"""EEG-as-image seizure detection with ImageNet transfer learning + explainability.

Leave-one-patient-out over 7 CHB-MIT patients. Variants:
  resnet18_imagenet   ImageNet-pre-trained ResNet-18, fully fine-tuned
  resnet18_scratch    same architecture, random init (does natural-image pre-training help?)
  linear_probe        frozen ImageNet ResNet-18 features + logistic regression

Explainability for the fine-tuned ImageNet model on each unseen patient:
  * Grad-CAM (layer3) averaged over ictal windows -> channel x frequency x time map
  * channel occlusion  : drop in seizure logit when one electrode's rows are blanked
  * frequency occlusion: drop when one 2 Hz band is blanked on every electrode
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chbmit import make_windows  # noqa: E402
from eeg_image import F_ROWS, N_CH, GradCAM, eeg_to_image, resnet18  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def batches(X, bs, perm=None):
    idx = np.arange(len(X)) if perm is None else perm
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield j, torch.from_numpy(X[j]).to(DEV).float()


def train(model, X, y, epochs, lr, bs=64, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), device=DEV)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEV)
    for ep in range(epochs):
        model.train()
        for j, xb in batches(X, bs, rng.permutation(len(X))):
            img = eeg_to_image(xb)
            # light augmentation: random circular time shift + one random time mask
            img = torch.roll(img, int(rng.integers(-8, 9)), dims=-1)
            t0 = int(rng.integers(0, img.shape[-1] - 6)); img[..., t0:t0 + 6] = 0
            with torch.autocast(DEV, dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(img).squeeze(-1), yt[j], pos_weight=pos_w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return model


@torch.no_grad()
def logits(model, X, bs=128, occlude=None):
    model.eval(); out = []
    for _, xb in batches(X, bs):
        img = eeg_to_image(xb)
        if occlude is not None:
            img = occlude(img)
        with torch.autocast(DEV, dtype=torch.float16):
            out.append(model(img).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed(model, X, bs=128):
    model.eval(); fc = model.fc; model.fc = torch.nn.Identity(); out = []
    for _, xb in batches(X, bs):
        with torch.autocast(DEV, dtype=torch.float16):
            out.append(model(eeg_to_image(xb)).float().cpu().numpy())
    model.fc = fc
    return np.concatenate(out)


def metrics(y, s, thr=0.0):
    pred = s >= thr
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "sensitivity": float(pred[y == 1].mean()), "specificity": float(1 - pred[y == 0].mean())}


def explain(model, Xi, n_examples=4):
    """Grad-CAM + occlusion on ictal windows of the unseen patient."""
    cam_engine = GradCAM(model, model.layer3)
    cams = []
    for _, xb in batches(Xi, 32):
        cams.append(cam_engine(eeg_to_image(xb)).cpu().numpy())
    cams = np.concatenate(cams)                      # (n, 360, 128)
    base = logits(model, Xi)
    ch_drop, f_drop = [], []
    for c in range(N_CH):
        def occ(img, c=c):
            img = img.clone(); img[:, :, c * F_ROWS:(c + 1) * F_ROWS] = 0; return img
        ch_drop.append(float((base - logits(model, Xi, occlude=occ)).mean()))
    for f in range(F_ROWS):
        rows = [c * F_ROWS + f for c in range(N_CH)]
        def occ(img, rows=rows):
            img = img.clone(); img[:, :, rows] = 0; return img
        f_drop.append(float((base - logits(model, Xi, occlude=occ)).mean()))
    ex = np.argsort(-base)[:n_examples]
    with torch.no_grad():
        ex_img = eeg_to_image(torch.from_numpy(Xi[ex]).to(DEV).float())[:, 0].cpu().numpy()
    return {"cam_mean": cams.mean(0), "ch_drop": np.array(ch_drop), "f_drop": np.array(f_drop),
            "ex_img": ex_img, "ex_cam": cams[ex]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True); (out / "xai").mkdir(exist_ok=True)
    X, y, pat, _ = make_windows(Path(a.cache), win_s=8.0, ictal_stride_s=2.0,
                                inter_stride_s=16.0, inter_ratio=5.0)
    X = X.astype(np.float16)
    patients = sorted(set(pat))
    print(f"windows {X.shape} ictal {y.sum()}", flush=True)
    rf = out / "lopo_results.json"
    res = json.loads(rf.read_text()) if rf.exists() else {}

    for p in patients:
        tr, te = pat != p, pat == p
        for variant in ("resnet18_imagenet", "resnet18_scratch", "linear_probe"):
            key = f"{variant}|{p}"
            if key in res:
                continue
            t0 = time.time()
            if variant == "linear_probe":
                m = resnet18(True).to(DEV)
                Etr, Ete = embed(m, X[tr]), embed(m, X[te])
                mu, sd = Etr.mean(0), Etr.std(0) + 1e-6
                clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
                clf.fit((Etr - mu) / sd, y[tr])
                s = clf.decision_function((Ete - mu) / sd)
            else:
                pre = variant.endswith("imagenet")
                m = train(resnet18(pre).to(DEV), X[tr], y[tr], a.epochs, 3e-4 if pre else 1e-3)
                s = logits(m, X[te])
                if pre:
                    e = explain(m, X[te][y[te] == 1])
                    np.savez_compressed(out / "xai" / f"{p}.npz", **e)
            res[key] = metrics(y[te], s)
            rf.write_text(json.dumps(res, indent=1))
            print(f"{variant:18s} test={p} " + " ".join(f"{k}={v:.3f}" for k, v in res[key].items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
