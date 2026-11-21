"""Tables and explainability figures from results/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "src")
from chbmit import MONTAGE  # noqa: E402
from eeg_image import F_ROWS, FREQS, N_CH  # noqa: E402

R = json.loads(Path("results/lopo_results.json").read_text())
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
V = {"resnet18_imagenet": "ResNet-18, ImageNet init, fine-tuned",
     "resnet18_scratch": "ResNet-18, random init",
     "linear_probe": "Frozen ImageNet features + logistic reg."}
pats = sorted({k.split("|")[1] for k in R})

lines = ["| model | " + " | ".join(pats) + " | mean AUROC | mean AUPRC |", "|---|" + "---|" * (len(pats) + 2)]
for v, name in V.items():
    rs = [R.get(f"{v}|{p}") for p in pats]
    if any(r is None for r in rs):
        continue
    lines.append(f"| {name} | " + " | ".join(f"{r['auroc']:.3f}" for r in rs)
                 + f" | **{np.mean([r['auroc'] for r in rs]):.3f}** | {np.mean([r['auprc'] for r in rs]):.3f} |")
Path("results/table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

xs = {p: np.load(f"results/xai/{p}.npz") for p in pats if Path(f"results/xai/{p}.npz").exists()}
ch = np.stack([x["ch_drop"] for x in xs.values()])
fq = np.stack([x["f_drop"] for x in xs.values()])

# channel x patient occlusion heat-map + frequency importance
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
im = ax[0].imshow(ch / (np.abs(ch).max(1, keepdims=True) + 1e-9), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_yticks(range(len(xs))); ax[0].set_yticklabels(list(xs))
ax[0].set_xticks(range(N_CH)); ax[0].set_xticklabels(MONTAGE, rotation=90)
ax[0].set_title("electrode importance (logit drop when occluded, row-normalised)")
fig.colorbar(im, ax=ax[0], fraction=0.03)
for p, f in zip(xs, fq):
    ax[1].plot([fr + 1 for fr in FREQS], f, alpha=0.6, label=p)
ax[1].plot([fr + 1 for fr in FREQS], fq.mean(0), "k-", lw=2.5, label="mean")
ax[1].set_xlabel("frequency band centre (Hz)"); ax[1].set_ylabel("logit drop when band occluded")
ax[1].set_title("which rhythms drive the seizure decision"); ax[1].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "occlusion_importance.png", dpi=160)

# Grad-CAM examples (one per patient)
n = len(xs)
fig, ax = plt.subplots(2, n, figsize=(2.1 * n, 6.5))
for j, (p, x) in enumerate(xs.items()):
    ax[0, j].imshow(x["ex_img"][0], aspect="auto", cmap="magma")
    ax[1, j].imshow(x["ex_img"][0], aspect="auto", cmap="gray")
    ax[1, j].imshow(x["ex_cam"][0], aspect="auto", cmap="jet", alpha=0.45)
    for a in ax[:, j]:
        a.set_xticks([0, 32, 63]); a.set_xticklabels(["0", "4", "8 s"])
        a.set_yticks(np.arange(N_CH) * F_ROWS + F_ROWS / 2)
        a.set_yticklabels(MONTAGE if j == 0 else [], fontsize=5)
    ax[0, j].set_title(p)
ax[0, 0].set_ylabel("spectrogram image"); ax[1, 0].set_ylabel("Grad-CAM")
fig.suptitle("Ictal windows of unseen patients: log-spectrogram (top), Grad-CAM overlay (bottom)")
fig.tight_layout(); fig.savefig(FIG / "gradcam_examples.png", dpi=160)

# mean Grad-CAM as channel x frequency map
fig, ax = plt.subplots(figsize=(5, 4))
cam = np.stack([x["cam_mean"].mean(1).reshape(N_CH, F_ROWS) for x in xs.values()]).mean(0)
im = ax.imshow(cam, aspect="auto", cmap="viridis")
ax.set_yticks(range(N_CH)); ax.set_yticklabels(MONTAGE, fontsize=7)
ax.set_xticks(range(0, F_ROWS, 4)); ax.set_xticklabels([f"{FREQS[i]}" for i in range(0, F_ROWS, 4)])
ax.set_xlabel("frequency (Hz)"); ax.set_title("mean Grad-CAM over ictal windows (all patients)")
fig.colorbar(im, fraction=0.04); fig.tight_layout(); fig.savefig(FIG / "gradcam_channel_freq.png", dpi=160)
