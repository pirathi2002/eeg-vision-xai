"""Animated demo: ictal spectrogram images of unseen subjects and where the model looks.

Cycles through the highest-scoring seizure windows of every cross-validation fold
(results/full_scale/xai/fold*.npz): the electrode x frequency x time image, the Grad-CAM
overlay, and the per-electrode Grad-CAM profile. Writes figures/demo_xai.gif and .mp4.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, "src")
from chbmit import MONTAGE  # noqa: E402

F_ROWS = 20
items = []
for f in sorted(Path("results/full_scale/xai").glob("fold*.npz")):
    d = np.load(f)
    for k in range(len(d["ex_img"])):
        items.append((f.stem, k, d["ex_img"][k], d["ex_cam"][k]))

fig, ax = plt.subplots(1, 3, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.55]})


def draw(i):
    fold, k, img, cam = items[i]
    for a in ax:
        a.clear()
    ax[0].imshow(img, aspect="auto", cmap="magma")
    ax[1].imshow(img, aspect="auto", cmap="gray"); ax[1].imshow(cam, aspect="auto", cmap="jet", alpha=0.45)
    prof = cam.reshape(18, F_ROWS, -1).mean((1, 2))
    ax[2].barh(np.arange(18), prof, color="#e67e22"); ax[2].invert_yaxis(); ax[2].set_xlim(0, 1)
    ax[2].set_yticks(np.arange(18)); ax[2].set_yticklabels(MONTAGE, fontsize=7); ax[2].set_xlabel("mean Grad-CAM")
    for a in ax[:2]:
        a.set_yticks(np.arange(18) * F_ROWS + F_ROWS / 2); a.set_yticklabels(MONTAGE, fontsize=6)
        a.set_xticks([0, img.shape[1] // 2, img.shape[1] - 1]); a.set_xticklabels(["0 s", "4 s", "8 s"])
    ax[0].set_title("EEG as image: electrode x (1-40 Hz) x time", fontsize=9)
    ax[1].set_title("Grad-CAM: where ResNet-18 looks", fontsize=9)
    fig.suptitle(f"Seizure window of an UNSEEN subject  ({fold}, example {k + 1})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])


anim = FuncAnimation(fig, draw, frames=len(items), interval=1200)
Path("figures").mkdir(exist_ok=True)
anim.save("figures/demo_xai.gif", writer=PillowWriter(fps=1), dpi=60)
anim.save("figures/demo_xai.mp4", writer=FFMpegWriter(fps=1, bitrate=800), dpi=100)
print(len(items), "frames")
