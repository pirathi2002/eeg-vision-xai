# Full-scale Colab notebooks

These notebooks run the three projects at full scale, which my laptop (GTX 1650 4 GB GPU, ~7 GB free disk) could not do.

| Notebook | Runtime | Data | Approx. time |
|---|---|---|---|
| `Project1_Robust_EEG_Foundation_Model.ipynb` | GPU (A100/L4 best, T4 OK) | Siena (pre-training, ~128 h) + CHB-MIT (all 24 cases) | T4: ~1.5–2 h pre-train + 4–7 h fine-tune |
| `Project2_TinySeizure_HW_Efficient.ipynb` | CPU (High-RAM if available) | **All** 686 CHB-MIT recordings (~980 h) | ~3–6 h (mostly download) |
| `Project3_EEG_Vision_XAI.ipynb` | GPU | CHB-MIT (cache shared with Project 1) | T4: ~4–7 h |

## How to run
1. Go to https://colab.research.google.com, then *File → Upload notebook*, and pick one `.ipynb` from this folder.
2. *Runtime → Change runtime type*: choose **GPU** for Projects 1 and 3, and **CPU** for Project 2.
3. *Runtime → Run all*. Allow Google Drive access when asked.
4. **If Colab disconnects** (free sessions do this after a few hours), reconnect and *Run all* again. Every stage resumes from
   what is already saved on Drive: data caches, the pre-training checkpoint, and each `(model, fold)` result.
5. When finished, the last cell downloads `projectX_results.zip`. Everything is also kept in Google Drive under
   `MyDrive/PhD_EEG/`.

**Suggested order:** Project 1 first, because it builds the CHB-MIT cache that Project 3 reuses. You can run Project 2 at the same
time in another Colab tab, since it only needs a CPU.

## Google Drive space needed
About 11 GB in `MyDrive/PhD_EEG/`: CHB-MIT cache tar ~4 GB, Siena cache tar ~2 GB, TinySeizure features ~4 GB, plus results.
Free Google accounts have 15 GB. Delete `MyDrive/PhD_EEG/cache/` after all runs finish if you need the space back.

## What changed compared with the laptop runs
| | Laptop run | Colab notebook |
|---|---|---|
| Subjects | 7 | **all 23** (24 cases; chb21 is merged with chb01, the same person) |
| Evaluation | leave-one-patient-out over 7 | subject-wise 5-fold over 23, with a separate validation set and early stopping (`--folds 0` = leave-one-subject-out) |
| P1 pre-training | 38 h CHB-MIT (17 other patients), 0.56 M params, 20 epochs | **Siena** (another hospital and country), ~3 M params, 50 epochs |
| P1 fine-tuning | 6 fixed epochs | up to 30 epochs, best epoch by validation AUPRC |
| P2 data | 66 h, 44 seizures | **~980 h, ~198 seizures** |
| P3 images | 360 × 64, 4 epochs, ResNet-18 only | 360 × 128, up to 20 epochs, + EfficientNet-B0, + frequency-mask augmentation |

`make_notebooks.py` regenerates the notebooks from `src/`. Edit the `.py` files and re-run it.
All code was smoke-tested locally on a small subset before packaging.
