"""CHB-MIT Scalp EEG utilities shared by the three projects.

Pipeline
--------
raw EDF (256 Hz, 23 bipolar ch)  ->  18-channel double-banana montage
  -> 0.5-45 Hz band-pass + 60 Hz notch  ->  resample to 128 Hz
  -> float16 .npy per recording + per-sample seizure mask (.json meta)

Each project copies this file (``src/chbmit.py``) so that every repository
is self-contained.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

# Standard 18-channel longitudinal bipolar ("double banana") montage that is
# present in every CHB-MIT recording used here.
MONTAGE = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
FS_RAW = 256
FS = 128  # working sampling rate


@dataclass
class Recording:
    patient: str
    name: str
    seizures: list[tuple[float, float]] = field(default_factory=list)  # seconds


def parse_summary(path: Path) -> dict[str, Recording]:
    """Parse ``chbXX-summary.txt`` into {file_name: Recording}."""
    patient = path.name.split("-")[0]
    recs: dict[str, Recording] = {}
    cur: Recording | None = None
    starts: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            cur = Recording(patient, line.split(":", 1)[1].strip())
            recs[cur.name] = cur
            starts = []
        elif cur is not None and re.match(r"Seizure( \d+)? Start Time", line):
            starts.append(float(re.findall(r"(\d+)\s*seconds", line)[0]))
        elif cur is not None and re.match(r"Seizure( \d+)? End Time", line):
            end = float(re.findall(r"(\d+)\s*seconds", line)[0])
            cur.seizures.append((starts.pop(0), end))
    return recs


def _read_edf(path: Path) -> tuple[np.ndarray, list[str], float]:
    import pyedflib

    with pyedflib.EdfReader(str(path)) as f:
        labels = [l.upper().strip() for l in f.getSignalLabels()]
        fs = f.getSampleFrequency(0)
        data = np.stack([f.readSignal(i) for i in range(f.signals_in_file)])
    return data, labels, fs


def load_montage(path: Path) -> np.ndarray:
    """Return (18, T) array in microvolts at 256 Hz, montage order."""
    data, labels, fs = _read_edf(path)
    assert abs(fs - FS_RAW) < 1e-6, f"unexpected fs {fs} in {path}"
    idx = []
    for ch in MONTAGE:
        # duplicated channel names (e.g. T8-P8) appear as 'T8-P8-0', 'T8-P8'
        cands = [i for i, l in enumerate(labels) if l == ch or l.startswith(ch + "-")]
        if not cands:
            raise KeyError(f"{ch} missing in {path.name}: {labels}")
        idx.append(cands[0])
    return data[idx]


def preprocess(x: np.ndarray) -> np.ndarray:
    """Band-pass 0.5-45 Hz, 60 Hz notch, resample 256 -> 128 Hz."""
    sos = signal.butter(4, [0.5, 45.0], btype="bandpass", fs=FS_RAW, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=-1)
    b, a = signal.iirnotch(60.0, 30.0, fs=FS_RAW)
    x = signal.filtfilt(b, a, x, axis=-1)
    x = signal.resample_poly(x, 1, FS_RAW // FS, axis=-1)
    return x.astype(np.float32)


def build_cache(raw_root: Path, cache_root: Path, files: list[str]) -> None:
    """Preprocess every ``chbXX/chbXX_YY.edf`` in *files* into *cache_root*."""
    cache_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Recording]] = {}
    index = []
    for rel in files:
        pat, fname = rel.split("/")
        if pat not in summaries:
            summaries[pat] = parse_summary(raw_root / f"{pat}-summary.txt")
        rec = summaries[pat][fname]
        out = cache_root / fname.replace(".edf", ".npy")
        if not out.exists():
            x = preprocess(load_montage(raw_root / rel))
            np.save(out, x.astype(np.float16))
        n = np.load(out, mmap_mode="r").shape[1]
        index.append({"patient": pat, "file": out.name, "n_samples": int(n),
                      "seizures": rec.seizures})
        print(f"cached {rel}: {n / FS / 60:.1f} min, seizures={rec.seizures}")
    (cache_root / "index.json").write_text(json.dumps(index, indent=1))


def load_index(cache_root: Path) -> list[dict]:
    return json.loads((cache_root / "index.json").read_text())


def seizure_mask(n: int, seizures: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in seizures:
        m[int(s * FS): int(e * FS)] = True
    return m


def make_windows(cache_root: Path, win_s: float = 4.0, ictal_stride_s: float = 1.0,
                 inter_stride_s: float = 8.0, inter_ratio: float = 5.0,
                 patients: list[str] | None = None, seed: int = 0,
                 guard_s: float = 300.0):
    """Labelled windows for seizure detection.

    Ictal windows: >=50% of samples inside a seizure, dense stride (1 s).
    Interictal windows: no seizure sample within ``guard_s`` seconds, sparse
    stride, randomly sub-sampled to ``inter_ratio`` x the ictal count per patient.

    Returns X (N, C, T) float32, y (N,), patient ids (N,), record ids (N,).
    """
    rng = np.random.default_rng(seed)
    W = int(win_s * FS)
    index = load_index(cache_root)
    cands = []  # (record_id, start, label, patient)
    for r_id, rec in enumerate(index):
        if patients and rec["patient"] not in patients:
            continue
        n = rec["n_samples"]
        m = seizure_mask(n, rec["seizures"])
        guard = seizure_mask(n, [(max(0, s - guard_s), e + guard_s) for s, e in rec["seizures"]])
        for s in range(0, n - W, int(ictal_stride_s * FS)):
            if m[s:s + W].mean() >= 0.5:
                cands.append((r_id, s, 1, rec["patient"]))
        for s in range(0, n - W, int(inter_stride_s * FS)):
            if not guard[s:s + W].any():
                cands.append((r_id, s, 0, rec["patient"]))
    lab = np.array([c[2] for c in cands]); pat = np.array([c[3] for c in cands])
    keep = []
    for p in np.unique(pat):
        ii = np.where((pat == p) & (lab == 1))[0]
        jj = np.where((pat == p) & (lab == 0))[0]
        k = min(len(jj), int(inter_ratio * max(len(ii), 1)))
        keep.extend(ii); keep.extend(rng.choice(jj, k, replace=False))
    keep = np.sort(np.array(keep))
    mm = {}
    X = np.empty((len(keep), len(MONTAGE), W), np.float32)
    for k, i in enumerate(keep):
        r_id, s, _, _ = cands[i]
        if r_id not in mm:
            mm[r_id] = np.load(cache_root / index[r_id]["file"], mmap_mode="r")
        X[k] = mm[r_id][:, s:s + W]
    rid = np.array([cands[i][0] for i in keep])
    return X, lab[keep], pat[keep], rid


def unlabeled_windows(cache_root: Path, n_per_record: int = 400, win_s: float = 4.0,
                      patients: list[str] | None = None, seed: int = 0) -> np.ndarray:
    """Random windows (no labels) for self-supervised pre-training."""
    rng = np.random.default_rng(seed)
    W = int(win_s * FS)
    out = []
    for rec in load_index(cache_root):
        if patients and rec["patient"] not in patients:
            continue
        x = np.load(cache_root / rec["file"], mmap_mode="r")
        starts = rng.integers(0, rec["n_samples"] - W, n_per_record)
        out.extend(np.asarray(x[:, s:s + W], dtype=np.float32) for s in starts)
    return np.stack(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the preprocessed CHB-MIT cache")
    ap.add_argument("--raw", default="data/chbmit")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--list", default="data/chbmit/download_list.txt")
    a = ap.parse_args()
    files = [l.strip() for l in Path(a.list).read_text().splitlines() if l.strip()]
    build_cache(Path(a.raw), Path(a.cache), files)
