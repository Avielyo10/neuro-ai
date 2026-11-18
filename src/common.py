"""Shared utilities for Homework 1 (NeuroAI): data loading, ROI masks, caching.

Data layout (see scripts/download_data.sh):
    data/<SUBJ>.mat              18 subjects, variable ``data_crop`` (271,633 voxels x 280 TRs)
    data/group2_all_52rois.voi   52 ROIs (BrainVoyager VOI text file, 3-mm world coordinates)
    data/MNI152_T1_3mm_brain.nii 61 x 73 x 61 template that defines the voxel grid
    data/Story_Original_MRI.wav  the "Pieman" stimulus
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
for _d in (CACHE, RESULTS, FIGS):
    _d.mkdir(exist_ok=True)

TR = 1.5            # seconds
N_TR = 280          # time points per subject
N_SEG = 14          # story segments (decoding classes)
SEG_LEN = 20        # TRs per segment (30 s)
GRID = (61, 73, 61) # voxel grid of the 3-mm MNI template
SIGNAL_THRESHOLD = 6000  # mean-signal threshold used by the course helper scripts
VOI_FILE = DATA / "group2_all_52rois.voi"
TEMPLATE = DATA / "MNI152_T1_3mm_brain.nii"
AUDIO = DATA / "Story_Original_MRI.wav"


def subject_files() -> list[str]:
    """Sorted list of the 18 subject .mat files."""
    files = sorted(glob.glob(str(DATA / "*_*.mat")))
    assert len(files) == 18, f"expected 18 subject files, found {len(files)}"
    return files


def load_subject(path: str) -> np.ndarray:
    """Return one subject's data as a (280 TRs, 271,633 voxels) float32 array."""
    return np.ascontiguousarray(loadmat(path)["data_crop"].T, dtype=np.float32)


def load_voi(path: Path = VOI_FILE) -> list[tuple[str, np.ndarray]]:
    """Parse a BrainVoyager .voi text file -> list of (name, (n_vox, 3) world coords)."""
    lines = [l.strip() for l in Path(path).read_text(errors="ignore").splitlines() if l.strip()]
    vois, i = [], 0
    while i < len(lines):
        if lines[i].startswith("NameOfVOI"):
            name = lines[i].split(":", 1)[1].strip()
            while not lines[i].startswith("NrOfVoxels"):
                i += 1
            n = int(re.findall(r"-?\d+", lines[i])[0])
            coords = np.array([lines[i + 1 + k].split()[:3] for k in range(n)], dtype=int)
            vois.append((name, coords))
            i += n
        i += 1
    return vois


def voi_to_flat_index(coords: np.ndarray) -> np.ndarray:
    """World (mm) coordinates -> unique flat (Fortran-order) voxel indices on the 61x73x61 grid.

    Same convention as the course example code: inverse template affine, then
    ``flatten(order='F')`` so indices match the rows of ``data_crop``.
    """
    aff = nib.load(str(TEMPLATE)).affine
    ijk = np.round(np.c_[coords, np.ones(len(coords))] @ np.linalg.inv(aff).T)[:, :3].astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(GRID)), axis=1)
    return np.unique(np.ravel_multi_index(ijk[ok].T, GRID, order="F"))


def roi_labels() -> list[str]:
    """Unique labels 'NN name' (several VOI names repeat in the file, so index them)."""
    return [f"{k + 1:02d} {name}" for k, (name, _) in enumerate(load_voi())]


def load_roi_data() -> dict:
    """Per-ROI data for all subjects, cached to disk.

    Returns dict with
        'labels' : list of 52 ROI labels
        'X'      : list of 52 arrays, each (18 subjects, 280 TRs, V voxels), float32, raw signal
        'n_decl' : number of voxels declared in the VOI file per ROI
    Only voxels that carry valid signal in *all* 18 subjects are kept (mean signal
    above SIGNAL_THRESHOLD, finite, non-constant) so every subject has identical features.
    """
    cache = CACHE / "roi_data.npz"
    labels = roi_labels()
    vois = load_voi()
    if cache.exists():
        z = np.load(cache)
        return {"labels": labels, "X": [z[f"roi{k}"] for k in range(len(vois))],
                "n_decl": [len(c) for _, c in vois]}
    idx = [voi_to_flat_index(c) for _, c in vois]
    union = np.unique(np.concatenate(idx))
    sub = np.stack([load_subject(f)[:, union] for f in subject_files()])  # (18, 280, U)
    good = (np.isfinite(sub).all(1) & (sub.mean(1) > SIGNAL_THRESHOLD) & (sub.std(1) > 0)).all(0)
    out = {}
    for k, ix in enumerate(idx):
        cols = np.searchsorted(union, ix)
        out[f"roi{k}"] = sub[:, :, cols[good[cols]]]
    np.savez(cache, **out)
    return {"labels": labels, "X": [out[f"roi{k}"] for k in range(len(vois))],
            "n_decl": [len(c) for _, c in vois]}


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score along ``axis`` (population std); constant series map to 0."""
    x = x - x.mean(axis, keepdims=True)
    s = x.std(axis, keepdims=True)
    return x / np.where(s == 0, 1, s)


def segment_labels(n_tr: int = N_TR) -> np.ndarray:
    """Class label (0..13) of each TR: 14 consecutive, non-overlapping 20-TR segments."""
    return np.repeat(np.arange(n_tr // SEG_LEN), SEG_LEN)
