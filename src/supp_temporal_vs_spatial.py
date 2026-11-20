"""Supplementary analysis - how much of the Task-A accuracy comes from the temporal profile of the
regional response, and how much from the spatial pattern across voxels?

A Task-A sample is the 20 TRs x V voxels response matrix of one story segment.  With the same data
(voxels z-scored within subject), the same LOSO design and the same linear SVM (C = 1, one-vs-one),
three feature sets are compared in all 52 ROIs:

  full      the 20 x V matrix, flattened (20*V features)  -> identical to the Task-A linear SVM;
  temporal  the ROI-mean time course of the segment: mean over voxels at each TR (20 features);
  spatial   the time-averaged spatial pattern of the segment: mean over the 20 TRs per voxel (V features).

`temporal` and `spatial` are the two marginal means of the sample matrix and are used without
rescaling (primary analysis).  Because a linear SVM with fixed C is not scale-invariant, the script
also stores a sensitivity variant in which both reduced feature sets are re-standardised within
subject (`temporal_z`: ROI-mean time course z-scored over the 280 TRs; `spatial_z`: each voxel's 14
segment means z-scored across segments).  No parameter is tuned.

Run:  python src/supp_temporal_vs_spatial.py     (after part2a_decoding.py; well under a minute)
Out:  results/supp_temporal_vs_spatial.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import time

import numpy as np
from joblib import Parallel, delayed

from common import N_SEG, RESULTS, SEG_LEN, load_roi_data, zscore
from decoding import CLF_NAMES, GramLinearSVC

SETS = ["full", "temporal", "spatial", "temporal_z", "spatial_z"]


def feature_sets(x):
    """x: (18, 280, V) raw ROI data -> dict of (18 subjects, 14 segments, n_features) arrays."""
    n_sub, _, V = x.shape
    z = zscore(x, axis=1)                                     # voxel-wise, within subject (as in Task A)
    t = z.reshape(n_sub, N_SEG, SEG_LEN, V)
    return {"full": t.reshape(n_sub, N_SEG, -1),
            "temporal": t.mean(3),
            "spatial": t.mean(2),
            "temporal_z": zscore(z.mean(2), axis=1).reshape(n_sub, N_SEG, SEG_LEN),
            "spatial_z": zscore(t.mean(2), axis=1)}


def loso_svm(X):
    """LOSO accuracy per held-out subject for X (n_sub, n_seg, n_features)."""
    n_sub, n_seg, p = X.shape
    y = np.arange(n_seg)
    acc = np.zeros(n_sub)
    for s in range(n_sub):
        train = np.delete(np.arange(n_sub), s)
        acc[s] = GramLinearSVC(C=1.0).fit(X[train].reshape(-1, p), np.tile(y, n_sub - 1)).score(X[s], y)
    return acc


def run_roi(x):
    return {k: loso_svm(v) for k, v in feature_sets(x).items()}


def sem(a):
    return float(a.std(ddof=1) / np.sqrt(len(a)))


if __name__ == "__main__":
    roi = load_roi_data()
    labels, nv = roi["labels"], np.array([x.shape[2] for x in roi["X"]])
    t0 = time.time()
    order = np.argsort(-nv)
    res = [None] * 52
    for k, o in zip(order, Parallel(n_jobs=4, verbose=1)(delayed(run_roi)(roi["X"][k]) for k in order)):
        res[k] = o
    print(f"done in {time.time() - t0:.0f} s")
    acc = {s: np.stack([r[s] for r in res]) * 100 for s in SETS}          # (52, 18)

    main = np.load(RESULTS / "part2a_accuracy.npz")
    d = np.abs(acc["full"] - main["acc"][:, :, CLF_NAMES.index("Linear SVM")] * 100).max()
    assert d < 1e-9, f"'full' does not reproduce the Task-A linear-SVM accuracies (max diff {d})"

    m = {s: acc[s].mean(1) for s in SETS}                                 # per-ROI means
    out = {"design": "linear SVM (C = 1, one-vs-one), 14 classes, LOSO over 18 subjects, 52 ROIs; accuracy in %",
           "feature_sets": {"full": "20 x V segment matrix (Task A)",
                            "temporal": "ROI-mean time course of the segment (20 features)",
                            "spatial": "time-averaged spatial pattern of the segment (V features)",
                            "temporal_z": "as temporal, ROI-mean time course z-scored within subject",
                            "spatial_z": "as spatial, segment means z-scored across segments per voxel"},
           "chance_pct": 100 / N_SEG, "full_reproduces_part2a_svm": True,
           "mean_over_rois": {s: float(m[s].mean()) for s in SETS},
           "median_over_rois": {s: float(np.median(m[s])) for s in SETS},
           "n_rois_temporal_gt_spatial": int((m["temporal"] > m["spatial"]).sum()),
           "n_rois_spatial_gt_temporal": int((m["spatial"] > m["temporal"]).sum()),
           "n_rois_full_ge_both": int(((m["full"] >= m["temporal"]) & (m["full"] >= m["spatial"])).sum()),
           "n_rois_temporal_ge_full": int((m["temporal"] >= m["full"]).sum()),
           "n_rois_temporal_z_gt_spatial_z": int((m["temporal_z"] > m["spatial_z"]).sum()),
           "table": [{"roi": labels[k], "n_vox": int(nv[k]), **{s: float(m[s][k]) for s in SETS},
                      **{f"{s}_sem": sem(acc[s][k]) for s in SETS}} for k in range(52)]}
    json.dump(out, open(RESULTS / "supp_temporal_vs_spatial.json", "w"), indent=1)

    print("mean over ROIs:", {s: round(v, 1) for s, v in out["mean_over_rois"].items()})
    print("temporal > spatial in", out["n_rois_temporal_gt_spatial"], "of 52 ROIs; full >= both in",
          out["n_rois_full_ge_both"])
    for k in np.argsort(-m["full"])[:12]:
        print(f"{labels[k]:24s} V={nv[k]:4d}  " + "  ".join(f"{s} {m[s][k]:5.1f}" for s in SETS))
