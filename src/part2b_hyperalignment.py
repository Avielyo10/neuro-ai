"""Part 2 / Task B - hyperalignment (Haxby et al., 2011) and re-evaluation of all classifiers.

Non-circular split-half design, repeated with the two halves swapped:
    alignment half : 140 TRs used ONLY to estimate the orthogonal Procrustes transformations
    decoding half  : the other 140 TRs = 7 story segments -> 7-class problem (chance 1/7 = 14.29 %)
Voxel time courses are z-scored within each half separately, so the halves share nothing.

Within every LOSO fold:
    1. the common-space template is built from the alignment half of the 17 TRAINING subjects only
       (Haxby's three steps, see `build_template`);
    2. the held-out subject gets its own transformation by aligning its alignment half to that template;
    3. every subject's decoding half is rotated into the common space (X @ R), cut into 7 segments
       of 20 TRs, and all Task-A classifiers are trained on 17 subjects / tested on the held-out one.
The anatomical baseline is the identical procedure with R = identity.
Optional analysis: the common space is reduced to the top-k principal axes of the training template.

Run:  python src/part2b_hyperalignment.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes

from common import RESULTS, SEG_LEN, load_roi_data, zscore
from decoding import CLF_NAMES, loso_accuracy, segment_samples

HALF = 140
K_LIST = [5, 10, 20, 40, 80]     # dimensionalities of the reduced common space


def procrustes(X, Y):
    """Orthogonal R (R^T R = I, no scaling) minimising ||X R - Y||_F."""
    return orthogonal_procrustes(X, Y)[0]


def build_template(data):
    """Haxby et al. (2011) common-space template from a list of (T, V) matrices.

    (i)   start from a reference subject; align the next subject to it and average; then align each
          remaining subject to the evolving template (running mean of everything aligned so far);
    (ii)  second pass: re-align every subject to the pass-1 template and average the re-aligned data;
    (iii) return that final template (each subject's final R is its Procrustes fit to it).
    """
    template = data[0].copy()
    for n, X in enumerate(data[1:], start=1):
        template = (template * n + X @ procrustes(X, template)) / (n + 1)
    return np.mean([X @ procrustes(X, template) for X in data], axis=0)


def run_roi(x):
    """x: (18, 280, V).  Returns accuracies (2 half-assignments, 18 folds, classifiers) for the
    anatomical baseline, full hyperalignment, and the reduced common space (len(K_LIST), ...)."""
    n_sub, _, V = x.shape
    acc_anat = np.zeros((2, n_sub, len(CLF_NAMES)))
    acc_hyp = np.zeros_like(acc_anat)
    acc_k = np.full((len(K_LIST), 2, n_sub, len(CLF_NAMES)), np.nan)
    halves = [zscore(x[:, :HALF], axis=1), zscore(x[:, HALF:], axis=1)]
    for h in range(2):                         # h = index of the DECODING half
        dec, ali = halves[h], halves[1 - h]
        acc_anat[h] = loso_accuracy(segment_samples(dec, SEG_LEN))
        for s in range(n_sub):
            train = [i for i in range(n_sub) if i != s]
            template = build_template([ali[i] for i in train])          # training subjects only
            R = {i: procrustes(ali[i], template) for i in range(n_sub)}  # incl. held-out subject
            common = np.stack([dec[i] @ R[i] for i in train + [s]])      # held-out subject last
            acc_hyp[h, s] = loso_fold(common)
            # reduced common space: top-k right singular vectors of the training template
            Vt = np.linalg.svd(template, full_matrices=False)[2]
            for j, k in enumerate(K_LIST):
                if k < min(V, HALF - 1):
                    acc_k[j, h, s] = loso_fold(common @ Vt[:k].T)
    return acc_anat, acc_hyp, acc_k


def loso_fold(common):
    """Train on all but the last subject of `common` (n_sub, T, features), test on the last."""
    from decoding import fit_score_all
    t = segment_samples(common, SEG_LEN)                                  # (7, 20, F, n_sub)
    X = np.transpose(t, (3, 0, 1, 2)).reshape(common.shape[0], t.shape[0], -1)
    y = np.arange(t.shape[0])
    return fit_score_all(X[:-1].reshape(-1, X.shape[2]), np.tile(y, X.shape[0] - 1), X[-1], y)


if __name__ == "__main__":
    roi = load_roi_data()
    t0 = time.time()
    order = np.argsort([-x.shape[2] for x in roi["X"]])
    out = Parallel(n_jobs=4, verbose=5)(delayed(run_roi)(roi["X"][k]) for k in order)
    n = len(CLF_NAMES)
    anat, hyp, red = np.zeros((52, 2, 18, n)), np.zeros((52, 2, 18, n)), np.zeros((52, len(K_LIST), 2, 18, n))
    for k, (a, h, r) in zip(order, out):
        anat[k], hyp[k], red[k] = a, h, r
    np.savez(RESULTS / "part2b_accuracy.npz", anat=anat, hyper=hyp, reduced=red, k_list=K_LIST,
             clf=CLF_NAMES, labels=roi["labels"], n_vox=[x.shape[2] for x in roi["X"]])
    print(f"done in {time.time() - t0:.0f} s")
    a, h = anat.mean((1, 2)) * 100, hyp.mean((1, 2)) * 100
    print("anat  mean over ROIs:", dict(zip(CLF_NAMES, a.mean(0).round(1))))
    print("hyper mean over ROIs:", dict(zip(CLF_NAMES, h.mean(0).round(1))))
    for j, k in enumerate(K_LIST):
        print(f"k={k}:", np.nanmean(red[:, j], axis=(0, 1, 2)).round(3) * 100)
    for k in np.argsort(-(h - a).mean(1))[:10]:
        print(f"{roi['labels'][k]:24s}", a[k].round(0), "->", h[k].round(0))
