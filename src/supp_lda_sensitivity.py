"""Supplementary analysis - how much does the Task-A LDA result depend on the covariance regulariser?

Same data, samples and LOSO design as part2a_decoding.py (14 classes, 20*V features, 18 folds, all
52 ROIs).  Three LDA variants that differ ONLY in how the pooled within-class covariance is regularised:

  span_lw       the model used in the report (decoding.ShrinkageLDA): Ledoit-Wolf shrinkage towards a
                scaled identity, with intensity and target scale estimated in the PCA subspace spanned
                by the training samples (q = 238 dimensions);
  voxel_lw      exact pooled Ledoit-Wolf shrinkage estimated in the full p = 20*V voxel space
                (same formula, applied to the 238 x p matrix of within-class residuals);
  sklearn_auto  the estimator of scikit-learn's LinearDiscriminantAnalysis(solver="lsqr",
                shrinkage="auto") on the raw features: per class, features are standardised, the
                Ledoit-Wolf estimate is computed and scaled back, and the class estimates are averaged.

For p up to 15,320 the p x p covariance is never formed.  All three estimators have the form
    Sigma = diag(g) + R^T diag(w) R        (R: 238 x p within-class residuals)
so Sigma^-1 is applied with the Woodbury identity (238 x 238 solves) and the Ledoit-Wolf intensity is
computed from Gram matrices.  `selfcheck()` verifies these shortcuts against scikit-learn's own dense
implementations on a small ROI before anything is reported.

Nothing here is used to choose a model: the report keeps span_lw; this script only documents that the
rank of LDA relative to the other classifiers depends on the regulariser.

Run:  python src/supp_lda_sensitivity.py     (after part2a_decoding.py; about 1 minute on 4 cores)
Out:  results/supp_lda_sensitivity.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import time

import numpy as np
from joblib import Parallel, delayed

from common import N_SEG, RESULTS, SEG_LEN, load_roi_data, zscore
from decoding import CLF_NAMES, ShrinkageLDA, segment_samples

VARIANTS = ["span_lw", "voxel_lw", "sklearn_auto"]
WIDE_V = 100          # "wide" ROIs for the shrinkage-intensity summary: V >= 100 (p = 20 V >= 2000 >> 238)


def lw_intensity(X):
    """Ledoit-Wolf shrinkage intensity and target scale mu = tr(S)/p of S = X^T X / n (X centred),
    computed from the n x n Gram matrix (identical to sklearn.covariance.ledoit_wolf_shrinkage)."""
    n, p = X.shape
    G = X @ X.T
    tr = np.trace(G) / n
    mu = tr / p
    delta_ = (G ** 2).sum() / n ** 2                       # ||S||_F^2
    beta_ = ((X ** 2).sum(1) ** 2).sum()
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * tr + p * mu ** 2) / p
    beta = min(beta, delta)
    return (0.0 if beta == 0 else float(beta / delta)), float(mu)


def woodbury_predict(M, R, w, g, Xte):
    """LDA prediction (equal priors) for Sigma = diag(g) + R^T diag(w) R without forming Sigma.

    M (classes, p) class means, R (n, p) residuals, w (n,) row weights, g (p,) > 0 diagonal part.
    """
    keep = w > 0
    R, w = R[keep], w[keep]
    Rg = R / g
    K = np.diag(1.0 / w) + Rg @ R.T
    Mg = M / g
    W = Mg - np.linalg.solve(K, R @ Mg.T).T @ Rg           # rows: Sigma^-1 m_c
    b = -0.5 * np.einsum("cd,cd->c", M, W)
    return np.argmax(Xte @ W.T + b, axis=1)


def split(X, s):
    """Training / test samples of LOSO fold s.  X: (n_sub, n_seg, p)."""
    n_sub, n_seg, p = X.shape
    train = np.delete(np.arange(n_sub), s)
    y = np.arange(n_seg)
    return (X[train].reshape(-1, p).astype(np.float64), np.tile(y, n_sub - 1),
            X[s].astype(np.float64), y)


def predict_voxel_lw(Xtr, ytr, Xte):
    """Pooled within-class covariance, Ledoit-Wolf in the full voxel space."""
    M = np.stack([Xtr[ytr == c].mean(0) for c in np.unique(ytr)])
    R = Xtr - M[ytr]
    n, p = R.shape
    a, mu = lw_intensity(R)
    return woodbury_predict(M, R, np.full(n, (1 - a) / n), np.full(p, a * mu), Xte), a


def predict_sklearn_auto(Xtr, ytr, Xte):
    """Equivalent of LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') on raw features.

    scikit-learn: Sigma = sum_c prior_c * D_c [(1 - a_c) Z_c^T Z_c / n_c + a_c mu_c I] D_c, where Z_c are
    the class-c samples standardised feature-wise (scale D_c) and (a_c, mu_c) the Ledoit-Wolf estimate.
    """
    classes = np.unique(ytr)
    M = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    R = Xtr - M[ytr]
    n, p = R.shape
    g, w = np.zeros(p), np.zeros(n)
    for c in classes:
        idx = np.flatnonzero(ytr == c)
        sd = R[idx].std(0)
        sd[sd == 0] = 1.0
        a, mu = lw_intensity(R[idx] / sd)
        prior = len(idx) / n
        g += prior * a * mu * sd ** 2
        w[idx] = prior * (1 - a) / len(idx)
    return woodbury_predict(M, R, w, g, Xte)


def run_roi(x):
    """x: (18, 280, V) raw ROI data.  Returns dict of per-fold arrays."""
    t = segment_samples(zscore(x, axis=1), SEG_LEN)                       # identical to part2a
    X = np.transpose(t, (3, 0, 1, 2)).reshape(t.shape[3], N_SEG, -1)      # (subject, segment, 20 V)
    n_sub, _, p = X.shape
    out = {k: np.zeros(n_sub) for k in VARIANTS + ["a_span", "a_span_voxel_equiv", "a_voxel"]}
    for s in range(n_sub):
        Xtr, ytr, Xte, y = split(X, s)
        lda = ShrinkageLDA().fit(Xtr, ytr)
        out["span_lw"][s] = lda.score(Xte, y)
        a, q = lda.shrinkage_, lda.means_.shape[1]
        out["a_span"][s] = a
        # (1-a) S + a (tr S / q) I  is proportional to  (1-a') S + a' (tr S / p) I  with:
        out["a_span_voxel_equiv"][s] = a * p / (a * p + (1 - a) * q)
        pred, a_vox = predict_voxel_lw(Xtr, ytr, Xte)
        out["voxel_lw"][s] = np.mean(pred == y)
        out["a_voxel"][s] = a_vox
        out["sklearn_auto"][s] = np.mean(predict_sklearn_auto(Xtr, ytr, Xte) == y)
    return out


def selfcheck(roi, max_p=700):
    """Compare the Gram / Woodbury shortcuts with scikit-learn's dense implementations on the largest
    ROI that is still small enough for dense p x p covariances (all 18 folds)."""
    from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    nv = np.array([x.shape[2] for x in roi["X"]])
    k = int(np.argmax(np.where(nv * SEG_LEN <= max_p, nv, -1)))
    t = segment_samples(zscore(roi["X"][k], axis=1), SEG_LEN)
    X = np.transpose(t, (3, 0, 1, 2)).reshape(t.shape[3], N_SEG, -1)
    d_a, same_vox, same_skl = 0.0, True, True
    for s in range(X.shape[0]):
        Xtr, ytr, Xte, y = split(X, s)
        M = np.stack([Xtr[ytr == c].mean(0) for c in y])
        R = Xtr - M[ytr]
        d_a = max(d_a, abs(lw_intensity(R)[0] - ledoit_wolf_shrinkage(R, assume_centered=True)))
        cov = LedoitWolf(assume_centered=True).fit(R).covariance_
        W = np.linalg.solve(cov, M.T).T
        dense = np.argmax(Xte @ W.T - 0.5 * np.einsum("cd,cd->c", M, W), axis=1)
        same_vox &= bool(np.array_equal(dense, predict_voxel_lw(Xtr, ytr, Xte)[0]))
        skl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte)
        same_skl &= bool(np.array_equal(skl, predict_sklearn_auto(Xtr, ytr, Xte)))
    res = {"roi": roi["labels"][k], "n_features": int(X.shape[2]), "n_folds": int(X.shape[0]),
           "max_abs_diff_lw_intensity_vs_sklearn": float(d_a),
           "voxel_lw_predictions_equal_dense": same_vox,
           "sklearn_auto_predictions_equal_sklearn_lda": same_skl}
    assert d_a < 1e-9 and same_vox and same_skl, res
    return res


def sem(a):
    return float(a.std(ddof=1) / np.sqrt(len(a)))


if __name__ == "__main__":
    roi = load_roi_data()
    labels, nv = roi["labels"], np.array([x.shape[2] for x in roi["X"]])
    t0 = time.time()
    check = selfcheck(roi)
    print("self-check passed:", check)
    order = np.argsort(-nv)
    res = [None] * 52
    for k, o in zip(order, Parallel(n_jobs=4, verbose=1)(delayed(run_roi)(roi["X"][k]) for k in order)):
        res[k] = o
    print(f"done in {time.time() - t0:.0f} s")

    main = np.load(RESULTS / "part2a_accuracy.npz")                      # (ROI, fold, classifier)
    assert list(main["clf"]) == CLF_NAMES
    jL = CLF_NAMES.index("LDA")
    acc = {v: np.stack([r[v] for r in res]) * 100 for v in VARIANTS}     # (52, 18)
    same = np.abs(acc["span_lw"] - main["acc"][:, :, jL] * 100).max()
    assert same < 1e-9, f"span_lw does not reproduce the Task-A LDA accuracies (max diff {same})"

    out = {"design": "Task A design: 14 classes, 20*V features, LOSO over 18 subjects, 52 ROIs; accuracy in %",
           "variants": {"span_lw": "main model: Ledoit-Wolf in the 238-dim PCA span of the training fold",
                        "voxel_lw": "exact pooled Ledoit-Wolf in the 20*V voxel space",
                        "sklearn_auto": "equivalent of sklearn LDA(solver='lsqr', shrinkage='auto') on raw features"},
           "selfcheck": check, "span_lw_reproduces_part2a_lda": True, "clf": CLF_NAMES,
           "mean_over_rois": {}, "median_over_rois": {}, "n_rois_lda_best": {}, "n_rois_best_by_clf": {},
           "clf_mean": {}, "top3_rois": {}, "top3_mean_acc": {}, "diff_vs_span_lw_pp": {}}
    for v in VARIANTS:
        m = main["acc"].mean(1) * 100                                    # (52, 7) with LDA replaced
        m[:, jL] = acc[v].mean(1)
        best = np.round(m, 6).argmax(1)                                  # exact ties: first-listed classifier
        top = np.argsort(-m.mean(1))[:3]
        out["mean_over_rois"][v] = float(acc[v].mean())
        out["median_over_rois"][v] = float(np.median(acc[v].mean(1)))
        out["n_rois_lda_best"][v] = int((best == jL).sum())
        out["n_rois_best_by_clf"][v] = [int((best == j).sum()) for j in range(len(CLF_NAMES))]
        out["clf_mean"][v] = m.mean(0).tolist()
        out["top3_rois"][v] = [labels[k] for k in top]
        out["top3_mean_acc"][v] = [float(m[k].mean()) for k in top]
        if v != "span_lw":
            d = acc[v].mean(1) - acc["span_lw"].mean(1)
            out["diff_vs_span_lw_pp"][v] = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max())}
    a = {k: np.array([r[k].mean() for r in res]) for k in ("a_span", "a_span_voxel_equiv", "a_voxel")}
    wide = nv >= WIDE_V
    out["shrinkage"] = {"wide_roi_min_voxels": WIDE_V, "n_wide_rois": int(wide.sum()),
                        **{f"{k}_range_wide": [float(v[wide].min()), float(v[wide].max())] for k, v in a.items()},
                        **{f"{k}_range_all": [float(v.min()), float(v.max())] for k, v in a.items()}}
    out["table"] = [{"roi": labels[k], "n_vox": int(nv[k]),
                     **{v: float(acc[v][k].mean()) for v in VARIANTS},
                     **{f"{v}_sem": sem(acc[v][k]) for v in VARIANTS},
                     **{key: float(val[k]) for key, val in a.items()}} for k in range(52)]
    json.dump(out, open(RESULTS / "supp_lda_sensitivity.json", "w"), indent=1)

    for v in VARIANTS:
        print(f"{v:13s} mean {out['mean_over_rois'][v]:.1f} %  LDA best in {out['n_rois_lda_best'][v]:2d} ROIs  "
              f"best-by-clf {out['n_rois_best_by_clf'][v]}  top-3 {out['top3_rois'][v]}")
    print("shrinkage:", out["shrinkage"])
    print("differences vs main model (pp):", out["diff_vs_span_lw_pp"])
