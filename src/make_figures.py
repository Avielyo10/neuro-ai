"""Figures and summary tables for Part 2 (Tasks A and B).  Run after part2a/part2b scripts."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

from common import FIGS, RESULTS

CH14, CH7 = 100 / 14, 100 / 7
COL = ["#9ecae1", "#6baed6", "#3182bd", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def sem(a, axis):
    return a.std(axis, ddof=1) / np.sqrt(a.shape[axis])


def part2a():
    z = np.load(RESULTS / "part2a_accuracy.npz")
    acc, clf, labels, nvox = z["acc"] * 100, list(z["clf"]), list(z["labels"]), z["n_vox"]
    m, s = acc.mean(1), sem(acc, 1)                      # (52, 7)
    order = np.argsort(-m.mean(1))

    fig, axes = plt.subplots(len(clf), 1, figsize=(13, 14.2), sharex=True)
    for j, ax in enumerate(axes):
        ax.bar(range(52), m[order, j], yerr=s[order, j], color=COL[j], ecolor="0.25", capsize=1.5,
               error_kw=dict(lw=0.7))
        ax.axhline(CH14, color="k", ls="--", lw=0.9)
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 92); ax.set_xlim(-0.7, 51.7)
        ax.set_title(f"{clf[j]}   (mean over ROIs {m[:, j].mean():.1f} %, best {labels[int(m[:, j].argmax())]} "
                     f"{m[:, j].max():.1f} %)", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(52))
    axes[-1].set_xticklabels([f"{labels[k]} ({nvox[k]})" for k in order], rotation=90, fontsize=9.5)
    axes[-1].set_xlabel("ROI (index name, number of voxels), sorted by mean accuracy over classifiers")
    fig.suptitle("Task A \u2013 LOSO 14-way story-segment decoding in anatomical space (mean ± SEM over 18 folds; dashed line = chance, 7.14 %)",
                 y=0.995, fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "part2a_bars_all_rois.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(m[order].T, aspect="auto", cmap="viridis", vmin=0, vmax=80)
    ax.set_yticks(range(len(clf))); ax.set_yticklabels(clf)
    ax.set_xticks(range(52)); ax.set_xticklabels([labels[k] for k in order], rotation=90, fontsize=9)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", pad=0.01)
    ax.set_title("Task A \u2013 accuracy of every classifier in every ROI", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGS / "part2a_heatmap.png", dpi=170); plt.close(fig)

    # summary: classifier means, significance vs chance per ROI (one-sided t-test over folds, FDR)
    p = np.array([[ttest_1samp(acc[k, :, j], CH14, alternative="greater").pvalue for j in range(len(clf))]
                  for k in range(52)])
    flat = np.sort(p.ravel()); n = flat.size
    thr = flat[np.nonzero(flat <= 0.05 * np.arange(1, n + 1) / n)[0].max()] if (flat <= 0.05 * np.arange(1, n + 1) / n).any() else 0
    # Best classifier per ROI from the integer number of correct test samples (18 folds x 14 samples), so
    # that exact ties are detected reliably; a tie goes to the first-listed classifier (argmax).
    n_correct = np.round(z["acc"].sum(1) * 14).astype(int)           # (52, 7); 14 test samples per fold
    best = n_correct.argmax(1)
    tied = (n_correct == n_correct.max(1, keepdims=True)).sum(1) > 1
    n_tied = int(tied.sum())
    out = {"clf": clf, "clf_mean": m.mean(0).tolist(), "clf_median": np.median(m, 0).tolist(),
           "n_sig_rois_fdr": (p <= thr).sum(0).tolist(),
           "n_rois_best": [int((best == j).sum()) for j in range(len(clf))], "n_rois_best_tied": n_tied,
           "rois_best_tied": [{"roi": labels[k], "tied": [clf[j] for j in np.flatnonzero(n_correct[k] == n_correct[k].max())]}
                              for k in np.flatnonzero(tied)],
           "best_median_nvox": [float(np.median(nvox[best == j])) if (best == j).any() else None for j in range(len(clf))],
           "n_big_rois": int((nvox >= 200).sum()),
           "n_big_rois_best": [int(((best == j) & (nvox >= 200)).sum()) for j in range(len(clf))],
           "rank_order": [labels[k] for k in order], "rank_mean": [float(m[k].mean()) for k in order],
           "top3_by_best_single_clf": [labels[k] for k in np.argsort(-m.max(1))[:3]],
           "top3_by_clf": {c: [labels[k] for k in np.argsort(-m[:, j])[:3]] for j, c in enumerate(clf)},
           "best_single": {"roi": labels[int(np.unravel_index(m.argmax(), m.shape)[0])],
                           "clf": clf[int(np.unravel_index(m.argmax(), m.shape)[1])], "acc": float(m.max())},
           "n_rois_total": 52, "n_vox_min": int(nvox.min()), "n_vox_max": int(nvox.max()),
           "top10": [{"roi": labels[k], "n_vox": int(nvox[k]), "mean": float(m[k].mean()),
                      "acc": m[k].tolist(), "sem": s[k].tolist()} for k in order[:10]],
           "bottom5": [{"roi": labels[k], "n_vox": int(nvox[k]), "mean": float(m[k].mean()),
                        "acc": m[k].tolist()} for k in order[-5:]],
           "corr_logV_acc": float(np.corrcoef(np.log(nvox), m.mean(1))[0, 1]),
           "table": [{"roi": labels[k], "n_vox": int(nvox[k]), "acc": m[k].tolist(), "sem": s[k].tolist()}
                     for k in range(52)]}
    json.dump(out, open(RESULTS / "part2a_summary.json", "w"), indent=1)
    return out


def part2b():
    z = np.load(RESULTS / "part2b_accuracy.npz")
    clf, labels, nvox, klist = list(z["clf"]), list(z["labels"]), z["n_vox"], list(z["k_list"])
    A, H = z["anat"].mean(1) * 100, z["hyper"].mean(1) * 100       # (52, 18, 7): mean of the two halves
    R = np.nanmean(z["reduced"], axis=2) * 100                       # (52, K, 18, 7)
    a, h = A.mean(1), H.mean(1)                                      # (52, 7)
    gain = H - A
    g, gs = gain.mean(1), sem(gain, 1)

    # (1) scatter per classifier
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for j, ax in enumerate(axes.ravel()[:7]):
        ax.scatter(a[:, j], h[:, j], s=8 + nvox / 12, color=COL[j], alpha=0.75, edgecolor="k", lw=0.3)
        ax.plot([0, 100], [0, 100], "k-", lw=0.7); ax.axhline(CH7, color="k", ls=":", lw=0.7); ax.axvline(CH7, color="k", ls=":", lw=0.7)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        ax.set_title(f"{clf[j]}\n{a[:, j].mean():.1f} \u2192 {h[:, j].mean():.1f} %  "
                     f"({(h[:, j] > a[:, j]).sum()}/52 ROIs improve)", fontsize=9.5)
        ax.set_xlabel("Anatomical alignment (%)"); ax.set_ylabel("Hyperalignment (%)")
    ax = axes.ravel()[7]
    x = np.arange(len(clf)); w = 0.38
    ax.bar(x - w / 2, a.mean(0), w, yerr=sem(A.mean(0), 0), color="0.6", label="anatomical", capsize=2)
    ax.bar(x + w / 2, h.mean(0), w, yerr=sem(H.mean(0), 0), color="#d62728", label="hyperaligned", capsize=2)
    ax.axhline(CH7, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(clf, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mean accuracy over 52 ROIs (%)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Average over ROIs (± SEM over folds)", fontsize=9.5)
    fig.suptitle("Task B \u2013 7-way split-half LOSO decoding: anatomical vs. hyperaligned features "
                 "(one dot per ROI, size ~ voxels; dotted = chance 14.29 %)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGS / "part2b_scatter.png", dpi=170); plt.close(fig)

    # (2) per-ROI gain (mean over classifiers) + anat/hyper bars
    gm = gain.mean(2)                                               # (52, 18)
    order = np.argsort(-gm.mean(1))
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    xx = np.arange(52)
    ax[0].bar(xx - 0.2, A.mean(2).mean(1)[order], 0.4, yerr=sem(A.mean(2), 1)[order], color="0.6", label="anatomical", error_kw=dict(lw=0.6))
    ax[0].bar(xx + 0.2, H.mean(2).mean(1)[order], 0.4, yerr=sem(H.mean(2), 1)[order], color="#d62728", label="hyperaligned", error_kw=dict(lw=0.6))
    ax[0].axhline(CH7, color="k", ls="--", lw=0.8, label="chance (14.29 %)")
    ax[0].set_ylabel("Accuracy, mean of 7 classifiers (%)")
    ax[0].legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9)   # outside the axes
    ax[1].bar(xx, gm.mean(1)[order], yerr=sem(gm, 1)[order], color=np.where(gm.mean(1)[order] >= 0, "#d62728", "#3182bd"), error_kw=dict(lw=0.6))
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_ylabel("Gain: hyper \u2212 anatomical (pct. points)")
    ax[1].set_xticks(xx); ax[1].set_xticklabels([f"{labels[k]} ({nvox[k]})" for k in order], rotation=90, fontsize=8.5)
    ax[1].set_xlim(-0.7, 51.7)
    for q in ax: q.spines[["top", "right"]].set_visible(False)
    ax[0].set_title("Task B \u2013 effect of hyperalignment per ROI (mean ± SEM over 18 folds; ROIs sorted by gain)",
                    fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(FIGS / "part2b_gain_per_roi.png", dpi=170); plt.close(fig)

    # (3) dimensionality of the common space (ROIs with V > 80 so every k is defined)
    big = nvox > max(klist)
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for j in range(len(clf)):
        dx = (j - (len(clf) - 1) / 2) * 0.05         # small horizontal offset so coinciding markers stay visible
        ax[0].plot(np.arange(len(klist) + 1) + dx, list(np.nanmean(R[big][:, :, :, j], axis=(0, 2))) + [h[big, j].mean()],
                   marker="o", ms=4.5, lw=1.3, color=COL[j], label=clf[j])
        ax[0].plot([len(klist) + 1 + dx], [a[big, j].mean()], marker="s", ms=5, ls="none", color=COL[j])
    ax[0].axvline(len(klist) + 0.5, color="0.8", lw=0.8)
    ax[0].set_xticks(list(range(len(klist) + 2))); ax[0].set_xticklabels([str(k) for k in klist] + ["all V", "anatomical\n(no alignment)"])
    ax[0].axhline(CH7, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("Common-space dimensions k (top PCs of the training template)"); ax[0].set_ylabel("Accuracy (%)")
    ax[0].set_title(f"Mean over the {int(big.sum())} ROIs with V > {max(klist)} (dashed: chance 14.29 %)", fontsize=10)
    ax[0].set_ylim(10, 72); ax[0].legend(fontsize=7.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.06))
    ax[0].spines[["top", "right"]].set_visible(False); ax[1].spines[["top", "right"]].set_visible(False)
    top = np.argsort(-h.mean(1))[:6]
    for k in top:
        ax[1].plot(range(len(klist) + 1), list(np.nanmean(R[k], axis=(1, 2))) + [h[k].mean()], marker="o", label=labels[k])
    ax[1].set_xticks(range(len(klist) + 1)); ax[1].set_xticklabels([str(k) for k in klist] + ["all V"])
    ax[1].set_xlabel("Common-space dimensions k"); ax[1].set_ylabel("Accuracy, mean of 7 classifiers (%)")
    ax[1].set_title("Six best hyperaligned ROIs", fontsize=10)
    ax[1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))       # outside the axes
    fig.tight_layout(); fig.savefig(FIGS / "part2b_dimensionality.png", dpi=170); plt.close(fig)

    def wil(x):
        try: return float(wilcoxon(x).pvalue)
        except ValueError: return 1.0
    # each split direction separately, and per-ROI paired tests over the 18 held-out subjects (FDR)
    Ah, Hh = z["anat"] * 100, z["hyper"] * 100                       # (52, 2, 18, 7)
    pr = np.array([ttest_1samp(gm[k], 0).pvalue for k in range(52)])
    srt = np.sort(pr); ok = srt <= 0.05 * np.arange(1, 53) / 52
    thr = srt[np.nonzero(ok)[0].max()] if ok.any() else 0
    subj_gain = gm.mean(0)
    # relative Procrustes residual on the alignment half: (ROI, half, fold, [train mean, held-out]) -> ROI means
    res = z["procrustes_resid"].mean((1, 2))                         # (52, 2)
    T_ALIGN = 140
    wide = nvox > T_ALIGN                                            # more voxels than alignment TRs
    rng_ = lambda v: [float(v.min()), float(v.max())]
    top6 = np.argsort(-h.mean(1))[:6]
    k20 = klist.index(20)
    out = {"clf": clf, "k_list": [int(k) for k in klist],
           "procrustes_residual": {
               "definition": "||X R - T||_F^2 / ||X||_F^2 on the alignment half, mean over folds and both splits",
               "n_align_TR": T_ALIGN, "n_rois_V_gt_T": int(wide.sum()),
               "train_range_all": rng_(res[:, 0]), "heldout_range_all": rng_(res[:, 1]),
               "train_range_V_gt_T": rng_(res[wide, 0]), "heldout_range_V_gt_T": rng_(res[wide, 1]),
               "min_any_fold": float(z["procrustes_resid"].min())},
           "top6_hyper_k20_vs_full": [{"roi": labels[k], "k20": float(np.nanmean(R[k, k20])), "full": float(h[k].mean())} for k in top6],
           "by_half": {"anat": Ah.mean((0, 2)).tolist(), "hyper": Hh.mean((0, 2)).tolist()},
           "n_sig_gain_fdr": int(((pr <= thr) & (gm.mean(1) > 0)).sum()), "n_sig_loss_fdr": int(((pr <= thr) & (gm.mean(1) < 0)).sum()),
           "sig_gain_rois": [labels[k] for k in order if pr[k] <= thr and gm[k].mean() > 0],
           "sig_loss_rois": [labels[k] for k in order if pr[k] <= thr and gm[k].mean() < 0],
           "subj_gain_mean": float(subj_gain.mean()), "subj_gain_p": float(ttest_1samp(subj_gain, 0).pvalue),
           "anat_mean": a.mean(0).tolist(), "hyper_mean": h.mean(0).tolist(),
           "gain_mean": (h - a).mean(0).tolist(), "n_improved": (h > a).sum(0).tolist(),
           "p_wilcoxon_rois": [wil(h[:, j] - a[:, j]) for j in range(len(clf))],
           "overall_anat": float(a.mean()), "overall_hyper": float(h.mean()),
           "n_rois_improved_mean": int((gm.mean(1) > 0).sum()),
           "p_wilcoxon_overall": wil(gm.mean(1)),
           "corr_gain_logV": float(np.corrcoef(np.log(nvox), gm.mean(1))[0, 1]),
           "corr_gain_anat": float(np.corrcoef(a.mean(1), gm.mean(1))[0, 1]),
           "gain_bigV": float(gm.mean(1)[nvox > 139].mean()), "gain_smallV": float(gm.mean(1)[nvox <= 139].mean()),
           "n_bigV": int((nvox > 139).sum()),
           "top_gain": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat": float(a[k].mean()), "hyper": float(h[k].mean()),
                         "gain": float(gm[k].mean()), "sem": float(sem(gm, 1)[k]), "anat_clf": a[k].tolist(), "hyper_clf": h[k].tolist()} for k in order[:10]],
           "worst_gain": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat": float(a[k].mean()), "hyper": float(h[k].mean()),
                           "gain": float(gm[k].mean())} for k in order[-5:]],
           "top_hyper": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat_clf": a[k].tolist(), "hyper_clf": h[k].tolist()} for k in np.argsort(-h.mean(1))[:5]],
           "reduced_mean_bigrois": {str(int(k)): np.nanmean(R[big][:, i], axis=(0, 1)).tolist() for i, k in enumerate(klist)},
           "hyper_mean_bigrois": h[big].mean(0).tolist(), "anat_mean_bigrois": a[big].mean(0).tolist(), "n_big": int(big.sum()),
           "table": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat": a[k].tolist(), "hyper": h[k].tolist(),
                      "gain": float(gm[k].mean()), "gain_sem": float(sem(gm, 1)[k]), "gain_p": float(pr[k]),
                      "gain_sig_fdr": bool(pr[k] <= thr),
                      "resid_train": float(res[k, 0]), "resid_heldout": float(res[k, 1])} for k in range(52)]}
    json.dump(out, open(RESULTS / "part2b_summary.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    o = part2a(); print("2A", dict(zip(o["clf"], np.round(o["clf_mean"], 1))), o["n_sig_rois_fdr"], o["n_rois_best"], o["corr_logV_acc"])
    if (RESULTS / "part2b_accuracy.npz").exists():
        o = part2b(); print("2B", {k: v for k, v in o.items() if k not in ("table", "top_gain", "top_hyper", "worst_gain")})
