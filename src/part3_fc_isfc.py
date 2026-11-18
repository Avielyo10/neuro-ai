"""Part 3 - whole-brain seed-based FC and ISFC maps for the top-3 decoding ROIs.

Seed time course  : mean of the z-scored voxel time courses of the seed ROI (per subject).
FC   (within)     : r(seed of subject i, every voxel of subject i), averaged over the 18 subjects.
ISFC (between)    : r(seed of subject i, every voxel of the AVERAGE of the other 17 subjects),
                    averaged over the 18 subjects (leave-one-out ISFC, Simony et al., 2016).
This is a Python re-implementation of the course helpers run_isfc_p2m.m / isfc_p2m_nifti.m
(same signal threshold of 6000 and the same 61 x 73 x 61 output grid); the leave-one-out averages
are computed here instead of being read from the `avg` folder.  Subject-level r maps are averaged
after Fisher z-transformation and transformed back to r.

Run:  python src/part3_fc_isfc.py     (after part2a_decoding.py)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting

from common import (CACHE, FIGS, GRID, N_TR, RESULTS, SIGNAL_THRESHOLD, TEMPLATE, load_subject, load_voi,
                    roi_labels, subject_files, voi_to_flat_index, zscore)

MIN_OTHERS = 9      # ISFC needs a voxel to be valid in at least this many of the other 17 subjects
N_TOP = 3


def top_rois():
    """Top-3 ROIs of Part 2 (Task A): highest LOSO accuracy averaged over the 7 classifiers."""
    z = np.load(RESULTS / "part2a_accuracy.npz")
    score = z["acc"].mean(axis=(1, 2))
    return [int(k) for k in np.argsort(-score)[:N_TOP]], score


def load_whole_brain():
    """Z-scored data of all subjects on the union of valid voxels.

    Returns Z (18, 280, U) float32 with invalid entries set to 0, valid (18, U) bool, union (U,) flat idx.
    """
    cache = CACHE / "wholebrain.npz"
    if cache.exists():
        c = np.load(cache)
        return c["Z"], c["valid"], c["union"]
    raw, masks = [], []
    for f in subject_files():
        x = load_subject(f)
        with np.errstate(invalid="ignore"):
            m = np.isfinite(x).all(0) & (np.nan_to_num(x).mean(0) > SIGNAL_THRESHOLD) & (np.nan_to_num(x).std(0) > 0)
        raw.append(x); masks.append(m)
    masks = np.stack(masks)
    union = np.flatnonzero(masks.any(0))
    valid = masks[:, union]
    Z = np.zeros((len(raw), N_TR, len(union)), dtype=np.float32)
    for i, x in enumerate(raw):
        Z[i][:, valid[i]] = zscore(x[:, union[valid[i]]], axis=0)
    np.savez(cache, Z=Z, valid=valid, union=union)
    return Z, valid, union


def seed_maps(Z, valid, seed_cols):
    """Return (fc, isfc): subject-level correlation maps, each (18, U) with NaN where undefined."""
    n_sub, T, U = Z.shape
    S, n_valid = Z.sum(0), valid.sum(0)
    fc, isfc = np.full((n_sub, U), np.nan, np.float32), np.full((n_sub, U), np.nan, np.float32)
    for i in range(n_sub):
        cols = seed_cols[valid[i, seed_cols]]
        seed = zscore(Z[i][:, cols].mean(1))                       # (280,)
        fc[i, valid[i]] = seed @ Z[i][:, valid[i]] / T             # Pearson r (both z-scored)
        n_oth = n_valid - valid[i]
        ok = n_oth >= MIN_OTHERS
        others = zscore((S[:, ok] - Z[i][:, ok]) / n_oth[ok], axis=0)   # mean of the other subjects
        isfc[i, ok] = seed @ others / T
    return fc, isfc


def fisher_mean(r):
    return np.tanh(np.nanmean(np.arctanh(np.clip(r, -0.999999, 0.999999)), axis=0))


def to_img(values, union, ref):
    vol = np.zeros(np.prod(GRID), dtype=np.float32)
    vol[union] = np.nan_to_num(values)
    return nib.Nifti1Image(vol.reshape(GRID, order="F"), ref.affine)


def main():
    top, score = top_rois()
    labels, vois = roi_labels(), load_voi()
    ref = nib.load(str(TEMPLATE))
    Z, valid, union = load_whole_brain()
    print("whole-brain voxels (valid in >=1 subject):", len(union), "| valid in all 18:", int(valid.all(0).sum()))
    summary = {"top_rois": [labels[k] for k in top], "mean_acc_pct": [float(score[k] * 100) for k in top], "seeds": {}}
    imgs = {}
    for k in top:
        flat = voi_to_flat_index(vois[k][1])
        seed_cols = np.searchsorted(union, flat[np.isin(flat, union)])
        fc_s, isfc_s = seed_maps(Z, valid, seed_cols)
        fc, isfc = fisher_mean(fc_s), fisher_mean(isfc_s)
        tag = labels[k].replace(" ", "_")
        for name, m in (("FC", fc), ("ISFC", isfc)):
            imgs[(k, name)] = to_img(m, union, ref)
            nib.save(imgs[(k, name)], str(RESULTS / f"map_{name}_{tag}.nii.gz"))
        outside = np.ones(len(union), bool); outside[seed_cols] = False
        both = np.isfinite(fc) & np.isfinite(isfc) & outside
        com = nib.affines.apply_affine(ref.affine, np.array(np.unravel_index(flat, GRID, order="F")).T.mean(0))
        summary["seeds"][labels[k]] = {
            "n_seed_voxels": int(len(seed_cols)), "seed_center_mm": [float(c) for c in com],
            "fc_mean_r": float(fc[both].mean()), "isfc_mean_r": float(isfc[both].mean()),
            "fc_median_r": float(np.median(fc[both])), "isfc_median_r": float(np.median(isfc[both])),
            "fc_frac_gt_0.2": float((fc[both] > 0.2).mean()), "isfc_frac_gt_0.2": float((isfc[both] > 0.2).mean()),
            "fc_frac_lt_-0.1": float((fc[both] < -0.1).mean()), "isfc_frac_lt_-0.1": float((isfc[both] < -0.1).mean()),
            "fc_max_r_outside_seed": float(fc[both].max()), "isfc_max_r_outside_seed": float(isfc[both].max()),
            "spatial_corr_fc_isfc": float(np.corrcoef(fc[both], isfc[both])[0, 1]),
            "isfc_within_seed_mean_r": float(np.nanmean(isfc[seed_cols])),
            "fc_within_seed_mean_r": float(np.nanmean(fc[seed_cols])),
        }
        print(labels[k], json.dumps(summary["seeds"][labels[k]]))
    json.dump(summary, open(RESULTS / "part3_summary.json", "w"), indent=1)

    # ---- multi-planar figure per seed: FC (top) vs ISFC (bottom), same colour scale ----
    for k in top:
        com = summary["seeds"][labels[k]]["seed_center_mm"]
        fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), gridspec_kw={"height_ratios": [1.5, 1, 1.5, 1]})
        for r, name in enumerate(("FC", "ISFC")):
            long = ("FC - within-subject functional connectivity" if name == "FC"
                    else "ISFC - inter-subject functional correlation")
            plotting.plot_stat_map(imgs[(k, name)], bg_img=ref, threshold=0.15, vmax=0.6, cmap="cold_hot",
                                   display_mode="ortho", cut_coords=[round(c) for c in com], axes=axes[2 * r],
                                   figure=fig, title=f"{long}   (seed: {labels[k]})", draw_cross=True,
                                   black_bg=True, annotate=True)
            plotting.plot_stat_map(imgs[(k, name)], bg_img=ref, threshold=0.15, vmax=0.6, cmap="cold_hot",
                                   display_mode="z", cut_coords=[-24, -9, 6, 21, 36, 51, 63], axes=axes[2 * r + 1],
                                   figure=fig, black_bg=True, colorbar=False)
        fig.savefig(FIGS / f"part3_maps_{labels[k].replace(' ', '_')}.png", dpi=170, facecolor="k")
        plt.close(fig)


if __name__ == "__main__":
    main()
