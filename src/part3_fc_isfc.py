"""Part 3 - whole-brain seed-based FC and ISFC maps for the top-3 decoding ROIs.

Seed time course  : mean of the z-scored voxel time courses of the seed ROI (per subject).
FC   (within)     : r(seed of subject i, every voxel of subject i), averaged over the 18 subjects.
ISFC (between)    : r(seed of subject i, every voxel of the AVERAGE of the other 17 subjects),
                    averaged over the 18 subjects (leave-one-out ISFC, Simony et al., 2016).
This is a Python re-implementation of the course helpers run_isfc_p2m.m / isfc_p2m_nifti.m
(same signal threshold of 6000 and the same 61 x 73 x 61 output grid); the leave-one-out averages
are computed here instead of being read from the `avg` folder.  As in the helper, the group map is
the plain (arithmetic) mean of the subject-level r maps; the mean after Fisher z-transformation is
computed for comparison only.  helper_fc_map() is a line-by-line port of the helper's FC branch
(within = 1) working on the raw .mat files; it is used to validate the vectorised implementation.

Common mask: a subject's FC is defined wherever that subject has valid signal, but leave-one-out ISFC
needs enough other subjects.  So that FC and ISFC are drawn and summarised on the SAME voxels, both
group maps are restricted to voxels that are valid in at least MIN_OTHERS + 1 = 10 of the 18 subjects.

Memory: the whole-brain array is 18 x 280 x ~76,000 float32 (1.5 GB on disk, cache/wholebrain.npz);
building it from the .mat files needs about 8 GB of RAM.

Run:  python src/part3_fc_isfc.py     (after part2a_decoding.py)
"""
import json
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import nibabel as nib
import numpy as np
from nilearn import plotting
from PIL import Image

from common import (CACHE, FIGS, GRID, N_TR, RESULTS, SIGNAL_THRESHOLD, TEMPLATE, load_subject, load_voi,
                    roi_labels, subject_files, voi_to_flat_index, zscore)

MIN_OTHERS = 9      # ISFC needs a voxel to be valid in at least this many of the other 17 subjects
N_TOP = 3
DISPLAY_THR = 0.15  # maps are displayed for |r| > 0.15
EXTRA_SEEDS = ["08 PREC_L"]   # supplementary non-auditory seed (best default-mode ROI in Task A)
# Approximate anatomical probe locations (template mm).  FC / ISFC are averaged over the 3 x 3 x 3
# voxel cube (9 mm) centred on each coordinate; used only to put numbers on the verbal description
# of the maps (which regions belong to the FC pattern of which seed).
PROBES = {"insula L": (-39, 3, 0), "insula R": (39, 3, 0), "SMA / medial frontal": (0, -3, 57),
          "anterior cingulate": (0, 21, 33), "sensorimotor L": (-39, -24, 57), "thalamus L": (-12, -18, 6),
          "thalamus R": (12, -18, 6), "cerebellum L": (-21, -60, -27), "cerebellum R": (21, -60, -27),
          "IFG L": (-48, 21, 12), "precuneus": (0, -57, 33), "medial prefrontal": (0, 54, 6)}


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


def plain_mean(r):
    """Arithmetic mean of r over subjects, as nanmean(csub_2, 2) in isfc_p2m_nifti.m."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns -> NaN, masked later
        return np.nanmean(r.astype(np.float64), axis=0)


def helper_fc_map(seed_flat):
    """Line-by-line port of the FC branch (within = 1) of isfc_p2m_nifti.m, from the raw files.

    seed_flat: flat indices of the seed voxels on the 61 x 73 x 61 grid (the helper's `roimask`).
    Returns the group map on the full grid (NaN where no subject has valid signal).
    """
    files = subject_files()
    csub = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for isub, f in enumerate(files):
            bold = load_subject(f).astype(np.float64)               # bold_one_temp: (280, Nvox)
            mask_single = bold.mean(0) > SIGNAL_THRESHOLD           # mean(bold_one_temp) > Bth
            if csub is None:
                csub = np.full((bold.shape[1], len(files)), np.nan)  # csub_2
            bold[:, ~mask_single] = np.nan
            roi = bold[:, seed_flat]                                # seed region, z-scored per voxel
            roi = (roi - roi.mean(0)) / roi.std(0, ddof=1)
            b = np.nanmean(roi, axis=1)                             # bold_one
            ns = len(b)
            good = np.flatnonzero(mask_single)                      # gg_avg
            b_avg = bold[:, good]
            b = (b - b.mean()) / (np.sqrt(ns - 1) * b.std(ddof=1))
            b_avg = (b_avg - b_avg.mean(0)) / (np.sqrt(ns - 1) * b_avg.std(0, ddof=1))
            csub[good, isub] = b_avg.T @ b                          # cc
        return np.nanmean(csub, axis=1)                             # avg_map_dmn


def fisher_mean(r):
    """Mean over subjects after Fisher z-transformation (NaN where no subject has a value)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns -> NaN, masked later
        return np.tanh(np.nanmean(np.arctanh(np.clip(r, -0.999999, 0.999999)), axis=0))


def typographic_minus(fig):
    """Replace the ASCII hyphen in negative numbers (nilearn's cut labels and colour-bar ticks) by a
    real minus sign."""
    fig.canvas.draw()
    for ax in fig.axes:                                       # colour bars: freeze the tick labels
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if any(re.search(r"-\d", l) for l in labels):
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([re.sub(r"-(?=\d)", "\u2212", l) for l in labels])
    for t in fig.findobj(mtext.Text):
        if re.search(r"-\d", t.get_text()):
            t.set_text(re.sub(r"-(?=\d)", "\u2212", t.get_text()))


def probe_cols(xyz, union, ref):
    """Columns (into `union`) of the 3 x 3 x 3 voxel cube centred on template coordinate xyz (mm)."""
    c = np.round(nib.affines.apply_affine(np.linalg.inv(ref.affine), xyz)).astype(int)
    off = np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    ijk = c + off
    ijk = ijk[np.all((ijk >= 0) & (ijk < np.array(GRID)), axis=1)]
    flat = np.ravel_multi_index(ijk.T, GRID, order="F")
    return np.searchsorted(union, flat[np.isin(flat, union)])


def to_img(values, union, ref):
    vol = np.zeros(np.prod(GRID), dtype=np.float32)
    vol[union] = np.nan_to_num(values)
    return nib.Nifti1Image(vol.reshape(GRID, order="F"), ref.affine)


def main():
    top, score = top_rois()
    labels, vois = roi_labels(), load_voi()
    ref = nib.load(str(TEMPLATE))
    Z, valid, union = load_whole_brain()
    keep = valid.sum(0) >= MIN_OTHERS + 1          # common FC / ISFC mask: valid in >= 10 subjects
    print("whole-brain voxels (valid in >=1 subject):", len(union), "| valid in all 18:", int(valid.all(0).sum()),
          f"| common map mask (valid in >= {MIN_OTHERS + 1}):", int(keep.sum()))
    seeds_all = top + [labels.index(e) for e in EXTRA_SEEDS]
    rank = [int(k) for k in np.argsort(-score)]
    summary = {"min_valid_subjects": MIN_OTHERS + 1, "n_voxels_map_mask": int(keep.sum()),
               "n_voxels_valid_any_subject": int(len(union)), "n_voxels_valid_all_subjects": int(valid.all(0).sum()),
               "threshold_display": DISPLAY_THR,
               "extra_seed_rank": {e: rank.index(labels.index(e)) + 1 for e in EXTRA_SEEDS},
               "extra_seeds": EXTRA_SEEDS, "top_rois": [labels[k] for k in top], "mean_acc_pct": [float(score[k] * 100) for k in top], "rank4": [labels[int(np.argsort(-score)[3])], float(np.sort(score)[-4] * 100)], "seeds": {}}
    imgs = {}
    for k in seeds_all:
        flat = voi_to_flat_index(vois[k][1])
        seed_cols = np.searchsorted(union, flat[np.isin(flat, union)])
        fc_s, isfc_s = seed_maps(Z, valid, seed_cols)
        fc, isfc = plain_mean(fc_s), plain_mean(isfc_s)   # helper convention: arithmetic mean of r
        fc[~keep] = np.nan; isfc[~keep] = np.nan    # same voxels in both maps
        assert np.array_equal(np.isfinite(fc), keep) and np.array_equal(np.isfinite(isfc), keep)
        # for comparison only: mean over subjects after Fisher z-transformation
        d_plain = max(float(np.max(np.abs(fisher_mean(m)[keep] - g[keep])))
                      for m, g in ((fc_s, fc), (isfc_s, isfc)))
        if k == top[0]:                             # validation against the port of the course helper
            ref_map = helper_fc_map(flat)[union]
            assert np.isfinite(ref_map[keep]).all()
            summary["helper_check"] = {"seed": labels[k], "n_voxels": int(keep.sum()),
                                       "max_abs_diff_fc": float(np.max(np.abs(ref_map[keep] - fc[keep])))}
            print("FC map vs. port of isfc_p2m_nifti.m:", summary["helper_check"])
        tag = labels[k].replace(" ", "_")
        for name, m in (("FC", fc), ("ISFC", isfc)):
            imgs[(k, name)] = to_img(m, union, ref)
            nib.save(imgs[(k, name)], str(RESULTS / f"map_{name}_{tag}.nii.gz"))
        outside = np.ones(len(union), bool); outside[seed_cols] = False
        both = keep & outside
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
            "n_seed_voxels_in_mask": int(keep[seed_cols].sum()),
            "n_fc_gt_thr": int((fc[keep] > DISPLAY_THR).sum()), "n_isfc_gt_thr": int((isfc[keep] > DISPLAY_THR).sum()),
            "n_fc_lt_neg_thr": int((fc[keep] < -DISPLAY_THR).sum()), "n_isfc_lt_neg_thr": int((isfc[keep] < -DISPLAY_THR).sum()),
            "max_abs_diff_fisher_vs_plain_mean": d_plain,
            "probes": {name: {"mm": list(xyz), "n_vox": int(keep[c].sum()),
                              "fc": float(np.nanmean(fc[c])), "isfc": float(np.nanmean(isfc[c]))}
                       for name, xyz in PROBES.items() for c in [probe_cols(xyz, union, ref)]},
        }
        print(labels[k], json.dumps({a: b for a, b in summary["seeds"][labels[k]].items() if a != "probes"}))
        print("   probes FC/ISFC:", "; ".join(f"{n} {v['fc']:.2f}/{v['isfc']:.2f}" for n, v in summary["seeds"][labels[k]]["probes"].items()))
    json.dump(summary, open(RESULTS / "part3_summary.json", "w"), indent=1)

    # ---- multi-planar figure per seed: FC (top) vs ISFC (bottom), same colour scale ----
    for k in seeds_all:
        com = summary["seeds"][labels[k]]["seed_center_mm"]
        fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), gridspec_kw={"height_ratios": [1.5, 1, 1.5, 1]})
        for r, name in enumerate(("FC", "ISFC")):
            long = ("FC \u2013 within-subject functional connectivity" if name == "FC"
                    else "ISFC \u2013 inter-subject functional correlation")
            disp = plotting.plot_stat_map(imgs[(k, name)], bg_img=ref, threshold=DISPLAY_THR, vmax=0.6, cmap="cold_hot",
                                          display_mode="ortho", cut_coords=[round(c) for c in com], axes=axes[2 * r],
                                          figure=fig, draw_cross=True, black_bg=True, annotate=True)
            disp.title(f"{long}   (seed: {labels[k]})", size=20)     # small enough not to cover the L/R labels
            plotting.plot_stat_map(imgs[(k, name)], bg_img=ref, threshold=DISPLAY_THR, vmax=0.6, cmap="cold_hot",
                                   display_mode="z", cut_coords=[-24, -9, 6, 21, 36, 51, 63], axes=axes[2 * r + 1],
                                   figure=fig, black_bg=True, colorbar=False)
        path = FIGS / f"part3_maps_{labels[k].replace(' ', '_')}.png"
        typographic_minus(fig)
        fig.savefig(path, dpi=170, facecolor="k")
        plt.close(fig)
        im = Image.open(path).convert("RGB")                       # trim the empty black margins
        box = im.point(lambda v: 255 if v > 12 else 0).getbbox()
        im.crop((max(box[0] - 20, 0), max(box[1] - 20, 0), min(box[2] + 20, im.width), min(box[3] + 20, im.height))).save(path)


if __name__ == "__main__":
    main()
