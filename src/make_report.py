"""Build report/report.html (and report/Homework1_Report.pdf via headless Chromium) from results/*.

Every number in the prose is read or computed from the result files.  Verbal statements that depend
on the data (which ROIs lead, signs of effects, ...) are guarded by `check(...)`: if a re-run makes a
sentence untrue, the build stops with a message instead of printing a wrong report.

The PDF step needs node and the `playwright` npm package with its Chromium (see README).
"""
import base64
import json
import re
import subprocess

import numpy as np

from common import FIGS, RESULTS, ROOT, TR, load_voi, voi_to_flat_index

# Author line of the title block.  Fill in before submitting, e.g.
#     AUTHORS = "Jane Doe (ID 123456789), John Roe (ID 987654321)"
# While it is empty, no "Submitted by" line is printed (one blank line is reserved in the title block,
# so that filling in a one-line author string does not move any page break).
AUTHORS = ""

t0 = json.load(open(RESULTS / "task0_alignment.json"))
A = json.load(open(RESULTS / "part2a_summary.json"))
B = json.load(open(RESULTS / "part2b_summary.json"))
P3 = json.load(open(RESULTS / "part3_summary.json"))
LD = json.load(open(RESULTS / "supp_lda_sensitivity.json"))
TS = json.load(open(RESULTS / "supp_temporal_vs_spatial.json"))
clf = A["clf"]
jS, jN, jL, jQ = (clf.index(c) for c in ("Linear SVM", "Naive Bayes", "LDA", "QDA"))
jK = [clf.index(f"KNN (K={k})") for k in (1, 2, 3)]
MINUS = "−"


# ---------------------------------------------------------------- helpers
def check(cond, msg):
    if not cond:
        raise SystemExit(f"make_report: the text no longer matches the data -> {msg}")


def n(x, spec=".1f"):
    """Format a number with a typographic minus sign (a value that rounds to zero is printed without a minus)."""
    txt = format(x, spec)
    if float(txt) == 0:
        txt = format(0.0, spec)
    return txt.replace("-", MINUS)


def span(vals, spec=".1f"):
    """'lo-hi' of a list of numbers (en dash); 'lo to hi' with explicit signs when lo is negative."""
    lo, hi = min(vals), max(vals)
    if lo < 0:
        return f"{n(lo, spec)} to {n(hi, '+' + spec if hi > 0 else spec)}"
    return f"{n(lo, spec)}&ndash;{n(hi, spec)}"


def pval(p):
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def psci(p):
    """p-value with two significant digits; scientific notation as a x 10^b below 0.001."""
    if p >= 0.001:
        return f"{p:.2g}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}&times;10<sup>{MINUS}{abs(int(exp))}</sup>"


def img(name, width="100%"):
    b = base64.b64encode(open(FIGS / name, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def short(label):
    """'48 A1_R' -> 'A1_R' (ROI name without the running index)."""
    return label.split(" ", 1)[1]


# numbering: order of appearance in the document (verified against the HTML below)
TAB = {k: i + 1 for i, k in enumerate(["subj", "clf", "top", "lda", "tvs", "hyp", "gain", "half", "dim", "fc"])}
FIG = {k: i + 1 for i, k in enumerate(["xcorr", "tc", "bars", "heat", "scatter", "gainroi", "dim",
                                       "map1", "map2", "map3", "map4"])}

# ---------------------------------------------------------------- Methods numbers
vois = load_voi()
n_mapped = [len(voi_to_flat_index(c)) for _, c in vois]
n_kept = [r["n_vox"] for r in A["table"]]
lost = [(A["table"][k]["roi"], n_mapped[k] - n_kept[k]) for k in range(len(vois)) if n_mapped[k] != n_kept[k]]
n_lost = sum(d for _, d in lost)
v_max, v_min = max(n_kept), min(n_kept)
n_train = 17 * 14
n_narrow = sum(20 * v < n_train for v in n_kept)          # ROIs whose feature count is below 238
sh = LD["shrinkage"]
fisher_diff = max(s["max_abs_diff_fisher_vs_plain_mean"] for s in P3["seeds"].values())
hc = P3["helper_check"]
check(hc["max_abs_diff_fc"] < 1e-6, "FC map no longer agrees with the port of the course helper")
hc_txt = "10\u207b\u2076"

# ---------------------------------------------------------------- Task 0
raw, hrf = t0["raw"], t0["hrf"]
sl, sr = np.array(raw["subj_peak_lag_TR"]), np.array(raw["subj_peak_r"])
hl = np.array(hrf["subj_peak_lag_TR"])
sp = np.array(raw["subj_p_phase"])
nsig = int((sp < 0.05).sum())
lag = raw["group_peak_lag_TR"]
r_by_lag = dict(zip(raw["lags_TR"], raw["group_r_by_lag"]))
off = t0["window_offset_scan"]
best_off = off["best_offset_TR"]
r_off = dict(zip(off["offset_TR"], off["r"]))
alt = t0["alt_scan_start"]
vz = t0["voxel_zscore_first"]
vz_min = f"{np.floor(vz['min_subj_corr_with_primary'] * 100) / 100:.2f}"
check(vz["raw_group_peak_lag_TR"] == t0["raw"]["group_peak_lag_TR"] and vz["min_subj_corr_with_primary"] > 0.95,
      "Task 0: the two A1 averaging conventions no longer agree")
hemi = t0["hemispheres"]
n_lags = len(raw["lags_TR"])
check(raw["group_n_exceed_phase"] == 0 and raw["group_n_exceed_circ"] == 0, "a surrogate reached the group peak")
check(best_off == t0["scan_start_TR"], "offset scan no longer peaks at the chosen crop")
check(lag == 3 and r_by_lag[lag + 1] > 0.9 * raw["group_peak_r"] and alt["raw_group_peak_lag_TR"] == lag + 1,
      "group peak is no longer between +3 and +4 TRs")
check(4 <= raw["group_peak_lag_s"] <= 6 and sl.min() > 0, "peak lag outside the 4-6 s range or not positive in all subjects")
check(hrf["group_peak_lag_TR"] == 0 and hrf["group_peak_r"] > raw["group_peak_r"], "HRF convolution result changed")
hemi_lags = sorted(h["peak_lag_TR"] for h in hemi.values())
check(hemi_lags == [lag, lag + 1], "hemispheres no longer peak at +3 and +4 TRs")
hemi_txt = " and ".join(f"{short(k)}{' peaks' if i == 0 else ''} at +{h['peak_lag_TR']} TRs with r = {h['peak_r']:.2f}"
                        for i, (k, h) in enumerate(hemi.items()))
N_SUBJ_COLS = 3                                           # per-subject table: three blocks side by side


def subj_table(idx):
    rows = "".join(f"<tr><td>{t0['subjects'][i]}</td><td>+{sl[i]}</td><td>{sl[i] * TR:.1f}</td><td>{sr[i]:.2f}</td>"
                   f"<td>{pval(sp[i])}</td></tr>" for i in idx)
    return ('<table style="width:32.6%;display:inline-table;vertical-align:top"><tr><th>Subject</th><th>Lag (TRs)</th>'
            f"<th>Lag (s)</th><th>Peak r</th><th>p</th></tr>{rows}</table>")


_per = -(-len(t0["subjects"]) // N_SUBJ_COLS)
subj_tables = " ".join(subj_table(range(a, min(a + _per, len(t0["subjects"])))) for a in range(0, len(t0["subjects"]), _per))


# ---------------------------------------------------------------- Task A
top4 = [short(r["roi"]) for r in A["top10"][:4]]
check(set(top4) == {"A1_R", "A1_L", "AUB_L", "AUB_R"}, "the four leading ROIs are no longer the four auditory ROIs")
best1 = A["best_single"]
check(best1["clf"] == "LDA" and best1["roi"] == A["top10"][0]["roi"], "best single result is no longer LDA in the top ROI")
knn_means = [A["clf_mean"][j] for j in jK]
check(max(knn_means) < min(A["clf_mean"][j] for j in (jS, jN, jL, jQ)) - 5, "KNN is no longer clearly the weakest family")
check(A["clf_mean"][jK[1]] < min(A["clf_mean"][jK[0]], A["clf_mean"][jK[2]]), "K = 2 is no longer the weakest KNN")
check(A["clf_mean"][jS] > A["clf_mean"][jL] > A["clf_mean"][jN] > A["clf_mean"][jQ], "order of classifier means changed")
check(all(A["n_sig_rois_fdr"][j] == 52 for j in (jS, jN, jL, jQ)), "not all ROIs above chance for SVM/NB/LDA/QDA")
check(A["n_rois_best"][jL] > A["n_rois_best"][jN] > A["n_rois_best"][jS], "order of '# ROIs where best' changed")
check(LD["n_rois_lda_best"]["span_lw"] == A["n_rois_best"][jL] and abs(LD["mean_over_rois"]["span_lw"] - A["clf_mean"][jL]) < 1e-9,
      "supp_lda_sensitivity.json is out of date (re-run supp_lda_sensitivity.py)")
check(len({tuple(v) for v in LD["top3_rois"].values()}) == 1, "top-3 ROIs differ between LDA variants")
check(abs(TS["mean_over_rois"]["full"] - A["clf_mean"][jS]) < 1e-9, "supp_temporal_vs_spatial.json is out of date")
bottom_v = [r["n_vox"] for r in A["bottom5"]]
tied_txt = "; ".join(f"{short(t['roi'])}: {' = '.join(t['tied'])}" for t in A["rois_best_tied"])
hdr = "".join(f"<th>{c}</th>" for c in clf)
top_rows = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td>"
    + "".join(f"<td{' class=b' if a > max(r['acc']) - 1e-6 else ''}>{a:.1f} ± {s:.1f}</td>" for a, s in zip(r["acc"], r["sem"]))
    + f"<td>{r['mean']:.1f}</td></tr>" for r in A["top10"])
clf_rows = "".join(
    f"<tr><td>{c}</td><td>{A['clf_mean'][j]:.1f}</td><td>{A['clf_median'][j]:.1f}</td><td>{A['n_rois_best'][j]}</td>"
    f"<td>{A['n_sig_rois_fdr'][j]}/52</td></tr>" for j, c in enumerate(clf))

# LDA regulariser sensitivity (supp_lda_sensitivity.py)
LDA_ROIS = ["48 A1_R", "08 PREC_L", "12 pSTS_R", "15 aMTG_R"]       # large auditory / largest DMN / mid-size / small
ld_tab = {r["roi"]: r for r in LD["table"]}
ld_names = {"span_lw": "Main model: LW in the PCA span of the training fold",
            "voxel_lw": "Exact pooled LW in the 20&middot;V voxel space",
            "sklearn_auto": "scikit-learn LDA, <code>lsqr</code> + <code>shrinkage=\"auto\"</code>, raw features"}
lda_rows = "".join(
    f"<tr><td>{ld_names[v]}</td><td>{LD['mean_over_rois'][v]:.1f}</td>"
    + "".join(f"<td>{ld_tab[r][v]:.1f}</td>" for r in LDA_ROIS)
    + f"<td>{LD['n_rois_lda_best'][v]}</td><td>{LD['n_rois_best_by_clf'][v][jS]} / {LD['n_rois_best_by_clf'][v][jN]}</td></tr>"
    for v in ("span_lw", "sklearn_auto", "voxel_lw"))
lda_hdr = "".join(f"<th style='white-space:nowrap'>{short(r)}<br>(V = {ld_tab[r]['n_vox']})</th>" for r in LDA_ROIS)
dsk, dvx = LD["diff_vs_span_lw_pp"]["sklearn_auto"], LD["diff_vs_span_lw_pp"]["voxel_lw"]
check(abs(LD["mean_over_rois"]["sklearn_auto"] - LD["mean_over_rois"]["span_lw"]) < 1.0, "sklearn-auto LDA no longer close to the main model")
check(LD["mean_over_rois"]["voxel_lw"] < LD["mean_over_rois"]["span_lw"] - 3 and LD["n_rois_lda_best"]["voxel_lw"] < 10,
      "voxel-space LW LDA no longer clearly worse")
check(LD["clf_mean"]["voxel_lw"][jL] < A["clf_mean"][jQ], "voxel-space LW LDA is no longer below QDA on average")

# temporal profile vs spatial pattern (supp_temporal_vs_spatial.py)
TS_ROIS = ["48 A1_R", "47 IFG_L_UNIQUE", "08 PREC_L", "39 PREC_L", "12 pSTS_R"]
ts_tab = {r["roi"]: r for r in TS["table"]}
ts_rows = "".join(
    f"<tr><td>{r}</td><td>{ts_tab[r]['n_vox']}</td>"
    + "".join(f"<td>{ts_tab[r][s]:.1f} ± {ts_tab[r][s + '_sem']:.1f}</td>" for s in ("full", "temporal", "spatial")) + "</tr>"
    for r in TS_ROIS) + "<tr><td><b>Mean over 52 ROIs</b></td><td></td>" + "".join(
    f"<td><b>{TS['mean_over_rois'][s]:.1f}</b></td>" for s in ("full", "temporal", "spatial")) + "</tr>"
tsm = TS["mean_over_rois"]
ts_aud = [ts_tab[r["roi"]]["temporal"] for r in A["top10"][:4]]
ts_exc = [r for r in TS["table"] if r["spatial"] > r["temporal"]]
check(TS["n_rois_temporal_gt_spatial"] >= 40 and tsm["temporal"] > tsm["spatial"] + 5, "temporal features no longer dominate")
check(all(("PREC" in r["roi"]) or ("IPL" in r["roi"]) for r in ts_exc) and len(ts_exc) == TS["n_rois_spatial_gt_temporal"],
      "ROIs where the spatial pattern beats the mean time course are no longer precuneus / IPL")
check(abs(tsm["temporal_z"] - tsm["temporal"]) < 1.5 and abs(tsm["spatial_z"] - tsm["spatial"]) < 1.5
      and TS["n_rois_temporal_z_gt_spatial_z"] >= 40, "re-standardised variant no longer agrees")
n_prec_exc = sum("PREC" in r["roi"] for r in ts_exc)
check(TS_ROIS[0] == A["top10"][0]["roi"] and ts_tab[TS_ROIS[1]]["n_vox"] == v_max and TS_ROIS[2] == A["rank_order"][4]
      and TS_ROIS[3] in [r["roi"] for r in ts_exc], "example ROIs of the temporal/spatial table no longer fit their description")

# ---------------------------------------------------------------- Task B
bh = B["by_half"]
bh_rows = "".join(
    f"<tr><td>{c}</td><td>{bh['anat'][0][j]:.1f}</td><td>{bh['hyper'][0][j]:.1f}</td><td>{n(bh['hyper'][0][j] - bh['anat'][0][j], '+.1f')}</td>"
    f"<td>{bh['anat'][1][j]:.1f}</td><td>{bh['hyper'][1][j]:.1f}</td><td>{n(bh['hyper'][1][j] - bh['anat'][1][j], '+.1f')}</td></tr>"
    for j, c in enumerate(clf))
bha = [float(np.mean(bh["anat"][h])) for h in (0, 1)]
bhh = [float(np.mean(bh["hyper"][h])) for h in (0, 1)]
b_rows = "".join(
    f"<tr><td>{c}</td><td>{B['anat_mean'][j]:.1f}</td><td>{B['hyper_mean'][j]:.1f}</td><td>{n(B['gain_mean'][j], '+.1f')}</td>"
    f"<td>{B['n_improved'][j]}/52</td><td>{B['n_tied'][j]}</td><td>{psci(B['p_wilcoxon_rois'][j])}</td></tr>" for j, c in enumerate(clf))
tie_txt = (f" ({B['n_rois_tied_mean']} exactly tied)" if B["n_rois_tied_mean"] else "")
N_GAIN = 8
gain_rows = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td><td>{r['anat']:.1f}</td><td>{r['hyper']:.1f}</td>"
    f"<td>{n(r['gain'], '+.1f')} ± {r['sem']:.1f}</td></tr>" for r in B["top_gain"][:N_GAIN])
btab = {r["roi"]: r for r in B["table"]}
loss_rows = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td><td>{r['anat']:.1f}</td><td>{r['hyper']:.1f}</td>"
    f"<td>{n(r['gain'], '+.1f')} ± {btab[r['roi']]['gain_sem']:.1f}</td></tr>" for r in B["worst_gain"][::-1])
k_rows = "".join(
    f"<tr><td>k = {k}</td>" + "".join(f"<td>{v:.1f}</td>" for v in B["reduced_mean_bigrois"][str(k)]) + "</tr>"
    for k in B["k_list"]) + "<tr><td>all V (full)</td>" + "".join(f"<td>{v:.1f}</td>" for v in B["hyper_mean_bigrois"]) \
    + "</tr><tr><td>anatomical</td>" + "".join(f"<td>{v:.1f}</td>" for v in B["anat_mean_bigrois"]) + "</tr>"

top_gain = B["top_gain"][:N_GAIN]
check(all(any(s in r["roi"] for s in ("STG", "STS", "MTG", "AUB", "TPJ")) for r in top_gain),
      "the largest gains are no longer all in superior/middle temporal or temporo-parietal ROIs")
check(all(r["roi"] in B["sig_gain_rois"] for r in top_gain), "one of the largest gains is not significant")
aud_svm = [btab[r["roi"]]["hyper"][jS] for r in A["top10"][:4]]
check(sorted(r["roi"] for r in B["top_hyper"][:4]) == sorted(r["roi"] for r in A["top10"][:4]),
      "the four auditory ROIs are no longer the four most accurate hyperaligned ROIs")
pi = [r for r in B["table"] if "PREC" in r["roi"] or "IPL" in r["roi"]]
pi_loss = sorted([r for r in pi if r["gain_sig_fdr"] and r["gain"] < 0], key=lambda r: r["gain"])
pi_rest = [r for r in pi if r not in pi_loss]
check(len(pi) == 8 and len(pi_loss) == 4 and not any(r["gain_sig_fdr"] for r in pi_rest),
      "precuneus / inferior-parietal pattern changed (expected 4 of 8 with a significant loss)")
check(set(B["sig_loss_rois"]) - {r["roi"] for r in pi_loss} == {"25 TP_L"}, "set of significant losses changed")
tp = btab["25 TP_L"]
pr = B["procrustes_residual"]
check(all(r["n_vox"] > pr["n_align_TR"] for r in pi_loss), "a losing precuneus/IPL ROI has V <= T")
check(pr["min_any_fold"] > 0.3, "Procrustes residual is close to zero in some fold")
par = [B["gain_mean"][j] for j in (jQ, jS, jL, jN)]
check(min(par) > 2.5 and max(abs(B["gain_mean"][j]) for j in jK) < 1.5 and max(B["gain_mean"][j] for j in jK) < 0.5,
      "pattern of classifier gains changed")
check(bhh[0] - bha[0] > 3 and abs(bhh[1] - bha[1]) < 1, "split-direction asymmetry changed")
check(abs(B["corr_gain_logV"]) < 0.15, "gain is now related to ROI size")
red20, redfull, redanat = B["reduced_mean_bigrois"]["20"], B["hyper_mean_bigrois"], B["anat_mean_bigrois"]
d20 = max(abs(red20[j] - redfull[j]) for j in (jS, jL, jQ))
check(d20 < 1.5 and all(red20[j] > redanat[j] for j in (jS, jL, jQ)), "k = 20 no longer matches the full common space")
nb_k = [B["reduced_mean_bigrois"][str(k)][jN] for k in B["k_list"]]
check(all(a > b for a, b in zip(nb_k, nb_k[1:])) and redfull[jN] > nb_k[0], "Naive Bayes vs k pattern changed")
d6 = [r["k20"] - r["full"] for r in B["top6_hyper_k20_vs_full"]]
k_max = max(B["k_list"])

# ---------------------------------------------------------------- Part 3
tops, extra = P3["top_rois"], P3["extra_seeds"][0]
seeds, xs, r4 = [P3["seeds"][t] for t in tops], P3["seeds"][extra], P3["rank4"]
a1 = [P3["seeds"][t] for t in tops if "A1" in t]
belt_name = [t for t in tops if "AUB" in t]
check(len(a1) == 2 and len(belt_name) == 1, "top-3 seeds are no longer two A1 ROIs and one belt ROI")
belt = P3["seeds"][belt_name[0]]
check(tops == [r["roi"] for r in A["top10"][:3]] and r4[0] == A["top10"][3]["roi"], "part3_summary.json is out of date")
check(short(r4[0]) == "AUB_R" and r4[0] in A["top3_by_best_single_clf"] and r4[0] in A["top3_by_clf"]["Linear SVM"],
      "AUB_R no longer enters the top 3 under the alternative criteria")
check(A["rank_order"][P3["extra_seed_rank"][extra] - 1] == extra and P3["extra_seed_rank"][extra] == 5,
      "the supplementary seed is no longer the best non-auditory ROI (rank 5)")
fc_fr = [100 * s["fc_frac_gt_0.2"] for s in seeds]
is_fr = [100 * s["isfc_frac_gt_0.2"] for s in seeds]
sc = [s["spatial_corr_fc_isfc"] for s in seeds]


def probe(seed_list, names, kind):
    return [s["probes"][p][kind] for s in seed_list for p in names]


INS, THA, CER = ["insula L", "insula R"], ["thalamus L", "thalamus R"], ["cerebellum L", "cerebellum R"]
SMA, IFG, PRE, MPF = ["SMA / medial frontal"], ["IFG L"], ["precuneus"], ["medial prefrontal"]
thr = P3["threshold_display"]
check(min(probe(a1, INS + THA + SMA + CER, "fc")) > thr, "A1 FC at insula/thalamus/SMA/cerebellum probes not above threshold")
check(max(probe([belt], INS + SMA, "fc")) < thr < min(probe([belt], IFG + PRE + MPF, "fc")), "belt-seed FC probe pattern changed")
check(max(probe(a1, INS, "isfc")) < 0.1 and max(probe(a1, THA + CER + SMA, "isfc")) < 0.2, "A1 ISFC probe pattern changed")
check(belt["probes"]["IFG L"]["isfc"] > 0.2 and belt["n_isfc_lt_neg_thr"] > 5 * max(s["n_isfc_lt_neg_thr"] for s in a1),
      "belt-seed ISFC pattern changed")
check(all(s["fc_within_seed_mean_r"] > s["isfc_within_seed_mean_r"] + 0.15 for s in seeds + [xs]), "FC inside seed not above ISFC")
check(all(s["n_fc_gt_thr"] > 1.5 * s["n_isfc_gt_thr"] for s in seeds), "FC maps no longer clearly more extensive")
p3_rows = "".join(
    f"<tr><td>{name}</td><td>{s['n_seed_voxels']}</td><td>{s['fc_within_seed_mean_r']:.2f} / {s['isfc_within_seed_mean_r']:.2f}</td>"
    f"<td>{s['fc_mean_r']:.3f} / {s['isfc_mean_r']:.3f}</td><td>{100 * s['fc_frac_gt_0.2']:.1f} % / {100 * s['isfc_frac_gt_0.2']:.1f} %</td>"
    f"<td>{s['fc_max_r_outside_seed']:.2f} / {s['isfc_max_r_outside_seed']:.2f}</td><td>{s['spatial_corr_fc_isfc']:.2f}</td></tr>"
    for name, s in P3["seeds"].items())

appA = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td>" + "".join(f"<td>{a:.1f} ± {s:.1f}</td>" for a, s in zip(r["acc"], r["sem"])) + "</tr>"
    for r in A["table"])
appB_anat = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td>" + "".join(f"<td>{a:.1f} ± {s:.1f}</td>" for a, s in zip(r["anat"], r["anat_sem"])) + "</tr>"
    for r in B["table"])
appB_hyper = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td>" + "".join(f"<td>{a:.1f} ± {s:.1f}</td>" for a, s in zip(r["hyper"], r["hyper_sem"]))
    + f"<td>{n(r['gain'], '+.1f')}</td></tr>" for r in B["table"])
author_line = ('<div class="meta author">' + (f"<b>Submitted by:</b> {AUTHORS}" if AUTHORS.strip() else "&nbsp;") + "</div>")


def mapfig(key, seed, text):
    return (f'<div class="fig">{img("part3_maps_" + seed.replace(" ", "_") + ".png", "66%")}'
            f'<div class="cap"><b>Figure {FIG[key]}.</b> {text}</div></div>')


CSS = f"""<!doctype html><html><head><meta charset="utf-8"><title>Homework 1 Report</title><style>
@page {{ size: A4; margin: 15mm 15mm 16mm; }}
body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 10pt; line-height: 1.35; color: #111; }}
h1 {{ font-size: 17pt; margin: 0 0 2pt; }} h2 {{ font-size: 13pt; border-bottom: 1.2px solid #1f3a5f; color: #1f3a5f; margin: 13pt 0 5pt; }}
h3 {{ font-size: 11pt; margin: 9pt 0 3pt; color: #1f3a5f; }} p {{ margin: 4pt 0; text-align: justify; orphans: 3; widows: 3; }}
h2, h3 {{ break-after: avoid; page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 7.9pt; margin: 5pt 0 3pt; break-inside: avoid; page-break-inside: avoid; }}
tr {{ break-inside: avoid; page-break-inside: avoid; }} table.small {{ break-inside: auto; page-break-inside: auto; }}
th {{ background: #1f3a5f; color: #fff; padding: 3px 4px; }} td {{ padding: 2px 4px; border-bottom: 0.5px solid #ccc; text-align: center; }}
td:first-child {{ text-align: left; white-space: nowrap; }} td.b {{ font-weight: bold; }}
.cap {{ font-size: 8.7pt; color: #333; margin: 2pt 0 8pt; text-align: justify; }}
.fig, .tbl {{ break-inside: avoid; page-break-inside: avoid; margin-top: 6pt; text-align: center; }}
.meta {{ font-family: Arial, sans-serif; font-size: 9pt; color: #444; }} .author {{ color: #111; }}
.pb {{ break-before: page; page-break-before: always; }} ul {{ margin: 3pt 0; padding-left: 16pt; }}
li {{ margin-bottom: 2pt; text-align: justify; }} code {{ font-size: 8.7pt; }} .small td {{ font-size: 6.9pt; padding: 1px 3px; }}
ol.refs {{ font-size: 8.8pt; color: #222; padding-left: 16pt; margin: 3pt 0; }} ol.refs li {{ margin-bottom: 1.5pt; text-align: left; }}
</style></head><body>
"""

# ---------------------------------------------------------------- content blocks
BLK = {}
BLK["title"] = f"""<h1>Brain State Decoding &amp; Functional Connectivity (FC / ISFC)</h1>
<div class="meta">NeuroAI: Bridging Neuroscience and Artificial Intelligence (2700759), Fall 2026 &middot; Homework Assignment 1<br>
Data: 18 subjects, "Pieman" story, TR = 1.5 s, 280 TRs, 52 ROIs &middot; Code: Python (<code>src/</code>, <code>scripts/run_all.sh</code>)</div>
{author_line}"""

BLK["methods"] = f"""<h2>1. Methodology</h2>
<p><b>Data and ROIs.</b> Each subject file holds <code>data_crop</code> (271,633 voxels &times; 280 TRs) on the 61&times;73&times;61, 3-mm template grid.
The 52 ROIs of <code>group2_all_52rois.voi</code> were mapped to voxel indices with the inverse template affine (as in the course example code; the file header labels the coordinates TAL, they are used as template mm coordinates).
Voxels were kept only if they carried valid signal in all 18 subjects (finite, mean &gt; 6000 &ndash; the helper scripts' threshold &ndash; and non-constant);
this removed {n_lost} of {sum(n_mapped):,} ROI voxels (ROIs {', '.join(l.split()[0] for l, _ in lost)}), leaving ROIs of {v_min} to {v_max} voxels. Several VOI names repeat in the file, so ROIs are referred to as "index name" (e.g. <i>48 A1_R</i>).
Every voxel time course was z-scored within subject.</p>
<p><b>Tensor reshaping and samples.</b> For an ROI with V voxels, each subject's 280&times;V matrix was reshaped to 14&times;20&times;V
(14 consecutive 30-s segments &times; 20 TRs &times; V voxels) and subjects were stacked into the 14&times;20&times;V&times;18 group tensor. One <i>sample</i> is one segment of one subject:
the 20&times;V spatio-temporal pattern, flattened to 20&middot;V features. The class label is the segment index (14 classes, chance 1/14 = 7.14 %).</p>
<p><b>LOSO cross-validation.</b> In each of 18 folds the classifiers were trained on 17 subjects (17&times;14 = {n_train} samples, 17 per class) and tested on the 14 segments of the held-out subject.
Reported values are the mean accuracy over the 18 folds &plusmn; SEM (SD/&radic;18).</p>
<p><b>Classifiers</b> (scikit-learn; all hyper-parameters fixed in advance, none tuned on test data).
<i>KNN</i>: Euclidean distance, K = 1, 2, 3; a tied vote goes to the lower class index (scikit-learn's default). <i>Linear SVM</i>: linear kernel, C = 1, one-vs-one (libsvm; the Gram matrix is precomputed, which is mathematically identical and faster).
<i>Naive Bayes</i>: Gaussian. <i>LDA and QDA</i>: features (up to 20&middot;{v_max} = {20 * v_max:,}) far outnumber training samples ({n_train}), so sample covariances are singular and both models need a regulariser.
<i>LDA</i> uses the pooled within-class covariance with Ledoit&ndash;Wolf (LW) shrinkage towards a scaled identity, estimated in the subspace spanned by the training samples
(PCA rotation fitted on the training fold, {n_train} dimensions except in the {n_narrow} ROIs with fewer than {n_train} features; the class means lie in this subspace; own implementation in <code>decoding.py</code>, double precision, equal priors).
The LW intensity and the target scale (trace/{n_train}) are therefore those of the subspace, not of the 20&middot;V voxel space. For wide ROIs this is a much stronger regulariser than LW applied in voxel space:
in the {sh['n_wide_rois']} ROIs with V &ge; {sh['wide_roi_min_voxels']} it corresponds to a voxel-space shrinkage intensity of {span(sh['a_span_voxel_equiv_range_wide'], '.2f')}, where voxel-space LW gives {span(sh['a_voxel_range_wide'], '.2f')}.
The LDA results depend on this choice; Section 3 (Table {TAB['lda']}) compares it with two alternatives.
<i>QDA</i>: class-specific covariances in the same subspace, each shrunk half-way towards a scaled identity (shrinkage 0.5; there are only 17 samples per class).</p>
<p><b>Hyperalignment (Task B).</b> Orthogonal Procrustes without scaling (<code>scipy.linalg.orthogonal_procrustes</code>, double precision), following Haxby et al. (2011):
(i) a reference subject (the first training subject) is taken as the initial template, each further subject is aligned to the evolving template, which is updated as the running mean;
(ii) every subject is re-aligned to that template and the re-aligned data are averaged into the final template;
(iii) each subject's final rotation R is its Procrustes fit to the final template. <b>Non-circular split-half design:</b> the rotations are estimated on one half of the story
(140 TRs) and applied to the other half, in which the 7 segments are decoded (chance 1/7 = 14.29 %); halves are then swapped and the two accuracies averaged. Voxels are z-scored within each half separately.
Within every LOSO fold the template is built from the 17 training subjects only; the held-out subject is brought into the common space by a rotation estimated from its own alignment half against that template.
The anatomical baseline is the identical 7-class split-half procedure with R = I. Optionally the common space was reduced to the top-k principal axes of the training template (k = {min(B['k_list'])}&ndash;{k_max}).</p>
<p><b>FC / ISFC (Part 3).</b> Python re-implementation of <code>run_isfc_p2m</code> / <code>isfc_p2m_nifti</code>. The seed time course is the mean of the z-scored seed voxels.
FC: Pearson r between a subject's seed and all of that subject's own voxels. ISFC: r between the seed of subject i and every voxel of the average of the other 17 subjects (leave-one-out; averages computed here, the <code>avg</code> folder was not used).
As in the helper, the group map is the arithmetic mean of the 18 subject maps of r (averaging after Fisher z-transformation instead changes no voxel by more than {fisher_diff:.2f}); the maps were written as NIfTI (<code>results/map_*.nii.gz</code>).
The MATLAB helper itself could not be run here (it needs MATLAB with NeuroElf), so its FC branch was ported line by line and run on the raw files for seed {hc['seed']}: the map agrees with the implementation used here to within {hc_txt} at all {hc['n_voxels']:,} mask voxels.
So that FC and ISFC are compared on identical voxels, both group maps are restricted to the {P3['n_voxels_map_mask']:,} voxels with valid signal in at least {P3['min_valid_subjects']} of the 18 subjects.</p>"""

BLK["task0"] = f"""<h2>2. Task 0 &ndash; Stimulus&ndash;brain alignment in A1</h2>
<p>A1 was defined as ROIs 48 A1_R + 52 A1_L ({t0['n_voxels_A1']} voxels). As the assignment specifies, a subject's A1 time course is the mean of the signal over these voxels, z-scored; the group time course is the z-scored mean over the 18 subjects. The audio envelope was |Hilbert(x)| of the mono signal, low-pass filtered at 8 Hz and averaged within 1.5-s bins.
The .wav lasts {t0['audio_s']:.0f} s ({t0['n_tr_bins_audio']} complete TR bins), longer than the 280-TR scan: it starts with ~13 s of music and ~2 s of silence before the story. Following the course example code, the envelope was cropped to the 280 bins starting {t0['scan_start_TR']} TRs ({t0['scan_start_TR'] * TR:.1f} s) into the file.
Cross-correlation was computed as MATLAB's <code>xcorr(&hellip;,'coeff')</code> for lags &minus;{t0['max_lag_TR']}&hellip;+{t0['max_lag_TR']} TRs (positive lag = BOLD follows the sound).</p>
<div class="fig">{img('task0_xcorr.png', '96%')}<div class="cap"><b>Figure {FIG['xcorr']}.</b> Cross-correlation between the audio envelope and A1 BOLD. Left: raw envelope. Middle: envelope convolved with the canonical (SPM double-gamma) HRF.
Grey: single subjects; red: group-average A1; dashed: 95th percentile of the surrogate peak distribution. Right: null distribution of the peak correlation (phase-randomised envelopes) with the observed group value.</div></div>
<div class="fig">{img('task0_timecourses.png', '74%')}<div class="cap"><b>Figure {FIG['tc']}.</b> Group-average A1 BOLD (red) against the TR-binned envelope (grey), before and after HRF convolution.</div></div>
<ul>
<li><b>Peak lag and correlation.</b> Group-average A1: peak at <b>+{lag} TRs ({raw['group_peak_lag_s']:.1f} s), r = {raw['group_peak_r']:.2f}</b>; the correlation at lag 0 is {n(raw['group_r_at_lag0'], '.2f')}.
The correlation at +{lag + 1} TRs is almost as high (r = {r_by_lag[lag + 1]:.2f}; taken alone, {hemi_txt}), so the delay lies between {lag} and {lag + 1} TRs.
Single subjects (Table {TAB['subj']}): peak lag {sl.mean():.1f} &plusmn; {sl.std(ddof=1):.1f} TRs (range {sl.min()}&ndash;{sl.max()}; mean &asymp; {sl.mean() * TR:.1f} s), peak r = {sr.mean():.2f} &plusmn; {sr.std(ddof=1):.2f}.</li>
<li><b>Significance.</b> The statistic is the maximum correlation over all {n_lags} lags, which corrects for the lag search. None of {raw['n_surr_phase']:,} phase-randomised envelopes reached the observed group peak
(p = {raw['group_p_phase']:.4f} = 1/{raw['n_surr_phase'] + 1}, the smallest attainable value). The circular-shift null was enumerated completely: none of the {raw['n_surr_circ']} admissible shifts (at least {2 * t0['max_lag_TR']} TRs away from zero) reached the observed peak
(p = {raw['group_p_circ']:.3f} = 1/{raw['n_surr_circ'] + 1}, again the minimum attainable). At the single-subject level {nsig}/18 subjects are significant at p &lt; 0.05 (phase randomisation).</li>
<li><b>Hemodynamic delay.</b> The lag is positive in every subject, and {raw['group_peak_lag_s']:.1f} s is within the expected 4&ndash;6 s hemodynamic range. Convolving the envelope with the canonical HRF moves the group peak from +{lag} TRs to lag {hrf['group_peak_lag_TR']}
(single subjects: {hl.mean():.1f} &plusmn; {hl.std(ddof=1):.1f} TRs), raises the zero-lag correlation from {n(raw['group_r_at_lag0'], '.2f')} to {hrf['group_r_at_lag0']:.2f} and the peak correlation from {raw['group_peak_r']:.2f} to {hrf['group_peak_r']:.2f}: with this crop, the canonical HRF accounts for the delay and for the temporal smoothing of the BOLD response.</li>
<li><b>Caveat on the absolute lag.</b> The position of the 280-TR scan inside the {t0['audio_s']:.0f}-s recording is not documented in the data files, so the lag is known only up to that offset. The {t0['scan_start_TR']}-TR crop of the example code is also the offset at which a canonical-HRF model fits best
(sliding the scan window over the HRF-convolved envelope of the whole file: r = {r_off[best_off]:.2f} at {best_off} TRs vs {r_off[best_off - 1]:.2f} at {best_off - 1} and {r_off[best_off + 1]:.2f} at {best_off + 1}). This is not an independent confirmation: it presumes the canonical delay, so crop and lag cannot be separated.
If the scan started exactly at story onset (15 s = {alt['scan_start_TR']} TRs), the raw peak lag would be +{alt['raw_group_peak_lag_TR']} TRs ({alt['raw_group_peak_lag_TR'] * TR:.0f} s). Either way the lag is {lag}&ndash;{lag + 1} TRs, i.e. hemodynamic.
The averaging convention does not matter either: z-scoring every voxel before averaging, as the course example notebook does, gives a group peak at +{vz['raw_group_peak_lag_TR']} TRs with r = {vz['raw_group_peak_r']:.2f} (the subject time courses of the two conventions correlate at r &ge; {vz_min}).</li>
<li><b>Why this matters.</b> A1 tracks the loudness of the story with a similar, physiologically plausible delay in {nsig} of 18 subjects individually (peak lags {sl.min()}&ndash;{sl.max()} TRs). This indicates that the scans are locked to the stimulus and to each other to within about one TR.
All later analyses assume this: LOSO decoding and ISFC both rely on TR <i>t</i> meaning the same story moment in every subject. A misaligned or shifted subject would lower decoding and ISFC for purely technical reasons. Because BOLD lags the sound by {lag}&ndash;{lag + 1} TRs, each 20-TR "segment" reflects story content from roughly 5 s earlier, which shifts segment boundaries equally in all subjects and does not affect classification.</li>
</ul>"""

BLK["subj"] = f"""<div class="tbl">{subj_tables}
<div class="cap"><b>Table {TAB['subj']}.</b> Task 0, single subjects (file IDs; raw envelope): lag and value of the peak cross-correlation with the subject's A1 time course, and p-value of the peak against {raw['n_surr_phase']:,} phase-randomised envelopes (maximum over lags; smallest attainable p = {1 / (raw['n_surr_phase'] + 1):.4f}).</div></div>"""

BLK["h3clf"] = f"""<h2>3. Decoding results in the original voxel space (Task A)</h2>
<div class="tbl"><table><tr><th>Classifier</th><th>Mean over 52 ROIs (%)</th><th>Median (%)</th><th># ROIs where best</th><th>ROIs above chance (FDR q &lt; .05)</th></tr>{clf_rows}</table>
<div class="cap"><b>Table {TAB['clf']}.</b> Classifier summary. "Best" is decided on the number of correct test samples; {A['n_rois_best_tied']} exact ties ({tied_txt}) are credited to the first-listed classifier.
Significance: one-sided t-test of the 18 fold accuracies against 7.14 %, Benjamini&ndash;Hochberg corrected over all 52&times;7 tests (LOSO folds share training data, so these p-values are approximate).</div></div>"""

BLK["regions"] = f"""<p><b>Which regions</b> (Figures {FIG['bars']} and {FIG['heat']}, Table {TAB['top']}). The four auditory ROIs ({', '.join(top4)}) lead by a wide margin (best single result: {best1['roi']}, {best1['clf']} {best1['acc']:.1f} %, more than {int(best1['acc'] // (100 / 14))}&times; chance), followed by precuneus, left IFG, middle and superior temporal, temporo-parietal and inferior parietal ROIs,
i.e. the language and default-mode regions that follow the narrative. With the SVM, Naive Bayes, LDA and QDA all 52 ROIs are above chance (Table {TAB['clf']}). Accuracy grows with ROI size (r = {A['corr_logV_acc']:.2f} between log V and mean accuracy), so comparisons between ROIs of very different size mix information content with feature count; the bottom five ROIs have {min(bottom_v)}&ndash;{max(bottom_v)} voxels.</p>"""

BLK["classifiers"] = f"""<p><b>Which classifiers.</b> The linear SVM ({A['clf_mean'][jS]:.1f} %) and LDA ({A['clf_mean'][jL]:.1f} %) are the best models on average, with Gaussian Naive Bayes close behind ({A['clf_mean'][jN]:.1f} %). LDA is the best model in {A['n_rois_best'][jL]} ROIs, including {A['n_big_rois_best'][jL]} of the {A['n_big_rois']} ROIs with V &ge; 200;
Naive Bayes wins in {A['n_rois_best'][jN]}, mostly small ROIs (median V = {A['best_median_nvox'][jN]:.0f}, against {A['best_median_nvox'][jL]:.0f} for the ROIs won by LDA), and the SVM in {A['n_rois_best'][jS]}.
With {n_train} training samples and thousands of features, strongly regularised linear models and a maximum-margin criterion do well.
<b>The position of LDA depends on its regulariser</b> (Table {TAB['lda']}). scikit-learn's standard shrinkage LDA on the raw features gives almost the same average accuracy ({LD['mean_over_rois']['sklearn_auto']:.1f} %; per-ROI differences from {n(dsk['min'])} to {n(dsk['max'], '+.1f')} pp),
whereas exact LW shrinkage in voxel space, a much weaker regulariser, gives {LD['mean_over_rois']['voxel_lw']:.1f} %; LDA is then below QDA on average and the best model in only {LD['n_rois_lda_best']['voxel_lw']} ROIs (SVM {LD['n_rois_best_by_clf']['voxel_lw'][jS]}, Naive Bayes {LD['n_rois_best_by_clf']['voxel_lw'][jN]}).
The three leading ROIs are the same under all three variants. The data therefore support "SVM, strongly shrunk LDA and Naive Bayes perform similarly", not a general advantage of LDA.
KNN is clearly worst ({span(knn_means, '.0f')} %): raw Euclidean distances in a very high-dimensional space are dominated by subject-specific noise, and with one exemplar per subject and class the neighbourhood is sparse. K = 2 is slightly worse than K = 1 and 3, probably because a two-neighbour vote is tied whenever the two neighbours belong to different classes, and the tie then goes to the lower class index.
QDA, which estimates 14 covariance matrices from 17 samples each, depends heavily on its fixed shrinkage and is on average below the LDA used here.</p>"""

BLK["lda"] = f"""<div class="tbl"><table><tr><th rowspan="2">LDA covariance regulariser</th><th rowspan="2">Mean over 52 ROIs</th><th colspan="{len(LDA_ROIS)}">Example ROIs</th><th rowspan="2"># ROIs: LDA best</th><th rowspan="2"># ROIs: SVM / NB best</th></tr>
<tr>{lda_hdr}</tr>{lda_rows}</table>
<div class="cap"><b>Table {TAB['lda']}.</b> Sensitivity of the Task-A LDA accuracy (%) to the covariance regulariser; same samples and LOSO folds, all other classifiers unchanged (<code>src/supp_lda_sensitivity.py</code>; NB = Naive Bayes).
The second and third variant are computed with Gram matrices; this was verified against scikit-learn's dense implementations on ROI {LD['selfcheck']['roi']} (identical predictions in all folds). No variant was selected on the basis of these numbers; the report model was fixed beforehand.</div></div>"""

BLK["samples"] = f"""<p><b>What the 20-TR samples contain.</b> A sample combines the temporal profile of the regional response with its spatial pattern. To separate the two, the linear SVM was re-run on reduced features (Table {TAB['tvs']}).
The ROI-mean time course of the segment alone (20 features, no spatial information) gives {tsm['temporal']:.1f} % on average, the time-averaged spatial pattern alone (V features, no temporal information) {tsm['spatial']:.1f} %, and the full sample {tsm['full']:.1f} %.
The mean time course is better than the spatial pattern in {TS['n_rois_temporal_gt_spatial']} of 52 ROIs and reaches {span(ts_aud, '.0f')} % in the four auditory ROIs; the {len(ts_exc)} exceptions are {n_prec_exc} precuneus ROIs and {len(ts_exc) - n_prec_exc} inferior parietal ROI, where the spatial pattern is slightly better.
In most regions, decoding therefore relies mainly on how the regional response evolves within a segment, which is shared across subjects, and less on fine-grained spatial patterns.
The full sample is still at least as good as both reduced sets in {TS['n_rois_full_ge_both']} ROIs, so voxel-wise time courses add information beyond the regional mean. These numbers are for one classifier with fixed C; re-standardising the reduced features changes the two averages by less than 1 pp ({tsm['temporal_z']:.1f} % and {tsm['spatial_z']:.1f} %).</p>"""

BLK["tvs"] = f"""<div class="tbl"><table style="width:78%;margin-left:auto;margin-right:auto"><tr><th>ROI</th><th>V</th><th>Full sample (20&middot;V features)</th><th>ROI-mean time course (20 features)</th><th>Time-averaged spatial pattern (V features)</th></tr>{ts_rows}</table>
<div class="cap"><b>Table {TAB['tvs']}.</b> Linear-SVM accuracy (% &plusmn; SEM over 18 folds, 14 classes, chance 7.14 %) with the full sample and with its temporal or spatial marginal mean only (<code>src/supp_temporal_vs_spatial.py</code>). Example ROIs: best auditory, largest, best non-auditory, one of the exceptions, and a mid-sized temporal ROI.</div></div>"""

BLK["top"] = f"""<div class="tbl"><table><tr><th>ROI</th><th>V</th>{hdr}<th>Mean</th></tr>{top_rows}</table>
<div class="cap"><b>Table {TAB['top']}.</b> Ten leading ROIs (accuracy % &plusmn; SEM; bold = best classifier of the ROI). The full 52-ROI table is in Appendix A.</div></div>"""

BLK["bars"] = f"""<div class="fig">{img('part2a_bars_all_rois.png', '84%')}<div class="cap"><b>Figure {FIG['bars']}.</b> LOSO 14-way segment-decoding accuracy (mean &plusmn; SEM over 18 folds) for all 52 ROIs and all classifiers. ROIs are in the same order in every panel (sorted by the mean over classifiers); dashed line: chance (7.14 %).</div></div>"""

BLK["heat"] = f"""<div class="fig">{img('part2a_heatmap.png')}<div class="cap"><b>Figure {FIG['heat']}.</b> Accuracy of every classifier in every ROI.</div></div>"""

BLK["h4"] = f"""<h2>4. Hyperalignment impact analysis (Task B)</h2>"""

BLK["scatter"] = f"""<div class="fig">{img('part2b_scatter.png', '82%')}<div class="cap"><b>Figure {FIG['scatter']}.</b> Anatomical vs. hyperaligned accuracy in the identical split-half 7-class LOSO design (average of both half assignments). One dot per ROI; points above the diagonal improve with hyperalignment.</div></div>"""

BLK["hyp"] = f"""<div class="tbl"><table><tr><th>Classifier</th><th>Anatomical (%)</th><th>Hyperaligned (%)</th><th>Gain (pp)</th><th>ROIs improved</th><th>ROIs tied</th><th>Wilcoxon p (52 ROIs)</th></tr>{b_rows}</table>
<div class="cap"><b>Table {TAB['hyp']}.</b> Mean accuracy over the 52 ROIs with anatomical alignment and after hyperalignment (chance 14.29 %). &ldquo;Improved&rdquo; and &ldquo;tied&rdquo; compare the numbers of correct test samples of a ROI ({B['n_test_per_roi']} per ROI: both halves, 18 folds, 7 samples). The Wilcoxon signed-rank test compares the two across ROIs (ROIs are not independent, so p-values are descriptive).</div></div>"""

BLK["gainroi"] = f"""<div class="fig">{img('part2b_gain_per_roi.png', '82%')}<div class="cap"><b>Figure {FIG['gainroi']}.</b> Per-ROI accuracy (mean of the 7 classifiers) and gain from hyperalignment, mean &plusmn; SEM over folds.</div></div>"""

BLK["gain"] = f"""<div class="tbl"><table style="width:49%;display:inline-table;vertical-align:top"><tr><th>Largest gains</th><th>V</th><th>Anat.</th><th>Hyper.</th><th style="white-space:nowrap">Gain &plusmn; SEM</th></tr>{gain_rows}</table>
<table style="width:49%;display:inline-table;vertical-align:top"><tr><th>Largest losses</th><th>V</th><th>Anat.</th><th>Hyper.</th><th style="white-space:nowrap">Gain &plusmn; SEM</th></tr>{loss_rows}</table>
<div class="cap"><b>Table {TAB['gain']}.</b> ROIs with the largest gains and losses from hyperalignment (accuracy: mean of the 7 classifiers, %; gain in percentage points &plusmn; SEM over the 18 held-out subjects).</div></div>"""

BLK["half"] = f"""<div class="tbl"><table><tr><th rowspan="2">Classifier</th><th colspan="3">Decode 1st half (align on 2nd)</th><th colspan="3">Decode 2nd half (align on 1st)</th></tr>
<tr><th>Anat.</th><th>Hyper.</th><th>Gain</th><th>Anat.</th><th>Hyper.</th><th>Gain</th></tr>{bh_rows}</table>
<div class="cap"><b>Table {TAB['half']}.</b> The two split directions separately (mean accuracy over 52 ROIs, %).</div></div>"""

BLK["help"] = f"""<p><b>Did hyperalignment help?</b> Modestly, and not uniformly. Averaged over all ROIs and classifiers accuracy rose from {B['overall_anat']:.1f} % to {B['overall_hyper']:.1f} % ({n(B['subj_gain_mean'], '+.1f')} pp per held-out subject, paired t-test over the 18 subjects p = {B['subj_gain_p']:.3f}); {B['n_rois_improved_mean']} of 52 ROIs improved{tie_txt}.
Two qualifications are important. First, the average gain comes almost entirely from one split direction (Table {TAB['half']}): when the first half is decoded, accuracy rises from {bha[0]:.1f} % to {bhh[0]:.1f} %, but when the second half is decoded the anatomical baseline is already higher ({bha[1]:.1f} %) and hyperalignment leaves it essentially unchanged ({bhh[1]:.1f} %).
Second, the effect differs strongly between regions. With a paired t-test over held-out subjects per ROI (FDR q &lt; .05), {B['n_sig_gain_fdr']} ROIs gain significantly and {B['n_sig_loss_fdr']} lose significantly.
The gains are concentrated in the <b>superior and middle temporal cortex around the auditory core</b>: the {N_GAIN} largest (Table {TAB['gain']}; {n(min(r['gain'] for r in top_gain), '+.1f')} to {n(max(r['gain'] for r in top_gain), '+.1f')} pp) are posterior STG/STS, STG, mid-MTG, auditory-belt and TPJ ROIs,
and the four auditory ROIs reach the highest absolute accuracies ({span(aud_svm, '.0f')} % with the linear SVM).
These are regions with a strong stimulus-driven response, where a finer functional correspondence between subjects than anatomy provides is plausible.
In contrast, <b>{len(pi_loss)} of the {len(pi)} precuneus / inferior-parietal ROIs lost</b> {span([-r['gain'] for r in pi_loss], '.0f')} pp ({', '.join(r['roi'] for r in pi_loss)}); the other {len(pi_rest)} changed by {span([r['gain'] for r in pi_rest])} pp (not significant), and the fifth significant loss is {tp['roi']} ({n(tp['gain'], '+.1f')} pp).
A plausible explanation, not tested here, is estimation noise: in these four ROIs a rotation with V(V&minus;1)/2 free parameters (V = {min(r['n_vox'] for r in pi_loss)}&ndash;{max(r['n_vox'] for r in pi_loss)}) is constrained by only {pr['n_align_TR']} TRs, so it is poorly determined and can partly fit stimulus-specific noise of the alignment half.
The fit itself is far from perfect, because an orthogonal map cannot change the similarity structure between time points: in the {pr['n_rois_V_gt_T']} ROIs with V &gt; {pr['n_align_TR']}, the relative Procrustes residual &Vert;XR&minus;T&Vert;&sup2;/&Vert;X&Vert;&sup2; on the alignment half is {span(pr['train_range_V_gt_T'], '.2f')} for the training subjects and {span(pr['heldout_range_V_gt_T'], '.2f')} for the held-out subject (ROI means).
The slow story response of these regions may also gain little from a finer alignment; this was not tested either. V &gt; T alone does not predict a loss (A1_L, V = {btab['52 A1_L']['n_vox']}, gains {n(btab['52 A1_L']['gain'], '+.1f')} pp), and across ROIs the gain is not related to ROI size (r = {n(B['corr_gain_logV'], '.2f')} with log V).
Haxby et al. derived their transformations from a full-length movie (thousands of TRs); with halves of {pr['n_align_TR']} TRs a smaller benefit is to be expected.</p>"""

BLK["clfs"] = f"""<p><b>Which classifiers changed most, and why?</b> The four parametric / linear models gained {span(par)} pp (QDA {n(B['gain_mean'][jQ], '+.1f')}, SVM {n(B['gain_mean'][jS], '+.1f')}, LDA {n(B['gain_mean'][jL], '+.1f')}, Naive Bayes {n(B['gain_mean'][jN], '+.1f')}), whereas KNN did not improve ({', '.join(n(B['gain_mean'][j], '+.1f') for j in jK)} pp for K = 1, 2, 3).
Models that learn one weight (or mean) per feature need feature j to mean the same thing in every subject; hyperalignment makes common-space dimensions functionally more similar across subjects, so class means become sharper and weights learned on 17 subjects transfer better to the 18th.
This argument does not explain the missing gain of KNN, whose Euclidean distances between subjects depend on voxel correspondence in the same way. A possible reason is that the poorly determined rotations add alignment noise to every single pattern, which offsets the gain in shared signal when individual exemplars are compared, while class means and weights estimated from 17 subjects average part of it out; this was not tested.</p>"""

BLK["why"] = f"""<p><b>Why must the alignment parameters come from independent data?</b> A Procrustes rotation has V(V&minus;1)/2 free parameters and is fitted to make subjects' time courses as similar as possible. If it were estimated on the same TRs that are later decoded, it would also align noise that happens to coincide in those TRs; with V &gt; T in {pr['n_rois_V_gt_T']} ROIs the rotation has ample freedom to do so in exactly the TRs that are later classified.
Since the classes here <i>are</i> time segments, "making time courses similar" is the same as "making the test samples resemble the correct class": information about the test labels would leak into the features and the accuracy would be inflated (double dipping).
The same logic requires the template to be built without the held-out subject. With the split-half design, any gain can only come from structure that generalises to new stimuli, i.e. from a functional correspondence between subjects that is not specific to the decoded time points.</p>"""

BLK["dimp"] = f"""<p><b>Dimensionality of the common space (optional analysis, Table {TAB['dim']} / Figure {FIG['dim']}).</b> For the {B['n_big']} ROIs with V &gt; {k_max}, a common space of only 20 dimensions gives SVM, LDA and QDA accuracies within {d20:.1f} pp of the full space and above the anatomical baseline,
and the six most accurate ROIs differ by {span(d6)} pp between k = 20 and all voxels (mean of the 7 classifiers): the shared story-driven signal used by these classifiers appears to be largely low-dimensional.
Naive Bayes is the exception: its accuracy falls as components are added ({nb_k[0]:.1f} % at k = {B['k_list'][0]} to {nb_k[-1]:.1f} % at k = {k_max}) although it is high again in the full voxel basis ({redfull[jN]:.1f} %). A plausible reason is that in the PC basis the later components carry little shared signal but enter the product of likelihoods with the same weight as the first ones; this was not tested separately.</p>"""

BLK["dimt"] = f"""<div class="tbl"><table><tr><th>Common space</th>{hdr}</tr>{k_rows}</table>
<div class="cap"><b>Table {TAB['dim']}.</b> Mean accuracy (%) over the {B['n_big']} ROIs with V &gt; {k_max} as a function of the number of common-space dimensions.</div></div>"""

BLK["dimf"] = f"""<div class="fig">{img('part2b_dimensionality.png', '96%')}<div class="cap"><b>Figure {FIG['dim']}.</b> Accuracy vs. common-space dimensionality. Left: markers of the seven classifiers are slightly offset horizontally so that coinciding values stay visible; squares: anatomical baseline. Right: mean of the 7 classifiers in the six most accurate hyperaligned ROIs.</div></div>"""

BLK["h5seed"] = f"""<h2>5. FC and ISFC cortical maps (Part 3)</h2>
<p><b>Seed selection.</b> The three ROIs with the highest Task-A accuracy (mean over the 7 classifiers) are <b>{tops[0]}</b> ({P3['mean_acc_pct'][0]:.1f} %), <b>{tops[1]}</b> ({P3['mean_acc_pct'][1]:.1f} %) and <b>{tops[2]}</b> ({P3['mean_acc_pct'][2]:.1f} %);
{r4[0]} is fourth ({r4[1]:.1f} %). With other criteria (best single classifier, or the SVM alone) {short(r4[0])} would replace one of the three, so the selection among the four auditory ROIs is not robust, but all candidates are auditory; the three LDA variants of Table {TAB['lda']} give the same three ROIs.
Because three auditory seeds give similar maps, the best non-auditory ROI, <b>{extra}</b> (rank {P3['extra_seed_rank'][extra]}), was added as a supplementary seed. Maps (Figures {FIG['map1']}&ndash;{FIG['map4']}) are displayed for |r| &gt; {thr} and share one colour scale; Table {TAB['fc']} summarises them.</p>"""

BLK["m1"] = f"""{mapfig('map1', tops[0], f'Seed {tops[0]}: FC (top) and ISFC (bottom), orthogonal views through the seed centre and axial slices (template z in mm).')}"""

BLK["fc"] = f"""<div class="tbl"><table><tr><th>Seed</th><th>V</th><th>Mean r inside seed (FC / ISFC)</th><th>Mean r, rest of brain (FC / ISFC)</th><th>Voxels with r &gt; 0.2 (FC / ISFC)</th><th>Max r outside seed (FC / ISFC)</th><th>Spatial corr. FC vs ISFC</th></tr>{p3_rows}</table>
<div class="cap"><b>Table {TAB['fc']}.</b> Quantitative comparison of the FC and ISFC maps for the three top seeds and the supplementary precuneus seed. Columns 4&ndash;7 refer to voxels outside the seed; both maps cover the same {P3['n_voxels_map_mask']:,} voxels.</div></div>"""

BLK["a"] = f"""<h3>a. Spatial differences between FC and ISFC</h3>
<p>FC maps are stronger and more widespread than ISFC maps of the same seed on identical voxels (mean r outside the seed {span([s['fc_mean_r'] for s in seeds], '.2f')} vs {span([s['isfc_mean_r'] for s in seeds], '.2f')}; {span(fc_fr, '.0f')} % vs {span(is_fr, '.0f')} % of voxels above r = 0.2), and they show a halo of very high values around the seed that decays with distance.
What FC adds beyond temporal cortex depends on the seed. For the two A1 seeds it is the insula and opercula, thalamus, medial frontal cortex (SMA) and cerebellum
(FC averaged over 27-voxel cubes at approximate anatomical locations: insula {span(probe(a1, INS, 'fc'), '.2f')}, thalamus {span(probe(a1, THA, 'fc'), '.2f')}, SMA {span(probe(a1, SMA, 'fc'), '.2f')}, cerebellum {span(probe(a1, CER, 'fc'), '.2f')}).
The belt seed {short(belt_name[0])} is different: its FC map does not include the insula or SMA ({span(probe([belt], INS + SMA, 'fc'), '.2f')}) but left inferior frontal cortex ({belt['probes']['IFG L']['fc']:.2f}), precuneus ({belt['probes']['precuneus']['fc']:.2f}) and medial prefrontal cortex ({belt['probes']['medial prefrontal']['fc']:.2f}).
ISFC maps are spatially selective. For the two A1 seeds they show essentially bilateral auditory cortex and STG, with small foci near the display threshold elsewhere (for example inferior frontal cortex, thalamus and cerebellum; insula {span(probe(a1, INS, 'isfc'), '.2f')});
for {short(belt_name[0])} the map extends along STG/STS into inferior frontal and premotor cortex, i.e. the speech and language network, and shows weak negative ISFC in medial parietal and cingulate regions.
The ISFC maximum lies in the corresponding auditory region of the other subjects, without a local halo: the mean correlation inside the seed drops from {span([s['fc_within_seed_mean_r'] for s in seeds], '.2f')} (FC) to {span([s['isfc_within_seed_mean_r'] for s in seeds], '.2f')} (ISFC).
The two maps are nevertheless spatially correlated (r = {min(sc):.2f}&ndash;{max(sc):.2f}), so ISFC looks like the stimulus-locked subset of FC.
The supplementary precuneus seed shows the same contrast in a different network: FC and ISFC both outline default-mode regions, but ISFC is weaker inside the seed ({xs['isfc_within_seed_mean_r']:.2f} vs {xs['fc_within_seed_mean_r']:.2f}) and lacks the local halo.</p>"""

BLK["m2"] = f"""{mapfig('map2', tops[1], f'Seed {tops[1]}; layout as in Figure {FIG["map1"]}.')}"""

BLK["m3"] = f"""{mapfig('map3', tops[2], f'Seed {tops[2]} (auditory belt); layout as in Figure {FIG["map1"]}.')}"""

BLK["b"] = f"""<h3>b. Why ISFC filters out non-neuronal artifacts and spontaneous fluctuations</h3>
<p>A voxel's signal can be written as stimulus-locked response + idiosyncratic neural activity + non-neuronal noise. Within one brain, all three components are shared between seed and target, so all three contribute to FC: respiration and cardiac pulsation, head motion and scanner drift affect many voxels of the same scan simultaneously, local smoothing and vascular coupling contribute to the halo around the seed, and spontaneous (resting-state-like) fluctuations are coherent within intrinsic networks.
ISFC correlates the seed of one subject with voxels in <i>other</i> subjects, who were scanned at different times. Whatever part of their breathing, heartbeat, motion, drift and spontaneous thoughts is unrelated to the stimulus is independent of the seed subject's, so its expected contribution to the cross-subject covariance is zero. What survives is everything that is time-locked to the common stimulus.
That is mostly, but not only, the neural response: physiology and motion that the story itself drives in a similar way in different listeners (for example changes of breathing or heart rate at an arousing passage) are synchronised across subjects and are not removed. ISFC therefore shows stimulus-locked signal; reading it as shared neural processing rests on the additional assumption that such stimulus-locked physiology is small.
Averaging the other 17 subjects before correlating additionally averages out their independent noise, which raises the signal-to-noise ratio of the stimulus-locked component. ISFC is therefore conservative: genuine but subject-specific neural coupling is removed along with the artifacts.</p>"""

BLK["m4"] = f"""{mapfig('map4', extra, f'Supplementary seed {extra} (precuneus, default-mode network); layout as in Figure {FIG["map1"]}.')}"""

BLK["c"] = f"""<h3>c. Stimulus-locked, narrative-driven processing</h3>
<p>The ISFC maps show which regions respond to the story in a way that is shared across listeners and coupled to auditory cortex: strongest in early auditory areas, which track the acoustics (Task 0), and extending for the belt seed into STG/STS and inferior frontal regions that process speech and language over longer time scales.
The broader FC pattern of the A1 seeds (insular, medial frontal, thalamic, cerebellar) is strongly reduced in ISFC (insula {span(probe(a1, INS, 'isfc'), '.2f')}; thalamus, SMA and cerebellum remain around the display threshold, {span(probe(a1, THA + SMA + CER, 'isfc'), '.2f')}), which suggests that much of this auditory "connectivity" is not locked to the narrative but reflects intrinsic, idiosyncratic or non-neuronal co-fluctuation.
This matches the decoding results: between-subject (LOSO) classification can only use response components shared across subjects, which is what ISFC isolates, and the regions with reliable stimulus-locked responses (auditory, STG/MTG, precuneus, IFG) are the ones that decode the story segment best.
It is consistent with Simony et al. (2016), who showed that ISFC reveals stimulus-dependent network configurations during narrative comprehension that standard FC, dominated by intrinsic correlations, does not.
All three top seeds are auditory, so their maps describe the auditory/speech end of the hierarchy; the supplementary precuneus seed (Figure {FIG['map4']}) shows that shared, story-locked coupling is also present within the default-mode network, with {100 * xs['isfc_frac_gt_0.2']:.0f} % of voxels above r = 0.2 in ISFC.</p>"""

BLK["limits"] = f"""<h2>6. Limitations</h2>
<ul><li>A sample is a 20-TR spatio-temporal segment. A linear SVM on the temporal profile of the ROI-mean signal alone reaches {tsm['temporal']:.1f} % on average, against {tsm['full']:.1f} % for the full sample (Table {TAB['tvs']}), so the results should not be read as evidence for fine-grained spatial codes; purely spatial (single-TR) decoding was not attempted.</li>
<li>Accuracy depends on ROI size, which ranges from {v_min} to {v_max} voxels; ROIs were not size-matched.</li>
<li>Hyperalignment was estimated from {pr['n_align_TR']} TRs per half, fewer than the number of voxels in {pr['n_rois_V_gt_T']} ROIs; gains may be underestimated relative to alignment on longer independent data, and the explanations offered for the losses were not tested.</li>
<li>LDA/QDA regularisation was fixed, not optimised, and the rank of LDA among the classifiers depends on the regulariser (Table {TAB['lda']}); nested tuning could change the ranking. Significance tests over LOSO folds are approximate because folds share training subjects.</li>
<li>The absolute audio&ndash;scan offset is uncertain by about one TR; crop and hemodynamic lag are not separately identifiable (Section 2).</li></ul>"""

BLK["refs"] = f"""<h2>References</h2>
<ol class="refs">
<li>Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O., Conroy, B. R., Gobbini, M. I., Hanke, M., &amp; Ramadge, P. J. (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. <i>Neuron, 72</i>(2), 404&ndash;416.</li>
<li>Ledoit, O., &amp; Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. <i>Journal of Multivariate Analysis, 88</i>(2), 365&ndash;411.</li>
<li>Nastase, S. A., Gazzola, V., Hasson, U., &amp; Keysers, C. (2019). Measuring shared responses across subjects using intersubject correlation. <i>Social Cognitive and Affective Neuroscience, 14</i>(6), 667&ndash;685.</li>
<li>Pereira, F., Mitchell, T., &amp; Botvinick, M. (2009). Machine learning classifiers and fMRI: A tutorial overview. <i>NeuroImage, 45</i>(1 Suppl), S199&ndash;S209.</li>
<li>Sch&ouml;nemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. <i>Psychometrika, 31</i>(1), 1&ndash;10.</li>
<li>Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., &amp; Hasson, U. (2016). Dynamic reconfiguration of the default mode network during narrative comprehension. <i>Nature Communications, 7</i>, 12141.</li>
</ol>"""

BLK["appA"] = f"""<h2 class="pb">Appendix A &ndash; Task A, all ROIs (14-class accuracy %, mean &plusmn; SEM over 18 folds)</h2>
<table class="small"><tr><th>ROI</th><th>V</th>{hdr}</tr>{appA}</table>"""

BLK["appB"] = f"""<h2 class="pb">Appendix B &ndash; Task B, all ROIs (7-class accuracy %, mean &plusmn; SEM over 18 folds)</h2>
<p><b>B.1 Anatomical alignment</b> (split-half 7-class baseline; a fold's accuracy is the mean of the two half assignments).</p>
<table class="small"><tr><th>ROI</th><th>V</th>{hdr}</tr>{appB_anat}</table>
<p class="pb"><b>B.2 After hyperalignment.</b> Last column: gain over B.1 in percentage points, mean of the 7 classifiers.</p>
<table class="small"><tr><th>ROI</th><th>V</th>{hdr}<th>Mean gain (pp)</th></tr>{appB_hyper}</table>"""

# Order of the blocks in the document.  Text paragraphs can break across pages, tables and figures cannot,
# so the order within each section was chosen to fill the pages (floats follow the text that cites them).
ORDER = ["title", "methods", "task0", "subj",
         "h3clf", "regions", "top", "classifiers", "samples", "bars", "lda", "tvs", "heat",
         "h4", "scatter", "hyp", "gainroi", "gain", "half", "help", "clfs", "why", "dimp", "dimt", "dimf",
         "h5seed", "m1", "fc", "m2", "m3", "a", "b", "m4", "c",
         "limits", "refs", "appA", "appB"]
assert sorted(ORDER) == sorted(BLK), set(ORDER) ^ set(BLK)


def main():
    """Write report/report.html and convert it to report/Homework1_Report.pdf."""
    html = CSS + "\n".join(BLK[k] for k in ORDER) + "\n</body></html>"

    # captions must appear in numerical order, and every table / figure must be present exactly once
    for kind, numbering in (("Table", TAB), ("Figure", FIG)):
        found = [int(x) for x in re.findall(rf"<b>{kind} (\d+)\.</b>", html)]
        check(found == list(range(1, len(numbering) + 1)), f"{kind} captions out of order: {found}")

    out = ROOT / "report" / "report.html"
    out.write_text(html)
    pdf = ROOT / "report" / "Homework1_Report.pdf"
    footer = ('<div style="font-family:Arial,sans-serif;font-size:7.5pt;color:#666;width:100%;text-align:center">'
              '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
          f"await p.goto('file://{out}');await p.pdf({{path:'{pdf}',format:'A4',printBackground:true,"
          f"displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:{json.dumps(footer)}}});await b.close();}})();")
    print(f"HTML written: {out}")
    try:
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("node not found: the PDF was NOT rebuilt (see README, 'PDF build'); print report.html from a browser instead")
    if r.returncode != 0:
        raise SystemExit("PDF step failed (is the playwright npm package with Chromium installed?):\n" + r.stderr)
    print(f"PDF written: {pdf}")


if __name__ == "__main__":
    main()
