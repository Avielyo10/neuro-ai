"""Build report/report.html (and report/Homework1_Report.pdf via headless Chromium) from results/*.json."""
import base64
import json
import subprocess
import sys

import numpy as np

from common import FIGS, RESULTS, ROOT

t0 = json.load(open(RESULTS / "task0_alignment.json"))
A = json.load(open(RESULTS / "part2a_summary.json"))
B = json.load(open(RESULTS / "part2b_summary.json"))
P3 = json.load(open(RESULTS / "part3_summary.json"))
clf = A["clf"]


def img(name, width="100%"):
    b = base64.b64encode(open(FIGS / name, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def f1(x):
    return f"{x:.1f}"


raw, hrf = t0["raw"], t0["hrf"]
sl, sr = np.array(raw["subj_peak_lag_TR"]), np.array(raw["subj_peak_r"])
nsig = int((np.array(raw["subj_p_phase"]) < 0.05).sum())
best_off = t0["window_offset_scan"]["best_offset_TR"]
hl = np.array(hrf["subj_peak_lag_TR"])
bh = B["by_half"]
bh_rows = "".join(
    f"<tr><td>{c}</td><td>{bh['anat'][0][j]:.1f}</td><td>{bh['hyper'][0][j]:.1f}</td><td>{bh['hyper'][0][j]-bh['anat'][0][j]:+.1f}</td>"
    f"<td>{bh['anat'][1][j]:.1f}</td><td>{bh['hyper'][1][j]:.1f}</td><td>{bh['hyper'][1][j]-bh['anat'][1][j]:+.1f}</td></tr>"
    for j, c in enumerate(json.load(open(RESULTS / "part2a_summary.json"))["clf"]))
bha = [float(np.mean(bh["anat"][h])) for h in (0, 1)]; bhh = [float(np.mean(bh["hyper"][h])) for h in (0, 1)]

# ---------- tables ----------
hdr = "".join(f"<th>{c}</th>" for c in clf)
top_rows = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td>"
    + "".join(f"<td{' class=b' if a == max(r['acc']) else ''}>{a:.1f} ± {s:.1f}</td>" for a, s in zip(r["acc"], r["sem"]))
    + f"<td>{r['mean']:.1f}</td></tr>" for r in A["top10"])
clf_rows = "".join(
    f"<tr><td>{c}</td><td>{A['clf_mean'][j]:.1f}</td><td>{A['clf_median'][j]:.1f}</td><td>{A['n_rois_best'][j]}</td>"
    f"<td>{A['n_sig_rois_fdr'][j]}/52</td></tr>" for j, c in enumerate(clf))
b_rows = "".join(
    f"<tr><td>{c}</td><td>{B['anat_mean'][j]:.1f}</td><td>{B['hyper_mean'][j]:.1f}</td><td>{B['gain_mean'][j]:+.1f}</td>"
    f"<td>{B['n_improved'][j]}/52</td><td>{B['p_wilcoxon_rois'][j]:.2g}</td></tr>" for j, c in enumerate(clf))
gain_rows = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td><td>{r['anat']:.1f}</td><td>{r['hyper']:.1f}</td>"
    f"<td>{r['gain']:+.1f} ± {r['sem']:.1f}</td></tr>" for r in B["top_gain"][:8])
loss_rows = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td><td>{r['anat']:.1f}</td><td>{r['hyper']:.1f}</td>"
    f"<td>{r['gain']:+.1f}</td></tr>" for r in B["worst_gain"][::-1])
k_rows = "".join(
    f"<tr><td>k = {k}</td>" + "".join(f"<td>{v:.1f}</td>" for v in B["reduced_mean_bigrois"][str(k)]) + "</tr>"
    for k in B["k_list"]) + "<tr><td>all V (full)</td>" + "".join(f"<td>{v:.1f}</td>" for v in B["hyper_mean_bigrois"]) \
    + "</tr><tr><td>anatomical</td>" + "".join(f"<td>{v:.1f}</td>" for v in B["anat_mean_bigrois"]) + "</tr>"
p3_rows = "".join(
    f"<tr><td>{n}</td><td>{s['n_seed_voxels']}</td><td>{s['fc_within_seed_mean_r']:.2f} / {s['isfc_within_seed_mean_r']:.2f}</td>"
    f"<td>{s['fc_mean_r']:.3f} / {s['isfc_mean_r']:.3f}</td><td>{100*s['fc_frac_gt_0.2']:.1f} % / {100*s['isfc_frac_gt_0.2']:.1f} %</td>"
    f"<td>{s['fc_max_r_outside_seed']:.2f} / {s['isfc_max_r_outside_seed']:.2f}</td><td>{s['spatial_corr_fc_isfc']:.2f}</td></tr>"
    for n, s in P3["seeds"].items())
appA = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td>" + "".join(f"<td>{a:.1f} ± {s:.1f}</td>" for a, s in zip(r["acc"], r["sem"])) + "</tr>"
    for r in A["table"])
appB = "".join(
    f"<tr><td>{r['roi']}</td><td>{r['n_vox']}</td>" + "".join(f"<td>{a:.0f} → {h:.0f}</td>" for a, h in zip(r["anat"], r["hyper"])) + "</tr>"
    for r in B["table"])
tops = P3["top_rois"]
extra = P3["extra_seeds"][0]
seeds = [P3["seeds"][t] for t in tops]
xs = P3["seeds"][extra]
r4 = P3["rank4"]
fc_fr = [100 * s["fc_frac_gt_0.2"] for s in seeds]; is_fr = [100 * s["isfc_frac_gt_0.2"] for s in seeds]
sc = [s["spatial_corr_fc_isfc"] for s in seeds]
jS, jN, jL, jQ, jK = clf.index("Linear SVM"), clf.index("Naive Bayes"), clf.index("LDA"), clf.index("QDA"), clf.index("KNN (K=1)")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Homework 1 Report</title><style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 10.3pt; line-height: 1.38; color: #111; }}
h1 {{ font-size: 17pt; margin: 0 0 2pt; }} h2 {{ font-size: 13pt; border-bottom: 1.2px solid #1f3a5f; color: #1f3a5f; margin: 16pt 0 5pt; }}
h3 {{ font-size: 11pt; margin: 10pt 0 3pt; color: #1f3a5f; }} p {{ margin: 4pt 0; text-align: justify; }}
table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 7.9pt; margin: 5pt 0; }}
th {{ background: #1f3a5f; color: #fff; padding: 3px 4px; }} td {{ padding: 2px 4px; border-bottom: 0.5px solid #ccc; text-align: center; }}
td:first-child {{ text-align: left; white-space: nowrap; }} td.b {{ font-weight: bold; }}
.cap {{ font-size: 8.8pt; color: #333; margin: 2pt 0 9pt; text-align: justify; }} .fig {{ page-break-inside: avoid; margin-top: 6pt; }}
.meta {{ font-family: Arial, sans-serif; font-size: 9pt; color: #444; }} .pb {{ page-break-before: always; }} ul {{ margin: 3pt 0; padding-left: 16pt; }}
li {{ margin-bottom: 2pt; text-align: justify; }} code {{ font-size: 8.8pt; }} .small td {{ font-size: 6.9pt; padding: 1px 3px; }}
</style></head><body>
<h1>Brain State Decoding &amp; Functional Connectivity (FC / ISFC)</h1>
<div class="meta">NeuroAI: Bridging Neuroscience and Artificial Intelligence (2700759), Fall 2026 &middot; Homework Assignment 1<br>
Data: 18 subjects, "Pieman" story, TR = 1.5 s, 280 TRs, 52 ROIs &middot; Code: Python (<code>src/</code>), fully reproducible with <code>scripts/run_all.sh</code></div>

<h2>1. Methodology</h2>
<p><b>Data and ROIs.</b> Each subject file holds <code>data_crop</code> (271,633 voxels &times; 280 TRs) on the 61&times;73&times;61, 3-mm template grid.
The 52 ROIs of <code>group2_all_52rois.voi</code> were mapped to voxel indices with the inverse template affine (as in the course example code; the file header labels the coordinates TAL, they are used as template mm coordinates).
Voxels were kept only if they carried valid signal in all 18 subjects (finite, mean &gt; 6000 &ndash; the helper scripts' threshold &ndash; and non-constant);
this removed 33 of 8,503 ROI voxels (ROIs 42, 45, 47). Several VOI names repeat in the file, so ROIs are referred to as "index name" (e.g. <i>48 A1_R</i>).
Every voxel time course was z-scored within subject.</p>
<p><b>Tensor reshaping and samples.</b> For an ROI with V voxels, each subject's 280&times;V matrix was reshaped to 14&times;20&times;V
(14 consecutive 30-s segments &times; 20 TRs &times; V voxels) and subjects were stacked into the 14&times;20&times;V&times;18 group tensor. One <i>sample</i> is one segment of one subject:
the 20&times;V spatio-temporal pattern, flattened to 20&middot;V features. The class label is the segment index (14 classes, chance 1/14 = 7.14 %).</p>
<p><b>LOSO cross-validation.</b> In each of 18 folds the classifiers were trained on 17 subjects (17&times;14 = 238 samples, 17 per class) and tested on the 14 segments of the held-out subject.
Reported values are the mean accuracy over the 18 folds &plusmn; SEM (SD/&radic;18).</p>
<p><b>Classifiers</b> (scikit-learn; all hyper-parameters fixed in advance, none tuned on test data).
<i>KNN</i>: Euclidean distance, K = 1, 2, 3. <i>Linear SVM</i>: linear kernel, C = 1, one-vs-one (libsvm; the Gram matrix is precomputed, which is mathematically identical and faster).
<i>Naive Bayes</i>: Gaussian. <i>LDA / QDA</i>: because features (up to 20&middot;766) far outnumber training samples (238), sample covariances are singular; both models are therefore fitted after a PCA rotation estimated on the training fold only
(all components kept) and regularised &ndash; LDA with the pooled within-class covariance shrunk towards a scaled identity by the Ledoit&ndash;Wolf amount (own implementation in <code>decoding.py</code>, computed in double precision and without rescaling the features,
because scikit-learn's <code>shrinkage="auto"</code> standardises features first, which after a PCA rotation cancels the regularisation), QDA with class-specific covariances shrunk half-way towards a scaled identity (shrinkage 0.5; only 17 samples per class).</p>
<p><b>Hyperalignment (Task B).</b> Orthogonal Procrustes without scaling (<code>scipy.linalg.orthogonal_procrustes</code>), following Haxby et al. (2011):
(i) a reference subject (the first training subject) is taken as the initial template, each further subject is aligned to the evolving template, which is updated as the running mean;
(ii) every subject is re-aligned to that template and the re-aligned data are averaged into the final template;
(iii) each subject's final rotation R is its Procrustes fit to the final template. <b>Non-circular split-half design:</b> the rotations are estimated on one half of the story
(140 TRs) and applied to the other half, in which the 7 segments are decoded (chance 1/7 = 14.29 %); halves are then swapped and the two accuracies averaged. Voxels are z-scored within each half separately.
Within every LOSO fold the template is built from the 17 training subjects only; the held-out subject is brought into the common space by a rotation estimated from its own alignment half against that template.
The anatomical baseline is the identical 7-class split-half procedure with R = I. Optionally the common space was reduced to the top-k principal axes of the training template (k = 5&ndash;80).</p>
<p><b>FC / ISFC (Part 3).</b> Python re-implementation of <code>run_isfc_p2m</code> / <code>isfc_p2m_nifti</code>. The seed time course is the mean of the z-scored seed voxels.
FC: Pearson r between a subject's seed and all of that subject's own voxels. ISFC: r between the seed of subject i and every voxel of the average of the other 17 subjects (leave-one-out; averages computed here, the <code>avg</code> folder was not used).
Subject maps were Fisher-z averaged over the 18 subjects (the helper uses a plain mean; the difference is below 0.04) and written as NIfTI (<code>results/map_*.nii.gz</code>).</p>

<h2>2. Task 0 &ndash; Stimulus&ndash;brain alignment in A1</h2>
<p>A1 was defined as ROIs 48 A1_R + 52 A1_L ({t0['n_voxels_A1']} voxels). The audio envelope was |Hilbert(x)| of the mono signal, low-pass filtered at 8 Hz and averaged within 1.5-s bins.
The .wav lasts {t0['audio_s']:.0f} s (291 complete TR bins), longer than the 280-TR scan: it starts with ~13 s of music and ~2 s of silence before the story. Following the course example code, the envelope was cropped to the 280 bins starting {t0['scan_start_TR']} TRs (13.5 s) into the file.
Cross-correlation was computed as MATLAB's <code>xcorr(&hellip;,'coeff')</code> for lags &minus;10&hellip;+10 TRs (positive lag = BOLD follows the sound).</p>
<div class="fig">{img('task0_xcorr.png')}<div class="cap"><b>Figure 1.</b> Cross-correlation between the audio envelope and A1 BOLD. Left: raw envelope. Middle: envelope convolved with the canonical (SPM double-gamma) HRF.
Grey: single subjects; red: group-average A1; dashed: 95th percentile of the surrogate peak distribution. Right: null distribution of the peak correlation (phase-randomised envelopes) with the observed group value.</div></div>
<ul>
<li><b>Peak lag and correlation.</b> Group-average A1: peak at <b>+{raw['group_peak_lag_TR']} TRs ({raw['group_peak_lag_s']:.1f} s), r = {raw['group_peak_r']:.2f}</b>; the correlation at lag 0 is {raw['group_r_at_lag0']:.2f}.
Single subjects: peak lag {sl.mean():.1f} &plusmn; {sl.std(ddof=1):.1f} TRs (range {sl.min()}&ndash;{sl.max()}), peak r = {sr.mean():.2f} &plusmn; {sr.std(ddof=1):.2f}.</li>
<li><b>Significance.</b> With 2,000 phase-randomised and 2,000 circularly shifted envelopes, using the maximum over all 21 lags as the statistic (which corrects for the lag search), no surrogate reached the observed group peak:
p = {raw['group_p_phase']:.4f} for both nulls (the smallest attainable p, 1/2001). At the single-subject level {nsig}/18 subjects are significant at p &lt; 0.05 (phase randomisation).</li>
<li><b>Hemodynamic delay.</b> The lag is positive and consistent across subjects, and {raw['group_peak_lag_s']:.1f} s is within the expected 4&ndash;6 s hemodynamic range. Convolving the envelope with the canonical HRF moves the group peak from +{raw['group_peak_lag_TR']} TRs to lag {hrf['group_peak_lag_TR']}
(single subjects: {hl.mean():.1f} &plusmn; {hl.std(ddof=1):.1f} TRs), raises the zero-lag correlation from {raw['group_r_at_lag0']:.2f} to {hrf['group_r_at_lag0']:.2f} and the peak correlation from {raw['group_peak_r']:.2f} to {hrf['group_peak_r']:.2f}: the canonical HRF accounts for the whole delay and for the temporal smoothing of the BOLD response.</li>
<li><b>Caveat on the absolute lag.</b> The position of the 280-TR scan inside the 438-s recording is not documented in the data files, so the lag is known only up to that offset. The 9-TR crop is supported by an independent check: sliding the scan window over the HRF-convolved envelope of the whole file gives the best zero-lag match at an offset of {best_off} TRs.
If the scan instead started exactly at story onset (15 s = 10 TRs), the raw peak lag would be +4 TRs (6 s). Either way the lag is 3&ndash;4 TRs, i.e. hemodynamic.</li>
<li><b>Why this matters.</b> A1 tracks the loudness of the story with a fixed, physiologically plausible delay in nearly every subject. This shows that the scans are locked to the stimulus and to each other to within about one TR.
All later analyses assume this: LOSO decoding and ISFC both rely on TR <i>t</i> meaning the same story moment in every subject. A misaligned or shifted subject would lower decoding and ISFC for purely technical reasons. Because BOLD lags the sound by ~3 TRs, each 20-TR "segment" reflects story content from roughly 4.5 s earlier, which shifts segment boundaries equally in all subjects and does not affect classification.</li>
</ul>
<div class="fig">{img('task0_timecourses.png', '92%')}<div class="cap"><b>Figure 2.</b> Group-average A1 BOLD (red) against the TR-binned envelope (grey), before and after HRF convolution.</div></div>

<h2 class="pb">3. Decoding results in the original voxel space (Task A)</h2>
<div class="fig">{img('part2a_bars_all_rois.png')}<div class="cap"><b>Figure 3.</b> LOSO 14-way segment-decoding accuracy (mean &plusmn; SEM over 18 folds) for all 52 ROIs and all classifiers. ROIs are in the same order in every panel (sorted by the mean over classifiers); dashed line: chance (7.14 %).</div></div>
<table><tr><th>Classifier</th><th>Mean over 52 ROIs (%)</th><th>Median (%)</th><th># ROIs where best</th><th>ROIs above chance (FDR q &lt; .05)</th></tr>{clf_rows}</table>
<div class="cap"><b>Table 1.</b> Classifier summary. Significance: one-sided t-test of the 18 fold accuracies against 7.14 %, Benjamini&ndash;Hochberg corrected over all 52&times;7 tests (LOSO folds share training data, so these p-values are approximate).</div>
<table><tr><th>ROI</th><th>V</th>{hdr}<th>Mean</th></tr>{top_rows}</table>
<div class="cap"><b>Table 2.</b> Ten leading ROIs (accuracy % &plusmn; SEM; bold = best classifier of the ROI). The full 52-ROI table is in the Appendix.</div>
<div class="fig">{img('part2a_heatmap.png')}<div class="cap"><b>Figure 4.</b> Accuracy of every classifier in every ROI.</div></div>
<p><b>Which regions.</b> The four auditory ROIs (A1_R, A1_L, AUB_L, AUB_R) lead by a wide margin (best single result: {A['top10'][0]['roi']}, LDA {max(A['top10'][0]['acc']):.1f} %, more than 11&times; chance), followed by precuneus, left IFG, middle temporal and inferior parietal ROIs,
i.e. the language and default-mode regions that follow the narrative. Even the weakest ROIs are mostly above chance. Accuracy grows with ROI size (r = {A['corr_logV_acc']:.2f} between log V and mean accuracy), so comparisons between ROIs of very different size mix information content with feature count; the bottom five ROIs have 5&ndash;28 voxels.</p>
<p><b>Which classifiers.</b> The linear SVM ({A['clf_mean'][jS]:.1f} %) and shrinkage LDA ({A['clf_mean'][jL]:.1f} %) are the best models on average, with Gaussian Naive Bayes close behind ({A['clf_mean'][jN]:.1f} %). LDA is the best model in {A['n_rois_best'][jL]} ROIs, including most of the large, high-signal ones; Naive Bayes wins in {A['n_rois_best'][jN]}, mostly small ROIs, and the SVM in {A['n_rois_best'][jS]}.
With 238 training samples and thousands of features, linear models with strong regularisation or a maximum-margin criterion do well, and a full (shrunk) covariance helps LDA where there is enough signal to estimate it.
KNN is clearly worst (22&ndash;24 %): raw Euclidean distances in a very high-dimensional space are dominated by subject-specific noise, and with one exemplar per subject and class the neighbourhood is sparse (K = 2 is slightly worse than K = 1 and 3 because two-neighbour votes are often tied).
QDA, which estimates 14 covariance matrices from 17 samples each, relies almost entirely on its shrinkage, behaves like a regularised nearest-centroid classifier and is on average below LDA.</p>

<h2 class="pb">4. Hyperalignment impact analysis (Task B)</h2>
<div class="fig">{img('part2b_scatter.png')}<div class="cap"><b>Figure 5.</b> Anatomical vs. hyperaligned accuracy in the identical split-half 7-class LOSO design (average of both half assignments). One dot per ROI; points above the diagonal improve with hyperalignment.</div></div>
<table><tr><th>Classifier</th><th>Anatomical (%)</th><th>Hyperaligned (%)</th><th>Gain (pp)</th><th>ROIs improved</th><th>Wilcoxon p (52 ROIs)</th></tr>{b_rows}</table>
<div class="cap"><b>Table 3.</b> Mean accuracy over the 52 ROIs with anatomical alignment and after hyperalignment (chance 14.29 %). The Wilcoxon signed-rank test compares the two across ROIs (ROIs are not independent, so p-values are descriptive).</div>
<div class="fig">{img('part2b_gain_per_roi.png')}<div class="cap"><b>Figure 6.</b> Per-ROI accuracy (mean of the 7 classifiers) and gain from hyperalignment, mean &plusmn; SEM over folds.</div></div>
<table style="width:49%;display:inline-table;vertical-align:top"><tr><th>Largest gains</th><th>V</th><th>Anat.</th><th>Hyper.</th><th>Gain &plusmn; SEM (pp)</th></tr>{gain_rows}</table>
<table style="width:49%;display:inline-table;vertical-align:top"><tr><th>Largest losses</th><th>V</th><th>Anat.</th><th>Hyper.</th><th>Gain (pp)</th></tr>{loss_rows}</table>
<div class="cap"><b>Table 4.</b> ROIs with the largest gains and losses (mean of the 7 classifiers, %).</div>
<table><tr><th rowspan="2">Classifier</th><th colspan="3">Decode 1st half (align on 2nd)</th><th colspan="3">Decode 2nd half (align on 1st)</th></tr>
<tr><th>Anat.</th><th>Hyper.</th><th>Gain</th><th>Anat.</th><th>Hyper.</th><th>Gain</th></tr>{bh_rows}</table>
<div class="cap"><b>Table 5.</b> The two split directions separately (mean accuracy over 52 ROIs, %).</div>
<p><b>Did hyperalignment help?</b> Modestly, and not uniformly. Averaged over all ROIs and classifiers accuracy rose from {B['overall_anat']:.1f} % to {B['overall_hyper']:.1f} % ({B['subj_gain_mean']:+.1f} pp per held-out subject, paired t-test over the 18 subjects p = {B['subj_gain_p']:.3f}); {B['n_rois_improved_mean']} of 52 ROIs improved.
Two qualifications are important. First, the average gain comes almost entirely from one split direction (Table 5): when the first half is decoded, accuracy rises from {bha[0]:.1f} % to {bhh[0]:.1f} %, but when the second half is decoded the anatomical baseline is already higher ({bha[1]:.1f} %) and hyperalignment leaves it unchanged ({bhh[1]:.1f} %).
Second, the effect differs strongly between regions. With a paired t-test over held-out subjects per ROI (FDR q &lt; .05), {B['n_sig_gain_fdr']} ROIs gain significantly and {B['n_sig_loss_fdr']} lose significantly.
The gains are concentrated in the <b>superior and middle temporal cortex around the auditory core</b> (pSTG, pSTS, STG, mid-MTG, TPJ; about +9 to +16 pp), and the four auditory ROIs reach the highest absolute accuracies (85&ndash;90 % with the linear SVM).
These are regions with a strong stimulus-driven response, where a finer functional correspondence between subjects than anatomy provides is plausible.
In contrast, the <b>precuneus and inferior parietal ROIs lost</b> 7&ndash;11 pp. The best-supported explanation is estimation noise: the rotation for these ROIs has to be estimated from only 140 TRs for 260&ndash;640 voxels (more free dimensions than time points, so the fit on the alignment half is perfect and partly fits noise),
and their slow story response may gain little from a finer alignment; this second point was not tested here.
Haxby et al. derived their transformations from a full-length movie (thousands of TRs); with our short halves the benefit is necessarily smaller. It is not related to ROI size (r = {B['corr_gain_logV']:.2f}).</p>
<p><b>Which classifiers changed most, and why?</b> The parametric / linear models gained 3&ndash;4.5 pp (QDA {B['gain_mean'][jQ]:+.1f}, SVM {B['gain_mean'][jS]:+.1f}, LDA {B['gain_mean'][jL]:+.1f}, Naive Bayes {B['gain_mean'][jN]:+.1f}), whereas KNN did not change ({B['gain_mean'][jK]:+.1f} pp for K = 1).
Models that learn one weight (or mean) per feature need feature j to mean the same thing in every subject; hyperalignment makes common-space dimensions functionally equivalent across subjects, so class means become sharper and weights learned on 17 subjects transfer to the 18th.
KNN compares whole patterns with Euclidean distance; a likely reason for the missing gain is that a rotation carries each pattern's noise along and the under-determined Procrustes fit adds alignment noise, which offsets the gain in shared signal.</p>
<p><b>Dimensionality of the common space (optional analysis, Table 6 / Figure 7).</b> For the 30 ROIs with V &gt; 80, a common space of only 20 dimensions already gives SVM, LDA and QDA accuracies within about one point of the full space and above the anatomical baseline,
and the best auditory ROIs are about as accurate at k = 20 as with all voxels: most of the shared story-driven signal is low-dimensional.
Naive Bayes is the exception: its accuracy falls as components are added (59.9 % at k = 5 to 37.7 % at k = 80) although it is high again in the full voxel basis. A plausible reason is that in the PC basis the later components carry little shared signal but enter the product of likelihoods with the same weight as the first ones; this was not tested separately.</p>
<table><tr><th>Common space</th>{hdr}</tr>{k_rows}</table>
<div class="cap"><b>Table 6.</b> Mean accuracy (%) over the 30 ROIs with V &gt; 80 as a function of the number of common-space dimensions.</div>
<div class="fig">{img('part2b_dimensionality.png', '88%')}<div class="cap"><b>Figure 7.</b> Accuracy vs. common-space dimensionality. Squares on the left panel: anatomical baseline.</div></div>
<p><b>Why must the alignment parameters come from independent data?</b> A Procrustes rotation has V(V&minus;1)/2 free parameters and is fitted to make subjects' time courses as similar as possible. If it were estimated on the same TRs that are later decoded, it would also align noise that happens to coincide in those TRs; in our case (V &gt; T for many ROIs) it could make the held-out subject's test data match the training template almost perfectly.
Since the classes here <i>are</i> time segments, "making time courses similar" is the same as "making the test samples resemble the correct class": information about the test labels would leak into the features and the accuracy would be inflated (double dipping).
The same logic requires the template to be built without the held-out subject. With the split-half design, any gain can only come from structure that generalises to new stimuli, i.e. from a real functional correspondence between subjects.</p>

<h2 class="pb">5. FC and ISFC cortical maps (Part 3)</h2>
<p><b>Seed selection.</b> The three ROIs with the highest Task-A accuracy (mean over the 7 classifiers) are <b>{tops[0]}</b> ({P3['mean_acc_pct'][0]:.1f} %), <b>{tops[1]}</b> ({P3['mean_acc_pct'][1]:.1f} %) and <b>{tops[2]}</b> ({P3['mean_acc_pct'][2]:.1f} %);
{r4[0]} is fourth ({r4[1]:.1f} %); with other criteria (best single classifier, or the SVM alone) AUB_R would replace one of the three, so the selection among the four auditory ROIs is not robust, but all candidates are auditory.
Because three auditory seeds give similar maps, the best non-auditory ROI, <b>{extra}</b> (rank 5), was added as a supplementary seed. Maps are thresholded at |r| &gt; 0.15 and share one colour scale.</p>
<div class="fig">{img('part3_maps_' + tops[0].replace(' ', '_') + '.png', '80%')}<div class="cap"><b>Figure 8.</b> Seed {tops[0]}: FC (top) and ISFC (bottom), orthogonal views through the seed centre and axial slices (MNI z in mm).</div></div>
<div class="fig">{img('part3_maps_' + tops[1].replace(' ', '_') + '.png', '80%')}<div class="cap"><b>Figure 9.</b> Seed {tops[1]}.</div></div>
<div class="fig">{img('part3_maps_' + tops[2].replace(' ', '_') + '.png', '80%')}<div class="cap"><b>Figure 10.</b> Seed {tops[2]}.</div></div>
<div class="fig">{img('part3_maps_' + extra.replace(' ', '_') + '.png', '80%')}<div class="cap"><b>Figure 11.</b> Supplementary seed {extra} (precuneus, default-mode network).</div></div>
<table><tr><th>Seed</th><th>V</th><th>Mean r inside seed (FC / ISFC)</th><th>Mean r, rest of brain (FC / ISFC)</th><th>Voxels with r &gt; 0.2 (FC / ISFC)</th><th>Max r outside seed (FC / ISFC)</th><th>Spatial corr. FC vs ISFC</th></tr>{p3_rows}</table>
<div class="cap"><b>Table 7.</b> Quantitative comparison of the FC and ISFC maps (three top seeds and the supplementary precuneus seed) (voxels outside the seed unless stated).</div>
<h3>a. Spatial differences between FC and ISFC</h3>
<p>FC maps are stronger and far more widespread. Besides bilateral superior temporal cortex they include the insula and opercula, thalamus and basal ganglia, medial frontal / cingulate and sensorimotor cortex and the cerebellum, with a positive offset across grey matter (mean r &asymp; {seeds[0]['fc_mean_r']:.2f} outside the seed; {min(fc_fr):.0f}&ndash;{max(fc_fr):.0f} % of voxels exceed r = 0.2) and a halo of very high values around the seed that decays with distance.
ISFC maps are weaker and spatially selective ({min(is_fr):.0f}&ndash;{max(is_fr):.0f} % of voxels above 0.2, mean close to zero): for the two A1 seeds essentially bilateral auditory cortex and STG, with small foci in inferior frontal cortex; for AUB_L (auditory belt) the map extends along the whole STG/STS into inferior frontal and premotor cortex, i.e. the speech and language network, and shows weak negative ISFC in medial parietal and midline regions.
The peak ISFC is in the homologous auditory cortex of the other subjects rather than in a local halo: correlation inside the seed region itself drops from ~0.68 (FC) to ~0.42 (ISFC). The two maps are nevertheless spatially correlated (r = {min(sc):.2f}&ndash;{max(sc):.2f}), so ISFC looks like the stimulus-locked subset of FC.
The supplementary precuneus seed shows the same contrast in a different network: FC and ISFC both outline default-mode regions, but ISFC is weaker inside the seed ({xs['isfc_within_seed_mean_r']:.2f} vs {xs['fc_within_seed_mean_r']:.2f}) and lacks the local halo.</p>
<h3>b. Why ISFC filters out non-neuronal artifacts and spontaneous fluctuations</h3>
<p>A voxel's signal can be written as stimulus-locked response + idiosyncratic neural activity + non-neuronal noise. Within one brain, all three components are shared between seed and target, so all three contribute to FC: respiration and cardiac pulsation, head motion and scanner drift affect many voxels of the same scan simultaneously, local smoothing and vascular coupling create the halo around the seed, and spontaneous (resting-state-like) fluctuations are coherent within intrinsic networks.
ISFC correlates the seed of one subject with voxels in <i>other</i> subjects, who were scanned at different times. Their breathing, heartbeat, motion, drift and spontaneous thoughts are statistically independent of the seed subject's, so their expected contribution to the cross-subject covariance is zero. The only component that is time-locked across brains is the response to the common stimulus.
Averaging the other 17 subjects before correlating additionally averages out their independent noise, which raises the signal-to-noise ratio of the stimulus-locked component. ISFC is therefore conservative: genuine but subject-specific neural coupling is removed along with the artifacts.</p>
<h3>c. Stimulus-locked, narrative-driven processing</h3>
<p>The ISFC maps show which regions respond to the story in a way that is shared across listeners and coupled to auditory cortex: strongest in early auditory areas, which track the acoustics (Task 0), and extending for the belt seed into STG/STS and inferior frontal regions that process speech and language over longer time scales.
The much broader FC pattern (medial frontal, sensorimotor, insular, subcortical, cerebellar) is strongly reduced in ISFC (thalamus and cerebellum remain at about the display threshold), indicating that much of this auditory "connectivity" reflects intrinsic or non-neuronal co-fluctuation rather than processing of the narrative.
This matches the decoding results: between-subject (LOSO) classification can only use response components shared across subjects, which is what ISFC isolates, and the regions with reliable stimulus-locked responses (auditory, STG/MTG, precuneus, IFG) are the ones that decode the story segment best.
It is consistent with Simony et al. (2016), who showed that ISFC reveals stimulus-dependent network configurations during narrative comprehension that standard FC, dominated by intrinsic correlations, does not.
All three top seeds are auditory, so their maps describe the auditory/speech end of the hierarchy; the supplementary precuneus seed (Figure 11) shows that shared, story-locked coupling is also present within the default-mode network, with {100*xs['isfc_frac_gt_0.2']:.0f} % of voxels above r = 0.2 in ISFC.</p>

<h2>6. Limitations</h2>
<ul><li>A sample is a 20-TR spatio-temporal segment, so decoding uses the temporal profile of the response as well as its spatial pattern; purely spatial (single-TR) decoding would be much harder.</li>
<li>Accuracy depends on ROI size, which ranges from 4 to 766 voxels; ROIs were not size-matched.</li>
<li>Hyperalignment was estimated from 140 TRs per half, fewer than the number of voxels in 22 ROIs; gains are probably underestimated relative to alignment on longer independent data.</li>
<li>LDA/QDA regularisation was fixed, not optimised; nested tuning could change their ranking. Significance tests over LOSO folds are approximate because folds share training subjects.</li>
<li>The absolute audio&ndash;scan offset is uncertain by about one TR (Section 2).</li></ul>

<h2>References</h2>
<p class="cap">Haxby, J. V., et al. (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. <i>Neuron, 72</i>, 404&ndash;416. &middot;
Nastase, S. A., et al. (2019). Measuring shared responses across subjects using intersubject correlation. <i>SCAN, 14</i>, 667&ndash;685. &middot;
Simony, E., et al. (2016). Dynamic reconfiguration of the default mode network during narrative comprehension. <i>Nature Communications, 7</i>, 12141. &middot;
Sch&ouml;nemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. <i>Psychometrika, 31</i>, 1&ndash;10. &middot;
Pereira, F., Mitchell, T., &amp; Botvinick, M. (2009). Machine learning classifiers and fMRI: a tutorial overview. <i>NeuroImage, 45</i>, S199&ndash;S209.</p>

<h2 class="pb">Appendix A &ndash; Task A, all ROIs (14-class accuracy %, mean &plusmn; SEM over 18 folds)</h2>
<table class="small"><tr><th>ROI</th><th>V</th>{hdr}</tr>{appA}</table>
<h2 class="pb">Appendix B &ndash; Task B, all ROIs (7-class accuracy %, anatomical &rarr; hyperaligned)</h2>
<table class="small"><tr><th>ROI</th><th>V</th>{hdr}</tr>{appB}</table>
</body></html>"""

out = ROOT / "report" / "report.html"
out.write_text(html)
js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
      f"await p.goto('file://{out}');await p.pdf({{path:'{ROOT / 'report' / 'Homework1_Report.pdf'}',format:'A4',printBackground:true}});await b.close();}})();")
r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
print(r.stderr or "PDF written")
