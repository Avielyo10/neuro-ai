# NeuroAI – Homework 1: Brain State Decoding & Functional Connectivity (FC / ISFC)

Solution to Homework Assignment 1 (course 2700759): MVPA decoding of story segments, hyperalignment,
and FC / ISFC mapping on the 18-subject "Pieman" naturalistic fMRI dataset.

**Report:** [`report/Homework1_Report.pdf`](report/Homework1_Report.pdf)

## Layout

| Path | Content |
|---|---|
| `src/common.py` | data loading, VOI parsing, ROI extraction, caching |
| `src/task0_a1_alignment.py` | Task 0 – audio envelope vs. A1 BOLD cross-correlation, surrogate tests, HRF |
| `src/decoding.py` | classifiers (KNN, linear SVM, Naive Bayes, LDA, QDA) and LOSO cross-validation |
| `src/part2a_decoding.py` | Task A – 14-class LOSO decoding in all 52 ROIs |
| `src/part2b_hyperalignment.py` | Task B – Procrustes hyperalignment, split-half non-circular evaluation, Procrustes residuals |
| `src/part3_fc_isfc.py` | Part 3 – whole-brain FC and ISFC maps for the top-3 ROIs (common voxel mask for both maps) |
| `src/supp_lda_sensitivity.py` | supplementary – Task-A LDA with three covariance regularisers (report Table 4) |
| `src/supp_temporal_vs_spatial.py` | supplementary – linear SVM on the full sample vs. ROI-mean time course vs. time-averaged spatial pattern (report Table 5) |
| `src/make_figures.py`, `src/make_report.py` | figures, summary tables, PDF report |
| `scripts/download_data.sh`, `scripts/run_all.sh` | data download; full pipeline |
| `results/` | accuracies (`.npz`), summaries (`.json`), FC / ISFC maps (`.nii.gz`) |
| `figures/` | all figures used in the report |

## Reproduce

```bash
pip install -r requirements.txt
scripts/download_data.sh     # ~1.2 GB from the course Google Drive folder into data/ (not committed)
scripts/run_all.sh           # about 15 minutes on 4 cores
```

Requirements and notes:

- **scikit-learn >= 1.8** (the QDA model uses `QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.5)`, which
  older versions do not have). Tested with Python 3.13 and scikit-learn 1.9.
- **RAM:** `part3_fc_isfc.py` needs about 8 GB the first time (it builds `cache/wholebrain.npz`, 1.5 GB on disk).
  `cache/` and `data/` are not committed; delete `cache/` to rebuild everything from the raw files.
- **PDF build:** `make_report.py` writes `report/report.html` and converts it with headless Chromium through
  node and the `playwright` npm package (`npm install -g playwright && npx playwright install chromium`, with
  `NODE_PATH` pointing to the global `node_modules`). Without node the HTML is still written and can be printed
  from any browser.
- **Author line:** set `AUTHORS = "Name (ID), ..."` near the top of `src/make_report.py` and re-run
  `python3 make_report.py` (from `src/`); the title block then shows "Submitted by: ...". While it is empty, no
  author line is printed.
- The text of the report is generated from `results/*.json`; statements that depend on the data are guarded by
  checks in `make_report.py`, which stops with a message if a re-run no longer supports a sentence.

The 18 subjects are those in `story_selected_reg_wm_csf_hsd_last_half2` (the folder used by the course
helper scripts and containing the example subject `KM_060412`). The `avg` sub-folder is not used.
