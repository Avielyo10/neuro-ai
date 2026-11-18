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
| `src/part2b_hyperalignment.py` | Task B – Procrustes hyperalignment, split-half non-circular evaluation |
| `src/part3_fc_isfc.py` | Part 3 – whole-brain FC and ISFC maps for the top-3 ROIs |
| `src/make_figures.py`, `src/make_report.py` | figures, summary tables, PDF report |
| `results/` | accuracies (`.npz`), summaries (`.json`), FC / ISFC maps (`.nii.gz`) |
| `figures/` | all figures used in the report |

## Reproduce

```bash
pip install -r requirements.txt
scripts/download_data.sh     # ~1.2 GB from the course Google Drive folder into data/ (not committed)
scripts/run_all.sh
```

The 18 subjects are those in `story_selected_reg_wm_csf_hsd_last_half2` (the folder used by the course
helper scripts and containing the example subject `KM_060412`). The `avg` sub-folder is not used.
