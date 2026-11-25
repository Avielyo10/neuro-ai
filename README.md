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

## Submission package: Colab notebook + PDF printout

The course asks for one zip file, named after the student ID, with a detailed Colab notebook and a PDF
printout of it. `notebook/build_submission.py` builds this package from the same analysis and text:

| Path | Content |
|---|---|
| `notebook/hw1_template.py` | source of the notebook (percent format): every code cell and the notebook-specific text. The notebook is self-contained (no imports from `src/`) and implements the same analysis as `src/` |
| `notebook/build_submission.py` | generates the notebook, executes it, checks its results against `results/`, stamps name and ID, renders the PDF, writes the zip |
| `notebook/print.css` | style sheet of the PDF printout |
| `submission/` | **not committed** (git-ignored): `student.json` and all outputs |

```bash
pip install nbformat nbclient nbconvert ipykernel psutil pandas
mkdir -p submission
echo '{"name": "Full Name", "id": "123456789"}' > submission/student.json   # never committed
python3 notebook/build_submission.py               # full build, about 15 minutes on 4 cores
python3 notebook/build_submission.py --stamp-only  # after editing student.json: seconds, no re-run
```

Outputs: `submission/HW1_<id>.ipynb` (executed, with outputs), `submission/HW1_<id>.pdf` and
`submission/<id>.zip`, which contains exactly these two files.

- **Text = results.** The markdown cells are generated from the prose blocks of `src/make_report.py`, so their
  numbers come from `results/*.json` and are protected by the same `check()` guards. The notebook computes
  everything again from the raw data in a clean work directory (`submission/build/work`) and saves its own
  results there; after the run the build compares them with `results/` (accuracies to 1e-9, correlations and
  maps to 1e-6) and stops if anything differs. It also stops if a cell wrote to stderr or raised an error.
- **Name and ID** are stamped into the title cell after execution. While `"name"` is empty the title shows
  "Name: (to be added)" and the build prints a warning. The PDF step needs node + playwright as described above
  and poppler (`pdftoppm`, `pdfimages`, `pdftotext`): where a figure would leave a large blank area at the bottom
  of a page, it is printed up to 15 % smaller so that it still fits.
- **Running the notebook elsewhere (Colab).** The first cells install missing packages (scikit-learn >= 1.8)
  and locate the data: environment variable `HW1_DATA_DIR`, a local folder, the course folder on Google Drive
  (`MyDrive/Data_for_fMRI_course`, sub-folder `story_selected_reg_wm_csf_hsd_last_half2`), or a download
  with `gdown`. Peak memory is about 4.5 GB; the text cells are static and describe the run embedded in the file.

The 18 subjects are those in `story_selected_reg_wm_csf_hsd_last_half2` (the folder used by the course
helper scripts and containing the example subject `KM_060412`). The `avg` sub-folder is not used.
