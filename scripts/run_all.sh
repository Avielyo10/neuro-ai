#!/usr/bin/env bash
# Reproduce every result, figure and the PDF report (about 15 minutes on 4 cores; Part 3 needs ~8 GB RAM
# the first time, when it builds cache/wholebrain.npz).
set -euo pipefail
cd "$(dirname "$0")/../src"
python3 task0_a1_alignment.py
python3 part2a_decoding.py
python3 supp_lda_sensitivity.py         # needs part2a results
python3 supp_temporal_vs_spatial.py     # needs part2a results
python3 part2b_hyperalignment.py        # ~8 min, uses 4 processes
python3 part3_fc_isfc.py                # needs part2a results (top-3 ROIs)
python3 make_figures.py
python3 make_report.py                  # PDF step needs node + the playwright npm package with Chromium (see README)
