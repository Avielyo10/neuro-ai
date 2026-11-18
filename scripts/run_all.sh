#!/usr/bin/env bash
# Reproduce every result, figure and the PDF report (about 10 minutes on 4 cores).
set -euo pipefail
cd "$(dirname "$0")/../src"
python3 task0_a1_alignment.py
python3 part2a_decoding.py
python3 part2b_hyperalignment.py
python3 part3_fc_isfc.py
python3 make_figures.py
python3 make_report.py     # needs node + playwright (Chromium) for the PDF step
