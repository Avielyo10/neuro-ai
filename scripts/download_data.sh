#!/usr/bin/env bash
# Download the course dataset (18 subjects of story_selected_reg_wm_csf_hsd_last_half2, ROI file,
# template, audio) from the course Google Drive folder into data/.  Requires: pip install gdown
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
python3 - <<'PY'
import os, gdown
files = gdown.download_folder(id="1nM5OJb3WSP1FSa3Ch1dioW2-Roff5CSH", skip_download=True, quiet=True, output="x")
for f in files:
    if "/avg/" in f.path or "last_half1" in f.path:      # not needed for this assignment
        continue
    out = os.path.join("data", os.path.basename(f.path))
    if not os.path.exists(out):
        gdown.download(id=f.id, output=out, quiet=False)
PY
