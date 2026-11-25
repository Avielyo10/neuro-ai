#!/usr/bin/env python3
"""Build the submission: one executed Colab notebook, its PDF printout and a zip of both.

    python3 notebook/build_submission.py               # full build (about 15 minutes on 4 cores)
    python3 notebook/build_submission.py --stamp-only  # new name / ID, no re-run (seconds)

Steps of a full build
  1. The markdown cells are generated from the prose blocks of src/make_report.py (values come
     from results/*.json and are protected by that module's check() guards); the code cells come
     from notebook/hw1_template.py.
  2. The notebook is executed from a clean work directory (submission/build/work) with the data
     in data/.  It saves its own results there.
  3. Those results are compared with the reference results in results/.  Any difference beyond
     the tolerances stops the build: the text of the notebook would not match its outputs.
  4. The student line is stamped into the title cell, the notebook is converted to HTML
     (nbconvert) and printed to PDF (headless Chromium through node + playwright), and the two
     files are zipped.

Student name and ID are read from submission/student.json ({"name": "...", "id": "..."}).
The whole submission/ folder is git-ignored; nothing in the tracked files contains the ID.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebook" / "hw1_template.py"
PRINT_CSS = ROOT / "notebook" / "print.css"
RESULTS = ROOT / "results"
SUBMISSION = ROOT / "submission"
BUILD = SUBMISSION / "build"
STUDENT_FILE = SUBMISSION / "student.json"
EXECUTED = BUILD / "HW1_executed.ipynb"          # executed, not yet stamped
HEADER_TAG = "hw1-header"
NAME_MISSING = "(to be added)"
MAX_CODE_LINE = 96                               # longer lines would wrap in the A4 printout

COURSE = "NeuroAI: Bridging Neuroscience and Artificial Intelligence"
COURSE_NO = "2700759"
TITLE = "Homework Assignment 1: Brain State Decoding & Functional Connectivity (FC / ISFC)"


# ------------------------------------------------------------------------------ student
def read_student() -> dict:
    if not STUDENT_FILE.exists():
        raise SystemExit(f'{STUDENT_FILE} is missing; create it with {{"name": "", "id": "<ID>"}}')
    student = json.loads(STUDENT_FILE.read_text())
    student = {"name": str(student.get("name", "")).strip(), "id": str(student.get("id", "")).strip()}
    if not re.fullmatch(r"\d{5,10}", student["id"]):
        raise SystemExit(f'{STUDENT_FILE}: "id" must be the student ID (digits only)')
    if not student["name"]:
        print(f"WARNING: no student name in {STUDENT_FILE}; the header will show "
              f'"Name: {NAME_MISSING}".  Fill in the name and run with --stamp-only.')
    return student


def header_markdown(student: dict | None) -> str:
    """Title cell.  `student` is None before stamping."""
    if student is None:
        line = "*(student name and ID are added by the build script)*"
    else:
        name = student["name"] or f"**{NAME_MISSING}**"
        line = f"Name: {name} &nbsp;·&nbsp; ID: {student['id']}"
    return (f"# {TITLE.replace('&', '&amp;')}\n\n"
            f"**Course:** {COURSE}, course no. {COURSE_NO}, Fall Semester 2026\n\n"
            f"**Submitted by:** {line}\n\n"
            'Multi-voxel pattern analysis, hyperalignment and inter-subject functional correlation on the '
            '"Pieman" data set: 18 subjects, TR = 1.5 s, 280 TRs, 52 ROIs. Code: Python, all of it in this notebook.')


# ------------------------------------------------------------------------------ prose
# Replacements that adapt the report prose to the notebook (each must match exactly once).
PROSE_EDITS = {
    "methods": [("own implementation in <code>decoding.py</code>, double precision",
                 "own implementation, class <code>ShrinkageLDA</code> in Section 1.1, double precision"),
                ("written as NIfTI (<code>results/map_*.nii.gz</code>)",
                 "written as NIfTI (<code>map_*.nii.gz</code> in the work directory)"),
                ("so its FC branch was ported line by line and run",
                 "so its FC branch was ported line by line (<code>helper_fc_map</code>, Section 5.1) and run")],
    "lda": [("(<code>src/supp_lda_sensitivity.py</code>; NB = Naive Bayes)", "(Section 3.5; NB = Naive Bayes)")],
    "tvs": [("(<code>src/supp_temporal_vs_spatial.py</code>)", "(Section 3.6)")],
}
UNUSED_BLOCKS = {"title"}                        # the notebook has its own title cell
KEEP_ENTITIES = {"&lt;": "\x00lt\x00", "&gt;": "\x00gt\x00", "&amp;": "\x00amp\x00"}
TOP_LEVEL = re.compile(
    r"\s*(?:<(h2|h3|p|ul|ol)(?:\s[^>]*)?>(.*?)</\1>"
    r"|<div class=\"(?:fig|tbl)\">.*?<div class=\"cap\">(.*?)</div></div>"
    r"|(<table class=\"small\">.*?</table>))", re.S)


def inline_md(text: str) -> str:
    """Inline HTML of the report (b, i, code, entities) -> Markdown, on one line."""
    text = " ".join(text.split())
    for ent, token in KEEP_ENTITIES.items():
        text = text.replace(ent, token)
    text = html.unescape(text)
    parts = re.split(r"(<code>.*?</code>)", text)
    for i, part in enumerate(parts):
        if part.startswith("<code>"):
            code = part[6:-7]
            for ent, token in KEEP_ENTITIES.items():
                code = code.replace(token, html.unescape(ent))
            assert "`" not in code and "<" not in code, code
            parts[i] = f"`{code}`"
        else:
            assert not re.search(r"[*`$\\\[\]]", part), f"character needs escaping: {part!r}"
            part = part.replace("~", r"\~")
            part = re.sub(r"</?b>", "**", part)
            part = re.sub(r"</?i>", "*", part)
            assert "<" not in part and ">" not in part, f"unexpected tag in {part!r}"
            parts[i] = part
    text = "".join(parts)
    for ent, token in KEEP_ENTITIES.items():
        text = text.replace(token, ent)
    return text


def parse_block(key: str, block: str) -> dict:
    """Split one HTML block of make_report.BLK into Markdown elements: {'h2': [...], 'p': [...], ...}."""
    block = re.sub(r'<img src="data:image/png;base64,[^"]*"[^>]*>', "", block)
    for old, new in PROSE_EDITS.get(key, []):
        assert block.count(old) == 1, f"prose edit for block {key!r} does not match exactly once: {old!r}"
        block = block.replace(old, new)
    assert not re.search(r"src/|scripts/|results/|\.py\b", re.sub(r"<table.*?</table>", "", block, flags=re.S)), \
        f"block {key!r} still refers to repository files"
    out, pos = {}, 0
    while block[pos:].strip():
        m = TOP_LEVEL.match(block, pos)
        assert m, f"cannot parse block {key!r} at: {block[pos:pos + 80]!r}"
        pos = m.end()
        tag, body, cap, table = m.groups()
        if tag in ("h2", "h3"):
            out.setdefault(tag, []).append(("## " if tag == "h2" else "#### ") + inline_md(body))
        elif tag == "p":
            out.setdefault("p", []).append(inline_md(body))
        elif tag in ("ul", "ol"):
            items = re.findall(r"<li>(.*?)</li>", body, re.S)
            assert items and not re.sub(r"<li>.*?</li>", "", body, flags=re.S).strip()
            mark = "- " if tag == "ul" else "1. "
            out.setdefault(tag, []).append("\n".join(mark + inline_md(it) for it in items))
        elif cap is not None:
            out.setdefault("cap", []).append(inline_md(cap))
        # appendix tables: produced by notebook code, nothing to keep
    return out


def load_prose():
    """Import src/make_report.py (runs its check() guards on results/*.json) and convert its blocks."""
    sys.path.insert(0, str(ROOT / "src"))
    import make_report as mr                     # noqa: E402  (no side effects at import)
    prose = {k: parse_block(k, mr.BLK[k]) for k in mr.ORDER if k not in UNUSED_BLOCKS}
    return prose, mr


# ------------------------------------------------------------------------------ notebook
def parse_template() -> list[tuple[str, str, str]]:
    """Cells of the percent-format template: (kind, argument, text)."""
    cells, kind, arg, buf = [], None, "", []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            lines = text.split("\n")
            assert all(ln == "#" or ln.startswith("# ") for ln in lines), f"bad markdown cell: {text[:80]!r}"
            text = "\n".join(ln[2:] for ln in lines)
        cells.append((kind, arg, text))

    for line in TEMPLATE.read_text().split("\n"):
        m = re.fullmatch(r"# %%(?: \[(\w+)\])?(?: (.*))?", line)
        if m:
            flush()
            kind, arg, buf = m.group(1) or "code", (m.group(2) or "").strip(), []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def build_notebook(prose: dict, mr):
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    used, cells = set(), []
    for kind, arg, text in parse_template():
        if kind == "code":
            long = [ln for ln in text.split("\n") if len(ln) > MAX_CODE_LINE]
            assert not long, f"code lines longer than {MAX_CODE_LINE} characters: {long}"
            cells.append(new_code_cell(text))
        elif kind == "markdown":
            cells.append(new_markdown_cell(text))
        elif kind == "header":
            cells.append(new_markdown_cell(header_markdown(None), metadata={"tags": [HEADER_TAG]}))
        elif kind == "prose":
            parts = []
            for ref in arg.split():
                m = re.fullmatch(r"(\w+)\.(\w+)(?:\[(\d+)\])?", ref)
                assert m, f"bad prose reference {ref!r}"
                block, elem, idx = m.group(1), m.group(2), m.group(3)
                items = prose[block][elem]
                for i in ([int(idx)] if idx is not None else range(len(items))):
                    assert (block, elem, i) not in used, f"prose element used twice: {ref}"
                    used.add((block, elem, i))
                    parts.append(items[i])
            cells.append(new_markdown_cell("\n\n".join(parts)))
        else:
            raise AssertionError(f"unknown cell kind {kind!r}")
    # every prose element of the report must appear in the notebook exactly once
    missing = [(b, e, i) for b, els in prose.items() for e, items in els.items()
               for i in range(len(items)) if (b, e, i) not in used]
    assert not missing, f"report prose not used in the notebook: {missing}"
    # captions must appear in numerical order and be complete
    text = "\n".join(c.source for c in cells if c.cell_type == "markdown")
    for kind, numbering in (("Table", mr.TAB), ("Figure", mr.FIG)):
        found = [int(x) for x in re.findall(rf"^\*\*{kind} (\d+)\.\*\*", text, re.M)]
        assert found == list(range(1, len(numbering) + 1)), f"{kind} captions out of order: {found}"
    # constants that the notebook code shares with the report prose
    code = "\n".join(c.source for c in cells if c.cell_type == "code")
    for const in (f"LDA_ROIS = {json.dumps(mr.LDA_ROIS)}", f"TS_ROIS = {json.dumps(mr.TS_ROIS)}",
                  f"N_GAIN = {mr.N_GAIN}"):
        assert const in code, f"notebook code and make_report.py disagree: {const}"
    for i, c in enumerate(cells):
        c["id"] = f"hw1-{i:03d}"
    nb = new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    nbformat.validate(nb)
    return nb


# ------------------------------------------------------------------------------ execution
class MemoryMonitor(threading.Thread):
    """Samples the memory of this process' descendants (the kernel and its workers)."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval, self.peak_rss, self.peak_pss, self.peak_kernel, self._stop_evt = interval, 0, 0, 0, threading.Event()

    def run(self):
        import psutil
        me = psutil.Process()
        while not self._stop_evt.wait(self.interval):
            rss = pss = kernel = 0
            for p in me.children(recursive=True):
                try:
                    if "node" in p.name():
                        continue
                    info = p.memory_full_info()
                    rss, pss = rss + info.rss, pss + getattr(info, "pss", info.rss)
                    if "ipykernel" in " ".join(p.cmdline()):
                        kernel = max(kernel, info.rss)
                except Exception:                # process ended between listing and reading
                    pass
            self.peak_rss, self.peak_pss = max(self.peak_rss, rss), max(self.peak_pss, pss)
            self.peak_kernel = max(self.peak_kernel, kernel)

    def stop(self):
        self._stop_evt.set()


def execute(nb, data_dir: Path, work_dir: Path):
    from nbclient import NotebookClient

    run_dir = BUILD / "run"                      # an empty directory: nothing of the repo is visible
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    os.environ["HW1_DATA_DIR"], os.environ["HW1_WORK_DIR"] = str(data_dir), str(work_dir)
    os.environ.pop("PYTHONPATH", None)
    monitor = MemoryMonitor()
    monitor.start()
    t0 = time.time()
    print(f"executing the notebook (data: {data_dir}, work directory: {work_dir}) ...", flush=True)
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(run_dir)}}).execute()
    for cell in nb.cells:                        # join consecutive chunks of the same stream; the order is kept
        merged = []
        for o in cell.get("outputs", []):
            if merged and o.output_type == "stream" and merged[-1].output_type == "stream" and merged[-1].name == o.name:
                merged[-1].text += o.text
            else:
                merged.append(o)
        if cell.cell_type == "code":
            cell.outputs = merged
    monitor.stop()
    info = {"wall_time_s": time.time() - t0, "peak_rss_all_processes_gb": monitor.peak_rss / 1e9,
            "peak_pss_all_processes_gb": monitor.peak_pss / 1e9, "peak_rss_kernel_gb": monitor.peak_kernel / 1e9}
    print(f"executed in {info['wall_time_s'] / 60:.1f} min; peak memory: kernel {info['peak_rss_kernel_gb']:.2f} GB, "
          f"kernel + workers {info['peak_pss_all_processes_gb']:.2f} GB (PSS; sum of RSS {info['peak_rss_all_processes_gb']:.2f} GB)")
    return nb, info


def check_outputs(nb):
    """No errors, no stderr, and every code cell has run."""
    problems = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if c.get("execution_count") is None:
            problems.append(f"cell {i}: not executed")
        for o in c.get("outputs", []):
            if o.output_type == "error":
                problems.append(f"cell {i}: error {o.ename}: {o.evalue}")
            if o.output_type == "stream" and o.name == "stderr":
                problems.append(f"cell {i}: stderr output: {o.text[:300]!r}")
    if problems:
        raise SystemExit("the executed notebook is not clean:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------------------ verification
TOL_ACC = 1e-9        # accuracies and statistics derived from them
TOL_CORR = 1e-6       # correlations, Procrustes residuals, FC / ISFC maps
JSON_FILES = {"task0_alignment.json": TOL_CORR, "part2a_summary.json": TOL_ACC, "supp_lda_sensitivity.json": TOL_ACC,
              "supp_temporal_vs_spatial.json": TOL_ACC, "part2b_summary.json": TOL_ACC, "part3_summary.json": TOL_CORR}
LOOSE_KEYS = {"procrustes_residual", "resid_train", "resid_heldout", "a_span", "a_span_voxel_equiv", "a_voxel",
              "shrinkage", "selfcheck"}       # floating-point quantities inside the accuracy files


def compare_json(a, b, tol, path, diffs, stats):
    """Recursive comparison of notebook value `a` with reference value `b`."""
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            diffs.append(f"{path}: keys differ")
            return
        for k in b:
            compare_json(a[k], b[k], TOL_CORR if k in LOOSE_KEYS else tol, f"{path}/{k}", diffs, stats)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            diffs.append(f"{path}: lists differ in length")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare_json(x, y, tol, f"{path}[{i}]", diffs, stats)
    elif isinstance(b, float) and not isinstance(a, (bool, str)) and a is not None:
        d = abs(a - b)
        stats["max"] = max(stats["max"], d)
        stats["n"] += 1
        if not d <= tol * max(1.0, abs(b)):
            diffs.append(f"{path}: notebook {a!r} vs reference {b!r} (difference {d:.3g})")
    else:                                        # int, bool, str, None: exact
        stats["n"] += 1
        if type(a) is not type(b) or a != b:
            diffs.append(f"{path}: notebook {a!r} vs reference {b!r}")


def verify(work: Path, mr) -> list[str]:
    """Compare everything the notebook saved with the reference results.  Returns report lines."""
    import nibabel as nib
    import numpy as np

    lines, diffs = [], []
    for name, tol in JSON_FILES.items():
        stats, n0 = {"max": 0.0, "n": 0}, len(diffs)
        compare_json(json.loads((work / name).read_text()), json.loads((RESULTS / name).read_text()), tol, name, diffs, stats)
        lines.append(f"{name:32s} {stats['n']:6d} values, max |diff| of floats {stats['max']:.2e}, "
                     f"{len(diffs) - n0} beyond tolerance")
    for name, exact in (("part2a_accuracy.npz", ["acc"]), ("part2b_accuracy.npz", ["anat", "hyper", "reduced"])):
        new, ref = np.load(work / name), np.load(RESULTS / name)
        assert set(new.files) == set(ref.files), f"{name}: arrays differ"
        for key in ref.files:
            x, y = new[key], ref[key]
            if x.shape != y.shape:
                diffs.append(f"{name}/{key}: shape {x.shape} vs {y.shape}")
            elif y.dtype.kind in "fc":
                same_nan = np.array_equal(np.isnan(x), np.isnan(y))
                d = float(np.nanmax(np.abs(x - y))) if same_nan else float("inf")
                n_diff = int(np.sum(np.abs(x - y) > 0)) if same_nan else -1
                tol = TOL_ACC if key in exact else TOL_CORR
                lines.append(f"{name + '/' + key:32s} {x.size:6d} values, max |diff| {d:.2e}, {n_diff} not bit-identical")
                if not d <= tol:
                    where = np.argwhere(np.nan_to_num(np.abs(x - y)) > tol)[:5].tolist()
                    diffs.append(f"{name}/{key}: max difference {d:.3g} (first positions {where})")
            elif not np.array_equal(x, y):
                diffs.append(f"{name}/{key}: differs")
    maps = sorted(p.name for p in RESULTS.glob("map_*.nii.gz"))
    assert maps == sorted(p.name for p in work.glob("map_*.nii.gz")), "set of NIfTI maps differs"
    dmax = 0.0
    for name in maps:
        x, y = nib.load(work / name), nib.load(RESULTS / name)
        d = float(np.abs(x.get_fdata() - y.get_fdata()).max())
        dmax = max(dmax, d)
        if not (d <= TOL_CORR and np.array_equal(x.affine, y.affine)):
            diffs.append(f"{name}: max difference {d:.3g}")
    lines.append(f"{len(maps)} FC / ISFC maps (NIfTI)        max |diff| {dmax:.2e}")
    info = json.loads((work / "roi_info.json").read_text())
    if info["n_mapped"] != mr.n_mapped or info["n_kept"] != mr.n_kept:
        diffs.append("roi_info.json: voxel counts differ from those quoted in the Methodology text")
    lines.append("ROI voxel counts (Methodology)   identical" if not diffs or "roi_info" not in diffs[-1] else "ROI voxel counts differ")
    if diffs:
        raise SystemExit("\n".join(lines) + f"\n\nNOTEBOOK RESULTS DIFFER FROM THE REFERENCE RESULTS ({len(diffs)} values); "
                         "the text of the notebook would not match its outputs:\n  " + "\n  ".join(diffs[:60]))
    return lines


# ------------------------------------------------------------------------------ PDF
def notebook_html(nb, student: dict, scales: dict | None = None) -> str:
    """HTML page of the notebook for printing: nbconvert 'basic' body + notebook/print.css.

    `scales` maps the running number of a figure (0, 1, ...) to a factor < 1 by which it is printed smaller."""
    from nbconvert import HTMLExporter
    from pygments.formatters import HtmlFormatter

    body, _ = HTMLExporter(template_name="basic").from_notebook_node(nb)
    # image width given in the notebook (px, 1000 = full width) -> percentage of the page width
    count = iter(range(10 ** 6))
    body = re.sub(r'(<img [^>]*?)\bwidth="(\d+)"',
                  lambda m: f'{m.group(1)}style="width:{min(int(m.group(2)) / 10, 100) * (scales or {}).get(next(count), 1):.1f}%"', body)
    # caption cells ("Figure n." / "Table n.") stay with the output above them
    body = re.sub(r'<div class="cell border-box-sizing text_cell rendered"( id="[^"]*")?>(?=<div class="prompt input_prompt">\s*</div>'
                  r'<div class="inner_cell">\s*<div class="text_cell_render border-box-sizing rendered_html">\s*<p><strong>(?:Figure|Table) \d+\.</strong>)',
                  r'<div class="cell border-box-sizing text_cell rendered caption"\1>', body)
    # each appendix starts on a new page
    body = re.sub(r'<div class="cell border-box-sizing text_cell rendered"(?=( id="[^"]*")?><div class="prompt input_prompt">\s*</div>'
                  r'<div class="inner_cell">\s*<div class="text_cell_render border-box-sizing rendered_html">\s*<h2 id="Appendix)',
                  '<div class="cell border-box-sizing text_cell rendered new-page"', body)
    # ... and so does the second table of Appendix B (its one-line introduction would otherwise be
    # stranded at the bottom of the page that the first table fills)
    body = re.sub(r'<div class="cell border-box-sizing text_cell rendered"(?=( id="[^"]*")?><div class="prompt input_prompt">\s*</div>'
                  r'<div class="inner_cell">\s*<div class="text_cell_render border-box-sizing rendered_html">\s*<p><strong>B\.2 )',
                  '<div class="cell border-box-sizing text_cell rendered new-page"', body)
    md = [c.source for c in nb.cells if c.cell_type == "markdown"]
    assert body.count("rendered caption") == sum(m.startswith(("**Figure ", "**Table ")) for m in md) > 0
    assert body.count("rendered new-page") == sum(m.startswith(("## Appendix", "**B.2 ")) for m in md) > 0
    # short tables are kept on one page
    def keep(m):
        size = "short" if m.group(0).count("<tr") <= KEEP_TABLE_ROWS else "long"
        return m.group(0).replace("<table", f'<table data-{size}="1"', 1)
    body = re.sub(r"<table.*?</table>", keep, body, flags=re.S)
    css = PRINT_CSS.read_text() + HtmlFormatter(style="default").get_style_defs(".highlight")
    title = f"HW1_{student['id']}"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f"<style>{css}</style></head><body>{body}</body></html>")


PAGE_H, MARGIN_TOP, MARGIN_BOTTOM, CONTENT_W = 297.0, 13.0, 14.0, 186.0     # mm; must agree with print.css
BODY_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
KEEP_TABLE_ROWS = 14        # tables up to this many rows are never split across pages
MAX_GAP = 0.20              # blank fraction at the bottom of a page that triggers a smaller figure
MIN_SCALE = 0.85            # a figure is never printed smaller than this fraction of its normal size (legibility)
CAPTION_MM = 16.0           # room kept for the caption below a figure
MAX_RENDERS = 40            # upper limit of trial renderings in the layout search


def page_gaps(pdf: Path) -> list[float]:
    """Blank fraction at the bottom of the text area of every page (from a low-resolution raster)."""
    import numpy as np
    from PIL import Image

    tmp = BUILD / "raster"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    subprocess.run(["pdftoppm", "-r", "30", "-gray", "-png", str(pdf), str(tmp / "p")], check=True)
    gaps = []
    for f in sorted(tmp.glob("p-*.png")):
        img = np.asarray(Image.open(f).convert("L"))
        mm = img.shape[0] / PAGE_H
        top, bottom = int(MARGIN_TOP * mm), int((PAGE_H - MARGIN_BOTTOM) * mm)
        rows = np.flatnonzero((img[top:bottom] < 245).any(axis=1))
        gaps.append(float(1.0 - (rows.max() + 1) / (bottom - top)) if len(rows) else 1.0)
    shutil.rmtree(tmp)
    return gaps


def figure_pages(pdf: Path) -> list[tuple[int, float]]:
    """(page, height / width) of every figure in the PDF, in document order."""
    out = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout
    rows = [ln.split() for ln in out.splitlines()[2:]]
    return [(int(r[0]), int(r[4]) / int(r[3])) for r in rows if r[2] == "image"]


def print_pdf(page: Path, pdf: Path, footer: str):
    js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
          f"await p.goto({json.dumps('file://' + str(page))});await p.pdf({{path:{json.dumps(str(pdf))},format:'A4',"
          "printBackground:true,preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<span></span>',"
          f"footerTemplate:{json.dumps(footer)}}});await b.close();}})();")
    try:
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("node not found: the PDF step needs node and the playwright npm package (see README)")
    if r.returncode != 0:
        raise SystemExit("PDF step failed (is the playwright npm package with Chromium installed?):\n" + r.stderr)


def render_pdf(nb, student: dict, pdf: Path, reuse: bool = False):
    """Print the notebook to PDF.  A figure that does not fit into the rest of a page moves to the next page and
    leaves a blank area; where that area is large and the figure would fit after a moderate reduction, the figure
    is printed smaller instead (checked on the rendered PDF, one figure at a time)."""
    page = BUILD / "printout.html"
    widths = [min(int(w) / 10, 100) / 100 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if "image/png" in o.get("data", {}) for w in [o.get("metadata", {}).get("image/png", {}).get("width", 1000)]]
    who = f"{student['name']}, ID {student['id']}" if student["name"] else f"ID {student['id']}"
    footer = ('<div style="font-family:Arial,sans-serif;font-size:7pt;color:#555;width:100%;padding:0 12mm;'
              'display:flex;justify-content:space-between">'
              f"<span>NeuroAI ({COURSE_NO}) – Homework 1 – {html.escape(who)}</span>"
              '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    saved = BUILD / "print_scales.json"

    def render(scales):
        page.write_text(notebook_html(nb, student, scales))
        print_pdf(page, pdf, footer)
        figs = figure_pages(pdf)
        assert len(figs) == len(widths), "figures of the notebook and images of the PDF do not match"
        return page_gaps(pdf), figs

    def worst(gaps):
        """Largest blank area at the bottom of a page; the last page and the pages before an appendix (forced
        page break) do not count."""
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
        forced = {p for p, t in enumerate(text.split("\f"), start=1) if t.lstrip().startswith("Appendix")}
        return max([g for p, g in enumerate(gaps[:-1], start=1) if p + 1 not in forced], default=0.0)

    budget = [MAX_RENDERS]

    def search(scales, fixed):
        """Depth-first search over 'print this figure smaller' / 'leave it': returns (largest gap, scales)."""
        budget[0] -= 1
        gaps, figs = render(scales)
        best = (worst(gaps), -sum(scales.values()) + len(scales), dict(scales))
        for p, gap in enumerate(gaps[:-1], start=1):
            i = next((k for k, (pg, _) in enumerate(figs) if pg == p + 1), None)
            if gap <= MAX_GAP or i is None or i in fixed or i in scales:
                continue
            scale = (gap * BODY_H - CAPTION_MM) / (widths[i] * CONTENT_W * figs[i][1])
            options = [{**scales, i: round(float(scale), 3)}] if MIN_SCALE <= scale < 1 else []
            for option in options + [scales]:                # smaller figure first, then "leave it"
                if budget[0] > 0:
                    best = min(best, search(option, fixed | {i}), key=lambda b: (round(b[0], 2), b[1]))
            break
        return best

    if reuse and saved.exists():                     # --stamp-only: the layout does not change
        scales = {int(k): v for k, v in json.loads(saved.read_text()).items()}
    else:
        scales = search({}, frozenset())[2]
        saved.write_text(json.dumps(scales))
    gaps, figs = render(scales)
    print(f"  PDF: {len(gaps)} pages; figures printed smaller to fill a page: "
          f"{ {i + 1: v for i, v in sorted(scales.items())} or 'none'}; largest blank area at the bottom of a page "
          f"(last page and pages before an appendix excluded): {worst(gaps) * 100:.0f} % of the text height")


# ------------------------------------------------------------------------------ stamping and packaging
def stamp_and_package(student: dict, reuse_layout: bool = False):
    """Stamp name / ID into the executed notebook, render the PDF and write the zip."""
    import nbformat

    if not EXECUTED.exists():
        raise SystemExit(f"{EXECUTED} not found: run a full build first (without --stamp-only)")
    nb = nbformat.read(EXECUTED, as_version=4)
    headers = [c for c in nb.cells if HEADER_TAG in c.metadata.get("tags", [])]
    assert len(headers) == 1 and headers[0] is nb.cells[0], "title cell not found"
    headers[0].source = header_markdown(student)
    nbformat.validate(nb)
    out = BUILD / "out"                           # build the three files here, then replace the old ones
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    names = [f"HW1_{student['id']}.ipynb", f"HW1_{student['id']}.pdf", f"{student['id']}.zip"]
    nbformat.write(nb, out / names[0])
    render_pdf(nb, student, out / names[1], reuse=reuse_layout)
    with zipfile.ZipFile(out / names[2], "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / names[0], names[0])
        z.write(out / names[1], names[1])
    with zipfile.ZipFile(out / names[2]) as z:
        assert sorted(z.namelist()) == sorted(names[:2])
    for old in list(SUBMISSION.glob("HW1_*.ipynb")) + list(SUBMISSION.glob("HW1_*.pdf")) + list(SUBMISSION.glob("*.zip")):
        old.unlink()                             # outputs of an earlier build (possibly another ID)
    ipynb, pdf, archive = (Path(shutil.move(out / n, SUBMISSION / n)) for n in names)
    out.rmdir()
    for path in (ipynb, pdf, archive):
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size / 1e6:.1f} MB)")
    if not student["name"]:
        print(f'WARNING: the header shows "Name: {NAME_MISSING}".  Put the name into {STUDENT_FILE.relative_to(ROOT)} '
              "and run:  python3 notebook/build_submission.py --stamp-only")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stamp-only", action="store_true",
                    help="re-stamp name / ID into the already executed notebook, re-render the PDF and the zip")
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="folder with the course data (default: data/)")
    ap.add_argument("--keep-work", action="store_true",
                    help="do not delete submission/build/work first (re-uses the cached ROI data)")
    ap.add_argument("--notebook-only", action="store_true",
                    help="only write the unexecuted notebook to submission/build/ (for inspection)")
    args = ap.parse_args()

    student = read_student()
    BUILD.mkdir(parents=True, exist_ok=True)
    if not args.stamp_only:
        import nbformat

        prose, mr = load_prose()
        nb = build_notebook(prose, mr)
        nbformat.write(nb, BUILD / "HW1_unexecuted.ipynb")
        print(f"notebook generated: {len(nb.cells)} cells")
        if args.notebook_only:
            return
        work = BUILD / "work"
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        EXECUTED.unlink(missing_ok=True)         # never stamp the notebook of an older build
        nb, info = execute(nb, args.data.resolve(), work)
        nbformat.write(nb, BUILD / "HW1_executed_unverified.ipynb")     # for debugging a failed check
        check_outputs(nb)
        lines = verify(work, mr)
        print("notebook results vs. reference results (results/):\n  " + "\n  ".join(lines))
        (BUILD / "build_info.json").write_text(json.dumps({**info, "verification": lines}, indent=1))
        nbformat.write(nb, EXECUTED)
    stamp_and_package(student, reuse_layout=args.stamp_only)


if __name__ == "__main__":
    main()
