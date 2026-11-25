# Source of the submission notebook (percent format).  notebook/build_submission.py turns this
# file into a Jupyter / Colab notebook:
#
#   "# %%"                      code cell (everything up to the next marker, verbatim)
#   "# %% [markdown]"           markdown cell; the text is written as "# " comment lines
#   "# %% [prose] a.p b.cap"    markdown cell filled with prose blocks of src/make_report.py
#                               (values come from results/*.json; see build_submission.py)
#   "# %% [header]"             title cell; the student line is stamped in after execution
#
# The file is plain Python, so it can also be run as a script for debugging:
#   HW1_DATA_DIR=data HW1_WORK_DIR=/tmp/hw1_work python3 notebook/hw1_template.py
# No student name or ID may appear in this file.

# %% [header]

# %% [markdown]
# ## About this notebook
#
# This notebook is the complete submission: for every part of the assignment it contains the
# code, the outputs of that code (figures and tables) and the written answers.
#
# | Section | Content | Run time (4 CPU cores) |
# |---|---|---|
# | 0 | Setup, data location, ROI extraction | under 1 min |
# | 1 | Methodology; classifiers and LOSO cross-validation | – |
# | 2 | Task 0: audio envelope vs. A1 BOLD | under 0.5 min |
# | 3 | Task A: decoding in the original voxel space, two supplementary analyses | about 2 min |
# | 4 | Task B: hyperalignment | about 7 min |
# | 5 | Part 3: FC and ISFC maps | about 2 min |
# | 6 | Limitations, references, appendix tables, run summary | – |
#
# **How to run.** Run all cells from top to bottom (about 12 minutes on 4 CPU cores; expect
# roughly twice that on the Google Colab free tier, which has 2 cores). Peak memory is about
# 4.5 GB, worker processes included. The only cell that may need editing is the
# configuration cell in Section 0.3, which locates the course data (a local folder, the
# course folder on Google Drive, or an automatic download of about 1.2 GB). If the first
# code cell asks for a restart of the runtime, restart and run all cells again.
#
# **Text and numbers.** The numbers quoted in the text cells are those of the run whose outputs
# are embedded below (library versions are printed in Section 0.2). All random numbers come
# from a fixed seed. A re-run with other library versions can change individual values
# slightly; the text is not updated automatically. Every analysis also saves a summary
# dictionary as a JSON file in the work directory. These files hold more values than the
# text quotes and serve as a record of the run.

# %% [markdown]
# ## 0. Setup and data
#
# ### 0.1 Packages
#
# The next cell installs what a fresh Colab runtime lacks, before any of these packages is
# imported. scikit-learn 1.8 or newer is required because the QDA model uses
# `QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.5)`. Packages that the runtime may
# already have imported (numpy, pandas, matplotlib, ...) are kept at their installed versions.
# If pip nevertheless had to replace a package that is already loaded, the cell stops and
# asks for a restart of the runtime. If everything is already installed, it does nothing.

# %%
import importlib.metadata as _metadata
import os
import re
import subprocess
import sys
import tempfile

# (pip name, minimum (major, minor) version or None)
REQUIREMENTS = [("scikit-learn", (1, 8)), ("numpy", None), ("scipy", None), ("pandas", None),
                ("matplotlib", None), ("joblib", None), ("pillow", None), ("nibabel", None),
                ("nilearn", None), ("gdown", None)]
# Installed packages that pip must not replace: the running kernel (or Colab itself) may
# already have imported them, and a new version on disk under an old one in memory breaks.
FROZEN = ["numpy", "scipy", "pandas", "matplotlib", "joblib", "pillow", "nibabel",
          "requests", "packaging", "jinja2", "threadpoolctl", "lxml"]


def unmet_requirements():
    """pip requirement strings for packages that are missing or too old."""
    unmet = []
    for name, minimum in REQUIREMENTS:
        try:
            have = tuple(int(x) for x in re.findall(r"\d+", _metadata.version(name))[:2])
        except _metadata.PackageNotFoundError:
            have = None
        if have is None or (minimum is not None and have < minimum):
            unmet.append(name if minimum is None else f"{name}>={minimum[0]}.{minimum[1]}")
    return unmet


def installed_versions():
    """{normalised distribution name: version} of everything that is installed."""
    return {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
            for d in _metadata.distributions() if d.metadata["Name"]}


def pip_install(requirements, constraints=None):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *requirements]
    if constraints:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(constraints))
        cmd += ["-c", fh.name]
    return subprocess.run(cmd, capture_output=True, text=True)


_todo = unmet_requirements()
if _todo:
    print("Installing:", ", ".join(_todo))
    _before = installed_versions()
    _pip = pip_install(_todo, [f"{n}=={_before[n]}" for n in FROZEN if n in _before])
    if _pip.returncode != 0:                 # no solution with the frozen versions
        _pip = pip_install(_todo)
    if _pip.returncode != 0 or unmet_requirements():
        raise RuntimeError("pip install failed:\n" + _pip.stdout + _pip.stderr)
    _after = installed_versions()
    _changed = {n for n in _before if _after.get(n) != _before[n]}
    _stale = sorted(mod for mod, dists in _metadata.packages_distributions().items()
                    if mod in sys.modules
                    and _changed & {re.sub(r"[-_.]+", "-", d).lower() for d in dists})
    if _stale:
        raise RuntimeError(
            "pip replaced packages that this runtime had already imported ("
            + ", ".join(_stale) + "). Restart the runtime (Colab: Runtime > Restart session) "
            "and run all cells again; this cell will then find everything installed.")
    print("Done.")
else:
    print("All required packages are already installed.")

# %% [markdown]
# ### 0.2 Imports and constants

# %%
import gc
import glob
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import HTML, Image, display
from joblib import Parallel, delayed
from nilearn import plotting
from PIL import Image as PILImage
from scipy.io import loadmat, wavfile
from scipy.linalg import orthogonal_procrustes
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import gamma, ttest_1samp, wilcoxon
from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

TR = 1.5                  # seconds
N_SUB = 18                # subjects
N_TR = 280                # time points per subject
N_SEG = 14                # story segments = decoding classes of Task A
SEG_LEN = 20              # TRs per segment (30 s)
GRID = (61, 73, 61)       # voxel grid of the 3-mm MNI template
SIGNAL_THRESHOLD = 6000   # mean-signal threshold of the course helper scripts
CH14, CH7 = 100 / 14, 100 / 7     # chance levels (%) of the 14-class and 7-class problems
MINUS = "\u2212"          # typographic minus sign for tables

plt.rcParams["font.family"] = "DejaVu Sans"    # ships with matplotlib: same figures everywhere

T_START = time.time()
TIMINGS = {}              # run time of the heavy cells, in seconds

print(f"Python {sys.version.split()[0]}, {os.cpu_count()} CPU cores")
for _mod in (np, scipy, sklearn, pd, matplotlib, nib, nilearn):
    print(f"  {_mod.__name__:12s} {_mod.__version__}")

# %% [markdown]
# ### 0.3 Configuration: where are the data?
#
# The notebook needs the 18 subject files of the folder
# `story_selected_reg_wm_csf_hsd_last_half2`, the ROI file `group2_all_52rois.voi`, the
# template `MNI152_T1_3mm_brain.nii` and the audio file `Story_Original_MRI.wav`. Two folder
# layouts are accepted: the layout of the course folder on Google Drive (subject files in the
# sub-folder, the other files at the top level) and a flat folder that holds all files.
#
# With `DATA_DIR = None` the data are located automatically, in this order:
#
# 1. the folder named in the environment variable `HW1_DATA_DIR`;
# 2. a local folder that already holds the files (`./data`, `./Data_for_fMRI_course`,
#    `./hw1_data`);
# 3. on Colab: Google Drive is mounted and the course folder
#    `MyDrive/Data_for_fMRI_course` is used (the path of the course example notebooks);
# 4. otherwise the files are downloaded from the course Drive folder with `gdown` into
#    `./hw1_data`. The `avg` sub-folders and the `..._half1` folder are not needed and are
#    skipped.
#
# Results, figures and a cache of the ROI data are written to the work directory.

# %%
# ------------------------------------------------------------------ CONFIGURATION
DATA_DIR = None           # e.g. "/content/drive/MyDrive/Data_for_fMRI_course"; None = automatic
WORK_DIR = None           # None = environment variable HW1_WORK_DIR or ./hw1_work
TRY_GOOGLE_DRIVE = True   # on Colab: mount Drive and look for the course folder first
DRIVE_DIR = "/content/drive/MyDrive/Data_for_fMRI_course"
GDRIVE_FOLDER_ID = "1nM5OJb3WSP1FSa3Ch1dioW2-Roff5CSH"     # course data folder
N_JOBS = -1               # worker processes for the decoding analyses (-1 = all CPU cores)
# ---------------------------------------------------------------------------------

SUBJECT_SUBDIR = "story_selected_reg_wm_csf_hsd_last_half2"
TOP_LEVEL_FILES = ["group2_all_52rois.voi", "MNI152_T1_3mm_brain.nii",
                   "Story_Original_MRI.wav"]


def subject_dir(root):
    """Folder with the subject files: the Drive sub-folder if it exists, else `root`."""
    sub = Path(root) / SUBJECT_SUBDIR
    return sub if sub.is_dir() else Path(root)


def list_subject_files(root):
    """Sorted list of the subject .mat files (names like KM_060412.mat) below `root`."""
    return sorted(glob.glob(str(subject_dir(root) / "*_*.mat")))


def data_complete(root):
    """True if `root` holds the three top-level files and exactly 18 subject files."""
    root = Path(root)
    return (root.is_dir() and all((root / f).is_file() for f in TOP_LEVEL_FILES)
            and len(list_subject_files(root)) == N_SUB)


def download_course_data(target):
    """Download the files used here from the course Drive folder (Drive layout is kept)."""
    import gdown
    target = Path(target)
    hint = (f"put the data into {DRIVE_DIR} on Google Drive, or set DATA_DIR in the "
            "configuration above to a folder that holds the files")
    try:
        listing = gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(target), quiet=True,
                                        skip_download=True)
    except Exception as exc:
        raise FileNotFoundError(f"could not list the course Drive folder ({exc}); {hint}")
    if not listing:
        raise FileNotFoundError(f"could not list the course Drive folder; {hint}")
    for item in listing:
        rel = Path(item.path)
        top_level = len(rel.parts) == 1 and rel.name in TOP_LEVEL_FILES
        subject = rel.parent == Path(SUBJECT_SUBDIR) and rel.suffix == ".mat"
        if (top_level or subject) and not (target / rel).exists():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            try:
                done = gdown.download(id=item.id, output=str(target / rel), quiet=True)
            except Exception as exc:
                done = None
                print(f"download of {rel} failed: {exc}")
            if not done:
                raise FileNotFoundError(f"could not download {rel}; {hint}")


def locate_data():
    """Return (data folder, description of how it was found); see the list above."""
    for folder, how in ((DATA_DIR, "DATA_DIR"),
                        (os.environ.get("HW1_DATA_DIR"), "environment variable HW1_DATA_DIR")):
        if folder:
            if not data_complete(folder):
                raise FileNotFoundError(f"{how} = {folder!r} does not hold the course data")
            return Path(folder).resolve(), how
    for folder in ("data", "Data_for_fMRI_course", "hw1_data"):
        if data_complete(folder):
            return Path(folder).resolve(), "local folder"
    try:
        from google.colab import drive
    except ImportError:
        drive = None
    if drive is not None and TRY_GOOGLE_DRIVE:
        try:
            drive.mount("/content/drive")
        except Exception as exc:                 # no authorisation, no Drive, ...
            print("Google Drive could not be mounted:", exc)
        if data_complete(DRIVE_DIR):
            return Path(DRIVE_DIR), "Google Drive"
        print(f"{DRIVE_DIR} not found or incomplete; downloading the data instead.")
    print("Downloading the course data (about 1.2 GB) ...")
    download_course_data("hw1_data")
    if not data_complete("hw1_data"):
        raise FileNotFoundError("download incomplete; set DATA_DIR to a folder with the data")
    return Path("hw1_data").resolve(), "downloaded with gdown"


DATA, _how = locate_data()
WORK = Path(WORK_DIR or os.environ.get("HW1_WORK_DIR") or "hw1_work").resolve()
FIG_DIR = WORK / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_FILES = list_subject_files(DATA)
SUBJECTS = [Path(f).stem for f in SUBJECT_FILES]
VOI_FILE = DATA / "group2_all_52rois.voi"
TEMPLATE = DATA / "MNI152_T1_3mm_brain.nii"
AUDIO = DATA / "Story_Original_MRI.wav"

print(f"Data folder   : {DATA}  ({_how})")
print(f"Subject files : {len(SUBJECT_FILES)} in {subject_dir(DATA)}")
print(f"Work directory: {WORK}")

# %% [markdown]
# ### 0.4 Loading, ROI extraction and display helpers
#
# - `load_subject` returns one subject as a 280 × 271,633 (TRs × voxels) array.
# - `load_voi` parses the BrainVoyager `.voi` text file; `voi_to_flat_index` converts the mm
#   coordinates of an ROI to voxel indices with the inverse template affine and
#   Fortran-order flattening, the convention of the course example code, so that the indices
#   match the rows of `data_crop`.
# - `load_roi_data` reads the subjects one at a time, keeps the ROI voxels only, and removes
#   voxels that are not valid in all 18 subjects. The result (about 170 MB) is cached in the
#   work directory, so that a second run of the notebook skips this step.
# - `zscore` uses the population standard deviation; a constant series is mapped to 0.

# %%
def load_subject(path):
    """One subject's data as a (280 TRs, 271,633 voxels) float32 array."""
    return np.ascontiguousarray(loadmat(path)["data_crop"].T, dtype=np.float32)


def load_voi(path):
    """Parse a BrainVoyager .voi text file -> list of (name, (n_vox, 3) mm coordinates)."""
    text = Path(path).read_text(errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    vois, i = [], 0
    while i < len(lines):
        if lines[i].startswith("NameOfVOI"):
            name = lines[i].split(":", 1)[1].strip()
            while not lines[i].startswith("NrOfVoxels"):
                i += 1
            n = int(re.findall(r"-?\d+", lines[i])[0])
            coords = np.array([lines[i + 1 + k].split()[:3] for k in range(n)], dtype=int)
            vois.append((name, coords))
            i += n
        i += 1
    return vois


TEMPLATE_IMG = nib.load(str(TEMPLATE))
VOIS = load_voi(VOI_FILE)
# several VOI names occur more than once in the file, so labels carry the running index
LABELS = [f"{k + 1:02d} {name}" for k, (name, _) in enumerate(VOIS)]


def voi_to_flat_index(coords):
    """mm coordinates -> unique flat (Fortran-order) voxel indices on the 61 x 73 x 61 grid."""
    inv = np.linalg.inv(TEMPLATE_IMG.affine)
    ijk = np.round(np.c_[coords, np.ones(len(coords))] @ inv.T)[:, :3].astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(GRID)), axis=1)
    return np.unique(np.ravel_multi_index(ijk[inside].T, GRID, order="F"))


def zscore(x, axis=0):
    """Z-score along `axis` (population SD); constant series map to 0."""
    x = x - x.mean(axis, keepdims=True)
    s = x.std(axis, keepdims=True)
    return x / np.where(s == 0, 1, s)


def load_roi_data():
    """Per-ROI data of all subjects (cached in the work directory).

    Returns a list of 52 float32 arrays (18 subjects, 280 TRs, V voxels) with the raw signal.
    Only voxels that are valid in ALL 18 subjects are kept (finite, mean signal above
    SIGNAL_THRESHOLD, non-constant), so that every subject has identical features.
    """
    cache = WORK / "roi_data.npz"
    if cache.exists():
        with np.load(cache) as z:
            return [z[f"roi{k}"] for k in range(len(VOIS))]
    idx = [voi_to_flat_index(c) for _, c in VOIS]
    union = np.unique(np.concatenate(idx))
    sub = np.stack([load_subject(f)[:, union] for f in SUBJECT_FILES])     # (18, 280, U)
    good = (np.isfinite(sub).all(1) & (sub.mean(1) > SIGNAL_THRESHOLD)
            & (sub.std(1) > 0)).all(0)
    out = {}
    for k, ix in enumerate(idx):
        cols = np.searchsorted(union, ix)
        # C-contiguous copy: the memory layout decides the order of float32 summation in the
        # z-scoring, and arrays read back from the cache are C-contiguous.  With this copy
        # the results are identical with and without the cache.
        out[f"roi{k}"] = np.ascontiguousarray(sub[:, :, cols[good[cols]]])
    np.savez(cache, **out)
    return [out[f"roi{k}"] for k in range(len(VOIS))]


def sem(a, axis=0):
    """Standard error of the mean (sample SD / sqrt(n)) along `axis`."""
    return a.std(axis, ddof=1) / np.sqrt(a.shape[axis])


def save_json(obj, name):
    """Write a result dictionary to the work directory."""
    with open(WORK / name, "w") as fh:
        json.dump(obj, fh, indent=1)


def signed(x, spec=".1f"):
    """Number with a typographic minus sign (a value that rounds to zero has no sign)."""
    txt = format(x, spec)
    if float(txt) == 0:
        txt = format(0.0, spec)
    return txt.replace("-", MINUS)


def show_table(df):
    """Display a DataFrame as a static HTML table (cells may contain HTML such as <b>)."""
    with pd.option_context("display.max_colwidth", None):       # never truncate a cell
        display(HTML(df.to_html(index=False, escape=False, notebook=True, border=0)))


def show_figure(fig, name, dpi, width):
    """Save a figure as PNG in the work directory, close it and display the saved file."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    display(Image(filename=str(path), width=width))

# %%
_t = time.time()
ROI_X = load_roi_data()
N_VOX = np.array([x.shape[2] for x in ROI_X])
N_MAPPED = np.array([len(voi_to_flat_index(c)) for _, c in VOIS])
TIMINGS["0.4 ROI extraction"] = time.time() - _t
save_json({"labels": LABELS, "n_declared": [len(c) for _, c in VOIS],
           "n_mapped": N_MAPPED.tolist(), "n_kept": N_VOX.tolist()}, "roi_info.json")

_lost = [LABELS[k].split()[0] for k in range(len(VOIS)) if N_MAPPED[k] != N_VOX[k]]
print(f"{len(VOIS)} ROIs, {N_MAPPED.sum():,} voxels on the template grid")
print(f"removed (not valid in all {N_SUB} subjects): {N_MAPPED.sum() - N_VOX.sum()} voxels, "
      f"in ROIs {', '.join(_lost)}")
print(f"ROI size after cleaning: {N_VOX.min()} to {N_VOX.max()} voxels "
      f"(median {np.median(N_VOX):.0f})")
print(f"array of the largest ROI: {ROI_X[int(N_VOX.argmax())].shape} "
      f"(subjects, TRs, voxels), {ROI_X[0].dtype}")
print(f"time: {TIMINGS['0.4 ROI extraction']:.0f} s")

# %% [prose] methods.h2 methods.p

# %% [markdown]
# ### 1.1 Implementation: classifiers and LOSO cross-validation
#
# The next cell defines the seven classifiers and the cross-validation used in Tasks A and B.
#
# - `segment_samples` reshapes the z-scored data (18 × 280 × V) into the group tensor
#   14 × 20 × V × 18 required by the assignment.
# - `loso_accuracy` flattens every 20 × V segment into one feature vector and runs the 18
#   leave-one-subject-out folds; `fit_score_all` trains and tests all classifiers on one fold.
# - `GramLinearSVC` is libsvm's linear SVC fed with the precomputed Gram matrix X Xᵀ.
# - `ShrinkageLDA` is the LDA described above (pooled covariance, Ledoit–Wolf shrinkage in
#   the PCA subspace of the training fold).
# - `print_by_classifier` only formats printed summaries.

# %%
CLF_NAMES = ["KNN (K=1)", "KNN (K=2)", "KNN (K=3)", "Linear SVM", "Naive Bayes", "LDA", "QDA"]


class GramLinearSVC:
    """Linear-kernel SVC (libsvm, one-vs-one) fed with a precomputed Gram matrix X X^T.

    Mathematically identical to SVC(kernel="linear") but much faster for wide fMRI data
    (features >> samples), because the kernel is computed once with BLAS.
    """

    def __init__(self, C=1.0):
        self.svc = SVC(kernel="precomputed", C=C, decision_function_shape="ovo")

    def fit(self, X, y):
        self.Xtr = np.asarray(X, dtype=np.float64)
        self.svc.fit(self.Xtr @ self.Xtr.T, y)
        return self

    def score(self, X, y):
        return self.svc.score(np.asarray(X, dtype=np.float64) @ self.Xtr.T, y)


class ShrinkageLDA:
    """LDA with a pooled within-class covariance shrunk towards a scaled identity
    (Ledoit-Wolf), estimated in the subspace spanned by the training samples.

    The training fold is rotated with a full PCA (q <= n_train components, float64).  The
    pooled within-class covariance S of the PCA scores is replaced by
    (1 - a) S + a (tr S / q) I, with the Ledoit-Wolf intensity a computed from the
    q-dimensional scores.  Both a and the target scale tr S / q therefore refer to the
    training subspace, not to the p = 20 V voxel space.  Features are not rescaled.

    Nothing is lost by working in the subspace: the class means lie in the training span and
    the shrinkage target is isotropic, so components of a test sample outside the span add
    the same constant to every class score.  Equal class priors (the design is balanced).
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.pca = PCA(svd_solver="full").fit(X)
        Z = self.pca.transform(X)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([Z[y == c].mean(0) for c in self.classes_])
        resid = Z - self.means_[np.searchsorted(self.classes_, y)]
        lw = LedoitWolf(assume_centered=True).fit(resid)
        self.shrinkage_ = float(lw.shrinkage_)       # LW intensity in the PCA subspace
        self.coef_ = np.linalg.solve(lw.covariance_, self.means_.T).T    # (classes, dims)
        self.intercept_ = -0.5 * np.einsum("cd,cd->c", self.means_, self.coef_)
        return self

    def predict(self, X):
        Z = self.pca.transform(np.asarray(X, dtype=np.float64))
        return self.classes_[np.argmax(Z @ self.coef_.T + self.intercept_, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

# %%
def make_classifiers():
    """The seven models, in CLF_NAMES order.  All hyper-parameters are fixed a priori.

    KNN          Euclidean distance, K = 1, 2, 3, uniform votes; a tied vote goes to the
                 lower class index (scikit-learn's default).
    Linear SVM   libsvm SVC, linear kernel, C = 1, one-vs-one.
    Naive Bayes  Gaussian; every (voxel, time point) feature is treated as independent.
    LDA          ShrinkageLDA (above).
    QDA          PCA rotation fitted on the training fold (all components), then one
                 covariance per class, shrunk half-way to a scaled identity (shrinkage 0.5).
    """
    qda = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.5)
    return {
        "KNN (K=1)": KNeighborsClassifier(1, algorithm="brute"),
        "KNN (K=2)": KNeighborsClassifier(2, algorithm="brute"),
        "KNN (K=3)": KNeighborsClassifier(3, algorithm="brute"),
        "Linear SVM": GramLinearSVC(C=1.0),
        "Naive Bayes": GaussianNB(),
        "LDA": ShrinkageLDA(),
        "QDA": make_pipeline(PCA(svd_solver="full"), qda),
    }


def fit_score_all(Xtr, ytr, Xte, yte):
    """Train every classifier on (Xtr, ytr); return the test accuracies (CLF_NAMES order)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # keep worker processes silent
        return np.array([clf.fit(Xtr, ytr).score(Xte, yte)
                         for clf in make_classifiers().values()])


def segment_samples(z, seg_len=SEG_LEN):
    """(n_subjects, T, V) z-scored data -> group tensor (n_seg, seg_len, V, n_subjects)."""
    n_sub, T, V = z.shape
    return np.transpose(z.reshape(n_sub, T // seg_len, seg_len, V), (1, 2, 3, 0))


def loso_accuracy(tensor):
    """LOSO cross-validation on a (n_seg, seg_len, V, n_subjects) group tensor.

    Each fold trains on (n_subjects - 1) x n_seg segment samples and predicts the segment
    identity of the n_seg samples of the held-out subject.
    Returns (n_subjects, n_classifiers) accuracies.
    """
    n_seg, seg_len, V, n_sub = tensor.shape
    # (subject, sample, feature): every 20 x V segment becomes one feature vector
    X = np.transpose(tensor, (3, 0, 1, 2)).reshape(n_sub, n_seg, seg_len * V)
    y = np.arange(n_seg)
    acc = np.zeros((n_sub, len(CLF_NAMES)))
    for s in range(n_sub):
        train = np.delete(np.arange(n_sub), s)
        acc[s] = fit_score_all(X[train].reshape(-1, X.shape[2]), np.tile(y, n_sub - 1),
                               X[s], y)
    return acc


def print_by_classifier(rows):
    """Print one line of values per label, in columns named after the classifiers."""
    print(" " * 13 + "".join(f"{c:>12s}" for c in CLF_NAMES))
    for label, values in rows.items():
        print(f"{label:13s}" + "".join(f"{v:12.1f}" for v in values))

# %%
# shape check on the largest ROI: 14 segments x 20 TRs x V voxels x 18 subjects
_k = int(N_VOX.argmax())
print(f"group tensor of ROI {LABELS[_k]}:",
      segment_samples(zscore(ROI_X[_k], axis=1)).shape)

# %% [prose] task0.h2 task0.p

# %% [markdown]
# ### 2.1 Code
#
# - `audio_envelope_tr`: envelope of the whole recording in consecutive 1.5-s bins.
# - `spm_hrf`: canonical double-gamma HRF (SPM default parameters) sampled every TR.
# - `xcorr`: normalised cross-correlation of the z-scored signals,
#   r[lag] = (1/N) Σ env[t] · bold[t + lag], so that a positive lag means that BOLD follows
#   the sound.
# - `phase_randomise` and `circ_shift` produce the surrogate envelopes of the two null
#   distributions. For every surrogate the test statistic is the maximum over all lags.

# %%
MAX_LAG = 10          # TRs
N_SURR = 2000         # phase-randomised surrogates (the circular-shift null is exhaustive)
SCAN_START_TR = 9     # scan TR 0 = 13.5 s into the .wav, the crop of the course example code


def audio_envelope_tr(path):
    """Envelope of the whole .wav in consecutive 1.5-s bins; also returns its length in s."""
    fs, y = wavfile.read(path)
    y = y.astype(float).mean(axis=1)                              # stereo -> mono
    n_fft = 1 << int(np.ceil(np.log2(len(y))))
    env = np.abs(hilbert(y, N=n_fft))[: len(y)]
    env = sosfiltfilt(butter(4, 8, fs=fs, output="sos"), env)     # 8-Hz low-pass
    n = int(round(fs * TR))
    nb = len(env) // n
    return env[: nb * n].reshape(nb, n).mean(axis=1), len(y) / fs


def spm_hrf(tr, duration=32.0):
    """Canonical double-gamma HRF (SPM defaults) sampled every `tr` seconds."""
    t = np.arange(0, duration, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def xcorr(env, bold, max_lag=MAX_LAG):
    """MATLAB-style xcorr(..., 'coeff') of z-scored signals.

    r[lag] = (1/N) sum_t env[t] * bold[t + lag]; a POSITIVE lag means BOLD follows the sound.
    `env` may be (n_surrogates, N); `bold` is (N,).  Returns lags and r (..., 2*max_lag+1).
    """
    env = np.atleast_2d(zscore(env, axis=-1))
    bold = zscore(bold)
    n = len(bold)
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.stack([(env[:, : n - L] * bold[L:]).sum(1) if L >= 0
                  else (env[:, -L:] * bold[: n + L]).sum(1) for L in lags], axis=1) / n
    return lags, r.squeeze()


def phase_randomise(x, n, rng):
    """n surrogates with the power spectrum of x but random Fourier phases."""
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, (n, len(f)))
    ph[:, 0] = 0
    if len(x) % 2 == 0:
        ph[:, -1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(x), axis=1)


def circ_shift(x):
    """All admissible circular shifts of x: every shift that is at least 2*MAX_LAG TRs away
    from zero in either direction (280 - 40 = 240 surrogates).  The null is enumerated
    completely, so the smallest attainable p-value is 1/241."""
    return np.stack([np.roll(x, s) for s in range(2 * MAX_LAG, len(x) - 2 * MAX_LAG)])


def peak(lags, r):
    """Lag and value of the maximum of a cross-correlation curve."""
    k = int(np.argmax(r))
    return int(lags[k]), float(r[k])

# %% [markdown]
# ### 2.2 Analysis
#
# As the assignment asks, the A1 time course of a subject is the mean of the signal over the
# voxels of ROIs 48 A1_R and 52 A1_L, z-scored; the group time course is the z-scored mean
# over subjects. The cell runs the cross-correlation for the raw and for the HRF-convolved
# envelope, both null distributions, each hemisphere separately, two checks of the position
# of the scan inside the recording, and one check of the averaging convention (the course
# example notebook z-scores every voxel before averaging). Most of the run time is the Hilbert transform of the recording.

# %%
_t = time.time()
rng = np.random.default_rng(0)

# --- A1 BOLD: per subject and group average
ia = [k for k, lab in enumerate(LABELS) if lab.split()[1] in ("A1_L", "A1_R")]
a1 = np.concatenate([ROI_X[k] for k in ia], axis=2)                 # (18, 280, V_A1)
subj_bold = zscore(a1.mean(axis=2), axis=1)       # (18, 280): mean over voxels, then z-score
group_bold = zscore(subj_bold.mean(axis=0))

# --- audio envelope, cropped to the scan window; raw and convolved with the HRF
env_full, audio_s = audio_envelope_tr(AUDIO)
onset = SCAN_START_TR
env = zscore(env_full[onset: onset + N_TR])
conv_full = np.convolve(env_full, spm_hrf(TR))[: len(env_full)]
env_hrf = zscore(conv_full[onset: onset + N_TR])

t0 = {"n_voxels_A1": int(a1.shape[2]), "scan_start_TR": onset, "audio_s": audio_s,
      "n_tr_bins_audio": int(len(env_full)), "max_lag_TR": MAX_LAG, "subjects": SUBJECTS}

# --- each hemisphere separately (raw envelope)
t0["hemispheres"] = {}
for k in ia:
    hemi_bold = zscore(zscore(ROI_X[k].mean(axis=2), axis=1).mean(axis=0))
    lags_h, r_h = xcorr(env, hemi_bold)
    t0["hemispheres"][LABELS[k]] = {"peak_lag_TR": peak(lags_h, r_h)[0],
                                    "peak_r": peak(lags_h, r_h)[1],
                                    "r_by_lag": r_h.tolist()}

# --- cross-correlation and surrogate tests, raw and HRF-convolved envelope
curves = {}
for name, e in (("raw", env), ("hrf", env_hrf)):
    lags, rg = xcorr(e, group_bold)
    rs = np.stack([xcorr(e, s)[1] for s in subj_bold])
    pk_g = peak(lags, rg)
    pk_s = [peak(lags, r) for r in rs]
    out = {"group_peak_lag_TR": pk_g[0], "group_peak_lag_s": pk_g[0] * TR,
           "group_peak_r": pk_g[1], "group_r_at_lag0": float(rg[MAX_LAG]),
           "lags_TR": lags.tolist(), "group_r_by_lag": rg.tolist(),
           "subj_peak_lag_TR": [p[0] for p in pk_s], "subj_peak_r": [p[1] for p in pk_s]}
    # null distributions of the peak (maximum over lags) correlation
    for null in ("phase", "circ"):
        surr = phase_randomise(e, N_SURR, rng) if null == "phase" else circ_shift(e)
        n_surr = len(surr)                           # 2000 (phase) or 240 (all shifts)
        out[f"n_surr_{null}"] = n_surr
        null_g = xcorr(surr, group_bold)[1].max(axis=1)
        out[f"group_n_exceed_{null}"] = int((null_g >= pk_g[1]).sum())
        out[f"group_p_{null}"] = float((1 + (null_g >= pk_g[1]).sum()) / (1 + n_surr))
        out[f"null95_{null}"] = float(np.quantile(null_g, 0.95))
        null_s = np.stack([xcorr(surr, s)[1].max(axis=1) for s in subj_bold])
        exceed = (null_s >= np.array([p[1] for p in pk_s])[:, None]).sum(1)
        out[f"subj_p_{null}"] = ((1 + exceed) / (1 + n_surr)).tolist()
        if name == "raw" and null == "phase":
            null_plot = null_g
    t0[name] = out
    curves[name] = (lags, rg, rs)
    print(f"[{name}] group peak: lag {pk_g[0]} TR ({pk_g[0] * TR:.1f} s), "
          f"r = {pk_g[1]:.3f}, p(phase) = {out['group_p_phase']:.4f}, "
          f"p(circular shift) = {out['group_p_circ']:.4f}")
    print(f"      subjects: peak lag {np.mean(out['subj_peak_lag_TR']):.2f} TR, "
          f"peak r {np.mean(out['subj_peak_r']):.3f}, significant (p < .05, phase): "
          f"{int((np.array(out['subj_p_phase']) < .05).sum())}/{N_SUB}")

# %%
# --- where does the scan start inside the .wav?  Slide the 280-TR window over the
#     HRF-convolved envelope of the whole file and correlate at zero lag.  (Not an independent
#     check of the crop: it assumes the canonical HRF delay.)
offs = np.arange(0, len(env_full) - N_TR + 1)
r_off = [float(np.corrcoef(conv_full[o: o + N_TR], group_bold)[0, 1]) for o in offs]
t0["window_offset_scan"] = {"offset_TR": offs.tolist(), "r": r_off,
                            "best_offset_TR": int(offs[np.argmax(r_off)])}
print("best scan-start offset in the .wav (HRF model, zero lag):",
      t0["window_offset_scan"]["best_offset_TR"], "TR")

# --- the same raw-envelope analysis if the scan had started one TR later (15 s, story onset)
alt = SCAN_START_TR + 1
lags_a, r_a = xcorr(zscore(env_full[alt: alt + N_TR]), group_bold)
t0["alt_scan_start"] = {"scan_start_TR": alt, "raw_group_peak_lag_TR": peak(lags_a, r_a)[0],
                        "raw_group_peak_r": peak(lags_a, r_a)[1]}
print(f"crop at {alt} TRs instead: raw peak lag {peak(lags_a, r_a)[0]} TR, "
      f"r = {peak(lags_a, r_a)[1]:.3f}")

# --- check: z-score every voxel before averaging, as in the course example notebook (equal
#     weight per voxel instead of weight proportional to the voxel's signal amplitude)
subj_vz = zscore(zscore(a1, axis=1).mean(axis=2), axis=1)
lags_v, r_v = xcorr(env, zscore(subj_vz.mean(axis=0)))
t0["voxel_zscore_first"] = {
    "raw_group_peak_lag_TR": peak(lags_v, r_v)[0], "raw_group_peak_r": peak(lags_v, r_v)[1],
    "min_subj_corr_with_primary": float(min(np.corrcoef(x, y)[0, 1]
                                            for x, y in zip(subj_bold, subj_vz))),
    "subj_peak_lag_TR": [peak(*xcorr(env, s))[0] for s in subj_vz]}
print(f"voxel-wise z-score before averaging instead: raw peak lag {peak(lags_v, r_v)[0]} TR, "
      f"r = {peak(lags_v, r_v)[1]:.3f}; correlation of the subject time courses with the "
      f"primary ones >= {t0['voxel_zscore_first']['min_subj_corr_with_primary']:.3f}")

# --- numbers quoted in the discussion below
_rg = dict(zip(t0["raw"]["lags_TR"], t0["raw"]["group_r_by_lag"]))
print("raw envelope, group r at lags 0..5 TRs: "
      + ", ".join(f"{lag}: {_rg[lag]:+.2f}" for lag in range(6)))
for _lab, _h in t0["hemispheres"].items():
    print(f"{_lab} alone: peak lag {_h['peak_lag_TR']} TR, r = {_h['peak_r']:.2f}")
_b = t0["window_offset_scan"]["best_offset_TR"]
print("zero-lag r of the HRF model at scan-start offsets "
      + ", ".join(f"{o} TR: {r_off[o]:.2f}" for o in (_b - 1, _b, _b + 1)))

save_json(t0, "task0_alignment.json")
TIMINGS["2 Task 0"] = time.time() - _t
print(f"A1: ROIs {[LABELS[k] for k in ia]}, {a1.shape[2]} voxels; audio {audio_s:.1f} s, "
      f"{len(env_full)} TR bins; time: {TIMINGS['2 Task 0']:.0f} s")

# %% [markdown]
# ### 2.3 Results

# %%
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
for a, name, ttl in ((ax[0], "raw", "Raw envelope vs. A1"),
                     (ax[1], "hrf", "HRF-convolved envelope vs. A1")):
    lags, rg, rs = curves[name]
    a.plot(lags, rs.T, color="0.75", lw=0.7)
    a.plot([], [], color="0.75", lw=0.7, label="single subjects (n = 18)")
    a.plot(lags, rg, color="C3", lw=2.2, marker="o", ms=3.5, label="group-average A1")
    a.axhline(t0[name]["null95_phase"], color="k", ls="--", lw=0.8,
              label="95th pct. of null peak (group)")
    a.axvline(0, color="k", lw=0.5)
    pk = t0[name]["group_peak_lag_TR"]
    where = "lag 0" if pk == 0 else f"lag {pk:+d} TR ({pk * TR:+.1f} s)".replace("-", MINUS)
    a.set_title(f"{ttl}\npeak: {where}, r = {t0[name]['group_peak_r']:.2f}", fontsize=10)
    a.set_xlabel("Lag (TRs; positive = BOLD follows sound)")
    a.set_ylabel("Cross-correlation (r)")
    a.set_xticks(lags[::2])
    sec = a.secondary_xaxis("top", functions=(lambda x: x * TR, lambda x: x / TR))
    sec.set_xlabel("Lag (s)", fontsize=8)
ax[2].hist(null_plot, bins=40, color="0.7")
ax[2].axvline(t0["raw"]["group_peak_r"], color="C3", lw=2)
ax[2].set_title(f"Null of peak r, {N_SURR} phase-randomised envelopes\nobserved r = "
                f"{t0['raw']['group_peak_r']:.2f}, p = {t0['raw']['group_p_phase']:.4f}",
                fontsize=10)
ax[2].set_xlabel(f"Peak cross-correlation over lags {MINUS}10 to +10 TRs")
ax[2].set_ylabel("Surrogates")
handles, legend_labels = ax[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=[0, 0.07, 1, 1])
show_figure(fig, "task0_xcorr.png", dpi=200, width=1000)

# %% [prose] task0.cap[0]

# %%
t_sec = np.arange(N_TR) * TR
fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
ax[0].plot(t_sec, env, color="0.45", lw=1, label="audio envelope (TR-binned, z)")
ax[0].plot(t_sec, group_bold, color="C3", lw=1.3, label="group A1 BOLD (z)")
ax[0].set_title("A1 BOLD and audio envelope (envelope cropped to the scan window, no shift)")
ax[1].plot(t_sec, env_hrf, color="0.45", lw=1, label="envelope * canonical HRF (z)")
ax[1].plot(t_sec, group_bold, color="C3", lw=1.3, label="group A1 BOLD (z)")
ax[1].set_title("After convolving the envelope with the canonical HRF "
                f"(r = {t0['hrf']['group_r_at_lag0']:.2f} at lag 0)")
ax[1].set_xlabel("Time from scan start (s)")
for a in ax:
    a.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    a.set_ylabel("z")
    a.set_xlim(0, N_TR * TR)
fig.tight_layout()
show_figure(fig, "task0_timecourses.png", dpi=200, width=780)

# %% [prose] task0.cap[1]

# %%
def pval(p):
    """p-value with three decimals (four below 0.001)."""
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


raw = t0["raw"]
_cells = [[t0["subjects"][i], f"+{raw['subj_peak_lag_TR'][i]}",
           f"{raw['subj_peak_lag_TR'][i] * TR:.1f}", f"{raw['subj_peak_r'][i]:.2f}",
           pval(raw["subj_p_phase"][i])] for i in range(N_SUB)]
_per = N_SUB // 3                 # three blocks of six subjects, side by side
show_table(pd.DataFrame(
    [_cells[i] + _cells[i + _per] + _cells[i + 2 * _per] for i in range(_per)],
    columns=["Subject", "Lag (TRs)", "Lag (s)", "Peak r", "p"] * 3))

# %% [prose] subj.cap

# %% [markdown]
# ### 2.4 Discussion

# %% [prose] task0.ul

# %% [prose] h3clf.h2

# %% [markdown]
# ### 3.1 LOSO decoding in all 52 ROIs
#
# For each ROI the voxel time courses are z-scored within subject, the group tensor
# 14 × 20 × V × 18 is built, and the seven classifiers are evaluated with LOSO
# cross-validation (Section 1.1). ROIs are processed in parallel, largest first. The result
# is an array of accuracies with shape (52 ROIs, 18 held-out subjects, 7 classifiers).
# Run time: about 1 minute on 4 cores.

# %%
def run_roi_task_a(x):
    """x: (18, 280, V) raw ROI data -> (18 folds, 7 classifiers) accuracies."""
    tensor = segment_samples(zscore(x, axis=1), SEG_LEN)
    assert tensor.shape == (N_SEG, SEG_LEN, x.shape[2], N_SUB)
    return loso_accuracy(tensor)


_t = time.time()
order = np.argsort(-N_VOX)                         # big ROIs first for better scheduling
_out = Parallel(n_jobs=N_JOBS)(delayed(run_roi_task_a)(ROI_X[k]) for k in order)
ACC_A = np.zeros((len(VOIS), N_SUB, len(CLF_NAMES)))
ACC_A[order] = np.stack(_out)                      # (ROI, held-out subject, classifier)
np.savez(WORK / "part2a_accuracy.npz", acc=ACC_A, clf=CLF_NAMES, labels=LABELS, n_vox=N_VOX)
TIMINGS["3.1 Task A decoding"] = time.time() - _t
print(f"accuracy array {ACC_A.shape}; time: {TIMINGS['3.1 Task A decoding']:.0f} s")
print("mean accuracy over the 52 ROIs (%):")
print_by_classifier({"Task A": ACC_A.mean((0, 1)) * 100})

# %% [markdown]
# ### 3.2 Summary statistics
#
# Mean and SEM over the 18 folds for every ROI and classifier, the ranking of the ROIs by
# their mean over classifiers, a one-sided t-test of the 18 fold accuracies against chance
# for every ROI and classifier (Benjamini–Hochberg FDR over all 52 × 7 tests), and the best
# classifier of every ROI. The best classifier is decided on the integer number of correct
# test samples, so that exact ties are detected; a tie goes to the first-listed classifier.

# %%
def fdr_threshold(p, q=0.05):
    """Largest p-value that is significant under Benjamini-Hochberg FDR (0 if none is)."""
    flat = np.sort(np.ravel(p))
    ok = flat <= q * np.arange(1, flat.size + 1) / flat.size
    return flat[np.nonzero(ok)[0].max()] if ok.any() else 0


def summarise_task_a(acc_frac):
    """Summary dictionary of the Task-A accuracies (fractions, shape (52, 18, 7))."""
    acc = acc_frac * 100
    clf, labels, nvox = CLF_NAMES, LABELS, N_VOX
    m, s = acc.mean(1), sem(acc, 1)                              # (52, 7)
    order = np.argsort(-m.mean(1))
    p = np.array([[ttest_1samp(acc[k, :, j], CH14, alternative="greater").pvalue
                   for j in range(len(clf))] for k in range(len(labels))])
    thr = fdr_threshold(p)
    n_correct = np.round(acc_frac.sum(1) * N_SEG).astype(int)    # 14 test samples per fold
    best = n_correct.argmax(1)
    is_max = n_correct == n_correct.max(1, keepdims=True)
    tied = is_max.sum(1) > 1
    i_best, j_best = np.unravel_index(m.argmax(), m.shape)
    return {
        "clf": clf, "clf_mean": m.mean(0).tolist(), "clf_median": np.median(m, 0).tolist(),
        "n_sig_rois_fdr": (p <= thr).sum(0).tolist(),
        "n_rois_best": [int((best == j).sum()) for j in range(len(clf))],
        "n_rois_best_tied": int(tied.sum()),
        "rois_best_tied": [{"roi": labels[k],
                            "tied": [clf[j] for j in np.flatnonzero(is_max[k])]}
                           for k in np.flatnonzero(tied)],
        "best_median_nvox": [float(np.median(nvox[best == j])) if (best == j).any() else None
                             for j in range(len(clf))],
        "n_big_rois": int((nvox >= 200).sum()),
        "n_big_rois_best": [int(((best == j) & (nvox >= 200)).sum())
                            for j in range(len(clf))],
        "rank_order": [labels[k] for k in order],
        "rank_mean": [float(m[k].mean()) for k in order],
        "top3_by_best_single_clf": [labels[k] for k in np.argsort(-m.max(1))[:3]],
        "top3_by_clf": {c: [labels[k] for k in np.argsort(-m[:, j])[:3]]
                        for j, c in enumerate(clf)},
        "best_single": {"roi": labels[int(i_best)], "clf": clf[int(j_best)],
                        "acc": float(m.max())},
        "n_rois_total": len(labels), "n_vox_min": int(nvox.min()),
        "n_vox_max": int(nvox.max()),
        "top10": [{"roi": labels[k], "n_vox": int(nvox[k]), "mean": float(m[k].mean()),
                   "acc": m[k].tolist(), "sem": s[k].tolist()} for k in order[:10]],
        "bottom5": [{"roi": labels[k], "n_vox": int(nvox[k]), "mean": float(m[k].mean()),
                     "acc": m[k].tolist()} for k in order[-5:]],
        "corr_logV_acc": float(np.corrcoef(np.log(nvox), m.mean(1))[0, 1]),
        "table": [{"roi": labels[k], "n_vox": int(nvox[k]), "acc": m[k].tolist(),
                   "sem": s[k].tolist()} for k in range(len(labels))],
    }


A = summarise_task_a(ACC_A)
save_json(A, "part2a_summary.json")
print("ROIs ranked by mean accuracy over the 7 classifiers (top 6):")
for _lab, _mean in list(zip(A["rank_order"], A["rank_mean"]))[:6]:
    print(f"  {_lab:18s} {_mean:5.1f} %")
print(f"best single result: {A['best_single']['roi']}, {A['best_single']['clf']}, "
      f"{A['best_single']['acc']:.1f} %")
print(f"correlation between log V and mean accuracy: r = {A['corr_logV_acc']:.2f}")
print(f"V of the bottom five ROIs: {[r['n_vox'] for r in A['bottom5']]}")

# %% [markdown]
# ### 3.3 Figures

# %%
COL = ["#9ecae1", "#6baed6", "#3182bd", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

acc_pct = ACC_A * 100
m, s = acc_pct.mean(1), sem(acc_pct, 1)            # (52, 7): mean and SEM over the 18 folds
order = np.argsort(-m.mean(1))

fig, axes = plt.subplots(len(CLF_NAMES), 1, figsize=(13, 14.2), sharex=True)
for j, ax in enumerate(axes):
    ax.bar(range(52), m[order, j], yerr=s[order, j], color=COL[j], ecolor="0.25",
           capsize=1.5, error_kw=dict(lw=0.7))
    ax.axhline(CH14, color="k", ls="--", lw=0.9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 92)
    ax.set_xlim(-0.7, 51.7)
    ax.set_title(f"{CLF_NAMES[j]}   (mean over ROIs {m[:, j].mean():.1f} %, best "
                 f"{LABELS[int(m[:, j].argmax())]} {m[:, j].max():.1f} %)",
                 fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[-1].set_xticks(range(52))
axes[-1].set_xticklabels([f"{LABELS[k]} ({N_VOX[k]})" for k in order], rotation=90,
                         fontsize=9.5)
axes[-1].set_xlabel("ROI (index name, number of voxels), sorted by mean accuracy "
                    "over classifiers")
fig.suptitle("Task A – LOSO 14-way story-segment decoding in anatomical space "
             "(mean ± SEM over 18 folds; dashed line = chance, 7.14 %)",
             y=0.995, fontsize=12)
fig.tight_layout()
show_figure(fig, "part2a_bars_all_rois.png", dpi=170, width=900)

# %% [prose] bars.cap

# %%
fig, ax = plt.subplots(figsize=(13, 3.6))
im = ax.imshow(m[order].T, aspect="auto", cmap="viridis", vmin=0, vmax=80)
ax.set_yticks(range(len(CLF_NAMES)))
ax.set_yticklabels(CLF_NAMES)
ax.set_xticks(range(52))
ax.set_xticklabels([LABELS[k] for k in order], rotation=90, fontsize=9)
fig.colorbar(im, ax=ax, label="Accuracy (%)", pad=0.01)
ax.set_title("Task A – accuracy of every classifier in every ROI", fontsize=11)
fig.tight_layout()
show_figure(fig, "part2a_heatmap.png", dpi=170, width=1000)

# %% [prose] heat.cap

# %% [markdown]
# ### 3.4 Summary tables and leading regions

# %%
show_table(pd.DataFrame({
    "Classifier": CLF_NAMES,
    "Mean over 52 ROIs (%)": [f"{v:.1f}" for v in A["clf_mean"]],
    "Median (%)": [f"{v:.1f}" for v in A["clf_median"]],
    "# ROIs where best": A["n_rois_best"],
    "ROIs above chance (FDR q &lt; .05)": [f"{v}/52" for v in A["n_sig_rois_fdr"]],
}))
print("exact ties for the best classifier:")
for _tie in A["rois_best_tied"]:
    print(f"   {_tie['roi']}: {' = '.join(_tie['tied'])}")
print(f"of the {A['n_big_rois']} ROIs with V >= 200, best classifier: "
      + ", ".join(f"{c} {v}" for c, v in zip(CLF_NAMES, A["n_big_rois_best"]) if v))
print("median V of the ROIs in which a classifier is best: "
      + ", ".join(f"{c} {v:.0f}" for c, v, k in
                  zip(CLF_NAMES, A["best_median_nvox"], A["n_rois_best"]) if k))

# %% [prose] h3clf.cap

# %%
def acc_sem_cells(acc, sems, mark_best=False):
    """'mean ± SEM' strings for one ROI; optionally the best classifier in bold."""
    cells = [f"{a:.1f} ± {e:.1f}" for a, e in zip(acc, sems)]
    if mark_best:
        cells = [f"<b>{c}</b>" if a > max(acc) - 1e-6 else c for c, a in zip(cells, acc)]
    return cells


show_table(pd.DataFrame(
    [[r["roi"], r["n_vox"], *acc_sem_cells(r["acc"], r["sem"], mark_best=True),
      f"{r['mean']:.1f}"] for r in A["top10"]],
    columns=["ROI", "V", *CLF_NAMES, "Mean"]))

# %% [prose] top.cap

# %% [prose] regions.p

# %% [markdown]
# ### 3.5 Supplementary analysis: how much does LDA depend on its covariance regulariser?
#
# Same samples and LOSO folds as above. Three LDA variants that differ only in how the pooled
# within-class covariance is regularised:
#
# - `span_lw`: the main model (`ShrinkageLDA`), Ledoit–Wolf (LW) shrinkage
#   estimated in the PCA subspace of the training fold;
# - `voxel_lw`: exact pooled LW shrinkage estimated in the full 20·V voxel space;
# - `sklearn_auto`: the estimator of scikit-learn's
#   `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` on the raw features (per
#   class: features are standardised, the LW estimate is computed and scaled back; the class
#   estimates are averaged).
#
# The p × p covariance (p = 20·V) is never formed. All three estimators have
# the form Σ = diag(g) + Rᵀ diag(w) R, where R holds the within-class residuals of the
# training samples, so Σ⁻¹ is applied with the Woodbury identity and the LW intensity is
# computed from Gram matrices. `selfcheck_lda` verifies these shortcuts against scikit-learn's
# dense implementations on a small ROI before anything is reported. Nothing here is used to
# choose a model. Run time: about 1 minute on 4 cores.

# %%
VARIANTS = ["span_lw", "voxel_lw", "sklearn_auto"]
WIDE_V = 100      # "wide" ROIs for the summary of the shrinkage intensity: V >= 100


def lw_intensity(X):
    """Ledoit-Wolf shrinkage intensity and target scale mu = tr(S)/p of S = X^T X / n
    (X centred), computed from the n x n Gram matrix."""
    n, p = X.shape
    G = X @ X.T
    tr = np.trace(G) / n
    mu = tr / p
    delta_ = (G ** 2).sum() / n ** 2                       # ||S||_F^2
    beta_ = ((X ** 2).sum(1) ** 2).sum()
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * tr + p * mu ** 2) / p
    beta = min(beta, delta)
    return (0.0 if beta == 0 else float(beta / delta)), float(mu)


def woodbury_predict(M, R, w, g, Xte):
    """LDA prediction (equal priors) for Sigma = diag(g) + R^T diag(w) R, Sigma not formed.

    M (classes, p) class means, R (n, p) residuals, w (n,) row weights, g (p,) > 0.
    """
    keep = w > 0
    R, w = R[keep], w[keep]
    Rg = R / g
    K = np.diag(1.0 / w) + Rg @ R.T
    Mg = M / g
    W = Mg - np.linalg.solve(K, R @ Mg.T).T @ Rg           # rows: Sigma^-1 m_c
    b = -0.5 * np.einsum("cd,cd->c", M, W)
    return np.argmax(Xte @ W.T + b, axis=1)


def split_fold(X, s):
    """Training / test samples of LOSO fold s.  X: (n_sub, n_seg, p)."""
    n_sub, n_seg, p = X.shape
    train = np.delete(np.arange(n_sub), s)
    y = np.arange(n_seg)
    return (X[train].reshape(-1, p).astype(np.float64), np.tile(y, n_sub - 1),
            X[s].astype(np.float64), y)


def predict_voxel_lw(Xtr, ytr, Xte):
    """Pooled within-class covariance, Ledoit-Wolf in the full voxel space."""
    M = np.stack([Xtr[ytr == c].mean(0) for c in np.unique(ytr)])
    R = Xtr - M[ytr]
    n, p = R.shape
    a, mu = lw_intensity(R)
    pred = woodbury_predict(M, R, np.full(n, (1 - a) / n), np.full(p, a * mu), Xte)
    return pred, a


def predict_sklearn_auto(Xtr, ytr, Xte):
    """Equivalent of LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').

    scikit-learn: Sigma = sum_c prior_c * D_c [(1 - a_c) Z_c^T Z_c / n_c + a_c mu_c I] D_c,
    where Z_c are the class-c samples standardised feature-wise (scale D_c) and (a_c, mu_c)
    is the Ledoit-Wolf estimate.
    """
    classes = np.unique(ytr)
    M = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    R = Xtr - M[ytr]
    n, p = R.shape
    g, w = np.zeros(p), np.zeros(n)
    for c in classes:
        idx = np.flatnonzero(ytr == c)
        sd = R[idx].std(0)
        sd[sd == 0] = 1.0
        a, mu = lw_intensity(R[idx] / sd)
        prior = len(idx) / n
        g += prior * a * mu * sd ** 2
        w[idx] = prior * (1 - a) / len(idx)
    return woodbury_predict(M, R, w, g, Xte)


def flat_samples(x):
    """(18, 280, V) raw ROI data -> (subject, segment, 20 V) samples, exactly as in Task A."""
    t = segment_samples(zscore(x, axis=1), SEG_LEN)
    return np.transpose(t, (3, 0, 1, 2)).reshape(t.shape[3], N_SEG, -1)


def run_roi_lda(x):
    """x: (18, 280, V) raw ROI data.  Returns a dict of per-fold arrays."""
    X = flat_samples(x)
    n_sub, _, p = X.shape
    out = {k: np.zeros(n_sub) for k in VARIANTS + ["a_span", "a_span_voxel_equiv", "a_voxel"]}
    for s in range(n_sub):
        Xtr, ytr, Xte, y = split_fold(X, s)
        lda = ShrinkageLDA().fit(Xtr, ytr)
        out["span_lw"][s] = lda.score(Xte, y)
        a, q = lda.shrinkage_, lda.means_.shape[1]
        out["a_span"][s] = a
        # (1-a) S + a (tr S / q) I  is proportional to  (1-a') S + a' (tr S / p) I  with:
        out["a_span_voxel_equiv"][s] = a * p / (a * p + (1 - a) * q)
        pred, a_vox = predict_voxel_lw(Xtr, ytr, Xte)
        out["voxel_lw"][s] = np.mean(pred == y)
        out["a_voxel"][s] = a_vox
        out["sklearn_auto"][s] = np.mean(predict_sklearn_auto(Xtr, ytr, Xte) == y)
    return out


def selfcheck_lda(max_p=700):
    """Compare the Gram / Woodbury shortcuts with scikit-learn's dense implementations on
    the largest ROI that is still small enough for dense p x p covariances (all 18 folds)."""
    k = int(np.argmax(np.where(N_VOX * SEG_LEN <= max_p, N_VOX, -1)))
    X = flat_samples(ROI_X[k])
    d_a, same_vox, same_skl = 0.0, True, True
    for s in range(X.shape[0]):
        Xtr, ytr, Xte, y = split_fold(X, s)
        M = np.stack([Xtr[ytr == c].mean(0) for c in y])
        R = Xtr - M[ytr]
        d_a = max(d_a, abs(lw_intensity(R)[0] - ledoit_wolf_shrinkage(R, assume_centered=True)))
        cov = LedoitWolf(assume_centered=True).fit(R).covariance_
        W = np.linalg.solve(cov, M.T).T
        dense = np.argmax(Xte @ W.T - 0.5 * np.einsum("cd,cd->c", M, W), axis=1)
        same_vox &= bool(np.array_equal(dense, predict_voxel_lw(Xtr, ytr, Xte)[0]))
        skl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
        same_skl &= bool(np.array_equal(skl.predict(Xte), predict_sklearn_auto(Xtr, ytr, Xte)))
    res = {"roi": LABELS[k], "n_features": int(X.shape[2]), "n_folds": int(X.shape[0]),
           "max_abs_diff_lw_intensity_vs_sklearn": float(d_a),
           "voxel_lw_predictions_equal_dense": same_vox,
           "sklearn_auto_predictions_equal_sklearn_lda": same_skl}
    assert d_a < 1e-9 and same_vox and same_skl, res
    return res

# %%
_t = time.time()
_check = selfcheck_lda()
print(f"self-check passed on ROI {_check['roi']} ({_check['n_features']} features, "
      f"{_check['n_folds']} folds):")
print("   LW intensity, largest difference to scikit-learn: "
      f"{_check['max_abs_diff_lw_intensity_vs_sklearn']:.1e}")
print("   predictions identical to the dense implementations: voxel_lw "
      f"{_check['voxel_lw_predictions_equal_dense']}, sklearn_auto "
      f"{_check['sklearn_auto_predictions_equal_sklearn_lda']}")
order = np.argsort(-N_VOX)
_res = [None] * len(VOIS)
_out = Parallel(n_jobs=N_JOBS)(delayed(run_roi_lda)(ROI_X[k]) for k in order)
for k, o in zip(order, _out):
    _res[k] = o

# %%
jL = CLF_NAMES.index("LDA")
acc_v = {v: np.stack([r[v] for r in _res]) * 100 for v in VARIANTS}       # (52, 18)
_same = np.abs(acc_v["span_lw"] - ACC_A[:, :, jL] * 100).max()
assert _same < 1e-9, f"span_lw does not reproduce the Task-A LDA accuracies ({_same})"

LD = {"design": "Task A design: 14 classes, 20*V features, LOSO over 18 subjects, 52 ROIs; "
                "accuracy in %",
      "variants": {
          "span_lw": "main model: Ledoit-Wolf in the 238-dim PCA span of the training fold",
          "voxel_lw": "exact pooled Ledoit-Wolf in the 20*V voxel space",
          "sklearn_auto": "equivalent of sklearn LDA(solver='lsqr', shrinkage='auto') "
                          "on raw features"},
      "selfcheck": _check, "span_lw_reproduces_part2a_lda": True, "clf": CLF_NAMES,
      "mean_over_rois": {}, "median_over_rois": {}, "n_rois_lda_best": {},
      "n_rois_best_by_clf": {}, "clf_mean": {}, "top3_rois": {}, "top3_mean_acc": {},
      "diff_vs_span_lw_pp": {}}
for v in VARIANTS:
    mv = ACC_A.mean(1) * 100                          # (52, 7), LDA column replaced below
    mv[:, jL] = acc_v[v].mean(1)
    best = np.round(mv, 6).argmax(1)                  # exact ties: first-listed classifier
    top = np.argsort(-mv.mean(1))[:3]
    LD["mean_over_rois"][v] = float(acc_v[v].mean())
    LD["median_over_rois"][v] = float(np.median(acc_v[v].mean(1)))
    LD["n_rois_lda_best"][v] = int((best == jL).sum())
    LD["n_rois_best_by_clf"][v] = [int((best == j).sum()) for j in range(len(CLF_NAMES))]
    LD["clf_mean"][v] = mv.mean(0).tolist()
    LD["top3_rois"][v] = [LABELS[k] for k in top]
    LD["top3_mean_acc"][v] = [float(mv[k].mean()) for k in top]
    if v != "span_lw":
        d = acc_v[v].mean(1) - acc_v["span_lw"].mean(1)
        LD["diff_vs_span_lw_pp"][v] = {"mean": float(d.mean()), "min": float(d.min()),
                                       "max": float(d.max())}
_a = {k: np.array([r[k].mean() for r in _res])
      for k in ("a_span", "a_span_voxel_equiv", "a_voxel")}
wide = N_VOX >= WIDE_V
LD["shrinkage"] = {
    "wide_roi_min_voxels": WIDE_V, "n_wide_rois": int(wide.sum()),
    **{f"{k}_range_wide": [float(v[wide].min()), float(v[wide].max())]
       for k, v in _a.items()},
    **{f"{k}_range_all": [float(v.min()), float(v.max())] for k, v in _a.items()}}
LD["table"] = [{"roi": LABELS[k], "n_vox": int(N_VOX[k]),
                **{v: float(acc_v[v][k].mean()) for v in VARIANTS},
                **{f"{v}_sem": float(sem(acc_v[v][k])) for v in VARIANTS},
                **{key: float(val[k]) for key, val in _a.items()}}
               for k in range(len(VOIS))]
save_json(LD, "supp_lda_sensitivity.json")
TIMINGS["3.5 LDA regulariser"] = time.time() - _t

for v in VARIANTS:
    print(f"{v:13s} mean {LD['mean_over_rois'][v]:.1f} %, LDA best in "
          f"{LD['n_rois_lda_best'][v]:2d} ROIs, top-3 {LD['top3_rois'][v]}")
_sh = LD["shrinkage"]
print(f"shrinkage intensity in the {_sh['n_wide_rois']} ROIs with V >= {WIDE_V} (ROI means):")
print("   main model, in voxel-space terms: "
      f"{_sh['a_span_voxel_equiv_range_wide'][0]:.2f} to "
      f"{_sh['a_span_voxel_equiv_range_wide'][1]:.2f}")
print(f"   voxel-space LW                    : {_sh['a_voxel_range_wide'][0]:.2f} to "
      f"{_sh['a_voxel_range_wide'][1]:.2f}")
for v, d in LD["diff_vs_span_lw_pp"].items():
    print(f"per-ROI difference {v} minus main model: {d['min']:+.1f} to {d['max']:+.1f} pp")
print(f"ROIs with fewer than {17 * N_SEG} features: "
      f"{int((N_VOX * SEG_LEN < 17 * N_SEG).sum())}")
print(f"time: {TIMINGS['3.5 LDA regulariser']:.0f} s")

# %%
jS, jN = CLF_NAMES.index("Linear SVM"), CLF_NAMES.index("Naive Bayes")
# example ROIs: large auditory / largest default-mode / mid-sized / small
LDA_ROIS = ["48 A1_R", "08 PREC_L", "12 pSTS_R", "15 aMTG_R"]
_ld_tab = {r["roi"]: r for r in LD["table"]}
_names = {"span_lw": "Main model: LW in the PCA span of the training fold",
          "sklearn_auto": 'scikit-learn LDA, lsqr + shrinkage="auto", raw features',
          "voxel_lw": "Exact pooled LW in the 20·V voxel space"}
show_table(pd.DataFrame(
    [[_names[v], f"{LD['mean_over_rois'][v]:.1f}",
      *[f"{_ld_tab[r][v]:.1f}" for r in LDA_ROIS], LD["n_rois_lda_best"][v],
      f"{LD['n_rois_best_by_clf'][v][jS]} / {LD['n_rois_best_by_clf'][v][jN]}"]
     for v in ("span_lw", "sklearn_auto", "voxel_lw")],
    columns=["LDA covariance regulariser", "Mean over 52 ROIs",
             *[f"{r.split(' ', 1)[1]} (V = {_ld_tab[r]['n_vox']})" for r in LDA_ROIS],
             "# ROIs: LDA best", "# ROIs: SVM / NB best"]))

# %% [prose] lda.cap

# %% [prose] classifiers.p

# %% [markdown]
# ### 3.6 Supplementary analysis: temporal profile or spatial pattern?
#
# A Task-A sample is the 20 TRs × V voxels response matrix of one story segment. With the
# same data, LOSO design and linear SVM (C = 1, one-vs-one), three feature sets are compared
# in all 52 ROIs:
#
# - `full`: the 20 × V matrix, flattened (identical to the Task-A linear SVM);
# - `temporal`: the ROI-mean time course of the segment, i.e. the mean over voxels at each TR
#   (20 features);
# - `spatial`: the time-averaged spatial pattern of the segment, i.e. the mean over the 20 TRs
#   per voxel (V features).
#
# Because a linear SVM with fixed C is not scale-invariant, a sensitivity variant
# re-standardises the two reduced feature sets within subject (`temporal_z`: ROI-mean time
# course z-scored over the 280 TRs; `spatial_z`: each voxel's 14 segment means z-scored
# across segments). No parameter is tuned. Run time: about 10 seconds.

# %%
SETS = ["full", "temporal", "spatial", "temporal_z", "spatial_z"]


def feature_sets(x):
    """x: (18, 280, V) raw ROI data -> dict of (18 subjects, 14 segments, n_features)."""
    n_sub, _, V = x.shape
    z = zscore(x, axis=1)                         # voxel-wise, within subject (as in Task A)
    t = z.reshape(n_sub, N_SEG, SEG_LEN, V)
    return {"full": t.reshape(n_sub, N_SEG, -1),
            "temporal": t.mean(3),
            "spatial": t.mean(2),
            "temporal_z": zscore(z.mean(2), axis=1).reshape(n_sub, N_SEG, SEG_LEN),
            "spatial_z": zscore(t.mean(2), axis=1)}


def loso_svm(X):
    """LOSO accuracy per held-out subject for X (n_sub, n_seg, n_features)."""
    n_sub, n_seg, p = X.shape
    y = np.arange(n_seg)
    acc = np.zeros(n_sub)
    for s in range(n_sub):
        train = np.delete(np.arange(n_sub), s)
        svm = GramLinearSVC(C=1.0).fit(X[train].reshape(-1, p), np.tile(y, n_sub - 1))
        acc[s] = svm.score(X[s], y)
    return acc


def run_roi_sets(x):
    return {k: loso_svm(v) for k, v in feature_sets(x).items()}


_t = time.time()
order = np.argsort(-N_VOX)
_res = [None] * len(VOIS)
_out = Parallel(n_jobs=N_JOBS)(delayed(run_roi_sets)(ROI_X[k]) for k in order)
for k, o in zip(order, _out):
    _res[k] = o
acc_s = {s: np.stack([r[s] for r in _res]) * 100 for s in SETS}            # (52, 18)
_d = np.abs(acc_s["full"] - ACC_A[:, :, CLF_NAMES.index("Linear SVM")] * 100).max()
assert _d < 1e-9, f"'full' does not reproduce the Task-A linear-SVM accuracies ({_d})"

ms = {s: acc_s[s].mean(1) for s in SETS}                                   # per-ROI means
TS = {"design": "linear SVM (C = 1, one-vs-one), 14 classes, LOSO over 18 subjects, "
                "52 ROIs; accuracy in %",
      "feature_sets": {
          "full": "20 x V segment matrix (Task A)",
          "temporal": "ROI-mean time course of the segment (20 features)",
          "spatial": "time-averaged spatial pattern of the segment (V features)",
          "temporal_z": "as temporal, ROI-mean time course z-scored within subject",
          "spatial_z": "as spatial, segment means z-scored across segments per voxel"},
      "chance_pct": 100 / N_SEG, "full_reproduces_part2a_svm": True,
      "mean_over_rois": {s: float(ms[s].mean()) for s in SETS},
      "median_over_rois": {s: float(np.median(ms[s])) for s in SETS},
      "n_rois_temporal_gt_spatial": int((ms["temporal"] > ms["spatial"]).sum()),
      "n_rois_spatial_gt_temporal": int((ms["spatial"] > ms["temporal"]).sum()),
      "n_rois_full_ge_both": int(((ms["full"] >= ms["temporal"])
                                  & (ms["full"] >= ms["spatial"])).sum()),
      "n_rois_temporal_ge_full": int((ms["temporal"] >= ms["full"]).sum()),
      "n_rois_temporal_z_gt_spatial_z": int((ms["temporal_z"] > ms["spatial_z"]).sum()),
      "table": [{"roi": LABELS[k], "n_vox": int(N_VOX[k]),
                 **{s: float(ms[s][k]) for s in SETS},
                 **{f"{s}_sem": float(sem(acc_s[s][k])) for s in SETS}}
                for k in range(len(VOIS))]}
save_json(TS, "supp_temporal_vs_spatial.json")
TIMINGS["3.6 Temporal vs. spatial"] = time.time() - _t

print("mean over the 52 ROIs (%):",
      ", ".join(f"{s} {v:.1f}" for s, v in TS["mean_over_rois"].items()))
print(f"temporal > spatial in {TS['n_rois_temporal_gt_spatial']} of 52 ROIs; "
      f"full >= both reduced sets in {TS['n_rois_full_ge_both']} ROIs")
print("spatial > temporal in:",
      ", ".join(r["roi"] for r in TS["table"] if r["spatial"] > r["temporal"]))
_ts = {r["roi"]: r["temporal"] for r in TS["table"]}
print("temporal, four leading ROIs (%):",
      ", ".join(f"{lab} {_ts[lab]:.1f}" for lab in A["rank_order"][:4]))
print(f"time: {TIMINGS['3.6 Temporal vs. spatial']:.0f} s")

# %%
# example ROIs: best auditory, largest, best non-auditory, one exception, mid-sized temporal
TS_ROIS = ["48 A1_R", "47 IFG_L_UNIQUE", "08 PREC_L", "39 PREC_L", "12 pSTS_R"]
_ts_tab = {r["roi"]: r for r in TS["table"]}
_cols = ("full", "temporal", "spatial")
_rows = [[r, _ts_tab[r]["n_vox"],
          *[f"{_ts_tab[r][s]:.1f} ± {_ts_tab[r][s + '_sem']:.1f}" for s in _cols]]
         for r in TS_ROIS]
_rows.append(["<b>Mean over 52 ROIs</b>", "",
              *[f"<b>{TS['mean_over_rois'][s]:.1f}</b>" for s in _cols]])
show_table(pd.DataFrame(_rows, columns=[
    "ROI", "V", "Full sample (20·V features)", "ROI-mean time course (20 features)",
    "Time-averaged spatial pattern (V features)"]))

# %% [prose] tvs.cap

# %% [prose] samples.p

# %% [prose] h4.h2

# %% [markdown]
# ### 4.1 Code
#
# - `procrustes(X, Y)` returns the orthogonal matrix R (RᵀR = I, no scaling) that minimises
#   ‖XR − Y‖.
# - `build_template` implements the three steps of Haxby et al. (2011) on a list of
#   time × voxel matrices.
# - `run_roi_task_b` runs the complete split-half design for one ROI. For each of the two
#   assignments of the halves and each LOSO fold it builds the template from the alignment
#   half of the 17 training subjects, estimates one rotation per subject (the held-out
#   subject is aligned to the training template with its own alignment half), rotates
#   every subject's decoding half into the common space and evaluates all classifiers.
#   The anatomical baseline is the same procedure without rotation. For the optional
#   analysis the common space is reduced to the top-k right singular vectors of the training
#   template.
# - The goodness of the Procrustes fit is recorded as the relative residual
#   ‖XR − T‖² / ‖X‖² on the alignment half (X: a subject's alignment half, T: the template
#   of the fold).
#
# All of Task B runs in double precision.

# %%
HALF = 140                        # TRs per half (7 segments)
K_LIST = [5, 10, 20, 40, 80]      # dimensionalities of the reduced common space


def procrustes(X, Y):
    """Orthogonal R (R^T R = I, no scaling) minimising ||X R - Y||_F."""
    return orthogonal_procrustes(X, Y)[0]


def build_template(data):
    """Haxby et al. (2011) common-space template from a list of (T, V) matrices.

    (i)   start from a reference subject; align the next subject to it and average; then
          align each remaining subject to the evolving template (running mean of everything
          aligned so far);
    (ii)  second pass: re-align every subject to the pass-1 template and average the
          re-aligned data;
    (iii) return that final template (a subject's final R is its Procrustes fit to it).
    """
    template = data[0].copy()
    for n, X in enumerate(data[1:], start=1):
        template = (template * n + X @ procrustes(X, template)) / (n + 1)
    return np.mean([X @ procrustes(X, template) for X in data], axis=0)


def rel_residual(X, R, template):
    """Relative Procrustes residual ||X R - T||_F^2 / ||X||_F^2 (0 = perfect fit)."""
    return float(np.linalg.norm(X @ R - template) ** 2 / np.linalg.norm(X) ** 2)


def loso_fold(common):
    """Train on all but the last subject of `common` (n_sub, T, features), test on the last."""
    t = segment_samples(common, SEG_LEN)                              # (7, 20, F, n_sub)
    X = np.transpose(t, (3, 0, 1, 2)).reshape(common.shape[0], t.shape[0], -1)
    y = np.arange(t.shape[0])
    return fit_score_all(X[:-1].reshape(-1, X.shape[2]), np.tile(y, X.shape[0] - 1),
                         X[-1], y)


def run_roi_task_b(x):
    """x: (18, 280, V) raw ROI data.

    Returns accuracies (2 half-assignments, 18 folds, 7 classifiers) for the anatomical
    baseline and for full hyperalignment, accuracies (len(K_LIST), 2, 18, 7) for the reduced
    common space, and the relative Procrustes residual on the alignment half, (2, 18, 2) =
    (half, fold, [mean of the 17 training subjects, held-out subject]).
    """
    x = x.astype(np.float64)                 # Procrustes SVD and classifiers in double
    n_sub, _, V = x.shape
    resid = np.zeros((2, n_sub, 2))
    acc_anat = np.zeros((2, n_sub, len(CLF_NAMES)))
    acc_hyp = np.zeros_like(acc_anat)
    acc_k = np.full((len(K_LIST), 2, n_sub, len(CLF_NAMES)), np.nan)
    # voxels are z-scored within each half separately, so the halves share nothing
    halves = [zscore(x[:, :HALF], axis=1), zscore(x[:, HALF:], axis=1)]
    for h in range(2):                       # h = index of the DECODING half
        dec, ali = halves[h], halves[1 - h]
        acc_anat[h] = loso_accuracy(segment_samples(dec, SEG_LEN))
        for s in range(n_sub):
            train = [i for i in range(n_sub) if i != s]
            template = build_template([ali[i] for i in train])        # training subjects only
            R = {i: procrustes(ali[i], template) for i in range(n_sub)}   # incl. held-out
            resid[h, s] = (np.mean([rel_residual(ali[i], R[i], template) for i in train]),
                           rel_residual(ali[s], R[s], template))
            common = np.stack([dec[i] @ R[i] for i in train + [s]])   # held-out subject last
            acc_hyp[h, s] = loso_fold(common)
            # reduced common space: top-k right singular vectors of the training template
            Vt = np.linalg.svd(template, full_matrices=False)[2]
            for j, k in enumerate(K_LIST):
                if k < min(V, HALF - 1):
                    acc_k[j, h, s] = loso_fold(common @ Vt[:k].T)
    return acc_anat, acc_hyp, acc_k, resid

# %% [markdown]
# ### 4.2 Run
#
# This is the longest step of the notebook: about 7 minutes on 4 cores (2 half assignments ×
# 18 folds × 52 ROIs, each with a new template, 18 Procrustes fits and up to seven
# evaluations of all classifiers).

# %%
_t = time.time()
order = np.argsort(-N_VOX)
_out = Parallel(n_jobs=N_JOBS)(delayed(run_roi_task_b)(ROI_X[k]) for k in order)
_n = len(CLF_NAMES)
ANAT = np.zeros((len(VOIS), 2, N_SUB, _n))
HYPER = np.zeros((len(VOIS), 2, N_SUB, _n))
REDUCED = np.zeros((len(VOIS), len(K_LIST), 2, N_SUB, _n))
RESID = np.zeros((len(VOIS), 2, N_SUB, 2))   # (ROI, decoding half, fold, [train, held-out])
for k, (acc_anat, acc_hyp, acc_k, resid) in zip(order, _out):
    ANAT[k], HYPER[k], REDUCED[k], RESID[k] = acc_anat, acc_hyp, acc_k, resid
np.savez(WORK / "part2b_accuracy.npz", anat=ANAT, hyper=HYPER, reduced=REDUCED,
         k_list=K_LIST, clf=CLF_NAMES, labels=LABELS, n_vox=N_VOX, procrustes_resid=RESID)
TIMINGS["4.2 Task B hyperalignment"] = time.time() - _t
print(f"time: {TIMINGS['4.2 Task B hyperalignment']:.0f} s")
print("mean accuracy over the 52 ROIs (%):")
print_by_classifier({"anatomical": ANAT.mean((0, 1, 2)) * 100,
                     "hyperaligned": HYPER.mean((0, 1, 2)) * 100})

# %% [markdown]
# ### 4.3 Summary statistics
#
# Accuracies are averaged over the two half assignments. The gain of an ROI is the difference
# hyperaligned − anatomical, averaged over the seven classifiers; it is tested against zero
# with a paired t-test over the 18 held-out subjects (FDR over the 52 ROIs). Across ROIs,
# anatomical and hyperaligned accuracies are compared with a Wilcoxon signed-rank test.

# %%
def wilcoxon_p(x):
    """Two-sided Wilcoxon signed-rank p-value (1.0 if all differences are zero)."""
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return 1.0


def min_max(v):
    return [float(v.min()), float(v.max())]


def mean_of_halves(reduced):
    """Reduced-space accuracies (%), mean of the two half assignments: (52, K, 18, 7).
    Entries stay NaN where k is not defined for the ROI (k >= V)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN entries
        return np.nanmean(reduced, axis=2) * 100


def summarise_task_b(anat, hyper, reduced, resid_all):
    """Summary dictionary of the Task-B results (accuracies as fractions)."""
    clf, labels, nvox, klist = CLF_NAMES, LABELS, N_VOX, K_LIST
    A_, H_ = anat.mean(1) * 100, hyper.mean(1) * 100     # (52, 18, 7): mean of both halves
    R_ = mean_of_halves(reduced)                         # (52, K, 18, 7)
    a, h = A_.mean(1), H_.mean(1)                        # (52, 7)
    # Correct test samples per ROI and classifier, summed over both halves and the 18 folds
    # (7 test samples per fold).  "Improved" is decided on these integers: comparing the
    # float means would count exact ties as gains or losses by rounding error.
    n_test = 7
    ca, ch = (np.rint(x * n_test).astype(int).sum(axis=(1, 2)) for x in (anat, hyper))
    assert max(np.abs(x * n_test - np.rint(x * n_test)).max() for x in (anat, hyper)) < 1e-9
    gm = (H_ - A_).mean(2)                               # (52, 18): gain, mean of classifiers
    gs = sem(gm, 1)
    order = np.argsort(-gm.mean(1))
    big = nvox > max(klist)                              # every k is defined in these ROIs
    Ah, Hh = anat * 100, hyper * 100                     # (52, 2, 18, 7): halves separately
    pr = np.array([ttest_1samp(gm[k], 0).pvalue for k in range(len(labels))])
    thr = fdr_threshold(pr)
    subj_gain = gm.mean(0)
    res = resid_all.mean((1, 2))                         # (52, 2): ROI means
    wide = nvox > HALF                                   # more voxels than alignment TRs
    top_h = np.argsort(-h.mean(1))
    k20 = klist.index(20)
    return {
        "clf": clf, "k_list": [int(k) for k in klist],
        "procrustes_residual": {
            "definition": "||X R - T||_F^2 / ||X||_F^2 on the alignment half, "
                          "mean over folds and both splits",
            "n_align_TR": HALF, "n_rois_V_gt_T": int(wide.sum()),
            "train_range_all": min_max(res[:, 0]), "heldout_range_all": min_max(res[:, 1]),
            "train_range_V_gt_T": min_max(res[wide, 0]),
            "heldout_range_V_gt_T": min_max(res[wide, 1]),
            "min_any_fold": float(resid_all.min())},
        "top6_hyper_k20_vs_full": [{"roi": labels[k], "k20": float(np.nanmean(R_[k, k20])),
                                    "full": float(h[k].mean())} for k in top_h[:6]],
        "by_half": {"anat": Ah.mean((0, 2)).tolist(), "hyper": Hh.mean((0, 2)).tolist()},
        "n_sig_gain_fdr": int(((pr <= thr) & (gm.mean(1) > 0)).sum()),
        "n_sig_loss_fdr": int(((pr <= thr) & (gm.mean(1) < 0)).sum()),
        "sig_gain_rois": [labels[k] for k in order if pr[k] <= thr and gm[k].mean() > 0],
        "sig_loss_rois": [labels[k] for k in order if pr[k] <= thr and gm[k].mean() < 0],
        "subj_gain_mean": float(subj_gain.mean()),
        "subj_gain_p": float(ttest_1samp(subj_gain, 0).pvalue),
        "anat_mean": a.mean(0).tolist(), "hyper_mean": h.mean(0).tolist(),
        "gain_mean": (h - a).mean(0).tolist(), "n_improved": (ch > ca).sum(0).tolist(),
        "n_tied": (ch == ca).sum(0).tolist(),
        "n_test_per_roi": int(n_test * anat.shape[1] * anat.shape[2]),
        "p_wilcoxon_rois": [wilcoxon_p(h[:, j] - a[:, j]) for j in range(len(clf))],
        "overall_anat": float(a.mean()), "overall_hyper": float(h.mean()),
        "n_rois_improved_mean": int((ch.sum(1) > ca.sum(1)).sum()),
        "n_rois_tied_mean": int((ch.sum(1) == ca.sum(1)).sum()),
        "p_wilcoxon_overall": wilcoxon_p(gm.mean(1)),
        "corr_gain_logV": float(np.corrcoef(np.log(nvox), gm.mean(1))[0, 1]),
        "corr_gain_anat": float(np.corrcoef(a.mean(1), gm.mean(1))[0, 1]),
        "gain_bigV": float(gm.mean(1)[nvox > 139].mean()),
        "gain_smallV": float(gm.mean(1)[nvox <= 139].mean()),
        "n_bigV": int((nvox > 139).sum()),
        "top_gain": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat": float(a[k].mean()),
                      "hyper": float(h[k].mean()), "gain": float(gm[k].mean()),
                      "sem": float(gs[k]), "anat_clf": a[k].tolist(),
                      "hyper_clf": h[k].tolist()} for k in order[:10]],
        "worst_gain": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat": float(a[k].mean()),
                        "hyper": float(h[k].mean()), "gain": float(gm[k].mean())}
                       for k in order[-5:]],
        "top_hyper": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat_clf": a[k].tolist(),
                       "hyper_clf": h[k].tolist()} for k in top_h[:5]],
        "reduced_mean_bigrois": {str(int(k)): np.nanmean(R_[big][:, i], axis=(0, 1)).tolist()
                                 for i, k in enumerate(klist)},
        "hyper_mean_bigrois": h[big].mean(0).tolist(),
        "anat_mean_bigrois": a[big].mean(0).tolist(), "n_big": int(big.sum()),
        "table": [{"roi": labels[k], "n_vox": int(nvox[k]), "anat": a[k].tolist(),
                   "hyper": h[k].tolist(), "anat_sem": sem(A_, 1)[k].tolist(),
                   "hyper_sem": sem(H_, 1)[k].tolist(), "gain": float(gm[k].mean()),
                   "gain_sem": float(gs[k]), "gain_p": float(pr[k]),
                   "gain_sig_fdr": bool(pr[k] <= thr), "resid_train": float(res[k, 0]),
                   "resid_heldout": float(res[k, 1])} for k in range(len(labels))],
    }

# %%
B = summarise_task_b(ANAT, HYPER, REDUCED, RESID)
save_json(B, "part2b_summary.json")
print(f"all ROIs and classifiers: {B['overall_anat']:.1f} % -> {B['overall_hyper']:.1f} %")
print(f"gain per held-out subject: {B['subj_gain_mean']:+.1f} pp "
      f"(paired t-test over the 18 subjects: p = {B['subj_gain_p']:.3f})")
print(f"ROIs with more correct predictions after hyperalignment (all classifiers together): "
      f"{B['n_rois_improved_mean']} of 52; exact ties: {B['n_rois_tied_mean']}")
print(f"significant (FDR q < .05) gains: {B['n_sig_gain_fdr']} ROIs, "
      f"losses: {B['n_sig_loss_fdr']} ROIs")
print(f"   losses in: {', '.join(B['sig_loss_rois'])}")
_pr = B["procrustes_residual"]
print(f"relative Procrustes residual in the {_pr['n_rois_V_gt_T']} ROIs with V > {HALF} "
      "(ROI means):")
print(f"   training subjects {_pr['train_range_V_gt_T'][0]:.2f} to "
      f"{_pr['train_range_V_gt_T'][1]:.2f}, held-out subject "
      f"{_pr['heldout_range_V_gt_T'][0]:.2f} to {_pr['heldout_range_V_gt_T'][1]:.2f}")
print(f"correlation of the gain with log V: r = {B['corr_gain_logV']:+.2f}")

# %% [markdown]
# ### 4.4 Results

# %%
A_, H_ = ANAT.mean(1) * 100, HYPER.mean(1) * 100          # (52, 18, 7)
R_ = mean_of_halves(REDUCED)                              # (52, K, 18, 7)
a, h = A_.mean(1), H_.mean(1)                             # (52, 7)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for j, ax in enumerate(axes.ravel()[:7]):
    ax.scatter(a[:, j], h[:, j], s=8 + N_VOX / 12, color=COL[j], alpha=0.75,
               edgecolor="k", lw=0.3)
    ax.plot([0, 100], [0, 100], "k-", lw=0.7)
    ax.axhline(CH7, color="k", ls=":", lw=0.7)
    ax.axvline(CH7, color="k", ls=":", lw=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_title(f"{CLF_NAMES[j]}\n{a[:, j].mean():.1f} → {h[:, j].mean():.1f} %  "
                 f"({B['n_improved'][j]}/52 ROIs improve)", fontsize=9.5)
    ax.set_xlabel("Anatomical alignment (%)")
    ax.set_ylabel("Hyperalignment (%)")
ax = axes.ravel()[7]
xpos, w = np.arange(len(CLF_NAMES)), 0.38
ax.bar(xpos - w / 2, a.mean(0), w, yerr=sem(A_.mean(0), 0), color="0.6",
       label="anatomical", capsize=2)
ax.bar(xpos + w / 2, h.mean(0), w, yerr=sem(H_.mean(0), 0), color="#d62728",
       label="hyperaligned", capsize=2)
ax.axhline(CH7, color="k", ls="--", lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(CLF_NAMES, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Mean accuracy over 52 ROIs (%)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Average over ROIs (± SEM over folds)", fontsize=9.5)
fig.suptitle("Task B – 7-way split-half LOSO decoding: anatomical vs. hyperaligned "
             "features (one dot per ROI, size ~ voxels; dotted = chance 14.29 %)",
             fontsize=11)
fig.tight_layout()
show_figure(fig, "part2b_scatter.png", dpi=170, width=900)

# %% [prose] scatter.cap

# %%
def psci(p):
    """p-value with two significant digits; a × 10^b notation below 0.001."""
    if p >= 0.001:
        return f"{p:.2g}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}×10<sup>{MINUS}{abs(int(exp))}</sup>"


show_table(pd.DataFrame({
    "Classifier": CLF_NAMES,
    "Anatomical (%)": [f"{v:.1f}" for v in B["anat_mean"]],
    "Hyperaligned (%)": [f"{v:.1f}" for v in B["hyper_mean"]],
    "Gain (pp)": [signed(v, "+.1f") for v in B["gain_mean"]],
    "ROIs improved": [f"{v}/52" for v in B["n_improved"]],
    "ROIs tied": B["n_tied"],
    "Wilcoxon p (52 ROIs)": [psci(p) for p in B["p_wilcoxon_rois"]],
}))

# %% [prose] hyp.cap

# %%
gm = (H_ - A_).mean(2)                                     # (52, 18): gain per ROI and fold
order = np.argsort(-gm.mean(1))
fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
xx = np.arange(52)
ax[0].bar(xx - 0.2, A_.mean(2).mean(1)[order], 0.4, yerr=sem(A_.mean(2), 1)[order],
          color="0.6", label="anatomical", error_kw=dict(lw=0.6))
ax[0].bar(xx + 0.2, H_.mean(2).mean(1)[order], 0.4, yerr=sem(H_.mean(2), 1)[order],
          color="#d62728", label="hyperaligned", error_kw=dict(lw=0.6))
ax[0].axhline(CH7, color="k", ls="--", lw=0.8, label="chance (14.29 %)")
ax[0].set_ylabel("Accuracy, mean of 7 classifiers (%)")
ax[0].legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9)
ax[1].bar(xx, gm.mean(1)[order], yerr=sem(gm, 1)[order],
          color=np.where(gm.mean(1)[order] >= 0, "#d62728", "#3182bd"),
          error_kw=dict(lw=0.6))
ax[1].axhline(0, color="k", lw=0.7)
ax[1].set_ylabel("Gain: hyper − anatomical (pct. points)")
ax[1].set_xticks(xx)
ax[1].set_xticklabels([f"{LABELS[k]} ({N_VOX[k]})" for k in order], rotation=90,
                      fontsize=8.5)
ax[1].set_xlim(-0.7, 51.7)
for q in ax:
    q.spines[["top", "right"]].set_visible(False)
ax[0].set_title("Task B – effect of hyperalignment per ROI (mean ± SEM over 18 folds; "
                "ROIs sorted by gain)", fontsize=11, loc="left")
fig.tight_layout()
show_figure(fig, "part2b_gain_per_roi.png", dpi=170, width=900)

# %% [prose] gainroi.cap

# %%
N_GAIN = 8        # number of ROIs listed in the table of the largest gains
_btab = {r["roi"]: r for r in B["table"]}
_cols = ["V", "Anat.", "Hyper.", "Gain ± SEM"]
show_table(pd.DataFrame(
    [[r["roi"], r["n_vox"], f"{r['anat']:.1f}", f"{r['hyper']:.1f}",
      f"{signed(r['gain'], '+.1f')} ± {r['sem']:.1f}"] for r in B["top_gain"][:N_GAIN]],
    columns=["Largest gains", *_cols]))
show_table(pd.DataFrame(
    [[r["roi"], r["n_vox"], f"{r['anat']:.1f}", f"{r['hyper']:.1f}",
      f"{signed(r['gain'], '+.1f')} ± {_btab[r['roi']]['gain_sem']:.1f}"]
     for r in B["worst_gain"][::-1]],
    columns=["Largest losses", *_cols]))

# %% [prose] gain.cap

# %%
_bh = B["by_half"]
_rows = []
for j, c in enumerate(CLF_NAMES):
    _row = [c]
    for half in (0, 1):
        _row += [f"{_bh['anat'][half][j]:.1f}", f"{_bh['hyper'][half][j]:.1f}",
                 signed(_bh["hyper"][half][j] - _bh["anat"][half][j], "+.1f")]
    _rows.append(_row)
_top = ["Decode 1st half (align on 2nd)"] * 3 + ["Decode 2nd half (align on 1st)"] * 3
_columns = pd.MultiIndex.from_arrays(
    [["Classifier", *_top], ["", "Anat.", "Hyper.", "Gain", "Anat.", "Hyper.", "Gain"]])
show_table(pd.DataFrame(_rows, columns=_columns))
print("mean of the 7 classifiers, anatomical -> hyperaligned:")
print(f"   1st half decoded: {np.mean(_bh['anat'][0]):.1f} -> "
      f"{np.mean(_bh['hyper'][0]):.1f} %;  2nd half decoded: "
      f"{np.mean(_bh['anat'][1]):.1f} -> {np.mean(_bh['hyper'][1]):.1f} %")

# %% [prose] half.cap

# %% [markdown]
# ### 4.5 Discussion

# %% [prose] help.p

# %% [prose] clfs.p

# %% [prose] why.p

# %% [markdown]
# ### 4.6 Optional analysis: dimensionality of the common space

# %% [prose] dimp.p

# %%
_rows = [[f"k = {k}", *[f"{v:.1f}" for v in B["reduced_mean_bigrois"][str(k)]]]
         for k in B["k_list"]]
_rows.append(["all V (full)", *[f"{v:.1f}" for v in B["hyper_mean_bigrois"]]])
_rows.append(["anatomical", *[f"{v:.1f}" for v in B["anat_mean_bigrois"]]])
show_table(pd.DataFrame(_rows, columns=["Common space", *CLF_NAMES]))
print("six most accurate hyperaligned ROIs, accuracy at k = 20 minus all voxels (pp):")
_d6 = [f"{r['roi']} {r['k20'] - r['full']:+.1f}" for r in B["top6_hyper_k20_vs_full"]]
print("   " + ", ".join(_d6[:3]) + ",\n   " + ", ".join(_d6[3:]))

# %% [prose] dimt.cap

# %%
big = N_VOX > max(K_LIST)          # ROIs with V > 80, so that every k is defined
nk = len(K_LIST)
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
for j in range(len(CLF_NAMES)):
    dx = (j - (len(CLF_NAMES) - 1) / 2) * 0.05     # small offset: coinciding markers visible
    ax[0].plot(np.arange(nk + 1) + dx,
               list(np.nanmean(R_[big][:, :, :, j], axis=(0, 2))) + [h[big, j].mean()],
               marker="o", ms=4.5, lw=1.3, color=COL[j], label=CLF_NAMES[j])
    ax[0].plot([nk + 1 + dx], [a[big, j].mean()], marker="s", ms=5, ls="none", color=COL[j])
ax[0].axvline(nk + 0.5, color="0.8", lw=0.8)
ax[0].set_xticks(list(range(nk + 2)))
ax[0].set_xticklabels([str(k) for k in K_LIST] + ["all V", "anatomical\n(no alignment)"])
ax[0].axhline(CH7, color="k", ls="--", lw=0.8)
ax[0].set_xlabel("Common-space dimensions k (top PCs of the training template)")
ax[0].set_ylabel("Accuracy (%)")
ax[0].set_title(f"Mean over the {int(big.sum())} ROIs with V > {max(K_LIST)} "
                "(dashed: chance 14.29 %)", fontsize=10)
ax[0].set_ylim(10, 72)
ax[0].legend(fontsize=7.5, frameon=False, ncol=4, loc="lower center",
             bbox_to_anchor=(0.5, 0.06))
ax[0].spines[["top", "right"]].set_visible(False)
ax[1].spines[["top", "right"]].set_visible(False)
for k in np.argsort(-h.mean(1))[:6]:
    ax[1].plot(range(nk + 1), list(np.nanmean(R_[k], axis=(1, 2))) + [h[k].mean()],
               marker="o", label=LABELS[k])
ax[1].set_xticks(range(nk + 1))
ax[1].set_xticklabels([str(k) for k in K_LIST] + ["all V"])
ax[1].set_xlabel("Common-space dimensions k")
ax[1].set_ylabel("Accuracy, mean of 7 classifiers (%)")
ax[1].set_title("Six best hyperaligned ROIs", fontsize=10)
ax[1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
show_figure(fig, "part2b_dimensionality.png", dpi=170, width=1000)

# %% [prose] dimf.cap

# %% [prose] h5seed.h2 h5seed.p

# %% [markdown]
# ### 5.1 Code
#
# - `top_rois` ranks the ROIs by their Task-A accuracy (mean over the 7 classifiers).
# - `load_whole_brain` reads the subjects one at a time and keeps, for each subject, only the
#   z-scored time courses of its valid voxels (finite, mean signal above 6000, non-constant).
#   The result is one array Z (18 × 280 × U, float32, about 1.5 GB) on the union U of all
#   voxels that are valid in at least one subject, with zeros where a subject has no valid
#   signal, and a mask `valid` (18 × U). Peak memory of this step is about 2.5 GB.
# - `seed_maps` computes the subject-level maps. FC: correlation of the subject's seed with
#   the subject's own voxels. ISFC: correlation of the subject's seed with the z-scored mean
#   of the other subjects at every voxel (leave-one-out), where at least 10 of the other 17
#   subjects have valid signal. This is the mask stored in the course's `avg` files.
# - `plain_mean` averages the subject maps with an arithmetic mean of r, as the course helper
#   does (`nanmean(csub_2, 2)` in `isfc_p2m_nifti.m`). `fisher_mean` (mean after Fisher
#   z-transformation) is used for comparison only.
# - `helper_maps` is a line-by-line port of `isfc_p2m_nifti.m`. It works on the raw subject
#   files, independently of `load_whole_brain` and `seed_maps`, and is used to validate them
#   for one seed. Its FC branch (`within = 1`) needs nothing else. Its ISFC branch
#   (`within = 0`) reads the leave-one-out averages that the course precomputed (folder
#   `avg` next to the subject files). That folder is not needed for the analysis; if it is
#   not there, the ISFC comparison is skipped.
# - `probe_cols` returns the 3 × 3 × 3 voxel cube centred on a template coordinate; these
#   cubes are used only to put numbers on the verbal description of the maps.

# %%
MIN_OTHERS = 10      # ISFC: a voxel must be valid in at least this many OTHER subjects
MIN_VALID = 10       # both group maps: voxels valid in at least this many of the 18 subjects
N_TOP = 3
DISPLAY_THR = 0.15   # maps are displayed for |r| > 0.15
EXTRA_SEEDS = ["08 PREC_L"]    # supplementary non-auditory seed (best default-mode ROI)
# Approximate anatomical probe locations (template mm); FC / ISFC are averaged over the
# 3 x 3 x 3 voxel cube (9 mm) centred on each coordinate.
PROBES = {"insula L": (-39, 3, 0), "insula R": (39, 3, 0),
          "SMA / medial frontal": (0, -3, 57), "anterior cingulate": (0, 21, 33),
          "sensorimotor L": (-39, -24, 57), "thalamus L": (-12, -18, 6),
          "thalamus R": (12, -18, 6), "cerebellum L": (-21, -60, -27),
          "cerebellum R": (21, -60, -27), "IFG L": (-48, 21, 12),
          "precuneus": (0, -57, 33), "medial prefrontal": (0, 54, 6)}


def top_rois(acc):
    """Indices of the N_TOP ROIs with the highest mean Task-A accuracy, and all scores."""
    score = acc.mean(axis=(1, 2))
    return [int(k) for k in np.argsort(-score)[:N_TOP]], score


def load_whole_brain():
    """Z-scored data of all subjects on the union of valid voxels.

    Returns Z (18, 280, U) float32 with invalid entries set to 0, valid (18, U) bool and
    union (U,) flat voxel indices.  Subjects are read one at a time; only the z-scored valid
    voxels of each subject are kept until the union is known.
    """
    zs, masks = [], []
    for f in SUBJECT_FILES:
        x = load_subject(f)
        xn = np.nan_to_num(x)
        with np.errstate(invalid="ignore"):
            m = np.isfinite(x).all(0) & (xn.mean(0) > SIGNAL_THRESHOLD) & (xn.std(0) > 0)
        del xn
        zs.append(zscore(x[:, np.flatnonzero(m)], axis=0))
        masks.append(m)
        del x
    masks = np.stack(masks)
    union = np.flatnonzero(masks.any(0))
    valid = masks[:, union]
    Z = np.zeros((len(zs), N_TR, len(union)), dtype=np.float32)
    for i in range(len(zs)):
        Z[i][:, valid[i]] = zs[i]
        zs[i] = None                                   # free the per-subject copy
    return Z, valid, union


def seed_maps(Z, valid, seed_cols):
    """Subject-level maps (fc, isfc), each (18, U) with NaN where undefined."""
    n_sub, T, U = Z.shape
    S, n_valid = Z.sum(0), valid.sum(0)
    fc = np.full((n_sub, U), np.nan, np.float32)
    isfc = np.full((n_sub, U), np.nan, np.float32)
    for i in range(n_sub):
        cols = seed_cols[valid[i, seed_cols]]
        seed = zscore(Z[i][:, cols].mean(1))                         # (280,)
        fc[i, valid[i]] = seed @ Z[i][:, valid[i]] / T               # Pearson r (z-scored)
        n_oth = n_valid - valid[i]
        ok = n_oth >= MIN_OTHERS
        # z-scored mean of the other subjects at every voxel
        others = zscore((S[:, ok] - Z[i][:, ok]) / n_oth[ok], axis=0)
        isfc[i, ok] = seed @ others / T
    return fc, isfc

# %%
def plain_mean(r):
    """Arithmetic mean of r over subjects, as nanmean(csub_2, 2) in isfc_p2m_nifti.m."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns -> NaN
        return np.nanmean(r.astype(np.float64), axis=0)


def helper_maps(seed_flat, avg_files=None):
    """Line-by-line port of isfc_p2m_nifti.m, working on the raw files.

    seed_flat: flat indices of the seed voxels on the 61 x 73 x 61 grid (`roimask`).
    FC is the helper's branch within = 1.  ISFC (within = 0) is computed if `avg_files` is
    given: the course's precomputed leave-one-out averages (folder `avg`, one file per subject
    in the sorted order of the subject files; 280 rows of data, last row = voxel mask).
    Returns (fc_map, isfc_map or None) on the full grid, NaN where no subject has a value.
    The variable names of the MATLAB helper are given in the comments.
    """
    n_vox = int(np.prod(GRID))
    c_fc = np.full((n_vox, N_SUB), np.nan)                       # csub_2, within = 1
    c_isfc = np.full((n_vox, N_SUB), np.nan) if avg_files else None    # csub_2, within = 0

    def unit(x):                                 # B = (B - mean(B)) / (sqrt(Ns - 1) * std(B))
        return (x - x.mean(0)) / (np.sqrt(len(x) - 1) * x.std(0, ddof=1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for isub, f in enumerate(SUBJECT_FILES):
            bold = load_subject(f)                               # bold_one_temp (280, Nvox)
            mask_single = bold.mean(0, dtype=np.float64) > SIGNAL_THRESHOLD   # mean > Bth
            roi = bold[:, seed_flat].astype(np.float64)          # ROI: seed voxels, z-scored
            roi[:, ~mask_single[seed_flat]] = np.nan
            roi = (roi - roi.mean(0)) / roi.std(0, ddof=1)
            b = unit(np.nanmean(roi, axis=1))                    # bold_one -> B
            good = np.flatnonzero(mask_single)                   # gg_avg = find(mask_single)
            c_fc[good, isub] = unit(bold[:, good].astype(np.float64)).T @ b           # cc
            if avg_files:
                bold_avg = loadmat(avg_files[isub])["bold_avg"]  # (281, Nvox), last row: mask
                good = np.flatnonzero(bold_avg[-1])              # gg_avg = find(mask_avg)
                c_isfc[good, isub] = unit(bold_avg[:-1][:, good].astype(np.float64)).T @ b
            del bold
        isfc_map = np.nanmean(c_isfc, axis=1) if avg_files else None
        return np.nanmean(c_fc, axis=1), isfc_map                # avg_map_dmn


def fisher_mean(r):
    """Mean over subjects after Fisher z-transformation (NaN where no subject has a value)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns -> NaN
        return np.tanh(np.nanmean(np.arctanh(np.clip(r, -0.999999, 0.999999)), axis=0))


def probe_cols(xyz, union):
    """Columns (into `union`) of the 3 x 3 x 3 voxel cube centred on template mm `xyz`."""
    inv = np.linalg.inv(TEMPLATE_IMG.affine)
    c = np.round(nib.affines.apply_affine(inv, xyz)).astype(int)
    off = np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    ijk = c + off
    ijk = ijk[np.all((ijk >= 0) & (ijk < np.array(GRID)), axis=1)]
    flat = np.ravel_multi_index(ijk.T, GRID, order="F")
    return np.searchsorted(union, flat[np.isin(flat, union)])


def to_img(values, union):
    """Vector on the union voxels -> NIfTI image on the template grid (NaN -> 0)."""
    vol = np.zeros(np.prod(GRID), dtype=np.float32)
    vol[union] = np.nan_to_num(values)
    return nib.Nifti1Image(vol.reshape(GRID, order="F"), TEMPLATE_IMG.affine)

# %% [markdown]
# The last cell of this section defines the plotting function used in Section 5.3: for one
# seed, FC (top) and ISFC (bottom), each as orthogonal views through the centre of the seed
# and as a row of axial slices, on the same colour scale.

# %%
def typographic_minus(fig):
    """Replace the ASCII hyphen of negative numbers (cut labels, colour-bar ticks) by a
    real minus sign."""
    fig.canvas.draw()
    for ax in fig.axes:                                   # colour bars: freeze the tick labels
        ticks = [t.get_text() for t in ax.get_yticklabels()]
        if any(re.search(r"-\d", t) for t in ticks):
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([re.sub(r"-(?=\d)", MINUS, t) for t in ticks])
    for t in fig.findobj(mtext.Text):
        if re.search(r"-\d", t.get_text()):
            t.set_text(re.sub(r"-(?=\d)", MINUS, t.get_text()))


def plot_seed_maps(k, width=720):
    """Multi-planar FC / ISFC figure of seed k (same colour scale for both maps)."""
    com = P3["seeds"][LABELS[k]]["seed_center_mm"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5),
                             gridspec_kw={"height_ratios": [1.5, 1, 1.5, 1]})
    kw = dict(bg_img=TEMPLATE_IMG, threshold=DISPLAY_THR, vmax=0.6, cmap="cold_hot",
              figure=fig, black_bg=True)
    for r, name in enumerate(("FC", "ISFC")):
        long = ("FC – within-subject functional connectivity" if name == "FC"
                else "ISFC – inter-subject functional correlation")
        disp = plotting.plot_stat_map(imgs[(k, name)], display_mode="ortho",
                                      cut_coords=[round(c) for c in com], axes=axes[2 * r],
                                      draw_cross=True, annotate=True, **kw)
        disp.title(f"{long}   (seed: {LABELS[k]})", size=20)
        plotting.plot_stat_map(imgs[(k, name)], display_mode="z",
                               cut_coords=[-24, -9, 6, 21, 36, 51, 63], axes=axes[2 * r + 1],
                               colorbar=False, **kw)
    path = FIG_DIR / f"part3_maps_{LABELS[k].replace(' ', '_')}.png"
    typographic_minus(fig)
    fig.savefig(path, dpi=170, facecolor="k")
    plt.close(fig)
    im = PILImage.open(path).convert("RGB")                # trim the empty black margins
    box = im.point(lambda v: 255 if v > 12 else 0).getbbox()
    im.crop((max(box[0] - 20, 0), max(box[1] - 20, 0), min(box[2] + 20, im.width),
             min(box[3] + 20, im.height))).save(path)
    display(Image(filename=str(path), width=width))

# %% [markdown]
# ### 5.2 Whole-brain data and maps
#
# The first cell loads the whole-brain data (about half a minute). The second cell computes the
# FC and ISFC maps of the three top seeds and of the supplementary precuneus seed, restricts
# both group maps to the common mask (voxels that are valid in at least 10 of the 18
# subjects), writes them as NIfTI files to the work directory, and collects the summary
# numbers of each seed (about half a minute). For the first seed the maps are also computed
# with the port of the course helper from the raw files (about one more minute) and compared.

# %%
_t = time.time()
Z, valid, union = load_whole_brain()
keep = valid.sum(0) >= MIN_VALID               # common FC / ISFC mask: valid in >= 10 subjects
TIMINGS["5.2 whole-brain loading"] = time.time() - _t
print(f"Z: {Z.shape}, {Z.dtype}, {Z.nbytes / 1e9:.2f} GB")
print(f"voxels valid in at least one subject: {len(union):,}; valid in all {N_SUB}: "
      f"{int(valid.all(0).sum()):,}")
print(f"common map mask (valid in >= {MIN_VALID} subjects): {int(keep.sum()):,} voxels")
print(f"time: {TIMINGS['5.2 whole-brain loading']:.0f} s")

# %%
_t = time.time()
top, score = top_rois(ACC_A)
rank = [int(k) for k in np.argsort(-score)]
seeds_all = top + [LABELS.index(e) for e in EXTRA_SEEDS]
P3 = {"min_valid_subjects": MIN_VALID, "min_other_subjects": MIN_OTHERS,
      "n_voxels_map_mask": int(keep.sum()),
      "n_voxels_valid_any_subject": int(len(union)),
      "n_voxels_valid_all_subjects": int(valid.all(0).sum()),
      "threshold_display": DISPLAY_THR,
      "extra_seed_rank": {e: rank.index(LABELS.index(e)) + 1 for e in EXTRA_SEEDS},
      "extra_seeds": EXTRA_SEEDS, "top_rois": [LABELS[k] for k in top],
      "mean_acc_pct": [float(score[k] * 100) for k in top],
      "rank4": [LABELS[rank[3]], float(np.sort(score)[-4] * 100)], "seeds": {}}
print("top-3 ROIs of Task A (mean over the 7 classifiers):")
print("   " + ", ".join(f"{LABELS[k]} ({score[k] * 100:.1f} %)" for k in top)
      + f"; rank 4: {P3['rank4'][0]} ({P3['rank4'][1]:.1f} %)")
print("top 3 by best single classifier:", ", ".join(A["top3_by_best_single_clf"]))
print("top 3 by the linear SVM        :", ", ".join(A["top3_by_clf"]["Linear SVM"]))
print(f"\nvoxels beyond the display threshold (|r| > {DISPLAY_THR}), FC / ISFC:")
print(f"   {'seed':12s} {'V':>4s}   {'r > ' + str(DISPLAY_THR):>16s}   "
      f"{'r < ' + MINUS + str(DISPLAY_THR):>16s}")

imgs = {}
for k in seeds_all:
    flat = voi_to_flat_index(VOIS[k][1])
    seed_cols = np.searchsorted(union, flat[np.isin(flat, union)])
    fc_s, isfc_s = seed_maps(Z, valid, seed_cols)
    fc, isfc = plain_mean(fc_s), plain_mean(isfc_s)     # helper convention: mean of r
    fc[~keep] = np.nan                           # the same voxels in both maps
    isfc[~keep] = np.nan
    assert np.array_equal(np.isfinite(fc), keep) and np.array_equal(np.isfinite(isfc), keep)
    # for comparison only: largest difference to the mean after Fisher z-transformation
    d_plain = max(float(np.max(np.abs(fisher_mean(m_)[keep] - g_[keep])))
                  for m_, g_ in ((fc_s, fc), (isfc_s, isfc)))
    if k == top[0]:           # validation against the port of the course helper (raw files)
        avg_files = sorted(glob.glob(str(subject_dir(DATA) / "avg" / "*.mat")))
        assert len(avg_files) in (0, N_SUB), "the avg folder is incomplete"
        ref_fc, ref_isfc = helper_maps(flat, avg_files)
        assert np.isfinite(ref_fc[union][keep]).all()
        P3["helper_check"] = {
            "seed": LABELS[k], "n_voxels": int(keep.sum()),
            "max_abs_diff_fc": float(np.max(np.abs(ref_fc[union][keep] - fc[keep])))}
        if avg_files:         # is the helper's ISFC map defined on the same voxels?
            _same = np.array_equal(np.flatnonzero(np.isfinite(ref_isfc)), union[keep])
            P3["helper_check"].update(
                isfc_same_voxels=bool(_same),
                max_abs_diff_isfc=float(np.max(np.abs(ref_isfc[union][keep] - isfc[keep]))))
        del ref_fc, ref_isfc
    tag = LABELS[k].replace(" ", "_")
    for name, m_ in (("FC", fc), ("ISFC", isfc)):
        imgs[(k, name)] = to_img(m_, union)
        nib.save(imgs[(k, name)], str(WORK / f"map_{name}_{tag}.nii.gz"))
    outside = np.ones(len(union), bool)
    outside[seed_cols] = False
    both = keep & outside                        # map voxels outside the seed
    ijk = np.array(np.unravel_index(flat, GRID, order="F")).T
    com = nib.affines.apply_affine(TEMPLATE_IMG.affine, ijk.mean(0))
    P3["seeds"][LABELS[k]] = {
        "n_seed_voxels": int(len(seed_cols)), "seed_center_mm": [float(c) for c in com],
        "fc_mean_r": float(fc[both].mean()), "isfc_mean_r": float(isfc[both].mean()),
        "fc_median_r": float(np.median(fc[both])),
        "isfc_median_r": float(np.median(isfc[both])),
        "fc_frac_gt_0.2": float((fc[both] > 0.2).mean()),
        "isfc_frac_gt_0.2": float((isfc[both] > 0.2).mean()),
        "fc_frac_lt_-0.1": float((fc[both] < -0.1).mean()),
        "isfc_frac_lt_-0.1": float((isfc[both] < -0.1).mean()),
        "fc_max_r_outside_seed": float(fc[both].max()),
        "isfc_max_r_outside_seed": float(isfc[both].max()),
        "spatial_corr_fc_isfc": float(np.corrcoef(fc[both], isfc[both])[0, 1]),
        "isfc_within_seed_mean_r": float(np.nanmean(isfc[seed_cols])),
        "fc_within_seed_mean_r": float(np.nanmean(fc[seed_cols])),
        "n_seed_voxels_in_mask": int(keep[seed_cols].sum()),
        "n_fc_gt_thr": int((fc[keep] > DISPLAY_THR).sum()),
        "n_isfc_gt_thr": int((isfc[keep] > DISPLAY_THR).sum()),
        "n_fc_lt_neg_thr": int((fc[keep] < -DISPLAY_THR).sum()),
        "n_isfc_lt_neg_thr": int((isfc[keep] < -DISPLAY_THR).sum()),
        "max_abs_diff_fisher_vs_plain_mean": d_plain,
        "probes": {name: {"mm": list(xyz), "n_vox": int(keep[c].sum()),
                          "fc": float(np.nanmean(fc[c])), "isfc": float(np.nanmean(isfc[c]))}
                   for name, xyz in PROBES.items() for c in [probe_cols(xyz, union)]},
    }
    _s = P3["seeds"][LABELS[k]]
    print(f"   {LABELS[k]:12s} {_s['n_seed_voxels']:4d}   "
          f"{_s['n_fc_gt_thr']:7,} / {_s['n_isfc_gt_thr']:6,}   "
          f"{_s['n_fc_lt_neg_thr']:7,} / {_s['n_isfc_lt_neg_thr']:6,}")
save_json(P3, "part3_summary.json")
_dmax = max(s["max_abs_diff_fisher_vs_plain_mean"] for s in P3["seeds"].values())
print(f"largest difference between the plain mean of r (used) and the Fisher-z mean: "
      f"{_dmax:.3f}")
_hc = P3["helper_check"]
print(f"seed {_hc['seed']}, maps vs. the port of isfc_p2m_nifti.m over {_hc['n_voxels']:,} "
      f"voxels:\n   FC: largest difference {_hc['max_abs_diff_fc']:.1e}")
if "max_abs_diff_isfc" in _hc:
    print(f"   ISFC (course avg files): largest difference {_hc['max_abs_diff_isfc']:.1e}; "
          f"defined on the same voxels: {_hc['isfc_same_voxels']}")
else:
    print("   ISFC: not compared (no avg folder next to the subject files)")

print("\nFC / ISFC at the probe locations (mean over the 3 x 3 x 3 voxel cube), per seed:")
show_table(pd.DataFrame(
    [[name, ", ".join(str(c).replace("-", MINUS) for c in xyz),
      *[f"{signed(s['probes'][name]['fc'], '.2f')} / {signed(s['probes'][name]['isfc'], '.2f')}"
        for s in P3["seeds"].values()]] for name, xyz in PROBES.items()],
    columns=["Probe location", "x, y, z (mm)", *P3["seeds"]]))

# the whole-brain array is not needed any more
del Z, fc_s, isfc_s
gc.collect()
TIMINGS["5.2 FC / ISFC maps"] = time.time() - _t
print(f"time: {TIMINGS['5.2 FC / ISFC maps']:.0f} s")

# %% [markdown]
# ### 5.3 Results

# %%
show_table(pd.DataFrame(
    [[name, s["n_seed_voxels"],
      f"{s['fc_within_seed_mean_r']:.2f} / {s['isfc_within_seed_mean_r']:.2f}",
      f"{s['fc_mean_r']:.3f} / {s['isfc_mean_r']:.3f}",
      f"{100 * s['fc_frac_gt_0.2']:.1f} % / {100 * s['isfc_frac_gt_0.2']:.1f} %",
      f"{s['fc_max_r_outside_seed']:.2f} / {s['isfc_max_r_outside_seed']:.2f}",
      f"{s['spatial_corr_fc_isfc']:.2f}"] for name, s in P3["seeds"].items()],
    columns=["Seed", "V", "Mean r inside seed (FC / ISFC)",
             "Mean r, rest of brain (FC / ISFC)", "Voxels with r &gt; 0.2 (FC / ISFC)",
             "Max r outside seed (FC / ISFC)", "Spatial corr. FC vs ISFC"]))

# %% [prose] fc.cap

# %%
_t = time.time()
plot_seed_maps(seeds_all[0])

# %% [prose] m1.cap

# %%
plot_seed_maps(seeds_all[1])

# %% [prose] m2.cap

# %%
plot_seed_maps(seeds_all[2])

# %% [prose] m3.cap

# %%
plot_seed_maps(seeds_all[3])
TIMINGS["5.3 map figures"] = time.time() - _t

# %% [prose] m4.cap

# %% [markdown]
# ### 5.4 Discussion

# %% [prose] a.h3 a.p

# %% [prose] b.h3 b.p

# %% [prose] c.h3 c.p

# %% [prose] limits.h2 limits.ul

# %% [prose] refs.h2 refs.ol

# %% [prose] appA.h2

# %%
show_table(pd.DataFrame(
    [[r["roi"], r["n_vox"], *acc_sem_cells(r["acc"], r["sem"])] for r in A["table"]],
    columns=["ROI", "V", *CLF_NAMES]))

# %% [prose] appB.h2 appB.p[0]

# %%
show_table(pd.DataFrame(
    [[r["roi"], r["n_vox"], *acc_sem_cells(r["anat"], r["anat_sem"])] for r in B["table"]],
    columns=["ROI", "V", *CLF_NAMES]))

# %% [prose] appB.p[1]

# %%
show_table(pd.DataFrame(
    [[r["roi"], r["n_vox"], *acc_sem_cells(r["hyper"], r["hyper_sem"]),
      signed(r["gain"], "+.1f")] for r in B["table"]],
    columns=["ROI", "V", *CLF_NAMES, "Mean gain (pp)"]))

# %% [markdown]
# ## Run summary
#
# Run time of the heavy cells and of the whole notebook, and the peak memory of the notebook
# process (worker processes of the decoding analyses not included).

# %%
import resource

TIMINGS["total (all cells)"] = time.time() - T_START
_steps = [f"{k:26s}{v:5.0f} s" for k, v in TIMINGS.items()]
_half = -(-len(_steps) // 2)
for _i in range(_half):                          # two columns
    print(_steps[_i] + "        " + (_steps[_i + _half] if _i + _half < len(_steps) else ""))
_peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # Linux: kilobytes
print(f"peak memory of the notebook process: {_peak_gb:.1f} GB; CPU cores: {os.cpu_count()}")
_files = [p.name for p in WORK.iterdir() if p.is_file() and p.name != "roi_data.npz"]
print(f"result files in the work directory: {len(_files)} ("
      + ", ".join(f"{sum(f.endswith(ext) for f in _files)} {ext}"
                  for ext in (".json", ".npz", ".nii.gz")) + ")")
save_json({"timings_s": TIMINGS, "peak_rss_gb": _peak_gb, "n_cpu": os.cpu_count()},
          "run_info.json")
