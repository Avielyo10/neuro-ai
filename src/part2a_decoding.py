"""Part 2 / Task A - classifier benchmarking in the original (anatomically aligned) voxel space.

For each of the 52 ROIs:
  * every voxel time course is z-scored within subject, and the 280 x V subject matrix is reshaped
    to 14 x 20 x V (14 story segments x 20 TRs x V voxels); stacking the 18 subjects gives the
    4-D group tensor 14 x 20 x V x 18;
  * LOSO cross-validation: train on the 17 x 14 segment samples of 17 subjects and predict which of
    the 14 segments each 20 x V sample of the held-out subject belongs to (chance = 1/14 = 7.14 %);
  * mean accuracy and SEM over the 18 folds for KNN (K = 1, 2, 3), linear SVM, Naive Bayes, LDA, QDA.

Run:  python src/part2a_decoding.py
"""
import time

import numpy as np
from joblib import Parallel, delayed

from common import N_SEG, RESULTS, SEG_LEN, load_roi_data, zscore
from decoding import CLF_NAMES, loso_accuracy, segment_samples


def run_roi(x):
    """x: (18, 280, V) raw ROI data -> (18 folds, 7 classifiers) accuracies."""
    tensor = segment_samples(zscore(x, axis=1), SEG_LEN)
    assert tensor.shape == (N_SEG, SEG_LEN, x.shape[2], 18)
    return loso_accuracy(tensor)


if __name__ == "__main__":
    roi = load_roi_data()
    t0 = time.time()
    order = np.argsort([-x.shape[2] for x in roi["X"]])     # big ROIs first for better scheduling
    out = Parallel(n_jobs=4, verbose=1)(delayed(run_roi)(roi["X"][k]) for k in order)
    acc = np.zeros((52, 18, len(CLF_NAMES)))
    acc[order] = np.stack(out)                              # (ROI, held-out subject, classifier)
    np.savez(RESULTS / "part2a_accuracy.npz", acc=acc, clf=CLF_NAMES, labels=roi["labels"],
             n_vox=[x.shape[2] for x in roi["X"]])
    print(f"done in {time.time() - t0:.0f} s")
    m = acc.mean(1) * 100
    print("mean over ROIs:", dict(zip(CLF_NAMES, m.mean(0).round(1))))
    for k in np.argsort(-m.mean(1)):
        print(f"{roi['labels'][k]:24s} V={roi['X'][k].shape[2]:4d}", m[k].round(1))
