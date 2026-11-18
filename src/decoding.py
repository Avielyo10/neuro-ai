"""Classifier definitions and leave-one-subject-out (LOSO) helpers shared by Parts 2A and 2B.

A *sample* is one story segment of one subject: the 20 TRs x V voxels response matrix,
flattened to a 20*V feature vector.  Each subject therefore contributes one sample per class.
"""
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

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


def make_classifiers() -> dict:
    """All hyper-parameters are fixed a priori (never tuned on test data).

    KNN         Euclidean distance, K = 1, 2, 3 (ties in the K = 2 vote go to the lower label,
                scikit-learn's default).
    Linear SVM  libsvm SVC, linear kernel, C = 1, one-vs-one multi-class.
    Naive Bayes Gaussian, every (voxel, time point) feature independent.
    LDA / QDA   The feature count (20*V, up to ~15,000) far exceeds the number of training samples
                (17 subjects x 14 segments = 238; 17 per class), so the sample covariance is singular.
                Both are therefore fitted after a PCA rotation estimated on the training fold only
                (all components kept, i.e. lossless for the training data) and are regularised:
                LDA uses one pooled covariance with Ledoit-Wolf shrinkage; QDA uses one covariance
                per class, shrunk half-way to a scaled identity (shrinkage = 0.5; with 17 samples
                per class the automatic estimate is rank-deficient).
    """
    return {
        "KNN (K=1)": KNeighborsClassifier(1, algorithm="brute"),
        "KNN (K=2)": KNeighborsClassifier(2, algorithm="brute"),
        "KNN (K=3)": KNeighborsClassifier(3, algorithm="brute"),
        "Linear SVM": GramLinearSVC(C=1.0),
        "Naive Bayes": GaussianNB(),
        "LDA": make_pipeline(PCA(svd_solver="full"), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "QDA": make_pipeline(PCA(svd_solver="full"), QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.5)),
    }


def fit_score_all(Xtr, ytr, Xte, yte) -> np.ndarray:
    """Train every classifier on (Xtr, ytr); return test accuracies in CLF_NAMES order."""
    return np.array([clf.fit(Xtr, ytr).score(Xte, yte) for clf in make_classifiers().values()])


def segment_samples(z, seg_len=20):
    """(n_subjects, T, V) z-scored data -> group tensor (n_seg, seg_len, V, n_subjects)."""
    n_sub, T, V = z.shape
    return np.transpose(z.reshape(n_sub, T // seg_len, seg_len, V), (1, 2, 3, 0))


def loso_accuracy(tensor) -> np.ndarray:
    """LOSO cross-validation on a (n_seg, seg_len, V, n_subjects) group tensor.

    Each fold trains on (n_subjects - 1) x n_seg segment samples and predicts the segment
    identity of the n_seg samples of the held-out subject.
    Returns (n_subjects, n_classifiers) accuracies.
    """
    n_seg, seg_len, V, n_sub = tensor.shape
    X = np.transpose(tensor, (3, 0, 1, 2)).reshape(n_sub, n_seg, seg_len * V)  # (subj, sample, feature)
    y = np.arange(n_seg)
    acc = np.zeros((n_sub, len(CLF_NAMES)))
    for s in range(n_sub):
        train = np.delete(np.arange(n_sub), s)
        acc[s] = fit_score_all(X[train].reshape(-1, X.shape[2]), np.tile(y, n_sub - 1), X[s], y)
    return acc
