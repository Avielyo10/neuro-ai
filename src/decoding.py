"""Classifier definitions and leave-one-subject-out (LOSO) helpers shared by Parts 2A and 2B.

A *sample* is one story segment of one subject: the 20 TRs x V voxels response matrix,
flattened to a 20*V feature vector.  Each subject therefore contributes one sample per class.
"""
import warnings

import numpy as np
from sklearn.covariance import LedoitWolf
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

    KNN         Euclidean distance, K = 1, 2, 3, uniform votes.  A tied vote (frequent for K = 2,
                where the two neighbours often belong to different classes) goes to the lower class
                index, scikit-learn's default.
    Linear SVM  libsvm SVC, linear kernel, C = 1, one-vs-one multi-class.
    Naive Bayes Gaussian, every (voxel, time point) feature independent.
    LDA / QDA   The feature count (20*V, up to ~15,000) far exceeds the number of training samples
                (17 subjects x 14 segments = 238; 17 per class), so the sample covariance is singular.
                Both are therefore fitted after a PCA rotation estimated on the training fold only
                (all components kept; the rotated data have at most n_train - 1 = 237 non-trivial
                dimensions) and are regularised IN THAT SUBSPACE:
                LDA (class ShrinkageLDA) uses one pooled within-class covariance shrunk towards a
                scaled identity.  The Ledoit-Wolf intensity and the target scale (trace / q, q = number
                of PCA dimensions) are estimated from the PCA scores, i.e. they are those of the
                training subspace and NOT those of the 20*V-dimensional voxel space.  The PCA rotation
                is lossless for the training data, but the regulariser is not rotation-equivalent: the
                same Ledoit-Wolf formula applied in voxel space gives a much weaker shrinkage for wide
                ROIs, and a different accuracy (see supp_lda_sensitivity.py, which compares this
                model with exact voxel-space Ledoit-Wolf and with the equivalent of scikit-learn's
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto") on the raw features).
                Features are not rescaled before shrinkage (scikit-learn's shrinkage="auto"
                standardises features first, which after a PCA rotation whitens the components and
                removes the regularisation).
                QDA uses one covariance per class, shrunk half-way to a scaled identity
                (shrinkage = 0.5, again in the PCA subspace; with 17 samples per class the automatic
                estimate is rank-deficient).  Needs scikit-learn >= 1.8 (QDA `solver` / `shrinkage`).
    """
    return {
        "KNN (K=1)": KNeighborsClassifier(1, algorithm="brute"),
        "KNN (K=2)": KNeighborsClassifier(2, algorithm="brute"),
        "KNN (K=3)": KNeighborsClassifier(3, algorithm="brute"),
        "Linear SVM": GramLinearSVC(C=1.0),
        "Naive Bayes": GaussianNB(),
        "LDA": ShrinkageLDA(),
        "QDA": make_pipeline(PCA(svd_solver="full"), QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.5)),
    }


class ShrinkageLDA:
    """LDA with a pooled within-class covariance shrunk towards a scaled identity (Ledoit-Wolf),
    estimated in the subspace spanned by the training samples.

    Model: the training fold is rotated with a full PCA (q <= n_train components, float64); the
    pooled within-class covariance S of the PCA scores is replaced by (1 - a) S + a (tr S / q) I,
    with the Ledoit-Wolf intensity a computed from the q-dimensional scores.  Both a and the target
    scale tr S / q therefore refer to the training subspace, not to the p = 20*V voxel space
    (voxel-space Ledoit-Wolf would use tr S / p and a different, for wide ROIs much smaller, a).
    In voxel-space terms the model is a considerably stronger regulariser when p >> q; it is a
    sensible but specific choice, and the accuracy of LDA relative to the other classifiers depends
    on it (supp_lda_sensitivity.py).

    Within the model as defined, nothing is lost by working in the subspace: class means lie in the
    training span and the shrinkage target is isotropic, so components of a test sample outside the
    span add the same constant to every class score and cannot change the prediction.
    Equal class priors (the design is balanced).
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.pca = PCA(svd_solver="full").fit(X)
        Z = self.pca.transform(X)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([Z[y == c].mean(0) for c in self.classes_])
        resid = Z - self.means_[np.searchsorted(self.classes_, y)]
        lw = LedoitWolf(assume_centered=True).fit(resid)
        cov = lw.covariance_
        self.shrinkage_ = float(lw.shrinkage_)      # Ledoit-Wolf intensity in the PCA subspace (diagnostic only)
        self.coef_ = np.linalg.solve(cov, self.means_.T).T                 # (classes, dims)
        self.intercept_ = -0.5 * np.einsum("cd,cd->c", self.means_, self.coef_)  # equal priors
        return self

    def predict(self, X):
        Z = self.pca.transform(np.asarray(X, dtype=np.float64))
        return self.classes_[np.argmax(Z @ self.coef_.T + self.intercept_, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


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
