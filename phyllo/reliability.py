"""Uncertainty, OOD scoring, calibration and selective-classification utilities.

Convention: every OOD score returned here is "higher = more in-distribution (more trustworthy)".

Methods (all post-hoc, no retraining needed — feasible on an RTX 3050):
  * MSP       max softmax probability                (Hendrycks & Gimpel, ICLR 2017)
  * MaxLogit  max un-normalised logit                (Hendrycks et al., ICML 2022)
  * Energy    T*logsumexp(logits/T)                  (Liu et al., NeurIPS 2020)
  * Entropy   negative predictive entropy
  * Mahalanobis  class-conditional Gaussians, tied covariance, on embeddings (Lee et al., NeurIPS 2018)
  * kNN       k-th nearest-neighbour cosine similarity on L2-normalised embeddings (Sun et al., ICML 2022)
Calibration: temperature scaling (Guo et al., ICML 2017), ECE.
Selective classification: risk-coverage curve, AURC (Geifman & El-Yaniv, NeurIPS 2017).
"""
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# ----------------------------------------------------------------- logit-based scores
def msp(logits, T=1.0):
    return softmax(logits / T, axis=1).max(1)


def max_logit(logits):
    return logits.max(1)


def energy(logits, T=1.0):
    return T * logsumexp(logits / T, axis=1)


def neg_entropy(logits, T=1.0):
    p = softmax(logits / T, axis=1)
    return (p * np.log(p + 1e-12)).sum(1)


# ----------------------------------------------------------------- feature-based scores
class Mahalanobis:
    """Class-conditional Gaussian with shared (tied) covariance + Ledoit-Wolf-style shrinkage.
    Shrinkage is essential: we have ~60 samples/class in 768 dimensions."""

    def __init__(self, shrinkage=0.1):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([X[y == c].mean(0) for c in self.classes_])
        centred = np.concatenate([X[y == c] - self.means_[i] for i, c in enumerate(self.classes_)])
        cov = centred.T @ centred / len(centred)
        mu = np.trace(cov) / cov.shape[0]
        cov = (1 - self.shrinkage) * cov + self.shrinkage * mu * np.eye(cov.shape[0])
        self.precision_ = np.linalg.inv(cov)
        return self

    def distances(self, X):
        X = np.asarray(X, np.float64)
        d = [np.einsum("ij,jk,ik->i", X - m, self.precision_, X - m) for m in self.means_]
        return np.stack(d, 1)                      # (n, n_classes) squared distances

    def score(self, X):
        return -self.distances(X).min(1)           # higher = more in-distribution

    def state(self):
        return {"means": self.means_, "precision": self.precision_, "classes": self.classes_}

    @classmethod
    def from_state(cls, means, precision, classes):
        m = cls()
        m.means_, m.precision_, m.classes_ = means, precision, classes
        return m


class KNNScore:
    def __init__(self, k=5):
        self.k = k

    @staticmethod
    def _norm(X):
        X = np.asarray(X, np.float32)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def fit(self, X, y=None):
        self.bank_ = self._norm(X)
        return self

    def score(self, X, exclude_self=False):
        sims = self._norm(X) @ self.bank_.T
        if exclude_self:
            np.fill_diagonal(sims, -np.inf)
        k = min(self.k, sims.shape[1] - (1 if exclude_self else 0))
        return np.sort(sims, axis=1)[:, -k]        # k-th largest cosine similarity


# ----------------------------------------------------------------- OOD metrics
def ood_metrics(score_id, score_ood):
    """In-distribution is the positive class. Returns AUROC, AUPR-In, AUPR-Out, FPR@95%TPR."""
    s = np.concatenate([score_id, score_ood])
    y = np.concatenate([np.ones(len(score_id)), np.zeros(len(score_ood))])
    fpr, tpr, _ = roc_curve(y, s)
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95, side="left")])
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "AUPR_in": float(average_precision_score(y, s)),
        "AUPR_out": float(average_precision_score(1 - y, -s)),
        "FPR@95TPR": fpr95,
    }


def threshold_at_tpr(score_id, tpr=0.95):
    """Score threshold that keeps `tpr` of in-distribution samples (fit on validation data)."""
    return float(np.quantile(score_id, 1 - tpr))


# ----------------------------------------------------------------- calibration
def nll(logits, y, T):
    p = softmax(logits / T, axis=1)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def fit_temperature(logits, y, bounds=(0.05, 20.0)):
    """NB: if the calibration set is classified perfectly, NLL decreases monotonically as T->0 and the
    optimum is unidentifiable; use constrained bounds in that case (see E4)."""
    res = minimize_scalar(lambda t: nll(logits, y, t), bounds=bounds, method="bounded")
    return float(res.x)


def ece(probs, y, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y[m]).mean(), conf[m].mean()
            e += m.mean() * abs(acc - c)
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "acc": float(acc), "conf": float(c)})
    return float(e), rows


# ----------------------------------------------------------------- selective classification
def risk_coverage(confidence, correct):
    order = np.argsort(-confidence)
    c = np.asarray(correct, float)[order]
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(1 - c) / np.arange(1, n + 1)
    return coverage, risk, float(np.mean(risk))   # AURC = mean risk over coverages
