from __future__ import annotations
from typing import Literal, Union, Sequence
import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["KNNClassifier", "KNNRegressor"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _as2d_float(X, name="X"):
    arr = np.asarray(X, float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr

def _as1d(y, name="y"):
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    return arr

def _dist(XA, XB, metric):
    if metric == "euclidean":
        aa = np.sum(XA*XA, 1)[:, None]
        bb = np.sum(XB*XB, 1)[None, :]
        return np.sqrt(np.maximum(aa + bb - 2 * XA @ XB.T, 0.0))
    elif metric == "manhattan":
        return np.sum(np.abs(XA[:, None, :] - XB[None, :, :]), axis=2)
    else:
        raise ValueError("Bad metric.")

def _weights(dist, scheme, eps=1e-12):
    if scheme == "uniform":
        return np.ones_like(dist)
    zero = dist <= eps
    w = np.where(zero.any(1)[:,None],
                 zero.astype(float),
                 1.0 / np.maximum(dist, eps))
    return w


# -------------------------------------------------------------------
# Base
# -------------------------------------------------------------------

class _KNNBase:
    def __init__(self, n_neighbors=5, *, metric="euclidean", weights="uniform"):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if metric not in ("euclidean","manhattan"):
            raise ValueError("Bad metric.")
        if weights not in ("uniform","distance"):
            raise ValueError("Bad weights.")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights
        self._X = None
        self._y = None

    def fit(self, X: ArrayLike, y: ArrayLike):
        X = _as2d_float(X, "X")
        y = _as1d(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors exceeds sample count.")
        self._X, self._y = X, y
        return self

    # nearest neighbors (common)
    def kneighbors(self, X):
        if self._X is None:
            raise RuntimeError("Call fit first.")
        Xq = _as2d_float(X, "X")
        if Xq.shape[1] != self._X.shape[1]:
            raise ValueError("Feature mismatch.")
        D = _dist(Xq, self._X, self.metric)
        idx = np.argpartition(D, self.n_neighbors-1, axis=1)[:, :self.n_neighbors]
        # sort the selected neighbors
        sort = np.take_along_axis(D, idx, 1).argsort(1)
        idx = np.take_along_axis(idx, sort, 1)
        dist = np.take_along_axis(D, idx, 1)
        return dist, idx


# -------------------------------------------------------------------
# Classifier
# -------------------------------------------------------------------

class KNNClassifier(_KNNBase):
    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.unique(self._y)
        return self

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        w = _weights(dist, self.weights)
        labels = self._y[idx]                      # (nq,k)
        class_index = np.searchsorted(self.classes_, labels)

        # one-hot counting with weights
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        # accumulate weights per class using an eye-matrix trick
        proba = (np.eye(n_classes)[class_index] * w[..., None]).sum(axis=1)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == _as1d(y)))


# -------------------------------------------------------------------
# Regressor
# -------------------------------------------------------------------

class KNNRegressor(_KNNBase):
    def fit(self, X, y):
        X = _as2d_float(X, "X")
        y = _as1d(y, "y").astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Length mismatch.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("Too few samples.")
        self._X, self._y = X, y
        return self

    def predict(self, X):
        dist, idx = self.kneighbors(X)
        w = _weights(dist, self.weights)
        y_neighbors = self._y[idx]
        wsum = w.sum(1)
        ypred = (w * y_neighbors).sum(1) / np.maximum(wsum, 1e-12)
        return ypred

    def score(self, X, y):
        y_true = _as1d(y).astype(float)
        y_pred = self.predict(X)
        ss_res = ((y_true - y_pred)**2).sum()
        ss_tot = ((y_true - y_true.mean())**2).sum()
        if ss_tot == 0:
            raise ValueError("R^2 undefined for constant y.")
        return float(1 - ss_res/ss_tot)