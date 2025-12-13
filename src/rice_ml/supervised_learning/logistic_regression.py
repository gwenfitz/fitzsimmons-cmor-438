"""
Logistic Regression (Binary Classification)

This module provides a robust, sklearn-style implementation of
binary logistic regression using gradient descent.

Supported Features
------------------
• Binary classification (labels {0,1})
• L2 regularization
• Optional intercept term
• Gradient descent optimizer with convergence checks
• Predict probabilities, class labels, decision function
• ROC curve + AUC implemented manually

"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


__all__ = ["LogisticRegression"]


# ----------------------------------------------------------------------
# Input validation utilities
# ----------------------------------------------------------------------

def _validate_binary_y(y: np.ndarray) -> None:
    """Ensure y contains only {0,1} labels."""
    unique_vals = np.unique(y)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError("LogisticRegression only supports binary labels {0,1}.")


def _as_2d_float(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return arr


def _prepare_X_y(X, y=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert X and y to NumPy and validate shape consistency.
    """
    X = _as_2d_float(X)

    if y is None:
        return X, None

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    _validate_binary_y(y)
    return X, y


# ----------------------------------------------------------------------
# Logistic Regression Model
# ----------------------------------------------------------------------

class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.

    Parameters
    ----------
    penalty : {"l2", "none"}
        Type of regularization to apply.

    C : float
        Inverse regularization strength (like sklearn).
        Smaller C → stronger regularization.

    fit_intercept : bool
        Whether to add a bias term.

    max_iter : int
        Maximum GD iterations.

    tol : float
        Convergence tolerance.

    learning_rate : float
        Step size for gradient descent.

    random_state : int or None
        Seed for internal RNG.

    solver : {"gd"}
        Currently gradient descent only.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 5000,
        tol: float = 1e-4,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
        solver: str = "gd",
    ):
        if penalty not in {"l2", "none"}:
            raise ValueError("penalty must be 'l2' or 'none'.")
        if C <= 0:
            raise ValueError("C must be positive.")
        if solver != "gd":
            raise ValueError("Only solver='gd' is supported.")

        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.solver = solver
        self.rng = np.random.default_rng(random_state)

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None


    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Stable sigmoid implementation."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def _compute_gradient(self, X, y, w):
        """
        Gradient of the logistic loss:
            grad = X^T (sigmoid(Xw) - y) / n
        """
        y_pred = self._sigmoid(X @ w)
        error = y_pred - y
        grad = (X.T @ error) / X.shape[0]

        # L2 regularization
        if self.penalty == "l2":
            reg_term = (1.0 / self.C) * w
            if self.fit_intercept:
                reg_term[0] = 0  # do not regularize bias
            grad += reg_term

        return np.clip(grad, -1e6, 1e6)


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = _prepare_X_y(X, y)
        X_aug = self._add_intercept(X)

        n_features = X_aug.shape[1]
        w = np.zeros(n_features)

        # Adaptive LR for stability
        lr = self.learning_rate
        if self.penalty == "l2":
            lr *= min(1.0, self.C)

        for _ in range(self.max_iter):
            grad = self._compute_gradient(X_aug, y, w)
            w_new = w - lr * grad

            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                break

            w = w_new

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self


    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def decision_function(self, X):
        X, _ = _prepare_X_y(X, None)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        z = self.decision_function(X)
        p1 = self._sigmoid(z)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def score(self, X, y):
        X, y = _prepare_X_y(X, y)
        return float(np.mean(self.predict(X) == y))


    # ------------------------------------------------------------------
    # ROC curve + AUC
    # ------------------------------------------------------------------

    def roc_curve(self, X, y, num_thresholds=200):
        """
        Compute ROC curve (FPR, TPR) and AUC manually without sklearn.
        """
        X, y = _prepare_X_y(X, y)
        probs = self.predict_proba(X)[:, 1]

        thresholds = np.linspace(1, 0, num_thresholds)
        tprs, fprs = [], []

        for t in thresholds:
            preds = (probs >= t).astype(int)

            TP = np.sum((preds == 1) & (y == 1))
            FP = np.sum((preds == 1) & (y == 0))
            FN = np.sum((preds == 0) & (y == 1))
            TN = np.sum((preds == 0) & (y == 0))

            TPR = TP / (TP + FN) if TP + FN > 0 else 0.0
            FPR = FP / (FP + TN) if FP + TN > 0 else 0.0

            tprs.append(TPR)
            fprs.append(FPR)

        auc = np.trapz(tprs, fprs)
        return np.array(fprs), np.array(tprs), float(auc)