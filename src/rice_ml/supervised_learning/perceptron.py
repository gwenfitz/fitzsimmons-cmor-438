"""
Perceptron Classifier (Binary Classification)

This module implements the classic Perceptron learning algorithm
for linearly separable binary classification problems.

Features
--------
• Binary classification (0 / 1)
• Optional intercept (bias term)
• Online (stochastic) weight updates
• Early stopping when convergence is reached
• Sklearn-like API: fit / predict / score

References
----------
Rosenblatt, F. (1958). The Perceptron: A probabilistic model for information storage.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

__all__ = ["Perceptron"]


# ---------------------------------------------------------------------
# Input validation utilities
# ---------------------------------------------------------------------

def _validate_inputs(X, y: Optional[np.ndarray] = None):
    """
    Validate feature matrix X and optional target vector y.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,), optional

    Returns
    -------
    X : ndarray
    y : ndarray or None
    """
    X = np.asarray(X, dtype=float)

    if y is None:
        return X

    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # Enforce binary labels {0, 1}
    unique = np.unique(y)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("Perceptron supports binary labels {0, 1} only.")

    return X, y


# ---------------------------------------------------------------------
# Perceptron Classifier
# ---------------------------------------------------------------------

class Perceptron:
    """
    Binary Perceptron Classifier.

    Parameters
    ----------
    learning_rate : float
        Step size for weight updates.
    max_iter : int
        Maximum number of passes over the training data.
    fit_intercept : bool
        Whether to include a bias term.
    shuffle : bool
        Whether to shuffle samples each epoch.
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned weight vector.
    intercept_ : float
        Bias term.
    n_iter_ : int
        Number of epochs run.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the Perceptron model.

        The update rule is:
            w := w + η (y - ŷ) x

        where ŷ ∈ {0,1}.
        """
        X, y = _validate_inputs(X, y)
        X_aug = self._add_intercept(X)

        n_samples, n_features = X_aug.shape
        w = np.zeros(n_features)

        for epoch in range(self.max_iter):
            errors = 0

            indices = np.arange(n_samples)
            if self.shuffle:
                self._rng.shuffle(indices)

            for i in indices:
                xi = X_aug[i]
                yi = y[i]

                y_pred = 1 if np.dot(w, xi) >= 0 else 0
                update = self.learning_rate * (yi - y_pred)

                if update != 0:
                    w += update * xi
                    errors += 1

            if errors == 0:
                break

        self.n_iter_ = epoch + 1

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def decision_function(self, X):
        """
        Compute signed distance to the decision boundary.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(
                "This Perceptron instance is not fitted yet. "
                "Call 'fit(X, y)' before using this method."
            )

        X = _validate_inputs(X)
        return X @ self.coef_ + self.intercept_


    def predict(self, X):
        """
        Predict binary labels {0,1}.
        """
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def score(self, X, y):
        """
        Compute classification accuracy.
        """
        X, y = _validate_inputs(X, y)
        preds = self.predict(X)
        return float(np.mean(preds == y))
