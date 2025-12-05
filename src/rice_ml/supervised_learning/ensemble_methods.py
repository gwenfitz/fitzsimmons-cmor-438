"""
Ensemble learning methods.

This module implements common ensemble techniques for supervised
classification using base learners implemented within the rice_ml package.
Supported ensembles include:

- Bagging (Bootstrap Aggregating)
- Voting (hard voting)
- Random Forest (tree ensemble with bootstrap sampling)

All ensemble models share a consistent API with fit(X, y) and predict(X).
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Optional
import numpy as np
from collections import Counter

from .decision_tree import DecisionTreeClassifier
from .knn import KNNClassifier


__all__ = [
    "BaggingClassifier",
    "VotingClassifier",
    "RandomForestClassifier",
]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _validate_inputs(X, y=None):
    """
    Validate input feature matrix X and (optional) label vector y.

    Returns
    -------
    X : ndarray
        Ensured to be 2D array.
    y : ndarray or None
        Ensured to be 1D or 2D numeric array.

    Raises
    ------
    ValueError
        If shapes do not align.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional; got {X.ndim}D.")

    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        return X, y

    return X


def _majority_vote(column: Iterable) -> int:
    """
    Compute the majority class in a vector of predicted labels.

    Ties are resolved by selecting the smallest class label (consistent rule).

    Parameters
    ----------
    column : iterable of labels

    Returns
    -------
    int
        Label chosen by majority vote.
    """
    counts = Counter(column)
    # Sort by: highest count, then lowest label
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


# ---------------------------------------------------------------------------
# Bagging Classifier
# ---------------------------------------------------------------------------

class BaggingClassifier:
    """
    Bagging ensemble for classification.

    Each estimator is trained on a bootstrap sample of the training data.

    Parameters
    ----------
    base_learner : Callable
        Constructor for a base classifier (e.g., DecisionTreeClassifier).
    n_estimators : int
        Number of base learners to train.
    max_samples : float
        Fraction of samples to draw for each bootstrap dataset.
    random_state : int or None
        Seed for reproducible sampling.
    """

    def __init__(
        self,
        base_learner: Callable = DecisionTreeClassifier,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.base_learner = base_learner
        self.n_estimators = int(n_estimators)
        self.max_samples = float(max_samples)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.models: List = []

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)
        n = X.shape[0]
        sample_size = max(1, int(self.max_samples * n))

        self.models = []
        for _ in range(self.n_estimators):
            idx = self._rng.choice(n, size=sample_size, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            model = self.base_learner()
            model.fit(X_boot, y_boot)
            self.models.append(model)

        return self

    def predict(self, X):
        X = _validate_inputs(X)
        all_preds = np.array([m.predict(X) for m in self.models])
        return np.array([_majority_vote(all_preds[:, j]) for j in range(X.shape[0])])


# ---------------------------------------------------------------------------
# Voting Classifier
# ---------------------------------------------------------------------------

class VotingClassifier:
    """
    Hard voting ensemble classifier.

    All models must implement fit(X, y) and predict(X).

    Parameters
    ----------
    models : list
        List of instantiated base models.
    """

    def __init__(self, models: List):
        self.models = models

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)
        for model in self.models:
            if hasattr(model, "fit"):
                model.fit(X, y)
        return self

    def predict(self, X):
        X = _validate_inputs(X)
        preds = np.array([model.predict(X) for model in self.models])
        return np.array([_majority_vote(preds[:, j]) for j in range(X.shape[0])])


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------

class RandomForestClassifier:
    """
    Random Forest classifier.

    Uses bootstrap samples and a collection of DecisionTreeClassifiers.
    This implementation uses your custom DecisionTreeClassifier without
    feature subsampling (for simplicity). Feature subsampling can be added
    later if needed.

    Parameters
    ----------
    n_estimators : int
        Number of decision trees.
    max_samples : float
        Fraction of training samples per tree.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = int(n_estimators)
        self.max_samples = float(max_samples)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.trees: List[DecisionTreeClassifier] = []

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)
        n = X.shape[0]
        sample_size = max(1, int(self.max_samples * n))

        self.trees = []
        for _ in range(self.n_estimators):
            idx = self._rng.choice(n, size=sample_size, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            tree = DecisionTreeClassifier()
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = _validate_inputs(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([_majority_vote(tree_preds[:, j]) for j in range(X.shape[0])])