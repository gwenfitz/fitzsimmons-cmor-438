import numpy as np
import pytest

from rice_ml.supervised_learning.ensemble_methods import (
    BaggingClassifier,
    VotingClassifier,
    RandomForestClassifier,
)

from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier
from rice_ml.supervised_learning.knn import KNNClassifier


# ---------------------------------------------------------------------------
# Dummy model for faster / isolated testing
# ---------------------------------------------------------------------------

class DummyClassifier:
    """
    A minimal deterministic classifier for testing ensembles.
    Always predicts the most frequent class seen during fit().
    """

    def __init__(self):
        self.most_common = None

    def fit(self, X, y):
        (values, counts) = np.unique(y, return_counts=True)
        self.most_common = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.most_common)


# ---------------------------------------------------------------------------
# BaggingClassifier tests
# ---------------------------------------------------------------------------

def test_bagging_basic():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 1, 0])  # mixed classes

    model = BaggingClassifier(base_learner=DummyClassifier, n_estimators=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    # Dummy always predicts the most common class (tie → lowest label)
    assert (preds == 0).all()


def test_bagging_bootstrap_sampling():
    X = np.arange(20).reshape(-1, 1)
    y = np.array([0] * 10 + [1] * 10)

    model = BaggingClassifier(base_learner=DummyClassifier, n_estimators=3, random_state=0)
    model.fit(X, y)

    # Each model should be a DummyClassifier instance
    assert len(model.models) == 3
    assert all(isinstance(m, DummyClassifier) for m in model.models)


# ---------------------------------------------------------------------------
# VotingClassifier tests
# ---------------------------------------------------------------------------

def test_voting_classifier_simple():
    X = np.array([[0], [0], [0]])
    y = np.array([1, 1, 1])

    m1 = DummyClassifier()
    m2 = DummyClassifier()
    m3 = DummyClassifier()

    vote = VotingClassifier([m1, m2, m3])
    vote.fit(X, y)
    preds = vote.predict(X)

    assert (preds == 1).all()


def test_voting_breaks_ties_by_smallest_label():
    X = np.array([[0], [0], [0]])
    y = np.array([0, 1, 2])

    # Build 3 models that always output different labels:
    class FixedPredictor:
        def __init__(self, value):
            self.value = value
        def fit(self, X, y): return self
        def predict(self, X): return np.full(X.shape[0], self.value)

    m1 = FixedPredictor(2)
    m2 = FixedPredictor(0)
    m3 = FixedPredictor(1)

    vote = VotingClassifier([m1, m2, m3])
    vote.fit(X, y)

    preds = vote.predict(X)

    # Tie among {0,1,2} → choose smallest = 0
    assert (preds == 0).all()


# ---------------------------------------------------------------------------
# RandomForestClassifier tests
# ---------------------------------------------------------------------------

def test_random_forest_basic():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 1, 0])

    rf = RandomForestClassifier(n_estimators=4, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)

    # Should return integer labels of correct shape
    assert preds.shape == (4,)
    assert preds.dtype.kind in {"i", "u"}  # int or uint


def test_random_forest_trees_created():
    X = np.arange(10).reshape(-1, 1)
    y = np.array([0, 1] * 5)

    rf = RandomForestClassifier(n_estimators=5, random_state=123)
    rf.fit(X, y)

    # Should contain exactly 5 trees
    assert len(rf.trees) == 5
    assert all(isinstance(t, DecisionTreeClassifier) for t in rf.trees)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

def test_shape_mismatch_error():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1])  # wrong length

    model = BaggingClassifier()

    with pytest.raises(ValueError):
        model.fit(X, y)