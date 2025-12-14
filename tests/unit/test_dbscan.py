import numpy as np
import pytest

from rice_ml.unsupervised_learning.dbscan import DBSCAN


# ==========================================================
# Basic functionality
# ==========================================================

def test_simple_two_clusters():
    rng = np.random.default_rng(0)

    X1 = rng.normal(loc=0.0, scale=0.1, size=(50, 2))
    X2 = rng.normal(loc=5.0, scale=0.1, size=(50, 2))
    X = np.vstack([X1, X2])

    model = DBSCAN(eps=0.3, min_samples=5)
    labels = model.fit_predict(X)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    assert len(unique_labels) == 2


def test_detects_noise():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [5.0, 5.0],  # isolated point
    ])

    model = DBSCAN(eps=0.3, min_samples=3)
    labels = model.fit_predict(X)

    assert -1 in labels


def test_all_noise():
    X = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
        [20.0, 20.0],
    ])

    model = DBSCAN(eps=0.1, min_samples=2)
    labels = model.fit_predict(X)

    assert np.all(labels == -1)


# ==========================================================
# Error handling
# ==========================================================

def test_invalid_eps():
    with pytest.raises(ValueError):
        DBSCAN(eps=0)


def test_invalid_min_samples():
    with pytest.raises(ValueError):
        DBSCAN(min_samples=0)


def test_non_2d_input():
    model = DBSCAN()
    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]))
