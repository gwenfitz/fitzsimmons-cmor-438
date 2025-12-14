import numpy as np
import pytest

from rice_ml.unsupervised_learning.k_means_clustering import KMeans


# ---------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------

def test_fit_sets_attributes():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.1, 5.1],
    ])

    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X)

    assert model.cluster_centers_ is not None
    assert model.labels_ is not None
    assert model.inertia_ is not None
    assert model.cluster_centers_.shape == (2, 2)
    assert len(model.labels_) == len(X)


# ---------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------

def test_predict_matches_fit_labels():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.1, 5.1],
    ])

    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)

    preds = model.predict(X)

    assert preds.shape == model.labels_.shape
    assert np.array_equal(preds, model.labels_)


def test_predict_before_fit_raises():
    model = KMeans(n_clusters=2)
    X = np.array([[0.0, 0.0]])

    with pytest.raises(RuntimeError):
        model.predict(X)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def test_invalid_n_clusters():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)


def test_fit_with_non_2d_input_raises():
    X = np.array([1.0, 2.0, 3.0])

    model = KMeans(n_clusters=2)

    with pytest.raises(ValueError):
        model.fit(X)


def test_fit_with_empty_input_raises():
    X = np.empty((0, 2))

    model = KMeans(n_clusters=2)

    with pytest.raises(ValueError):
        model.fit(X)


# ---------------------------------------------------------------------
# Clustering correctness (simple case)
# ---------------------------------------------------------------------

def test_separable_clusters():
    """
    Two well-separated clusters should be identified correctly.
    """
    X = np.vstack([
        np.random.normal(loc=0.0, scale=0.1, size=(20, 2)),
        np.random.normal(loc=5.0, scale=0.1, size=(20, 2)),
    ])

    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X)

    labels = model.labels_

    # Expect two clusters
    assert len(np.unique(labels)) == 2

    # Centroids should be far apart
    c0, c1 = model.cluster_centers_
    dist = np.linalg.norm(c0 - c1)
    assert dist > 3.0


# ---------------------------------------------------------------------
# Inertia sanity check
# ---------------------------------------------------------------------

def test_inertia_non_negative():
    X = np.random.randn(30, 3)

    model = KMeans(n_clusters=3, random_state=0)
    model.fit(X)

    assert model.inertia_ >= 0.0
