import numpy as np
import pytest
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


# -------------------------- KNNClassifier --------------------------

def test_knn_classifier_predict_basic():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=1, metric="euclidean", weights="uniform").fit(X, y)

    # single prediction
    y_pred = clf.predict([[0.1, 0.1]])
    assert y_pred[0] in [0, 1]

    # probabilities sum to 1
    probs = clf.predict_proba([[0.1, 0.1]])
    assert np.isclose(probs.sum(), 1.0)


def test_knn_classifier_predict_multiple():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=1).fit(X, y)

    Xq = np.array([[0, 0], [1, 1]])
    y_pred = clf.predict(Xq)
    assert len(y_pred) == 2
    assert set(y_pred).issubset({0, 1})


def test_knn_classifier_score_perfect():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=1).fit(X, y)  # use 1 neighbor for perfect accuracy

    score = clf.score(X, y)
    assert np.isclose(score, 1.0)


def test_knn_classifier_n_neighbors_exceeds_samples():
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    # n_neighbors > samples should raise
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=3).fit(X, y)


def test_knn_classifier_manhattan_distance():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=1, metric="manhattan").fit(X, y)

    probs = clf.predict_proba([[0.5, 0.5]])
    assert np.isclose(probs.sum(), 1.0)


# -------------------------- KNNRegressor --------------------------

def test_knn_regressor_predict_basic():
    X = np.array([[0], [1], [2], [3]], float)
    y = np.array([0.0, 1.0, 1.5, 3.0])

    # Test both uniform and distance weighting
    reg_uniform = KNNRegressor(n_neighbors=2, weights="uniform").fit(X, y)
    y_pred_uniform = reg_uniform.predict([[1.5]])
    assert 1.0 <= y_pred_uniform[0] <= 2.0  # prediction is between nearest neighbors

    reg_distance = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
    y_pred_distance = reg_distance.predict([[1.5]])
    # weighted average falls between nearest neighbors
    assert 1.0 <= y_pred_distance[0] <= 1.5


def test_knn_regressor_score_r2_perfect():
    X = np.array([[0], [1], [2], [3]], float)
    y = np.array([0.0, 1.0, 1.5, 3.0])
    reg = KNNRegressor(n_neighbors=1).fit(X, y)

    score = reg.score(X, y)
    assert np.isclose(score, 1.0)


def test_knn_regressor_weights_output_shape():
    X = np.array([[0], [1], [2]], float)
    y = np.array([0.0, 1.0, 2.0])

    reg_uniform = KNNRegressor(n_neighbors=2, weights="uniform").fit(X, y)
    reg_distance = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)

    pred_uniform = reg_uniform.predict([[0.5]])
    pred_distance = reg_distance.predict([[0.5]])

    # Just ensure numeric output and correct shape
    assert pred_uniform.shape == (1,)
    assert pred_distance.shape == (1,)


def test_knn_regressor_invalid_targets():
    X = np.array([[0], [1]])
    y = np.array(["a", "b"])  # non-numeric

    # Should raise ValueError when converting non-numeric target
    with pytest.raises(ValueError):
        KNNRegressor(n_neighbors=1).fit(X, y)