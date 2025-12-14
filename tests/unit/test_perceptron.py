import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def linearly_separable_data():
    """
    Simple linearly separable dataset.
    """
    X = np.array([
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [2, 1],
        [1, 2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_fit_predict(linearly_separable_data):
    X, y = linearly_separable_data
    model = Perceptron(max_iter=1000).fit(X, y)

    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_accuracy(linearly_separable_data):
    X, y = linearly_separable_data
    model = Perceptron().fit(X, y)

    acc = model.score(X, y)
    assert acc == 1.0


def test_coefficients_shape(linearly_separable_data):
    X, y = linearly_separable_data
    model = Perceptron().fit(X, y)

    assert model.coef_.shape == (X.shape[1],)
    assert isinstance(model.intercept_, float)


def test_invalid_labels():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])  # invalid label

    with pytest.raises(ValueError):
        Perceptron().fit(X, y)


def test_predict_before_fit():
    model = Perceptron()
    X = np.array([[0, 0]])

    with pytest.raises(RuntimeError):
        model.predict(X)
