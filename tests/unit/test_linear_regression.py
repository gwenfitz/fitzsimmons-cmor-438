import numpy as np
import pytest

from rice_ml.supervised_learning.linear_regression import LinearRegression


# ---------------------------------------------------------------------
# Helper synthetic dataset
# ---------------------------------------------------------------------

@pytest.fixture
def simple_data():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 2 * X.ravel() + 1
    return X, y


# ---------------------------------------------------------------------
# Closed-form regression tests
# ---------------------------------------------------------------------

def test_ols_fit(simple_data):
    X, y = simple_data
    model = LinearRegression().fit(X, y)

    assert np.isclose(model.intercept_, 1.0, atol=1e-6)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-6)

    preds = model.predict(X)
    assert np.allclose(preds, y)


def test_ridge_regularization(simple_data):
    X, y = simple_data
    model = LinearRegression(regularization=10.0).fit(X, y)

    assert model.coef_.size == 1
    assert model.intercept_ is not None


# ---------------------------------------------------------------------
# Gradient descent tests
# ---------------------------------------------------------------------

def test_gradient_descent_fit(simple_data):
    X, y = simple_data
    model = LinearRegression(use_gradient_descent=True, learning_rate=0.01).fit(X, y)

    assert pytest.approx(model.coef_[0], rel=0.05) == 2.0
    assert pytest.approx(model.intercept_, rel=0.05) == 1.0


# ---------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------

def test_metrics(simple_data):
    X, y = simple_data
    model = LinearRegression().fit(X, y)

    assert np.isclose(model.mse(X, y), 0.0)
    assert np.isclose(model.rmse(X, y), 0.0)
    assert np.isclose(model.mae(X, y), 0.0)
    assert np.isclose(model.score(X, y), 1.0)


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_mismatched_input_lengths():
    X = np.array([[1],[2],[3]], dtype=float)
    y = np.array([1,2])

    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)