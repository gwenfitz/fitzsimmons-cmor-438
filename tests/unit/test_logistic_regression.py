import numpy as np
import pytest
from rice_ml.supervised_learning.logistic_regression import LogisticRegression


# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture
def simple_binary_data():
    # Perfectly separable line: x > 0 → class 1
    X = np.array([[-2], [-1], [0], [1], [2]], float)
    y = np.array([0, 0, 0, 1, 1], float)
    return X, y


# ----------------------------------------------------------
# Tests
# ----------------------------------------------------------

def test_fit_shapes(simple_binary_data):
    X, y = simple_binary_data
    model = LogisticRegression().fit(X, y)

    assert model.coef_.shape == (1,)
    assert isinstance(model.intercept_, float)


def test_predict_binary_labels(simple_binary_data):
    X, y = simple_binary_data
    model = LogisticRegression(max_iter=5000, learning_rate=0.5).fit(X, y)

    preds = model.predict(X)
    assert set(preds.tolist()) <= {0, 1}


def test_accuracy_on_simple_line(simple_binary_data):
    X, y = simple_binary_data
    model = LogisticRegression(max_iter=5000, learning_rate=0.5).fit(X, y)

    acc = model.score(X, y)
    assert acc >= 0.8  # hard to perfectly separate this dataset


def test_predict_proba_valid(simple_binary_data):
    X, y = simple_binary_data
    model = LogisticRegression().fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_roc_curve(simple_binary_data):
    X, y = simple_binary_data
    model = LogisticRegression().fit(X, y)

    fpr, tpr, auc = model.roc_curve(X, y)
    assert len(fpr) == len(tpr)
    assert 0 <= auc <= 1