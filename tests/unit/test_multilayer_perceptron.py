import numpy as np
import pytest

from rice_ml.supervised_learning.multilayer_perceptron import MultilayerPerceptron


def test_fit_and_predict_simple():
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    model = MultilayerPerceptron(hidden_layers=[4], max_iter=2000, learning_rate=0.1)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_accuracy_on_xor():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])

    model = MultilayerPerceptron(
        hidden_layers=[8],
        learning_rate=0.1,
        max_iter=5000,
        random_state=0
    )

    model.fit(X, y)
    acc = model.score(X, y)

    assert acc >= 0.95


def test_predict_before_fit_raises():
    model = MultilayerPerceptron(hidden_layers=[4])
    X = np.array([[0.0, 0.0]])

    with pytest.raises(TypeError):
        model.predict(X)


def test_invalid_labels():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])

    model = MultilayerPerceptron(hidden_layers=[4])

    with pytest.raises(ValueError):
        model.fit(X, y)
