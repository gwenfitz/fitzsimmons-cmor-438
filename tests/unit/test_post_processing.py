import numpy as np
from rice_ml.supervised_learning.post_processing import (
    accuracy_score,
    r2_score,
    confusion_matrix,
)


def test_accuracy_score():
    assert accuracy_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy_score([1, 2], [1, 0]) == 0.5


def test_r2_score_simple():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert r2_score(y_true, y_pred) == 1.0


def test_r2_score_nonperfect():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 2])
    r2 = r2_score(y_true, y_pred)
    assert np.isclose(r2, -0.5)


def test_r2_constant_y_true_perfect_fit():
    y_true = np.array([4, 4, 4])
    y_pred = np.array([4, 4, 4])
    assert r2_score(y_true, y_pred) == 1.0


def test_confusion_matrix_basic():
    cm = confusion_matrix([0, 1, 1], [0, 1, 0])
    assert cm.shape == (2, 2)
    assert np.array_equal(cm, np.array([[1, 0], [1, 1]]))