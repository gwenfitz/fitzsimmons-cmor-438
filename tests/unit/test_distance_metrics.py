import numpy as np
import pytest

from rice_ml.supervised_learning.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
)


# ---------------------------------------------------------------------------
# Euclidean distance tests
# ---------------------------------------------------------------------------

def test_euclidean_basic():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0


def test_euclidean_identical():
    assert euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0


def test_euclidean_negative_values():
    a = [-1, -2]
    b = [3, 2]
    expected = np.linalg.norm(np.array(a) - np.array(b))
    assert euclidean_distance(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Manhattan distance tests
# ---------------------------------------------------------------------------

def test_manhattan_basic():
    assert manhattan_distance([1, 2, 3], [4, 0, 3]) == 5


def test_manhattan_zero():
    assert manhattan_distance([0, 0], [0, 0]) == 0


# ---------------------------------------------------------------------------
# Shape mismatch tests
# ---------------------------------------------------------------------------

def test_shape_mismatch():
    with pytest.raises(ValueError):
        euclidean_distance([1, 2], [1, 2, 3])

    with pytest.raises(ValueError):
        manhattan_distance(np.array([1, 2]), np.array([[1, 2]]))


# ---------------------------------------------------------------------------
# Non-numeric input tests
# ---------------------------------------------------------------------------

def test_non_numeric_inputs():
    with pytest.raises(TypeError):
        euclidean_distance(["a", "b"], ["c", "d"])

    with pytest.raises(TypeError):
        manhattan_distance([1, 2], ["x", "y"])


# ---------------------------------------------------------------------------
# Float precision tests
# ---------------------------------------------------------------------------

def test_float_precision():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.4, 0.6, 0.9])
    expected = np.linalg.norm(a - b)
    assert euclidean_distance(a, b) == pytest.approx(float(expected))


# ---------------------------------------------------------------------------
# Stress test with large vectors
# ---------------------------------------------------------------------------

def test_large_vectors():
    a = np.arange(10000, dtype=float)
    b = a + 1
    expected = np.linalg.norm(a - b)
    assert euclidean_distance(a, b) == pytest.approx(float(expected))