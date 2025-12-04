import numpy as np
from rice_ml.processing.preprocessing import (
    minmax_scale,
    standardize,
    train_test_split,
)


def test_minmax_scale_basic():
    X = np.array([[0, 10], [2, 20]], float)
    out = minmax_scale(X)
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[1], [1.0, 1.0])


def test_standardize_basic():
    X = np.array([[0, 10], [2, 20]], float)
    out = standardize(X)
    assert np.allclose(out.mean(axis=0), [0, 0], atol=1e-7)
    assert np.allclose(out.std(axis=0), [1, 1], atol=1e-7)


def test_train_test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    assert Xte.shape[0] == 3    # 30% of 10
    assert Xtr.shape[0] == 7
    assert len(ytr) == 7
    assert len(yte) == 3


def test_train_test_split_no_shuffle():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    Xtr, Xte, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    # first 2 become test
    assert np.array_equal(Xte, X[:2])
    assert np.array_equal(Xtr, X[2:])