import numpy as np
import pytest

from rice_ml.unsupervised_learning.pca import PCA


# ==========================================================
# Helper data generators
# ==========================================================

def make_simple_2d_data(seed=0):
    """
    Simple 2D dataset with a clear principal direction.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 2))
    X[:, 1] = 0.5 * X[:, 0] + rng.normal(scale=0.1, size=100)
    return X


def make_high_dim_data(seed=1):
    """
    Higher-dimensional random dataset.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(50, 5))


# ==========================================================
# Shape & API tests
# ==========================================================

def test_fit_transform_shape():
    X = make_simple_2d_data()
    pca = PCA(n_components=1)

    X_pca = pca.fit_transform(X)

    assert X_pca.shape == (X.shape[0], 1)


def test_transform_after_fit():
    X = make_simple_2d_data()
    pca = PCA(n_components=2)

    pca.fit(X)
    X_pca = pca.transform(X)

    assert X_pca.shape == (X.shape[0], 2)


def test_fit_returns_self():
    X = make_simple_2d_data()
    pca = PCA(n_components=1)

    out = pca.fit(X)

    assert out is pca


def test_repr():
    pca = PCA(n_components=3)
    assert repr(pca) == "PCA(n_components=3)"


# ==========================================================
# Numerical correctness
# ==========================================================

def test_explained_variance_order():
    X = make_simple_2d_data()
    pca = PCA(n_components=2)

    pca.fit(X)
    ev = pca.explained_variance_

    assert ev[0] >= ev[1]


def test_explained_variance_ratio_sum():
    X = make_simple_2d_data()
    pca = PCA(n_components=2)

    pca.fit(X)
    ratio_sum = pca.explained_variance_ratio_.sum()

    assert pytest.approx(ratio_sum, rel=1e-6) == 1.0


def test_components_orthonormal():
    X = make_simple_2d_data()
    pca = PCA(n_components=2)

    pca.fit(X)
    C = pca.components_

    identity = C @ C.T
    assert np.allclose(identity, np.eye(2), atol=1e-6)


# ==========================================================
# Inverse transform tests
# ==========================================================

def test_inverse_transform_shape():
    X = make_simple_2d_data()
    pca = PCA(n_components=1)

    X_pca = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_pca)

    assert X_rec.shape == X.shape


def test_inverse_transform_reconstruction_error():
    """
    Reconstruction error should be small when retaining enough components.
    """
    X = make_high_dim_data()
    pca = PCA(n_components=5)

    X_pca = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_pca)

    mse = np.mean((X - X_rec) ** 2)
    assert mse < 1e-10


# ==========================================================
# Error handling
# ==========================================================

def test_transform_before_fit_raises():
    X = make_simple_2d_data()
    pca = PCA(n_components=1)

    with pytest.raises(RuntimeError):
        pca.transform(X)


def test_inverse_transform_before_fit_raises():
    X = make_simple_2d_data()
    pca = PCA(n_components=1)

    with pytest.raises(RuntimeError):
        pca.inverse_transform(X)


def test_invalid_n_components():
    with pytest.raises(ValueError):
        PCA(n_components=0)


def test_too_many_components():
    X = make_high_dim_data()
    pca = PCA(n_components=10)

    with pytest.raises(ValueError):
        pca.fit(X)


def test_non_2d_input():
    pca = PCA(n_components=1)

    with pytest.raises(ValueError):
        pca.fit(np.array([1, 2, 3]))
