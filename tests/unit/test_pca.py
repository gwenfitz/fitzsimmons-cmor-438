import numpy as np
import pytest

from rice_ml.unsupervised_learning.pca import PCA


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def simple_data():
    """
    Simple 2D dataset with clear principal direction.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    X[:, 1] = 0.5 * X[:, 0] + rng.normal(scale=0.1, size=100)
    return X


@pytest.fixture
def higher_dim_data():
    """
    Higher-dimensional random dataset.
    """
    rng = np.random.default_rng(1)
    return rng.normal(size=(50, 5))


# ==========================================================
# Shape & API tests
# ==========================================================

def test_fit_transform_shape(simple_data):
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(simple_data)

    assert X_pca.shape == (simple_data.shape[0], 1)


def test_transform_after_fit(simple_data):
    pca = PCA(n_components=2)
    pca.fit(simple_data)
    X_pca = pca.transform(simple_data)

    assert X_pca.shape == (simple_data.shape[0], 2)


def test_fit_returns_self(simple_data):
    pca = PCA(n_components=1)
    out = pca.fit(simple_data)

    assert out is pca


# ==========================================================
# Numerical correctness
# ==========================================================

def test_explained_variance_order(simple_data):
    pca = PCA(n_components=2)
    pca.fit(simple_data)

    ev = pca.explained_variance_
    assert ev[0] >= ev[1]


def test_explained_variance_ratio_sum(simple_data):
    pca = PCA(n_components=2)
    pca.fit(simple_data)

    ratio_sum = pca.explained_variance_ratio_.sum()
    assert pytest.approx(ratio_sum, rel=1e-6) == 1.0


def test_components_orthonormal(simple_data):
    pca = PCA(n_components=2)
    pca.fit(simple_data)

    C = pca.components_
    identity = C @ C.T

    assert np.allclose(identity, np.eye(2), atol=1e-6)


# ==========================================================
# Error handling
# ==========================================================

def test_transform_before_fit_raises(simple_data):
    pca = PCA(n_components=1)
    with pytest.raises(RuntimeError):
        pca.transform(simple_data)


def test_invalid_n_components():
    with pytest.raises(ValueError):
        PCA(n_components=0)


def test_too_many_components(higher_dim_data):
    pca = PCA(n_components=10)
    with pytest.raises(ValueError):
        pca.fit(higher_dim_data)


def test_non_2d_input():
    pca = PCA(n_components=1)
    with pytest.raises(ValueError):
        pca.fit(np.array([1, 2, 3]))
