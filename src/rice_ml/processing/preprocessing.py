import numpy as np
from typing import Tuple, Optional

__all__ = [
    "minmax_scale",
    "standardize",
    "train_test_split",
]

def minmax_scale(X: np.ndarray) -> np.ndarray:
    """Scale each feature to [0, 1].

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)

    Returns
    -------
    X_scaled : ndarray, shape (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1.0   # avoid divide-by-zero
    return (X - X_min) / denom


def standardize(X: np.ndarray) -> np.ndarray:
    """Standardize features (zero mean, unit variance)."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple train/test split using NumPy.

    Parameters
    ----------
    X, y : ndarray
        Training data.
    test_size : float
        Fraction of samples to place in test set.
    shuffle : bool
        Whether to shuffle before splitting.
    random_state : int or None
        RNG seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)

    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)

    test_n = max(1, int(n * test_size))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]