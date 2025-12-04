import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Simple train/test split similar to sklearn.

    Parameters
    ----------
    X : array-like
        Features.
    y : array-like
        Labels.
    test_size : float
        Fraction of the dataset to include in the test split.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )