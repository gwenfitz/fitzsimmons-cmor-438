import numpy as np
from typing import Any

__all__ = [
    "accuracy_score",
    "r2_score",
    "mean_squared_error",
    "confusion_matrix",
]

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")
    return float(np.mean(y_true == y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 for regression."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)

    if ss_tot == 0:
        if ss_res == 0:
            return 1.0
        raise ValueError("R^2 is undefined when y_true is constant.")
    return float(1 - ss_res/ss_tot)

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).

    MSE = (1 / n) * Σ (y_i - ŷ_i)²
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")

    return float(np.mean((y_true - y_pred) ** 2))

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix with classes sorted in increasing order."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), int)

    for t, p in zip(y_true, y_pred):
        i = np.where(classes == t)[0][0]
        j = np.where(classes == p)[0][0]
        cm[i, j] += 1

    return cm