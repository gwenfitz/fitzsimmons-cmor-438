"""
Post-processing Metrics and Evaluation Utilities

This module implements common evaluation metrics used to assess the
performance of supervised learning models. All metrics are implemented
from scratch using NumPy and are designed for clarity, correctness,
and instructional use.

Design Goals
------------
- Provide transparent, minimal implementations of standard metrics
- Avoid reliance on external machine learning libraries
- Enforce explicit input validation and well-defined behavior
- Serve as educational references for how evaluation metrics work

Implemented Metrics
-------------------
- accuracy_score:
    Computes the fraction of correctly classified samples.

- r2_score:
    Computes the coefficient of determination (R²) for regression tasks,
    measuring the proportion of variance explained by the model.

- mean_squared_error:
    Computes the average squared difference between true and predicted
    continuous targets.

- confusion_matrix:
    Computes a confusion matrix for classification tasks, with class
    labels sorted in increasing order.

Implementation Notes
--------------------
- All functions operate on NumPy arrays or array-like inputs
- Input shapes and lengths are validated explicitly
- Metrics follow their standard mathematical definitions
- Edge cases (such as constant targets in R²) are handled explicitly

These utilities are intended to be used in conjunction with the
supervised learning models provided in the rice_ml package and to
reinforce understanding of how model evaluation metrics are computed.
"""

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