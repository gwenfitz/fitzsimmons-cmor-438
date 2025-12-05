"""
Distance metric utilities.

This module implements common vector distance functions used in machine
learning, including Euclidean (L2) and Manhattan (L1). All operations
use NumPy for performance and include strict validation to ensure that
inputs are 1-dimensional numeric arrays with matching shapes.
"""

from __future__ import annotations
from typing import Tuple, Iterable
import numpy as np

__all__ = ["euclidean_distance", "manhattan_distance"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_1d_float_array(x: Iterable, name: str) -> np.ndarray:
    """
    Convert input to a validated 1D float array.

    Parameters
    ----------
    x : array_like
        Input vector.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        1D float array.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional.
    TypeError
        If the input contains non-numeric values.
    """
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"Input '{name}' must be 1-dimensional; got {arr.ndim}D.")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Input '{name}' must contain only numeric values.")

    try:
        return arr.astype(float, copy=False)
    except (TypeError, ValueError):
        raise TypeError(f"Input '{name}' must contain only numeric values.")


def _validate_pair(a: Iterable, b: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate two input vectors for distance computation.

    Returns
    -------
    (a_arr, b_arr) : tuple of np.ndarray

    Raises
    ------
    ValueError
        If shapes differ.
    """
    a_arr = _to_1d_float_array(a, "a")
    b_arr = _to_1d_float_array(b, "b")

    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"Vectors must have the same shape; got {a_arr.shape} and {b_arr.shape}."
        )

    return a_arr, b_arr


# ---------------------------------------------------------------------------
# Public distance metrics
# ---------------------------------------------------------------------------

def euclidean_distance(a: Iterable, b: Iterable) -> float:
    """
    Compute the Euclidean (L2) distance between two 1D vectors.

    Defined as:
        d(a, b) = sqrt(sum((a_i - b_i)^2))

    Parameters
    ----------
    a, b : array_like
        Input vectors.

    Returns
    -------
    float
        Euclidean distance.

    Raises
    ------
    ValueError, TypeError
        For invalid shapes or non-numeric input.
    """
    a_arr, b_arr = _validate_pair(a, b)
    return float(np.linalg.norm(a_arr - b_arr))


def manhattan_distance(a: Iterable, b: Iterable) -> float:
    """
    Compute the Manhattan (L1) distance between two 1D vectors.

    Defined as:
        d(a, b) = sum(|a_i - b_i|)

    Parameters
    ----------
    a, b : array_like
        Input vectors.

    Returns
    -------
    float
        Manhattan distance.

    Raises
    ------
    ValueError, TypeError
        For invalid shapes or non-numeric input.
    """
    a_arr, b_arr = _validate_pair(a, b)
    return float(np.sum(np.abs(a_arr - b_arr)))