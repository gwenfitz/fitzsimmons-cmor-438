"""
Gradient Descent (NumPy-only).

This module provides simple, dependency-free implementations of gradient
descent for educational and lightweight optimization tasks. It supports:

- Single-variable gradient descent (`GradientDescent1D`)
- Multi-variable gradient descent (`GradientDescentND`)
- Step-size (learning rate) tuning
- Custom derivative functions
- Iteration logging

NumPy-only, suitable for teaching and testing without heavy dependencies.

Examples
--------
Basic single-variable example:

>>> import numpy as np
>>> from rice_ml.supervised_learning.gradient_descent import GradientDescent1D
>>> f = lambda w: (w - 2)**2 + 1
>>> df = lambda w: 2 * (w - 2)
>>> gd = GradientDescent1D(df, alpha=0.8, w0=5.0, tol=1e-3)
>>> w_path = gd.run()
>>> np.round(w_path[-1], 3)
2.0

Basic multi-variable example:

>>> import numpy as np
>>> from rice_ml.supervised_learning.gradient_descent import GradientDescentND
>>> f = lambda w: w[0]**2 + w[1]**2 + 1
>>> df = lambda w: np.array([2*w[0], 2*w[1]])
>>> gd = GradientDescentND(df, alpha=0.1, W0=np.array([5.0, -5.0]), max_iter=500)
>>> W_path = gd.run()
>>> np.round(W_path[-1], 2).tolist()
[0.0, 0.0]
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Sequence, Union

__all__ = ["GradientDescent1D", "GradientDescentND"]

ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Helper validation
# ---------------------------------------------------------------------------

def _ensure_callable(fn: Callable, name: str) -> None:
    """Ensure the given function is callable."""
    if not callable(fn):
        raise TypeError(f"{name} must be callable, got {type(fn).__name__}.")


def _ensure_float(value, name: str) -> float:
    """Ensure a value is a float."""
    try:
        return float(value)
    except Exception as e:
        raise TypeError(f"{name} must be a float-compatible value.") from e


# ---------------------------------------------------------------------------
# Gradient Descent (1D)
# ---------------------------------------------------------------------------

class GradientDescent1D:
    """
    Single-variable gradient descent optimizer.

    Parameters
    ----------
    df : callable
        Derivative function `df(w)` returning float.
    alpha : float, default=0.8
        Learning rate (step size).
    w0 : float, default=0.0
        Initial weight.
    tol : float, default=1e-3
        Convergence tolerance based on |df(w)|.
    max_iter : int, default=1000
        Maximum number of iterations.

    Attributes
    ----------
    history_ : list[float]
        Sequence of weight values across iterations.
    n_iter_ : int
        Number of completed iterations.
    """

    def __init__(
        self,
        df: Callable[[float], float],
        alpha: float = 0.8,
        w0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ) -> None:
        _ensure_callable(df, "df")
        self.df = df
        self.alpha = _ensure_float(alpha, "alpha")
        self.w0 = _ensure_float(w0, "w0")
        self.tol = _ensure_float(tol, "tol")
        self.max_iter = int(max_iter)
        self.history_: list[float] = []
        self.n_iter_: int = 0

    def run(self) -> np.ndarray:
        """
        Execute the gradient descent process.

        Returns
        -------
        np.ndarray
            Array of weight values at each iteration.
        """
        w = self.w0
        self.history_ = [w]
        for _ in range(self.max_iter):
            grad = self.df(w)
            if abs(grad) < self.tol:
                break
            w -= self.alpha * grad
            self.history_.append(w)
        self.n_iter_ = len(self.history_) - 1
        return np.array(self.history_, dtype=float)


# ---------------------------------------------------------------------------
# Gradient Descent (ND)
# ---------------------------------------------------------------------------

class GradientDescentND:
    """
    Multi-variable gradient descent optimizer.

    Parameters
    ----------
    df : callable
        Gradient function `df(W)` returning a NumPy array of same shape as W.
    alpha : float, default=0.1
        Learning rate (step size).
    W0 : array_like
        Initial parameter vector.
    max_iter : int, default=1000
        Maximum iterations.

    Attributes
    ----------
    history_ : list[np.ndarray]
        List of W vectors per iteration.
    n_iter_ : int
        Number of completed iterations.
    """

    def __init__(
        self,
        df: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.1,
        W0: ArrayLike = (0.0, 0.0),
        max_iter: int = 1000,
    ) -> None:
        _ensure_callable(df, "df")
        self.df = df
        self.alpha = _ensure_float(alpha, "alpha")
        self.W0 = np.asarray(W0, dtype=float)
        if self.W0.ndim != 1:
            raise ValueError("W0 must be a 1D array or sequence.")
        self.max_iter = int(max_iter)
        self.history_: list[np.ndarray] = []
        self.n_iter_: int = 0

    def run(self) -> np.ndarray:
        """
        Execute the gradient descent process.

        Returns
        -------
        np.ndarray
            Array of shape (n_iter, n_features) with parameter values.
        """
        W = self.W0.copy()
        self.history_ = [W.copy()]
        for _ in range(self.max_iter):
            grad = np.asarray(self.df(W), dtype=float)
            if grad.shape != W.shape:
                raise ValueError("Gradient shape mismatch with W.")
            W -= self.alpha * grad
            self.history_.append(W.copy())
        self.n_iter_ = len(self.history_) - 1
        return np.vstack(self.history_)