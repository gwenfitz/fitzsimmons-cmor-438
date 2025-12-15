"""
Gradient Descent Optimization Algorithms

This module implements simple gradient descent optimizers from scratch
for both one-dimensional and multi-dimensional objective functions.
The implementations are intentionally minimal and designed to illustrate
the core mechanics of gradient-based optimization.

Design Goals
------------
- Educational clarity over performance or generality
- Explicit update rules with minimal abstraction
- Support for both scalar and vector-valued parameters
- Transparent convergence behavior via parameter history tracking

Implemented Optimizers
----------------------
- GradientDescent1D:
    Gradient descent for scalar-valued parameters w ∈ ℝ, given an
    explicit derivative df/dw.

- GradientDescentND:
    Gradient descent for vector-valued parameters w ∈ ℝⁿ, given a
    gradient function ∇f(w).

Key Characteristics
-------------------
- Fixed learning rate (no adaptive schedules)
- Simple stopping criterion based on parameter change magnitude
- Stores full optimization trajectory for visualization and analysis
- NumPy-based implementation with no external dependencies

Implementation Notes
--------------------
- These optimizers are not tied to any specific model class
- They are intended for demonstration, experimentation, and
  instructional use rather than large-scale optimization
- Numerical stability and advanced features (momentum, Adam, etc.)
  are intentionally omitted for clarity

This module provides a foundation for understanding how gradient-based
learning algorithms operate at a fundamental level.
"""


import numpy as np
from typing import Callable

class GradientDescent1D:
    """Gradient descent for 1D functions."""
    
    def __init__(self, df: Callable[[float], float], alpha: float = 0.1, tol: float = 1e-6, max_iter: int = 1000):
        """
        Parameters:
            df : derivative of the function f(w)
            alpha : learning rate
            tol : tolerance for stopping
            max_iter : maximum number of iterations
        """
        self.df = df
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.history = []

    def fit(self, w0: float) -> list[float]:
        """Run gradient descent starting from w0."""
        w = w0
        self.history = [w]
        
        for i in range(self.max_iter):
            grad = self.df(w)
            w_new = w - self.alpha * grad
            self.history.append(w_new)
            if abs(w_new - w) < self.tol:
                break
            w = w_new
        
        return self.history


class GradientDescentND:
    """Gradient descent for N-dimensional functions."""
    
    def __init__(self, grad_f: Callable[[np.ndarray], np.ndarray], alpha: float = 0.1, tol: float = 1e-6, max_iter: int = 1000):
        """
        Parameters:
            grad_f : gradient function ∇f(w)
            alpha : learning rate
            tol : tolerance for stopping
            max_iter : maximum number of iterations
        """
        self.grad_f = grad_f
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.history = []

    def fit(self, w0: np.ndarray) -> list[np.ndarray]:
        """Run gradient descent starting from w0."""
        w = np.array(w0, dtype=float)
        self.history = [w.copy()]
        
        for i in range(self.max_iter):
            grad = self.grad_f(w)
            w_new = w - self.alpha * grad
            self.history.append(w_new.copy())
            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new
        
        return self.history