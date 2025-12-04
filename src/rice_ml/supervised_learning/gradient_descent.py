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