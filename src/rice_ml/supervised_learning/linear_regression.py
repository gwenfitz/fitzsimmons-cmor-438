"""
Linear Regression (OLS + Ridge + Gradient Descent)

A self-contained implementation that supports:
- Closed-form OLS / Ridge regression
- Optional gradient descent optimizer
- R², MSE, RMSE, MAE metrics
- Residual diagnostics
- Automatic intercept handling
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------
# Input validation utility
# ---------------------------------------------------------------------

def _validate_inputs(X, y: Optional[np.ndarray] = None):
    """
    Validate X and optional y; ensure matching rows.

    Parameters
    ----------
    X : array_like
        Feature matrix.
    y : array_like or None
        Target vector.

    Returns
    -------
    X : ndarray
    y : ndarray or None

    Raises
    ------
    ValueError
        If X and y lengths mismatch.
    """
    X = np.asarray(X, dtype=float)

    if y is None:
        return X

    y = np.asarray(y, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    return X, y


# ---------------------------------------------------------------------
# Linear Regression Class
# ---------------------------------------------------------------------

class LinearRegression:
    """
    Linear Regression (OLS + optional Ridge + optional Gradient Descent).

    Parameters
    ----------
    fit_intercept : bool
        Whether to include a bias term.
    regularization : float
        L2 penalty λ (Ridge). If 0, plain OLS.
    use_gradient_descent : bool
        Whether to fit using gradient descent.
    learning_rate : float
        Gradient descent step size.
    max_iter : int
        Maximum iterations for gradient descent.
    tol : float
        Early stopping tolerance.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Model coefficients.
    intercept_ : float
        Bias parameter.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        regularization: float = 0.0,
        use_gradient_descent: bool = False,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.use_gradient_descent = use_gradient_descent
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    # ------------------------------------------------------------------
    # Closed-form solution
    # ------------------------------------------------------------------

    def _closed_form_fit(self, X, y):
        """
        Solve:
            w = (XᵀX + λI)⁻¹ Xᵀ y
        """
        n_features = X.shape[1]
        I = np.eye(n_features)

        # Do NOT regularize intercept
        if self.fit_intercept:
            I[0, 0] = 0

        A = X.T @ X + self.regularization * I
        b = X.T @ y

        w = np.linalg.solve(A, b)
        return w

    # ------------------------------------------------------------------
    # Gradient descent solution
    # ------------------------------------------------------------------

    def _gradient_descent_fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)

        for _ in range(self.max_iter):
            y_pred = X @ w
            grad = (2 / n_samples) * (X.T @ (y_pred - y))

            # Ridge penalty
            if self.regularization > 0:
                reg = 2 * self.regularization * w
                if self.fit_intercept:
                    reg[0] = 0
                grad += reg

            w_new = w - self.learning_rate * grad

            if np.linalg.norm(w_new - w) < self.tol:
                break

            w = w_new

        return w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit model using either closed-form OLS or gradient descent.
        """
        X, y = _validate_inputs(X, y)
        X_aug = self._add_intercept(X)

        if self.use_gradient_descent:
            w = self._gradient_descent_fit(X_aug, y)
        else:
            w = self._closed_form_fit(X_aug, y)

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        X = _validate_inputs(X)
        return X @ self.coef_ + self.intercept_

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def residuals(self, X, y):
        X, y = _validate_inputs(X, y)
        return y - self.predict(X)

    def mse(self, X, y):
        r = self.residuals(X, y)
        return np.mean(r**2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))

    def mae(self, X, y):
        r = self.residuals(X, y)
        return np.mean(np.abs(r))

    def score(self, X, y):
        """
        R² score:
              SS_res
        1 - ---------
              SS_tot
        """
        X, y = _validate_inputs(X, y)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)

        return 1 - ss_res / ss_tot

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_residuals(self, X, y):
        import matplotlib.pyplot as plt

        res = self.residuals(X, y)
        y_pred = self.predict(X)

        plt.figure(figsize=(7,5))
        plt.scatter(y_pred, res, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title("Residuals vs Predicted")
        plt.show()

    def summary(self, X, y):
        print("Linear Regression Summary")
        print("-"*40)
        print(f"Intercept: {self.intercept_}")
        print(f"Coefficients: {self.coef_}")
        print(f"R² Score: {self.score(X, y):.4f}")
        print(f"MSE: {self.mse(X, y):.4f}")
        print(f"RMSE: {self.rmse(X, y):.4f}")
        print(f"MAE: {self.mae(X, y):.4f}")
        print("-"*40)