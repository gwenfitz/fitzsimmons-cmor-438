"""
Principal Component Analysis (PCA)

This module provides a from-scratch implementation of PCA using NumPy.
PCA is an unsupervised dimensionality reduction technique that projects
data onto orthogonal directions of maximum variance.

No sklearn dependencies are used.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


__all__ = ["PCA"]


# ==========================================================
# PCA Class
# ==========================================================
class PCA:
    """
    Principal Component Analysis (PCA).

    PCA projects data onto a lower-dimensional subspace spanned by the
    eigenvectors of the covariance matrix corresponding to the largest
    eigenvalues.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each selected component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean.
    """

    def __init__(self, n_components: int):
        if n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)

        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    # ======================================================
    # Fit
    # ======================================================
    def fit(self, X: np.ndarray) -> "PCA":
        """
        Fit PCA on X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self : PCA
            Fitted PCA instance.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot exceed number of features."
            )

        # Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Eigen-decomposition (symmetric matrix)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues/vectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Store selected components
        self.components_ = eigvecs[:, : self.n_components].T
        self.explained_variance_ = eigvals[: self.n_components]

        total_var = eigvals.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    # ======================================================
    # Transform
    # ======================================================
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto the principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_pca : ndarray of shape (n_samples, n_components)
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    # ======================================================
    # Fit + Transform
    # ======================================================
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA on X and return the transformed data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_pca : ndarray of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
