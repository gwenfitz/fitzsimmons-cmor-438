"""
Principal Component Analysis (PCA)

This module provides a from-scratch implementation of Principal Component
Analysis using NumPy only.

PCA is an unsupervised dimensionality reduction technique that projects data
onto orthogonal directions of maximum variance. It is commonly used for:

- Visualization of high-dimensional data
- Noise reduction
- Feature compression
- Preprocessing for other learning algorithms
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
    # Representation
    # ======================================================
    def __repr__(self) -> str:
        return f"PCA(n_components={self.n_components})"

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
                "n_components cannot exceed the number of features."
            )

        # Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # Eigen-decomposition (covariance matrix is symmetric)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues/vectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Store selected components
        self.components_ = eigvecs[:, : self.n_components].T
        self.explained_variance_ = eigvals[: self.n_components]

        total_variance = eigvals.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )

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
        return X_centered @ self.components_.T

    # ======================================================
    # Inverse Transform
    # ======================================================
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from PCA space back to original feature space.

        Parameters
        ----------
        X_pca : ndarray of shape (n_samples, n_components)

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA has not been fitted yet.")

        X_pca = np.asarray(X_pca, dtype=float)

        if X_pca.ndim != 2:
            raise ValueError("X_pca must be a 2D array.")

        return X_pca @ self.components_ + self.mean_

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
