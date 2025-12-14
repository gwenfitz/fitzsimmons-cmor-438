"""
K-Means Clustering (From Scratch)

This module implements the K-Means clustering algorithm using NumPy.
K-Means is an unsupervised learning method that partitions data into
K clusters by minimizing within-cluster variance.

Features
--------
• Random or user-defined initialization
• Euclidean distance metric
• Lloyd's algorithm
• Convergence based on centroid movement
• Inertia (within-cluster sum of squares)

Limitations
-----------
• Requires specifying number of clusters K
• Sensitive to initialization
• Assumes spherical clusters of similar scale

This implementation is designed for educational clarity and
integration into the rice_ml package.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _validate_inputs(X):
    """
    Validate feature matrix.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)

    Returns
    -------
    X : ndarray
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    return X


# ---------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------

class KMeans:
    """
    K-Means Clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance for centroid movement.
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    cluster_centers_ : ndarray, shape (K, n_features)
        Final centroid locations.
    labels_ : ndarray, shape (n_samples,)
        Cluster assignment for each sample.
    inertia_ : float
        Sum of squared distances to cluster centers.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_centroids(self, X):
        """
        Initialize centroids by random sampling from data.
        """
        indices = self._rng.choice(
            X.shape[0], self.n_clusters, replace=False
        )
        return X[indices]

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _assign_clusters(self, X, centroids):
        """
        Assign each point to the nearest centroid.
        """
        distances = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2
        )
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """
        Update centroid locations as cluster means.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                # Reinitialize empty cluster
                centroids[k] = X[self._rng.integers(0, X.shape[0])]
            else:
                centroids[k] = members.mean(axis=0)
        return centroids

    def _compute_inertia(self, X, centroids, labels):
        """
        Compute within-cluster sum of squares.
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            members = X[labels == k]
            inertia += np.sum((members - centroids[k]) ** 2)
        return inertia

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit K-Means clustering.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _validate_inputs(X)

        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, centroids, labels)

        return self

    def predict(self, X):
        """
        Assign clusters to new data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit before predict.")

        X = _validate_inputs(X)
        return self._assign_clusters(X, self.cluster_centers_)
