"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

This module provides a from-scratch implementation of DBSCAN using NumPy.

DBSCAN groups points based on density rather than distance to centroids.
It is capable of discovering arbitrarily-shaped clusters and identifying
outliers (noise).

Key properties:
- No need to specify number of clusters
- Robust to noise
- Sensitive to distance scale and density parameters

No sklearn dependencies are used.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

__all__ = ["DBSCAN"]


# ==========================================================
# Utility
# ==========================================================
def _validate_inputs(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X


# ==========================================================
# DBSCAN Class
# ==========================================================
class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps : float
        Neighborhood radius.
    min_samples : int
        Minimum number of points required to form a dense region.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels (-1 indicates noise).
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1.")

        self.eps = eps
        self.min_samples = min_samples
        self.labels_: Optional[np.ndarray] = None

    # ======================================================
    # Public API
    # ======================================================
    def fit(self, X):
        """
        Fit DBSCAN clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _validate_inputs(X)
        n_samples = X.shape[0]

        labels = np.full(n_samples, -1)  # -1 = noise
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # noise
            else:
                self._expand_cluster(
                    X, labels, visited, i, neighbors, cluster_id
                )
                cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        """
        Fit DBSCAN and return cluster labels.
        """
        self.fit(X)
        return self.labels_

    # ======================================================
    # Internal helpers
    # ======================================================
    def _region_query(self, X, idx):
        """
        Find all points within eps of point idx.
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(
        self, X, labels, visited, point_idx, neighbors, cluster_id
    ):
        """
        Expand a new cluster using density reachability.
        """
        labels[point_idx] = cluster_id
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._region_query(X, neighbor_idx)

                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.unique(
                        np.concatenate([neighbors, neighbor_neighbors])
                    )

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            i += 1
