"""
Community detection algorithms from scratch.

This module implements simple, fully self-contained community detection
methods on graphs without relying on external libraries such as sklearn
or scipy. The focus is on clarity and educational value rather than
state-of-the-art performance.

Currently implemented:
- Label Propagation Algorithm (LPA)

Graphs are represented using adjacency lists or adjacency matrices
built from NumPy arrays.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

__all__ = ["LabelPropagation"]


class LabelPropagation:
    """
    Label Propagation community detection algorithm.

    Each node starts with a unique label. At each iteration, nodes update
    their label to the most frequent label among their neighbors. The
    algorithm converges when labels no longer change.

    This implementation is deterministic given a fixed node order.
    """

    def __init__(self, max_iter: int = 100):
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.max_iter = max_iter
        self.labels_: np.ndarray | None = None

    @staticmethod
    def _validate_adjacency(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        return A

    def fit(self, A: np.ndarray):
        """
        Fit the label propagation algorithm.

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix of the graph (n x n). Nonzero values indicate
            edges. The graph is assumed to be undirected.
        """
        A = self._validate_adjacency(A)
        n = A.shape[0]

        # initialize each node with a unique label
        labels = np.arange(n)

        for _ in range(self.max_iter):
            changed = False

            for i in range(n):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) == 0:
                    continue

                neighbor_labels = labels[neighbors]
                values, counts = np.unique(neighbor_labels, return_counts=True)
                new_label = values[counts.argmax()]

                if labels[i] != new_label:
                    labels[i] = new_label
                    changed = True

            if not changed:
                break

        self.labels_ = labels
        return self

    def fit_predict(self, A: np.ndarray) -> np.ndarray:
        """
        Fit the model and return community labels.
        """
        self.fit(A)
        return self.labels_.copy()
