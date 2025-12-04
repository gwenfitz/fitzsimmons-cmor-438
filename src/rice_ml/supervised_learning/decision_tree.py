from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


# ==========================
# Tree Node Structure
# ==========================
@dataclass
class _TreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    proba: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        return self.feature_index is None


# ==========================
# Decision Tree Classifier
# ==========================
class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int | float] = None,
        random_state: Optional[int] = None,
    ):
        # Hyperparameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        # Set during fit
        self.n_classes_ = None
        self.n_features_ = None
        self.tree_ = None
        self._rng = None

    # ==========================
    # Fit the Tree
    # ==========================
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        # Basic checks
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched sample sizes.")
        if X.size == 0:
            raise ValueError("Empty training data.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer class labels.")
        if np.min(y) < 0:
            raise ValueError("Labels must be non-negative.")

        # Store metadata
        self.n_features_ = X.shape[1]
        self.n_classes_ = int(np.max(y)) + 1
        self._rng = np.random.default_rng(self.random_state)

        # Grow tree
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    # ==========================
    # Prediction API
    # ==========================
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Call fit first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")

        out = np.zeros((X.shape[0], self.n_classes_))
        for i, row in enumerate(X):
            node = self._traverse_tree(row, self.tree_)
            out[i] = node.proba
        return out

    # ==========================
    # Build Tree Recursively
    # ==========================
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        proba = self._class_proba(y)
        labels_unique = np.unique(y)
        n_samples = len(y)

        # Stop if pure, depth reached, or too few samples
        if (
            len(labels_unique) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
        ):
            return _TreeNode(proba=proba)

        # Find best split
        feat, thresh, (left_mask, right_mask) = self._best_split(X, y)

        # No valid split → make leaf
        if feat is None:
            return _TreeNode(proba=proba)

        # Build children
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature_index=feat,
            threshold=thresh,
            left=left,
            right=right,
            proba=proba,
        )

    # ==========================
    # Find Best Split (Gini)
    # ==========================
    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Tuple[np.ndarray, np.ndarray]]:

        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, (np.array([]), np.array([]))

        # Feature subsampling
        if self.max_features is None:
            features = np.arange(n_features)
        elif isinstance(self.max_features, int):
            features = self._rng.choice(n_features, self.max_features, replace=False)
        else:
            k = max(1, int(self.max_features * n_features))
            features = self._rng.choice(n_features, k, replace=False)

        best_gini = 1.0
        best_feat = None
        best_thresh = None
        best_left = None
        best_right = None

        for feat in features:
            col = X[:, feat]
            thresholds = np.unique(col)

            if len(thresholds) == 1:
                continue

            for t in thresholds:
                left = col <= t
                right = ~left

                # Leaf-size constraint
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue

                g_left = self._gini(y[left])
                g_right = self._gini(y[right])
                g_split = (left.sum() * g_left + right.sum() * g_right) / n_samples

                if g_split < best_gini:
                    best_gini = g_split
                    best_feat = feat
                    best_thresh = float(t)
                    best_left = left
                    best_right = right

        if best_feat is None:
            return None, None, (np.array([]), np.array([]))

        return best_feat, best_thresh, (best_left, best_right)

    # ==========================
    # Helper: Gini Impurity
    # ==========================
    def _gini(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / counts.sum()
        return 1.0 - np.sum(p * p)

    # ==========================
    # Helper: Class Probabilities
    # ==========================
    def _class_proba(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / counts.sum()

    # ==========================
    # Traverse Tree for One Sample
    # ==========================
    def _traverse_tree(self, x: np.ndarray, node: _TreeNode) -> _TreeNode:
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node
