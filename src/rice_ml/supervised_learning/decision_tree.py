"""
Decision Tree Classifier (CART-style)

This module implements a decision tree classifier from scratch using
Gini impurity as the splitting criterion. The tree is constructed
recursively by selecting feature–threshold splits that minimize the
weighted Gini impurity of the resulting child nodes.

Design Goals
------------
- Educational clarity over performance optimization
- Explicit, readable implementation of the CART algorithm
- Minimal dependencies (NumPy only)
- sklearn-style public API: fit, predict, predict_proba, score

Key Characteristics
-------------------
- Binary splits on continuous features
- Greedy, top-down recursive tree construction
- Support for:
    * Maximum tree depth
    * Minimum samples per split
    * Minimum samples per leaf
    * Feature subsampling (for ensemble methods)
- Stores class probabilities at each node for probabilistic prediction

Implementation Notes
--------------------
- Tree nodes are represented by a lightweight internal _TreeNode class
- Stopping conditions enforce purity, depth limits, and sample thresholds
- Feature subsampling enables compatibility with Random Forest-style ensembles
- This implementation is intended for instructional use and small datasets

This module does not attempt to match the performance, memory efficiency,
or numerical stability of production-grade libraries such as scikit-learn.
Instead, it aims to closely mirror the underlying algorithmic logic of
decision trees in a transparent and inspectable manner.
"""


from __future__ import annotations
from typing import Optional
import numpy as np

# ==========================================================
# Tree Node Structure
# ==========================================================
class _TreeNode:
    """
    Internal tree node representation.
    """

    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_TreeNode"] = None,
        right: Optional["_TreeNode"] = None,
        proba: Optional[np.ndarray] = None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.proba = proba

    def is_leaf(self) -> bool:
        return self.feature_index is None


# ==========================================================
# Decision Tree Classifier (Gini Impurity)
# ==========================================================
class DecisionTreeClassifier:
    """
    Decision Tree classifier using Gini impurity.

    The tree is built recursively by selecting feature-threshold
    splits that minimize weighted Gini impurity.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. If None, grows until pure.
    min_samples_split : int
        Minimum samples required to split a node.
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    max_features : int or float or None
        Number (or fraction) of features to consider at each split.
    random_state : int or None
        Random seed for feature subsampling.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int | float] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.n_classes_ = None
        self.n_features_ = None
        self.tree_ = None
        self._rng = None

    # ======================================================
    # Fit
    # ======================================================
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer class labels.")
        if np.min(y) < 0:
            raise ValueError("Class labels must be non-negative.")

        self.n_features_ = X.shape[1]
        self.n_classes_ = int(np.max(y)) + 1
        self._rng = np.random.default_rng(self.random_state)

        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    # ======================================================
    # Prediction API
    # ======================================================
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Call fit before predict.")

        X = np.asarray(X)
        out = np.zeros((X.shape[0], self.n_classes_))

        for i, x in enumerate(X):
            node = self._traverse_tree(x, self.tree_)
            out[i] = node.proba

        return out

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Classification accuracy.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        preds = self.predict(X)
        return float(np.mean(preds == y))

    # ======================================================
    # Tree Construction
    # ======================================================
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        proba = self._class_proba(y)

        # Stopping conditions
        if (
            len(np.unique(y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
        ):
            return _TreeNode(proba=proba)

        feat, thresh, (left, right) = self._best_split(X, y)

        if feat is None:
            return _TreeNode(proba=proba)

        return _TreeNode(
            feature_index=feat,
            threshold=thresh,
            left=self._grow_tree(X[left], y[left], depth + 1),
            right=self._grow_tree(X[right], y[right], depth + 1),
            proba=proba,
        )

    # ======================================================
    # Split Selection (Gini)
    # ======================================================
    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Tuple[np.ndarray, np.ndarray]]:

        n_samples, n_features = X.shape
        best_gini = 1.0
        best_feat, best_thresh = None, None
        best_left, best_right = None, None

        if self.max_features is None:
            features = np.arange(n_features)
        elif isinstance(self.max_features, int):
            features = self._rng.choice(n_features, self.max_features, replace=False)
        else:
            k = max(1, int(self.max_features * n_features))
            features = self._rng.choice(n_features, k, replace=False)

        for feat in features:
            values = np.unique(X[:, feat])
            for t in values:
                left = X[:, feat] <= t
                right = ~left

                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue

                g = (
                    left.sum() * self._gini(y[left])
                    + right.sum() * self._gini(y[right])
                ) / n_samples

                if g < best_gini:
                    best_gini = g
                    best_feat = feat
                    best_thresh = float(t)
                    best_left = left
                    best_right = right

        if best_feat is None:
            return None, None, (None, None)

        return best_feat, best_thresh, (best_left, best_right)

    # ======================================================
    # Impurity & Traversal
    # ======================================================
    def _gini(self, y: np.ndarray) -> float:
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / counts.sum()
        return 1.0 - np.sum(p * p)

    def _class_proba(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / counts.sum()

    def _traverse_tree(self, x: np.ndarray, node: _TreeNode) -> _TreeNode:
        while not node.is_leaf():
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node


# sklearn-style alias
DecisionTree = DecisionTreeClassifier
