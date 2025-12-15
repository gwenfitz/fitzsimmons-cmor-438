"""
rice_ml

A from-scratch machine learning library designed for educational use.
Includes supervised learning, unsupervised learning, and data processing
utilities with clear, sklearn-style APIs.
"""

# Subpackages
from . import supervised_learning
from . import unsupervised_learning
from . import processing

# --------------------------------------------------
# Common supervised learning models
# --------------------------------------------------
from .supervised_learning import (
    LinearRegression,
    LogisticRegression,
    KNNClassifier,
    KNNRegressor,
    Perceptron,
    MultilayerPerceptron,
    DecisionTreeClassifier,
    DecisionTree,
    RegressionTree,
)

# --------------------------------------------------
# Common unsupervised learning models
# --------------------------------------------------
from .unsupervised_learning import (
    KMeans,
    DBSCAN,
    PCA,
    LabelPropagation,
)

# --------------------------------------------------
# Processing utilities
# --------------------------------------------------
from .processing import (
    standardize,
    minmax_scale,
    train_test_split,
    accuracy_score,
    r2_score,
    mean_squared_error,
    confusion_matrix,
)

__all__ = [
    # Subpackages
    "supervised_learning",
    "unsupervised_learning",
    "processing",

    # Supervised learning
    "LinearRegression",
    "LogisticRegression",
    "KNNClassifier",
    "KNNRegressor",
    "Perceptron",
    "MultilayerPerceptron",
    "DecisionTreeClassifier",
    "DecisionTree",
    "RegressionTree",

    # Unsupervised learning
    "KMeans",
    "DBSCAN",
    "PCA",
    "LabelPropagation",

    # Processing utilities
    "standardize",
    "minmax_scale",
    "train_test_split",
    "accuracy_score",
    "r2_score",
    "mean_squared_error",
    "confusion_matrix",
]
