# Core models
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression

# Neighbors
from .knn import KNNClassifier, KNNRegressor

# Perceptron & neural networks
from .perceptron import Perceptron
from .multilayer_perceptron import MultilayerPerceptron

# Trees
from .decision_tree import DecisionTreeClassifier, DecisionTree
from .regression_trees import RegressionTree

# Optimization
from .gradient_descent import GradientDescent1D, GradientDescentND

# Ensembles
from .ensemble_methods import (
    BaggingClassifier,
    VotingClassifier,
    RandomForestClassifier,
)

# Distance metrics
from .distance_metrics import euclidean_distance, manhattan_distance


__all__ = [
    # Linear models
    "LinearRegression",
    "LogisticRegression",

    # Neighbors
    "KNNClassifier",
    "KNNRegressor",

    # Perceptron / NN
    "Perceptron",
    "MultilayerPerceptron",

    # Trees
    "DecisionTreeClassifier",
    "DecisionTree",
    "RegressionTree",

    # Optimization
    "GradientDescent1D",
    "GradientDescentND",

    # Ensembles
    "BaggingClassifier",
    "VotingClassifier",
    "RandomForestClassifier",

    # Metrics
    "euclidean_distance",
    "manhattan_distance",
]