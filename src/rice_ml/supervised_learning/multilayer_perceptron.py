"""
Multilayer Perceptron (MLP) — Binary Classification

This module implements a fully connected Multilayer Perceptron (MLP)
from scratch using NumPy. The model supports arbitrary hidden-layer
architectures and is trained using batch gradient descent with
backpropagation.

Key Characteristics
-------------------
• Nonlinear classifier (unlike the single-layer perceptron)
• Supports one or more hidden layers
• ReLU activations in hidden layers
• Sigmoid activation in the output layer
• Binary cross-entropy loss
• Gradient-based optimization (no sklearn)

Educational Purpose
-------------------
This implementation demonstrates:

• How multilayer neural networks extend linear models
• How backpropagation computes gradients efficiently
• Why nonlinear activations are required to solve problems like XOR
• The relationship between perceptrons, logistic regression, and MLPs
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _validate_inputs(X, y=None):
    X = np.asarray(X, dtype=float)

    if y is None:
        return X

    y = np.asarray(y, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    return X, y


# ---------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(a):
    return (a > 0).astype(float)


# ---------------------------------------------------------------------
# Multilayer Perceptron
# ---------------------------------------------------------------------

class MultilayerPerceptron:
    """
    Multilayer Perceptron (MLP) for binary classification.

    Architecture
    ------------
    input → hidden layers (ReLU) → output (sigmoid)
    """

    def __init__(
        self,
        hidden_layers,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.weights_: Optional[List[np.ndarray]] = None
        self.biases_: Optional[List[np.ndarray]] = None
        self.loss_history_: List[float] = []

        # Explicit binary class semantics
        self.classes_ = np.array([0, 1])

        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_parameters(self, n_features: int):
        layer_sizes = [n_features] + self.hidden_layers + [1]

        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            W = self._rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(layer_sizes[i]),
                size=(layer_sizes[i], layer_sizes[i + 1])
            )
            b = np.zeros(layer_sizes[i + 1])

            self.weights_.append(W)
            self.biases_.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X):
        activations = [X]

        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            Z = activations[-1] @ W + b

            if i == len(self.weights_) - 1:
                A = sigmoid(Z)   # output layer
            else:
                A = relu(Z)      # hidden layers

            activations.append(A)

        return activations

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backward(self, activations, y):
        grads_W = []
        grads_b = []

        y = y.reshape(-1, 1)
        delta = activations[-1] - y  # BCE + sigmoid

        for i in reversed(range(len(self.weights_))):
            A_prev = activations[i]
            W = self.weights_[i]

            dW = A_prev.T @ delta / len(y)
            db = delta.mean(axis=0)

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ W.T) * relu_derivative(activations[i])

        return grads_W, grads_b

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)

        # Enforce binary labels explicitly
        if not np.array_equal(np.unique(y), self.classes_):
            raise ValueError("MLP supports binary labels {0, 1} only.")

        self._initialize_parameters(X.shape[1])

        prev_loss = np.inf

        for _ in range(self.max_iter):
            activations = self._forward(X)
            y_pred = activations[-1]

            loss = -np.mean(
                y * np.log(y_pred + 1e-9)
                + (1 - y) * np.log(1 - y_pred + 1e-9)
            )
            self.loss_history_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            grads_W, grads_b = self._backward(activations, y)

            for i in range(len(self.weights_)):
                self.weights_[i] -= self.learning_rate * grads_W[i]
                self.biases_[i] -= self.learning_rate * grads_b[i]

        return self

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        if self.weights_ is None:
            raise TypeError("Model has not been fit yet.")

        X = _validate_inputs(X)
        activations = self._forward(X)
        return activations[-1].ravel()

    def predict(self, X, threshold: float = 0.5):
        probs = self.predict_proba(X)
        return np.where(probs >= threshold, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        X, y = _validate_inputs(X, y)
        preds = self.predict(X)
        return np.mean(preds == y)
