"""
Multilayer Perceptron (MLP) — Binary Classification

This module implements a fully-connected Multilayer Perceptron (MLP)
from scratch using NumPy. The model supports an arbitrary number of
hidden layers and is trained using batch gradient descent with
backpropagation.

The implementation is intentionally simple and educational, designed
to complement other supervised learning models in the rice_ml package
(e.g., Perceptron, LogisticRegression) while demonstrating how nonlinear
decision boundaries can be learned using layered representations.

--------------------------------------------------
Model Overview
--------------------------------------------------
• Feedforward neural network
• Fully connected layers
• Sigmoid activation for all layers
• Binary cross-entropy loss
• Batch gradient descent optimization
• Backpropagation via chain rule

--------------------------------------------------
Mathematical Formulation
--------------------------------------------------
Forward pass:
    z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
    a^{(l)} = σ(z^{(l)})

Binary cross-entropy loss:
    L = -1/n ∑ [ y log(ŷ) + (1 - y) log(1 - ŷ) ]

Backpropagation:
    Gradients are computed using the chain rule
    and propagated backward through the network.

--------------------------------------------------
Design Notes
--------------------------------------------------
• Output layer has a single neuron with sigmoid activation
• Supports binary labels {0, 1} only
• Uses full-batch gradient descent (not mini-batch)
• No regularization or momentum (by design)
• Deterministic behavior via random_state

--------------------------------------------------
Intended Use
--------------------------------------------------
This implementation is designed for:
• Educational demonstrations
• Small-to-medium datasets
• Course projects requiring from-scratch models

It is not intended as a replacement for optimized
libraries such as PyTorch or TensorFlow.

--------------------------------------------------
Author: Gwenyth FitzSimmons
Course: CMOR 438
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


def sigmoid_derivative(a):
    return a * (1.0 - a)


# ---------------------------------------------------------------------
# Multilayer Perceptron
# ---------------------------------------------------------------------

class MultilayerPerceptron:
    """
    Multilayer Perceptron (MLP) for binary classification.

    Architecture
    ------------
    input → hidden layers → output (sigmoid)

    Training
    --------
    • Batch gradient descent
    • Binary cross-entropy loss
    • Backpropagation

    Parameters
    ----------
    hidden_layers : list[int]
        Number of neurons in each hidden layer.
    learning_rate : float
        Gradient descent step size.
    max_iter : int
        Maximum number of training iterations.
    tol : float
        Early stopping tolerance.
    random_state : int or None
        Random seed.

    Attributes
    ----------
    weights_ : list[np.ndarray]
        Weight matrices.
    biases_ : list[np.ndarray]
        Bias vectors.
    """

    def __init__(
        self,
        hidden_layers: List[int],
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.weights_: Optional[List[np.ndarray]] = None
        self.biases_: Optional[List[np.ndarray]] = None
        self.loss_history_: list[float] = []

        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_parameters(self, n_features):
        layer_sizes = [n_features] + self.hidden_layers + [1]

        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            W = self._rng.normal(0, 0.1, size=(layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights_.append(W)
            self.biases_.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X):
        activations = [X]

        for W, b in zip(self.weights_, self.biases_):
            Z = activations[-1] @ W + b
            A = sigmoid(Z)
            activations.append(A)

        return activations

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backward(self, activations, y):
        grads_W = []
        grads_b = []

        y = y.reshape(-1, 1)
        delta = activations[-1] - y  # BCE + sigmoid simplification

        for i in reversed(range(len(self.weights_))):
            A_prev = activations[i]
            W = self.weights_[i]

            dW = A_prev.T @ delta / len(y)
            db = delta.mean(axis=0, keepdims=True)

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ W.T) * sigmoid_derivative(activations[i])

        return grads_W, grads_b

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)

        if not np.all(np.isin(np.unique(y), [0, 1])):
            raise ValueError("MLP supports binary labels 0/1 only.")

        self._initialize_parameters(X.shape[1])

        prev_loss = np.inf

        for _ in range(self.max_iter):
            activations = self._forward(X)
            y_pred = activations[-1]

            # Binary cross-entropy
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
        X = _validate_inputs(X)
        activations = self._forward(X)
        return activations[-1].ravel()

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def score(self, X, y):
        X, y = _validate_inputs(X, y)
        preds = self.predict(X)
        return np.mean(preds == y)
