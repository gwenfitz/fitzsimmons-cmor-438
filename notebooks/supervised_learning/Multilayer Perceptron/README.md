# Multilayer Perceptron (MLP) — Binary Classification

This directory contains a Jupyter notebook demonstrating a Multilayer Perceptron (MLP) classifier implemented entirely from scratch using the rice_ml package.

The notebook builds directly on the single-layer Perceptron and shows how adding hidden layers and nonlinear activation functions allows the model to learn nonlinear decision boundaries.

No scikit-learn neural networks are used.

## What This Notebook Covers

This notebook presents a complete supervised learning workflow:
1. Reviewing the limitations of the single-layer Perceptron
2. Introducing hidden layers and nonlinear activations
3. Implementing an MLP classifier from scratch
4. Training the model using gradient descent and backpropagation
5. Comparing performance against a Perceptron baseline
6. Visualizing model behavior using PCA
7. Interpreting results from a geometric and learning-theoretic perspective

All code relies exclusively on custom implementations in the rice_ml package.

## Why Go Beyond the Perceptron?

The single-layer Perceptron can only learn linearly separable decision boundaries.

As a result, it cannot solve problems such as:
- XOR-type patterns
- Nonlinear medical or signal datasets
- Overlapping or curved class boundaries

The Multilayer Perceptron overcomes these limitations by introducing:
- One or more hidden layers
- Nonlinear activation functions
- A differentiable loss function optimized with backpropagation

## Model Overview: Multilayer Perceptron

An MLP consists of:
- An input layer
- One or more hidden layers
- An output layer

Each layer applies a linear transformation followed by a nonlinear activation.

### Forward Pass (Conceptual)

For a network with one hidden layer:

Hidden layer: h = phi(X · W1 + b1)
Output layer: y_hat = sigmoid(h · W2 + b2)

Where:

phi is a nonlinear activation function (e.g., ReLU)

sigmoid maps outputs to probabilities for binary classification

### Activation Functions

Activation functions introduce nonlinearity into the model.

ReLU (hidden layers)
phi(z) = max(0, z)

Sigmoid (output layer)
sigmoid(z) = 1 / (1 + exp(-z))

Without nonlinear activations, stacking layers would collapse into a single linear model.

### Loss Function

For binary classification, the MLP minimizes binary cross-entropy loss:

loss = -[y · log(y_hat) + (1 − y) · log(1 − y_hat)]


This loss:

Is convex with respect to the output layer

Penalizes confident incorrect predictions

Produces smooth gradients for optimization

Training with Gradient Descent and Backpropagation

Model parameters are updated using gradient descent:

W = W − learning_rate · gradient


Gradients are computed efficiently using backpropagation, which applies the chain rule to propagate errors backward through the network.

Backpropagation makes training multilayer networks computationally feasible.

## Dataset Overview

The notebook applies both the Perceptron and the MLP to the Ionosphere dataset from the UCI Machine Learning Repository.

### Dataset Characteristics

Number of samples: 351
Number of features: 34 continuous numerical values
Target labels:
- 1 → good radar return
- 0 → bad radar return

Moderate class balance

No missing values

This dataset is not linearly separable, making it ideal for demonstrating the advantages of nonlinear models.

## Exploratory Data Analysis (EDA)

EDA focuses on:
- Target class distribution
- Feature scale differences
- Feature distributions and outliers

Neural networks are sensitive to feature scale, so understanding these properties is essential before training.

## Preprocessing

All features are standardized to zero mean and unit variance:

X_std = (X − mean) / std

This ensures:
- Equal contribution from all features
- Stable gradient updates
- Faster convergence during training

The dataset is then split into training and test sets using a custom train_test_split function.

## Baseline Model: Perceptron

Before training the MLP, a single-layer Perceptron is trained as a baseline.

This comparison isolates the effect of:
- Hidden layers
- Nonlinear activations
- Backpropagation

The Perceptron establishes a lower bound on performance.

## Training the Multilayer Perceptron

The MLP is trained with:
- One hidden layer
- ReLU activation in hidden layers
- Sigmoid activation in the output layer
- Gradient descent optimization

The model learns nonlinear feature interactions that the Perceptron cannot represent.

## Model Evaluation

Performance is evaluated using classification accuracy:

Accuracy = (number of correct predictions) / (total samples)

The MLP consistently outperforms the Perceptron, demonstrating the benefit of nonlinear modeling.

## PCA Visualization

Because the dataset has 34 features, direct visualization is not possible.

Principal Component Analysis (PCA) is used to project the data into two dimensions for visualization only.

Important notes:
- PCA does not affect model training
- The MLP is trained in the full feature space
- Overlap in PCA space explains why accuracy is not perfect

## Limitations

Despite improved performance, the MLP still has limitations:
- Sensitive to hyperparameter choices
- Requires careful feature scaling
- Can overfit without regularization
- Less interpretable than linear models

These issues motivate deeper architectures and regularization techniques in modern deep learning.

## Purpose of This Notebook

This notebook is designed to:
- Demonstrate a correct from-scratch MLP implementation
- Highlight the limitations of linear classifiers
- Show how hidden layers enable nonlinear learning
- Reinforce the role of backpropagation
- Serve as a conceptual bridge to modern neural networks

## Notes

All models are implemented using the rice_ml package

No scikit-learn neural networks are used

Visualizations rely on Matplotlib

This notebook complements the Perceptron and Logistic Regression examples