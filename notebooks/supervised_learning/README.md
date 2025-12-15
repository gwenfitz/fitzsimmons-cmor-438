# Supervised Learning Notebooks

This directory contains Jupyter notebooks demonstrating **supervised learning
algorithms implemented from scratch** as part of the CMOR 438 project.

Each notebook provides an end-to-end workflow that includes:

- Dataset loading and preprocessing
- Exploratory data analysis (EDA)
- Model implementation and training
- Visualization of model behavior
- Evaluation and interpretation of results

All models used in these notebooks are implemented in the project’s custom
machine learning package. No scikit-learn classifiers or regressors are used.

---

## What Is Supervised Learning?

Supervised learning refers to machine learning tasks where models are trained
using **labeled data**, meaning each observation includes both input features
and a known target value.

The goal is to learn a mapping from inputs to outputs that generalizes to new,
unseen data.

Supervised learning problems fall into two main categories:

- **Classification**: predicting discrete labels
- **Regression**: predicting continuous values

The notebooks in this directory illustrate how these models are implemented
from first principles and how their performance can be analyzed on real
datasets.

---

## 📁 Included Notebooks and Methods

Below is an overview of the supervised learning methods demonstrated in this
directory, organized by subfolder.

---

### 🌳 Decision Tree

**Folder:** `Decision Tree/`  
**Notebook:** `decision_tree_ex.ipynb`

This notebook demonstrates:

- Building a decision tree classifier from scratch
- Impurity-based splitting criteria
- Interpreting recursive partitioning
- Understanding overfitting and depth constraints
- Evaluating classification performance

---

### 🪶 Ensemble Methods

**Folder:** `Ensemble Methods/`  
**Notebook:** `ensemble_methods_example.ipynb`

This notebook explores ensemble learning techniques built from simpler base
models.

Topics include:

- Bagging and random forests
- Variance reduction through ensembling
- Comparing ensemble performance to single decision trees
- Evaluating ensemble accuracy and robustness

Explicit comparisons are made between ensemble methods and their individual
constituents to illustrate why ensembling improves performance.

---

### 📉 Gradient Descent

**Folder:** `Gradient Descent/`  
**Notebook:** `gradient_descent_example.ipynb`

This notebook focuses on optimization rather than a specific model.

Topics covered include:

- Gradient descent from first principles
- Convergence behavior and learning rates
- Visualization of loss minimization
- Relationship between gradient descent and linear/logistic regression

---

### 🔢 k-Nearest Neighbors (KNN)

**Folder:** `knn_example/`  
**Notebook:** `knn_example.ipynb`  
**Dataset:** `iris.csv`

This notebook demonstrates:

- Distance-based classification using k-nearest neighbors
- Feature scaling and distance metrics
- The effect of the choice of \( k \)
- Visualization of decision boundaries
- Classification accuracy on the Iris dataset

---

### 📈 Linear Regression

**Folder:** `Linear Regression/`  
**Notebook:** `linear_regression_boston.ipynb`

This notebook demonstrates:

- Linear regression using gradient descent
- Predicting continuous target variables
- Interpreting learned coefficients
- Visualizing regression fits
- Evaluating performance with error metrics

---

### ✔️ Logistic Regression

**Folder:** `Logistic Regression/`  
**Notebook:** `logistic_regression_pima.ipynb`

This notebook covers:

- Binary classification using logistic regression
- Training from scratch with gradient descent
- Interpreting coefficients and probabilities
- Evaluating performance using accuracy and related metrics

---

### 🧠 Multilayer Perceptron

**Folder:** `Multilayer Perceptron/`  
**Notebook:** `multilayer_perceptron_example.ipynb`

This notebook explains:

- A feedforward neural network implemented from scratch
- Forward propagation and backpropagation
- Nonlinear activation functions
- Training dynamics and loss convergence
- Comparing training and test performance

---

### ➕ Perceptron

**Folder:** `Perceptron/`  
**Notebook:** `perceptron_example.ipynb`

This notebook demonstrates:

- Binary classification using a single-layer perceptron
- The perceptron update rule
- Convergence behavior
- Limitations of linear decision boundaries

---

### 🌲 Regression Trees

**Folder:** `Regression Trees/`  
**Notebook:** `regression_trees_example.ipynb`

This notebook applies a tree-based learning algorithm to a dataset with a
continuous target variable.

Topics include:

- Discretizing continuous targets for classification-style trees
- CART-style tree construction
- Evaluating predictive performance
- Analyzing the effect of tree depth on generalization

---

## 🎯 Purpose of This Directory

The notebooks in this directory are intended to:

- Demonstrate correct usage of the project’s supervised learning algorithms
- Provide clear, reproducible examples for coursework
- Reinforce intuition behind classical machine learning methods
- Serve as reference implementations for testing and validation

Together, these notebooks complement the source code and unit tests by
illustrating how each algorithm behaves in practice.

---

## 📝 Notes

- All models are implemented from scratch in this project.
- No scikit-learn supervised models are used.
- Visualizations are created using Matplotlib.
- Datasets are stored locally within the relevant subfolders or loaded from
  public sources.
- Mathematical expressions are written using GitHub-compatible LaTeX.

## Folder Organization

supervised_learning/
├── Decision Tree/
│   └── decision_tree_ex.ipynb
├── Ensemble Methods/
│   └── ensemble_methods_example.ipynb
├── KNN/
│   └── knn_example.ipynb
├── Linear Regression/
│   └── linear_regression_boston.ipynb
├── Logistic Regression/
│   └── logistic_regression_pima.ipynb
├── Multilayer Perceptron/
│   └── multilayer_perceptron_example.ipynb
├── Perceptron/
│   └── perceptron_example.ipynb
Regression Trees/
│   └── regression_trees_example.ipynb