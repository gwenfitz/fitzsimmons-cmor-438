# k-Nearest Neighbors (KNN)

This directory contains a Jupyter notebook demonstrating the **k-Nearest
Neighbors (KNN)** algorithm for both **classification** and **regression**,
implemented entirely from scratch using the `rice_ml` package.

The notebook emphasizes geometric intuition, distance-based learning, and the
bias–variance tradeoff inherent in instance-based models.

---

## 📘 Notebook Overview

**Notebook:** `knn_example.ipynb`  
**Models:** `KNNClassifier`, `KNNRegressor` (custom implementations)

This notebook provides an end-to-end workflow including:

- Dataset loading and preprocessing
- Exploratory data analysis (EDA)
- Feature standardization for distance-based learning
- KNN classification on a real multiclass dataset
- Visualization of decision boundaries using 2D slices
- KNN regression on a nonlinear function
- Empirical demonstration of the bias–variance tradeoff

All algorithms are implemented from scratch without using scikit-learn.

---

## 📊 Datasets Used

### Iris Dataset (Classification)

The Iris dataset consists of 150 samples of iris flowers from three species.
Each sample includes four continuous features:

- sepal length
- sepal width
- petal length
- petal width

The target variable is the flower species (three classes).  
This dataset is well-suited for KNN due to its low dimensionality and strong
class separation in petal features.

---

### Synthetic Sine Wave (Regression)

A one-dimensional synthetic dataset is generated using a noisy sine function.
This example illustrates how KNN regression can approximate nonlinear
functions without assuming a parametric form.

---

## 🧠 What Is k-Nearest Neighbors?

KNN is a **non-parametric, instance-based learning algorithm**. Instead of
learning explicit model parameters, it stores the training data and makes
predictions based on distances to nearby points.

For a query point:

1. Compute distances to all training points
2. Select the k nearest neighbors
3. Aggregate their labels (classification) or values (regression)

Distance-based weighting is used so that closer neighbors contribute more to
the prediction than distant ones.

---

## 🔍 Exploratory Data Analysis

Exploratory analysis focuses on understanding:

- Feature distributions
- Class separability
- Relative feature scales

Univariate plots and pairwise visualizations reveal that petal features provide
strong class separation, while sepal features overlap significantly. This
explains both the strong performance of KNN on the Iris dataset and the
limitations of low-dimensional visualizations.

---

## ⚙️ Preprocessing

Because KNN relies directly on distance calculations, feature scaling is
critical.

- Features are standardized to zero mean and unit variance
- Training statistics are used to scale both training and test data to avoid
  data leakage

This ensures that all features contribute equally to distance computations.

---

## 📈 KNN Classification

The KNN classifier is evaluated on the Iris dataset using Euclidean distance
and distance-based weighting.

Model performance is measured using classification accuracy on both training
and test sets. Results demonstrate that KNN achieves high accuracy on
well-structured, low-dimensional data.

---

## 📐 Decision Boundary Visualization

Because the Iris dataset has four features, direct visualization is not
possible. To provide geometric intuition:

- Two features are visualized at a time
- Remaining features are fixed at their mean values
- Predictions are evaluated over a dense grid

This produces a 2D slice of the full decision boundary.

These visualizations illustrate both the flexibility of KNN and the limitations
of projecting high-dimensional decision rules into lower dimensions.

---

## 📉 KNN Regression

KNN regression is demonstrated on a noisy sine wave dataset.

Key observations:

- KNN regression naturally captures nonlinear patterns
- Predictions become smoother as k increases
- Larger k increases bias but reduces variance

This example highlights the strength of KNN as a local function approximator.

---

## ⚖️ Bias–Variance Tradeoff

The complexity of KNN is controlled directly by the number of neighbors k.

- Small k → low bias, high variance
- Large k → higher bias, lower variance

The notebook explicitly compares models with different k values to demonstrate
this tradeoff empirically, reinforcing the theoretical discussion.

---

## ⚠️ Limitations of KNN

While simple and intuitive, KNN has important limitations:

- Prediction time grows with dataset size
- Performance degrades in high-dimensional spaces
- Requires careful feature scaling
- Sensitive to the choice of distance metric and k

These limitations motivate the use of parametric and ensemble methods for
larger or more complex datasets.

---

## 🎯 Purpose of This Notebook

This notebook is designed to:

- Demonstrate KNN implemented entirely from scratch
- Reinforce geometric intuition behind distance-based learning
- Illustrate the bias–variance tradeoff empirically
- Serve as an educational reference for instance-based models
- Provide a reusable template for KNN experiments in this project

---

## 📝 Notes

- All models are implemented in the `rice_ml` package
- No scikit-learn classifiers or regressors are used
- Visualizations rely on Matplotlib and Seaborn
- All mathematical expressions are written in GitHub-compatible format

This notebook complements the project’s supervised learning suite by providing
a clear and interpretable example of distance-based machine learning.
