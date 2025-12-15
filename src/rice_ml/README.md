# rice_ml

`rice_ml` is a **from-scratch machine learning library** designed for educational and exploratory purposes. It provides clean, readable implementations of core machine learning algorithms across **supervised learning**, **unsupervised learning**, and **data preprocessing**, with a strong emphasis on mathematical clarity and algorithmic transparency.

This package is intended for learning, experimentation, and deeper understanding of how machine learning algorithms work internally—without relying on black-box frameworks.

---

## Design Philosophy

The guiding principles of `rice_ml` are:

- **From-scratch implementations** (NumPy-based, minimal abstraction)
- **Mathematical transparency** over performance optimization
- **Modular structure** that mirrors standard ML taxonomy
- **Readable code** suitable for coursework and self-study
- **Explicit assumptions and limitations**

This library is not intended to replace production ML frameworks, but to **explain them**.

---

## Package Structure

```text
rice_ml/
├── processing/
│   ├── preprocessing.py
│   └── post_processing.py
│
├── supervised_learning/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── gradient_descent.py
│   ├── knn.py
│   ├── distance_metrics.py
│   ├── perceptron.py
│   ├── multilayer_perceptron.py
│   ├── decision_tree.py
│   ├── regression_trees.py
│   └── ensemble_methods.py
│
├── unsupervised_learning/
│   ├── k_means_clustering.py
│   ├── dbscan.py
│   ├── pca.py
│   └── community_detection.py
│
└── __init__.py

Each module is self-contained and mirrors the structure and behavior of its theoretical counterpart.

## Processing

### `processing`

Utilities for preparing data before and after modeling.

**Includes:**
- Feature standardization  
- Common preprocessing transformations  
- Post-processing helpers for model outputs  

These utilities are intentionally minimal and designed to expose how preprocessing affects downstream algorithms.

---

## Supervised Learning

### `supervised_learning`

Implements classic supervised learning algorithms where labeled data is available.

**Algorithms included:**

- **Linear Regression**  
  Closed-form and gradient-based solutions for regression tasks.

- **Logistic Regression**  
  Binary classification using sigmoid activation and log-loss.

- **Gradient Descent**  
  Generic optimization routines shared across models.

- **k-Nearest Neighbors (kNN)**  
  Distance-based classification with customizable metrics.

- **Distance Metrics**  
  Euclidean and related distance functions used across models.

- **Perceptron**  
  Linear binary classifier using mistake-driven updates.

- **Multilayer Perceptron (MLP)**  
  Feedforward neural network with backpropagation.

- **Decision Trees**  
  Tree-based models using greedy recursive splitting.

- **Regression Trees**  
  Tree-based regression using variance reduction.

- **Ensemble Methods**  
  Foundations for combining multiple weak learners.

These implementations emphasize algorithmic logic over performance tricks.

---

## Unsupervised Learning

### `unsupervised_learning`

Implements algorithms that discover structure **without labels**.

**Algorithms included:**

- **K-Means Clustering**  
  Centroid-based clustering using Euclidean distance.

- **DBSCAN**  
  Density-based clustering with explicit noise detection.

- **Principal Component Analysis (PCA)**  
  Variance-based dimensionality reduction via eigen-decomposition.

- **Community Detection (Label Propagation)**  
  Graph-based clustering using connectivity and local consensus.

These methods illustrate different definitions of similarity:
**distance, density, variance, and connectivity**.

---

## Educational Focus

The `rice_ml` package is especially suited for:

- Machine learning coursework  
- Algorithm walkthroughs and demos  
- Understanding tradeoffs and assumptions  
- Debugging intuition for real-world ML tools  
- Connecting math to code  

Each algorithm is implemented in a way that mirrors its mathematical definition as closely as possible.

---

## Limitations

- Not optimized for large-scale or production use  
- Limited numerical stability safeguards  
- Focuses on clarity rather than speed  
- Assumes clean, well-structured input data  

These tradeoffs are intentional and aligned with the library’s educational goals.

---

## Conclusion

`rice_ml` provides a cohesive, transparent collection of machine learning algorithms implemented from first principles.

By organizing supervised learning, unsupervised learning, and preprocessing under a single consistent interface, the package serves as a practical companion for understanding **how machine learning works**.