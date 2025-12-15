# Example Notebooks

This directory contains Jupyter notebooks that demonstrate the use of the
custom machine learning algorithms implemented in the `rice_ml` package.
All notebooks are designed to provide **end-to-end, reproducible examples**
that connect theory, implementation, and empirical behavior.

Each notebook emphasizes **algorithmic transparency** and **conceptual
understanding**, with all models implemented from scratch using NumPy
(no scikit-learn or SciPy models are used).

---

## 📘 Supervised Learning

**Folder:** `supervised/`

These notebooks demonstrate supervised learning algorithms implemented in
the `rice_ml` package, including:

- k-Nearest Neighbors (classification and regression)
- Logistic Regression
- Decision Trees
- Ensemble methods (Bagging, Random Forests, Voting Classifiers)
- Multilayer Perceptron (neural network)

Each supervised learning notebook includes:

- Dataset description and variable overview
- Exploratory data analysis (EDA)
- Preprocessing and feature scaling
- Model training using custom implementations
- Quantitative evaluation (accuracy, precision, etc.)
- Visualizations and interpretation of results
- Comparisons between base learners and ensemble methods when applicable

These notebooks are designed to illustrate both **how to use the models**
and **why the algorithms behave as they do**.

---

## 🔍 Unsupervised Learning

**Folder:** `unsupervised_learning/`

These notebooks focus on unsupervised learning and structure discovery,
including:

- Density-based clustering (DBSCAN)
- Dimensionality reduction (PCA)
- Graph-based community detection (Label Propagation)
- K-Means Clustering

Each unsupervised notebook provides:

- Intuition and mathematical motivation for the algorithm
- Dataset exploration and geometric interpretation
- Step-by-step application using `rice_ml` implementations
- Visualizations of clusters, embeddings, or communities
- Sensitivity and stability analysis when appropriate

Special emphasis is placed on comparing **feature-space clustering**
(e.g., DBSCAN) with **graph-based structure discovery**
(e.g., community detection).

---

## 🎯 Purpose of This Directory

The notebooks in this folder are intended to:

- Demonstrate correct and idiomatic usage of the `rice_ml` package
- Provide clear, well-documented examples for coursework and review
- Connect theoretical concepts to practical implementations
- Highlight the strengths and limitations of each algorithm
- Complement the unit tests found in the `tests/` directory

Together, these notebooks serve as both **tutorials** and **validation tools**
for the algorithms implemented in this project.

---

## 📝 Notes

- All algorithms used in the notebooks are implemented from scratch in
  the `rice_ml` package.
- Datasets are either stored locally within each notebook folder or
  downloaded from public sources (e.g., UCI Machine Learning Repository).
- Visualizations are created using Matplotlib.
- Mathematical expressions are written in GitHub-compatible LaTeX.

These notebooks are meant to be read alongside the source code and tests,
providing a complete picture of the design, implementation, and behavior
of classical machine learning algorithms.
