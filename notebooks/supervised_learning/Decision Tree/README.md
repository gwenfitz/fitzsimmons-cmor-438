# Decision Tree Classifier

This directory contains a Jupyter notebook demonstrating a **Decision Tree
classifier implemented entirely from scratch** using the custom
`rice_ml` package.

The notebook emphasizes both **algorithmic intuition** and **practical
application**, following a standard supervised learning workflow without
reliance on scikit-learn models.

---

## 📘 Notebook Overview

**Notebook:** `decision_tree_ex.ipynb`  
**Model:** Custom `DecisionTree` classifier  
**Dataset:** Wine Quality (Red) — UCI Machine Learning Repository

The notebook walks through the complete process of applying a decision tree
to a real-world dataset, including:

- Dataset loading and inspection
- Exploratory data analysis (EDA)
- Target construction for binary classification
- Training a decision tree from scratch
- Model evaluation on training and test data
- Visualization of decision regions using PCA

---

## 🌳 What Is a Decision Tree?

A Decision Tree is a supervised learning model that predicts outcomes by
recursively partitioning the feature space into regions that are increasingly
homogeneous with respect to the target variable.

At each internal node, the model applies a rule of the form:

- *Is feature* j *less than or equal to some threshold?*

Each leaf node stores class statistics and predicts the most likely class for
observations that fall into that region.

The implementation in this project uses **Gini impurity** to measure node
purity and selects splits greedily to minimize weighted impurity.

---

## 📊 Dataset Description

The Wine Quality (Red) dataset consists of **1,599 Portuguese red wines**,
each described by **11 continuous physicochemical features**, such as acidity,
alcohol content, and sulphates.

The original quality score is an integer between 0 and 10. For this notebook,
the target variable is converted into a **binary classification problem**:

- **1** — good quality wine (quality ≥ 6)  
- **0** — lower quality wine (quality < 6)

The dataset contains **no missing values**, making it well-suited for
tree-based modeling.

---

## 🔍 Exploratory Data Analysis

The notebook includes exploratory analysis to:

- Examine the distribution of wine quality scores
- Visualize feature behavior across the binary target
- Identify relationships between features
- Provide intuition for how the decision tree may split the data

Although decision trees do not require feature scaling, EDA plays an important
role in interpreting learned decision rules.

---

## 🧠 Model Training and Evaluation

A depth-constrained decision tree is trained to balance model complexity and
generalization.

The notebook evaluates:

- Training accuracy
- Test accuracy
- Evidence of overfitting
- The effect of limiting tree depth

This analysis highlights the bias–variance tradeoff inherent in tree-based
models.

---

## 📐 PCA-Based Visualization

Because the dataset has 11 features, direct visualization of decision
boundaries is not possible. Principal Component Analysis (PCA) is used to
project the data into two dimensions **for visualization only**.

A separate decision tree is trained on the PCA-transformed data to illustrate:

- Axis-aligned decision regions
- Approximate class separation
- The geometric behavior of tree-based models

This visualization provides qualitative insight but does not represent the
exact decision logic of the original high-dimensional model.

---

## 🎯 Purpose of This Notebook

This notebook is designed to:

- Demonstrate a decision tree classifier implemented from first principles
- Reinforce intuition behind recursive partitioning and impurity measures
- Show how decision trees behave on real, noisy data
- Serve as a reusable template for tree-based modeling in this project

---

## 📝 Notes

- The decision tree is implemented from scratch in the `rice_ml` package.
- No scikit-learn decision tree models are used.
- Visualizations are created using Matplotlib and Seaborn.
- PCA is used strictly for visualization and interpretation purposes.
- Mathematical expressions are written using GitHub-compatible LaTeX.

This notebook complements the source code and unit tests by providing a clear,
interpretable example of decision tree learning in practice.
