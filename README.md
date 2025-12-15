# CMOR 438: Data Science and Machine Learning Repo

# Gwen FitzSimmons

## Overview

This repository contains a custom machine learning package developed for **CMOR 438**.  
The project implements classic **supervised and unsupervised learning algorithms from scratch** using NumPy, organized into a clean and modular Python package called **`rice_ml`**.

The package is paired with structured **Jupyter notebooks** that demonstrate each algorithm on real and synthetic datasets, emphasizing **mathematical intuition, algorithmic transparency, and interpretability** rather than black-box usage.

---

## Project Highlights

This repository showcases:

- Fully custom implementations of core machine learning algorithms  
- A well-structured, installable Python package (`rice_ml`)  
- Separate modules for supervised learning, unsupervised learning, and preprocessing  
- Educational notebooks demonstrating each algorithm step-by-step  
- A comprehensive **pytest test suite** covering all major components  

---

## 🚀 Capabilities

### Supervised Learning

Implemented in `rice_ml/supervised_learning`:

- **Linear Regression**  
- **Logistic Regression**  
- **k-Nearest Neighbors (KNN)**  
- **Perceptron**  
- **Multilayer Perceptron (Neural Network)**  
- **Decision Trees**  
- **Regression Trees**  
- **Gradient Descent utilities**  
- **Basic ensemble methods**  
- **Distance metrics**

These implementations prioritize clarity and correctness over performance optimizations.

---

### Unsupervised Learning

Implemented in `rice_ml/unsupervised_learning`:

- **K-Means Clustering**  
- **DBSCAN**  
- **Principal Component Analysis (PCA)**  
- **Community Detection (Label Propagation)**  

Each method highlights a different notion of structure:
distance, density, variance, and graph connectivity.

---

### Data Processing Utilities

Implemented in `rice_ml/processing`:

- Feature standardization  
- Common preprocessing transformations  
- Post-processing helpers  

These tools are intentionally minimal and designed to illustrate how preprocessing impacts downstream algorithms.

---

## 📁 Repository Structure

High-level structure of the repository:

```text
.
├── notebooks/
│   ├── supervised_learning/
│   └── unsupervised_learning/
│
├── src/
│   └── rice_ml/
│       ├── processing/
│       ├── supervised_learning/
│       ├── unsupervised_learning/
│       └── __init__.py
│
├── tests/
│   ├── test_linear_regression.py
│   ├── test_knn.py
│   ├── test_dbscan.py
│   ├── test_pca.py
│   └── ...
│
├── README.md
├── requirements.txt
├── LICENSE
└── pyproject.toml
```
## Repository Overview

- **`notebooks/`**  
  Contains demonstration notebooks for each algorithm.

- **`src/rice_ml/`**  
  Contains the full from-scratch package implementation.

- **`tests/`**  
  Contains pytest-based unit tests for every major module.

---

## 📘 Notebooks

Each algorithm has a corresponding notebook that walks through:

- Dataset loading and exploration  
- Preprocessing and scaling  
- Training the custom implementation from `rice_ml`  
- Visualization of predictions, clusters, or embeddings  
- Discussion of assumptions, behavior, and limitations  

These notebooks are designed to be **teaching resources**, not just demos.

---

## 🧪 Testing

All major algorithms and utilities are tested using **pytest**.

Tests cover:

- Numerical correctness  
- Input validation and shape handling  
- Edge cases  
- Consistency of outputs  
- Expected behavior on small, known datasets  

To run the test suite:

```bash
pytest -q
```

## 🔧 Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install the package in editable mode:

```bash
pip install -e .
```
Example usage:

```python
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.unsupervised_learning.k_means_clustering import KMeans
from rice_ml.processing.preprocessing import standardize
``` 
## 🎯 Project Goals

This project was built to:
- Deepen understanding of machine learning algorithms by implementing them from first principles
- Practice professional-quality Python package organization
- Integrate testing, documentation, and examples into a single codebase
- Emphasize algorithmic assumptions, limitations, and interpretation
- Cover the full ML workflow: preprocessing → modeling → evaluation → visualization

## 📜 License

This project is intended for educational use as part of CMOR 438.
Refer to the repository for licensing details.

## 👤 Author

Gwenyth FitzSimmons
Rice University — CMOR 438