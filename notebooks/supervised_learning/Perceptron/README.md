# Perceptron Classifier — Ionosphere Dataset

This directory contains a Jupyter notebook demonstrating the Perceptron algorithm implemented entirely from scratch using the rice_ml package. No scikit-learn models are used.

The notebook applies the Perceptron to the Ionosphere dataset, a real-world radar signal classification task, and focuses on linear decision boundaries, mistake-driven learning, and interpretability.

## What This Notebook Covers

The notebook presents a complete supervised learning workflow:
- Loading a real dataset from a public URL
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training a Perceptron classifier from scratch
- Evaluating classification accuracy
- Visualizing high-dimensional data using PCA
- Interpreting results through theory and geometry

All algorithms and utilities come from the custom rice_ml package.

## Model Overview: The Perceptron

The Perceptron is one of the earliest learning algorithms in machine learning. It learns a linear decision boundary of the form:

f(x) = w · x + b


Predictions are made as:

f(x) >= 0 → class 1

f(x) < 0 → class 0

Learning Rule

For each training example:

If the prediction is correct → no update

If the prediction is incorrect → update weights and bias

w = w + η (y − ŷ) x
b = b + η (y − ŷ)

where:

η is the learning rate
y is the true label
ŷ is the predicted label

### Key Properties

Updates occur only on misclassified samples

No explicit loss function is minimized

Guaranteed to converge only if the data are linearly separable

## Dataset Overview

Dataset: Ionosphere
Source: UCI Machine Learning Repository
Task: Binary classification

Characteristics
- Samples: 351
- Features: 34 continuous numerical values

Target:
- 1 → good radar return
- 0 → bad radar return

No missing values

Moderate class balance

This dataset is useful for studying linear classifiers, but the classes are not perfectly separable, which limits Perceptron performance.

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:
- Target class balance
- Feature scale differences
- Identifying the need for preprocessing

Because Perceptron updates are proportional to feature values, differences in feature scale can strongly influence learning behavior.

## Preprocessing

Before training:
- Features are standardized to zero mean and unit variance

This ensures:
- Balanced feature contributions
- Stable updates
- Faster convergence

The dataset is then split into training and test sets using a custom train_test_split function.

## Model Evaluation

Model performance is evaluated using classification accuracy:

Accuracy = (number of correct predictions) / (total samples)

## Visualization with PCA

Because the dataset has 34 features, direct visualization is not possible.

The notebook applies Principal Component Analysis (PCA) to project the data into two dimensions for visualization only.

Important notes:
- PCA is used only for visualization
- The Perceptron is trained in the full feature space
- Overlap in the PCA projection explains why accuracy is limited

## Limitations of the Perceptron

While simple and interpretable, the Perceptron has important limitations:
- Learns only linear decision boundaries
- Cannot solve non-linearly separable problems
- Produces no probabilistic outputs
- Sensitive to noisy or overlapping data

These limitations motivate more advanced models such as:
- Logistic Regression
- Support Vector Machines
- Neural Networks

## Purpose of This Notebook

This notebook is designed to:
- Demonstrate a correct from-scratch implementation of the Perceptron
- Reinforce the concept of linear separability
- Provide intuition for gradient-based learning
- Serve as a foundational supervised learning example
- Support comparison with more advanced classifiers in the project

## Notes

All models are implemented using the rice_ml package

No scikit-learn classifiers are used

Visualizations rely on Matplotlib

This notebook complements logistic regression and ensemble examples in the repository