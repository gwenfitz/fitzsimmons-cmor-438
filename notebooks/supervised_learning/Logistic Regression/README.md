# Logistic Regression — Pima Indians Diabetes Dataset

This directory contains a Jupyter notebook demonstrating binary logistic regression implemented entirely from scratch using the rice_ml package. No scikit-learn models are used.

The notebook applies logistic regression to the Pima Indians Diabetes dataset, a classic and challenging benchmark in medical classification, and emphasizes interpretability, probability modeling, and proper evaluation.

## What This Notebook Covers

The notebook provides a complete supervised learning workflow:
- Loading a real-world medical dataset from a public URL
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training logistic regression using gradient descent
- Evaluating performance with multiple metrics
- Interpreting results from both statistical and ML perspectives

All components are implemented from scratch using the rice_ml package.

## Dataset Overview

**Dataset:** Pima Indians Diabetes
**Source:** UCI / public GitHub mirror
**Task:** Binary classification (diabetes prediction)

### Features

Each observation represents a patient with the following measurements:
- pregnancies — number of pregnancies
- glucose — plasma glucose concentration
- blood_pressure — diastolic blood pressure
- skin_thickness — triceps skin fold thickness
- insulin — 2-hour serum insulin
- bmi — body mass index
- diabetes_pedigree — genetic risk score
- age — age in years

Target: 
- 1 → diabetes
- 0 → no diabetes

### Data Notes

All features are numeric

Several features contain zero values that represent missing or implausible measurements

Feature scales differ substantially

The dataset is noisy and not linearly separable

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:
- Feature distributions (histograms)
- Scale differences across variables (boxplots)
- Identifying skewness, outliers, and measurement issues

These observations motivate feature standardization, which is critical for stable gradient-based optimization.

## Preprocessing

Before training the model:
- Features and labels are separated
- Features are standardized to zero mean and unit variance using a custom standardize function
- The data is split into training and test sets using a custom train_test_split implementation

This ensures fair evaluation and numerical stability.

## Logistic Regression Model

The custom LogisticRegression class implements:
- Sigmoid activation for probabilistic outputs
- Binary cross-entropy (log-loss)
- Gradient descent optimization
- L2 regularization support
- Probability predictions and hard classification

Logistic regression models the probability of diabetes directly, making it especially useful for risk assessment in medical settings.

## Evaluation Metrics

The notebook evaluates model performance using:
- Accuracy — overall classification correctness
- Confusion Matrix — insight into false positives and false negatives
- ROC Curve — performance across classification thresholds
- AUC — threshold-independent measure of discriminative power

Special emphasis is placed on ROC–AUC, which is more informative than accuracy for imbalanced and noisy datasets.

## Interpreting Performance

Accuracy on this dataset is intentionally modest. This reflects:
- Significant class overlap
- Measurement noise
- Linear decision boundary limitations
- The presence of implicit missing values

Logistic regression is presented as an interpretable baseline, not a high-capacity model. Ensemble and tree-based methods are shown elsewhere in the project to achieve higher accuracy.

## Purpose of This Notebook

This notebook is designed to:
- Demonstrate a correct from-scratch implementation of logistic regression
- Highlight the importance of preprocessing and evaluation choices
- Reinforce the probabilistic interpretation of classification models
- Serve as a baseline comparison for more complex models
- Provide a clear, reproducible example for INDE 577 coursework

## Notes

All models are implemented using the rice_ml package

No scikit-learn classifiers are used

Visualizations rely on Matplotlib

The notebook complements unit tests and ensemble examples elsewhere in the repository