# Linear Regression

This directory contains a Jupyter notebook demonstrating **linear regression
implemented entirely from scratch** using the custom `rice_ml` package.

The notebook emphasizes both the **mathematical intuition** behind linear
regression and its **practical application** to a real-world dataset.

---

## 📘 Notebook Overview

**Notebook:** `linear_regression_boston.ipynb`  
**Model:** `LinearRegression` (custom implementation)  
**Dataset:** Boston Housing Dataset (UCI Machine Learning Repository)

This notebook walks through a complete supervised learning workflow, including:

- Dataset loading and inspection
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training a linear regression model using Ordinary Least Squares
- Model evaluation using multiple regression metrics
- Visualization of predictions and residuals
- Interpretation of learned coefficients

All models and utilities are implemented from scratch without using
scikit-learn.

---

## 📊 Dataset Description

The Boston Housing dataset contains **506 observations** describing housing
values in suburban Boston. Each observation consists of **13 numerical
features** related to socioeconomic and environmental factors.

The target variable is:

- **MEDV** — median value of owner-occupied homes (in thousands of dollars)

All features are numeric, and no missing values are present in the dataset.
Feature scales vary significantly, motivating standardization prior to
model training.

---

## 🧠 What Is Linear Regression?

Linear regression models the relationship between a set of input features
and a continuous target variable as a linear combination of those features.

In matrix form, the model assumes:

y = Xβ + ε

where:
- `β` represents the model coefficients
- `ε` represents random noise

In this notebook, model parameters are estimated using **Ordinary Least
Squares (OLS)**, which minimizes the mean squared error between predictions
and observed values.

---

## 🔍 Exploratory Data Analysis

Exploratory analysis is used to:

- Examine feature distributions and ranges
- Understand the distribution of the target variable
- Identify linear relationships between features and the target
- Visualize correlation structure using a heatmap

This analysis provides intuition for which features are likely to have
strong influence on housing prices.

---

## ⚙️ Preprocessing

Before training the model, the data is:

- Separated into feature and target matrices
- Standardized to zero mean and unit variance
- Split into training and test sets using a custom implementation

Standardization improves numerical stability and allows model coefficients
to be interpreted on a common scale.

---

## 📈 Model Training and Evaluation

The linear regression model is trained using the **closed-form normal
equation**, without regularization.

Model performance is evaluated using:

- R² score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Evaluation is performed on both training and test sets to assess
generalization.

---

## 📐 Visualization and Diagnostics

The notebook includes standard diagnostic visualizations, such as:

- Predicted vs. actual target values
- Residuals vs. predicted values

These plots help assess model fit, bias, and variance, and reveal limitations
of the linear modeling assumption.

---

## ⚠️ Model Limitations

Linear regression assumes a linear relationship between features and the
target variable. While the model performs reasonably well on the Boston
Housing dataset, nonlinear relationships and feature interactions are not
captured.

More flexible models such as decision trees or ensemble methods may achieve
higher predictive accuracy, but at the cost of interpretability.

---

## 🎯 Purpose of This Notebook

This notebook is designed to:

- Demonstrate linear regression implemented from first principles
- Reinforce the intuition behind regression modeling
- Provide a clear example of regression diagnostics and evaluation
- Serve as a reusable template for regression analysis in this project

---

## 📝 Notes

- The linear regression model is implemented from scratch in the `rice_ml`
  package.
- No scikit-learn regression models are used.
- Visualizations are created using Matplotlib.
- All mathematical expressions are written in GitHub-compatible format.

This notebook complements the project’s source code and unit tests by
providing a transparent and interpretable example of linear regression in
practice.

