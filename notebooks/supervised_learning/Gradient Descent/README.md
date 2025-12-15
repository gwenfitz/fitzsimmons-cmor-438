# Gradient Descent Regression — Boston Housing Dataset (From Scratch)

## Overview

This notebook demonstrates **gradient descent optimization implemented entirely from scratch** using the `rice_ml` package. We train a **linear regression model** on the classic **Boston Housing dataset**, without using external machine learning libraries such as scikit-learn.

The notebook emphasizes mathematical clarity, modular design, and the role of preprocessing in gradient-based optimization.

---

## Objectives

In this notebook, we:

- Load and inspect the Boston Housing dataset  
- Perform exploratory data analysis (EDA)  
- Standardize features using custom preprocessing utilities  
- Train a linear regression model using gradient descent  
- Evaluate performance using regression metrics  
- Visualize convergence behavior and predictions  

All steps are implemented using **NumPy only**.

---

## Dataset Description

The Boston Housing dataset contains housing values for suburbs of Boston. Each sample corresponds to a census tract and includes **13 numerical features** describing socioeconomic and environmental factors.

**Target variable:**  
- **MEDV** — Median value of owner-occupied homes (in \$1000s)

The dataset contains no missing values and all variables are numeric.

---

## Exploratory Data Analysis

EDA reveals that feature scales vary significantly across predictors (e.g., `TAX`, `RAD`, `CRIM`). Because gradient descent is sensitive to feature magnitude, this motivates **feature standardization** prior to training.

---

## Preprocessing

Before training, we:

- Separate features and target values  
- Standardize all variables to zero mean and unit variance  
- Split the data into training and test sets using a custom `train_test_split` function  

Standardization ensures stable gradients and efficient convergence.

---

## Gradient Descent for Linear Regression

We minimize the **Mean Squared Error (MSE)** loss using gradient descent:

- Gradients are computed analytically
- Weights are updated iteratively using a fixed learning rate
- Convergence is determined by a tolerance threshold

### Design Choice

The gradient is defined using a **closure** that captures the training data. This allows the `GradientDescentND` class to remain fully generic, reusable, and independent of the specific loss function.

---

## Training and Evaluation

The optimizer stores the full weight history, allowing reconstruction of the loss curve to verify convergence. A smoothly decreasing loss confirms correct implementation and appropriate hyperparameters.

Model performance is evaluated using the **R² score** on a held-out test set, achieving a value of approximately **0.77**, indicating that the linear model explains a substantial portion of the variance in housing prices.

---

## Key Takeaways

- Feature scaling is essential for gradient descent  
- Separating optimization from model structure improves design clarity  
- Linear regression captures meaningful trends but cannot model nonlinear effects  
- From-scratch implementations provide valuable insight into optimization behavior  

---

## Conclusion

This notebook demonstrates that gradient descent, when implemented from first principles and paired with proper preprocessing, can successfully train a meaningful regression model. It highlights both the strengths of gradient-based optimization and the limitations of linear models on real-world data.
