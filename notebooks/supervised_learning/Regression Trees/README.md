# Regression Trees (From Scratch)

This directory contains a complete implementation and demonstration of Regression Trees built entirely from scratch using the rice_ml package.

The notebook applies a CART-style regression tree to the California Housing dataset and walks through data exploration, model training, evaluation, and interpretation.

No scikit-learn tree models are used.

## Overview

Regression trees are non-parametric supervised learning models that predict continuous targets by recursively partitioning the feature space and assigning each region a constant prediction.

In this example, we:
1. Load a real-world housing dataset
2. Explore the target distribution and feature scales
3. Train a custom regression tree
4. Evaluate performance using R² and Mean Squared Error (MSE)
5. Visualize predictions and residuals
6. Discuss bias–variance tradeoffs

## What Is a Regression Tree?

A regression tree models a function by splitting the feature space into regions using axis-aligned rules of the form:

feature_j ≤ threshold

Each terminal node (leaf) predicts the mean value of the training samples that fall into that region.

## Model Formulation

### Prediction Rule:

For a leaf containing target values:

{y₁, y₂, ..., yₖ}

the prediction is:

ŷ = (1 / k) * Σ yᵢ

This is the optimal constant predictor under mean squared error.

### Split Criterion (MSE)

At each node, the tree chooses the split that minimizes the weighted Mean Squared Error:

MSE = (1 / n) * Σ (yᵢ − ŷ)²

The algorithm greedily selects the feature and threshold that produce the largest reduction in MSE.

### Stopping Conditions

Tree growth stops when:
- Maximum depth is reached
- Too few samples remain to split
- No split reduces error

These controls prevent uncontrolled growth and excessive variance.

### Dataset

California Housing Dataset

Each row represents a geographic block group from the 1990 U.S. Census.

Features (numerical)
- Median income
- House age
- Average rooms
- Average bedrooms
- Population
- Average occupancy
- Latitude
- Longitude

Target
- Median house value (continuous)

This dataset is well suited for regression trees because:
- Relationships are nonlinear
- Feature interactions are complex
- Interpretability is important

## Exploratory Data Analysis

The notebook examines:
- Target distribution (right-skewed)
- Feature scale differences
- Presence of extreme values

Regression trees do not require feature scaling, since splits depend only on ordering comparisons.

## Model Training

The regression tree is trained using a CART-style greedy algorithm with:
- Maximum depth constraint
- Minimum samples per leaf
- Variance-based splitting criterion

This balances expressiveness and generalization.

## Evaluation Metrics

### R² Score: 
- R² measures the proportion of variance explained by the model: R² = 1 − (Σ (y − ŷ)² / Σ (y − ȳ)²). 
- Values close to 1 indicate strong explanatory power.

### Mean Squared Error (MSE): 
- MSE = (1 / n) * Σ (y − ŷ)². 
- MSE reflects the average squared prediction error.
Large numerical values are expected because housing prices are measured in dollars.

## Visualization

The notebook includes:
- Predicted vs. true value plots
- Residuals vs. predictions
- Diagnostic interpretation of heteroscedasticity

These plots confirm:
- Strong overall fit
- Mildly increasing error for extreme values
- No systematic bias or curvature

## Bias–Variance Tradeoff

Regression trees illustrate the bias–variance tradeoff clearly:
- Shallow trees → high bias, low variance
- Deep trees → low bias, high variance

Tree depth and minimum leaf size directly control this balance.

Because single trees are high-variance models, they are often used as base learners in Bagging and Random Forests.

## Note on Model Performance

The near-perfect R² values observed in this notebook reflect the expressive power of regression trees on structured tabular data under random train–test splits.

This result highlights representational capacity rather than guaranteed real-world generalization. In practice, stronger validation strategies and ensemble methods are used to obtain more robust performance estimates.

## Key Takeaways

Regression trees naturally model nonlinear relationships

No feature scaling is required

Splits are interpretable and rule-based

Tree depth controls overfitting

Single trees are powerful but high-variance models

This notebook complements the classification trees, ensemble methods, and neural network models in the project.