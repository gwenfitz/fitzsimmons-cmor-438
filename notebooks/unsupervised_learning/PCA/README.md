# Principal Component Analysis (PCA)

## Overview

Principal Component Analysis (**PCA**) is an **unsupervised dimensionality reduction** technique that projects high-dimensional data onto a lower-dimensional space while preserving as much variance as possible.

Rather than selecting or discarding features, PCA constructs **new orthogonal features** (principal components) that capture the dominant patterns in the data.

This notebook demonstrates PCA **from scratch** and applies it to a real-world dataset to illustrate both the mathematical foundations and practical interpretation of PCA.

---

## Key Idea

Given a dataset with many correlated features, PCA:

- Identifies directions of **maximum variance**
- Orders these directions by importance
- Projects the data onto a smaller number of directions
- Preserves most of the original information with fewer dimensions

Each principal component is a **linear combination of the original features**, and components are mutually orthogonal.

---

## Mathematical Intuition

PCA operates through the following steps:

1. **Mean centering**  
   Subtract the feature-wise mean so the data is centered at the origin.

2. **Covariance matrix computation**  
   Measure how features vary together.

3. **Eigen-decomposition**  
   - Eigenvectors define principal directions  
   - Eigenvalues quantify variance along those directions  

4. **Projection**  
   Data is projected onto the top components with the largest eigenvalues.

This produces a lower-dimensional representation that retains maximal variance.

---

## Dataset

The notebook uses the **Wine Quality** dataset, which contains physicochemical measurements of red wines.

Dataset characteristics:

- Moderate dimensionality  
- Continuous numeric features  
- Strong correlations between variables  
- No requirement for labels  

These properties make it well-suited for demonstrating PCA.

---

## Exploratory Analysis

Before applying PCA, exploratory analysis examines:

- Feature distributions  
- Differences in feature scale  
- Correlation structure  

Key observations include:

- Features vary widely in scale and variance  
- Several features are strongly correlated  
- Outliers and skewed distributions are present  

Because PCA is variance-based, **feature standardization** is essential prior to applying PCA.

---

## Explained Variance

Each principal component explains a portion of the total variance.

Important observations:

- The first few components explain a large fraction of variance  
- Later components contribute diminishing information  
- Variance is concentrated in a low-dimensional subspace  

This indicates redundancy among features and motivates dimensionality reduction.

---

## Scree and Cumulative Variance Analysis

Two common tools are used to guide component selection:

- **Scree plot**: identifies diminishing returns in additional components  
- **Cumulative explained variance**: determines how many components are needed to retain a desired percentage of variance  

These plots reveal an effective dimensionality much smaller than the original feature space.

---

## Low-Dimensional Projection

The data is projected onto the first two principal components to visualize its structure.

Key insights:

- PCA captures dominant variance directions  
- No clear cluster separation is visible in 2D  
- PCA preserves variance, not class separation  

This highlights an important distinction between **dimensionality reduction** and **clustering**.

---

## Reconstruction and Information Loss

PCA allows approximate reconstruction of the original data from reduced components.

Reconstruction error:

- Quantifies information loss  
- Decreases as more components are retained  
- Is non-zero when dimensionality is reduced  

This provides a concrete measure of the tradeoff between compression and fidelity.

---

## When to Use PCA

PCA is well-suited for:

- Visualizing high-dimensional data  
- Reducing noise  
- Preprocessing for clustering or classification  
- Mitigating multicollinearity  

PCA is **not ideal** when:

- Feature interpretability is critical  
- Data contains strong non-linear structure  
- Variables are categorical  

---

## Limitations

Despite its usefulness, PCA has limitations:

- Captures only linear relationships  
- Sensitive to outliers  
- Maximizes variance, not task-specific performance  
- Principal components can be difficult to interpret  

---

## Conclusion

This notebook demonstrates PCA as a foundational unsupervised learning technique.

Key takeaways:

- PCA reduces dimensionality by exploiting correlation  
- Standardization is essential for meaningful results  
- Most variance often lies in a small number of directions  
- PCA is invaluable for visualization, preprocessing, and noise reduction  

PCA remains a core tool in modern machine learning pipelines, particularly as a precursor to clustering and exploratory analysis.
