# Unsupervised Learning

## Overview

**Unsupervised learning** focuses on discovering structure in data **without labeled outcomes**. Instead of predicting predefined targets, unsupervised methods analyze patterns, relationships, and geometry inherent in the data itself.

This folder contains a collection of **from-scratch implementations and notebooks** demonstrating core unsupervised learning techniques. Each method addresses a different notion of structure, including distance, density, variance, and connectivity.

All notebooks emphasize **conceptual understanding**, **mathematical intuition**, and **interpretability**, rather than reliance on black-box libraries.

---

## Contents

The `unsupervised_learning` directory contains the following notebooks, each organized in its own subfolder:

### K-Means Clustering
A centroid-based clustering algorithm that partitions data into a fixed number of clusters by minimizing within-cluster variance.

**Key ideas:**
- Distance-based clustering
- Requires specifying the number of clusters
- Sensitive to feature scaling
- Best suited for roughly spherical clusters

---

### DBSCAN Clustering
A density-based clustering algorithm that groups points based on local neighborhood density and explicitly identifies noise.

**Key ideas:**
- Density-based clustering
- No need to specify the number of clusters
- Detects outliers naturally
- Handles non-spherical cluster shapes

---

### Principal Component Analysis (PCA)
An unsupervised dimensionality reduction technique that projects data onto orthogonal directions of maximum variance.

**Key ideas:**
- Variance-based dimensionality reduction
- Identifies latent structure in high-dimensional data
- Useful for visualization, noise reduction, and preprocessing
- Preserves variance, not class separation

---

### Community Detection (Label Propagation)
A graph-based method that identifies communities based on connectivity rather than feature similarity.

**Key ideas:**
- Operates on graph structure, not feature space
- Discovers groups through local label consensus
- No predefined number of communities
- Well-suited for network and relational data

---

## Learning Themes

Across all notebooks, the following themes are emphasized:

- Structure discovery without labels  
- The role of geometry, density, variance, and connectivity  
- Importance of preprocessing and data representation  
- Strengths and limitations of different unsupervised methods  
- Interpretation of results rather than optimization alone  

Each method captures a **different definition of similarity**, highlighting that no single unsupervised technique is universally optimal.

---

## Folder Organization

unsupervised_learning/
├── Community Detection/
│   └── community_detection_example.ipynb
├── DBSCAN/
│   └── dbscan_example.ipynb
├── K-Means Clustering/
│   └── k_means_clustering.ipynb
├── PCA/
│   └── pca_example.ipynb