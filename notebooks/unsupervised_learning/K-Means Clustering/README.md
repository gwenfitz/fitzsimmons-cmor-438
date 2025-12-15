# K-Means Clustering from Scratch

## Overview

K-Means is one of the most widely used **unsupervised learning** algorithms. Its goal is to partition a dataset into **K distinct clusters** such that points within the same cluster are as similar as possible, typically measured using Euclidean distance.

Unlike supervised learning methods, K-Means does **not** rely on labeled data. Instead, it discovers underlying structure by minimizing the distance between data points and their assigned cluster centroids.

In this notebook, we implement K-Means **from scratch** using a custom machine learning package (`rice_ml`) and apply it to a real-world dataset to explore clustering behavior, evaluation techniques, and visualization strategies.

## Objectives

In this notebook, we:

- Implement K-Means using a custom `rice_ml` implementation  
- Apply the algorithm to a real dataset  
- Perform exploratory data analysis (EDA)  
- Highlight the importance of feature scaling  
- Analyze clustering quality using inertia  
- Use the elbow method to select the number of clusters  
- Visualize high-dimensional clusters using PCA  
- Discuss strengths and limitations of K-Means  

## Dataset Description

We use the Wine dataset from the UCI Machine Learning Repository.

The dataset consists of 178 wine samples described by 13 physicochemical features, including:
- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Color intensity
- Hue

Although the dataset includes class labels, they are not used in this notebook, as K-Means is an unsupervised learning algorithm.

Source: UCI Machine Learning Repository – Wine Dataset

## Data Preparation

The class label column is removed before clustering

Only the feature matrix is used for training

Feature scaling is applied using standardization (zero mean, unit variance)

Standardization is critical because K-Means relies on Euclidean distance, and features with larger scales can otherwise dominate the clustering process.

## Exploratory Data Analysis (EDA)

Before clustering, we examine:
- Dataset shape and structure
- Distribution of true class labels (for reference only)
- Differences in feature scales
- Boxplots reveal significant variation in feature magnitudes, reinforcing the need for preprocessing prior to clustering.

## K-Means Algorithm

K-Means proceeds iteratively through the following steps:
1. Initialize K cluster centroids
2. Assign each data point to the nearest centroid
3. Update centroids as the mean of assigned points
4. Repeat until convergence or until a maximum number of iterations is reached

The algorithm minimizes the within-cluster sum of squared distances, commonly referred to as inertia.

## Model Training

The model is trained with:
- n_clusters = 3
- max_iter = 300
- tol = 1e-4
- Fixed random seed for reproducibility

After fitting, we extract:
- Cluster labels
- Cluster centroids
- Inertia

## Inertia and the Elbow Method

Inertia measures how tightly points are clustered around their centroids. Lower inertia indicates more compact clusters, but inertia always decreases as the number of clusters increases.

To select an appropriate value of K, we use the elbow method, plotting inertia as a function of K and identifying where improvements begin to diminish.

For this dataset, the curve begins to flatten around K = 3, suggesting a good balance between model complexity and clustering quality.

## PCA for Visualization

Because the dataset is high-dimensional, we use Principal Component Analysis (PCA) to project the data into two dimensions for visualization.

### Important notes:

- PCA is used only for visualization
- Clustering is performed in the original standardized feature space
- Cluster assignments remain unchanged

## Limitations of K-Means

While K-Means is simple and efficient, it has several limitations:
- Requires specifying the number of clusters in advance
- Assumes roughly spherical clusters
- Sensitive to initialization
- Sensitive to outliers
- Requires feature scaling

## Conclusion

This notebook demonstrates how K-Means clustering can be implemented and applied from scratch to uncover structure in unlabeled data.

Key takeaways include:
- The importance of feature standardization
- How inertia and the elbow method guide cluster selection
- The value of PCA for visualizing high-dimensional clusters
- The strengths and limitations of K-Means as an unsupervised learning method

Overall, K-Means provides a powerful yet accessible approach to exploratory data analysis and unsupervised learning.