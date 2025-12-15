# DBSCAN Clustering

## Overview

Density-Based Spatial Clustering of Applications with Noise (**DBSCAN**) is an **unsupervised clustering** algorithm that groups data points based on **local density** rather than distance to a centroid.

Unlike K-Means, DBSCAN:

- Does not require specifying the number of clusters  
- Can identify **arbitrarily shaped** clusters  
- Explicitly detects **noise and outliers**  
- Is well-suited for **non-spherical** cluster structures  

This notebook applies DBSCAN using a from-scratch implementation to illustrate its behavior on a non-linear dataset.

---

## Key Idea

DBSCAN defines clusters as **connected regions of high point density**.

It relies on two parameters:

- **ε (epsilon)**: neighborhood radius  
- **min_samples**: minimum number of points required to form a dense region  

Points are classified as:

- **Core points**: dense points that initiate clusters  
- **Border points**: near dense regions but not dense themselves  
- **Noise points**: isolated points not belonging to any cluster  

Clusters grow by recursively connecting density-reachable points.

---

## Dataset

The notebook uses the **Two Moons** dataset, a common benchmark for evaluating clustering algorithms on complex geometries.

Dataset characteristics:

- Two interleaving crescent-shaped clusters  
- Non-linear, non-spherical structure  
- Two continuous features  
- No meaningful centroid representation  

This dataset highlights scenarios where centroid-based methods fail.

---

## Exploratory Analysis

Exploratory analysis focuses on **spatial structure and local density**, including:

- Visualization of raw data geometry  
- Analysis of local neighborhood density  
- Empirical guidance for selecting ε  

k-nearest neighbor distance plots help identify transitions between dense regions and sparse boundaries, guiding parameter selection.

---

## Clustering Results

DBSCAN successfully:

- Recovers the curved cluster structure  
- Preserves connectivity along each crescent  
- Labels boundary and isolated points as noise  

Clusters emerge naturally from density patterns without requiring a predefined cluster count.

---

## Sensitivity to Parameters

DBSCAN is sensitive to the choice of ε:

- Smaller ε increases noise detection  
- Larger ε can cause cluster merging  

Selecting ε requires balancing cluster connectivity and noise robustness.

---

## Comparison with K-Means

| Method   | Requires K | Detects Noise | Cluster Shape |
|----------|------------|---------------|---------------|
| K-Means  | Yes        | No            | Spherical     |
| DBSCAN  | No         | Yes           | Arbitrary     |

The Two Moons dataset strongly favors DBSCAN due to its irregular, density-driven structure.

---

## Limitations

DBSCAN may struggle when:

- Cluster densities vary significantly  
- Data dimensionality is high  
- Parameter tuning is difficult  

Despite these limitations, DBSCAN remains a powerful method for density-based clustering.

---

## Conclusion

This notebook demonstrates how DBSCAN identifies clusters based on **density and connectivity** rather than centroids.

Key takeaways:

- No need to specify the number of clusters  
- Natural detection of noise points  
- Effective handling of non-linear cluster shapes  

DBSCAN is a strong choice for unsupervised learning problems involving complex geometric structure.
