# Community Detection with Label Propagation

## Overview

**Community detection** is an unsupervised learning task that identifies groups of nodes in a graph such that connections are **dense within groups** and **sparse between groups**.

Unlike clustering in Euclidean feature space, community detection operates purely on **graph connectivity** rather than distances between feature vectors.

This notebook demonstrates **Label Propagation**, a classic community detection algorithm, using a fully from-scratch implementation. The method is applied to a real-world social network dataset to uncover community structure directly from graph topology.

---

## Key Idea

Given a graph with nodes and edges, community detection aims to partition nodes into groups where:

- Nodes within the same group are highly connected  
- Nodes across different groups have relatively few connections  

Label Propagation achieves this by allowing labels to **diffuse through dense regions** of the graph until stable communities emerge.

---

## Label Propagation Algorithm

Label Propagation is an iterative, local algorithm:

1. Assign each node a unique label  
2. At each iteration, update a node’s label to the **most frequent label among its neighbors**  
3. Repeat until labels stabilize or a maximum number of iterations is reached  

Key properties:

- No need to specify the number of communities  
- Relies only on graph connectivity  
- Communities emerge naturally through majority voting  

---

## Dataset

The notebook uses the **Facebook Large Page–Page Network** dataset from the UCI Machine Learning Repository.

Dataset characteristics:

- Nodes represent Facebook pages  
- Edges represent mutual links between pages  
- Large, sparse social network  
- Node metadata is available for interpretation but not required  

To make analysis tractable and visualizable, the notebook focuses on a **subgraph induced by high-degree nodes**, preserving dense and informative structure.

---

## Exploratory Graph Analysis

Exploratory analysis focuses on **network structure**, including:

- Number of nodes and edges  
- Degree distribution  
- Presence of hubs and heavy-tailed connectivity  

Social networks typically exhibit hub nodes with very high degree. These hubs often anchor communities and motivate subgraph extraction for analysis.

---

## Community Detection Results

Label Propagation is applied to the induced subgraph using its adjacency matrix.

Key observations:

- A small number of large communities dominate  
- Several smaller communities also emerge  
- Community sizes arise naturally from graph structure  

Community labels are **categorical identifiers** with no inherent ordering. Only label equality matters.

---

## Connectivity and Communities

Community assignments correlate strongly with connectivity:

- High-degree nodes often anchor large communities  
- Low-degree nodes may attach to nearby communities or form small groups  

This confirms that Label Propagation responds directly to graph structure rather than randomness.

---

## Visualization via Spectral Embedding

Because communities are defined on graphs rather than feature space, visualization uses a **spectral embedding** derived from the graph Laplacian.

Key ideas:

- The graph Laplacian encodes connectivity structure  
- Eigenvectors associated with small eigenvalues provide a low-dimensional embedding  
- Nodes in the same community cluster together in this embedding  

The visualization offers a geometric interpretation of communities derived purely from topology.

---

## Stability Analysis

Label Propagation can be sensitive to update order and randomness.

To assess robustness:

- The algorithm is rerun with different random seeds  
- The number of detected communities is compared across runs  

Results show relatively stable community counts, especially when damping is introduced to reduce oscillations. This suggests the detected structure reflects genuine graph organization.

---

## Comparison with DBSCAN

| Method              | Input Representation | Groups Based On  | Handles Noise |
|---------------------|---------------------|------------------|---------------|
| DBSCAN              | Feature space        | Density          | Yes           |
| Community Detection | Graph                | Connectivity     | No            |

DBSCAN discovers structure in metric spaces, while community detection uncovers structure in **relational data**.

---

## Limitations

Label Propagation may struggle when:

- Graphs are very sparse  
- Community structure is weak or ambiguous  
- Multiple equally valid partitions exist  

Because the algorithm is local and greedy, different runs may converge to different solutions.

---

## Conclusion

This notebook demonstrates community detection using **Label Propagation** on a real-world graph.

Key takeaways:

- Community detection operates on graphs, not feature vectors  
- Label propagation discovers structure through local consensus  
- Dense connectivity naturally yields stable communities  
- No prior knowledge of the number of communities is required  

Together with clustering methods like DBSCAN, community detection provides a complementary perspective on unsupervised structure discovery in complex data.
