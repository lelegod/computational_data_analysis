# Week 9 — Unsupervised Clustering

## Overview
This week covers unsupervised clustering methods: K-means, K-medoids, hierarchical clustering (agglomerative and divisive), Gaussian Mixture Models (GMM) with the EM algorithm, and cluster validation methods (silhouette, gap statistic). Clustering is fundamentally different from supervised classification: there is no response variable y, and clustering will ALWAYS produce a grouping — even on random data.

---

## 1. K-means Clustering

### Key Concepts
- Partitions $n$ observations into $K$ clusters by minimizing within-cluster variance.
- Requires pre-specifying $K$.
- Each observation belongs to exactly one cluster (hard assignment).
- Algorithm is iterative and not guaranteed to find the global optimum — result depends on initialization.
- Works with Euclidean distance.

### Algorithm
1. Initialize: randomly assign observations to $K$ clusters (or choose $K$ random centroids).
2. **Assignment step:** assign each observation to the nearest centroid (by Euclidean distance).
3. **Update step:** recalculate each centroid as the mean of its assigned observations.
4. Repeat steps 2–3 until assignments do not change (convergence).

### Objective Function

$$\min \; \sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2$$

- $\mu_k$: centroid (mean) of cluster $k$.
- $C_k$: set of observations in cluster $k$.

### Properties
- Favors **convex, spherical** clusters of similar size.
- Sensitive to outliers (centroids pulled by extreme values).
- Multiple random restarts recommended to avoid local optima.
- K-means converges but not necessarily to the global optimum.

---

## 2. K-medoids Clustering

### Key Concepts
- Similar to K-means, but cluster centers are **actual data points** (medoids), not means.
- More **robust to outliers** than K-means.
- Works with any distance measure (not just Euclidean).
- Common implementation: PAM (Partitioning Around Medoids).

---

## 3. Hierarchical Clustering

### Key Concepts
- Produces a **dendrogram** (tree structure) rather than a fixed set of clusters.
- No need to pre-specify $K$ — cut the dendrogram at any level to get any number of clusters.
- Two strategies:
  - **Agglomerative (bottom-up):** Start with each observation as its own cluster; merge most similar clusters at each step.
  - **Divisive (top-down):** Start with all observations in one cluster; split recursively.
- The **linkage criterion** defines how inter-cluster distance is computed.

### Linkage Methods (how cluster-cluster distance is defined)
- **Single linkage:** Distance between clusters = distance between their **closest** pair of points. Tends to create chained clusters.
- **Complete linkage:** Distance between clusters = distance between their **farthest** pair of points. Tends to create compact, balanced clusters.
- **Average linkage:** Distance = average of all pairwise distances between clusters.
- **Ward linkage:** Merges clusters that minimize the total within-cluster variance increase. Can only be used with **Euclidean distance**. Tends to give a good compromise between balanced and unbalanced clusters.

### Bottom-up Example (Cities)
Starting distances: Copenhagen–Edinburgh = 983 km (closest pair).
- Step 1: Merge {Copenhagen} and {Edinburgh} (smallest distance = 983 km).
- Step 2 (single linkage): Use minimum distance to remaining clusters.
  - {Cph, Edi}–Reykavik: $\min(2107, 1374) = 1374$ → merge → {Cph, Edi, Reykavik}
  - (complete linkage would use max: 2107, giving different merge order)
- Step 3: Final merge with {Nuuq}.

### Dendrogram Interpretation
- Height of merge = dissimilarity at which two clusters joined.
- Cut the dendrogram horizontally → number of clusters = number of branches below the cut.
- Microarray example: Ward-linkage, Euclidean distance, dendrogram cut at 10 nodes.

### Two-Way Clustering (Biclustering)
- Cluster both **observations (rows)** and **features (columns)** simultaneously.
- Visualized as a heatmap with dendrograms on both axes.
- Python: `SpectralBiclustering` in scikit-learn.

---

## 4. Gaussian Mixture Models (GMM) and EM Algorithm

### Key Concepts
- **Probabilistic** model giving **soft** (probabilistic) cluster assignments.
- Each observation comes from one of $K$ Gaussian distributions.
- An unobserved (latent) random variable $Z_i$ selects which Gaussian generates observation $x_i$.
- Unknown parameters: means $\mu_j$, covariances $\Sigma_j$, mixing proportions $\pi_j$.

### Model Specification
- $X_i \sim \mathcal{N}(\mu_j, \Sigma_j)$ if $Z_i = j$
- $P(Z_i = j) = \pi_j$ (mixing proportions)
- $\pi_1 + \pi_2 + \cdots + \pi_K = 1$

### Parameter Vector

$$\theta = (\pi_1, \ldots, \pi_K, \mu_1, \ldots, \mu_K, \Sigma_1, \ldots, \Sigma_K)$$

$$Z = (Z_1, \ldots, Z_n) \quad \text{(latent variables, unknown)}$$

### Joint Log-Likelihood

$$\ell(\theta; x, Z) = \log p_\theta(x, Z) = \sum_i \sum_j \mathbf{1}\{Z_i=j\} \bigl(\log \mathcal{N}(x_i; \mu_j, \Sigma_j) + \log \pi_j\bigr)$$

Maximum likelihood estimate: $\hat{\theta}_{ML} = \arg\max_{\theta,Z} \ell(\theta; x, Z)$. Finding the solution is simplified by the **EM algorithm**.

### The EM Algorithm for GMM

**Step 0 — Initialize:** Choose initial values for $\mu_j$, $\Sigma_j$, $\pi_j$.

**Step E (Expectation) — Compute posterior probabilities:**

$$\gamma_{ij} = P_{\theta^{(k)}}(Z_i = j \mid x_i) = \frac{\pi_j^{(k)} \mathcal{N}(x_i; \mu_j^{(k)}, \Sigma_j^{(k)})}{\sum_{j'} \pi_{j'}^{(k)} \mathcal{N}(x_i; \mu_{j'}^{(k)}, \Sigma_{j'}^{(k)})}$$

- $\gamma_{ij}$ = soft assignment of observation $i$ to cluster $j$ (probability between 0 and 1).
- Uses Bayes' formula: prior × likelihood / marginal.

**Step M (Maximization) — Update parameters using $\gamma_{ij}$:**

$$\mu_j^{(k+1)} = \frac{\sum_i \gamma_{ij} x_i}{\sum_i \gamma_{ij}}$$

$$\Sigma_j^{(k+1)} = \frac{\sum_i \gamma_{ij} (x_i - \mu_j)(x_i - \mu_j)^T}{\sum_i \gamma_{ij}}$$

$$\pi_j^{(k+1)} = \frac{1}{n} \sum_i \gamma_{ij}$$

- These are **weighted** versions of the standard MLE formulas, weighted by soft assignments.

**Step 4 — Iterate until convergence.**

### Q-function (theoretical basis of M-step)

$$Q(\theta \mid \theta^{(k)}) = \mathbb{E}_{Z \mid x,\theta^{(k)}}[\log p_\theta(x, Z)] = \sum_i \sum_j p_{\theta^{(k)}}(Z_i=j \mid x_i) \bigl(\log \mathcal{N}(x_i; \mu_j, \Sigma_j) + \log \pi_j\bigr)$$

The M-step maximizes this Q-function with respect to $\theta$.

### GMM Properties
- **Soft assignments:** Each point has a probability of belonging to each cluster (unlike K-means).
- **Flexible covariance:** Each cluster can have its own full covariance matrix $\Sigma_j$.
- K-means is a special case of GMM with equal, spherical covariances and hard assignments.
- **High-dimensional tricks ($p \gg n$):**
  - Share covariance: $\Sigma = \Sigma_1 = \cdots = \Sigma_K$ (fewer parameters)
  - Diagonal covariance: $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_K^2)$
  - Regularize: $\Sigma = \Sigma + \lambda I$
  - Use only first few PCA dimensions

---

## 5. Cluster Validation: Selecting K

### Why This Is Hard
- **Clustering will ALWAYS generate a grouping** — even on completely random data.
- Use domain knowledge to validate that structure is meaningful.
- Cross-validation does not apply directly to unsupervised clustering.

### Methods for Selecting K

#### Silhouette Method (Heuristic)
Measures how well each point fits its assigned cluster vs. the neighboring cluster.

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

Where:
- $a(i)$ = average distance from point $i$ to all other points in **the same cluster** (cohesion).
- $b(i)$ = average distance from point $i$ to all points in the **neighboring cluster** (the nearest cluster $i$ is NOT a member of).
- $s(i) \in [-1, 1]$: closer to 1 = well clustered; closer to −1 = mis-clustered; 0 = on boundary.

**Decision rule:**
- Select $K$ where all clusters have observations above average silhouette width, OR
- Select $K^*$ with the **maximum average silhouette** width.

**Properties:**
- Favors convex and spherical clusters.
- Outlier sensitive.
- Reference: Rousseeuw 1987.

#### Gap Statistic
Compares the log within-cluster dissimilarity to what is expected for **uniformly distributed** data (via Monte Carlo simulation with 20 samples).

**Within-cluster dissimilarity:**

$$D_\ell = \sum_{\substack{C_k(i)=\ell \\ C_k(j)=\ell}} \|x_i - x_j\|^2 = N_l \sum_{C_k(i)=\ell} \|x_i - \bar{x}_l\|^2$$

$$W_k = \sum_\ell \frac{1}{2N_l} D_\ell$$

Where $C_k(i) = \ell$ means observation $i$ is in cluster $\ell$ when there are $k$ clusters, and $N_l$ is cluster size.

**Gap Statistic:**

$$G(K) = \log(U_k) - \log(W_k)$$

Where:
- $U_k$ = within-cluster dissimilarity for simulated uniform data (mean over 20 samples)
- $W_k$ = within-cluster dissimilarity for actual data

**Selection rule:**

$$K^* = \arg\min_k \{ K \mid G(K) \geq G(K+1) - s'_{K+1} \}$$

Where:

$$s'_{K+1} = \text{std}(\log(U_K)) \cdot \sqrt{1 + 1/20}$$

Choose smallest $K$ where gap is large enough relative to next gap (within simulation variability).

**Properties:**
- Works for K-means, K-medoids, and hierarchical clustering.
- Works with different measures of within-cluster dissimilarity.
- Based on simulation — may differ between runs.
- For zip code data example: Gap statistic → $K^* = 9$.

#### Goodness-of-Fit (for GMM)
- **AIC** (Akaike Information Criterion): penalizes model complexity.
- **BIC** (Bayesian Information Criterion): heavier penalty for complexity → prefers simpler models.
- Chi-squared statistics, Kolmogorov-Smirnov statistics.
- Plot BIC/AIC vs. number of clusters; choose the elbow or minimum.

#### Biological/Physical Interpretation
- Domain knowledge often the most reliable guide.

---

## 6. Other Clustering Methods

- **Louvain/Leiden clustering:** Network-based. Standard pipeline in scRNA-seq: PCA + Louvain/Leiden.
- **HDBSCAN:** Density-based hierarchical clustering. Can find clusters of arbitrary shape.
- **Deep clustering:** Uses neural networks for representation learning + clustering.
- **Biclustering (Two-way):** Cluster both rows and columns (SpectralBiclustering in sklearn).

---

## 7. Key Warning

**Clustering will ALWAYS generate a grouping, even on random data.**
- Even when data has no cluster structure, K-means/hierarchical will return clusters.
- It is the application and domain knowledge that tells whether the structure is meaningful.
- Always validate clusters using external information or the above methods.

---

## 8. Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| K-means objective | $\min \sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2$ |
| Within-cluster dissimilarity | $W_k = \sum_\ell \frac{1}{2N_l} D_\ell$ |
| $D_\ell$ (pairwise) | $D_\ell = N_l \sum_{i \in C_l} \|x_i - \bar{x}_l\|^2$ |
| Gap statistic | $G(K) = \log(U_k) - \log(W_k)$ |
| Gap selection rule | $K^* = \arg\min_k \{K \mid G(K) \geq G(K+1) - s'_{K+1}\}$ |
| $s'_{K+1}$ | $\text{std}(\log(U_K)) \cdot \sqrt{1+1/20}$ |
| Silhouette | $s(i) = (b(i)-a(i)) / \max\{a(i),b(i)\}$ |
| GMM E-step | $\gamma_{ij} = \pi_j \mathcal{N}(x_i;\mu_j,\Sigma_j) / \sum_{j'} \pi_{j'} \mathcal{N}(x_i;\mu_{j'},\Sigma_{j'})$ |
| GMM M-step mean | $\mu_j = \sum_i \gamma_{ij} x_i / \sum_i \gamma_{ij}$ |
| GMM M-step covariance | $\Sigma_j = \sum_i \gamma_{ij}(x_i-\mu_j)(x_i-\mu_j)^T / \sum_i \gamma_{ij}$ |
| GMM M-step mixing | $\pi_j = (1/n) \sum_i \gamma_{ij}$ |
| Joint log-likelihood | $\ell(\theta;x,Z) = \sum_i\sum_j \mathbf{1}\{Z_i=j\}(\log \mathcal{N}(x_i;\mu_j,\Sigma_j) + \log \pi_j)$ |
