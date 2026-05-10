# Week 9 — Unsupervised Clustering

## Overview
This week covers unsupervised clustering methods: K-means, K-medoids, hierarchical clustering (agglomerative and divisive), Gaussian Mixture Models (GMM) with the EM algorithm, and cluster validation methods (silhouette, gap statistic). Clustering is fundamentally different from supervised classification: there is no response variable y, and clustering will ALWAYS produce a grouping — even on random data.

---

## 1. K-means Clustering

### Key Concepts
- Partitions n observations into K clusters by minimizing within-cluster variance.
- Requires pre-specifying K.
- Each observation belongs to exactly one cluster (hard assignment).
- Algorithm is iterative and not guaranteed to find the global optimum — result depends on initialization.
- Works with Euclidean distance.

### Algorithm
1. Initialize: randomly assign observations to K clusters (or choose K random centroids).
2. **Assignment step:** assign each observation to the nearest centroid (by Euclidean distance).
3. **Update step:** recalculate each centroid as the mean of its assigned observations.
4. Repeat steps 2–3 until assignments do not change (convergence).

### Objective Function
```
min  Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖²
```
- μₖ: centroid (mean) of cluster k.
- Cₖ: set of observations in cluster k.

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
- No need to pre-specify K — cut the dendrogram at any level to get any number of clusters.
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
  - {Cph, Edi}–Reykavik: min(2107, 1374) = 1374 → merge → {Cph, Edi, Reykavik}
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
- Each observation comes from one of K Gaussian distributions.
- An unobserved (latent) random variable Zᵢ selects which Gaussian generates observation xᵢ.
- Unknown parameters: means μⱼ, covariances Σⱼ, mixing proportions πⱼ.

### Model Specification
- `Xᵢ ~ N(μⱼ, Σⱼ)` if `Zᵢ = j`
- `P(Zᵢ = j) = πⱼ`  (mixing proportions)
- `π₁ + π₂ + ... + πₖ = 1`

### Parameter Vector
```
θ = (π₁, ..., πₖ, μ₁, ..., μₖ, Σ₁, ..., Σₖ)
Z = (Z₁, ..., Zₙ)   (latent variables, unknown)
```

### Joint Log-Likelihood
```
ℓ(θ; x, Z) = log p_θ(x, Z) = Σᵢ Σⱼ 𝟙{Zᵢ=j} (log N(xᵢ; μⱼ, Σⱼ) + log πⱼ)
```
Maximum likelihood estimate: `θ_ML = arg max_{θ,Z} ℓ(θ; x, Z)`
Finding the solution is simplified by the **EM algorithm**.

### The EM Algorithm for GMM

**Step 0 — Initialize:** Choose initial values for μⱼ, Σⱼ, πⱼ.

**Step E (Expectation) — Compute posterior probabilities:**
```
γᵢⱼ = P_θ(k)(Zᵢ = j | xᵢ) = [πⱼ^(k) N(xᵢ; μⱼ^(k), Σⱼ^(k))] / [Σⱼ' πⱼ'^(k) N(xᵢ; μⱼ'^(k), Σⱼ'^(k))]
```
- γᵢⱼ = soft assignment of observation i to cluster j (probability between 0 and 1).
- Uses Bayes' formula: prior × likelihood / marginal.

**Step M (Maximization) — Update parameters using γᵢⱼ:**
```
μⱼ^(k+1) = Σᵢ γᵢⱼ xᵢ / Σᵢ γᵢⱼ

Σⱼ^(k+1) = Σᵢ γᵢⱼ (xᵢ − μⱼ)(xᵢ − μⱼ)^T / Σᵢ γᵢⱼ

πⱼ^(k+1) = (1/n) Σᵢ γᵢⱼ
```
- These are **weighted** versions of the standard MLE formulas, weighted by soft assignments.

**Step 4 — Iterate until convergence.**

### Q-function (theoretical basis of M-step)
```
Q(θ | θ^(k)) = E_{Z|x,θ^(k)}[log p_θ(x, Z)]
             = Σᵢ Σⱼ p_θ^(k)(Zᵢ=j|xᵢ) (log N(xᵢ; μⱼ, Σⱼ) + log πⱼ)
```
The M-step maximizes this Q-function with respect to θ.

### GMM Properties
- **Soft assignments:** Each point has a probability of belonging to each cluster (unlike K-means).
- **Flexible covariance:** Each cluster can have its own full covariance matrix Σⱼ.
- K-means is a special case of GMM with equal, spherical covariances and hard assignments.
- **High-dimensional tricks (p >> n):**
  - Share covariance: Σ = Σ₁ = ... = Σₖ (fewer parameters)
  - Diagonal covariance: Σ = diag(σ₁², ..., σₖ²)
  - Regularize: Σ = Σ + λI
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

```
s(i) = (b(i) − a(i)) / max{a(i), b(i)}
```
Where:
- `a(i)` = average distance from point i to all other points in **the same cluster** (cohesion).
- `b(i)` = average distance from point i to all points in the **neighboring cluster** (the nearest cluster i is NOT a member of).
- s(i) ∈ [−1, 1]: closer to 1 = well clustered; closer to −1 = mis-clustered; 0 = on boundary.

**Decision rule:**
- Select K where all clusters have observations above average silhouette width, OR
- Select K* with the **maximum average silhouette** width.

**Properties:**
- Favors convex and spherical clusters.
- Outlier sensitive.
- Reference: Rousseeuw 1987.

#### Gap Statistic
Compares the log within-cluster dissimilarity to what is expected for **uniformly distributed** data (via Monte Carlo simulation with 20 samples).

**Within-cluster dissimilarity:**
```
D_ℓ = Σ_{Cₖ(i)=ℓ, Cₖ(j)=ℓ} ‖xᵢ − xⱼ‖²  =  Nₗ Σ_{Cₖ(i)=ℓ} ‖xᵢ − x̄ₗ‖²

W_k = Σ_ℓ (1/2Nₗ) D_ℓ
```
Where `Cₖ(i) = ℓ` means observation i is in cluster ℓ when there are k clusters, and Nₗ is cluster size.

**Gap Statistic:**
```
G(K) = log(U_k) − log(W_k)
```
Where:
- `U_k` = within-cluster dissimilarity for simulated uniform data (mean over 20 samples)
- `W_k` = within-cluster dissimilarity for actual data

**Selection rule:**
```
K* = arg min_k {K | G(K) ≥ G(K+1) − s'_{K+1}}
```
Where:
```
s'_{K+1} = std(log(U_K)) · √(1 + 1/20)
```
Choose smallest K where gap is large enough relative to next gap (within simulation variability).

**Properties:**
- Works for K-means, K-medoids, and hierarchical clustering.
- Works with different measures of within-cluster dissimilarity.
- Based on simulation — may differ between runs.
- For zip code data example: Gap statistic → K* = 9.

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
| K-means objective | `min Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖²` |
| Within-cluster dissimilarity | `W_k = Σ_ℓ (1/2Nₗ) D_ℓ` |
| D_ℓ (pairwise) | `D_ℓ = Nₗ Σ_{Cₖ(i)=ℓ} ‖xᵢ − x̄ₗ‖²` |
| Gap statistic | `G(K) = log(U_k) − log(W_k)` |
| Gap selection rule | `K* = arg min_k {K | G(K) ≥ G(K+1) − s'_{K+1}}` |
| s'_{K+1} | `std(log(U_K)) · √(1+1/20)` |
| Silhouette | `s(i) = (b(i)−a(i)) / max{a(i),b(i)}` |
| GMM E-step | `γᵢⱼ = πⱼ N(xᵢ;μⱼ,Σⱼ) / Σⱼ' πⱼ' N(xᵢ;μⱼ',Σⱼ')` |
| GMM M-step mean | `μⱼ = Σᵢ γᵢⱼ xᵢ / Σᵢ γᵢⱼ` |
| GMM M-step covariance | `Σⱼ = Σᵢ γᵢⱼ(xᵢ−μⱼ)(xᵢ−μⱼ)^T / Σᵢ γᵢⱼ` |
| GMM M-step mixing | `πⱼ = (1/n) Σᵢ γᵢⱼ` |
| Joint log-likelihood | `ℓ(θ;x,Z) = ΣᵢΣⱼ 𝟙{Zᵢ=j}(log N(xᵢ;μⱼ,Σⱼ) + log πⱼ)` |
