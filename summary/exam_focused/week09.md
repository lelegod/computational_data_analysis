# Week 9 — Unsupervised Clustering (Exam Focus)

## Must-Know Facts

### General Clustering
- Clustering is **unsupervised** — no response variable y.
- **Clustering will ALWAYS generate a grouping, even on completely random data.** This is the fundamental warning.
- Domain knowledge is needed to interpret whether discovered clusters are meaningful.
- Cross-validation does NOT directly apply to unsupervised clustering.

### K-means
- Partitions n observations into K clusters minimizing total within-cluster squared Euclidean distance.
- K must be specified in advance.
- Assignments are **hard** (each point belongs to exactly one cluster).
- Algorithm: assign to nearest centroid → update centroids as cluster means → repeat.
- **Not guaranteed to find the global optimum** — use multiple random restarts.
- K-means uses **Euclidean distance** only.
- Favors **convex, spherical clusters of similar size**.
- Sensitive to **outliers** (centroids can be pulled by extreme values).

### K-medoids
- Like K-means but cluster centers are **actual data points** (medoids, not means).
- More **robust to outliers** than K-means.
- Works with **any distance measure** (not just Euclidean).

### Hierarchical Clustering
- Produces a **dendrogram** — no need to pre-specify K.
- **Agglomerative (bottom-up):** starts with n singleton clusters, merges iteratively.
- **Divisive (top-down):** starts with one cluster, splits iteratively.
- **Single linkage:** cluster distance = distance of **closest** pair → tends to chain.
- **Complete linkage:** cluster distance = distance of **farthest** pair → compact clusters.
- **Ward linkage:** minimizes increase in total within-cluster variance. Only valid with **Euclidean distance**.
- Ward linkage gives a good compromise between balanced and unbalanced clusters.
- Cut the dendrogram at a chosen height to get a specific number of clusters.

### Gaussian Mixture Models (GMM)
- Probabilistic model with **soft** (probability) cluster assignments.
- `Xᵢ ~ N(μⱼ, Σⱼ)` if latent variable `Zᵢ = j`.
- Mixing proportions πⱼ = P(Zᵢ = j); must sum to 1: `Σⱼ πⱼ = 1`.
- Parameters: θ = (π₁,...,πₖ, μ₁,...,μₖ, Σ₁,...,Σₖ).
- Solved by the **EM algorithm** (alternates E-step and M-step until convergence).
- **E-step:** computes posterior probability γᵢⱼ = P(Zᵢ=j | xᵢ) using Bayes' rule.
- **M-step:** updates parameters as weighted means/covariances using γᵢⱼ as weights.
- πⱼ^(new) = (1/n) Σᵢ γᵢⱼ (average of soft assignments).
- K-means is a special case of GMM with hard assignments and equal spherical covariances.
- High-dimensional GMM tricks: shared covariance, diagonal covariance, regularize Σ = Σ + λI, use PCA first.
- Validate GMM using **AIC or BIC** (not silhouette or gap).

### Silhouette Method
- `s(i) = (b(i) − a(i)) / max{a(i), b(i)}`
- `a(i)` = average distance to points in **same cluster** (cohesion, lower is better).
- `b(i)` = average distance to points in **neighboring cluster** (the nearest cluster i is not in).
- s(i) ∈ [−1, 1]: 1 = perfect, 0 = boundary, negative = misclassified.
- Select K where ALL clusters have observations above average silhouette, OR choose K* with maximum average silhouette.
- Favors **convex, spherical clusters**; **outlier sensitive**.

### Gap Statistic
- `G(K) = log(U_k) − log(W_k)` where U_k = simulated uniform data dissimilarity, W_k = actual data dissimilarity.
- Compare actual within-cluster dissimilarity to what is expected for **uniformly distributed** (structureless) data.
- Uses 20 simulations to estimate U_k.
- Choose `K* = arg min_k {K | G(K) ≥ G(K+1) − s'_{K+1}}` where `s'_{K+1} = std(log(U_K))·√(1+1/20)`.
- Works for K-means, K-medoids, and hierarchical clustering.
- Result depends on simulation — may vary between runs.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|------------|-------------|
| `min Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖²` | K-means objective | Defining K-means |
| `W_k = Σ_ℓ (1/2Nₗ) D_ℓ` | Within-cluster dissimilarity | Gap statistic |
| `D_ℓ = Nₗ Σᵢ∈Cₗ ‖xᵢ − x̄ₗ‖²` | Cluster sum of squares | Computing W_k |
| `G(K) = log(U_k) − log(W_k)` | Gap statistic | Choosing K |
| `K* = arg min_k {K | G(K) ≥ G(K+1) − s'_{K+1}}` | Gap selection rule | Finding optimal K |
| `s'_{K+1} = std(log(U_K))·√(1+1/20)` | Gap uncertainty | Gap selection rule |
| `s(i) = (b(i)−a(i))/max{a(i),b(i)}` | Silhouette score | Cluster quality |
| `γᵢⱼ = πⱼN(xᵢ;μⱼ,Σⱼ) / Σⱼ'πⱼ'N(xᵢ;μⱼ',Σⱼ')` | GMM E-step (soft assignments) | EM algorithm |
| `μⱼ = Σᵢγᵢⱼxᵢ / Σᵢγᵢⱼ` | GMM M-step mean update | EM algorithm |
| `Σⱼ = Σᵢγᵢⱼ(xᵢ−μⱼ)(xᵢ−μⱼ)^T / Σᵢγᵢⱼ` | GMM M-step covariance update | EM algorithm |
| `πⱼ = (1/n) Σᵢ γᵢⱼ` | GMM M-step mixing proportion | EM algorithm |
| `Σπⱼ = 1` | Mixing proportions sum to 1 | GMM constraint |

---

## Common Traps (Wrong Answers in Exams)

- **❌ K-means guarantees the global optimum** → ✓ K-means finds a local optimum; multiple restarts needed.
- **❌ Clustering only produces results when there really are clusters** → ✓ Clustering ALWAYS produces a grouping, even on random data.
- **❌ K-medoids uses cluster means as centers** → ✓ K-medoids uses actual data points (medoids); K-means uses means.
- **❌ Ward linkage can be used with any distance** → ✓ Ward linkage requires **Euclidean distance** specifically.
- **❌ Single linkage produces compact clusters** → ✓ Single linkage tends to chain; complete linkage produces more compact clusters.
- **❌ GMM gives hard cluster assignments** → ✓ GMM gives soft (probabilistic) assignments γᵢⱼ ∈ [0,1].
- **❌ In GMM, we observe Zᵢ (cluster labels)** → ✓ Zᵢ are **latent (unobserved)** variables; that's why we need EM.
- **❌ πⱼ in GMM is the estimated mean of cluster j** → ✓ πⱼ is the **mixing proportion** (prior probability that an obs belongs to cluster j).
- **❌ The E-step in EM updates the parameters** → ✓ The E-step computes soft assignments γᵢⱼ; the **M-step updates parameters**.
- **❌ Cross-validation is the standard way to choose K in clustering** → ✓ CV doesn't directly apply; use silhouette, gap statistic, or AIC/BIC.
- **❌ Silhouette works well for non-spherical clusters** → ✓ Silhouette favors convex, spherical clusters and is unreliable for other shapes.
- **❌ High s(i) means point i is far from its cluster** → ✓ High s(i) (close to 1) means well-clustered; the formula is b-a (far from neighbors, close to own cluster).
- **❌ Gap statistic uses the actual data twice** → ✓ Gap statistic compares actual data to **simulated uniform random data** (20 simulations).
- **❌ K-means can use Manhattan distance** → ✓ K-means specifically uses Euclidean; K-medoids can use any distance.

---

## Quick Decision Rules

- If you need to pre-specify K → K-means or K-medoids.
- If you want a dendrogram (explore all K) → hierarchical clustering.
- If you have outliers → prefer **K-medoids** over K-means.
- If clusters may be non-spherical or of very different sizes → avoid K-means/silhouette.
- If you want probabilistic cluster membership → use **GMM with EM**.
- If you need to choose K:
  - Have a distribution assumption (GMM) → use **AIC/BIC**.
  - No distribution assumption → use **gap statistic** (most principled) or **silhouette** (heuristic).
- If gap statistic G(K) first satisfies G(K) ≥ G(K+1) − s'_{K+1} → that is K*.
- If s(i) is negative for many points in a cluster → that cluster may be wrongly specified.
- If you need to cluster both rows AND columns → use **biclustering** (SpectralBiclustering).
- Ward linkage microarray data → Euclidean distances, top-down divisive approach, dendrogram cut at desired number of nodes.
- Silhouette selects K=2 or K=9 for zip data; Gap statistic selects K=9 → **trust the gap statistic** when the two methods disagree (gap is more statistically principled).
