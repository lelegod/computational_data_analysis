# Q21-V — Cluster Validation: Silhouette, Gap Statistic, Elbow
> Week 9. How do you choose K? Could be asked alongside K-means or as standalone.

---

## The Problem

Clustering algorithms (K-means, GMM, hierarchical) require choosing $K$. But unlike supervised learning, there is no held-out label to validate against. We need internal or relative criteria.

---

## Elbow Method

**Criterion**: Within-Cluster Sum of Squares (WCSS):
$$\text{WCSS}(K) = \sum_{k=1}^K\sum_{x_i\in C_k}\|x_i-\mu_k\|^2$$

**Procedure**: compute WCSS for $K=1,2,\ldots,K_\text{max}$. Plot WCSS vs $K$.

**Shape**: WCSS always decreases as $K$ increases (more clusters = smaller groups = closer to centers). The "elbow" = point of diminishing returns where the curve bends sharply.

**Limitation**: the elbow is often ambiguous — the curve may be smooth with no clear kink, especially for high-dimensional or non-spherical data. The method is subjective.

---

## Silhouette Score

For each observation $i$ in cluster $C_k$:
- $a(i)$ = mean distance to all other points in the same cluster $C_k$ (intra-cluster cohesion)
- $b(i)$ = mean distance to all points in the nearest other cluster $C_{k'}\neq C_k$ (inter-cluster separation)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$

**Interpretation**:
- $s(i) \approx 1$: well-clustered (far from neighboring cluster, close to own)
- $s(i) \approx 0$: on the boundary between two clusters
- $s(i) < 0$: misclassified (closer to another cluster than its own)

**Average silhouette width**: $\bar{s}(K) = \frac{1}{N}\sum_i s(i)$. Plot vs $K$; choose $K$ at maximum.

**Advantages over elbow**: quantitative, interpretable, uses both cohesion AND separation (not just cohesion like WCSS). Can diagnose individual point quality.

**Limitations**: assumes convex, roughly equal-sized clusters (same assumption as K-means). May fail for elongated or non-convex clusters.

---

## Gap Statistic (Tibshirani et al., 2001)

**Core idea**: compare WCSS to what you would expect under a null reference distribution (random data with no cluster structure).

**Algorithm**:
1. Compute $\log\text{WCSS}(K)$ on the real data
2. Generate $B$ reference datasets by sampling uniformly from the bounding box of the data
3. Compute $\log\text{WCSS}^*(K)$ on each reference dataset
4. Gap statistic: $\text{Gap}(K) = \frac{1}{B}\sum_b\log\text{WCSS}^*_b(K) - \log\text{WCSS}(K)$
5. Standard deviation: $s_K = \text{SD}[\log\text{WCSS}^*(K)]\cdot\sqrt{1+1/B}$

**Selection rule**: choose the smallest $K$ such that:
$$\text{Gap}(K) \geq \text{Gap}(K+1) - s_{K+1}$$

i.e., the gap at $K$ is at least as large as the gap at $K+1$ minus one standard error — analogous to the 1-SE rule.

**Advantage**: principled statistical test against a null hypothesis of no cluster structure. $K=1$ (no clusters) is selected when the data is random. Works for any clustering algorithm.

**Limitation**: computationally expensive ($B$ reference datasets $\times$ K_max fits). Sensitive to the choice of reference distribution.

---

## BIC for GMM

When using Gaussian Mixture Models, BIC gives a principled model selection criterion:
$$\text{BIC}(K) = -2\ell_K(\hat{\theta}) + p_K\log N$$

where $p_K$ = number of free parameters for a $K$-component GMM:
- Full covariance: $p_K = K-1 + Kd + Kd(d+1)/2$ (mixing weights + means + covariance entries)
- Spherical: $p_K = K-1 + Kd + K$ (much fewer)

Plot BIC vs $K$; choose $K$ at minimum. BIC balances fit (log-likelihood) against complexity (parameter count).

---

## Comparison of Methods

| Method | Measures | Works for non-spherical? | Requires distance? | Principled? |
|--------|---------|------------------------|-------------------|------------|
| Elbow (WCSS) | Cohesion only | No | Yes (Euclidean) | No (subjective) |
| Silhouette | Cohesion + separation | Partially | Yes (any) | Partially |
| Gap statistic | vs random null | Yes | Yes (any) | Yes |
| BIC (GMM) | Probabilistic fit | Yes (full $\Sigma_k$) | No (model-based) | Yes |

**Practical recommendation**: use Silhouette for quick visual inspection; Gap statistic or BIC for rigorous selection.

---

## Additional Possible Exam Questions

**Q: Why can silhouette give misleading results for non-convex clusters?**
Silhouette computes mean distance to all points in a cluster. For a crescent-shaped cluster, distant points within the same cluster may have large $a(i)$, even if they are correctly clustered. Similarly, $b(i)$ may be small if a neighboring cluster is convex and compact. The result: $s(i)$ can be low for correctly clustered points in non-convex shapes. Methods like DBSCAN or spectral clustering better capture non-convex structure; WCSS and silhouette are designed for convex clusters.

**Q: What does a Gap statistic of zero mean?**
The WCSS on real data equals the expected WCSS on random data. The data has no more cluster structure than a random dataset — $K=1$ (no clusters) is the correct choice.

**Q: How does BIC for GMM relate to AIC?**
Both penalize the log-likelihood: AIC uses $2p_K$, BIC uses $p_K\log N$. For $N>7$: BIC penalizes extra components more → selects fewer clusters. In practice, AIC-selected GMMs tend to overfit (too many components); BIC is preferred for GMM model selection.

**Q: Can you use cross-validation to choose K in clustering?**
Not straightforwardly — clustering is unsupervised, so there is no held-out label to evaluate. One approach: prediction strength (Tibshirani & Walther): fit clusters on training set, predict cluster membership for test set, measure stability. Another: hold out a subset, fit clusters on training, assign test points to nearest centroid, measure WCSS on test points. Both are valid but less common than silhouette/gap/BIC.

**Q: What is the difference between internal and external cluster validation?**
Internal: evaluate cluster quality using only the data itself (no ground truth). Examples: silhouette, WCSS, BIC. Used when true labels are unknown (unsupervised setting). External: compare clustering to known ground-truth labels using metrics like Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), or purity. Used when ground truth exists (evaluation/benchmarking). In a real unsupervised problem, only internal validation is available.
