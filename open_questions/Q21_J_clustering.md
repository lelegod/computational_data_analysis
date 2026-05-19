# Q21-J — K-means vs Hierarchical Clustering
> Week 9. Classic comparison question; could also ask to derive the K-means algorithm.

---

## K-means Algorithm

**Goal**: partition $N$ observations into $K$ clusters to minimize within-cluster variance:
$$\min_{C_1,\ldots,C_K} \sum_{k=1}^K \sum_{x_i\in C_k} \|x_i - \mu_k\|^2$$

where $\mu_k = \frac{1}{|C_k|}\sum_{x_i\in C_k} x_i$ is the cluster centroid.

**Algorithm** (Lloyd's algorithm):
1. Initialize: randomly assign each observation to one of $K$ clusters (or use K-means++)
2. **Assignment step** (E-step): assign each $x_i$ to nearest centroid:
$$C(i) = \arg\min_k \|x_i - \mu_k\|^2$$
3. **Update step** (M-step): recompute centroids:
$$\mu_k = \frac{1}{|C_k|}\sum_{i:C(i)=k} x_i$$
4. Repeat steps 2–3 until assignments don't change

**Convergence**: guaranteed to converge (objective decreases or stays the same at each step) but to a **local minimum**, not global.

**Connection to GMM**: K-means is a special case of GMM EM with spherical equal covariances $\Sigma_k=\sigma^2 I$ and hard (0/1) assignments instead of soft probabilities.

**Key properties**:
- Requires $K$ specified in advance
- Sensitive to initialization (run multiple times, keep best)
- **K-means++ initialization**: choose first centroid uniformly at random, then each subsequent centroid with probability $\propto d(x_i,\text{nearest existing centroid})^2$ → better coverage, faster convergence
- Result depends on scale: standardize features before running (unless scale is meaningful)
- Cannot handle non-convex, non-spherical clusters

---

## Hierarchical Clustering

**Goal**: build a tree (dendrogram) of nested clusters — no need to specify $K$ in advance.

**Two types**:
- **Agglomerative** (bottom-up): start with $N$ singleton clusters, merge the two closest at each step
- **Divisive** (top-down): start with one cluster containing all points, split recursively. Rarely used.

### Agglomerative Algorithm

1. Start: $N$ clusters, each containing one observation
2. Compute $N\times N$ dissimilarity matrix $D$
3. Find the two clusters $C_i$, $C_j$ with smallest inter-cluster distance (linkage)
4. Merge them into one cluster
5. Update $D$: compute distances from merged cluster to all others
6. Repeat steps 3–5 until one cluster remains
7. Cut the dendrogram at desired height to obtain $K$ clusters

**Output**: dendrogram — read off any number of clusters by choosing the cut height.

### Linkage Methods

How to define distance between two sets of points:

| Linkage | Definition | Behavior |
|---------|-----------|---------|
| Single | $\min_{i\in C_1, j\in C_2} d(x_i,x_j)$ | Chaining effect, elongated clusters |
| Complete | $\max_{i\in C_1, j\in C_2} d(x_i,x_j)$ | Compact, roughly equal-sized clusters |
| Average | $\frac{1}{|C_1||C_2|}\sum_{i\in C_1}\sum_{j\in C_2} d(x_i,x_j)$ | Compromise |
| Ward | Minimize increase in total within-cluster variance | Tends to produce equal-sized compact clusters |
| Centroid | $d(\mu_{C_1}, \mu_{C_2})$ | Can produce inversions in dendrogram |

**Ward** is the default for most practical use. Equivalent to K-means objective in the merge criterion.

---

## Choosing K

For K-means and to decide where to cut a dendrogram:

**Elbow method**: plot within-cluster sum of squares (WCSS) vs $K$. Look for the "elbow" — point where adding more clusters gives diminishing returns.

**Silhouette score**:
$$s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))} \in [-1,1]$$
where $a(i)$ = mean distance to points in same cluster, $b(i)$ = mean distance to points in nearest other cluster. High = well-clustered. Average silhouette width as function of $K$ → pick $K$ at maximum.

**Gap statistic**: compare WCSS to expected WCSS under a null reference distribution (random data). $K$ = smallest $K$ where gap is within 1 SE of the maximum.

---

## Comparison Table

| Property | K-means | Hierarchical (Agglomerative) |
|----------|---------|------------------------------|
| $K$ required upfront? | Yes | No (choose after seeing dendrogram) |
| Algorithm type | Iterative (EM-like) | Sequential merging |
| Result | Flat partition | Dendrogram (nested hierarchy) |
| Sensitivity to init? | Yes (local optima) | No (deterministic given linkage) |
| Scalability | $O(NKd)$ per iteration — fast | $O(N^2\log N)$ — slow for large $N$ |
| Cluster shape | Spherical, equal-sized | Depends on linkage (complete → compact) |
| Handles non-convex? | No | Single linkage can (chaining) |
| Sensitive to outliers? | Yes (centroid pulled) | Complete/Ward: less; Single: yes |
| Reproducible? | No (random init) | Yes |

---

## Additional Possible Exam Questions

**Q: Why does K-means converge but not necessarily to the global minimum?**
Each step (assignment + update) is guaranteed to decrease or maintain the objective (WCSS). Since there are finitely many possible assignments, the algorithm terminates. But the objective is non-convex — there are many local minima. The final solution depends on initialization. Run $k$ times with different random starts and keep the run with lowest WCSS.

**Q: What is K-medoids (PAM) and how does it differ from K-means?**
K-medoids uses actual data points as cluster centers (medoids), not means. Objective: minimize sum of distances to nearest medoid. More robust to outliers (medoid is a real point, cannot be pulled by extreme values). Works with any dissimilarity measure (not just Euclidean). More expensive: $O(N^2K)$ per iteration.

**Q: What is an "inversion" in a dendrogram and which linkage causes it?**
An inversion occurs when a merge at a lower level has a greater inter-cluster distance than a merge at a higher level — the dendrogram branches cross. Centroid linkage can produce inversions because moving from two points to their average can create a centroid closer to an existing cluster than the original points were. Single, complete, average, and Ward linkages are monotone (no inversions).

**Q: How does hierarchical clustering with Ward linkage relate to K-means?**
Ward linkage minimizes the increase in total within-cluster sum of squares at each merge step. This is exactly the K-means objective. Cutting the Ward dendrogram at $K$ clusters gives a partition that is a reasonable starting point for K-means. Ward + K-means refinement is a common practical approach.

**Q: When would you choose GMM over K-means?**
GMM: when clusters have different shapes, sizes, or orientations (different $\Sigma_k$), or when you need soft assignments (probabilistic cluster membership) and density estimates. K-means: when clusters are roughly spherical and equal-sized, you need speed, or interpretability of hard assignments is important.
