# Q21-Y — K-medoids (PAM) vs K-means
> Week 9. Robustness comparison; could be asked alongside or instead of K-means.

---

## The Core Difference

**K-means**: cluster centers = **means** of assigned points (arithmetic average). Can be any point in $\mathbb{R}^p$, not necessarily a training observation.

**K-medoids (PAM — Partitioning Around Medoids)**: cluster centers = **medoids** — actual training observations that minimize total distance to other points in the cluster. Centers are always real data points.

---

## K-medoids Objective

$$\min_{m_1,\ldots,m_K} \sum_{k=1}^K \sum_{x_i \in C_k} d(x_i, m_k)$$

where $m_k \in \{x_1,\ldots,x_N\}$ (medoid must be a training observation) and $d$ is any dissimilarity measure.

Compare to K-means: $\min \sum_k\sum_{x_i\in C_k}\|x_i-\mu_k\|^2$ where $\mu_k$ can be any vector.

---

## PAM Algorithm

**Initialize**: choose $K$ initial medoids (randomly or using BUILD heuristic).

**Iterate until convergence**:
1. **Assignment**: assign each non-medoid $x_i$ to the nearest medoid
2. **Update**: for each cluster $k$ and each non-medoid $x_j \in C_k$:
   - Compute total cost if $x_j$ replaces $m_k$ as medoid
   - If total cost decreases, swap $m_k \leftarrow x_j$
3. Repeat assignment + update until no swap improves the objective

**Complexity**: $O(K(N-K)^2)$ per iteration — much more expensive than K-means ($O(NK)$ per iteration).

---

## Why K-medoids is More Robust to Outliers

**K-means**: mean is sensitive to outliers. One extreme observation pulls the centroid far from the true cluster center.

**K-medoids**: the medoid is the observation minimizing total distance to others in the cluster — it is the most "centrally located" actual point. A single outlier cannot become the medoid unless it is genuinely central, and it cannot shift the medoid to an extreme position.

**Example**: cluster of 99 points near origin + 1 outlier at $(1000, 0)$.
- K-means centroid shifts to $\approx(10, 0)$ — far from the true cluster center
- K-medoids: the outlier is never chosen as medoid (total distance would be huge); centroid stays near origin

---

## Works with Any Dissimilarity

K-means requires Euclidean distances (to compute means). K-medoids works with **any dissimilarity matrix** $D_{ij} = d(x_i,x_j)$:
- Categorical data (Hamming distance)
- Strings (edit distance)
- Sequences (DTW distance)
- Graphs (graph kernels)
- Any precomputed $N\times N$ dissimilarity matrix

This makes K-medoids applicable to non-vectorial data where means are not defined.

---

## Comparison Table

| Property | K-means | K-medoids (PAM) |
|----------|---------|----------------|
| Centers | Arithmetic means (any point) | Actual data points |
| Objective | Minimize squared Euclidean distance | Minimize total dissimilarity |
| Distance | Euclidean only | Any dissimilarity |
| Robust to outliers? | No | Yes |
| Computational cost | $O(NKd)$ per iter — fast | $O(K(N-K)^2)$ per iter — slow |
| Scalable to large $N$? | Yes | No (use CLARA for large $N$) |
| Result interpretable? | Centroid may not exist in data | Medoid is always a real observation |
| Reproducible? | No (random init) | No (random init) |

---

## CLARA: K-medoids for Large Datasets

PAM is $O(N^2)$ — infeasible for large $N$. **CLARA** (Clustering LARge Applications):
1. Draw multiple random subsamples of size $s \ll N$
2. Apply PAM to each subsample → get candidate medoids
3. Assign all $N$ points to nearest candidate medoid
4. Report the partition with minimum total cost

Trades exactness for scalability.

---

## Additional Possible Exam Questions

**Q: Why is K-medoids preferred for categorical data?**
For categorical variables, the mean is not defined — you cannot average "red, blue, green" to get a center. K-medoids only needs a pairwise dissimilarity matrix (e.g., Hamming distance for binary vectors, or any custom metric). The medoid is simply the observation with minimum total distance to others in the cluster — no notion of mean required.

**Q: Show that K-means is a special case of K-medoids under Euclidean distance when medoids are unrestricted.**
If we relax the constraint that medoids must be training observations and allow them to be any point in $\mathbb{R}^p$, the optimal "medoid" minimizing total squared Euclidean distance to cluster members is the mean: $\arg\min_m \sum_{x_i\in C_k}\|x_i-m\|^2 = \frac{1}{|C_k|}\sum_{x_i\in C_k}x_i$. So K-means = K-medoids with unrestricted centers and squared Euclidean distance.

**Q: What is the medoid of a cluster?**
The medoid is the observation $m_k \in C_k$ that minimizes the total dissimilarity to all other points in the cluster: $m_k = \arg\min_{x_j\in C_k}\sum_{x_i\in C_k}d(x_i,x_j)$. It is the "most central" actual data point — geometrically, it is the point closest to the centroid of the cluster on average.

**Q: When would you choose K-means over K-medoids despite outlier sensitivity?**
When $N$ is large (K-medoids is $O(N^2)$, impractical). When the data is truly Euclidean and outliers have been pre-processed/removed. When speed is more important than robustness. When you plan to use the centroid only for prediction (nearest centroid classification), not for interpretation. K-means with preprocessing (outlier removal, robust scaling) often matches K-medoids in practice while being far faster.
