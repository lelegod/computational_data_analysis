# Week 9 — Lecture Notes
## Computational Data Analysis (02582)

---

## Cluster Analysis

**Goal:** Partition observations into groups (clusters) such that points within a cluster are close to each other, and points in different clusters are far apart. Unsupervised — no labels.

---

## Similarity and Dissimilarity

- **Similarity**: large value when points are close
- **Dissimilarity**: large value when points are far apart (= distance)

Any monotone-decreasing function converts one to the other.

### Distance Measures

| Distance | Formula | Use case |
|---|---|---|
| Euclidean | $d(x_i,x_j) = \sqrt{\sum_{k=1}^p (x_{ik}-x_{jk})^2}$ | Quantitative variables |
| Manhattan | $d(x_i,x_j) = \sum_{k=1}^p \|x_{ik}-x_{jk}\|$ | Quantitative, robust to outliers |
| Mahalanobis | $d(x_i,x_j) = \sqrt{(x_i-x_j)^T \Sigma^{-1}(x_i-x_j)}$ | Accounts for covariance structure |
| Tanimoto | $d(x_i,x_j) = \dfrac{x_i^T x_j}{x_i^T x_i + x_j^T x_j - x_i^T x_j}$ | Binary / categorical variables |

**Weighted distances:** give different weight $w_k$ to each feature:

$$d(x_i,x_j) = \sum_{k=1}^p w_k\, d_k(x_{ik}, x_{jk}), \qquad \sum_{k=1}^p w_k = 1$$

Note: $w_k = 1/p$ does not give equal influence — must normalise by average distance per feature.

---

## Clustering as an Optimisation Problem

### Within-Cluster Variability

How tight (compact) a cluster $c$ is:

$$\text{Variability}(c) = \sum_{i \in c} \text{dist}(\text{mean}(c),\, x_i)^2$$

### Total Dissimilarity

Sum of variability over all clusters:

$$\text{Dissimilarity}(C) = \sum_{c \in C} \text{Variability}(c)$$

**Clustering objective:** minimise total dissimilarity.

**Trivial solution trap:** putting every point in its own cluster gives variability = 0. This is why you fix $K$.

### Two Goals of Any Clustering

| Goal | Want |
|---|---|
| Points within cluster are close (within-cluster variability) | Small |
| Points across clusters are far apart (between-cluster dissimilarity) | Large |

---

## K-means Clustering

**Algorithm:**
1. Choose $K$, randomly initialise $K$ centroids
2. Repeat until assignments do not change:
   - (a) Assign each point to its nearest centroid
   - (b) Recompute each centroid as the mean of its assigned points
3. Output: $K$ cluster assignments

Each step can only **decrease** total dissimilarity → guaranteed to converge (but may reach a local minimum — run multiple times with different initialisations).

**K-means minimises:**

$$\min_{C} \sum_{c \in C} \sum_{i \in c} \|x_i - \text{mean}(c)\|^2$$

This is within-cluster sum of squares (WCSS).
