# Week 9 — Group Discussion Questions

Five embedded bee-icon group discussion questions were found in the Week 9 lecture slides (`02582_2026_week9.pdf`).

---

## Q1: What makes a good clustering? (Slide 5)

**Question:** In what sense are points close in one cluster and far from points in another cluster? How do we measure that?

**Answer:**

A good clustering simultaneously satisfies two criteria:

**Within-cluster compactness:** Points assigned to the same cluster should be similar to each other — i.e., the within-cluster dissimilarity should be small. Common measures:

$$W(C) = \sum_{k=1}^{K} \sum_{i \in C_k} d(x_i, \bar{x}_k)$$

where $\bar{x}_k$ is the centroid of cluster $k$ and $d$ is a dissimilarity measure (e.g., Euclidean distance, squared Euclidean, correlation-based).

For K-means specifically, within-cluster sum of squares (WCSS) is used:

$$W(C) = \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \bar{x}_k\|^2$$

**Between-cluster separation:** Points in different clusters should be dissimilar — i.e., the between-cluster dissimilarity should be large:

$$B(C) = \sum_{k=1}^{K} n_k \|\bar{x}_k - \bar{x}\|^2$$

where $\bar{x}$ is the global mean and $n_k$ is the size of cluster $k$.

**How we measure it in practice:**

- **Euclidean distance:** $d(x_i, x_j) = \|x_i - x_j\|_2$ — sensitive to scale, assumes isotropic clusters
- **Manhattan distance:** $d(x_i, x_j) = \|x_i - x_j\|_1$ — more robust to outliers
- **Correlation-based distance:** $d(x_i, x_j) = 1 - \text{corr}(x_i, x_j)$ — measures shape similarity, scale-invariant

The choice of dissimilarity measure is a fundamental modeling decision and affects cluster shape assumptions.

---

## Q2: What happens if we minimize dissimilarity? (Slide 8)

**Question:** What happens if we minimize dissimilarity (instead of within-cluster dissimilarity)?

**Answer:**

If we minimize the total pairwise dissimilarity $\sum_{i,j} d(x_i, x_j)$ without any clustering constraint, the trivial solution is to put **all points into a single cluster** — this is not a meaningful result.

More precisely, the K-means objective minimizes within-cluster dissimilarity $W(C)$. If instead you minimized total dissimilarity (including both within and between), or if you minimized between-cluster similarity $B(C)$, you would degenerate:

- **Minimizing $W(C)$ only with no $K$ constraint** $\Rightarrow$ put every point in its own singleton cluster, giving $W(C) = 0$. Trivial and useless.
- **Minimizing total similarity** $\Rightarrow$ collapse everything into one cluster.

This is why clustering algorithms must constrain the number of clusters $K$ (K-means, GMM), or use a stopping criterion (hierarchical), or validate the result post-hoc using internal measures (silhouette, gap statistic).

The key insight: **clustering requires a balance between compactness and the number of clusters**. The tradeoff is captured by criteria like:
$$\text{minimize} \quad W(C) \quad \text{subject to} \quad K \text{ clusters}$$

---

## Q3: Single linkage vs complete linkage (Slide 30)

**Question:** Given the current state of agglomerative clustering with clusters $\{\text{Copenhagen, Edinburgh}\}$, $\{\text{Nuuq}\}$, $\{\text{Reykavik}\}$ and the distance matrix below, which clusters should we merge if we use **single linkage**? Which clusters should we merge if we use **complete linkage**?

**Distance matrix:**

|           | Copenhagen | Nuuq   | Edinburgh | Reykavik |
|-----------|-----------|--------|-----------|----------|
| Copenhagen | 0 km      | 3535 km | 983 km   | 2107 km  |
| Nuuq       | 3535 km   | 0 km   | 2765 km   | 1430 km  |
| Edinburgh  | 983 km    | 2765 km | 0 km     | 1374 km  |
| Reykavik   | 2107 km   | 1403 km | 1374 km  | 0 km     |

Current clusters after step 1: $\{\text{Copenhagen, Edinburgh}\}$, $\{\text{Nuuq}\}$, $\{\text{Reykavik}\}$

**Answer:**

We need to compute the cluster-cluster distances between the three current clusters.

**Clusters to compare:**
- $G = \{\text{Copenhagen, Edinburgh}\}$
- $H_1 = \{\text{Nuuq}\}$
- $H_2 = \{\text{Reykavik}\}$

**Single linkage** ($d_{SL} = \min_{i \in G, j \in H} d_{ij}$, nearest-neighbor):

$d_{SL}(G, H_1)$: $\min(d(\text{Cph, Nuuq}), d(\text{Edi, Nuuq})) = \min(3535, 2765) = 2765$ km

$d_{SL}(G, H_2)$: $\min(d(\text{Cph, Reykavik}), d(\text{Edi, Reykavik})) = \min(2107, 1374) = 1374$ km

$d_{SL}(H_1, H_2)$: $d(\text{Nuuq, Reykavik}) = 1430$ km

Minimum is $1374$ km $\Rightarrow$ merge $G$ and $H_2$: **$\{\text{Copenhagen, Edinburgh, Reykavik}\}$ and $\{\text{Nuuq}\}$**

**Complete linkage** ($d_{CL} = \max_{i \in G, j \in H} d_{ij}$, furthest-neighbor):

$d_{CL}(G, H_1)$: $\max(3535, 2765) = 3535$ km

$d_{CL}(G, H_2)$: $\max(2107, 1374) = 2107$ km

$d_{CL}(H_1, H_2)$: $d(\text{Nuuq, Reykavik}) = 1430$ km

Minimum is $1430$ km $\Rightarrow$ merge $H_1$ and $H_2$: **$\{\text{Copenhagen, Edinburgh}\}$ and $\{\text{Nuuq, Reykavik}\}$**

**Key takeaway:** Single linkage and complete linkage give different cluster structures because single linkage can be dominated by a single close pair (chaining effect), while complete linkage enforces that all members of two clusters are close.

---

## Q4: Unknown model parameters — supervised case (Slide 33)

**Question:** In the supervised Gaussian model (where cluster labels $Z_i$ are known), what are the unknown model parameters?

**Setup on the slide:** Observations come from three known Gaussian distributions:
$$X_i \in N(\mu_1, \Sigma_1) \text{ if } Z_i = 1, \quad X_i \in N(\mu_2, \Sigma_2) \text{ if } Z_i = 2, \quad X_i \in N(\mu_3, \Sigma_3) \text{ if } Z_i = 3$$
With $Z_i = 1$ for $i = 1, \ldots, 100$, $Z_i = 2$ for $i = 101, \ldots, 200$, $Z_i = 3$ for $i = 201, \ldots, 300$ (known labels).

**Answer:**

Since the cluster assignments $Z_i$ are **known** (observed/labeled data), the only unknown parameters are the parameters of each Gaussian distribution:

$$\boldsymbol{\theta} = (\mu_1, \mu_2, \mu_3, \Sigma_1, \Sigma_2, \Sigma_3)$$

That is: the **mean vector** $\mu_j \in \mathbb{R}^p$ and **covariance matrix** $\Sigma_j \in \mathbb{R}^{p \times p}$ for each of the $K = 3$ clusters.

These can be estimated directly by maximum likelihood — since labels are known, this reduces to computing the empirical mean and covariance within each labeled group:

$$\hat{\mu}_j = \frac{1}{n_j} \sum_{i: Z_i = j} x_i, \qquad \hat{\Sigma}_j = \frac{1}{n_j} \sum_{i: Z_i = j} (x_i - \hat{\mu}_j)(x_i - \hat{\mu}_j)^\top$$

No EM algorithm is needed because there are no latent variables — this is just supervised MLE, which is what LDA and QDA use (with the additional simplification that LDA assumes $\Sigma_1 = \Sigma_2 = \Sigma_3$).

---

## Q5: Unknown model parameters — unsupervised GMM case (Slide 34)

**Question:** In the unsupervised Gaussian Mixture Model (where cluster labels $Z_i$ are **unknown**), what are the unknown model parameters?

**Setup on the slide:** Same three Gaussian distributions as above, but now $Z_i$ are latent (unobserved) variables with prior probabilities:
$$P(Z_i = 1) = \pi_1, \quad P(Z_i = 2) = \pi_2, \quad P(Z_i = 3) = \pi_3, \quad \pi_1 + \pi_2 + \pi_3 = 1$$

**Answer:**

Now that the $Z_i$ are unobserved, they become part of what must be inferred. The **unknown model parameters** are:

$$\boldsymbol{\theta} = (\pi_1, \pi_2, \pi_3, \mu_1, \mu_2, \mu_3, \Sigma_1, \Sigma_2, \Sigma_3)$$

Specifically:
- $\pi_j$ — the **mixing proportions** (prior probability that an observation comes from component $j$), subject to $\sum_j \pi_j = 1$. These have no analogue in the supervised case.
- $\mu_j$ — the **mean** of component $j$
- $\Sigma_j$ — the **covariance matrix** of component $j$

For $K$ components in $p$ dimensions, the total parameter count is:
- $K - 1$ free mixing proportions
- $Kp$ mean parameters
- $K \cdot p(p+1)/2$ covariance parameters (if full covariance)

Since the complete-data log-likelihood is:
$$\ell(\boldsymbol{\theta}; \mathbf{x}, \mathbf{Z}) = \sum_{i=1}^{n} \sum_{j=1}^{K} \mathbf{1}_{\{Z_i = j\}} \left(\log N(x_i; \mu_j, \Sigma_j) + \log \pi_j\right)$$

and $\mathbf{Z}$ is unobserved, direct maximization is intractable. The **EM algorithm** resolves this by alternating between:
- **E-step:** Compute soft assignments $\gamma_{ij} = P(Z_i = j \mid x_i, \boldsymbol{\theta}^{(k)})$ via Bayes' rule
- **M-step:** Update $\boldsymbol{\theta}$ using the soft-weighted sufficient statistics
