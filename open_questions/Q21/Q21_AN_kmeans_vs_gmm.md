# Q21-AN — K-means vs GMM
> Week 9. Could ask: compare hard and soft clustering, explain why K-means is a special case of GMM, and discuss when likelihood-based clustering is preferable.

---

## The Shared Goal

Both methods try to partition data into $K$ groups, but they do so with very different modeling philosophies.

- **K-means** is geometric and deterministic once initialized
- **GMM** is probabilistic and models uncertainty in cluster membership

---

## K-means

K-means minimizes within-cluster sum of squares:
$$
\min_{C_1,\dots,C_K} \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i-\mu_k\|^2
$$

It alternates:
1. assign each point to the nearest centroid
2. recompute centroids as cluster means

### Main characteristics

- hard assignments
- spherical equal-size bias
- fast and simple
- no explicit probability model

---

## GMM

GMM models the density as:
$$
p(x)=\sum_{k=1}^K \pi_k \mathcal{N}(x;\mu_k,\Sigma_k)
$$

It is fitted by EM:

1. **E-step**: compute responsibilities
   $$
   \gamma_{ik}=P(Z_i=k \mid x_i)
   $$
2. **M-step**: update $\pi_k,\mu_k,\Sigma_k$

### Main characteristics

- soft assignments
- ellipsoidal cluster shapes possible
- likelihood-based
- gives uncertainty and density estimates

---

## Hard vs Soft Clustering

This is one of the most important distinctions.

### K-means

Each point belongs to exactly one cluster:
$$
z_{ik} \in \{0,1\}, \quad \sum_k z_{ik}=1
$$

### GMM

Each point has a probability of belonging to each component:
$$
\gamma_{ik} \in [0,1], \quad \sum_k \gamma_{ik}=1
$$

So GMM can express ambiguity near cluster overlaps, while K-means cannot.

---

## Why K-means Is a Special Case of GMM

If in a GMM we assume:

- all covariances are spherical
  $$
  \Sigma_k = \sigma^2 I
  $$
- all components have equal shape
- assignments are hardened to 0/1

then maximizing the GMM likelihood becomes equivalent to minimizing K-means WCSS.

So K-means can be viewed as a constrained hard-assignment limit of GMM.

---

## Cluster Shape Assumptions

| Property | K-means | GMM |
|----------|---------|-----|
| Shape | Spherical | Ellipsoidal possible |
| Size flexibility | Limited | Flexible |
| Orientation | No | Yes |
| Uncertainty | No | Yes |

This is why GMM is better when clusters have different spreads or tilted covariance structures.

---

## Objective Function Difference

**K-means** minimizes:
$$
\text{WCSS}
$$

**GMM** maximizes:
$$
\ell(\theta)=\sum_{i=1}^N \log\left(\sum_{k=1}^K \pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)\right)
$$

So K-means is a pure geometric optimization problem, while GMM is a statistical modeling problem.

---

## Comparison Table

| Property | K-means | GMM |
|----------|---------|-----|
| Assignment | Hard | Soft |
| Objective | Min WCSS | Max likelihood |
| Fitting | Lloyd's algorithm | EM |
| Cluster shape | Spherical | General Gaussian |
| Density model | No | Yes |
| Uncertainty | No | Yes |
| Speed | Faster | Slower |
| Model selection | Elbow, silhouette | BIC / AIC / likelihood |

---

## When to Use Which

**Use K-means when**:
- speed matters
- clusters are roughly spherical
- hard partitioning is enough
- you want a simple baseline

**Use GMM when**:
- cluster overlap matters
- uncertainty matters
- shapes differ in size or orientation
- you want density estimation or BIC-based model selection

---

## Limitations

1. K-means is sensitive to scaling and initialization.
2. GMM can converge to local maxima.
3. GMM can suffer degeneracy if a covariance collapses.
4. Both require choosing $K$.
5. Neither handles highly non-convex structure particularly well without extensions.

---

## Additional Possible Exam Questions

**Q: Why does GMM provide more information than K-means?**
Because it outputs posterior cluster probabilities, not just a single label. This quantifies uncertainty and also defines a full density model for the data.

**Q: Why is K-means often still used even though GMM is more flexible?**
Because K-means is simpler, faster, and often works well enough when clusters are compact and roughly spherical.

**Q: When does the extra flexibility of GMM become important?**
When clusters differ in size, orientation, or overlap, so that a single spherical centroid-based partition is too restrictive.
