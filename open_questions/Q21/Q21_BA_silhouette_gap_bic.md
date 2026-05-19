# Q21-BA — Silhouette vs Gap Statistic vs BIC
> Week 9. Could ask: compare three methods for choosing the number of clusters and explain when each is appropriate.

---

## The Shared Problem

Clustering always produces some grouping, even on random data.

So one of the hardest practical questions is:

**How many clusters should we keep?**

Three important tools from the course are:

- silhouette
- gap statistic
- BIC

---

## Silhouette

For observation $i$:
$$
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
$$

where:
- $a(i)$ = average distance to points in its own cluster
- $b(i)$ = average distance to points in the nearest other cluster

### Main idea

- reward high within-cluster cohesion
- reward good separation from neighboring clusters

### Interpretation

- near 1: well-clustered
- near 0: on the boundary
- negative: likely misclustered

Silhouette is geometric and heuristic.

---

## Gap Statistic

The gap statistic compares observed clustering quality to what would be expected under random structureless data.

### Main idea

Compare:
- observed within-cluster dispersion
- dispersion under a null reference distribution

Choose the smallest $K$ such that:
$$
\text{Gap}(K)\ge \text{Gap}(K+1)-s_{K+1}
$$

### Interpretation

Gap statistic asks:

“Is this clustering better than what random uniform data would produce?”

So it is more statistically principled than silhouette.

---

## BIC

For model-based clustering, especially GMM:
$$
\text{BIC}(K)=-2\ell(\hat\theta_K)+p_K\log N
$$

### Main idea

- reward likelihood fit
- penalize model complexity

### Interpretation

BIC is appropriate when clustering is explicitly modeled probabilistically, such as with Gaussian mixtures.

It is not a generic geometric criterion like silhouette.

---

## Core Comparison

| Method | Main idea | Best for | Main limitation |
|--------|-----------|----------|-----------------|
| Silhouette | Cohesion vs separation | Geometric clustering like K-means | Favors convex/spherical clusters |
| Gap statistic | Compare to random null data | Generic clustering with stronger justification | Computationally heavier |
| BIC | Likelihood + complexity penalty | GMM / model-based clustering | Requires probabilistic model |

---

## Which Is Most Principled?

This is a common exam-style question.

- **Silhouette** is intuitive and easy
- **Gap statistic** is more principled for general clustering
- **BIC** is principled when a probabilistic model such as GMM is assumed

So there is no single universal winner; the correct method depends on the clustering framework.

---

## K-means vs GMM Context

### For K-means

Use:
- silhouette
- gap statistic
- elbow as a weaker heuristic

### For GMM

Use:
- BIC
- or AIC, though BIC is usually preferred for choosing $K$

This context dependence is often what the examiner wants you to say.

---

## When to Use Which

**Use silhouette when**:
- the clustering method is geometric
- you want a simple internal validation score

**Use gap statistic when**:
- you want a more principled nonparametric comparison to a null structure

**Use BIC when**:
- clustering is model-based
- likelihood is defined, especially for GMM

---

## Limitations

1. Silhouette can fail for non-convex cluster shapes.
2. Gap statistic depends on simulated null data and is more computationally expensive.
3. BIC inherits the assumptions of the probabilistic model.

---

## Additional Possible Exam Questions

**Q: Which method is most natural for GMM?**
BIC.

**Q: Why is silhouette not ideal for GMM?**
Because silhouette is purely geometric and does not use the likelihood model or soft assignments.

**Q: Which is the most general-purpose geometric method here?**
Gap statistic, because it compares observed structure to a null reference distribution.
