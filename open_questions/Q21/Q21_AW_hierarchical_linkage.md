# Q21-AW — Hierarchical Linkage Methods
> Week 9. Could ask: compare single, complete, average, and Ward linkage, and explain how the choice of linkage changes the cluster geometry.

---

## The Shared Framework

Agglomerative hierarchical clustering starts with $n$ singleton clusters and repeatedly merges the two closest clusters.

The key design choice is:

**How is the distance between two clusters defined?**

That is exactly what linkage specifies.

---

## Single Linkage

Distance between two clusters:
$$
d(C_1,C_2)=\min_{i\in C_1,\; j\in C_2} d(x_i,x_j)
$$

### Behavior

- merges clusters if any pair of points is close
- tends to create long chains
- can recover elongated/non-convex shapes

### Main weakness

- extremely sensitive to noise bridges
- one accidental connection can merge two clusters too early

---

## Complete Linkage

Distance between two clusters:
$$
d(C_1,C_2)=\max_{i\in C_1,\; j\in C_2} d(x_i,x_j)
$$

### Behavior

- requires all points across the two clusters to be relatively close
- favors compact clusters
- avoids chaining

### Main weakness

- can split large diffuse clusters
- may be too strict when clusters are not compact

---

## Average Linkage

Distance between two clusters:
$$
d(C_1,C_2)=\frac{1}{|C_1||C_2|}\sum_{i\in C_1}\sum_{j\in C_2} d(x_i,x_j)
$$

### Behavior

- compromise between single and complete linkage
- less chaining than single
- less extreme compactness bias than complete

This is often a moderate, balanced choice.

---

## Ward Linkage

Ward linkage merges the pair of clusters that produces the smallest increase in total within-cluster variance.

### Main idea

It is the hierarchical analogue closest to K-means.

### Behavior

- prefers compact, roughly spherical clusters
- often gives balanced cluster sizes
- usually performs well in practice

### Important restriction

Ward linkage is tied to **Euclidean distance**.

---

## Core Comparison

| Linkage | Definition | Tends to produce | Main risk |
|---------|------------|------------------|-----------|
| Single | Minimum pair distance | Chains / elongated clusters | Noise bridges |
| Complete | Maximum pair distance | Compact clusters | Over-splitting diffuse clusters |
| Average | Mean pair distance | Moderate compromise | Less distinctive geometry |
| Ward | Min increase in WCSS | Compact balanced clusters | Assumes Euclidean, spherical tendency |

---

## Chaining vs Compactness

This is the central conceptual tradeoff.

- **Single linkage** is the most chaining-prone
- **Complete linkage** is the most compactness-seeking
- **Average linkage** sits in between
- **Ward linkage** actively optimizes a variance criterion

If the exam asks “why do dendrograms differ under different linkage rules?”, this is the answer.

---

## Relation to K-means

Ward linkage is most closely related to K-means because both are driven by within-cluster variance.

Difference:

- Ward is hierarchical and produces a dendrogram
- K-means gives one flat partition for a chosen $K$

So cutting a Ward dendrogram at $K$ clusters often gives a partition similar to K-means.

---

## When to Use Which

**Use single linkage when**:
- non-convex or elongated structure is expected

**Use complete linkage when**:
- compact well-separated clusters are expected

**Use average linkage when**:
- you want a neutral compromise

**Use Ward linkage when**:
- Euclidean distance is appropriate
- compact balanced clusters are expected
- you want a strong general-purpose default

---

## Limitations

1. Hierarchical clustering is greedy and cannot undo early bad merges.
2. Single linkage is very noise-sensitive.
3. Complete linkage can fragment broad clusters.
4. Ward linkage is not appropriate with arbitrary non-Euclidean dissimilarities.

---

## Additional Possible Exam Questions

**Q: Which linkage is most vulnerable to chaining?**
Single linkage.

**Q: Which linkage is most closely related to K-means?**
Ward linkage, because it minimizes increase in within-cluster variance.

**Q: Why can hierarchical methods disagree so much for the same dataset?**
Because each linkage defines inter-cluster distance differently, so the sequence of greedy merges changes.
