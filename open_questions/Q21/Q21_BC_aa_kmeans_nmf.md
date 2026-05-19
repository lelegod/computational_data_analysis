# Q21-BC — Archetypal Analysis vs K-means vs NMF
> Weeks 9/11. Could ask: compare three unsupervised methods by where their prototypes live and what kind of structure they uncover.

---

## The Shared Question

All three methods summarize a dataset using a smaller set of representative structures.

But the representatives are fundamentally different:

- **K-means** finds centroids
- **Archetypal Analysis** finds extremes
- **NMF** finds additive parts

That makes this a very strong compare-and-contrast topic.

---

## K-means

K-means minimizes:
$$
\sum_{k=1}^K \sum_{x_i\in C_k}\|x_i-\mu_k\|^2
$$

### Main idea

- each cluster has a centroid
- each observation belongs to one cluster

### Geometric interpretation

Centroids lie in the **interior** of the data cloud as averages.

---

## Archetypal Analysis

AA approximates the data as mixtures of archetypes, where archetypes themselves are convex combinations of observed data:
$$
X \approx XSH
$$

### Main idea

- archetypes are extreme profiles
- observations are mixtures of those extremes

### Geometric interpretation

Archetypes lie on or near the **convex hull boundary**.

So AA emphasizes extremes rather than centers.

---

## NMF

NMF factorizes:
$$
X \approx WH, \quad W,H\ge 0
$$

### Main idea

- observations are additive sums of nonnegative parts
- components are not forced to be extremes or cluster centers

### Geometric interpretation

NMF components usually live in the interior of the nonnegative cone and act like building blocks.

---

## Core Comparison

| Property | K-means | Archetypal Analysis | NMF |
|----------|---------|---------------------|-----|
| Representative objects | Centroids | Archetypes | Parts / basis vectors |
| Hard or soft? | Hard assignment | Soft convex mixtures | Soft additive mixtures |
| Prototype location | Interior | Boundary/extremes | Additive latent factors |
| Best interpretation | Cluster centers | Pure types / end-members | Parts-based decomposition |

---

## What Structure Each Method Assumes

### K-means

Assumes the data consist of compact groups around centers.

### Archetypal Analysis

Assumes the data can be explained as mixtures of extreme points or end-members.

### NMF

Assumes the data can be built as additive combinations of nonnegative latent parts.

This assumption-level comparison is exactly the sort of thing that earns marks in Q21.

---

## When to Use Which

**Use K-means when**:
- you want a flat clustering
- hard assignment is meaningful
- cluster centers are the right summary

**Use Archetypal Analysis when**:
- extremes matter more than averages
- you want end-member interpretation
- observations are mixtures of pure types

**Use NMF when**:
- data are nonnegative
- additive parts interpretation is meaningful

---

## Examples

- **K-means**: customer segments with typical center profiles
- **AA**: extreme patient phenotypes or material end-members
- **NMF**: face parts, spectra, document-term factors

---

## Limitations

1. K-means is sensitive to outliers and assumes spherical structure.
2. AA can be sensitive to extreme noisy points because it seeks the boundary.
3. NMF is non-unique and requires nonnegative data.

---

## Additional Possible Exam Questions

**Q: Which method is most appropriate when you care about extreme phenotypes?**
Archetypal Analysis.

**Q: Which method is most appropriate when you want actual cluster centers?**
K-means.

**Q: Which method is most appropriate when each observation is a sum of parts?**
NMF.
