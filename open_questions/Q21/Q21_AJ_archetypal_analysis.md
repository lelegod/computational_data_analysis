# Q21-AJ — Archetypal Analysis (AA)
> Week 11. Could ask: explain the AA objective, why archetypes lie on the convex hull, and compare AA to PCA, NMF, and K-means.

---

## Model

Archetypal Analysis represents data using extreme prototype profiles called **archetypes**.

The key construction is:
$$
Z = XS
$$
where:
- $X \in \mathbb{R}^{N \times p}$ is the data matrix
- $Z \in \mathbb{R}^{K \times p}$ contains the archetypes
- $S$ contains convex weights defining each archetype as a convex combination of data points

Then each observation is reconstructed as a convex combination of those archetypes:
$$
X \approx HZ
$$

Combining both steps gives:
$$
X \approx HXS
$$

Depending on convention, the course may write the equivalent form as $X \approx XSH$. The essential idea is unchanged: data are approximated by mixtures of archetypes, and archetypes themselves are mixtures of actual data points.

---

## Convexity Constraints

AA imposes two convexity constraints:

1. Archetypes must be convex combinations of data points
2. Each data point must be a convex combination of archetypes

So the weights satisfy:
- non-negativity
- rows or columns summing to 1, depending on matrix convention

These constraints force the archetypes to lie on or near the **convex hull** of the data.

---

## Objective Function

The AA objective is:
$$
\min_{S,H} \|X - HXS\|_F^2
$$

or equivalently, under the alternative convention,
$$
\min_{S,H} \|X - XSH\|_F^2
$$

subject to the convexity constraints on $S$ and $H$.

The point is not the exact matrix orientation, but the geometry:
- archetypes are built from data
- observations are rebuilt from archetypes

---

## Why Archetypes Are Extreme

Because each archetype is a convex combination of observed data points, it cannot leave the convex hull of the dataset.

To minimize reconstruction error well, the optimal archetypes tend to move toward the boundary, where they capture the extreme directions of the cloud.

So AA finds:
- "pure types"
- "end-members"
- extreme phenotypes

rather than average profiles.

---

## Why This Is Different from PCA

PCA finds directions of maximum variance through the middle of the cloud.

AA instead tries to explain the cloud by its corners.

This makes AA especially useful when intermediate observations are mixtures of extreme profiles.

Example:
- PCA gives axes of variation
- AA gives extreme patient or material profiles

---

## Optimization

The problem is not jointly convex in $(S,H)$, so AA is fitted by alternating optimization:

1. Fix archetypes, update mixture weights
2. Fix mixture weights, update archetype weights
3. Repeat until the reconstruction error stabilizes

This again gives a nonconvex problem with possible local optima.

---

## AA vs K-means

| Property | AA | K-means |
|----------|----|---------|
| Prototypes | Extreme archetypes | Interior centroids |
| Geometry | Convex hull based | Mean-based clustering |
| Representation | Mixture of archetypes | Hard assignment |
| Prototype location | Boundary | Interior |

**Key distinction**: K-means finds cluster centers; AA finds extremes.

---

## AA vs NMF

| Property | AA | NMF |
|----------|----|-----|
| Main idea | Extremes / convex hull | Additive parts |
| Prototype source | Must come from data convex hull | Factors not anchored to data |
| Constraints | Doubly convex | Non-negative |
| Best interpretation | End-members / pure types | Parts-based decomposition |

NMF can produce interpretable components, but they are not forced to correspond to extreme observed profiles. AA explicitly searches for those extremes.

---

## AA vs PCA

| Property | AA | PCA |
|----------|----|-----|
| Target | Extreme profiles | Maximum variance directions |
| Prototypes | Boundary | Interior/global directions |
| Components orthogonal? | No | Yes |
| Interpretability | High when extremes matter | High when variance structure matters |

**Key distinction**: PCA summarizes variation; AA summarizes extremal structure.

---

## When AA Is Useful

AA is especially useful when:

- data are mixtures of pure types
- extreme profiles matter scientifically
- interpretability is more important than purely minimizing variance

Typical examples:
- patient phenotypes
- material end-members
- consumer behavior extremes
- environmental mixtures

---

## Limitations

1. The optimization is nonconvex and initialization matters.
2. Archetypes may be sensitive to outliers because they live near the boundary.
3. Choosing the number of archetypes $K$ requires model selection.
4. AA is usually more computationally demanding than PCA or K-means.
5. If the true structure is not mixture-of-extremes, AA may be less natural than PCA or NMF.

---

## Additional Possible Exam Questions

**Q: Why do archetypes lie on the convex hull of the data?**
Because each archetype is a convex combination of observed data points. Convex combinations cannot leave the convex hull, so the feasible set for archetypes is restricted to that geometry.

**Q: Why does AA find extremes while K-means finds centers?**
K-means minimizes squared distance to centroids, so the optimal representatives are means in the interior. AA instead explains the cloud as mixtures of prototypes and therefore pushes the prototypes toward the boundary to span the data.

**Q: When would you use AA rather than PCA?**
When you care about finding extreme interpretable profiles rather than average directions of variation. PCA is better for compression and variance structure; AA is better for end-member interpretation.

**Q: What is the main statistical risk of AA?**
Because archetypes are boundary-seeking, they can be distorted by outliers or noisy extreme observations. So robust preprocessing and careful validation matter.
