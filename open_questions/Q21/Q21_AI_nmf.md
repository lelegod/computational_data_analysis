# Q21-AI — Non-Negative Matrix Factorization (NMF)
> Week 11. Could ask: explain the NMF objective, derive the multiplicative updates, and explain why non-negativity gives parts-based representations.

---

## Model

NMF approximates a non-negative data matrix by a product of two non-negative matrices:
$$
X \approx WH
$$
where:
- $X \in \mathbb{R}^{N \times p}$ with $X_{ij} \ge 0$
- $W \in \mathbb{R}^{N \times K}$ with $W_{ik} \ge 0$
- $H \in \mathbb{R}^{K \times p}$ with $H_{kj} \ge 0$

Each observation is represented as an additive combination of $K$ latent components.

---

## Objective Function

The standard Frobenius-loss NMF problem is:
$$
\min_{W,H \ge 0} \frac{1}{2}\|X-WH\|_F^2
$$

A common alternative for counts is KL-divergence NMF, but the basic exam version is usually the Frobenius objective.

---

## Why Non-Negativity Matters

The non-negativity constraint prevents cancellation:

- components cannot subtract from each other
- each reconstruction is purely additive
- factors are often easier to interpret as parts

This is why NMF is called a **parts-based** method. In image data, a face can be reconstructed as a sum of nose, mouth, and eye patterns, rather than as positive and negative global eigenfaces as in PCA.

---

## Alternating Optimization

The objective is **not jointly convex** in $(W,H)$, but it is convex in one block when the other is fixed.

So NMF is fitted by alternating updates:

1. Fix $W$, update $H$
2. Fix $H$, update $W$
3. Repeat until convergence

Because of non-negativity, ordinary least squares updates must be modified or projected.

---

## Multiplicative Updates

The classic Lee-Seung updates are:
$$
H \leftarrow H \odot \frac{W^T X}{W^T W H}
$$
$$
W \leftarrow W \odot \frac{X H^T}{W H H^T}
$$
where:
- $\odot$ means elementwise multiplication
- division is also elementwise

### Why these updates are useful

They have two important properties:

1. They preserve non-negativity automatically if initialized with non-negative values
2. They monotonically decrease the objective

So no explicit projection step is needed after each update.

---

## Why the Updates Preserve Non-Negativity

If all entries of $W$ and $H$ start non-negative, then:

- numerators such as $W^T X$ and $XH^T$ are non-negative
- denominators such as $W^TWH$ and $WHH^T$ are non-negative
- multiplying a non-negative matrix by a non-negative ratio keeps it non-negative

This is a major practical reason the multiplicative scheme is attractive.

---

## Interpretation of the Factors

- Rows of $H$ are latent components / parts
- Rows of $W$ contain the weights of those parts for each observation

If row $i$ of $W$ has large weight on component 2 and small weights elsewhere, then observation $i$ is mainly explained by part 2.

Because everything is non-negative, this interpretation is often much more intuitive than PCA.

---

## NMF vs PCA

| Property | NMF | PCA |
|----------|-----|-----|
| Signs allowed? | No | Yes |
| Representation | Additive parts | Global variance directions |
| Orthogonality | No | Yes |
| Uniqueness | No | Yes up to sign |
| Components nested? | No | Yes |

**Key distinction**: PCA explains variance, while NMF emphasizes interpretable additive structure.

---

## Non-Uniqueness

NMF is not unique.

If
$$
X \approx WH
$$
then for an invertible matrix $Q$,
$$
WH = (WQ^{-1})(QH)
$$
also gives the same reconstruction, as long as the transformed matrices remain non-negative.

This is the NMF **ambiguity problem**. So NMF components are not uniquely identified without extra assumptions such as sparsity or geometric constraints.

---

## NMF vs ICA

| Property | NMF | ICA |
|----------|-----|-----|
| Main constraint | Non-negativity | Independence + non-Gaussianity |
| Main goal | Parts-based decomposition | Source separation |
| Signs allowed? | No | Yes |
| Uniqueness | No | Essentially yes |
| Typical data | Images, counts, spectra | Mixed signals, EEG, audio |

**Key distinction**: NMF is about additive interpretability; ICA is about recovering independent latent sources.

---

## NMF vs Sparse Coding

| Property | NMF | Sparse Coding |
|----------|-----|---------------|
| Constraint | Non-negative factors | Sparse coefficients |
| Signs allowed? | No | Usually yes |
| Dictionary size | Often low-rank | Can be overcomplete |
| Interpretability | Additive parts | Few active atoms |

Sparse coding allows positive and negative atoms but forces only a few to be used. NMF instead allows many active atoms, but all are additive.

---

## Limitations

1. The optimization is nonconvex, so local minima depend on initialization.
2. The factorization is not unique.
3. Choosing the number of components $K$ requires model selection or cross-validation.
4. NMF requires non-negative data, so it is not natural for centered data with negative values.
5. Reconstruction quality may improve with larger $K$, but interpretation can deteriorate.

---

## Additional Possible Exam Questions

**Q: Why does NMF tend to give parts-based decompositions?**
Because both $W$ and $H$ are constrained to be non-negative, each reconstruction is an additive sum of components with no subtraction. This means each latent component behaves like a "part" that can be present to varying degrees.

**Q: Why is NMF not jointly convex?**
The objective is bilinear in $W$ and $H$ through the product $WH$. If one block is fixed, the problem becomes convex in the other, but jointly the landscape has many local optima.

**Q: Why are multiplicative updates preferred over naive gradient descent?**
Because they preserve non-negativity automatically and decrease the objective without requiring a projection step after every update.

**Q: When would you prefer NMF over PCA?**
When the data are naturally non-negative and interpretability of additive components matters more than pure variance maximization. Examples include face images, document-term matrices, spectra, and count data.
