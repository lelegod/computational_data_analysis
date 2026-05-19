# Q21-AH — Sparse Coding
> Week 11. Could ask: explain the sparse coding objective, why sparsity gives useful representations, and compare sparse coding to ICA and NMF.

---

## Model

Sparse coding represents each observation as a sparse linear combination of dictionary atoms:
$$
X \approx WH
$$
where:
- $X \in \mathbb{R}^{p \times N}$ is the data matrix
- $W \in \mathbb{R}^{p \times K}$ is the dictionary
- $H \in \mathbb{R}^{K \times N}$ contains the coefficients

The key idea is that each column $x_i$ is reconstructed using only a few active coefficients in $h_i$.

---

## Objective Function

The standard sparse coding problem is:
$$
\min_{W,H} \frac{1}{2}\|X-WH\|_F^2 + \lambda \sum_{i=1}^N \|h_i\|_1
$$
subject to a normalization constraint on the dictionary columns, typically
$$
\|w_k\|_2 \leq 1 \quad \text{for all } k.
$$

**Why the constraint is needed**:
without constraining $W$, we could scale one column of $W$ up and the matching row of $H$ down, leaving the reconstruction unchanged while making the $L_1$ penalty arbitrarily small. The norm constraint removes this scaling degeneracy.

---

## Why Sparsity Helps

The $L_1$ penalty encourages most coefficients to be exactly zero:

- each observation uses only a few atoms
- the representation becomes more interpretable
- irrelevant variation is suppressed
- the model can capture local structure better than dense low-rank factorizations

This is useful when the true signal is believed to be composed of a small number of latent building blocks, such as image edges, spectral peaks, or neural activity patterns.

---

## Alternating Optimization

The full problem is **not jointly convex** in $(W,H)$, but it is convex in one block when the other is fixed.

### Step 1 — Update coefficients $H$

Fix $W$, then for each sample solve a Lasso-type problem:
$$
\min_{h_i} \frac{1}{2}\|x_i-Wh_i\|_2^2 + \lambda \|h_i\|_1
$$

This can be solved by coordinate descent, soft-thresholding methods, or LARS-type algorithms.

### Step 2 — Update dictionary $W$

Fix $H$, then solve:
$$
\min_W \|X-WH\|_F^2 \quad \text{s.t. } \|w_k\|_2 \leq 1
$$

This is a constrained least-squares problem. After updating, the dictionary columns are renormalized if needed.

### Step 3 — Repeat

Alternate the two steps until the objective stops improving.

**Convergence**: the objective decreases monotonically, but because the problem is nonconvex, the algorithm may reach only a local optimum.

---

## Geometric Interpretation

Sparse coding seeks a dictionary such that data points can be reconstructed from a few active directions.

- PCA uses dense orthogonal directions
- sparse coding uses an overcomplete, non-orthogonal dictionary
- each sample selects only a small subset of atoms

So the model is flexible globally, but sparse locally.

This is why sparse coding often learns edge-like basis functions in images: each patch activates only a few oriented edge atoms.

---

## Overcomplete Dictionaries

A major strength of sparse coding is that the dictionary can be **overcomplete**, meaning:
$$
K > p
$$

This is impossible in PCA, where components are limited by rank and orthogonality.

An overcomplete dictionary gives:
- more expressive representations
- multiple possible atoms for different local patterns
- better reconstruction of structured data

Sparsity is what keeps this flexibility from becoming arbitrary overfitting.

---

## Relation to ICA

Sparse coding and ICA are closely related.

**ICA model**:
$$
x = As, \quad \text{with independent non-Gaussian sources } s
$$

If the source prior is super-Gaussian, such as Laplacian, then MAP estimation of the latent coefficients leads to an $L_1$-type penalty. This makes sparse coding closely connected to ICA with a sparse prior.

### Sparse Coding vs ICA

| Property | Sparse Coding | ICA |
|----------|---------------|-----|
| Main goal | Sparse representation / reconstruction | Recover independent latent sources |
| Dictionary | Can be overcomplete | Usually square or constrained for identifiability |
| Penalty / assumption | $L_1$ sparsity | Statistical independence + non-Gaussianity |
| Uniqueness | No strong uniqueness | Essentially unique up to permutation/sign/scale |
| Typical interpretation | Flexible feature learning | Source separation |

**Key distinction**: ICA emphasizes identifiability of sources; sparse coding emphasizes useful sparse representation, even if the representation is not unique.

---

## Relation to NMF

Both sparse coding and NMF factorize $X \approx WH$, but the constraints differ.

### Sparse Coding vs NMF

| Property | Sparse Coding | NMF |
|----------|---------------|-----|
| Signs allowed? | Yes | No, $W,H \ge 0$ |
| Main structure | Sparsity | Additive parts |
| Dictionary shape | Unconstrained except norm bounds | Nonnegative |
| Interpretation | Few active atoms per sample | Parts-based decomposition |
| Uniqueness | No | No, $Q$-ambiguity |

**Key distinction**:
- NMF prevents cancellation because everything is nonnegative
- sparse coding allows positive and negative atoms but forces only a few to be active

So NMF is often more naturally interpretable for counts, intensities, and spectra, while sparse coding is more flexible for general signals.

---

## Sparse Coding vs PCA

| Property | PCA | Sparse Coding |
|----------|-----|---------------|
| Basis | Orthogonal | Not necessarily orthogonal |
| Representation | Dense | Sparse |
| Number of atoms | Up to rank | Can be overcomplete |
| Objective | Max variance / min rank-$K$ reconstruction error | Reconstruction + sparsity penalty |
| Interpretability | Moderate | Often higher for local patterns |

**Important consequence**: PCA spreads information over all components, while sparse coding concentrates information in a few active atoms.

---

## Limitations

1. The optimization is nonconvex, so solutions depend on initialization.
2. Choosing $K$ and $\lambda$ requires cross-validation or model selection.
3. The learned dictionary is not unique.
4. If $\lambda$ is too large, the model underfits and uses too few atoms.
5. If $\lambda$ is too small, the representation becomes dense and loses the benefit of sparsity.

---

## Additional Possible Exam Questions

**Q: Why does sparse coding typically use an $L_1$ penalty instead of an $L_0$ penalty?**
The $L_0$ penalty directly counts nonzero coefficients and would express sparsity most literally, but it leads to a combinatorial optimization problem that is computationally intractable in general. The $L_1$ penalty is a convex surrogate that still promotes exact zeros and is practical to optimize.

**Q: Why can sparse coding use more dictionary atoms than the data dimension?**
Because sparsity prevents all atoms from being used simultaneously. Even if $K>p$, each observation activates only a few atoms, so the effective local representation remains simple. This gives more flexibility than PCA without making each reconstruction dense.

**Q: What is the bias-variance intuition for increasing $\lambda$?**
Larger $\lambda$ forces sparser coefficients, which increases bias because reconstructions become less flexible, but reduces variance because the representation is more stable and less sensitive to noise. The optimal $\lambda$ balances reconstruction fidelity and sparsity.

**Q: When would sparse coding be preferred over PCA?**
When the data are thought to be generated by a small number of localized or interpretable features rather than broad global variance directions. Typical examples are image patches, neural signals, and high-dimensional structured measurements where local patterns matter more than orthogonal compression.
