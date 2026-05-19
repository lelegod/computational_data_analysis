# Q21-B — ICA: Uniqueness and Non-Gaussianity
> Appeared: 2024 Q21

---

## The ICA Model

$$x = As, \quad x \in \mathbb{R}^p, \; A \in \mathbb{R}^{p\times p}, \; s \in \mathbb{R}^p$$

- $x$: observed mixed signals (known)
- $A$: unknown mixing matrix
- $s$: unknown independent source signals
- Goal: find unmixing matrix $W \approx A^{-1}$ such that $\hat{s} = Wx$ recovers $s$

---

## Why Non-Gaussianity is Required

**Central Limit Theorem argument**: Linear mixtures of independent random variables converge toward Gaussian. Therefore:
- Mixed signals $x = As$ are MORE Gaussian than original sources $s$
- To unmix: find directions that are LEAST Gaussian (moving away from the mixture toward the sources)
- If sources were Gaussian: ALL rotations of a Gaussian are equally Gaussian → no direction is identifiable → ICA is completely unidentifiable

**Formal condition**: At most one source may be Gaussian for ICA to work.

---

## Measuring Non-Gaussianity

### Excess Kurtosis
$$\kappa_4 = \frac{\mu_4}{\sigma^4} - 3$$
- Gaussian: $\kappa_4 = 0$
- Super-Gaussian (heavy-tailed, leptokurtic): $\kappa_4 > 0$ (e.g., Laplace: $\kappa_4=3$, speech signals)
- Sub-Gaussian (flat, platykurtic): $\kappa_4 < 0$ (e.g., Uniform: $\kappa_4 \approx -1.2$)

**Limitation**: very sensitive to outliers (depends on 4th moment).

### Negentropy
$$J(y) = H(y_\text{Gauss}) - H(y) \geq 0$$
where $H$ is differential entropy and $y_\text{Gauss}$ has the same variance as $y$.
- $J(y) = 0$ iff $y$ is Gaussian (Gaussian maximizes entropy at fixed variance)
- More robust than kurtosis; used in FastICA

---

## The FastICA Algorithm

**Preprocessing**: Whiten the data so $E[\tilde{x}\tilde{x}^T] = I$ (zero mean, identity covariance). This reduces the search to orthogonal transformations only.

**Fixed-point iteration** (for one component $w$):
$$w_\text{new} \leftarrow E[\tilde{x} \, g(w^T\tilde{x})] - E[g'(w^T\tilde{x})]w$$
$$w \leftarrow w / \|w\|$$

where $g = \tanh$ (for super-Gaussian) or $g(u) = u\exp(-u^2/2)$ (general).

**Multiple components**: After finding $w_1$, orthogonalize:
$$w_2 \leftarrow w_2 - (w_2^T w_1)w_1, \quad w_2 \leftarrow w_2/\|w_2\|$$

**Convergence**: cubic/quadratic (much faster than gradient descent).

---

## What ICA CAN and CANNOT Determine (Uniqueness)

ICA is unique **up to**:
1. **Permutation** of components: the order of recovered sources is arbitrary
2. **Sign** of components: $s$ and $-s$ produce identical distributions
3. **Scale** (absorbed into $A$): by convention, sources have unit variance

These are **fundamental indeterminacies** — not solvable without additional constraints.

ICA is unique (up to these) when sources are non-Gaussian and independent — this is the key theorem.

---

## ICA vs PCA

| Property | PCA | ICA |
|----------|-----|-----|
| Objective | Maximize variance | Maximize non-Gaussianity / independence |
| Constraint | Orthogonal components | Statistically independent components |
| Order of components | Ranked by variance (nested) | No ordering |
| Uniqueness | Unique | Up to permutation/sign |
| Requires non-Gaussian? | No | Yes |
| Second-order stats? | Yes (covariance only) | Higher-order statistics |
| Works on Gaussian data? | Yes | No (fully unidentifiable) |

**Key distinction**: Uncorrelated $\neq$ Independent. PCA finds uncorrelated components (zero covariance). ICA finds statistically independent components (zero covariance AND all higher-order cross-cumulants zero). Independence is strictly stronger than uncorrelation for non-Gaussian distributions.

Example: Let $X_1 \sim \mathcal{N}(0,1)$ and $X_2 = X_1^2$. Then $\text{Cov}(X_1, X_2) = 0$ (uncorrelated) but $X_2$ is fully determined by $X_1$ (not independent). PCA would not detect this dependency; ICA would.

---

## Additional Possible Exam Questions

**Q: When would you choose ICA over PCA?**
When you believe the data has statistically independent non-Gaussian sources and you want to recover those sources (not just variance-maximizing directions). Examples: EEG artifact removal (heartbeat, eye movements), audio source separation (cocktail party), financial return factors.

**Q: What happens if you apply ICA to data with only Gaussian sources?**
The algorithm has no gradient to follow — all rotations of the whitened data are equally Gaussian. FastICA may converge to an arbitrary rotation, not the true sources. The result is meaningless.

**Q: Why whiten the data before ICA?**
Whitening ($E[\tilde{x}\tilde{x}^T]=I$) removes second-order correlations and normalizes scales. After whitening, the unmixing matrix is orthogonal (rotation only), reducing the search space from all invertible matrices to $O(p)$ (orthogonal group). This dramatically reduces optimization difficulty.

**Q: How many sources can ICA recover?**
At most $\min(p, N)$ sources, where $p$ is the number of mixtures (channels) and $N$ the number of observations. Typically: apply PCA first to reduce to $k<p$ components (keeping non-Gaussian ones), then apply ICA to the $k$-dimensional space.

**Q: What is the connection between ICA and sparse coding?**
Both seek a dictionary $A$ such that $x = As$ where $s$ is sparse (ICA with super-Gaussian prior ≈ sparse). The Laplace prior on $s$ in sparse coding is a super-Gaussian → equivalent to ICA with Laplacian sources under MAP estimation.
