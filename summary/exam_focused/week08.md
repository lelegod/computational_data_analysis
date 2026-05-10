# Week 8 — Subspace Methods: PCA, Sparse PCA, PLS, CCA (Exam Focus)

## Must-Know Facts

### PCA
- PCA maximizes **variance** of projected data: `max_v Var(Xv)`.
- PCA is **unsupervised** — it ignores any response y.
- EVD is computed on the **covariance/correlation matrix** (p×p); SVD is computed on the **data matrix X** (n×p).
- Both EVD and SVD give the **same loading vectors V**.
- Eigenvalue λₖ = proportion of variance explained by PC k = `λₖ/Σλⱼ`.
- SVD standard deviation for component l: `σₗ = dₗ/√(n−1)` where dₗ is the l-th singular value.
- PCA on **unscaled data** is dominated by high-variance features (units matter).
- Principal components are orthogonal to each other; scores are uncorrelated.
- Mode of variation: `μ ± 2.5σₗ vₗ` (mean ± 2.5 SD along loading vₗ).

### Sparse PCA
- Standard PCA loadings use ALL p features — hard to interpret.
- Sparse PCA zeros out many loadings for interpretability.
- **Three methods:** Thresholding, Varimax rotation, Elastic Net.
- After **thresholding**, scores must be **recomputed** from the new loading matrix.
- After thresholding or Varimax: **uncorrelatedness of scores is NOT guaranteed**.
- Elastic Net (L1 + L2 penalty) produces the most principled sparse solution.

### PLS
- PLS is **supervised** — uses y to find the relevant subspace of X.
- PLS objective: `max Cov(Xu, Yv)` — maximizes covariance between projected X and projected y.
- `Cov(Xu,Yv)² = Var(Xu)·Var(Yv)·Corr²(Xu,Yv)` — PLS balances both variance and correlation.
- **PCR flaw:** PCA may discard dimensions of X most predictive of y (highest variance ≠ most predictive).
- PLS automatically ignores X-features with zero covariance to y.
- PLS latent components zᵢ and zⱼ are **uncorrelated** (`zᵢ^T zⱼ = 0`).
- **OLS equivalence:** When M = p (all components), PLS = OLS regression.
- When M < p: PLS is a regularized/dimensionality-reduced regression.

### CCA
- CCA finds associations between **two matrices X and Y**.
- CCA objective: `max Corr²(Yu_m, Xv_m)` — pure correlation, no variance emphasis.
- CCA ignores internal variance of X and Y; PLS does not.
- CCA requires inverting Σ_XX and Σ_YY — fails when p > n (singular matrices).
- **Regularized CCA (Ridge):** adds λI to covariance matrices before inverting.
- **Sparse CCA (PMD):** applies L1 penalty to u and v — fixes high-dim problem + sparsity.
- CCA produces at most `min(p, q)` canonical variate pairs.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|------------|-------------|
| `max_v Var(Xv)` | PCA objective | State what PCA optimizes |
| `max Cov(Xu, Yv)` | PLS objective | Distinguish PLS from PCA |
| `max_α Corr²(y,Xα)·Var(Xα)` | PLS m-th direction criterion | PLS vs PCR distinction |
| `Σ_XY Σ_YY⁻¹ Σ_YX u = λ² Σ_XX u` | CCA generalized eigenvalue problem | CCA solution |
| `(u^T Σ_XY v)/√(u^T Σ_XX u · v^T Σ_YY v)` | CCA ratio objective | CCA formulation |
| `φ̂ₘⱼ = xⱼ^(m−1)^T y` | PLS weight (Step 1) | PLS algorithm |
| `zₘ = Σⱼ φ̂ₘⱼ xⱼ^(m−1)` | PLS latent component (Step 2) | PLS algorithm |
| `θ̂ₘ = zₘ^T y / zₘ^T zₘ` | PLS regression coeff (Step 3) | PLS algorithm |
| `ŷ^(m) = ŷ^(m−1) + θ̂ₘ zₘ` | PLS prediction update (Step 4) | PLS algorithm |
| `xⱼ^(m) = xⱼ^(m−1) − (zₘ^T xⱼ^(m−1)/zₘ^T zₘ) zₘ` | PLS deflation (Step 5) | PLS algorithm |
| `zᵢ^T zⱼ = 0` for i≠j | PLS orthogonality guarantee | Key PLS property |
| `λₖ / Σλⱼ` | Proportion of variance explained by PC k | PCA interpretation |
| `σₗ = dₗ/√(n−1)` | SD from SVD singular value | Mode of variation plots |
| `(Σ_XX + λI)⁻¹ Σ_XY` | Regularized CCA | High-dim CCA |

---

## Common Traps (Wrong Answers in Exams)

- **❌ PCA maximizes correlation with y** → ✓ PCA maximizes variance of Xv; it has no knowledge of y.
- **❌ PCR is always better than PLS** → ✓ PCR can fail if the highest-variance X directions have zero correlation with y; PLS avoids this.
- **❌ EVD and SVD give different loading vectors** → ✓ Both give the same V (loading vectors); they differ only in how scores are computed.
- **❌ EVD is computed on the data matrix X** → ✓ EVD is computed on the covariance/correlation matrix; SVD is on X directly.
- **❌ PLS components zᵢ are correlated with each other** → ✓ PLS components are orthogonal (uncorrelated) by design of the deflation step.
- **❌ PLS with M = p introduces regularization** → ✓ PLS with M = p is exactly equivalent to OLS; regularization only happens when M < p.
- **❌ CCA maximizes variance** → ✓ CCA maximizes CORRELATION only; it ignores internal variance of X and Y.
- **❌ CCA works fine when p > n** → ✓ CCA requires inverting Σ_XX; when p > n, Σ_XX is singular — use Regularized or Sparse CCA.
- **❌ Sparse PCA scores remain uncorrelated after thresholding** → ✓ Thresholding destroys the orthogonality property; scores must be recomputed AND may be correlated.
- **❌ PLS and CCA have the same objective** → ✓ PLS maximizes covariance (variance × correlation); CCA maximizes correlation only.
- **❌ Scaling doesn't matter for PCA** → ✓ Unscaled PCA is dominated by high-variance features; use correlation matrix (scaled) for equal weighting.
- **❌ All sparse PCA methods preserve orthogonality** → ✓ None of the three sparse PCA methods (threshold, varimax, elastic net) guarantee uncorrelated scores.

---

## Quick Decision Rules

- Use **PCA** when you want to explore variance structure in X with no response variable.
- Use **PLS** when you want to predict y from X and suspect high-variance X dimensions are irrelevant to y.
- Use **PCR** only when you are sure PCA directions are relevant — otherwise prefer PLS.
- Use **CCA** when you have two separate data matrices (X and Y) and want to find their shared structure.
- If p > n and you need CCA → use **Regularized CCA** (Ridge) or **Sparse CCA** (PMD).
- If PLS with M = p → same predictions as OLS; reduce M for regularization.
- If you want interpretable (anatomically labeled) components → use **Elastic Net** sparse PCA.
- Choose M (number of PLS components) by **cross-validation** on prediction error.
- If PCA components capture all variance but regression fails → switch to PLS.
