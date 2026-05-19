# Week 8 — Subspace Methods: PCA, Sparse PCA, PLS, CCA (Exam Focus)

## Must-Know Facts

### PCA
- PCA maximizes **variance** of projected data: $\max_v \text{Var}(Xv)$.
- PCA is **unsupervised** — it ignores any response y.
- EVD is computed on the **covariance/correlation matrix** ($p \times p$); SVD is computed on the **data matrix X** ($n \times p$).
- Both EVD and SVD give the **same loading vectors $V$**.
- Eigenvalue $\lambda_k$ = proportion of variance explained by PC $k$ = $\lambda_k/\sum_j \lambda_j$.
- SVD standard deviation for component $l$: $\sigma_l = d_l/\sqrt{n-1}$ where $d_l$ is the $l$-th singular value.
- PCA on **unscaled data** is dominated by high-variance features (units matter).
- Principal components are orthogonal to each other; scores are uncorrelated.
- Mode of variation: $\mu \pm 2.5\sigma_l v_l$ (mean ± 2.5 SD along loading $v_l$).

### Sparse PCA
- Standard PCA loadings use ALL $p$ features — hard to interpret.
- Sparse PCA zeros out many loadings for interpretability.
- **Three methods:** Thresholding, Varimax rotation, Elastic Net.
- After **thresholding**, scores must be **recomputed** from the new loading matrix.
- After thresholding, Varimax, or Elastic Net: **uncorrelatedness of scores is NOT guaranteed** (all three methods break PCA orthogonality).
- Elastic Net (L1 + L2 penalty) produces the most principled sparse solution.

### PLS
- PLS is **supervised** — uses y to find the relevant subspace of X.
- PLS objective: $\max \text{Cov}(Xu, Yv)$ — maximizes covariance between projected X and projected y.
- $\text{Cov}(Xu,Yv)^2 = \text{Var}(Xu)\cdot\text{Var}(Yv)\cdot\text{Corr}^2(Xu,Yv)$ — PLS balances both variance and correlation.
- **PCR flaw:** PCA may discard dimensions of X most predictive of y (highest variance ≠ most predictive).
- PLS automatically ignores X-features with zero covariance to y.
- PLS latent components $z_i$ and $z_j$ are **uncorrelated** ($z_i^T z_j = 0$).
- **OLS equivalence:** When $M = p$ (all components), PLS = OLS regression.
- When $M < p$: PLS is a regularized/dimensionality-reduced regression.

### CCA
- CCA finds associations between **two matrices X and Y**.
- CCA objective: $\max \text{Corr}^2(Xu_m, Yv_m)$ — pure correlation, no variance emphasis ($u$ paired with $X$, $v$ with $Y$).
- CCA ignores internal variance of X and Y; PLS does not.
- CCA requires inverting $\Sigma_{XX}$ and $\Sigma_{YY}$ — fails when $p > n$ (singular matrices).
- **Regularized CCA (Ridge):** adds $\lambda I$ to covariance matrices before inverting.
- **Sparse CCA (PMD):** applies L1 penalty to $u$ and $v$ — fixes high-dim problem + sparsity.
- CCA produces at most $\min(p, q)$ canonical variate pairs.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|------------|-------------|
| $\max_v \text{Var}(Xv)$ | PCA objective | State what PCA optimizes |
| $\max \text{Cov}(Xu, Yv)$ | PLS objective | Distinguish PLS from PCA |
| $\max_\alpha \text{Corr}^2(y,X\alpha)\cdot\text{Var}(X\alpha)$ | PLS $m$-th direction criterion | PLS vs PCR distinction |
| $\Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX} u = \lambda^2 \Sigma_{XX} u$ | CCA generalized eigenvalue problem | CCA solution |
| $(u^T \Sigma_{XY} v)/\sqrt{u^T \Sigma_{XX} u \cdot v^T \Sigma_{YY} v}$ | CCA ratio objective | CCA formulation |
| $\hat{\phi}_{mj} = x_j^{(m-1)^T} y$ | PLS weight (Step 1) | PLS algorithm |
| $z_m = \sum_j \hat{\phi}_{mj} x_j^{(m-1)}$ | PLS latent component (Step 2) | PLS algorithm |
| $\hat{\theta}_m = z_m^T y / z_m^T z_m$ | PLS regression coeff (Step 3) | PLS algorithm |
| $\hat{y}^{(m)} = \hat{y}^{(m-1)} + \hat{\theta}_m z_m$ | PLS prediction update (Step 4) | PLS algorithm |
| $x_j^{(m)} = x_j^{(m-1)} - (z_m^T x_j^{(m-1)}/z_m^T z_m) z_m$ | PLS deflation (Step 5) | PLS algorithm |
| $z_i^T z_j = 0$ for $i \neq j$ | PLS orthogonality guarantee | Key PLS property |
| $\lambda_k / \sum_j \lambda_j$ | Proportion of variance explained by PC $k$ | PCA interpretation |
| $\sigma_l = d_l/\sqrt{n-1}$ | SD from SVD singular value | Mode of variation plots |
| $(\Sigma_{XX} + \lambda I)^{-1} \Sigma_{XY}$ | Regularized CCA | High-dim CCA |

---

## Common Traps (Wrong Answers in Exams)

- **❌ PCA maximizes correlation with y** → ✓ PCA maximizes variance of $Xv$; it has no knowledge of y.
- **❌ PCR is always better than PLS** → ✓ PCR can fail if the highest-variance X directions have zero correlation with y; PLS avoids this.
- **❌ EVD and SVD give different loading vectors** → ✓ Both give the same $V$ (loading vectors); they differ only in how scores are computed.
- **❌ EVD is computed on the data matrix X** → ✓ EVD is computed on the covariance/correlation matrix; SVD is on X directly.
- **❌ PLS components $z_i$ are correlated with each other** → ✓ PLS components are orthogonal (uncorrelated) by design of the deflation step.
- **❌ PLS with $M = p$ introduces regularization** → ✓ PLS with $M = p$ is exactly equivalent to OLS; regularization only happens when $M < p$.
- **❌ CCA maximizes variance** → ✓ CCA maximizes CORRELATION only; it ignores internal variance of X and Y.
- **❌ CCA works fine when $p > n$** → ✓ CCA requires inverting $\Sigma_{XX}$; when $p > n$, $\Sigma_{XX}$ is singular — use Regularized or Sparse CCA.
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
- If $p > n$ and you need CCA → use **Regularized CCA** (Ridge) or **Sparse CCA** (PMD).
- If PLS with $M = p$ → same predictions as OLS; reduce $M$ for regularization.
- If you want interpretable (anatomically labeled) components → use **Elastic Net** sparse PCA.
- Choose $M$ (number of PLS components) by **cross-validation** on prediction error.
- If PCA components capture all variance but regression fails → switch to PLS.
