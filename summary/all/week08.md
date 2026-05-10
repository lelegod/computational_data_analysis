# Week 8 — Subspace Methods (PCA, Sparse PCA, PLS, CCA)

## Overview
This week covers four related methods for finding low-dimensional subspaces in data: PCA (unsupervised, maximizes variance), Sparse PCA (interpretable sparse loadings), Partial Least Squares/PLS (supervised, maximizes covariance with a response y), and Canonical Correlation Analysis/CCA (finds associations between two data matrices X and Y). All methods are dimensionality reduction techniques but with different objectives.

---

## 1. Principal Component Analysis (PCA)

### Key Concepts
- **Goal:** Find a low-dimensional linear subspace that captures maximum variance in X.
- **Unsupervised:** Uses only X, ignores any response variable y.
- PCA finds directions v (loading vectors) such that the projected scores Xv have maximum variance.
- Principal components are **orthogonal** to each other.
- The loadings (eigenvectors) are orthonormal: vᵢ^T vⱼ = 0 for i ≠ j.
- Scores matrix S = XV (projections of data onto loading vectors).

### Formulas
- **PCA objective:** `max_v Var(Xv)` subject to `‖v‖ = 1`
- **Covariance matrix decomposition (EVD):** `Σ = VΛV^T`
  - V: matrix of eigenvectors (loading vectors / principal axes)
  - Λ: diagonal matrix of eigenvalues (variance along each PC)
- **SVD of data matrix:** `X = UDV^T`
  - U: left singular vectors (scores, up to scaling)
  - D: diagonal matrix of singular values
  - V: right singular vectors = loading vectors (same as from EVD of covariance)
  - Relationship: eigenvalues λᵢ = dᵢ²/(n−1), scores S = UD
- **Variance explained by component k:** `λₖ / Σⱼ λⱼ`
- **Standard deviations from SVD:** `σₗ = dₗ/√(n−1)` where dₗ is the l-th singular value.
- **Mode of variation at ±2.5 SD:** `μ ± 2.5σₗ vₗ` (mean face ± 2.5 standard deviations along PC l)

### EVD vs SVD
- **EVD** is computed on the **correlation or covariance matrix** Σ (size p×p).
- **SVD** is computed on the **data matrix** X itself (size n×p).
- Both give the same loading vectors V; use SVD for efficiency when n < p.

### Pitfall: Scaling
- PCA on **unscaled data**: dominated by high-variance features (units matter).
- PCA on **scaled/correlation** data: each feature equally weighted.
- Rule: use correlation matrix (scale features) unless features are already on the same scale.

---

## 2. Sparse PCA

### Key Concepts
- Standard PCA loadings use all p features (no zeros) — hard to interpret.
- Sparse PCA forces many loadings to be **exactly zero** → interpretable components.
- Trade-off: sparse components explain less variance but are more interpretable.
- Sparsity breaks the exact orthogonality of loadings (scores may also become correlated).

### Three Methods for Sparse PCA (as taught in exercises)

#### Method A: Thresholding
- Run standard PCA, then zero out all loadings with |loading| < threshold (e.g., 0.15).
- Scores must be **recomputed** after thresholding (using the modified loading matrix).
- Warning: uncorrelatedness of scores is no longer guaranteed after thresholding.

#### Method B: Varimax Rotation
- Rotate loading matrix to maximize sparsity while preserving explained variance.
- The number of columns rotated affects sparsity.
- Warning: Varimax-rotated scores are generally no longer uncorrelated.
- Use `varimax()` function in R or equivalent.

#### Method C: Elastic Net (Penalized Regression)
- Solve PCA as a regression problem with L1 (LASSO) + L2 (Ridge) penalty → Elastic Net.
- Produces sparse loading vectors (exactly zero coefficients for many features).
- Normalize the resulting loading vector to unit length.
- Start with a chosen number of nonzero loadings and try different values.
- Unlike Varimax, this can produce more anatomically interpretable components.

### Formulas
- **Varimax criterion:** Maximize `Σⱼ Σₖ (vⱼₖ⁴) − (Σⱼ vⱼₖ²)²` (simplified) — maximizes variance of squared loadings.
- **Elastic Net:** adds `λ₁‖v‖₁ + λ₂‖v‖₂²` penalty to induce sparsity.

---

## 3. Partial Least Squares (PLS)

### Key Concepts
- **Supervised** dimensionality reduction: finds subspace of X that best predicts y.
- PCA ignores y and may keep directions with high variance but zero correlation to y.
- PLS explicitly uses the relationship between X and y.
- **PCR (Principal Component Regression)** = PCA on X, then OLS regression. The flaw: high-variance directions in X may have zero correlation with y.
- PLS fixes this by finding directions that **maximize covariance with y**.

### Objective
- **PCA objective:** `max_v Var(Xv)`
- **PLS objective:** `max_{u,v} Cov(Xu, Yv)` (for multivariate Y)
- For univariate y: `max_α Corr²(y, Xα) · Var(Xα)`
- Key identity: `Cov(Xu, Yv)² = Var(Xu) · Var(Yv) · Corr²(Xu, Yv)` — PLS explicitly balances variance and correlation.
- The m-th PLS direction αₘ solves: `max_α Corr²(y, Xα) · Var(Xα)` subject to `‖α‖ = 1` and orthogonality to previous components.

### PLS Algorithm (Iterative)

**Variable Definitions:**
- X ∈ ℝⁿˣᵖ: Standardized feature matrix (n samples, p features)
- y ∈ ℝⁿ: Standardized target vector
- M: Desired number of latent components (M ≤ p)
- m: Current iteration counter (m = 1,...,M)
- xⱼ^(m−1): j-th feature column after m−1 deflations
- φ̂ₘⱼ: Weight (covariance) of feature j for component m
- zₘ ∈ ℝⁿ: The m-th extracted latent component (score vector)
- θ̂ₘ: Scalar regression coefficient for component m
- ŷ^(m): Prediction vector at iteration m

**Step 0: Initialization**
- Standardize columns of X (mean 0, variance 1)
- Standardize y (mean 0, variance 1)
- Initialize: `ŷ^(0) = μ_y = 0` (since y is standardized)
- Initialize features: `xⱼ^(0) = xⱼ` for j = 1,...,p

**Loop for m = 1 to M:**

Step 1 — Calculate Weights (Covariance):
```
φ̂ₘⱼ = xⱼ^(m−1)^T y
```
Features highly correlated with y get larger weights.

Step 2 — Construct Latent Component:
```
zₘ = Σⱼ φ̂ₘⱼ xⱼ^(m−1)
```
Weighted sum of features — the m-th PLS score.

Step 3 — Calculate Regression Coefficient:
```
θ̂ₘ = zₘ^T y / (zₘ^T zₘ)
```
OLS regression of y onto zₘ.

Step 4 — Update Prediction:
```
ŷ^(m) = ŷ^(m−1) + θ̂ₘ zₘ
```

Step 5 — Orthogonalize (Deflation):
```
xⱼ^(m) = xⱼ^(m−1) − (zₘ^T xⱼ^(m−1) / zₘ^T zₘ) zₘ    for j = 1,...,p
```
Strips variance explained by zₘ so the next component captures entirely new information.

**Key Properties of PLS:**
- PLS latent components zᵢ and zⱼ are **uncorrelated** (orthogonal): `zᵢ^T zⱼ = 0` for i ≠ j.
- **OLS Equivalence:** If M = p (keep all components), PLS predictions = OLS predictions.
- Choosing M < p provides dimensionality reduction and regularization.

### Example Intuition
- Wrist OCD data: X = wearable biosignals, y = OCD severity.
  - PCA might capture child running (high variance, irrelevant to OCD).
  - PLS finds signals that directly predict OCD episodes.
- Audio emotion data (y = anger level):
  - PCR failure: keeps loud background hum (high variance, zero correlation to anger).
  - PLS success: elevates subtle vocal quiver (low variance, high correlation to anger).

---

## 4. Canonical Correlation Analysis (CCA)

### Key Concepts
- **Goal:** Find associations between **two data matrices** X (n×p) and Y (n×q).
- Finds pairs of linear combinations (canonical variates) U = Xu and V = Yv that maximize correlation.
- Unlike PLS, CCA **ignores internal variance** of X and Y; focuses purely on **cross-correlation**.
- CCA produces at most `min(p, q)` canonical variate pairs.

### Objective
```
max_{u,v}  Corr²(Yu_m, Xv_m)
subject to  uₘ^T uⱼ = 0  and  vₘ^T vⱼ = 0  for m ≠ j
```
The linear combinations are uncorrelated across different pairs.

### CCA Formulation (Optimization)
- Seek canonical variates U = Xu and V = Yv.
- **Objective (ratio form):**
```
max_{u,v}  (u^T Σ_XY v) / sqrt(u^T Σ_XX u · v^T Σ_YY v)
```
Where:
  - Σ_XY = X^T Y: cross-covariance matrix
  - Σ_XX = X^T X: within-X covariance
  - Σ_YY = Y^T Y: within-Y covariance

### CCA Derivation: Lagrangian
Maximize covariance subject to unit variance constraints: `u^T Σ_XX u = 1` and `v^T Σ_YY v = 1`.

Lagrangian:
```
L(u, v, λ₁, λ₂) = u^T Σ_XY v − (λ₁/2)(u^T Σ_XX u − 1) − (λ₂/2)(v^T Σ_YY v − 1)
```
Why constrain variance? Without constraints, algorithm scales u, v to infinity to make covariance arbitrarily large. Constraints force focus on structural relationship, not magnitude.

### CCA Solution: Generalized Eigenvalue Problem
Taking partial derivatives and substituting yields:
```
Σ_XY Σ_YY⁻¹ Σ_YX u = λ² Σ_XX u
```
- u: canonical weights (eigenvectors)
- λ: canonical correlation (sqrt of eigenvalues)
- Python: `M = inv(S_XX) @ S_XY @ inv(S_YY) @ S_YX; eig_vals, eig_vecs = np.linalg.eig(M)`
  - Equivalent to `sklearn.cross_decomposition.CCA`

### Sparse and Regularized CCA
- **Problem:** CCA requires inverting Σ_XX and Σ_YY. When p >> n, Σ_XX is **singular** (non-invertible) — CCA crashes.
- **Regularized CCA (Ridge):** `(Σ_XX + λ_X I)⁻¹ Σ_XY ...` (adds λI to make invertible)
  - References: Vinod 1976, Leurgans et al. 1993
- **Sparse CCA (PMD — Penalized Matrix Decomposition):** Applies L1 penalties to u and v to yield sparse canonical vectors. Solves the invertibility problem + selects informative features.
  - Reference: Witten et al. 2009
- Example: 3000 audio features, 50 patients → Σ_XX is singular. Sparse CCA solves this while zeroing out uninformative features.

---

## 5. Comparison of Methods

| Method | Objective | Supervised? | Two matrices? |
|--------|-----------|-------------|---------------|
| PCA | max Var(Xv) | No | No |
| Sparse PCA | max Var(Xv) + sparsity | No | No |
| PLS | max Cov(Xu, Yv) | Yes (y) | Optional |
| CCA | max Corr²(Xu, Yv) | — | Yes (X and Y) |

- **PCA vs PLS:** PCA maximizes variance in X only; PLS maximizes covariance between X and y.
- **PLS vs CCA:** PLS balances variance and correlation; CCA ignores internal variance, focuses only on cross-correlation.
- **PCR vs PLS:** PCR first reduces X (may discard y-relevant dimensions); PLS uses y to guide reduction.

---

## 6. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| PCA objective | `max_v Var(Xv)` s.t. `‖v‖=1` |
| PLS objective | `max_{u,v} Cov(Xu, Yv)` |
| PLS m-th direction | `max_α Corr²(y, Xα)·Var(Xα)` s.t. `‖α‖=1` |
| CCA objective | `max_{u,v} (u^T Σ_XY v)/sqrt(u^T Σ_XX u · v^T Σ_YY v)` |
| CCA eigenvalue problem | `Σ_XY Σ_YY⁻¹ Σ_YX u = λ² Σ_XX u` |
| PLS weight step | `φ̂ₘⱼ = xⱼ^(m−1)^T y` |
| PLS score | `zₘ = Σⱼ φ̂ₘⱼ xⱼ^(m−1)` |
| PLS regression coeff | `θ̂ₘ = zₘ^T y / zₘ^T zₘ` |
| PLS deflation | `xⱼ^(m) = xⱼ^(m−1) − (zₘ^T xⱼ^(m−1)/zₘ^T zₘ) zₘ` |
| PLS orthogonality | `zᵢ^T zⱼ = 0` for i≠j |
| Regularized CCA | `(Σ_XX + λI)⁻¹ Σ_XY ...` |
| Variance explained | `λₖ/Σλⱼ` |
| SVD standard deviation | `σₗ = dₗ/√(n−1)` |
