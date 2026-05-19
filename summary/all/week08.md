# Week 8 — Subspace Methods (PCA, Sparse PCA, PLS, CCA)

## Overview
This week covers four related methods for finding low-dimensional subspaces in data: PCA (unsupervised, maximizes variance), Sparse PCA (interpretable sparse loadings), Partial Least Squares/PLS (supervised, maximizes covariance with a response y), and Canonical Correlation Analysis/CCA (finds associations between two data matrices X and Y). All methods are dimensionality reduction techniques but with different objectives.

---

## 1. Principal Component Analysis (PCA)

### Key Concepts
- **Goal:** Find a low-dimensional linear subspace that captures maximum variance in X.
- **Unsupervised:** Uses only X, ignores any response variable y.
- PCA finds directions $v$ (loading vectors) such that the projected scores $Xv$ have maximum variance.
- Principal components are **orthogonal** to each other.
- The loadings (eigenvectors) are orthonormal: $v_i^T v_j = 0$ for $i \neq j$.
- Scores matrix $S = XV$ (projections of data onto loading vectors).

### Formulas
- **PCA objective:** $\max_v \text{Var}(Xv)$ subject to $\|v\| = 1$
- **Covariance matrix decomposition (EVD):** $\Sigma = V\Lambda V^T$
  - $V$: matrix of eigenvectors (loading vectors / principal axes)
  - $\Lambda$: diagonal matrix of eigenvalues (variance along each PC)
- **SVD of data matrix:** $X = UDV^T$
  - $U$: left singular vectors (scores, up to scaling)
  - $D$: diagonal matrix of singular values
  - $V$: right singular vectors = loading vectors (same as from EVD of covariance)
  - Relationship: eigenvalues $\lambda_i = d_i^2/(n-1)$, scores $S = UD$
- **Variance explained by component $k$:** $\lambda_k / \sum_j \lambda_j$
- **Standard deviations from SVD:** $\sigma_l = d_l/\sqrt{n-1}$ where $d_l$ is the $l$-th singular value.
- **Mode of variation at $\pm 2.5$ SD:** $\mu \pm 2.5\sigma_l v_l$ (mean face ± 2.5 standard deviations along PC $l$)

### EVD vs SVD
- **EVD** is computed on the **correlation or covariance matrix** $\Sigma$ (size $p \times p$).
- **SVD** is computed on the **data matrix** $X$ itself (size $n \times p$).
- Both give the same loading vectors $V$; use SVD for efficiency when $n < p$.

### Pitfall: Scaling
- PCA on **unscaled data**: dominated by high-variance features (units matter).
- PCA on **scaled/correlation** data: each feature equally weighted.
- Rule: use correlation matrix (scale features) unless features are already on the same scale.

---

## 2. Sparse PCA

### Key Concepts
- Standard PCA loadings use all $p$ features (no zeros) — hard to interpret.
- Sparse PCA forces many loadings to be **exactly zero** → interpretable components.
- Trade-off: sparse components explain less variance but are more interpretable.
- Sparsity breaks the exact orthogonality of loadings (scores may also become correlated).

### Three Methods for Sparse PCA (as taught in exercises)

#### Method A: Thresholding
- Run standard PCA, then zero out all loadings with $|\text{loading}| < \text{threshold}$ (e.g., 0.15).
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
- **Varimax criterion:** Maximize $\sum_j \sum_k (v_{jk}^4) - (\sum_j v_{jk}^2)^2$ (simplified) — maximizes variance of squared loadings.
- **Elastic Net:** adds $\lambda_1 \|v\|_1 + \lambda_2 \|v\|_2^2$ penalty to induce sparsity.

---

## 3. Partial Least Squares (PLS)

### Key Concepts
- **Supervised** dimensionality reduction: finds subspace of X that best predicts y.
- PCA ignores y and may keep directions with high variance but zero correlation to y.
- PLS explicitly uses the relationship between X and y.
- **PCR (Principal Component Regression)** = PCA on X, then OLS regression. The flaw: high-variance directions in X may have zero correlation with y.
- PLS fixes this by finding directions that **maximize covariance with y**.

### Objective
- **PCA objective:** $\max_v \text{Var}(Xv)$
- **PLS objective:** $\max_{u,v} \text{Cov}(Xu, Yv)$ (for multivariate $Y$)
- For univariate $y$: $\max_\alpha \text{Corr}^2(y, X\alpha) \cdot \text{Var}(X\alpha)$
- Key identity: $\text{Cov}(Xu, Yv)^2 = \text{Var}(Xu) \cdot \text{Var}(Yv) \cdot \text{Corr}^2(Xu, Yv)$ — PLS explicitly balances variance and correlation.
- The $m$-th PLS direction $\alpha_m$ solves: $\max_\alpha \text{Corr}^2(y, X\alpha) \cdot \text{Var}(X\alpha)$ subject to $\|\alpha\| = 1$ and orthogonality to previous components.

### PLS Algorithm (Iterative)

**Variable Definitions:**
- $X \in \mathbb{R}^{n \times p}$: Standardized feature matrix ($n$ samples, $p$ features)
- $y \in \mathbb{R}^n$: Standardized target vector
- $M$: Desired number of latent components ($M \leq p$)
- $m$: Current iteration counter ($m = 1,\ldots,M$)
- $x_j^{(m-1)}$: $j$-th feature column after $m-1$ deflations
- $\hat{\phi}_{mj}$: Weight (covariance) of feature $j$ for component $m$
- $z_m \in \mathbb{R}^n$: The $m$-th extracted latent component (score vector)
- $\hat{\theta}_m$: Scalar regression coefficient for component $m$
- $\hat{y}^{(m)}$: Prediction vector at iteration $m$

**Step 0: Initialization**
- Standardize columns of X (mean 0, variance 1)
- Standardize y (mean 0, variance 1)
- Initialize: $\hat{y}^{(0)} = \mu_y = 0$ (since y is standardized)
- Initialize features: $x_j^{(0)} = x_j$ for $j = 1,\ldots,p$

**Loop for $m = 1$ to $M$:**

Step 1 — Calculate Weights (Covariance):

$$\hat{\phi}_{mj} = x_j^{(m-1)^T} y$$

Features highly correlated with y get larger weights.

Step 2 — Construct Latent Component:

$$z_m = \sum_j \hat{\phi}_{mj} x_j^{(m-1)}$$

Weighted sum of features — the $m$-th PLS score.

Step 3 — Calculate Regression Coefficient:

$$\hat{\theta}_m = z_m^T y / (z_m^T z_m)$$

OLS regression of $y$ onto $z_m$.

Step 4 — Update Prediction:

$$\hat{y}^{(m)} = \hat{y}^{(m-1)} + \hat{\theta}_m z_m$$

Step 5 — Orthogonalize (Deflation):

$$x_j^{(m)} = x_j^{(m-1)} - \frac{z_m^T x_j^{(m-1)}}{z_m^T z_m} z_m \quad \text{for } j = 1,\ldots,p$$

Strips variance explained by $z_m$ so the next component captures entirely new information.

**Key Properties of PLS:**
- PLS latent components $z_i$ and $z_j$ are **uncorrelated** (orthogonal): $z_i^T z_j = 0$ for $i \neq j$.
- **OLS Equivalence:** If $M = p$ (keep all components), PLS predictions = OLS predictions.
- Choosing $M < p$ provides dimensionality reduction and regularization.

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
- **Goal:** Find associations between **two data matrices** $X$ ($n \times p$) and $Y$ ($n \times q$).
- Finds pairs of linear combinations (canonical variates) $U = Xu$ and $V = Yv$ that maximize correlation.
- Unlike PLS, CCA **ignores internal variance** of X and Y; focuses purely on **cross-correlation**.
- CCA produces at most $\min(p, q)$ canonical variate pairs.

### Objective

$$\max_{u,v} \; \text{Corr}^2(Xu_m, Yv_m) \quad \text{subject to} \quad u_m^T u_j = 0 \text{ and } v_m^T v_j = 0 \text{ for } m \neq j$$

The linear combinations are uncorrelated across different pairs.

### CCA Formulation (Optimization)
- Seek canonical variates $U = Xu$ and $V = Yv$.
- **Objective (ratio form):**

$$\max_{u,v} \; \frac{u^T \Sigma_{XY} v}{\sqrt{u^T \Sigma_{XX} u \cdot v^T \Sigma_{YY} v}}$$

Where:
  - $\Sigma_{XY} = X^T Y$: cross-covariance matrix
  - $\Sigma_{XX} = X^T X$: within-X covariance
  - $\Sigma_{YY} = Y^T Y$: within-Y covariance

### CCA Derivation: Lagrangian
Maximize covariance subject to unit variance constraints: $u^T \Sigma_{XX} u = 1$ and $v^T \Sigma_{YY} v = 1$.

Lagrangian:

$$L(u, v, \lambda_1, \lambda_2) = u^T \Sigma_{XY} v - \frac{\lambda_1}{2}(u^T \Sigma_{XX} u - 1) - \frac{\lambda_2}{2}(v^T \Sigma_{YY} v - 1)$$

Why constrain variance? Without constraints, algorithm scales $u$, $v$ to infinity to make covariance arbitrarily large. Constraints force focus on structural relationship, not magnitude.

### CCA Solution: Generalized Eigenvalue Problem
Taking partial derivatives and substituting yields:

$$\Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX} u = \lambda^2 \Sigma_{XX} u$$

- $u$: canonical weights (eigenvectors)
- $\lambda$: canonical correlation (sqrt of eigenvalues)
- Python: `M = inv(S_XX) @ S_XY @ inv(S_YY) @ S_YX; eig_vals, eig_vecs = np.linalg.eig(M)`
  - Equivalent to `sklearn.cross_decomposition.CCA`

### Sparse and Regularized CCA
- **Problem:** CCA requires inverting $\Sigma_{XX}$ and $\Sigma_{YY}$. When $p \gg n$, $\Sigma_{XX}$ is **singular** (non-invertible) — CCA crashes.
- **Regularized CCA (Ridge):** $(\Sigma_{XX} + \lambda_X I)^{-1} \Sigma_{XY} \ldots$ (adds $\lambda I$ to make invertible)
  - References: Vinod 1976, Leurgans et al. 1993
- **Sparse CCA (PMD — Penalized Matrix Decomposition):** Applies L1 penalties to $u$ and $v$ to yield sparse canonical vectors. Solves the invertibility problem + selects informative features.
  - Reference: Witten et al. 2009
- Example: 3000 audio features, 50 patients → $\Sigma_{XX}$ is singular. Sparse CCA solves this while zeroing out uninformative features.

---

## 5. Comparison of Methods

| Method | Objective | Supervised? | Two matrices? |
|--------|-----------|-------------|---------------|
| PCA | $\max \text{Var}(Xv)$ | No | No |
| Sparse PCA | $\max \text{Var}(Xv)$ + sparsity | No | No |
| PLS | $\max \text{Cov}(Xu, Yv)$ | Yes (y) | Optional |
| CCA | $\max \text{Corr}^2(Xu, Yv)$ | — | Yes (X and Y) |

- **PCA vs PLS:** PCA maximizes variance in X only; PLS maximizes covariance between X and y.
- **PLS vs CCA:** PLS balances variance and correlation; CCA ignores internal variance, focuses only on cross-correlation.
- **PCR vs PLS:** PCR first reduces X (may discard y-relevant dimensions); PLS uses y to guide reduction.

---

## 6. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| PCA objective | $\max_v \text{Var}(Xv)$ s.t. $\|v\|=1$ |
| PLS objective | $\max_{u,v} \text{Cov}(Xu, Yv)$ |
| PLS $m$-th direction | $\max_\alpha \text{Corr}^2(y, X\alpha)\cdot\text{Var}(X\alpha)$ s.t. $\|\alpha\|=1$ |
| CCA objective | $\max_{u,v} (u^T \Sigma_{XY} v)/\sqrt{u^T \Sigma_{XX} u \cdot v^T \Sigma_{YY} v}$ |
| CCA eigenvalue problem | $\Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX} u = \lambda^2 \Sigma_{XX} u$ |
| PLS weight step | $\hat{\phi}_{mj} = x_j^{(m-1)^T} y$ |
| PLS score | $z_m = \sum_j \hat{\phi}_{mj} x_j^{(m-1)}$ |
| PLS regression coeff | $\hat{\theta}_m = z_m^T y / z_m^T z_m$ |
| PLS deflation | $x_j^{(m)} = x_j^{(m-1)} - (z_m^T x_j^{(m-1)}/z_m^T z_m) z_m$ |
| PLS orthogonality | $z_i^T z_j = 0$ for $i \neq j$ |
| Regularized CCA | $(\Sigma_{XX} + \lambda I)^{-1} \Sigma_{XY} \ldots$ |
| Variance explained | $\lambda_k/\sum_j \lambda_j$ |
| SVD standard deviation | $\sigma_l = d_l/\sqrt{n-1}$ |
