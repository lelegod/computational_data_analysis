# Week 3 — Sparse Regression: Curse of Dimensionality, Lasso, Elastic Net, Variable Selection Instability

## Overview
Week 3 deepens the regularization framework. It begins with the curse (and blessings) of dimensionality, revisits the 1-SE rule for λ selection, recaps Ridge and introduces Lasso and Elastic Net in full detail. The second half covers variable selection instability (multiple testing: FWER, Bonferroni, FDR via Benjamini-Hochberg). This lecture is taught by Sneha Das (DTU Compute).

---

## Part I — Recap and Outstanding Questions

### 1-SE Rule (from Week 1/2 recap)
- After cross-validation, instead of choosing the λ with minimum CV error, choose the **largest λ whose CV error is within 1 standard error of the minimum**.
- Rationale (Breiman, Friedman, Olsen, Stone 1984 — CART monograph): "the 1-SE rule yields a stable tree/model size across replications, whereas the 0-SE (minimum) size can vary substantially across replications."
- Effect: Selects a **simpler, more regularized** model that is statistically indistinguishable from the best.

### Methods for Model Selection Recap
- **AIC**: grows slowly with N (more data → "I can afford more complexity for prediction").
- **BIC**: grows faster with N ("with more data, I can confidently detect whether extra parameters are truly warranted" — requires a penalty that increases with N).
- **CV**: often aligned with AIC (asymptotically equivalent to LOO-CV).
- On a plot: AIC and CV select similar λ; BIC selects a more regularized (larger λ) model.

---

## Part II — Curse of Dimensionality

### What Happens as Dimension Grows
- As dimension D increases, the number of regions in the solution space grows **exponentially** with D.
- A fixed number of training points N becomes exponentially sparse in high dimensions.

### Five Manifestations of the Curse
1. **Sparsity**: Data becomes incredibly sparse; "local" neighborhoods become empty (nearest-neighbor methods break down).
2. **Distances**: Euclidean distances lose meaning; all points become roughly equidistant.
3. **Overfitting**: With p > N, models can perfectly fit noise (degrees of freedom issues — more parameters than observations).
4. **Edge Effect**: Most data points reside at the boundaries (corners) of the sample space, not in the interior.
5. **Computational Cost**: Search algorithms slow down significantly.

### Blessings of Dimensionality (Donoho, 2000)
Not all bad — Donoho (2000) identified 3 blessings:
1. **Correlations**: Several features will be correlated → we can average over them (e.g., Ridge exploits this).
2. **Low-dimensional manifold**: Underlying distributions are often finite-dimensional; informative data lies on a low-dimensional manifold (PCA, latent variable models exploit this).
3. **Approximate finite dimensionality**: Underlying structure in data (samples from continuous processes, images, etc.) will give approximate finite dimensionality.

---

## Part III — Dimension Reduction Overview

### Approaches to Dimension Reduction
1. **Regularization of parameters**: Ridge, Lasso, Elastic Net (focus of weeks 1-3).
2. **Combinatorial search**: Forward/backward selection, all-subsets (recap with multiple hypothesis testing this week).
3. **Projection to lower dimensions (latent variables)**: PCA, unsupervised decomposition, multi-way models (later lectures).
4. **Clustering of features**: Group similar features (later lectures).
5. **Structuring parameter estimates**: Related to regularization.

---

## Part IV — Norms and Shrinkage Methods

### L₂ and L₁ Norms of β
- **L₂ norm squared**: `||β||²₂ = Σⱼ βⱼ²` (sum of squares)
- **L₁ norm**: `||β||₁ = Σⱼ |βⱼ|` (sum of absolute values)

### Three Shrinkage Methods
1. **Ridge regression**: Quadratic shrinkage, L₂ norm penalty.
2. **Lasso regression**: Absolute-value shrinkage, L₁ norm penalty.
3. **Elastic Net**: Hybrid method (L₁ + L₂).

Instead of controlling model complexity by setting a subset of coefficients to zero (variable selection), shrinkage methods shrink ALL coefficients toward zero continuously.

---

## Part V — Ridge Regression (Recap/Detail)

### Objective
- **Penalized form**: `min_β (Y - Xβ)^T(Y - Xβ) + λβ^Tβ`
- **Constrained form**: `min_β (Y - Xβ)^T(Y - Xβ)` subject to `Σⱼ βⱼ² ≤ s`

### Key Properties
- Increasing λ makes estimated β's smaller but **never exactly zero**.
- We typically do **not penalize the intercept β₀**.
- The contour plots: RSS ellipses (blue) intersect the L₂ sphere (red circles) at a point that is NOT on an axis.

### Regularization Path
- As λ increases from 0 to ∞, all β traces a smooth path from OLS solution to 0.
- The path is smooth (no sharp changes) — characteristic of L₂ penalty.

---

## Part VI — The Lasso (Full Detail)

### Objective
- **Penalized form**: `min_β (Y - Xβ)^T(Y - Xβ) + λ||β||₁`
- **Constrained form (basis pursuit)**: `min_β (Y - Xβ)^T(Y - Xβ)` subject to `Σⱼ|βⱼ| ≤ s`
- The L₂ penalty of Ridge is replaced by an **L₁ penalty**.

### Geometry (Why Lasso Produces Zeros)
- L₁ constraint region = **diamond** with corners on coordinate axes (in 2D).
- L₂ constraint region = **sphere** (no corners).
- RSS ellipses typically first touch the diamond at a **corner** → that coordinate is exactly 0.
- This is why Lasso performs **automatic variable selection**.

---

## Part VII — LARS Algorithm (Full Detail)

### Overview
- LARS = Least Angle Regression Selection.
- Efficiently computes the **entire regularization path** for Lasso (all λ values) at the cost of one OLS fit.
- Lasso modification of LARS: when a coefficient estimate crosses zero, set it to zero and recompute the equiangular direction.

### Step-by-Step with Example (2D Data)
**Setup**: y = [2, 1]^T, x₁ = [1, 0]^T, x₂ = [0.5, 0.866]^T

**Step 1: Initialization and First Correlation**
- Start: β = 0, μ₀ = 0, r = y
- Correlations: c = X^T r
  - c₁ = x₁^T y = 1(2) + 0(1) = **2.0** (maximum)
  - c₂ = x₂^T y = 0.5(2) + 0.866(1) = 1.866
- Action: start moving along x₁ (highest correlation).

**Step 2: Step Size Calculation**
- Move along x₁ until residual is equally correlated with x₂.
- Step size formula (2 variables): `γ = (cⱼ - cₖ) / (1 - ρⱼₖ)`
  - c₁ = 2.0, c₂ = 1.866
  - ρ₁₂ = x₁^T x₂ = 0.5
  - γ = (2.0 - 1.866) / (1 - 0.5) = 0.134 / 0.5 = **0.268**
- Move 0.268 units along x₁.

**Step 3: Update and Verify**
- Update: μ_new = 0 + 0.268·[1, 0]^T = [0.268, 0]^T
- New residual: r_new = [2, 1]^T - [0.268, 0]^T = [1.732, 1]^T
- Check correlations:
  - c₁ = 1(1.732) + 0(1) = **1.732**
  - c₂ = 0.5(1.732) + 0.866(1) ≈ **1.732**
- Equal correlations: x₂ now enters the model. Success!

**Step 4: Equiangular Direction**
- LARS moves along u_A — the equiangular direction.
- For 2 vectors: u is the normalized sum (bisector).
- Calculation: sum = x₁ + x₂ = [1+0.5, 0+0.866]^T = [1.5, 0.866]^T
- Norm: ||v|| = √(1.5² + 0.866²) = √3 ≈ 1.732
- u = (1/1.732)[1.5, 0.866]^T = [0.866, 0.5]^T

**General Matrix Formula for Equiangular Direction**:
```
u_A = X_A · w,    where w = A(X_A^T X_A)^{-1} · 1
```
- X_A^T X_A: the Gram (correlation) matrix of active set.
- A: normalization factor so ||u_A|| = 1.

### LARS Algorithm Summary
**Assumptions**: Data is centered and normalized (each variable has length 1), so X^T X ≈ Corr(X).

**Lasso modification**: If a parameter estimate crosses zero → set to zero and recompute direction. This gives a piecewise linear path for all λ values.

### LARS Cp Statistic (Model Selection)
`Cp = (1/σ̂²) Σᵢ(yᵢ - ŷᵢ)² - n + 2k`
- k = number of LARS steps.
- Choose k that minimizes Cp.

### Parameter Trace: LARS vs LASSO
- **Pure LARS** (9 iterations): Coefficients move together continuously; once included, never dropped.
- **LASSO** (14 iterations in example): One feature (Feature 0) is DROPPED at step 8 — it crosses zero and is removed.

---

## Part VIII — Cyclical Coordinate Descent (Full Detail)

### Setting
Fix λ. Solve `min_β (1/2n) Σᵢ(yᵢ - xᵢβ)² + λ|β|` by updating one coordinate at a time.

### Algorithm
1. Compute partial residual (all contributions except βⱼ):
   `r_i^(j) = yᵢ - Σ_{k≠j} x_{ik} β̃_k(λ)`
2. Compute OLS solution for this partial residual:
   `β̃_j^OLS = (1/n) Σᵢ x_{ij} r_i^(j)`
   (under standardization: Σᵢ xᵢⱼ = 0 and (1/n)Σᵢ xᵢⱼ² = 1)
3. Apply soft thresholding:
   `β̃_j(λ) = sign(β̃_j^OLS)(|β̃_j^OLS| - λ)₊`
4. Cycle through j = 1, ..., p until convergence.

### Soft Thresholding in Detail
`S(x, Δ) = sign(x)(|x| - Δ)₊`
- |x| ≤ Δ → 0 (coefficient zeroed out)
- |x| > Δ → x shrunk toward 0 by Δ
- This is the operation that produces sparsity in coordinate descent.
- The soft thresholding function looks like: zero in the middle, then 45° lines on each side (shifted inward by Δ).

---

## Part IX — The Elastic Net (Full Detail)

### Motivation: Lasso's Three Limitations
1. **High Dimensionality** (p > n): Lasso selects at most n variables.
2. **Grouping Effect**: With correlated variables, Lasso arbitrarily picks one.
3. **Predictive Power**: When n > p with high correlations, Ridge often outperforms Lasso.

Source: Zou and Hastie (2005), "Regularization and variable selection via the elastic net."

### Elastic Net Objective
- **Penalized form**: `min_β (1/2n)||Y - Xβ||²₂ + λ[(1/2)(1-α)||β||²₂ + α||β||₁]`
- **Constrained form**: `min_β (1/2n)||Y - Xβ||²₂` s.t. `(1/2)(1-α)||β||²₂ + α||β||₁ ≤ t`

### α Controls the Mix
- α = 1: Pure Lasso
- α = 0: Pure Ridge
- 0 < α < 1: Elastic Net ("elastic" region)

### Contour Plot (α = 0.5)
- The elastic net constraint is intermediate between Ridge (sphere) and Lasso (diamond).
- It has softened corners: tends to produce sparse solutions but with some grouping.

### Advantage
Combines:
- **Shrinkage** of Ridge (handles multicollinearity, keeps correlated variables)
- **Variable selection** of Lasso (sparse coefficients)
Result: robust sparse estimate.

### Implementation via Data Augmentation

**Step 1a — Construction**: Stack original data with m additional "Ridge rows":
```
X*_{(n+m)×m} = [X; √λ₂ · I_m],    y*_{(n+m)} = [y; 0_m]
```
- I_m is the m×m identity matrix.
- Bottom m rows of y* are zeros.
- Factor √λ₂ controls Ridge influence within X*.

**Step 1b — Absorption**: RSS on augmented data equals:
```
||y* - X*β||² = ||y - Xβ||² + ||0 - √λ₂·I·β||²
              = ||y - Xβ||² + λ₂||β||²₂
              = Original RSS + The Ridge Penalty!
```
The L₂ penalty is automatically embedded in the residual.

**Step 2 — Equivalent Lasso**: After absorption, solve:
```
min_{β*} ||y* - X*β*||² + λ₁||β*||₁
```
Can now use LARS or Coordinate Descent on (y*, X*). They are blind to the hybrid penalty.

**Alternative formulation** (scaled):
```
X* = (1+λ₂)^{-1/2} [X; √λ₂ · I_p],    y* = [y; 0_p]
```
OLS solution: (1/√(1+λ₂))(X^T X + λ₂ I_p^T I_p)β* = X^T y → scaled ridge solution.

### Elastic Net Example (Diabetes Dataset)
- OLS Full: non-zero coefficients for all features.
- Ridge (L₂): all coefficients shrunk, none zero.
- Lasso (L₁): sparse — e.g., "age" coefficient zeroed out.
- Elastic Net: compromise — sparse but less aggressive than Lasso on correlated features.

---

## Part X — Combinatorial Search, Forward, and Backward Selection

### Combinatorial (All-Subsets) Search
- Try all possible 2^p subsets of features.
- Pro: Guaranteed to find the best combination.
- Con: Exponential growth — infeasible for large p.

### Forward Selection
- Start with no variables. Add one at a time (variable with highest information criterion gain).
- Pro: Feasible O(p²) models; works when p > n.
- Con: May not find globally optimal combination.

### Backward Elimination
- Start with all variables. Remove one at a time (lowest information criterion loss).
- Pro: Feasible O(p²) models; usually better than forward.
- Con: Numerical issues when p is large; requires n > p initially.

---

## Part XI — Variable Selection Instability and Multiple Testing

### The Problem of Multiple Testing in Feature Assessment
- Traditional t-test: tests if estimated parameters are zero (tests for each feature independently).
- Traditional F-test: tests overall parameter significance.
- Testing p features independently at α: FWER grows rapidly.

### Family-Wise Error Rate (FWER)
- **Definition**: Probability of at least one false rejection across M independent tests.
- **Formula**: `FWER = 1 - (1 - α)^M`
- **Example** (jelly bean xkcd example): 20 colors tested at α = 0.05:
  - FWER = 1 - (0.95)^20 ≈ **0.64**
  - 64% chance of finding at least one "significant" result by chance!
  - "Green jelly beans linked to acne! 95% confidence." — but 1 of 20 passes by chance.

### Bonferroni Correction
- **Method**: Reject H_i if p-value < α/M.
- **Tradeoff**: Controls FWER at α but has **low power** (we miss many true effects).

### False Discovery Rate (FDR)
- **Definition**: `FDR = E[FP / (FP + TP)]`
  - FP = false positives (false discoveries)
  - TP = true positives (true discoveries)
- FDR controls the **expected proportion** of false discoveries among all discoveries.
- Set FDR threshold q: we accept that up to fraction q of our findings may be mistakes.
- Proposed by Benjamini and Hochberg (1995).
- Gain: More power than Bonferroni (we detect more true effects).
- Cost: Increased false negatives if q is too loose.

### Benjamini-Hochberg (BH) Algorithm
Given m tests with null hypotheses H₁, ..., H_m and p-values p₁, ..., p_m:
1. Sort p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
2. For chosen q, find: `k = max{i : p_(i) ≤ (i/m)q}`
3. Reject H_(1), H_(2), ..., H_(k) (all up to and including k).

**Intuition**: Walk down sorted p-values. Reject while `(i/m)q` is greater than or equal to p_(i). Stop when it fails — do NOT continue after a failure. The threshold (i/m)q increases linearly.

**q is your choice** of acceptable fraction of mistakes. Typically α = 0.05 for single tests, but q = 0.1 or 0.2 for FDR (different type of error control).

### BH Example 1 (m=5, q=0.1, p-values: 0.01, 0.05, 0.1, 0.4, 0.6)
- i=1: 0.01 ≤ (1/5)(0.1) = 0.02 → ✓ (pass)
- i=2: 0.05 ≤ (2/5)(0.1) = 0.04 → ✗ (fail — stop)
- k=1, reject only H_(1).

### BH Example 2 (m=5, q=0.20, p-values: 0.01, 0.03, 0.15, 0.40, 0.50)
Thresholds: (1/5)×0.20=0.04, (2/5)×0.20=0.08, (3/5)×0.20=0.12, (4/5)×0.20=0.16, (5/5)×0.20=0.20
- i=1: 0.01 ≤ 0.04 → ✓
- i=2: 0.03 ≤ 0.08 → ✓
- i=3: 0.15 ≤ 0.12 → ✗ (fail — stop)
- k=2, reject H_(1) and H_(2). **Answer B** (first and second).

### Comparison: Bonferroni vs BH
| | Bonferroni | BH (FDR) |
|--|--|--|
| Controls | FWER (any false positive) | FDR (proportion of false positives) |
| Power | Low | Higher |
| Threshold | α/M (fixed, very small) | (i/m)q (adaptive, increases with rank) |
| Use when | Cannot afford any false positive | Can tolerate a few, want more discoveries |
