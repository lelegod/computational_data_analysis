# Week 3 — Sparse Regression: Curse of Dimensionality, Lasso, Elastic Net, Variable Selection Instability

## Overview
Week 3 deepens the regularization framework. It begins with the curse (and blessings) of dimensionality, revisits the 1-SE rule for $\lambda$ selection, recaps Ridge and introduces Lasso and Elastic Net in full detail. The second half covers variable selection instability (multiple testing: FWER, Bonferroni, FDR via Benjamini-Hochberg). This lecture is taught by Sneha Das (DTU Compute).

---

## Part I — Recap and Outstanding Questions

### 1-SE Rule (from Week 1/2 recap)
- After cross-validation, instead of choosing the $\lambda$ with minimum CV error, choose the **largest $\lambda$ whose CV error is within 1 standard error of the minimum**.
- Rationale (Breiman, Friedman, Olsen, Stone 1984 — CART monograph): "the 1-SE rule yields a stable tree/model size across replications, whereas the 0-SE (minimum) size can vary substantially across replications."
- Effect: Selects a **simpler, more regularized** model that is statistically indistinguishable from the best.

### Methods for Model Selection Recap
- **AIC**: grows slowly with $N$ (more data → "I can afford more complexity for prediction").
- **BIC**: grows faster with $N$ ("with more data, I can confidently detect whether extra parameters are truly warranted" — requires a penalty that increases with $N$).
- **CV**: often aligned with AIC (asymptotically equivalent to LOO-CV).
- On a plot: AIC and CV select similar $\lambda$; BIC selects a more regularized (larger $\lambda$) model.

---

## Part II — Curse of Dimensionality

### What Happens as Dimension Grows
- As dimension $D$ increases, the number of regions in the solution space grows **exponentially** with $D$.
- A fixed number of training points $N$ becomes exponentially sparse in high dimensions.

### Five Manifestations of the Curse
1. **Sparsity**: Data becomes incredibly sparse; "local" neighborhoods become empty (nearest-neighbor methods break down).
2. **Distances**: Euclidean distances lose meaning; all points become roughly equidistant.
3. **Overfitting**: With $p > N$, models can perfectly fit noise (degrees of freedom issues — more parameters than observations).
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

### $L_2$ and $L_1$ Norms of $\boldsymbol{\beta}$
- **$L_2$ norm squared**: $\|\boldsymbol{\beta}\|_2^2 = \sum_j \beta_j^2$ (sum of squares)
- **$L_1$ norm**: $\|\boldsymbol{\beta}\|_1 = \sum_j |\beta_j|$ (sum of absolute values)

### Three Shrinkage Methods
1. **Ridge regression**: Quadratic shrinkage, $L_2$ norm penalty.
2. **Lasso regression**: Absolute-value shrinkage, $L_1$ norm penalty.
3. **Elastic Net**: Hybrid method ($L_1 + L_2$).

Instead of controlling model complexity by setting a subset of coefficients to zero (variable selection), shrinkage methods shrink ALL coefficients toward zero continuously.

---

## Part V — Ridge Regression (Recap/Detail)

### Objective
- **Penalized form**: $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) + \lambda\boldsymbol{\beta}^T\boldsymbol{\beta}$
- **Constrained form**: $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$ subject to $\sum_j \beta_j^2 \leq s$

### Key Properties
- Increasing $\lambda$ makes estimated $\beta$'s smaller but **never exactly zero**.
- We typically do **not penalize the intercept $\beta_0$**.
- The contour plots: RSS ellipses (blue) intersect the $L_2$ sphere (red circles) at a point that is NOT on an axis.

### Regularization Path
- As $\lambda$ increases from $0$ to $\infty$, all $\hat{\beta}$ traces a smooth path from OLS solution to $0$.
- The path is smooth (no sharp changes) — characteristic of $L_2$ penalty.

---

## Part VI — The Lasso (Full Detail)

### Objective
- **Penalized form**: $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) + \lambda\|\boldsymbol{\beta}\|_1$
- **Constrained form (basis pursuit)**: $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$ subject to $\sum_j |\beta_j| \leq s$
- The $L_2$ penalty of Ridge is replaced by an **$L_1$ penalty**.

### Geometry (Why Lasso Produces Zeros)
- $L_1$ constraint region = **diamond** with corners on coordinate axes (in 2D).
- $L_2$ constraint region = **sphere** (no corners).
- RSS ellipses typically first touch the diamond at a **corner** → that coordinate is exactly $0$.
- This is why Lasso performs **automatic variable selection**.

---

## Part VII — LARS Algorithm (Full Detail)

### Overview
- LARS = Least Angle Regression Selection.
- Efficiently computes the **entire regularization path** for Lasso (all $\lambda$ values) at the cost of one OLS fit.
- Lasso modification of LARS: when a coefficient estimate crosses zero, set it to zero and recompute the equiangular direction.

### Step-by-Step with Example (2D Data)
**Setup**: $\mathbf{y} = [2,\ 1]^T$, $x_1 = [1,\ 0]^T$, $x_2 = [0.5,\ 0.866]^T$

**Step 1: Initialization and First Correlation**
- Start: $\boldsymbol{\beta} = \mathbf{0}$, $\mu_0 = 0$, $r = y$
- Correlations: $c = \mathbf{X}^T r$
  - $c_1 = x_1^T y = 1(2) + 0(1) = \mathbf{2.0}$ (maximum)
  - $c_2 = x_2^T y = 0.5(2) + 0.866(1) = 1.866$
- Action: start moving along $x_1$ (highest correlation).

**Step 2: Step Size Calculation**
- Move along $x_1$ until residual is equally correlated with $x_2$.
- Step size formula (2 variables): $\gamma = \dfrac{c_j - c_k}{1 - \rho_{jk}}$
  - $c_1 = 2.0$, $c_2 = 1.866$
  - $\rho_{12} = x_1^T x_2 = 0.5$
  - $\gamma = \dfrac{2.0 - 1.866}{1 - 0.5} = \dfrac{0.134}{0.5} = \mathbf{0.268}$
- Move $0.268$ units along $x_1$.

**Step 3: Update and Verify**
- Update: $\mu_{\text{new}} = \mathbf{0} + 0.268\cdot[1,\ 0]^T = [0.268,\ 0]^T$
- New residual: $r_{\text{new}} = [2,\ 1]^T - [0.268,\ 0]^T = [1.732,\ 1]^T$
- Check correlations:
  - $c_1 = 1(1.732) + 0(1) = \mathbf{1.732}$
  - $c_2 = 0.5(1.732) + 0.866(1) \approx \mathbf{1.732}$
- Equal correlations: $x_2$ now enters the model. Success!

**Step 4: Equiangular Direction**
- LARS moves along $u_A$ — the equiangular direction.
- For 2 vectors: $u$ is the normalized sum (bisector).
- Calculation: $\text{sum} = x_1 + x_2 = [1+0.5,\ 0+0.866]^T = [1.5,\ 0.866]^T$
- Norm: $\|v\| = \sqrt{1.5^2 + 0.866^2} = \sqrt{3} \approx 1.732$
- $u = \dfrac{1}{1.732}[1.5,\ 0.866]^T = [0.866,\ 0.5]^T$

**General Matrix Formula for Equiangular Direction**:

$$u_A = \mathbf{X}_A \cdot w, \qquad w = A(\mathbf{X}_A^T \mathbf{X}_A)^{-1} \cdot \mathbf{1}$$

- $\mathbf{X}_A^T \mathbf{X}_A$: the Gram (correlation) matrix of active set.
- $A$: normalization factor so $\|u_A\| = 1$.

### LARS Algorithm Summary
**Assumptions**: Data is centered and normalized (each variable has length 1), so $\mathbf{X}^T \mathbf{X} \approx \text{Corr}(\mathbf{X})$.

**Lasso modification**: If a parameter estimate crosses zero → set to zero and recompute direction. This gives a piecewise linear path for all $\lambda$ values.

### LARS Cp Statistic (Model Selection)
$$C_p = \frac{1}{\hat{\sigma}^2} \sum_i (y_i - \hat{y}_i)^2 - n + 2k$$
- $k$ = number of LARS steps.
- Choose $k$ that minimizes $C_p$.

### Parameter Trace: LARS vs LASSO
- **Pure LARS** (9 iterations): Coefficients move together continuously; once included, never dropped.
- **LASSO** (14 iterations in example): One feature (Feature 0) is DROPPED at step 8 — it crosses zero and is removed.

---

## Part VIII — Cyclical Coordinate Descent (Full Detail)

### Setting
Fix $\lambda$. Solve $\min_{\boldsymbol{\beta}}\ \frac{1}{2n} \sum_i (y_i - x_i^T\boldsymbol{\beta})^2 + \lambda|\beta_j|$ by updating one coordinate at a time.

### Algorithm
1. Compute partial residual (all contributions except $\beta_j$):
   $$r_i^{(j)} = y_i - \sum_{k \neq j} x_{ik} \tilde{\beta}_k(\lambda)$$
2. Compute OLS solution for this partial residual:
   $$\tilde{\beta}_j^{\text{OLS}} = \frac{1}{n} \sum_i x_{ij} r_i^{(j)}$$
   (under standardization: $\sum_i x_{ij} = 0$ and $\frac{1}{n}\sum_i x_{ij}^2 = 1$)
3. Apply soft thresholding:
   $$\tilde{\beta}_j(\lambda) = \text{sign}(\tilde{\beta}_j^{\text{OLS}})\left(|\tilde{\beta}_j^{\text{OLS}}| - \lambda\right)_+$$
4. Cycle through $j = 1, \ldots, p$ until convergence.

### Soft Thresholding in Detail
$$S(x, \Delta) = \text{sign}(x)(|x| - \Delta)_+$$
- $|x| \leq \Delta$ → $0$ (coefficient zeroed out)
- $|x| > \Delta$ → $x$ shrunk toward $0$ by $\Delta$
- This is the operation that produces sparsity in coordinate descent.
- The soft thresholding function looks like: zero in the middle, then 45° lines on each side (shifted inward by $\Delta$).

---

## Part IX — The Elastic Net (Full Detail)

### Motivation: Lasso's Three Limitations
1. **High Dimensionality** ($p > n$): Lasso selects at most $n$ variables.
2. **Grouping Effect**: With correlated variables, Lasso arbitrarily picks one.
3. **Predictive Power**: When $n > p$ with high correlations, Ridge often outperforms Lasso.

Source: Zou and Hastie (2005), "Regularization and variable selection via the elastic net."

### Elastic Net Objective
- **Penalized form**: $\min_{\boldsymbol{\beta}}\ \frac{1}{2n}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\!\left[\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1\right]$
- **Constrained form**: $\min_{\boldsymbol{\beta}}\ \frac{1}{2n}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$ s.t. $\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1 \leq t$

### $\alpha$ Controls the Mix
- $\alpha = 1$: Pure Lasso
- $\alpha = 0$: Pure Ridge
- $0 < \alpha < 1$: Elastic Net ("elastic" region)

### Contour Plot ($\alpha = 0.5$)
- The elastic net constraint is intermediate between Ridge (sphere) and Lasso (diamond).
- It has softened corners: tends to produce sparse solutions but with some grouping.

### Advantage
Combines:
- **Shrinkage** of Ridge (handles multicollinearity, keeps correlated variables)
- **Variable selection** of Lasso (sparse coefficients)
Result: robust sparse estimate.

### Implementation via Data Augmentation

**Step 1a — Construction**: Stack original data with $m$ additional "Ridge rows":

$$\mathbf{X}^*_{(n+m)\times m} = \begin{bmatrix} \mathbf{X} \\ \sqrt{\lambda_2} \cdot \mathbf{I}_m \end{bmatrix}, \qquad \mathbf{y}^*_{(n+m)} = \begin{bmatrix} \mathbf{y} \\ \mathbf{0}_m \end{bmatrix}$$

- $\mathbf{I}_m$ is the $m \times m$ identity matrix.
- Bottom $m$ rows of $\mathbf{y}^*$ are zeros.
- Factor $\sqrt{\lambda_2}$ controls Ridge influence within $\mathbf{X}^*$.

**Step 1b — Absorption**: RSS on augmented data equals:

$$\|\mathbf{y}^* - \mathbf{X}^*\boldsymbol{\beta}\|^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \|\mathbf{0} - \sqrt{\lambda_2}\cdot\mathbf{I}\cdot\boldsymbol{\beta}\|^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_2\|\boldsymbol{\beta}\|_2^2$$

The $L_2$ penalty is automatically embedded in the residual.

**Step 2 — Equivalent Lasso**: After absorption, solve:

$$\min_{\boldsymbol{\beta}^*}\ \|\mathbf{y}^* - \mathbf{X}^*\boldsymbol{\beta}^*\|^2 + \lambda_1\|\boldsymbol{\beta}^*\|_1$$

Can now use LARS or Coordinate Descent on $(\mathbf{y}^*, \mathbf{X}^*)$. They are blind to the hybrid penalty.

**Alternative formulation** (scaled):

$$\mathbf{X}^* = (1+\lambda_2)^{-1/2} \begin{bmatrix} \mathbf{X} \\ \sqrt{\lambda_2} \cdot \mathbf{I}_p \end{bmatrix}, \qquad \mathbf{y}^* = \begin{bmatrix} \mathbf{y} \\ \mathbf{0}_p \end{bmatrix}$$

OLS solution: $\frac{1}{\sqrt{1+\lambda_2}}(\mathbf{X}^T \mathbf{X} + \lambda_2 \mathbf{I}_p)\boldsymbol{\beta}^* = \mathbf{X}^T \mathbf{y}$ → scaled ridge solution.

### Elastic Net Example (Diabetes Dataset)
- OLS Full: non-zero coefficients for all features.
- Ridge ($L_2$): all coefficients shrunk, none zero.
- Lasso ($L_1$): sparse — e.g., "age" coefficient zeroed out.
- Elastic Net: compromise — sparse but less aggressive than Lasso on correlated features.

---

## Part X — Combinatorial Search, Forward, and Backward Selection

### Combinatorial (All-Subsets) Search
- Try all possible $2^p$ subsets of features.
- Pro: Guaranteed to find the best combination.
- Con: Exponential growth — infeasible for large $p$.

### Forward Selection
- Start with no variables. Add one at a time (variable with highest information criterion gain).
- Pro: Feasible $O(p^2)$ models; works when $p > n$.
- Con: May not find globally optimal combination.

### Backward Elimination
- Start with all variables. Remove one at a time (lowest information criterion loss).
- Pro: Feasible $O(p^2)$ models; usually better than forward.
- Con: Numerical issues when $p$ is large; requires $n > p$ initially.

---

## Part XI — Variable Selection Instability and Multiple Testing

### The Problem of Multiple Testing in Feature Assessment
- Traditional $t$-test: tests if estimated parameters are zero (tests for each feature independently).
- Traditional $F$-test: tests overall parameter significance.
- Testing $p$ features independently at $\alpha$: FWER grows rapidly.

### Family-Wise Error Rate (FWER)
- **Definition**: Probability of at least one false rejection across $M$ independent tests.
- **Formula**: $\text{FWER} = 1 - (1 - \alpha)^M$
- **Example** (jelly bean xkcd example): 20 colors tested at $\alpha = 0.05$:
  - $\text{FWER} = 1 - (0.95)^{20} \approx \mathbf{0.64}$
  - 64% chance of finding at least one "significant" result by chance!
  - "Green jelly beans linked to acne! 95% confidence." — but 1 of 20 passes by chance.

### Bonferroni Correction
- **Method**: Reject $H_i$ if p-value $< \alpha/M$.
- **Tradeoff**: Controls FWER at $\alpha$ but has **low power** (we miss many true effects).

### False Discovery Rate (FDR)
- **Definition**: $\text{FDR} = E\!\left[\frac{\text{FP}}{\text{FP} + \text{TP}}\right]$
  - FP = false positives (false discoveries)
  - TP = true positives (true discoveries)
- FDR controls the **expected proportion** of false discoveries among all discoveries.
- Set FDR threshold $q$: we accept that up to fraction $q$ of our findings may be mistakes.
- Proposed by Benjamini and Hochberg (1995).
- Gain: More power than Bonferroni (we detect more true effects).
- Cost: Increased false negatives if $q$ is too loose.

### Benjamini-Hochberg (BH) Algorithm
Given $m$ tests with null hypotheses $H_1, \ldots, H_m$ and p-values $p_1, \ldots, p_m$:
1. Sort p-values ascending: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. For chosen $q$, find: $k = \max\!\left\{i : p_{(i)} \leq \frac{i}{m}q\right\}$
3. Reject $H_{(1)}, H_{(2)}, \ldots, H_{(k)}$ (all up to and including $k$).

**Intuition**: Walk down sorted p-values. Reject while $\frac{i}{m}q$ is greater than or equal to $p_{(i)}$. Stop when it fails — do NOT continue after a failure. The threshold $\frac{i}{m}q$ increases linearly.

**$q$ is your choice** of acceptable fraction of mistakes. Typically $\alpha = 0.05$ for single tests, but $q = 0.1$ or $0.2$ for FDR (different type of error control).

### BH Example 1 ($m=5$, $q=0.1$, p-values: 0.01, 0.05, 0.1, 0.4, 0.6)
- $i=1$: $0.01 \leq \frac{1}{5}(0.1) = 0.02$ → ✓ (pass)
- $i=2$: $0.05 \leq \frac{2}{5}(0.1) = 0.04$ → ✗ (fail — stop)
- $k=1$, reject only $H_{(1)}$.

### BH Example 2 ($m=5$, $q=0.20$, p-values: 0.01, 0.03, 0.15, 0.40, 0.50)
Thresholds: $\frac{1}{5}\times 0.20=0.04$, $\frac{2}{5}\times 0.20=0.08$, $\frac{3}{5}\times 0.20=0.12$, $\frac{4}{5}\times 0.20=0.16$, $\frac{5}{5}\times 0.20=0.20$
- $i=1$: $0.01 \leq 0.04$ → ✓
- $i=2$: $0.03 \leq 0.08$ → ✓
- $i=3$: $0.15 \leq 0.12$ → ✗ (fail — stop)
- $k=2$, reject $H_{(1)}$ and $H_{(2)}$. **Answer B** (first and second).

### Comparison: Bonferroni vs BH
| | Bonferroni | BH (FDR) |
|--|--|--|
| Controls | FWER (any false positive) | FDR (proportion of false positives) |
| Power | Low | Higher |
| Threshold | $\alpha/M$ (fixed, very small) | $\frac{i}{m}q$ (adaptive, increases with rank) |
| Use when | Cannot afford any false positive | Can tolerate a few, want more discoveries |
