# Week 2 — Lasso, Elastic Net, Model Assessment, Bootstrap, Multiple Testing

## Overview
Week 2 extends week 1's regularization framework. It introduces Lasso (L₁ penalty) which performs variable selection by setting some coefficients to exactly zero. Two algorithms for solving the Lasso are covered: LARS and Cyclical Coordinate Descent. The elastic net combines L₁ and L₂ penalties. Model assessment topics include nested cross-validation, the bootstrap, classifier performance metrics (confusion matrix, ROC), and multiple testing correction (FWER, Bonferroni, FDR / Benjamini-Hochberg).

---

## Part I — The Lasso

### Key Concepts
- Lasso = "Least Absolute Shrinkage and Selection Operator"
- Uses an L₁ penalty instead of ridge's L₂ penalty.
- Critical difference: L₁ penalty produces **exact zeros** in the coefficient vector → automatic variable selection.

### Lasso Objective Function
- **Penalized form**: `min_β (Y - Xβ)^T(Y - Xβ) + λ||β||₁`
  - `||β||₁ = Σ|βⱼ|` — sum of absolute values
- **Constrained form** (basis pursuit): `min_β (Y - Xβ)^T(Y - Xβ)` subject to `Σ|β| ≤ s`
- λ (or equivalently s) controls sparsity; larger λ → more zeros.

### Why Lasso Produces Zeros (Geometry)
- The L₁ constraint region is a **diamond** (in 2D) with corners on the axes.
- The RSS ellipsoid typically first touches the diamond at a **corner**, where one or more βⱼ = 0.
- Contrast with ridge: L₂ constraint is a **sphere** with no corners → solutions are never exactly zero.

### Lasso Properties
- **Non-differentiable** at β = 0 (the L₁ norm has a kink there).
- **No closed-form solution** — must use iterative algorithms.
- For large enough λ: some βⱼ will be set to **exactly zero**.
- Effective number of parameters df = **number of non-zero coefficients** (coefficients different from zero).
- In the p > n case: Lasso selects **at most n variables**.
- If predictors are correlated, Lasso tends to pick **one** from a correlated group (arbitrarily).

### Lasso Limitations (motivating Elastic Net)
1. **High dimensionality** (p > n): Lasso selects at most n variables.
2. **Grouping effect**: With correlated predictors, Lasso picks one arbitrarily — Ridge tends to include the whole group.
3. **Predictive power**: When n > p and predictors are highly correlated, Ridge often outperforms Lasso.

---

## Part II — Algorithms for Lasso

### Algorithm 1: Least Angle Regression Selection (LARS)

#### Overview
- LARS is the computational "engine" for finding the entire Lasso/Elastic Net regularization path.
- Computes all λ values at the **speed of a single OLS fit**.
- LASSO is a **modification** of LARS: if a parameter estimate crosses zero, set it to zero and recompute direction.

#### LARS Algorithm Steps
1. **Initialize**: Start with all β = 0, current estimate μ₀ = 0, residual r = y.
2. **Find most correlated variable**: Compute correlations c = X^T r. Find xⱼ with max |correlation|.
3. **Move** βⱼ in the direction of its least-squares coefficient.
4. **Stop** when another variable xₖ has as much correlation with the current residual as xⱼ.
5. **Move in equiangular direction**: Move in the direction that bisects the angle between xⱼ and xₖ.
6. Repeat until all variables are included or residuals are zero.

#### Key Terminology
- μ: current prediction estimate
- Active set A: set of variables currently being moved
- Equiangular direction u_A = X_A w, where w = A(X_A^T X_A)^{-1} **1**, A is a normalization factor so ||u_A|| = 1
- Step size γ: chosen so the residual becomes equally correlated with all active variables

#### LARS Step Size Formula (2 variables)
`γ = (cⱼ - cₖ) / (1 - ρⱼₖ)`
- cⱼ, cₖ: current correlations of xⱼ and xₖ with residual
- ρⱼₖ: correlation between xⱼ and xₖ

#### LARS vs Forward Selection (Greedy vs Polite)
| Forward Selection (Greedy) | LARS (Polite) |
|---------------------------|---------------|
| Finds best variable | Finds best variable |
| Moves along it completely until it can't improve | Moves only until a second becomes equally helpful |
| Aggressive, jerky path | Efficient, equiangular path |

#### LARS Assumptions
- Data is centered and normalized (each variable has length 1).
- This means X^T X ≈ Corr(X).

#### Cp for LARS (choosing number of steps)
`Cp = (1/σ̂²) Σ(yᵢ - ŷᵢ)² - n + 2k`
where k is the number of LARS steps taken.

### Algorithm 2: Cyclical Coordinate Descent

#### Overview
Fix λ and solve `min_β (1/2n) Σ(yᵢ - xᵢβ)² + λ|β|` iteratively, updating one coordinate at a time.

#### Steps
1. Compute partial residual for coordinate j (holding all others fixed):
   `r_i^(j) = yᵢ - Σ_{k≠j} x_{ik} β̃_k(λ)`
2. Compute the OLS solution for this partial residual:
   `β̃_j^OLS = (1/n) Σᵢ x_{ij} r_i^(j)` (under standardization: Σᵢ xᵢⱼ = 0 and (1/n)Σᵢ xᵢⱼ² = 1)
3. Apply **soft thresholding** to get the Lasso update:
   `β̃_j(λ) = sign(β̃_j^OLS)(|β̃_j^OLS| - λ)₊`
4. Cycle through j = 1, ..., p repeatedly until convergence.

#### Soft Thresholding
`S(x, Δ) = sign(x)(|x| - Δ)₊`
- If |x| ≤ Δ: result is 0 (shrinks to zero).
- If |x| > Δ: result is x - Δ·sign(x) (shrinks toward zero by Δ).
- This is the key operation that produces sparsity.

---

## Part III — Elastic Net

### Motivation
Combines L₁ (Lasso) and L₂ (Ridge) penalties to get sparsity AND grouping behavior.

### Elastic Net Objective
- **Penalized form**: `min_β (1/2n)||Y - Xβ||²₂ + λ[(1/2)(1-α)||β||²₂ + α||β||₁]`
- **Constrained form**: `min_β (1/2n)||Y - Xβ||²₂` s.t. `(1/2)(1-α)||β||²₂ + α||β||₁ ≤ t`

### α Parameter (mixing)
- α = 1: Pure Lasso (L₁ only)
- α = 0: Pure Ridge (L₂ only)
- 0 < α < 1: Elastic Net (the "elastic" region)

### Advantage
- Combines shrinkage of Ridge with variable selection of Lasso.
- Handles the grouping effect: tends to include or exclude correlated variables together.
- More robust sparse estimate than Lasso alone.

### Implementation via Augmented Data
To use standard LARS/Lasso solvers for Elastic Net, "hide" the L₂ penalty inside the data:

**Step 1 — Construct augmented matrices**:
```
X*_{(n+m)×m} = [X; √λ₂ I_m],    y*_{(n+m)} = [y; 0_m]
```
- m = number of features, I_m is the m×m identity matrix.
- Bottom m rows of y* are zeros.

**Step 2 — Absorption**:
```
||y* - X*β||² = ||y - Xβ||² + ||0 - √λ₂ I β||²
              = ||y - Xβ||²  +  λ₂||β||²₂
```
The L₂ penalty is now absorbed into the residual term.

**Step 3 — Solve the LASSO** on (y*, X*):
`min_{β*} ||y* - X*β*||² + λ₁||β*||₁`
The solution β* is a **scaled ridge solution**: (1/√(1+λ₂)) is the scaling factor.

**Why**: LARS/Coordinate Descent only see a standard LASSO problem; they are blind to the hybrid penalty.

---

## Part IV — Feature / Variable Selection Methods

### Combinatorial Search
- Try all possible subsets of features; select optimal.
- Pro: Guaranteed to find best combination.
- Con: 2^p combinations — computationally infeasible for large p.

### Forward Selection
- Start with no variables; add one at a time (highest information criterion gain).
- Pro: O(p²) models; works when p > n.
- Con: May not find the globally optimal combination.

### Backward Elimination
- Start with all variables; remove one at a time (lowest information criterion loss).
- Pro: O(p²) models.
- Con: Numerical issues with many features; requires n > p to start.
- Usually better than forward selection.

---

## Part V — Model Assessment

### The Selection-Induced Bias Problem
- When you test many λ values and pick the one with **minimum CV error**, the minimum is **optimistically biased**.
- Reason: You have "spent" the independence of the validation folds by using them to select λ.
- The resulting error estimate is **not** an unbiased estimate of future performance.
- Insight: "We didn't just fit the model; we fitted the hyperparameter to the noise in the CV folds."

### Nested Cross-Validation (Solution)
Separate **model selection** from **model assessment** using two loops:

**Inner loop (Selection)**:
- Used to tune λ.
- Finds the best configuration for a specific training set.

**Outer loop (Assessment)**:
- Used to audit the **entire procedure** (including the selection step).
- Estimates how well the "Selection + Training" pipeline generalizes.

**Nested CV Algorithm**:
1. Split data into K_outer folds.
2. For each outer fold j (test):
   a. Take remaining data as "Training Set."
   b. Inner loop: Perform K_inner-fold CV on training set to find best λ*.
   c. Train final model with λ* on the **entire** training set.
   d. Evaluate on held-out outer fold j.
3. Final report: Average the K_outer test scores.

**Computational cost**: Total fits = K_outer × (K_inner × N_lambdas + 1)
- Example: 10 × 10 × 100 = 10,000 model fits.

**Key insight**: Nested CV audits the **methodology** (the whole pipeline), not a specific single model.
- It's OK if the best λ changes across outer folds.
- A large gap between inner error (5%) and outer error (12%) indicates selection-induced overfitting.

---

## Part VI — The Bootstrap

### What is Bootstrap?
- A general method for **assessing statistical accuracy** (standard errors, confidence intervals, bias).
- Invented by Efron. Key idea: use the data itself as a "mirror copy of the real world."
- Bootstrap estimates ≈ Monte Carlo estimates (but drawing from empirical distribution instead of true P).

### Conceptual Framework (Freedman's terminology)
- **Real world**: Unknown P → observed data x = (x₁, ..., xₙ) → statistic θ̂ = s(x)
- **Bootstrap world**: Estimated P̂ → bootstrap sample x* = (x₁*, ..., xₙ*) → bootstrap replication θ̂* = s(x*)

### Bootstrap Method
1. Given training set Z = (z₁, ..., z_N) where zᵢ = (xᵢ, yᵢ).
2. Randomly draw with replacement from Z, same size N → bootstrap sample Z*b.
3. Repeat B times (B = 100 or more), producing B bootstrap datasets.
4. Refit the model to each Z*b, compute statistic S(Z*b).
5. Variance estimate: `Var̂[S(Z)] = (1/(B-1)) Σᵦ (S(Z*b) - S̄*)²`
   where `S̄* = (Σᵦ S(Z*b)) / B`

### Practical Remarks
- For **standard deviation**: a few hundred replicates suffice.
- For **confidence intervals**: 1000–2000 replicates recommended.
- Try different B and check if results change.
- Bootstrap works well for statistics "in the middle" of the distribution; works **poorly for tail statistics** (extremes).
- **Tibshirani's warning**: Do NOT use bootstrap for model selection — it is intended for assessing statistical accuracy of a given model/statistic.

---

## Part VII — Classifier Performance

### Confusion Matrix
For a binary classifier (Positive/Negative):
| | Predicted Positive | Predicted Negative |
|--|--|--|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

**Derived metrics**:
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) — fraction correct. Dangerous for imbalanced data.
- **Sensitivity / Recall / TPR** = TP / (TP + FN) — fraction of actual positives detected.
- **Specificity / TNR** = TN / (TN + FP) — fraction of actual negatives correctly identified.
- **Precision / PPV** = TP / (TP + FP) — fraction of predicted positives that are true positives.
- **FPR (False Positive Rate)** = FP / (FP + TN) = 1 - Specificity
- **FNR (False Negative Rate)** = FN / (FN + TP) = 1 - Sensitivity
- **F1 score** = 2·TP / (2·TP + FP + FN) = harmonic mean of precision and recall
- **Balanced accuracy** = (Sensitivity + Specificity) / 2

**Note**: If prevalence is low (e.g., 0.1%), ignore accuracy — use Precision-Recall curves instead.

### ROC Curve — Receiver Operating Characteristics
- Plots **TPR (Sensitivity)** vs **FPR (1-Specificity)** as the classification threshold varies.
- **AUC-ROC**: Area under the ROC curve.
  - AUC = 1.0: perfect classifier.
  - AUC = 0.5: random classifier (diagonal line).
  - AUC > 0.5: better than random.
- ROC/AUC gives general performance across ALL classification thresholds.
- Can be extended to multiclass: micro-average and macro-average.

### Regression Performance Metrics
- **MSE/RMSE**: `(1/n)Σ(yᵢ - ŷᵢ)²` — outlier sensitive; useful for safety-critical audits.
- **MAE**: `(1/n)Σ|yᵢ - ŷᵢ|` — robust to outliers; direct physical interpretation.
- **R²**: Fraction of variance explained. Relative, not absolute measure.
- **Residual plots**: Final sanity check; if residuals show patterns, model is incomplete regardless of MSE.

---

## Part VIII — Multiple Testing

### The Problem
- Testing one hypothesis at significance level α: probability of false rejection = α.
- Testing M hypotheses: the probability of **at least one false rejection** is much larger than α.

### Family-Wise Error Rate (FWER)
- **Definition**: Probability of at least one false rejection across all M tests.
- **Formula** (independent tests): `FWER = 1 - (1 - α)^M`
- Example: M=20 tests at α=0.05 → FWER = 1 - (0.95)²⁰ ≈ 0.64 (64% chance of at least one false discovery!)

### Bonferroni Correction
- **Method**: Reject hypothesis if p-value < α/M.
- **Effect**: Controls FWER at level α (assuming independence).
- **Cost**: Low power — we miss many true effects.

### False Discovery Rate (FDR)
- **Definition**: `FDR = E[FP / (FP + TP)]`
  - FP = false positives (false discoveries)
  - TP = true positives (true discoveries)
- **Trade-off**: Allows a controlled fraction of false discoveries → more power than Bonferroni.
- Proposed by Benjamini and Hochberg (1995).
- Set FDR threshold q: among all findings, we expect at most fraction q to be mistakes.

### Benjamini-Hochberg (BH) Algorithm for FDR
Given m hypothesis tests with p-values p₁, ..., p_m and target FDR level q:
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
2. Find the largest k such that: `p_(k) ≤ (k/m) · q`
3. Reject all hypotheses H_(1), H_(2), ..., H_(k)

**Intuition**: Walk down sorted p-values; reject as long as p_(i) ≤ (i/m)q. The threshold (i/m)q increases linearly — it is more lenient for lower-ranked (more significant) tests.

**Example** (m=5 tests, q=0.1, p-values: 0.01, 0.05, 0.1, 0.4, 0.6):
- i=1: 0.01 ≤ (1/5)×0.1 = 0.02 ✓
- i=2: 0.05 ≤ (2/5)×0.1 = 0.04 ✗
- k=1, reject only H_(1)

**Example** (m=5 tests, q=0.20, p-values: 0.01, 0.03, 0.15, 0.40, 0.50):
- i=1: 0.01 ≤ 1/5×0.20 = 0.04 ✓
- i=2: 0.03 ≤ 2/5×0.20 = 0.08 ✓
- i=3: 0.15 ≤ 3/5×0.20 = 0.12 ✗
- k=2, reject H_(1) and H_(2)

**BH vs Bonferroni**:
- BH (FDR control): More discoveries, controlled proportion of false ones.
- Bonferroni (FWER control): Fewer discoveries, controls probability of ANY false one.

**q vs α**: q (FDR level) is often set higher than α (e.g., q = 0.1 or 0.2) because the cost metric is different.
