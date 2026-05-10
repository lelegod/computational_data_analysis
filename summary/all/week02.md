# Week 2 — Lasso, Elastic Net, Model Assessment, Bootstrap, Multiple Testing

## Overview
Week 2 extends week 1's regularization framework. It introduces Lasso ($L_1$ penalty) which performs variable selection by setting some coefficients to exactly zero. Two algorithms for solving the Lasso are covered: LARS and Cyclical Coordinate Descent. The elastic net combines $L_1$ and $L_2$ penalties. Model assessment topics include nested cross-validation, the bootstrap, classifier performance metrics (confusion matrix, ROC), and multiple testing correction (FWER, Bonferroni, FDR / Benjamini-Hochberg).

---

## Part I — The Lasso

### Key Concepts
- Lasso = "Least Absolute Shrinkage and Selection Operator"
- Uses an $L_1$ penalty instead of ridge's $L_2$ penalty.
- Critical difference: $L_1$ penalty produces **exact zeros** in the coefficient vector → automatic variable selection.

### Lasso Objective Function
- **Penalized form**: $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) + \lambda\|\boldsymbol{\beta}\|_1$
  - $\|\boldsymbol{\beta}\|_1 = \sum_j |\beta_j|$ — sum of absolute values
- **Constrained form** (basis pursuit): $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$ subject to $\sum_j |\beta_j| \leq s$
- $\lambda$ (or equivalently $s$) controls sparsity; larger $\lambda$ → more zeros.

### Why Lasso Produces Zeros (Geometry)
- The $L_1$ constraint region is a **diamond** (in 2D) with corners on the axes.
- The RSS ellipsoid typically first touches the diamond at a **corner**, where one or more $\beta_j = 0$.
- Contrast with ridge: $L_2$ constraint is a **sphere** with no corners → solutions are never exactly zero.

### Lasso Properties
- **Non-differentiable** at $\beta = 0$ (the $L_1$ norm has a kink there).
- **No closed-form solution** — must use iterative algorithms.
- For large enough $\lambda$: some $\beta_j$ will be set to **exactly zero**.
- Effective number of parameters $\text{df}$ = **number of non-zero coefficients** (coefficients different from zero).
- In the $p > n$ case: Lasso selects **at most $n$ variables**.
- If predictors are correlated, Lasso tends to pick **one** from a correlated group (arbitrarily).

### Lasso Limitations (motivating Elastic Net)
1. **High dimensionality** ($p > n$): Lasso selects at most $n$ variables.
2. **Grouping effect**: With correlated predictors, Lasso picks one arbitrarily — Ridge tends to include the whole group.
3. **Predictive power**: When $n > p$ and predictors are highly correlated, Ridge often outperforms Lasso.

---

## Part II — Algorithms for Lasso

### Algorithm 1: Least Angle Regression Selection (LARS)

#### Overview
- LARS is the computational "engine" for finding the entire Lasso/Elastic Net regularization path.
- Computes all $\lambda$ values at the **speed of a single OLS fit**.
- LASSO is a **modification** of LARS: if a parameter estimate crosses zero, set it to zero and recompute direction.

#### LARS Algorithm Steps
1. **Initialize**: Start with all $\boldsymbol{\beta} = \mathbf{0}$, current estimate $\mu_0 = 0$, residual $r = y$.
2. **Find most correlated variable**: Compute correlations $c = \mathbf{X}^T r$. Find $x_j$ with max $|\text{correlation}|$.
3. **Move** $\beta_j$ in the direction of its least-squares coefficient.
4. **Stop** when another variable $x_k$ has as much correlation with the current residual as $x_j$.
5. **Move in equiangular direction**: Move in the direction that bisects the angle between $x_j$ and $x_k$.
6. Repeat until all variables are included or residuals are zero.

#### Key Terminology
- $\mu$: current prediction estimate
- Active set $A$: set of variables currently being moved
- Equiangular direction $u_A = \mathbf{X}_A w$, where $w = A(\mathbf{X}_A^T \mathbf{X}_A)^{-1} \mathbf{1}$, $A$ is a normalization factor so $\|u_A\| = 1$
- Step size $\gamma$: chosen so the residual becomes equally correlated with all active variables

#### LARS Step Size Formula (2 variables)
$$\gamma = \frac{c_j - c_k}{1 - \rho_{jk}}$$
- $c_j,\ c_k$: current correlations of $x_j$ and $x_k$ with residual
- $\rho_{jk}$: correlation between $x_j$ and $x_k$

#### LARS vs Forward Selection (Greedy vs Polite)
| Forward Selection (Greedy) | LARS (Polite) |
|---------------------------|---------------|
| Finds best variable | Finds best variable |
| Moves along it completely until it can't improve | Moves only until a second becomes equally helpful |
| Aggressive, jerky path | Efficient, equiangular path |

#### LARS Assumptions
- Data is centered and normalized (each variable has length 1).
- This means $\mathbf{X}^T \mathbf{X} \approx \text{Corr}(\mathbf{X})$.

#### Cp for LARS (choosing number of steps)
$$C_p = \frac{1}{\hat{\sigma}^2} \sum_i (y_i - \hat{y}_i)^2 - n + 2k$$
where $k$ is the number of LARS steps taken.

### Algorithm 2: Cyclical Coordinate Descent

#### Overview
Fix $\lambda$ and solve $\min_{\boldsymbol{\beta}}\ \frac{1}{2n} \sum_i (y_i - x_i^T \boldsymbol{\beta})^2 + \lambda|\beta_j|$ iteratively, updating one coordinate at a time.

#### Steps
1. Compute partial residual for coordinate $j$ (holding all others fixed):
   $$r_i^{(j)} = y_i - \sum_{k \neq j} x_{ik} \tilde{\beta}_k(\lambda)$$
2. Compute the OLS solution for this partial residual:
   $$\tilde{\beta}_j^{\text{OLS}} = \frac{1}{n} \sum_i x_{ij} r_i^{(j)}$$
   (under standardization: $\sum_i x_{ij} = 0$ and $\frac{1}{n}\sum_i x_{ij}^2 = 1$)
3. Apply **soft thresholding** to get the Lasso update:
   $$\tilde{\beta}_j(\lambda) = \text{sign}(\tilde{\beta}_j^{\text{OLS}})\left(|\tilde{\beta}_j^{\text{OLS}}| - \lambda\right)_+$$
4. Cycle through $j = 1, \ldots, p$ repeatedly until convergence.

#### Soft Thresholding
$$S(x, \Delta) = \text{sign}(x)(|x| - \Delta)_+$$
- If $|x| \leq \Delta$: result is $0$ (shrinks to zero).
- If $|x| > \Delta$: result is $x - \Delta \cdot \text{sign}(x)$ (shrinks toward zero by $\Delta$).
- This is the key operation that produces sparsity.

---

## Part III — Elastic Net

### Motivation
Combines $L_1$ (Lasso) and $L_2$ (Ridge) penalties to get sparsity AND grouping behavior.

### Elastic Net Objective
- **Penalized form**: $\min_{\boldsymbol{\beta}}\ \frac{1}{2n}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\!\left[\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1\right]$
- **Constrained form**: $\min_{\boldsymbol{\beta}}\ \frac{1}{2n}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$ s.t. $\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1 \leq t$

### $\alpha$ Parameter (mixing)
- $\alpha = 1$: Pure Lasso ($L_1$ only)
- $\alpha = 0$: Pure Ridge ($L_2$ only)
- $0 < \alpha < 1$: Elastic Net (the "elastic" region)

### Advantage
- Combines shrinkage of Ridge with variable selection of Lasso.
- Handles the grouping effect: tends to include or exclude correlated variables together.
- More robust sparse estimate than Lasso alone.

### Implementation via Augmented Data
To use standard LARS/Lasso solvers for Elastic Net, "hide" the $L_2$ penalty inside the data:

**Step 1 — Construct augmented matrices**:

$$\mathbf{X}^*_{(n+m)\times m} = \begin{bmatrix} \mathbf{X} \\ \sqrt{\lambda_2}\, \mathbf{I}_m \end{bmatrix}, \qquad \mathbf{y}^*_{(n+m)} = \begin{bmatrix} \mathbf{y} \\ \mathbf{0}_m \end{bmatrix}$$

- $m$ = number of features, $\mathbf{I}_m$ is the $m \times m$ identity matrix.
- Bottom $m$ rows of $\mathbf{y}^*$ are zeros.

**Step 2 — Absorption**:

$$\|\mathbf{y}^* - \mathbf{X}^*\boldsymbol{\beta}\|^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \|\mathbf{0} - \sqrt{\lambda_2}\,\mathbf{I}\,\boldsymbol{\beta}\|^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_2\|\boldsymbol{\beta}\|_2^2$$

The $L_2$ penalty is now absorbed into the residual term.

**Step 3 — Solve the LASSO** on $(\mathbf{y}^*, \mathbf{X}^*)$:

$$\min_{\boldsymbol{\beta}^*}\ \|\mathbf{y}^* - \mathbf{X}^*\boldsymbol{\beta}^*\|^2 + \lambda_1\|\boldsymbol{\beta}^*\|_1$$

The solution $\boldsymbol{\beta}^*$ is a **scaled ridge solution**: $\frac{1}{\sqrt{1+\lambda_2}}$ is the scaling factor.

**Why**: LARS/Coordinate Descent only see a standard LASSO problem; they are blind to the hybrid penalty.

---

## Part IV — Feature / Variable Selection Methods

### Combinatorial Search
- Try all possible subsets of features; select optimal.
- Pro: Guaranteed to find best combination.
- Con: $2^p$ combinations — computationally infeasible for large $p$.

### Forward Selection
- Start with no variables; add one at a time (highest information criterion gain).
- Pro: $O(p^2)$ models; works when $p > n$.
- Con: May not find the globally optimal combination.

### Backward Elimination
- Start with all variables; remove one at a time (lowest information criterion loss).
- Pro: $O(p^2)$ models.
- Con: Numerical issues with many features; requires $n > p$ to start.
- Usually better than forward selection.

---

## Part V — Model Assessment

### The Selection-Induced Bias Problem
- When you test many $\lambda$ values and pick the one with **minimum CV error**, the minimum is **optimistically biased**.
- Reason: You have "spent" the independence of the validation folds by using them to select $\lambda$.
- The resulting error estimate is **not** an unbiased estimate of future performance.
- Insight: "We didn't just fit the model; we fitted the hyperparameter to the noise in the CV folds."

### Nested Cross-Validation (Solution)
Separate **model selection** from **model assessment** using two loops:

**Inner loop (Selection)**:
- Used to tune $\lambda$.
- Finds the best configuration for a specific training set.

**Outer loop (Assessment)**:
- Used to audit the **entire procedure** (including the selection step).
- Estimates how well the "Selection + Training" pipeline generalizes.

**Nested CV Algorithm**:
1. Split data into $K_{\text{outer}}$ folds.
2. For each outer fold $j$ (test):
   a. Take remaining data as "Training Set."
   b. Inner loop: Perform $K_{\text{inner}}$-fold CV on training set to find best $\lambda^*$.
   c. Train final model with $\lambda^*$ on the **entire** training set.
   d. Evaluate on held-out outer fold $j$.
3. Final report: Average the $K_{\text{outer}}$ test scores.

**Computational cost**: Total fits $= K_{\text{outer}} \times (K_{\text{inner}} \times N_{\lambda} + 1)$
- Example: $10 \times 10 \times 100 = 10{,}000$ model fits.

**Key insight**: Nested CV audits the **methodology** (the whole pipeline), not a specific single model.
- It's OK if the best $\lambda$ changes across outer folds.
- A large gap between inner error (5%) and outer error (12%) indicates selection-induced overfitting.

---

## Part VI — The Bootstrap

### What is Bootstrap?
- A general method for **assessing statistical accuracy** (standard errors, confidence intervals, bias).
- Invented by Efron. Key idea: use the data itself as a "mirror copy of the real world."
- Bootstrap estimates $\approx$ Monte Carlo estimates (but drawing from empirical distribution instead of true $P$).

### Conceptual Framework (Freedman's terminology)
- **Real world**: Unknown $P$ → observed data $x = (x_1, \ldots, x_n)$ → statistic $\hat{\theta} = s(x)$
- **Bootstrap world**: Estimated $\hat{P}$ → bootstrap sample $x^* = (x_1^*, \ldots, x_n^*)$ → bootstrap replication $\hat{\theta}^* = s(x^*)$

### Bootstrap Method
1. Given training set $Z = (z_1, \ldots, z_N)$ where $z_i = (x_i, y_i)$.
2. Randomly draw with replacement from $Z$, same size $N$ → bootstrap sample $Z^{*b}$.
3. Repeat $B$ times ($B = 100$ or more), producing $B$ bootstrap datasets.
4. Refit the model to each $Z^{*b}$, compute statistic $S(Z^{*b})$.
5. Variance estimate:

$$\widehat{\text{Var}}[S(Z)] = \frac{1}{B-1} \sum_b \left(S(Z^{*b}) - \bar{S}^*\right)^2, \quad \bar{S}^* = \frac{1}{B}\sum_b S(Z^{*b})$$

### Practical Remarks
- For **standard deviation**: a few hundred replicates suffice.
- For **confidence intervals**: 1000–2000 replicates recommended.
- Try different $B$ and check if results change.
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
- **Accuracy** $= \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ — fraction correct. Dangerous for imbalanced data.
- **Sensitivity / Recall / TPR** $= \frac{\text{TP}}{\text{TP} + \text{FN}}$ — fraction of actual positives detected.
- **Specificity / TNR** $= \frac{\text{TN}}{\text{TN} + \text{FP}}$ — fraction of actual negatives correctly identified.
- **Precision / PPV** $= \frac{\text{TP}}{\text{TP} + \text{FP}}$ — fraction of predicted positives that are true positives.
- **FPR (False Positive Rate)** $= \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{Specificity}$
- **FNR (False Negative Rate)** $= \frac{\text{FN}}{\text{FN} + \text{TP}} = 1 - \text{Sensitivity}$
- **F1 score** $= \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}$ = harmonic mean of precision and recall
- **Balanced accuracy** $= \frac{\text{Sensitivity} + \text{Specificity}}{2}$

**Note**: If prevalence is low (e.g., 0.1%), ignore accuracy — use Precision-Recall curves instead.

### ROC Curve — Receiver Operating Characteristics
- Plots **TPR (Sensitivity)** vs **FPR ($1-$Specificity)** as the classification threshold varies.
- **AUC-ROC**: Area under the ROC curve.
  - AUC $= 1.0$: perfect classifier.
  - AUC $= 0.5$: random classifier (diagonal line).
  - AUC $> 0.5$: better than random.
- ROC/AUC gives general performance across ALL classification thresholds.
- Can be extended to multiclass: micro-average and macro-average.

### Regression Performance Metrics
- **MSE/RMSE**: $\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$ — outlier sensitive; useful for safety-critical audits.
- **MAE**: $\frac{1}{n}\sum_i |y_i - \hat{y}_i|$ — robust to outliers; direct physical interpretation.
- **$R^2$**: Fraction of variance explained. Relative, not absolute measure.
- **Residual plots**: Final sanity check; if residuals show patterns, model is incomplete regardless of MSE.

---

## Part VIII — Multiple Testing

### The Problem
- Testing one hypothesis at significance level $\alpha$: probability of false rejection $= \alpha$.
- Testing $M$ hypotheses: the probability of **at least one false rejection** is much larger than $\alpha$.

### Family-Wise Error Rate (FWER)
- **Definition**: Probability of at least one false rejection across all $M$ tests.
- **Formula** (independent tests): $\text{FWER} = 1 - (1 - \alpha)^M$
- Example: $M=20$ tests at $\alpha=0.05$ → $\text{FWER} = 1 - (0.95)^{20} \approx 0.64$ (64% chance of at least one false discovery!)

### Bonferroni Correction
- **Method**: Reject hypothesis if p-value $< \alpha/M$.
- **Effect**: Controls FWER at level $\alpha$ (assuming independence).
- **Cost**: Low power — we miss many true effects.

### False Discovery Rate (FDR)
- **Definition**: $\text{FDR} = E\!\left[\frac{\text{FP}}{\text{FP} + \text{TP}}\right]$
  - FP = false positives (false discoveries)
  - TP = true positives (true discoveries)
- **Trade-off**: Allows a controlled fraction of false discoveries → more power than Bonferroni.
- Proposed by Benjamini and Hochberg (1995).
- Set FDR threshold $q$: among all findings, we expect at most fraction $q$ to be mistakes.

### Benjamini-Hochberg (BH) Algorithm for FDR
Given $m$ hypothesis tests with p-values $p_1, \ldots, p_m$ and target FDR level $q$:
1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find the largest $k$ such that: $p_{(k)} \leq \frac{k}{m} \cdot q$
3. Reject all hypotheses $H_{(1)}, H_{(2)}, \ldots, H_{(k)}$

**Intuition**: Walk down sorted p-values; reject as long as $p_{(i)} \leq \frac{i}{m}q$. The threshold $\frac{i}{m}q$ increases linearly — it is more lenient for lower-ranked (more significant) tests.

**Example** ($m=5$ tests, $q=0.1$, p-values: 0.01, 0.05, 0.1, 0.4, 0.6):
- $i=1$: $0.01 \leq \frac{1}{5}\times 0.1 = 0.02$ ✓
- $i=2$: $0.05 \leq \frac{2}{5}\times 0.1 = 0.04$ ✗
- $k=1$, reject only $H_{(1)}$

**Example** ($m=5$ tests, $q=0.20$, p-values: 0.01, 0.03, 0.15, 0.40, 0.50):
- $i=1$: $0.01 \leq \frac{1}{5}\times 0.20 = 0.04$ ✓
- $i=2$: $0.03 \leq \frac{2}{5}\times 0.20 = 0.08$ ✓
- $i=3$: $0.15 \leq \frac{3}{5}\times 0.20 = 0.12$ ✗
- $k=2$, reject $H_{(1)}$ and $H_{(2)}$

**BH vs Bonferroni**:
- BH (FDR control): More discoveries, controlled proportion of false ones.
- Bonferroni (FWER control): Fewer discoveries, controls probability of ANY false one.

**$q$ vs $\alpha$**: $q$ (FDR level) is often set higher than $\alpha$ (e.g., $q = 0.1$ or $0.2$) because the cost metric is different.
