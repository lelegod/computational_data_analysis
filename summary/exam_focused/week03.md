# Week 3 — Sparse Regression, Curse of Dimensionality, Multiple Testing (Exam Focus)

## Must-Know Facts

### Curse of Dimensionality
- Number of regions grows **exponentially** with dimension $D$.
- With $p > N$, models can perfectly fit noise → OLS is unreliable.
- Euclidean distances lose meaning in high dimensions — all points become roughly equidistant.
- Most data points concentrate at the **boundaries/corners** of the sample space (edge effect).
- "Local neighborhoods become empty" — KNN-type methods break down.

### Blessings of Dimensionality (Donoho 2000)
- 3 blessings: (1) correlated features can be averaged, (2) data lies on low-dimensional manifold, (3) continuous processes have approximate finite dimensionality.

### 1-SE Rule
- After cross-validation, choose the **largest $\lambda$** whose CV error is within **1 standard error** of the minimum CV error.
- Effect: Picks a simpler, more regularized model — more stable across replications.
- Source: Breiman et al. (1984) CART monograph.

### Norms
- $L_2$ norm squared: $\|\boldsymbol{\beta}\|_2^2 = \sum_j \beta_j^2$ — used in Ridge
- $L_1$ norm: $\|\boldsymbol{\beta}\|_1 = \sum_j |\beta_j|$ — used in Lasso

### Ridge (Recap)
- Penalty: $L_2$ (quadratic shrinkage). Constraint: sphere.
- Never zeros out coefficients. No variable selection.
- Does NOT penalize the intercept $\beta_0$.

### Lasso
- Penalty: $L_1$ (absolute-value shrinkage). Constraint: diamond.
- Diamond has corners → RSS ellipsoid touches corner → exact zeros → variable selection.
- No closed form. $\text{df}$ = number of non-zero coefficients.
- $p > n$: selects at most $n$ variables.

### Elastic Net
- Penalty: $\lambda\!\left[\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1\right]$
- $\alpha=1$ → Lasso; $\alpha=0$ → Ridge; $0<\alpha<1$ → Elastic Net.
- Solves the three Lasso limitations: high-D, grouping, predictive power.
- Implemented by augmenting $\mathbf{X}$ with $\sqrt{\lambda_2}\cdot\mathbf{I}$ rows and $\mathbf{y}$ with zeros, then solving standard Lasso.

### LARS
- Computes entire Lasso path at cost of one OLS fit.
- Data must be centered and normalized ($\mathbf{X}^T \mathbf{X} \approx \text{Corr}(\mathbf{X})$).
- Moves in equiangular direction (bisects active variables).
- Lasso-LARS: if a coefficient crosses zero, drop it from active set.
- $C_p$ for LARS: $C_p = \frac{1}{\hat{\sigma}^2}\sum_i(y_i-\hat{y}_i)^2 - n + 2k$

### Coordinate Descent
- Cyclically updates one $\beta_j$ at a time (others fixed).
- Update: compute partial residual, get OLS solution, apply soft thresholding.
- Soft thresholding: $\text{sign}(x)(|x|-\lambda)_+$ — zeroes if $|x|\leq\lambda$, else shrinks by $\lambda$.

### BIC vs AIC behavior with $N$
- AIC: "more data → I can afford more complexity" — penalty doesn't grow with $N$.
- BIC: "more data → I can confidently detect if extra parameters are warranted" — penalty grows with $\log(N)$.

### Multiple Testing: FWER
- Testing 1 hypothesis at $\alpha$: $P(\text{false rejection}) = \alpha$.
- Testing $M$ independent hypotheses: $\text{FWER} = 1 - (1-\alpha)^M$.
- Example: 20 tests at $\alpha=0.05$ → $\text{FWER} = 1-(0.95)^{20} \approx 0.64$.

### Bonferroni
- Reject if p-value $< \alpha/M$.
- Controls FWER but has low power (misses real effects).

### FDR and BH Algorithm
- $\text{FDR} = E[\text{FP}/(\text{FP}+\text{TP})]$ — expected fraction of discoveries that are false.
- BH: sort p-values ascending, find largest $k$ where $p_{(k)} \leq \frac{k}{m}q$, reject top $k$.
- BH is more powerful than Bonferroni.
- $q$ (FDR threshold) is typically set to 0.1 or 0.2.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $\|\boldsymbol{\beta}\|_2^2 = \sum_j \beta_j^2$ | $L_2$ norm (Ridge penalty) | Ridge regression |
| $\|\boldsymbol{\beta}\|_1 = \sum_j |\beta_j|$ | $L_1$ norm (Lasso penalty) | Lasso regression |
| Ridge: $\min\ (\mathbf{Y}-\mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y}-\mathbf{X}\boldsymbol{\beta}) + \lambda\boldsymbol{\beta}^T\boldsymbol{\beta}$ | Ridge objective | Ridge |
| Ridge solution: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ | Closed form | Ridge |
| Lasso: $\min\ (\mathbf{Y}-\mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y}-\mathbf{X}\boldsymbol{\beta}) + \lambda\|\boldsymbol{\beta}\|_1$ | Lasso objective | Lasso |
| EN: $\lambda\!\left[\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1\right]$ | Elastic net penalty | $\alpha=1\to$Lasso, $\alpha=0\to$Ridge |
| Soft thresh: $\text{sign}(x)(|x|-\lambda)_+$ | Coordinate descent update | Lasso coordinate descent |
| LARS step: $\gamma = (c_j-c_k)/(1-\rho_{jk})$ | LARS step size | LARS algorithm |
| LARS $C_p$: $\frac{1}{\hat{\sigma}^2}\sum_i(y_i-\hat{y}_i)^2-n+2k$ | Model selection for LARS | Choosing LARS steps |
| $\text{FWER} = 1-(1-\alpha)^M$ | Family-wise error rate | Multiple testing |
| Bonferroni: reject if $p < \alpha/M$ | FWER correction | Conservative testing |
| BH: find max $k$: $p_{(k)} \leq \frac{k}{m}q$ | FDR control | Multiple testing with power |
| $\text{FDR} = E[\text{FP}/(\text{FP}+\text{TP})]$ | Expected false discovery rate | Multiple testing |

---

## Common Traps (wrong answers in exams)

- ❌ Lasso always outperforms Ridge when predictors are correlated → ✓ With high correlation and $n>p$, Ridge often OUTPERFORMS Lasso (limitation #3 of Lasso)
- ❌ Elastic net with $\alpha=0$ is Lasso → ✓ $\alpha=0$ is Ridge; $\alpha=1$ is Lasso
- ❌ 1-SE rule selects the model with minimum CV error → ✓ 1-SE selects the MOST REGULARIZED model within 1-SE of minimum (simpler model)
- ❌ LARS is the same as forward stepwise selection → ✓ LARS is "polite" — moves equiangularly, not by adding one full variable at a time
- ❌ BH uses a fixed threshold for all hypotheses → ✓ BH threshold is ADAPTIVE: $\frac{i}{m}q$ — increases with rank
- ❌ FWER and FDR control the same thing → ✓ FWER controls probability of ANY false positive; FDR controls PROPORTION of discoveries that are false
- ❌ BH algorithm: continue rejecting after a failure → ✓ Stop at the FIRST failure; reject all hypotheses up to and including $k$ (not just the ones that passed individually)
- ❌ Bonferroni has more power than BH → ✓ BH has MORE power; Bonferroni is more conservative (lower threshold)
- ❌ BIC penalizes less than AIC for large $N$ → ✓ BIC penalty grows as $\log(N)$ — BIC penalizes MORE than AIC for large $N$ ($N > e^2 \approx 7.4$)
- ❌ Soft thresholding shrinks all values equally → ✓ Soft thresholding ZEROES values below $\lambda$ and shrinks larger values by exactly $\lambda$
- ❌ Curse of dimensionality only affects distance metrics → ✓ It also affects overfitting ($p>n$), sparsity of data, edge effects, and computational cost
- ❌ Elastic net augmented data approach changes the objective → ✓ Augmentation ABSORBS the $L_2$ penalty into the residuals — it is mathematically equivalent, not approximate
- ❌ The intercept $\beta_0$ is penalized in Ridge → ✓ The intercept is typically NOT penalized in Ridge

---

## Quick Decision Rules

- "Which method handles correlated groups of predictors best?" → Elastic Net (grouping effect)
- "Which method works when $p > n$ and you need variable selection?" → Lasso or Elastic Net (not Ridge alone)
- "Largest $\lambda$ within 1-SE of CV minimum" → 1-SE rule for simpler/more stable model
- "Control any false positive" → Bonferroni (FWER)
- "Allow some false positives, maximize discoveries" → BH (FDR)
- BH procedure step-by-step: sort p-values, compute threshold $\frac{i}{m}q$ for each rank, find the LAST rank $k$ where $p_{(k)} \leq$ threshold, reject ranks 1 through $k$
- Computing FWER: $1 - (1-\alpha)^M$; for $\alpha=0.05$ and $M=20$ → $\approx 64\%$
- If $q=0.1$ and $m=10$: thresholds are $0.01, 0.02, 0.03, \ldots, 0.10$ for $i=1,\ldots,10$
- "AIC selects more complex or simpler than BIC?" → AIC selects more COMPLEX (penalty $2d$ vs BIC's $\log(N)d$ for large $N$)
- "Is Ridge or Lasso used as the basis for LARS?" → LARS computes the Lasso path; LARS modified = Lasso
