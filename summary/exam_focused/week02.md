# Week 2 — Lasso, Elastic Net, Model Assessment, Bootstrap, Multiple Testing (Exam Focus)

## Must-Know Facts

### Lasso
- Lasso uses **$L_1$ penalty**: $\lambda\|\boldsymbol{\beta}\|_1 = \lambda\sum_j|\beta_j|$
- Lasso **DOES set coefficients to exactly zero** → performs variable selection.
- Lasso has **NO closed-form solution** — requires iterative algorithms (LARS, coordinate descent).
- The df for Lasso = **number of non-zero coefficients** (not $p$, not $n$).
- In $p > n$ setting: Lasso selects **at most $n$ variables**.
- With highly correlated predictors: Lasso picks **one arbitrarily** from the group (problem!).
- Geometry: $L_1$ constraint is a **diamond** with corners → solutions hit corners → exact zeros.
- Ridge constraint is a **sphere** → no corners → never exact zeros.

### LARS Algorithm
- LARS = Least Angle Regression Selection.
- Computes the **entire regularization path** (all $\lambda$) at speed of **one OLS fit**.
- LARS is **not** the same as forward selection — it is "polite" (moves only until the next variable becomes equally correlated with the residual).
- LASSO is LARS with a modification: **drop variables when their estimate crosses zero**.
- LARS assumes data is **centered and normalized**.
- Step size formula (2 vars): $\gamma = \dfrac{c_j - c_k}{1 - \rho_{jk}}$
- The direction LARS moves in is the **equiangular direction** (bisects angles between active variables).

### Cyclical Coordinate Descent
- Fixes $\lambda$, updates one $\beta_j$ at a time while holding others fixed.
- Key operation is **soft thresholding**: $\tilde{\beta}_j(\lambda) = \text{sign}(\tilde{\beta}_j^{\text{OLS}})\left(|\tilde{\beta}_j^{\text{OLS}}| - \lambda\right)_+$
- Soft thresholding: if $|\text{OLS estimate}| \leq \lambda$ → set to 0; else shrink by $\lambda$.

### Elastic Net
- Elastic net = $L_1 + L_2$ combined: $\min\ \frac{1}{2n}\|\mathbf{Y}-\mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\!\left[\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|^2 + \alpha\|\boldsymbol{\beta}\|_1\right]$
- $\alpha = 1$ → pure Lasso; $\alpha = 0$ → pure Ridge; $0 < \alpha < 1$ → elastic net.
- Elastic net handles **grouping effect** (includes/excludes correlated variables together).
- Implementation: augment data by appending $\sqrt{\lambda_2}\cdot\mathbf{I}$ to $\mathbf{X}$ and zeros to $\mathbf{y}$, then solve Lasso on augmented data.

### Nested Cross-Validation
- Standard CV error is **optimistically biased** when used for both selection AND assessment of $\lambda$.
- Nested CV separates: inner loop for **selection** (tune $\lambda$), outer loop for **assessment** (audit pipeline).
- The outer error estimates the generalization of the whole methodology, not just one model.
- It's normal for $\lambda^*$ to change between outer folds.
- Large gap (inner 5% vs outer 12%) → selection-induced overfitting.
- Cost: $K_{\text{outer}} \times (K_{\text{inner}} \times N_{\lambda} + 1)$ model fits.

### Bootstrap
- Bootstrap: sample **with replacement**, same size $N$, $B$ times.
- Bootstrap estimates statistical accuracy (standard errors, CIs), NOT used for model selection.
- For std dev: ~100–200 replicates; for CIs: 1000–2000 replicates.
- Bootstrap works poorly for **tail statistics** (extremes).
- Variance formula: $\widehat{\text{Var}}[S] = \frac{1}{B-1} \sum_b (S(Z^{*b}) - \bar{S}^*)^2$

### Classifier Performance
- **Sensitivity (Recall, TPR)** $= \text{TP}/(\text{TP}+\text{FN})$ — "of all actual positives, how many found?"
- **Specificity (TNR)** $= \text{TN}/(\text{TN}+\text{FP})$ — "of all actual negatives, how many correctly identified?"
- **Precision (PPV)** $= \text{TP}/(\text{TP}+\text{FP})$ — "of all predicted positives, how many are real?"
- **FPR** $= \text{FP}/(\text{FP}+\text{TN}) = 1 - \text{Specificity}$
- ROC plots **TPR vs FPR** as threshold varies. AUC $= 1$ is perfect; AUC $= 0.5$ is random.
- AUC-ROC is threshold-independent — general classifier performance.
- Low prevalence → accuracy is misleading; use **Precision-Recall** instead.

### Multiple Testing
- $\text{FWER} = 1 - (1-\alpha)^M$ — probability of at least one false positive in $M$ independent tests.
- Bonferroni: reject if $p < \alpha/M$ — controls FWER but has **low power**.
- $\text{FDR} = E[\text{FP}/(\text{FP}+\text{TP})]$ — expected proportion of false discoveries among all discoveries.
- Benjamini-Hochberg (BH): sort p-values, find largest $k$ where $p_{(k)} \leq \frac{k}{m}q$, reject top $k$.
- BH has **more power** than Bonferroni but allows a controlled fraction of false discoveries.
- FDR level $q$ is often set to 0.1 or 0.2 (higher than typical $\alpha = 0.05$).

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $\min_{\boldsymbol{\beta}}\ (\mathbf{Y}-\mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y}-\mathbf{X}\boldsymbol{\beta}) + \lambda\|\boldsymbol{\beta}\|_1$ | Lasso objective | Sparse regression |
| $\tilde{\beta}_j(\lambda) = \text{sign}(\tilde{\beta}_j^{\text{OLS}})(|\tilde{\beta}_j^{\text{OLS}}| - \lambda)_+$ | Soft thresholding | Coordinate descent update |
| $\gamma = (c_j - c_k)/(1 - \rho_{jk})$ | LARS step size | LARS algorithm |
| EN penalty: $\lambda\!\left[\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1\right]$ | Elastic net penalty | $\alpha=1\to$Lasso, $\alpha=0\to$Ridge |
| $\text{FWER} = 1-(1-\alpha)^M$ | Family-wise error rate | Multiple testing |
| Bonferroni: reject if $p < \alpha/M$ | FWER correction | Conservative multiple testing |
| BH: find max $k$ where $p_{(k)} \leq \frac{k}{m}q$ | FDR control | Multiple testing with power |
| $\text{FDR} = E[\text{FP}/(\text{FP}+\text{TP})]$ | False discovery rate | Multiple testing |
| $\widehat{\text{Var}}[S] = \frac{1}{B-1}\sum_b(S(Z^{*b})-\bar{S}^*)^2$ | Bootstrap variance | Statistical accuracy |
| $\text{Sensitivity} = \text{TP}/(\text{TP}+\text{FN})$ | True positive rate | Classifier evaluation |
| $\text{Specificity} = \text{TN}/(\text{TN}+\text{FP})$ | True negative rate | Classifier evaluation |

---

## Common Traps (wrong answers in exams)

- ❌ Lasso has a closed-form solution → ✓ Lasso has NO closed form; uses LARS or coordinate descent
- ❌ Ridge sets some coefficients to zero → ✓ Only LASSO ($L_1$) sets to zero; Ridge only shrinks
- ❌ LARS computes solutions for one $\lambda$ at a time → ✓ LARS computes the ENTIRE path (all $\lambda$) at once
- ❌ LARS is just forward selection → ✓ LARS is more efficient ("polite"): moves equiangularly, not greedily
- ❌ LASSO = LARS → ✓ LASSO is a MODIFICATION of LARS (drops variables when they hit zero)
- ❌ Elastic net with $\alpha=1$ is Ridge → ✓ $\alpha=1$ → Lasso; $\alpha=0$ → Ridge
- ❌ df for Lasso $= p$ → ✓ df for Lasso = number of NON-ZERO coefficients
- ❌ Bootstrap can be used for model selection → ✓ Bootstrap is for statistical accuracy (SEs, CIs); NOT for model selection (Tibshirani)
- ❌ Standard CV error is unbiased for assessing a pipeline that includes hyperparameter tuning → ✓ It is OPTIMISTICALLY BIASED — need nested CV for honest assessment
- ❌ Accuracy is always the right metric → ✓ With class imbalance, use Precision-Recall or AUC-ROC
- ❌ AUC $= 0.5$ means 50% accurate → ✓ AUC $= 0.5$ means RANDOM performance (like flipping a coin)
- ❌ Bonferroni is more powerful than BH → ✓ BH has MORE power; Bonferroni is MORE conservative (fewer discoveries)
- ❌ FWER $= \alpha$ when testing $M$ hypotheses → ✓ $\text{FWER} = 1-(1-\alpha)^M \gg \alpha$ for large $M$
- ❌ BH rejects all hypotheses below a fixed threshold → ✓ BH threshold is ADAPTIVE: $\frac{i}{m}q$ increases with rank $i$
- ❌ For $p > n$, Lasso can select any number of variables → ✓ Lasso selects AT MOST $n$ variables when $p > n$

---

## Quick Decision Rules

- "Produces sparse solutions / variable selection" → Lasso ($L_1$), not Ridge ($L_2$)
- "Closed-form solution for regularized regression" → Ridge only
- "Entire regularization path efficiently" → LARS algorithm
- "Correlated predictors, include/exclude together" → Elastic Net (grouping effect)
- "Unbiased assessment of tuning + training pipeline" → Nested CV (not simple CV)
- "Assessing standard error of a statistic" → Bootstrap
- "Control probability of any false positive" → Bonferroni (FWER)
- "Allow some false discoveries, maximize power" → Benjamini-Hochberg (FDR)
- "Imbalanced classes" → use Precision-Recall or AUC-ROC, not accuracy
- BH procedure: sort p-values ascending, walk down, reject while $p_{(i)} \leq \frac{i}{m}q$; once you fail, stop
- If $q=0.2$ and $m=5$: thresholds are $0.04, 0.08, 0.12, 0.16, 0.20$ for $i=1,2,3,4,5$
