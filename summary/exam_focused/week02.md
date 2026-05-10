# Week 2 — Lasso, Elastic Net, Model Assessment, Bootstrap, Multiple Testing (Exam Focus)

## Must-Know Facts

### Lasso
- Lasso uses **L₁ penalty**: λ||β||₁ = λΣ|βⱼ|
- Lasso **DOES set coefficients to exactly zero** → performs variable selection.
- Lasso has **NO closed-form solution** — requires iterative algorithms (LARS, coordinate descent).
- The df for Lasso = **number of non-zero coefficients** (not p, not n).
- In p > n setting: Lasso selects **at most n variables**.
- With highly correlated predictors: Lasso picks **one arbitrarily** from the group (problem!).
- Geometry: L₁ constraint is a **diamond** with corners → solutions hit corners → exact zeros.
- Ridge constraint is a **sphere** → no corners → never exact zeros.

### LARS Algorithm
- LARS = Least Angle Regression Selection.
- Computes the **entire regularization path** (all λ) at speed of **one OLS fit**.
- LARS is **not** the same as forward selection — it is "polite" (moves only until the next variable becomes equally correlated with the residual).
- LASSO is LARS with a modification: **drop variables when their estimate crosses zero**.
- LARS assumes data is **centered and normalized**.
- Step size formula (2 vars): `γ = (cⱼ - cₖ) / (1 - ρⱼₖ)`
- The direction LARS moves in is the **equiangular direction** (bisects angles between active variables).

### Cyclical Coordinate Descent
- Fixes λ, updates one βⱼ at a time while holding others fixed.
- Key operation is **soft thresholding**: `β̃ⱼ(λ) = sign(β̃ⱼ^OLS)(|β̃ⱼ^OLS| - λ)₊`
- Soft thresholding: if |OLS estimate| ≤ λ → set to 0; else shrink by λ.

### Elastic Net
- Elastic net = L₁ + L₂ combined: `min (1/2n)||Y-Xβ||² + λ[(1/2)(1-α)||β||² + α||β||₁]`
- α = 1 → pure Lasso; α = 0 → pure Ridge; 0 < α < 1 → elastic net.
- Elastic net handles **grouping effect** (includes/excludes correlated variables together).
- Implementation: augment data by appending √λ₂·I to X and zeros to y, then solve Lasso on augmented data.

### Nested Cross-Validation
- Standard CV error is **optimistically biased** when used for both selection AND assessment of λ.
- Nested CV separates: inner loop for **selection** (tune λ), outer loop for **assessment** (audit pipeline).
- The outer error estimates the generalization of the whole methodology, not just one model.
- It's normal for λ* to change between outer folds.
- Large gap (inner 5% vs outer 12%) → selection-induced overfitting.
- Cost: K_outer × (K_inner × N_lambdas + 1) model fits.

### Bootstrap
- Bootstrap: sample **with replacement**, same size N, B times.
- Bootstrap estimates statistical accuracy (standard errors, CIs), NOT used for model selection.
- For std dev: ~100-200 replicates; for CIs: 1000-2000 replicates.
- Bootstrap works poorly for **tail statistics** (extremes).
- Variance formula: `Var̂[S] = (1/(B-1)) Σ(S(Z*b) - S̄*)²`

### Classifier Performance
- **Sensitivity (Recall, TPR)** = TP/(TP+FN) — "of all actual positives, how many found?"
- **Specificity (TNR)** = TN/(TN+FP) — "of all actual negatives, how many correctly identified?"
- **Precision (PPV)** = TP/(TP+FP) — "of all predicted positives, how many are real?"
- **FPR** = FP/(FP+TN) = 1 - Specificity
- ROC plots **TPR vs FPR** as threshold varies. AUC = 1 is perfect; AUC = 0.5 is random.
- AUC-ROC is threshold-independent — general classifier performance.
- Low prevalence → accuracy is misleading; use **Precision-Recall** instead.

### Multiple Testing
- FWER = 1 - (1-α)^M — probability of at least one false positive in M independent tests.
- Bonferroni: reject if p < α/M — controls FWER but has **low power**.
- FDR = E[FP/(FP+TP)] — expected proportion of false discoveries among all discoveries.
- Benjamini-Hochberg (BH): sort p-values, find largest k where p_(k) ≤ (k/m)q, reject top k.
- BH has **more power** than Bonferroni but allows a controlled fraction of false discoveries.
- FDR level q is often set to 0.1 or 0.2 (higher than typical α = 0.05).

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| `min_β (Y-Xβ)^T(Y-Xβ) + λ\|\|β\|\|₁` | Lasso objective | Sparse regression |
| `β̃ⱼ(λ) = sign(β̃ⱼ^OLS)(\|β̃ⱼ^OLS\| - λ)₊` | Soft thresholding | Coordinate descent update |
| `γ = (cⱼ - cₖ)/(1 - ρⱼₖ)` | LARS step size | LARS algorithm |
| EN: `λ[(1/2)(1-α)\|\|β\|\|² + α\|\|β\|\|₁]` | Elastic net penalty | α=1→Lasso, α=0→Ridge |
| `FWER = 1-(1-α)^M` | Family-wise error rate | Multiple testing |
| Bonferroni: reject if `p < α/M` | FWER correction | Conservative multiple testing |
| BH: find max k where `p_(k) ≤ (k/m)q` | FDR control | Multiple testing with power |
| `FDR = E[FP/(FP+TP)]` | False discovery rate | Multiple testing |
| `Var̂[S] = (1/(B-1))Σ(S(Z*b)-S̄*)²` | Bootstrap variance | Statistical accuracy |
| `Sensitivity = TP/(TP+FN)` | True positive rate | Classifier evaluation |
| `Specificity = TN/(TN+FP)` | True negative rate | Classifier evaluation |

---

## Common Traps (wrong answers in exams)

- ❌ Lasso has a closed-form solution → ✓ Lasso has NO closed form; uses LARS or coordinate descent
- ❌ Ridge sets some coefficients to zero → ✓ Only LASSO (L₁) sets to zero; Ridge only shrinks
- ❌ LARS computes solutions for one λ at a time → ✓ LARS computes the ENTIRE path (all λ) at once
- ❌ LARS is just forward selection → ✓ LARS is more efficient ("polite"): moves equiangularly, not greedily
- ❌ LASSO = LARS → ✓ LASSO is a MODIFICATION of LARS (drops variables when they hit zero)
- ❌ Elastic net with α=1 is Ridge → ✓ α=1 → Lasso; α=0 → Ridge
- ❌ df for Lasso = p → ✓ df for Lasso = number of NON-ZERO coefficients
- ❌ Bootstrap can be used for model selection → ✓ Bootstrap is for statistical accuracy (SEs, CIs); NOT for model selection (Tibshirani)
- ❌ Standard CV error is unbiased for assessing a pipeline that includes hyperparameter tuning → ✓ It is OPTIMISTICALLY BIASED — need nested CV for honest assessment
- ❌ Accuracy is always the right metric → ✓ With class imbalance, use Precision-Recall or AUC-ROC
- ❌ AUC = 0.5 means 50% accurate → ✓ AUC = 0.5 means RANDOM performance (like flipping a coin)
- ❌ Bonferroni is more powerful than BH → ✓ BH has MORE power; Bonferroni is MORE conservative (fewer discoveries)
- ❌ FWER = α when testing M hypotheses → ✓ FWER = 1-(1-α)^M >> α for large M
- ❌ BH rejects all hypotheses below a fixed threshold → ✓ BH threshold is ADAPTIVE: (i/m)q increases with rank i
- ❌ For p > n, Lasso can select any number of variables → ✓ Lasso selects AT MOST n variables when p > n

---

## Quick Decision Rules

- "Produces sparse solutions / variable selection" → Lasso (L₁), not Ridge (L₂)
- "Closed-form solution for regularized regression" → Ridge only
- "Entire regularization path efficiently" → LARS algorithm
- "Correlated predictors, include/exclude together" → Elastic Net (grouping effect)
- "Unbiased assessment of tuning + training pipeline" → Nested CV (not simple CV)
- "Assessing standard error of a statistic" → Bootstrap
- "Control probability of any false positive" → Bonferroni (FWER)
- "Allow some false discoveries, maximize power" → Benjamini-Hochberg (FDR)
- "Imbalanced classes" → use Precision-Recall or AUC-ROC, not accuracy
- BH procedure: sort p-values ascending, walk down, reject while p_(i) ≤ (i/m)q; once you fail, stop
- If q=0.2 and m=5: thresholds are 0.04, 0.08, 0.12, 0.16, 0.20 for i=1,2,3,4,5
