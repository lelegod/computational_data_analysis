# Week 1 — Regression & Bias-Variance Tradeoff (Exam Focus)

## Must-Know Facts

### EPE / Bias-Variance
- EPE = σ² + Bias² + Variance — three terms, always.
- σ² is **irreducible** noise — no model can reduce it.
- Complex models have **low bias, high variance**; simple models have **high bias, low variance**.
- Training error always **decreases** with complexity; test error forms a **U-shape**.
- The decomposition holds at a specific point x₀ and averages over both y and the training set D.

### OLS
- OLS estimator: β̂ = (X^T X)^{-1} X^T y — requires X^T X to be invertible.
- OLS is **unbiased**: E[β̂] = β.
- OLS has **minimum variance** among all linear unbiased estimators (Gauss-Markov).
- OLS has df = p (number of predictors).
- Hat matrix: Ŷ = SY where S = X(X^T X)^{-1} X^T.

### Ridge Regression
- Ridge estimator: β̂_ridge = **(X^T X + λI)^{-1} X^T y** — closed form always exists.
- Ridge is **biased** (intentionally — to reduce variance).
- Ridge **never sets coefficients exactly to zero** — it shrinks toward zero but not to it.
- Ridge **does not perform variable selection**.
- As λ → 0: ridge → OLS. As λ → ∞: all β̂ → 0.
- Effective df(λ) = trace(X(X^T X + λI)^{-1} X^T), which decreases as λ increases.

### Cp Statistic
- Cp = err_train + 2(d/N)σ̂²_e — penalizes training error by model complexity.
- σ̂²_e is the noise estimate from a **low-bias model** (not from the current model).
- **Minimize** Cp to select the best model.

### AIC
- General: AIC = -(2/N) logL + 2(d/N)
- Gaussian: AIC(λ) = err(λ) + 2(d(λ)/N)σ̂²_e
- For Gaussian case, **Cp and AIC are identical**.
- AIC is asymptotically equivalent to **leave-one-out cross-validation** (Stone 1977).

### BIC
- General: BIC = -2 logL + log(N)d
- BIC penalizes complexity **more** than AIC (uses log(N) instead of 2).
- For large n: BIC selects the **correct model** with probability → 1 (consistent).
- For small n: BIC tends to pick **too simple** a model.
- AIC tends to pick **too complex** a model asymptotically.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| `EPE = σ² + Bias² + Var` | Bias-variance decomposition | Any EPE question |
| `β̂_OLS = (X^T X)^{-1} X^T y` | OLS closed form | Baseline regression |
| `β̂_ridge = (X^T X + λI)^{-1} X^T y` | Ridge closed form | Regularized regression |
| `Cp = err + 2(d/N)σ̂²_e` | In-sample model selection | Choosing between models |
| `AIC = -(2/N)logL + 2d/N` | Log-likelihood-based criterion | General AIC |
| `BIC = -2logL + log(N)d` | Bayesian criterion | Large sample model selection |
| `df(S) = trace(S)` | Effective parameters of linear smoother | Ridge df, AIC(λ) |
| `FWER = 1 - (1-α)^M` | Family-wise error rate for M tests | Multiple testing |

---

## Common Traps (wrong answers in exams)

- ❌ Ridge has no closed-form solution → ✓ Ridge DOES have a closed form: (X^T X + λI)^{-1} X^T y
- ❌ Ridge performs variable selection (sets some β to zero) → ✓ Ridge only SHRINKS toward zero; Lasso sets to zero
- ❌ OLS is better than ridge in all cases → ✓ OLS has higher variance when predictors are correlated; ridge trades bias for lower variance
- ❌ Adding more variables always improves Cp → ✓ More variables lower training error but increase the penalty; Cp can increase
- ❌ Cp and AIC are different for Gaussian models → ✓ They are IDENTICAL for Gaussian models
- ❌ AIC is more conservative than BIC → ✓ BIC is MORE conservative (larger penalty = simpler models)
- ❌ BIC is better for small samples → ✓ BIC selects too simple models for small n; AIC may be preferable
- ❌ AIC and k-fold CV are asymptotically equivalent → ✓ AIC is asymptotically equivalent to LEAVE-ONE-OUT CV, not k-fold
- ❌ σ̂²_e in Cp should come from the current model → ✓ σ̂²_e is the noise from a LOW-BIAS (e.g., full OLS) model — it is fixed
- ❌ The bias-variance decomposition has two terms → ✓ THREE terms: irreducible noise + bias² + variance
- ❌ Increasing λ in ridge increases df(λ) → ✓ Increasing λ DECREASES df(λ) from p toward 0

---

## Quick Decision Rules

- If question says "closed-form solution for regularized regression" → Ridge (not Lasso)
- If question says "sets coefficients to exactly zero" → Lasso (not Ridge)
- If Gaussian model, comparing Cp vs AIC → they are the same thing
- If n is very large and want consistent selection → use BIC
- If comparing AIC vs LOO-CV → asymptotically equivalent (Stone 1977)
- If λ increases in ridge → bias increases, variance decreases, df decreases
- If noise floor σ̂²_e is unknown → estimate it from the full (low-bias) model
- If p > n → OLS fails (non-invertible), Ridge still works (add λI makes it invertible)
