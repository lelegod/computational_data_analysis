# Week 1 — Regression & Bias-Variance Tradeoff (Exam Focus)

## Must-Know Facts

### EPE / Bias-Variance
- $\text{EPE} = \sigma^2 + \text{Bias}^2 + \text{Variance}$ — three terms, always.
- $\sigma^2$ is **irreducible** noise — no model can reduce it.
- Complex models have **low bias, high variance**; simple models have **high bias, low variance**.
- Training error always **decreases** with complexity; test error forms a **U-shape**.
- The decomposition holds at a specific point $x_0$ and averages over both $y$ and the training set $D$.

### OLS
- OLS estimator: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ — requires $\mathbf{X}^T \mathbf{X}$ to be invertible.
- OLS is **unbiased**: $E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$.
- OLS has **minimum variance** among all linear unbiased estimators (Gauss-Markov).
- OLS has $\text{df} = p$ (number of predictors).
- Hat matrix: $\hat{\mathbf{Y}} = \mathbf{S}\mathbf{Y}$ where $\mathbf{S} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$.

### Ridge Regression
- Ridge estimator: $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$ — closed form always exists.
- Ridge is **biased** (intentionally — to reduce variance).
- Ridge **never sets coefficients exactly to zero** — it shrinks toward zero but not to it.
- Ridge **does not perform variable selection**.
- As $\lambda \to 0$: ridge $\to$ OLS. As $\lambda \to \infty$: all $\hat{\beta} \to 0$.
- Effective $\text{df}(\lambda) = \text{trace}\!\left(\mathbf{X}(\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T\right)$, which decreases as $\lambda$ increases.

### Cp Statistic
- $C_p = \widehat{\text{err}}_{\text{train}} + 2\frac{d}{N}\hat{\sigma}^2_e$ — penalizes training error by model complexity.
- $\hat{\sigma}^2_e$ is the noise estimate from a **low-bias model** (not from the current model).
- **Minimize** $C_p$ to select the best model.

### AIC
- General: $\text{AIC} = -\frac{2}{N} \log L + \frac{2d}{N}$
- Gaussian: $\text{AIC}(\lambda) = \widehat{\text{err}}(\lambda) + 2\frac{d(\lambda)}{N}\hat{\sigma}^2_e$
- For Gaussian case, **Cp and AIC are identical**.
- AIC is asymptotically equivalent to **leave-one-out cross-validation** (Stone 1977).

### BIC
- General: $\text{BIC} = -2 \log L + \log(N)\,d$
- BIC penalizes complexity **more** than AIC (uses $\log(N)$ instead of $2$).
- For large $n$: BIC selects the **correct model** with probability $\to 1$ (consistent).
- For small $n$: BIC tends to pick **too simple** a model.
- AIC tends to pick **too complex** a model asymptotically.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $\text{EPE} = \sigma^2 + \text{Bias}^2 + \text{Var}$ | Bias-variance decomposition | Any EPE question |
| $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ | OLS closed form | Baseline regression |
| $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$ | Ridge closed form | Regularized regression |
| $C_p = \widehat{\text{err}} + 2\frac{d}{N}\hat{\sigma}^2_e$ | In-sample model selection | Choosing between models |
| $\text{AIC} = -\frac{2}{N}\log L + \frac{2d}{N}$ | Log-likelihood-based criterion | General AIC |
| $\text{BIC} = -2\log L + \log(N)\,d$ | Bayesian criterion | Large sample model selection |
| $\text{df}(\mathbf{S}) = \text{trace}(\mathbf{S})$ | Effective parameters of linear smoother | Ridge df, AIC($\lambda$) |
| $\text{FWER} = 1 - (1-\alpha)^M$ | Family-wise error rate for $M$ tests | Multiple testing |

---

## Common Traps (wrong answers in exams)

- ❌ Ridge has no closed-form solution → ✓ Ridge DOES have a closed form: $(\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$
- ❌ Ridge performs variable selection (sets some $\beta$ to zero) → ✓ Ridge only SHRINKS toward zero; Lasso sets to zero
- ❌ OLS is better than ridge in all cases → ✓ OLS has higher variance when predictors are correlated; ridge trades bias for lower variance
- ❌ Adding more variables always improves $C_p$ → ✓ More variables lower training error but increase the penalty; $C_p$ can increase
- ❌ Cp and AIC are different for Gaussian models → ✓ They are IDENTICAL for Gaussian models
- ❌ AIC is more conservative than BIC → ✓ BIC is MORE conservative (larger penalty = simpler models)
- ❌ BIC is better for small samples → ✓ BIC selects too simple models for small $n$; AIC may be preferable
- ❌ AIC and k-fold CV are asymptotically equivalent → ✓ AIC is asymptotically equivalent to LEAVE-ONE-OUT CV, not k-fold
- ❌ $\hat{\sigma}^2_e$ in $C_p$ should come from the current model → ✓ $\hat{\sigma}^2_e$ is the noise from a LOW-BIAS (e.g., full OLS) model — it is fixed
- ❌ The bias-variance decomposition has two terms → ✓ THREE terms: irreducible noise + $\text{bias}^2$ + variance
- ❌ Increasing $\lambda$ in ridge increases $\text{df}(\lambda)$ → ✓ Increasing $\lambda$ DECREASES $\text{df}(\lambda)$ from $p$ toward $0$

---

## Quick Decision Rules

- If question says "closed-form solution for regularized regression" → Ridge (not Lasso)
- If question says "sets coefficients to exactly zero" → Lasso (not Ridge)
- If Gaussian model, comparing $C_p$ vs AIC → they are the same thing
- If $n$ is very large and want consistent selection → use BIC
- If comparing AIC vs LOO-CV → asymptotically equivalent (Stone 1977)
- If $\lambda$ increases in ridge → bias increases, variance decreases, $\text{df}$ decreases
- If noise floor $\hat{\sigma}^2_e$ is unknown → estimate it from the full (low-bias) model
- If $p > n$ → OLS fails (non-invertible), Ridge still works (add $\lambda\mathbf{I}$ makes it invertible)
