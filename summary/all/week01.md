# Week 1 — Regression, Bias-Variance Tradeoff, Ridge Regression

## Overview
Week 1 establishes the statistical learning framework for supervised regression. It introduces the Expected Prediction Error (EPE) decomposition into irreducible noise, bias, and variance, and shows how regularization (ridge regression) manages the bias-variance tradeoff. The lecture also covers model selection criteria (Cp, AIC, BIC) and their relationship to cross-validation.

---

## Part I — The Statistical Learning Setup

### Key Concepts
- **Supervised learning**: Given training data D = {(x_i, y_i), i=1,...,N}, learn f such that y ≈ f(x).
- **Model**: y = f(x) + ε, where ε is irreducible noise with E(ε) = 0, E(ε²) = σ².
- **Goal**: Minimize prediction error on new (unseen) data — not just training data.
- **Prediction**: ŷ = f̂(x₀; D) — the fitted model applied to a new point x₀.

### The Expected Prediction Error (EPE)
The EPE at a new point x₀ is:

- **Formula**: `EPE = E(y - f̂)² = σ² + Bias²(f̂) + Var(f̂)`

Where the expectation is over both y and the training set D.

**Three components:**
| Term | Name | Meaning |
|------|------|---------|
| σ² | Irreducible noise | Cannot be reduced — inherent noise in y |
| (E[f̂] - f)² | Bias² | How far the average prediction is from truth |
| E[(f̂ - E[f̂])²] | Variance | How much the model fluctuates across training sets |

**EPE Derivation sketch** (from EPEderivation.pdf):
Given y = f + ε, E(ε)=0, E(ε²)=σ²:
```
E(y - f̂)² = E[(y - f + f - E[f̂] + E[f̂] - f̂)²]
           = σ² + (E[f̂] - f)² + E[(f̂ - E[f̂])²]
```
Steps use: E(y) = f, linearity of expectation, E(X²) = E(X)² for constants, cross-terms vanish.

### Bias-Variance Tradeoff
- **Complex models** (many parameters): Low bias, high variance.
- **Simple models** (few parameters): High bias, low variance.
- **Goal**: Find the sweet spot that minimizes total EPE.
- As model complexity increases: training error always decreases, test error forms a U-shape.

---

## Part II — Ordinary Least Squares (OLS)

### Key Concepts
- OLS minimizes the residual sum of squares (RSS): `RSS = ||y - Xβ||²`
- **OLS estimator**: `β̂_OLS = (X^T X)^{-1} X^T y`
- OLS is unbiased: E[β̂] = β
- OLS has the smallest variance among all linear unbiased estimators (Gauss-Markov theorem)

### Problems with OLS
- When p is large or predictors are correlated, (X^T X) may be nearly singular → large variance.
- In high-dimensional settings (p > n), (X^T X) is not invertible.
- OLS uses all features: no automatic variable selection.

### Linear Fitting — Hat Matrix
- Predictions: `Ŷ = X(X^T X)^{-1} X^T Y = SY` where S = X(X^T X)^{-1} X^T is the hat matrix.
- Effective degrees of freedom: `df(S) = trace(S)`
- For OLS with all p features: df = p (trace of the hat matrix equals number of parameters).

---

## Part III — Ridge Regression

### Key Concepts
- Ridge adds an L₂ penalty to the OLS objective to shrink coefficients toward zero.
- Controls variance at the cost of introducing bias.
- Particularly useful when predictors are correlated or p is large.

### Objective Function
- **Penalized form**: `min_β (Y - Xβ)^T(Y - Xβ) + λ||β||²₂`
- **Constrained form**: `min_β (Y - Xβ)^T(Y - Xβ)` subject to `||β||²₂ ≤ s`
- λ and s are in one-to-one correspondence: larger λ = smaller s = more shrinkage.

### Closed-Form Solution (derived in RidgeRegressionExplicitEstimate.pdf)
Take derivative of penalized objective and set to zero:
```
∂/∂β [(y - Xβ)^T(y - Xβ) + λβ^Tβ] = -2X^Ty + 2X^TXβ + 2λIβ = 0
```
Rearranging:
- **Ridge estimator**: `β̂_ridge = (X^T X + λI)^{-1} X^T y`

Key derivative rules used:
- ∂/∂β (b^T a) = a
- ∂/∂β (β^T A β) = (A + A^T)β = 2Aβ when A is symmetric

### Properties of Ridge
- **Always invertible**: (X^T X + λI) is positive definite for λ > 0, so solution always exists.
- **Biased**: E[β̂_ridge] ≠ β in general.
- **Lower variance** than OLS.
- **Shrinks all coefficients** toward zero but never exactly to zero.
- **Does not perform variable selection** — all predictors remain in model.

### Effective Degrees of Freedom for Ridge
Since Ŷ = X(X^T X + λI)^{-1} X^T Y = S_λ Y (linear fitting method):
- `df(λ) = trace(S_λ) = trace(X(X^T X + λI)^{-1} X^T)`
- As λ → 0: df → p (reduces to OLS)
- As λ → ∞: df → 0 (shrinks to null model)

### Choosing λ
- Use cross-validation to find λ that minimizes prediction error.
- Also use information criteria (Cp, AIC, BIC) with d = df(λ).

---

## Part IV — Model Selection Criteria (In-Sample Error Estimation)

These criteria avoid splitting the data by adding a penalty for model complexity.

### The Cp Statistic
- **Purpose**: Estimate the expected test error without a separate test set.
- **Formula**: `Cp = err_train + 2 * (d/N) * σ̂²_e`
  - `err_train = (1/N) Σ(yi - xi β̂)²` — average training error
  - `d` — number of parameters (complexity) in current model
  - `σ̂²_e` — estimated noise floor (MSE from a low-bias model, e.g., OLS with all features)
  - `N` — number of training samples
- **Selection rule**: Choose model that **minimizes** Cp.
- **Tradeoff**: Adding a variable always decreases training error, but increases the penalty term `2d/N * σ̂²_e`.

### AIC — Akaike Information Criterion
- **Used when**: The loss function is log-likelihood (not just squared error).
- **General form**: `AIC = -(2/N) logL + 2(d/N)`
  - `logL` = maximized log-likelihood
  - `d` = number of parameters
- **For Gaussian case with tuning parameter λ**: `AIC(λ) = err(λ) + 2(d(λ)/N) σ̂²_e`
  - `d(λ)` = effective number of parameters at regularization λ
- For the Gaussian model, **Cp and AIC are identical**.
- `d(λ)` is the **effective** number of parameters (trace of hat matrix).

### BIC — Bayes Information Criterion
- **Motivated by**: Bayesian model selection — selects model with highest posterior probability.
- **General form**: `BIC = -2 logL + log(N) d`
- **For Gaussian with λ**: `BIC(λ) = (N/σ̂²_e) * (err(λ) + log(N)(d(λ)/N) σ̂²_e)`
- **Selection rule**: Choose model that **minimizes** BIC.
- **Difference from AIC**: BIC uses `log(N)` penalty instead of `2`, making it more conservative (penalizes complexity more heavily).

### AIC vs BIC Comparison
| Property | AIC | BIC |
|----------|-----|-----|
| Penalty | 2d/N | log(N)d/N |
| Large n behavior | Tends to pick too complex | Selects correct model with probability → 1 |
| Small n behavior | Better | May pick too simple |
| Asymptotic | Equivalent to LOO-CV | Consistent model selection |

- Stone (1977): AIC and leave-one-out cross-validation are **asymptotically equivalent**.
- Both AIC and LOO-CV tend to choose models that are too complex.
- For n → ∞ with the true model in the candidate set: BIC selects the correct model (consistent), AIC does not.

---

## Part V — Cross-Validation (Review/Context)

### K-Fold Cross-Validation
- Split data into K folds. Train on K-1 folds, test on the held-out fold. Rotate K times.
- Average test errors over K folds for overall CV error.
- Common choices: K = 5 or K = 10.
- **LOO-CV**: K = N; most expensive but least bias.

### CV for Ridge Regression
- Compute CV error for a grid of λ values.
- Select λ with minimum CV error.
- Both Cp/AIC and CV identify similar optimal λ (AIC ≈ LOO-CV asymptotically).

---

## Part VI — Effective Parameters and Geometry

### Effective Number of Parameters (df)
For any linear smoother Ŷ = SY:
- `df(S) = trace(S)`
- OLS: S = X(X^T X)^{-1} X^T, trace(S) = p
- Ridge: S = X(X^T X + λI)^{-1} X^T, trace(S_λ) decreases from p to 0 as λ increases

### Geometry of Ridge vs OLS
- OLS solution is the unconstrained minimum of RSS — sits at the bottom of the elliptical contours.
- Ridge constrains β to lie within a sphere (||β||² ≤ s): solution is where the RSS ellipsoid first touches the sphere.
- The sphere constraint has no corners → solution is rarely exactly zero on any coordinate.

---

## Summary: When to Use What

| Method | Use when | Key property |
|--------|----------|--------------|
| OLS | n >> p, no multicollinearity | Unbiased, minimum variance linear unbiased |
| Ridge | Multicollinearity, large p | Closed-form, shrinks all, never zeros |
| Cp / AIC | Gaussian errors, want in-sample criterion | Equivalent to each other |
| BIC | Want consistent model selection | Penalizes complexity more at large n |
| CV | General purpose, any loss | Model-free, computationally intensive |
