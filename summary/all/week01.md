# Week 1 — Regression, Bias-Variance Tradeoff, Ridge Regression

## Overview
Week 1 establishes the statistical learning framework for supervised regression. It introduces the Expected Prediction Error (EPE) decomposition into irreducible noise, bias, and variance, and shows how regularization (ridge regression) manages the bias-variance tradeoff. The lecture also covers model selection criteria (Cp, AIC, BIC) and their relationship to cross-validation.

---

## Part I — The Statistical Learning Setup

### Key Concepts
- **Supervised learning**: Given training data $D = \{(x_i, y_i),\ i=1,\ldots,N\}$, learn $f$ such that $y \approx f(x)$.
- **Model**: $y = f(x) + \varepsilon$, where $\varepsilon$ is irreducible noise with $E(\varepsilon) = 0$, $E(\varepsilon^2) = \sigma^2$.
- **Goal**: Minimize prediction error on new (unseen) data — not just training data.
- **Prediction**: $\hat{y} = \hat{f}(x_0; D)$ — the fitted model applied to a new point $x_0$.

### The Expected Prediction Error (EPE)
The EPE at a new point $x_0$ is:

- **Formula**: $\text{EPE} = E(y - \hat{f})^2 = \sigma^2 + \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f})$

Where the expectation is over both $y$ and the training set $D$.

**Three components:**
| Term | Name | Meaning |
|------|------|---------|
| $\sigma^2$ | Irreducible noise | Cannot be reduced — inherent noise in $y$ |
| $(E[\hat{f}] - f)^2$ | Bias² | How far the average prediction is from truth |
| $E[(\hat{f} - E[\hat{f}])^2]$ | Variance | How much the model fluctuates across training sets |

**EPE Derivation sketch** (from EPEderivation.pdf):
Given $y = f + \varepsilon$, $E(\varepsilon)=0$, $E(\varepsilon^2)=\sigma^2$:

$$E(y - \hat{f})^2 = E[(y - f + f - E[\hat{f}] + E[\hat{f}] - \hat{f})^2]$$
$$= \sigma^2 + (E[\hat{f}] - f)^2 + E[(\hat{f} - E[\hat{f}])^2]$$

Steps use: $E(y) = f$, linearity of expectation, $E(X^2) = E(X)^2$ for constants, cross-terms vanish.

### Bias-Variance Tradeoff
- **Complex models** (many parameters): Low bias, high variance.
- **Simple models** (few parameters): High bias, low variance.
- **Goal**: Find the sweet spot that minimizes total EPE.
- As model complexity increases: training error always decreases, test error forms a U-shape.

---

## Part II — Ordinary Least Squares (OLS)

### Key Concepts
- OLS minimizes the residual sum of squares (RSS): $\text{RSS} = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$
- **OLS estimator**: $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
- OLS is unbiased: $E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$
- OLS has the smallest variance among all linear unbiased estimators (Gauss-Markov theorem)

### Problems with OLS
- When $p$ is large or predictors are correlated, $(\mathbf{X}^T \mathbf{X})$ may be nearly singular → large variance.
- In high-dimensional settings ($p > n$), $(\mathbf{X}^T \mathbf{X})$ is not invertible.
- OLS uses all features: no automatic variable selection.

### Linear Fitting — Hat Matrix
- Predictions: $\hat{\mathbf{Y}} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{S}\mathbf{Y}$ where $\mathbf{S} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$ is the hat matrix.
- Effective degrees of freedom: $\text{df}(\mathbf{S}) = \text{trace}(\mathbf{S})$
- For OLS with all $p$ features: $\text{df} = p$ (trace of the hat matrix equals number of parameters).

---

## Part III — Ridge Regression

### Key Concepts
- Ridge adds an $L_2$ penalty to the OLS objective to shrink coefficients toward zero.
- Controls variance at the cost of introducing bias.
- Particularly useful when predictors are correlated or $p$ is large.

### Objective Function
- **Penalized form**: $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) + \lambda\|\boldsymbol{\beta}\|_2^2$
- **Constrained form**: $\min_{\boldsymbol{\beta}}\ (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$ subject to $\|\boldsymbol{\beta}\|_2^2 \leq s$
- $\lambda$ and $s$ are in one-to-one correspondence: larger $\lambda$ = smaller $s$ = more shrinkage.

### Closed-Form Solution (derived in RidgeRegressionExplicitEstimate.pdf)
Take derivative of penalized objective and set to zero:

$$\frac{\partial}{\partial \boldsymbol{\beta}} \left[(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda\boldsymbol{\beta}^T\boldsymbol{\beta}\right] = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + 2\lambda\mathbf{I}\boldsymbol{\beta} = 0$$

Rearranging:
- **Ridge estimator**: $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$

Key derivative rules used:
- $\frac{\partial}{\partial \boldsymbol{\beta}} (\mathbf{b}^T \mathbf{a}) = \mathbf{a}$
- $\frac{\partial}{\partial \boldsymbol{\beta}} (\boldsymbol{\beta}^T \mathbf{A} \boldsymbol{\beta}) = (\mathbf{A} + \mathbf{A}^T)\boldsymbol{\beta} = 2\mathbf{A}\boldsymbol{\beta}$ when $\mathbf{A}$ is symmetric

### Properties of Ridge
- **Always invertible**: $(\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})$ is positive definite for $\lambda > 0$, so solution always exists.
- **Biased**: $E[\hat{\boldsymbol{\beta}}_{\text{ridge}}] \neq \boldsymbol{\beta}$ in general.
- **Lower variance** than OLS.
- **Shrinks all coefficients** toward zero but never exactly to zero.
- **Does not perform variable selection** — all predictors remain in model.

### Effective Degrees of Freedom for Ridge
Since $\hat{\mathbf{Y}} = \mathbf{X}(\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{S}_\lambda \mathbf{Y}$ (linear fitting method):
- $\text{df}(\lambda) = \text{trace}(\mathbf{S}_\lambda) = \text{trace}\!\left(\mathbf{X}(\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T\right)$
- As $\lambda \to 0$: $\text{df} \to p$ (reduces to OLS)
- As $\lambda \to \infty$: $\text{df} \to 0$ (shrinks to null model)

### Choosing $\lambda$
- Use cross-validation to find $\lambda$ that minimizes prediction error.
- Also use information criteria (Cp, AIC, BIC) with $d = \text{df}(\lambda)$.

---

## Part IV — Model Selection Criteria (In-Sample Error Estimation)

These criteria avoid splitting the data by adding a penalty for model complexity.

### The Cp Statistic
- **Purpose**: Estimate the expected test error without a separate test set.
- **Formula**: $C_p = \widehat{\text{err}}_{\text{train}} + 2 \cdot \frac{d}{N} \cdot \hat{\sigma}^2_e$
  - $\widehat{\text{err}}_{\text{train}} = \frac{1}{N} \sum_i (y_i - x_i^T \hat{\boldsymbol{\beta}})^2$ — average training error
  - $d$ — number of parameters (complexity) in current model
  - $\hat{\sigma}^2_e$ — estimated noise floor (MSE from a low-bias model, e.g., OLS with all features)
  - $N$ — number of training samples
- **Selection rule**: Choose model that **minimizes** $C_p$.
- **Tradeoff**: Adding a variable always decreases training error, but increases the penalty term $2d/N \cdot \hat{\sigma}^2_e$.

### AIC — Akaike Information Criterion
- **Used when**: The loss function is log-likelihood (not just squared error).
- **General form**: $\text{AIC} = -\frac{2}{N} \log L + \frac{2d}{N}$
  - $\log L$ = maximized log-likelihood
  - $d$ = number of parameters
- **For Gaussian case with tuning parameter $\lambda$**: $\text{AIC}(\lambda) = \widehat{\text{err}}(\lambda) + 2\frac{d(\lambda)}{N} \hat{\sigma}^2_e$
  - $d(\lambda)$ = effective number of parameters at regularization $\lambda$
- For the Gaussian model, **Cp and AIC are identical**.
- $d(\lambda)$ is the **effective** number of parameters (trace of hat matrix).

### BIC — Bayes Information Criterion
- **Motivated by**: Bayesian model selection — selects model with highest posterior probability.
- **General form**: $\text{BIC} = -2 \log L + \log(N)\, d$
- **For Gaussian with $\lambda$**: $\text{BIC}(\lambda) = \frac{N}{\hat{\sigma}^2_e} \left(\widehat{\text{err}}(\lambda) + \log(N)\frac{d(\lambda)}{N} \hat{\sigma}^2_e\right)$
- **Selection rule**: Choose model that **minimizes** BIC.
- **Difference from AIC**: BIC uses $\log(N)$ penalty instead of $2$, making it more conservative (penalizes complexity more heavily).

### AIC vs BIC Comparison
| Property | AIC | BIC |
|----------|-----|-----|
| Penalty | $2d/N$ | $\log(N)\,d/N$ |
| Large $n$ behavior | Tends to pick too complex | Selects correct model with probability $\to 1$ |
| Small $n$ behavior | Better | May pick too simple |
| Asymptotic | Equivalent to LOO-CV | Consistent model selection |

- Stone (1977): AIC and leave-one-out cross-validation are **asymptotically equivalent**.
- Both AIC and LOO-CV tend to choose models that are too complex.
- For $n \to \infty$ with the true model in the candidate set: BIC selects the correct model (consistent), AIC does not.

---

## Part V — Cross-Validation (Review/Context)

### K-Fold Cross-Validation
- Split data into $K$ folds. Train on $K-1$ folds, test on the held-out fold. Rotate $K$ times.
- Average test errors over $K$ folds for overall CV error.
- Common choices: $K = 5$ or $K = 10$.
- **LOO-CV**: $K = N$; most expensive but least bias.

### CV for Ridge Regression
- Compute CV error for a grid of $\lambda$ values.
- Select $\lambda$ with minimum CV error.
- Both $C_p$/AIC and CV identify similar optimal $\lambda$ (AIC $\approx$ LOO-CV asymptotically).

---

## Part VI — Effective Parameters and Geometry

### Effective Number of Parameters (df)
For any linear smoother $\hat{\mathbf{Y}} = \mathbf{S}\mathbf{Y}$:
- $\text{df}(\mathbf{S}) = \text{trace}(\mathbf{S})$
- OLS: $\mathbf{S} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$, $\text{trace}(\mathbf{S}) = p$
- Ridge: $\mathbf{S} = \mathbf{X}(\mathbf{X}^T \mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T$, $\text{trace}(\mathbf{S}_\lambda)$ decreases from $p$ to $0$ as $\lambda$ increases

### Geometry of Ridge vs OLS
- OLS solution is the unconstrained minimum of RSS — sits at the bottom of the elliptical contours.
- Ridge constrains $\boldsymbol{\beta}$ to lie within a sphere ($\|\boldsymbol{\beta}\|^2 \leq s$): solution is where the RSS ellipsoid first touches the sphere.
- The sphere constraint has no corners → solution is rarely exactly zero on any coordinate.

---

## Summary: When to Use What

| Method | Use when | Key property |
|--------|----------|--------------|
| OLS | $n \gg p$, no multicollinearity | Unbiased, minimum variance linear unbiased |
| Ridge | Multicollinearity, large $p$ | Closed-form, shrinks all, never zeros |
| $C_p$ / AIC | Gaussian errors, want in-sample criterion | Equivalent to each other |
| BIC | Want consistent model selection | Penalizes complexity more at large $n$ |
| CV | General purpose, any loss | Model-free, computationally intensive |
