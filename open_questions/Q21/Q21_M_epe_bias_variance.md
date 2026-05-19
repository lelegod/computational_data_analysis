# Q21-M — EPE Decomposition and Bias-Variance Tradeoff
> Week 1. The most fundamental theorem in the course. Could be asked to derive the decomposition.

---

## The Expected Prediction Error (EPE)

For a regression problem with squared loss:
$$\text{EPE}(x_0) = E\left[(y_0 - \hat{f}(x_0))^2\right]$$

where the expectation is over both the randomness in $y_0 = f(x_0)+\varepsilon$ (new test point) and in $\hat{f}$ (randomness from training data).

---

## The Decomposition

$$\boxed{\text{EPE}(x_0) = \sigma^2 + \text{Bias}^2[\hat{f}(x_0)] + \text{Var}[\hat{f}(x_0)]}$$

Where:
- $\sigma^2 = \text{Var}(\varepsilon)$: **irreducible noise** — cannot be reduced by any model
- $\text{Bias}^2[\hat{f}(x_0)] = (f(x_0) - E[\hat{f}(x_0)])^2$: systematic error from wrong assumptions
- $\text{Var}[\hat{f}(x_0)] = E[(\hat{f}(x_0) - E[\hat{f}(x_0)])^2]$: sensitivity to training data

---

## Full Derivation

Let $A = \varepsilon$, $B = f(x_0) - E[\hat{f}(x_0)]$ (bias), $C = E[\hat{f}(x_0)] - \hat{f}(x_0)$ (centered estimator).

Then $y_0 - \hat{f}(x_0) = \varepsilon + f(x_0) - \hat{f}(x_0) = A + B + C$.

$$\text{EPE} = E[(A+B+C)^2] = E[A^2] + B^2 + E[C^2] + 2BE[A] + 2E[AC] + 2BE[C]$$

**Why cross-terms vanish**:
- $2BE[A] = 2B\cdot E[\varepsilon] = 0$ (noise has zero mean)
- $2E[AC] = 2E[\varepsilon\cdot(E[\hat{f}]-\hat{f}(x_0))] = 0$ ($\varepsilon$ is independent of training data → independent of $\hat{f}$)
- $2BE[C] = 2B\cdot E[E[\hat{f}]-\hat{f}(x_0)] = 2B\cdot 0 = 0$ (by definition of $E[\hat{f}]$)

Remaining terms: $E[A^2] = \sigma^2$, $B^2 = \text{Bias}^2$, $E[C^2] = \text{Var}[\hat{f}]$.

---

## What Each Term Means

**Irreducible noise $\sigma^2$**:
- Fundamental randomness in the response (measurement error, unmeasured variables)
- Even the true $f$ cannot predict $y_0$ perfectly
- Lower bound on EPE for any model

**Bias$^2$**:
- How far the average prediction is from the truth
- Caused by: wrong model family (e.g., linear model for nonlinear truth), strong regularization
- High-bias examples: linear regression on nonlinear data, ridge with large $\lambda$, shallow trees

**Variance**:
- How much predictions change across different training sets
- Caused by: too complex model memorizes training noise, weak regularization
- High-variance examples: deep unpruned trees, KNN with $K=1$, OLS with many correlated features

---

## Bias-Variance Tradeoff

As model complexity increases:
- Bias decreases (model can fit more shapes)
- Variance increases (model fits noise)
- Optimal complexity = minimum EPE (U-shaped curve)

| Method | Bias | Variance |
|--------|------|---------|
| OLS | Low (if model correct) | Moderate |
| Ridge ($\lambda\uparrow$) | Increases | Decreases |
| Lasso ($\lambda\uparrow$) | Increases | Decreases |
| KNN ($K=1$) | Low | High |
| KNN ($K=N$) | High (= overall mean) | Zero |
| Deep tree | Low | High |
| Shallow tree | High | Low |
| Bagging deep trees | Low | Reduced |
| Boosting stumps | Reduced iteratively | Low per step |

---

## Bagging Reduces Variance (Not Bias)

For $B$ trees with individual variance $\sigma^2$ and pairwise correlation $\rho$:
$$\text{Var}(\text{average}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B\to\infty$: variance $\to\rho\sigma^2$. The bias of the ensemble = bias of one tree (unchanged).

**Implication**: bagging only helps when base learners have high variance (deep trees, low-K KNN). Using high-bias base learners (stumps) with bagging does not help — the bias floor $\rho\sigma^2$ remains high.

---

## Additional Possible Exam Questions

**Q: Can you reduce all three terms simultaneously?**
No. $\sigma^2$ is irreducible regardless of model. Bias and variance trade off against each other. The best you can do is minimize Bias$^2$ + Var, accepting $\sigma^2$ as the floor.

**Q: Why does OLS have zero bias (under correct specification)?**
If the true model is $y = X\beta + \varepsilon$ with $E[\varepsilon|X]=0$, then $E[\hat{\beta}_\text{OLS}] = E[(X^TX)^{-1}X^Ty] = \beta + (X^TX)^{-1}X^TE[\varepsilon] = \beta$. The estimator is unbiased. But OLS has high variance when predictors are correlated or $p$ is large → ridge trades a little bias for a large variance reduction, lowering EPE overall.

**Q: What does the training error measure vs EPE?**
Training error = $\frac{1}{N}\sum_i(y_i-\hat{f}(x_i))^2$ — evaluated on the same data used to fit $\hat{f}$. This underestimates EPE because $\hat{f}$ was fitted to minimize it. EPE measures performance on new, unseen data — the relevant quantity for deployment. The gap between training error and EPE grows with model complexity (overfitting).

**Q: What is the optimism of the training error?**
$\text{EPE} \approx \text{Training error} + \frac{2}{N}\sum_j\text{Cov}(\hat{y}_j, y_j)$. This is the basis for AIC ($2p/N$ optimism correction) and $C_p$ statistic. More parameters → more covariance between predictions and labels → larger optimism → larger gap between training error and EPE.
