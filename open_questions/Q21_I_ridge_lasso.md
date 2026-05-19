# Q21-I — Ridge vs Lasso vs Elastic Net
> Week 1/2/3. Most-tested topic in MC; strong Q21 candidate as comparison question.

---

## The General Framework: Penalized Regression

All three add a penalty to OLS to shrink coefficients and prevent overfitting:
$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 + \lambda \cdot \text{Penalty}(\beta)$$

| Method | Penalty | Shape |
|--------|---------|-------|
| Ridge | $\lambda\|\beta\|_2^2 = \lambda\sum_j\beta_j^2$ | $L_2$ — sphere |
| Lasso | $\lambda\|\beta\|_1 = \lambda\sum_j|\beta_j|$ | $L_1$ — diamond |
| Elastic Net | $\lambda[\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2]$ | Compromise |

---

## Ridge Regression

**Closed-form solution**:
$$\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1}X^Ty$$

**Key properties**:
- Invertibility: $X^TX + \lambda I$ is always positive definite → solution always exists even when $p > n$
- **Biased**: $E[\hat{\beta}_\text{ridge}] \neq \beta$ for $\lambda > 0$
- **Shrinks** all coefficients toward zero but **never exactly to zero**
- As $\lambda\to\infty$: all $\hat{\beta}_j \to 0$. As $\lambda\to0$: $\hat{\beta}_\text{ridge} \to \hat{\beta}_\text{OLS}$
- SVD view: $\hat{\beta}_\text{ridge} = \sum_j \frac{d_j^2}{d_j^2+\lambda}v_j\frac{u_j^Ty}{d_j}$ (shrinks each PC direction by $d_j^2/(d_j^2+\lambda)$)

**Geometric intuition**: the $L_2$ constraint region is a sphere — the solution (where the elliptical OLS contours touch the sphere) almost never lands on an axis → no exact zeros.

---

## Lasso

**No closed-form** (non-differentiable at $\beta_j = 0$).

**Fitted by**:
- LARS (Least Angle Regression) — computes full solution path efficiently
- Coordinate descent — cycle through each $\beta_j$ and apply soft-thresholding:
$$\hat{\beta}_j \leftarrow \text{sign}(\tilde{\beta}_j)\max(|\tilde{\beta}_j| - \lambda, 0)$$

where $\tilde{\beta}_j$ is the partial residual OLS estimate.

**Key properties**:
- **Biased** (like ridge)
- **Sparse**: sets some $\hat{\beta}_j = 0$ exactly → performs **variable selection**
- As $\lambda\to\infty$: all $\hat{\beta}_j \to 0$. As $\lambda\to0$: $\hat{\beta}_\text{lasso} \to \hat{\beta}_\text{OLS}$
- Solution path: coefficients enter the model one at a time as $\lambda$ decreases (LARS path)

**Geometric intuition**: the $L_1$ constraint region is a diamond with corners on the axes — the OLS contours are very likely to touch a corner → exact zero → sparsity.

**Limitation**: among correlated predictors, Lasso picks one arbitrarily and zeros the rest. Ridge keeps all (shrinks proportionally).

---

## Elastic Net

$$\hat{\beta}_\text{EN} = \arg\min_\beta \|y-X\beta\|^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$$

- $\alpha = \lambda_1/(\lambda_1+\lambda_2)$ controls $L_1$/$L_2$ balance
- $\alpha=1$: Lasso. $\alpha=0$: Ridge.
- Solves Lasso's correlated-predictor problem: correlated variables are grouped together (kept or dropped together)
- Still has closed-form solution after the soft-thresholding step

---

## Comparison Table

| Property | OLS | Ridge | Lasso | Elastic Net |
|----------|-----|-------|-------|-------------|
| Penalty | None | $L_2$ | $L_1$ | $L_1+L_2$ |
| Closed-form? | Yes | Yes | No | No |
| Biased? | No | Yes | Yes | Yes |
| Exact zeros? | No | No | **Yes** | Yes |
| Variable selection? | No | No | **Yes** | Yes |
| Works $p>n$? | No | Yes | Selects $\leq n$ | Yes |
| Correlated predictors | Unstable | Keeps all | Picks one | Groups together |
| Sensitivity to outliers | High | Moderate | Moderate | Moderate |

---

## Degrees of Freedom

- OLS: df $= p$ (all parameters free)
- Ridge: $\text{df}(\lambda) = \text{tr}[X(X^TX+\lambda I)^{-1}X^T] = \sum_j d_j^2/(d_j^2+\lambda) \in (0,p)$
  - Effective degrees of freedom decrease smoothly with $\lambda$
- Lasso: df $\approx$ number of non-zero coefficients (discontinuous)

---

## Choosing $\lambda$

- Cross-validation: compute CV error for grid of $\lambda$ values, pick minimum (or 1-SE rule)
- **1-SE rule**: find $\lambda_\text{min}$, then choose the largest $\lambda$ (most regularized) whose CV error is within 1 SE of the minimum → prefers simpler model

---

## Additional Possible Exam Questions

**Q: Why does Ridge never produce exact zeros but Lasso does?**
Ridge penalty $\lambda|\beta_j|^2$ has gradient $2\lambda\beta_j$ → smooth, never forces coordinate to zero unless $\lambda\to\infty$. Lasso penalty $\lambda|\beta_j|$ has subgradient $\lambda\cdot\text{sign}(\beta_j)$ → if $|X_j^Tr| < \lambda$ (residual correlation too small to overcome penalty), coefficient is set to exactly 0. Geometrically: $L_2$ ball is smooth (solution unlikely to land on axis); $L_1$ diamond has corners on axes (likely landing point).

**Q: What does the Lasso solution path look like?**
As $\lambda$ decreases from $\infty$ to $0$: coefficients enter the model one at a time. At $\lambda_\text{max} = \max_j|X_j^Ty|/N$, all coefficients are zero. Each time $\lambda$ crosses a threshold, one more coefficient becomes non-zero. LARS traces this path in $O(p^2n)$ time.

**Q: When should you use Ridge vs Lasso vs Elastic Net?**
- Ridge: when all predictors are expected to contribute (dense true signal), or for prediction stability with correlated predictors
- Lasso: when the true model is sparse (few relevant predictors out of many), and you want automatic variable selection
- Elastic Net: when predictors are correlated and you want grouping + sparsity. In genomics (correlated gene expression), Elastic Net almost always outperforms Lasso.

**Q: What is the bias-variance tradeoff for Ridge?**
Increasing $\lambda$: increases bias (more shrinkage from true $\beta$) but decreases variance (coefficients more stable, less sensitive to training data). The optimal $\lambda$ minimizes $\text{Bias}^2 + \text{Var}$ (EPE-optimal). OLS is the minimum-variance unbiased estimator (Gauss-Markov), but Ridge can have lower EPE by trading a small bias for a large variance reduction.

**Q: What happens to Ridge estimates when predictors are perfectly correlated?**
$X^TX$ becomes singular (rank deficient). OLS has no unique solution. Ridge adds $\lambda I$ → $(X^TX+\lambda I)$ is always invertible → unique solution exists. This is the primary motivation for Ridge when $p>n$ or with multicollinear predictors.
