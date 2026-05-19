# Q21-U — Bagging and Variance Reduction
> Week 5. Standalone treatment — distinct from Random Forest. Could ask to derive variance formula.

---

## The Core Idea

**Bagging** (Bootstrap AGGregatING): train $B$ models on $B$ bootstrap samples of the data, then average their predictions.

**Motivation**: a single high-variance model (e.g., deep decision tree) is unstable — small changes in training data produce very different models. Averaging many such models cancels out their idiosyncratic noise.

---

## The Algorithm

For $b = 1, \ldots, B$:
1. Draw bootstrap sample $Z^{*b}$ of size $N$ with replacement
2. Fit model $\hat{f}^{*b}(x)$ on $Z^{*b}$

Combine:
$$\hat{f}_\text{bag}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}^{*b}(x) \quad \text{(regression)}$$
$$\hat{G}_\text{bag}(x) = \text{majority vote}\{\hat{G}^{*b}(x)\} \quad \text{(classification)}$$

---

## Why Bagging Reduces Variance: The Math

For $B$ identically distributed models with variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}\!\left(\frac{1}{B}\sum_{b=1}^B X_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

**Two components**:
- $\rho\sigma^2$: irreducible floor — cannot be eliminated by averaging more models
- $(1-\rho)\sigma^2/B$: disappears as $B\to\infty$

**Key observations**:
1. As $B\to\infty$: variance $\to \rho\sigma^2$. Adding more trees always helps but with diminishing returns.
2. To reduce the floor, reduce $\rho$ (make models less correlated). This is exactly what Random Forest does via random feature subsampling.
3. If models were independent ($\rho=0$): variance $= \sigma^2/B \to 0$. But bootstrap samples overlap → $\rho > 0$ always.

---

## What Bagging Does (and Does NOT) Do

**Bagging reduces variance** — the spread of predictions across different training sets.

**Bagging does NOT reduce bias** — the expected prediction of the ensemble equals the expected prediction of a single model:
$$E[\hat{f}_\text{bag}(x)] = E[\hat{f}^{*b}(x)] \quad \text{(each bootstrap model is approximately unbiased for the original)}$$

**Implication**: bagging only helps high-variance base learners.
- Deep unpruned trees: high variance → bagging helps a lot
- Stumps (depth-1): low variance, high bias → bagging gives little variance reduction, bias remains
- Linear regression: already low variance → bagging barely helps
- KNN with small $K$: high variance → bagging helps significantly

---

## OOB (Out-of-Bag) Error Estimation

Each bootstrap sample leaves out ~36.8% of observations. For observation $i$:
- It is OOB for approximately $B/e \approx 0.368B$ trees
- Predict $x_i$ using only those trees → OOB prediction $\hat{f}^{-i}(x_i)$
- OOB error = $\frac{1}{N}\sum_i L(y_i, \hat{f}^{-i}(x_i))$

**OOB error $\approx$ LOO-CV error** — a nearly unbiased estimate of test error with no extra computation. This is a major practical advantage of bagging.

---

## Bagging vs Random Forest vs Boosting

| | Bagging | Random Forest | Boosting |
|--|---------|--------------|---------|
| Base learner | Any | Deep trees | Shallow trees |
| Sequential? | No (parallel) | No (parallel) | Yes |
| Feature subset per split | No (all $p$) | Yes ($m < p$) | No (all $p$) |
| Reduces | Variance | Variance (more) | Bias |
| Tree correlation $\rho$ | Higher | Lower | N/A |
| Can overfit? | No | No | Yes |
| OOB error? | Yes | Yes | No |

**Why RF beats bagging**: both use deep trees and bootstrap. RF additionally subsamples $m=\lfloor\sqrt{p}\rfloor$ features at each split → decorrelates trees → lowers $\rho$ → lower variance floor $\rho\sigma^2$.

---

## Which Base Learners Benefit Most from Bagging?

A base learner benefits from bagging when it has:
1. **High variance**: unstable estimates that change a lot with training data
2. **Low bias**: so averaging doesn't introduce systematic error

**Best base learners for bagging**: deep unpruned trees (prototypical), low-$K$ KNN.

**Poor base learners for bagging**: linear regression (already stable), stumps (high bias dominates), neural networks (slow to retrain, and variance is not the primary issue).

---

## Additional Possible Exam Questions

**Q: Does bagging reduce the training error?**
No. Each bootstrap model is trained on its own $Z^{*b}$, not on the original data. The ensemble's predictions on the original training data can actually be worse than a single model's training error, because each model was trained on a different (overlapping) sample. Bagging is designed to improve test error, not training error.

**Q: Can you bag linear regression? Does it help?**
Yes — draw $B$ bootstrap samples, fit OLS on each, average the $B$ coefficient vectors. The average of $B$ OLS estimates converges to a single OLS estimate (since OLS is linear in $y$, the average is the OLS estimate on the mean of bootstrap responses ≈ original OLS). So bagging linear regression gives approximately no improvement. Linear models have low variance to begin with; bagging cannot reduce the irreducible floor.

**Q: Why does the variance formula have a floor at $\rho\sigma^2$?**
The correlation $\rho$ between any two bootstrap models reflects shared structure in the training data — they were both trained on overlapping samples from the same distribution. Even with $B\to\infty$ trees, you cannot average away this shared component. It is an irreducible correlation imposed by the bootstrap mechanism itself. Random forests reduce $\rho$ by randomly masking features, pushing this floor lower.

**Q: What is the relationship between bagging and the bias-variance decomposition?**
Bias of ensemble = bias of individual model (unchanged). Variance of ensemble = $\rho\sigma^2 + (1-\rho)\sigma^2/B$ < $\sigma^2$ (reduced). Therefore EPE of ensemble = $\sigma^2_\varepsilon + \text{Bias}^2 + \text{Var}_\text{bag}$ < EPE of individual model. Bagging strictly reduces EPE for any $\rho < 1$, $B \geq 2$.
