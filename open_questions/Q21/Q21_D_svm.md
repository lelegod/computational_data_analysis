# Q21-D — SVM: Derivation and Kernel Trick
> High likelihood candidate (Week 7, geometric approach, kernel trick)

---

## Problem Setup

Binary classification: $y_i \in \{-1, +1\}$, data $x_i \in \mathbb{R}^p$.

Find hyperplane $\{x : x^T\beta + \beta_0 = 0\}$ that maximizes the margin (distance to nearest points on each side).

**Canonical margin**: Define scale so that the nearest points satisfy $y_i(x_i^T\beta+\beta_0) \geq 1$.
Then the margin = $2/\|\beta\|$ (distance between the two supporting hyperplanes $x^T\beta+\beta_0 = \pm 1$).

---

## Hard-Margin SVM (Separable Case)

**Primal problem**:
$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i^T\beta+\beta_0) \geq 1 \; \forall i$$

**Lagrangian** (introduce multipliers $\alpha_i \geq 0$):
$$L_P = \frac{1}{2}\|\beta\|^2 - \sum_i \alpha_i[y_i(x_i^T\beta+\beta_0) - 1]$$

**Stationarity conditions**:
$$\frac{\partial L}{\partial \beta} = 0 \;\Rightarrow\; \beta = \sum_i \alpha_i y_i x_i$$
$$\frac{\partial L}{\partial \beta_0} = 0 \;\Rightarrow\; \sum_i \alpha_i y_i = 0$$

**Substituting back** into $L_P$ gives the dual:
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \langle x_i, x_j\rangle \quad \text{s.t.} \quad \alpha_i \geq 0, \; \sum_i \alpha_i y_i = 0$$

---

## KKT Conditions and Support Vectors

**Complementary slackness**: $\alpha_i[y_i(x_i^T\beta+\beta_0)-1] = 0$ for all $i$.

This means:
- $\alpha_i > 0$ only when $y_i(x_i^T\beta+\beta_0) = 1$ (point is exactly on the margin)
- $\alpha_i = 0$ for all points strictly beyond the margin

**Support vectors**: the observations with $\alpha_i > 0$. The solution $\beta = \sum_i \alpha_i y_i x_i$ depends ONLY on support vectors — all other training points have zero influence.

**Recovering $\beta_0$**: from any support vector $s$ with $y_s(x_s^T\beta+\beta_0) = 1$:
$$\beta_0 = y_s - x_s^T\beta$$

---

## Soft-Margin SVM (Non-Separable Case)

Allow violations via slack variables $\xi_i \geq 0$ ($\xi_i = \max(0, 1 - y_i f(x_i))$ = hinge loss):
$$\min_{\beta,\beta_0,\xi} \frac{1}{2}\|\beta\|^2 + C\sum_i\xi_i \quad \text{s.t.} \quad y_i(x_i^T\beta+\beta_0) \geq 1-\xi_i, \; \xi_i\geq 0$$

- Large $C$: small margin tolerance → hard boundary (high variance, low bias)
- Small $C$: large margin, many violations allowed → soft boundary (high bias, low variance)
- The dual is identical except $0 \leq \alpha_i \leq C$ (box constraint)

This is equivalent to minimizing: $\sum_i \max(0, 1-y_if(x_i)) + \lambda\|\beta\|^2$ (hinge loss + $L_2$ regularization).

---

## The Kernel Trick

The dual depends ONLY on inner products $\langle x_i, x_j\rangle$. Replace with kernel $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad \alpha_i \geq 0, \; \sum_i\alpha_iy_i = 0$$

**Why this is powerful**: we never need to compute $\phi(x)$ explicitly. The kernel function $K$ computes $\phi(x_i)^T\phi(x_j)$ directly — even for infinite-dimensional feature spaces.

**Common kernels**:
| Kernel | Formula | Feature space |
|--------|---------|--------------|
| Linear | $\langle x, x'\rangle$ | Original $\mathbb{R}^p$ |
| Polynomial degree $d$ | $(1+\langle x,x'\rangle)^d$ | All monomials up to degree $d$ |
| RBF (Gaussian) | $\exp(-\gamma\|x-x'\|^2)$ | Infinite-dimensional |
| Sigmoid | $\tanh(\kappa\langle x,x'\rangle - \delta)$ | Approximate neural net |

**Prediction**: $\hat{y} = \text{sign}\!\left(\sum_i \alpha_i y_i K(x_i, x) + \beta_0\right)$

---

## SVM vs Other Classifiers

| Property | SVM | LDA | Logistic Regression |
|----------|-----|-----|---------------------|
| Probabilistic? | No | Yes | Yes |
| Loss function | Hinge loss | Gaussian log-likelihood | Logistic (cross-entropy) |
| Decision boundary | Max-margin hyperplane | Linear (Gaussian assumption) | Linear (logistic model) |
| Handles $p\gg n$? | Yes (dual, $N$ params) | No (singular $\Sigma$) | With regularization |
| Nonlinear extension | Kernel trick | Kernel LDA | Feature engineering |
| Focus | Boundary geometry | Class density | Class probability |

---

## Additional Possible Exam Questions

**Q: How does $C$ affect bias and variance in soft-margin SVM?**
Large $C$ → penalize violations heavily → small margin → fits training data tightly → low bias, high variance. Small $C$ → large margin, many violations allowed → high bias, low variance. Cross-validate $C$ to find the sweet spot.

**Q: Why does SVM work well in high dimensions?**
The dual formulation has $N$ free parameters ($\alpha_i$), not $p$. The solution is a sparse sum over support vectors — typically only a small fraction of training points. Regularization is implicit (maximizing margin = constraining $\|\beta\|^2$).

**Q: Why can't SVM output probabilities directly?**
SVM optimizes margin (geometric criterion), not a probabilistic objective. The decision function $f(x) = x^T\beta + \beta_0$ can be converted to probabilities via Platt scaling: $P(y=1|x) = \sigma(af(x)+b)$ where $a,b$ are fit by logistic regression on the SVM outputs.

**Q: What is the connection between the SVM objective and ridge regression?**
Ridge regression: $\min_\beta \sum_i(y_i-x_i^T\beta)^2 + \lambda\|\beta\|^2$ uses squared loss.
SVM: $\min_\beta \sum_i \max(0,1-y_if(x_i)) + \lambda\|\beta\|^2$ uses hinge loss.
Both use $L_2$ regularization on $\beta$; they differ only in loss function. Hinge loss ignores correctly classified points beyond the margin (sparse solution), while squared loss penalizes all deviations.

**Q: Can SVM do multi-class classification?**
Not natively — it is binary. Extensions: One-vs-One (train $K(K-1)/2$ binary SVMs, vote); One-vs-Rest (train $K$ binary SVMs, highest score wins). Both are heuristics; there is no single elegant multi-class SVM formulation.
