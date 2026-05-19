# Q21-P — Logistic Regression vs LDA
> Week 3/4. Classic comparison; both produce linear boundaries but from different assumptions.

---

## Logistic Regression

**Model**: directly model the posterior probability of class membership:
$$\log\frac{P(C_1|x)}{P(C_0|x)} = \beta_0 + x^T\beta \quad \Leftrightarrow \quad P(C_1|x) = \frac{1}{1+\exp(-\beta_0-x^T\beta)} = \sigma(\beta_0+x^T\beta)$$

The log-odds (logit) is linear in $x$. The decision boundary $P(C_1|x)=0.5$ is a hyperplane.

**Fitting**: Maximum Likelihood Estimation. Log-likelihood:
$$\ell(\beta) = \sum_{i=1}^N \left[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

where $\hat{p}_i = \sigma(\beta_0 + x_i^T\beta)$.

No closed-form solution — maximize via Newton-Raphson (IRLS: Iteratively Reweighted Least Squares):
$$\beta_\text{new} = \beta - \left(\frac{\partial^2\ell}{\partial\beta\partial\beta^T}\right)^{-1}\frac{\partial\ell}{\partial\beta}$$

The Hessian is negative definite (log-likelihood is concave) → Newton-Raphson converges to the **global maximum**.

---

## LDA (Linear Discriminant Analysis) — Recap

**Model**: assume class-conditional Gaussians with shared covariance:
$$P(x|C_k) = \mathcal{N}(x;\mu_k,\Sigma), \quad k=1,\ldots,K$$

Apply Bayes' rule → log-posterior ratio is linear in $x$ (quadratic terms cancel) → linear boundary.

**Fitting**: closed-form MLE — estimate $\hat{\mu}_k$, $\hat{\Sigma}$ (pooled), $\hat{\pi}_k$ directly from training data.

---

## Same Decision Boundary, Different Assumptions

Both produce **linear** decision boundaries. Key difference: where the linearity comes from.

| Property | Logistic Regression | LDA |
|----------|--------------------|----|
| What's modeled | $P(C_k\|x)$ directly (discriminative) | $P(x\|C_k)$ then Bayes (generative) |
| Distributional assumption | None on $x$ | $x\|C_k \sim \mathcal{N}(\mu_k,\Sigma)$ |
| Fitting | Iterative MLE (IRLS) | Closed-form MLE |
| Converges to global? | Yes (concave log-likelihood) | Yes (explicit formula) |
| Uses unlabeled data? | No | No |
| Boundary | Linear | Linear (shared $\Sigma$) |
| When Gaussian assumption holds | Less efficient than LDA | More efficient (uses more information) |
| When assumption violated | More robust | Less robust |
| Extrapolation behavior | Probabilistic at any $x$ | Relies on Gaussian tails |
| Multi-class | Multinomial logistic | Natural ($K$ classes) |

---

## When to Use Which

**Use LDA when**:
- You believe the Gaussian class-conditional assumption is approximately correct
- Features are roughly normally distributed within each class
- Small training set (fewer parameters to estimate vs logistic regression)
- You want dimensionality reduction (LDA can project to $K-1$ dimensions)

**Use Logistic Regression when**:
- Class-conditionals are clearly non-Gaussian (binary features, skewed distributions)
- You want probability calibration without assuming a distribution for $x$
- You need the discriminative boundary only (don't care about modeling $P(x)$)
- Regularized (L1/L2 penalized logistic) for high-dimensional feature selection

---

## Logistic Regression with Regularization

**Ridge (L2) logistic**: $\ell(\beta) - \lambda\|\beta\|_2^2$ → shrinks all coefficients, no zeros
**Lasso (L1) logistic**: $\ell(\beta) - \lambda\|\beta\|_1$ → sparse, variable selection

Both fitted by penalized IRLS or coordinate descent. No closed form.

---

## Multi-class Extension

**One-vs-Rest**: train $K$ binary classifiers (class $k$ vs all others). Predict class with highest probability. Simple but probabilities may not sum to 1.

**Multinomial logistic regression** (Softmax regression):
$$P(C_k|x) = \frac{\exp(\beta_{k0}+x^T\beta_k)}{\sum_{j=1}^K\exp(\beta_{j0}+x^T\beta_j)}$$

One class is the reference (set $\beta_{K}=0$). Log-likelihood is concave → global optimum.

---

## Additional Possible Exam Questions

**Q: Why is logistic regression called "discriminative" and LDA "generative"?**
Discriminative models directly model the decision boundary or posterior $P(C_k|x)$ — logistic regression directly parameterizes this. Generative models model the full data distribution $P(x,C_k) = P(x|C_k)P(C_k)$, then apply Bayes' rule to get $P(C_k|x)$. LDA models $P(x|C_k)$ and $P(C_k)$, which is a generative model. If the generative model is correct, it uses more information → more efficient. If wrong → discriminative model is more robust.

**Q: LDA and logistic regression produce the same linear boundary — when do they give different results?**
If $P(x|C_k)$ is truly Gaussian with equal covariances, LDA's boundary converges faster (more statistically efficient). If the Gaussian assumption fails, logistic regression's boundary is better calibrated because it doesn't force incorrect distributional structure into the posterior. In practice, for large datasets, both usually give similar results; for small datasets with Gaussian data, LDA wins.

**Q: Why does the logistic regression log-likelihood have no closed-form solution?**
The score equations $\partial\ell/\partial\beta = X^T(y-\hat{p}) = 0$ are nonlinear in $\beta$ (because $\hat{p} = \sigma(X\beta)$ is nonlinear). Unlike OLS (linear score equations), there is no algebraic solution. However, the Hessian $H = -X^TWX$ (where $W = \text{diag}(\hat{p}_i(1-\hat{p}_i))$) is negative definite everywhere → log-likelihood is strictly concave → unique global maximum → Newton-Raphson converges reliably.

**Q: What is complete separation in logistic regression?**
If the classes are perfectly linearly separable, the MLE $\hat{\beta}\to\infty$ (log-likelihood increases without bound as the boundary becomes sharper). The algorithm diverges. Fix: add regularization (L2 penalty keeps coefficients finite), or use Firth's method (modified score equations). LDA does not have this problem — it directly computes class means and covariances.

**Q: How does penalized logistic regression compare to SVM?**
Both produce linear classifiers. SVM minimizes hinge loss + $L_2$ penalty; L2 logistic regression minimizes log loss + $L_2$ penalty. Hinge loss is zero for correctly classified points beyond the margin (sparse solution) while log loss is never exactly zero (all points contribute). SVM is purely geometric (no probabilities); logistic gives calibrated probabilities. With the same $L_2$ penalty, they give similar boundaries but SVM is typically faster for large $p$.
