# Q21-T — AIC, BIC, and Model Selection Criteria
> Weeks 1/2. Could ask to derive AIC from KL divergence, compare AIC vs BIC, or contrast with CV.

---

## The Problem: Estimating Test Error Without a Test Set

Training error underestimates EPE. We want an analytical correction that avoids re-fitting the model $K$ times (unlike CV).

**General approach**: penalize the training log-likelihood for model complexity:
$$\text{Criterion} = -2\ell(\hat{\theta}) + \text{Penalty}(p, N)$$

where $\ell(\hat{\theta})$ is the maximized log-likelihood and $p$ = number of free parameters.

---

## AIC — Akaike Information Criterion

$$\text{AIC} = -2\ell(\hat{\theta}) + 2p$$

**Derivation** (from KL divergence): AIC estimates the expected Kullback-Leibler divergence between the true data-generating distribution $f$ and the fitted model $g_{\hat{\theta}}$:
$$\text{KL}(f\|g_{\hat{\theta}}) = \int f(x)\log\frac{f(x)}{g_{\hat{\theta}}(x)}dx = \text{const} - E_f[\log g_{\hat{\theta}}(x)]$$

The log-likelihood $\ell(\hat{\theta})$ estimates $E_f[\log g_{\hat{\theta}}]$ on training data — optimistically, because $\hat{\theta}$ was fitted to the same data. The bias is $\approx p$ (Takeuchi 1976; exactly $p$ for regular models). Correcting: $-2E[\ell(\hat{\theta})] \approx -2\ell(\hat{\theta}) + 2p$ = AIC.

**Asymptotic equivalence**: AIC is asymptotically equivalent to LOO-CV for linear models.

**What AIC minimizes**: expected KL divergence from the true model → selects the model that best approximates the truth in terms of information loss.

---

## BIC — Bayesian Information Criterion

$$\text{BIC} = -2\ell(\hat{\theta}) + p\log N$$

**Derivation** (from Bayesian model comparison): BIC approximates $-2\log P(\text{data}|\text{model})$ via Laplace approximation of the marginal likelihood. The Bayesian approach assigns equal prior to each model, then selects the model with highest posterior. The $p\log N$ penalty comes from the Laplace approximation of the integral over parameter space.

**Key difference from AIC**: penalty grows with $N$. For $N > 7$: $\log N > 2$, so BIC penalizes complexity more than AIC.

---

## AIC vs BIC: When Each Wins

| Property | AIC | BIC |
|----------|-----|-----|
| Penalty | $2p$ (constant in $N$) | $p\log N$ (grows with $N$) |
| Goal | Best predictive model | True model identification |
| Consistent? | No — over-selects complex models as $N\to\infty$ | Yes — selects true model as $N\to\infty$ (if in candidate set) |
| When true model in set | May over-select | Selects correctly for large $N$ |
| When true model NOT in set | Better approximation | Worse (penalizes too much) |
| Small $N$ | Under-penalizes (AICc was introduced to correct this bias at small $N$) | May over-penalize for very small $N$ (large $\log N$ effect) |
| Typical use | Predictive modeling, time series (ARIMA) | Model identification, variable selection |

**Rule**: use AIC when you want the best predictive model. Use BIC when you believe the true model is in the candidate set and want to identify it.

---

## Mallow's $C_p$ Statistic

For linear models with Gaussian errors, an equivalent to AIC:
$$C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} - N + 2p$$

where $\text{RSS}_p$ = residual sum of squares for a model with $p$ parameters, $\hat{\sigma}^2$ estimated from the full model.

Models with $C_p \approx p$ (near the diagonal) are good. $C_p \gg p$ indicates underfitting.

**Connection to AIC**: $C_p = \text{AIC}/\hat{\sigma}^2 + \text{const}$ — they select the same model.

---

## AIC/BIC vs Cross-Validation

| | AIC/BIC | Cross-Validation |
|--|---------|-----------------|
| Distributional assumption | Requires likelihood (Gaussian errors for regression) | None |
| Computation | One model fit | $K$ model fits |
| Valid for | Nested parametric models | Any model, any loss |
| Penalizes what? | Number of free parameters $p$ | Implicit via validation set |
| Consistent? | BIC yes; AIC no | Approximately unbiased |

**AIC is preferred** when: computation is expensive, Gaussian model is reasonable, many models to compare.
**CV is preferred** when: non-Gaussian loss, nonparametric models, small $N$, complex preprocessing pipelines.

---

## Additional Possible Exam Questions

**Q: Why does BIC select sparser models than AIC for large $N$?**
BIC penalty = $p\log N$ grows without bound as $N\to\infty$. Adding one parameter increases BIC by $\log N$, which eventually outweighs any improvement in $\ell$. AIC penalty = $2p$ is fixed regardless of $N$ → AIC always requires the same log-likelihood improvement per parameter, so it selects the same complexity level regardless of sample size.

**Q: What does it mean for BIC to be "consistent"?**
Consistent model selection: as $N\to\infty$, the probability that BIC selects the true model approaches 1 (if the true model is in the candidate set). AIC does not have this property — it over-selects: even with infinite data, AIC has nonzero probability of choosing a more complex model than the true one. In practice for finite $N$: AIC tends to be better for prediction, BIC for identification.

**Q: How does AIC relate to KL divergence?**
AIC = $-2\ell(\hat{\theta}) + 2p$ estimates $2N \cdot E[\text{KL}(f\|g_{\hat{\theta}})] + \text{const}$. Minimizing AIC = minimizing expected KL divergence from the true distribution to the fitted model. KL divergence is zero iff the distributions are identical → AIC selects the model that most efficiently encodes the data's information.

**Q: Can you use AIC/BIC to compare models fitted on different datasets or with different $N$?**
No. AIC/BIC are only comparable across models fitted on the same dataset with the same $N$. The $\log N$ factor in BIC changes with $N$, and the log-likelihood scale depends on $N$ as well. Comparing AIC across different sample sizes is meaningless.
