# Q21-R — The Bootstrap
> Week 2. Could ask to explain the algorithm, confidence intervals, or contrast with CV.

---

## The Core Idea

The bootstrap approximates the sampling distribution of a statistic $\hat{\theta}$ by **resampling from the observed data** rather than from the unknown population.

If we could repeat the experiment many times, we'd compute $\hat{\theta}^{(1)},\hat{\theta}^{(2)},\ldots$ and estimate variability directly. We can't — but we can treat the observed data as a proxy for the population and resample from it.

---

## The Bootstrap Algorithm

1. From training data $\mathcal{Z} = \{(x_1,y_1),\ldots,(x_N,y_N)\}$, draw $B$ bootstrap samples
2. Each bootstrap sample $\mathcal{Z}^{*b}$: draw $N$ observations **with replacement** from $\mathcal{Z}$
3. Each sample $\mathcal{Z}^{*b}$ contains ~63.2% unique observations (some repeated, ~36.8% never drawn)
4. Compute $\hat{\theta}^{*b}$ on each $\mathcal{Z}^{*b}$
5. Estimate variance: $\widehat{\text{Var}}(\hat{\theta}) = \frac{1}{B-1}\sum_{b=1}^B(\hat{\theta}^{*b}-\bar{\theta}^*)^2$

where $\bar{\theta}^* = \frac{1}{B}\sum_b\hat{\theta}^{*b}$.

---

## Bootstrap Confidence Intervals

**Percentile method** (simplest):
$$\text{CI}_{95\%} = [\hat{\theta}^*_{(0.025)},\; \hat{\theta}^*_{(0.975)}]$$
Take the 2.5th and 97.5th percentiles of the bootstrap distribution directly.

**Basic (pivot) method**:
$$\text{CI}_{95\%} = [2\hat{\theta} - \hat{\theta}^*_{(0.975)},\; 2\hat{\theta} - \hat{\theta}^*_{(0.025)}]$$
Corrects for bias in the percentile method by reflecting around $\hat{\theta}$.

**Standard normal**:
$$\text{CI}_{95\%} = \hat{\theta} \pm 1.96\cdot\widehat{\text{SE}}(\hat{\theta})$$
Only valid when bootstrap distribution is approximately Gaussian.

---

## Parametric vs Nonparametric Bootstrap

**Nonparametric bootstrap** (default): resample observations with replacement. Makes no distributional assumption — the empirical distribution is the proxy for the population.

**Parametric bootstrap**: fit a parametric model $\hat{F}$ to the data (e.g., fit $\mathcal{N}(\hat{\mu},\hat{\sigma}^2)$), then draw new samples from $\hat{F}$. More efficient when the parametric form is correct; less robust when it is wrong.

---

## Bootstrap vs Cross-Validation

Both estimate test error / variability. Key differences:

| Property | Bootstrap | Cross-Validation |
|----------|-----------|-----------------|
| Purpose | Estimate variability of $\hat{\theta}$, CIs | Estimate prediction error (EPE) |
| Sampling | With replacement (63.2% unique) | Without replacement (disjoint folds) |
| Training size | $N$ (same as original) | $N(1-1/K)$ |
| Test contamination | ~36.8% of test obs are in training | Zero (folds are disjoint) |
| Bias for error estimation | Optimistic (training ≈ test) | Nearly unbiased |
| Variance | Moderate | Higher (small $K$) / lower (large $K$) |

**Bootstrap for error estimation**: because bootstrap training samples overlap with the "test" (OOB) set only via resampling, naive bootstrap error is optimistically biased. The **.632 estimator** corrects for this:
$$\hat{\text{Err}}^{.632} = 0.368\cdot\overline{\text{err}} + 0.632\cdot\hat{\text{Err}}^1$$
where $\overline{\text{err}}$ = training error, $\hat{\text{Err}}^1$ = leave-one-out bootstrap error (average over OOB predictions). The weights 0.368 and 0.632 reflect the fractions of OOB and in-bag observations.

---

## The OOB Connection to Random Forest

In Random Forest, each tree is trained on a bootstrap sample. The ~36.8% OOB observations serve as a built-in test set for that tree. Averaging OOB predictions across all trees gives the OOB error — a nearly unbiased estimate of generalization error with no extra computation. This is the bootstrap directly embedded in the Random Forest algorithm.

---

## Additional Possible Exam Questions

**Q: Why does sampling with replacement give ~63.2% unique observations?**
Probability that observation $i$ is NOT selected in one draw = $(1-1/N)$. Probability it is NEVER selected in $N$ draws = $(1-1/N)^N \to e^{-1} \approx 0.368$ as $N\to\infty$. So ~36.8% are left out (OOB) and ~63.2% are included (at least once).

**Q: When does the bootstrap fail?**
(1) Heavy tails / extreme values: the bootstrap distribution of $\max(x_i)$ converges very slowly because the true maximum depends on rare events not well-represented in the empirical distribution. (2) Non-smooth statistics: the median of a discrete distribution can have a discrete bootstrap distribution with poor coverage. (3) Dependent data: standard bootstrap assumes IID — for time series, use block bootstrap (resample contiguous blocks to preserve temporal structure).

**Q: What is the difference between the bootstrap standard error and the analytical standard error for OLS?**
OLS: $\widehat{\text{SE}}(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2[(X^TX)^{-1}]_{jj}}$ — analytical, requires Gaussian errors and correct model specification. Bootstrap SE: simulate $B$ datasets, compute $\hat{\beta}_j^{*b}$ each time, take standard deviation. Valid under weaker assumptions — no Gaussian requirement. Especially useful for nonlinear estimators or complex statistics (e.g., SE of a cross-validated AUC) where no analytical formula exists.

**Q: How would you use the bootstrap to test a hypothesis?**
Under $H_0$, construct a bootstrap distribution of the test statistic centered at zero (e.g., subtract the observed statistic from each bootstrap replicate). The p-value = fraction of bootstrap statistics more extreme than the observed one. This is the "bootstrap hypothesis test" or permutation test variant. Unlike normal-theory tests, it does not assume Gaussian null distribution.
