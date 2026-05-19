# Q21-AE — Regularized Discriminant Analysis (RDA)
> Week 4. Could ask: derive why RDA interpolates between LDA and QDA, explain bias-variance tradeoff, and when RRDA is needed.

---

## Core Idea

RDA is a continuum between LDA and QDA:

- **LDA** assumes one shared covariance matrix across classes (high bias, low variance)
- **QDA** estimates one covariance matrix per class (low bias, high variance)
- **RDA** shrinks class-specific covariance estimates toward a pooled target to stabilize estimation

This is useful when class covariances differ, but sample size is too small for full QDA.

---

## Model

For class $k$ with sample covariance $\hat{\Sigma}_k$ and pooled covariance $\hat{\Sigma}$, the RDA covariance is

$$
\hat{\Sigma}_k(\alpha, \gamma) = \alpha \hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}
$$

and optionally a second shrinkage toward the identity:

$$
\hat{\Sigma}_k^{(\gamma)} = (1-\gamma)\hat{\Sigma}_k(\alpha) + \gamma\frac{\operatorname{tr}(\hat{\Sigma}_k(\alpha))}{p}I
$$

with tuning parameters $\alpha,\gamma \in [0,1]$.

The class discriminant score is

$$
\delta_k(x) = -\frac{1}{2}\log|\hat{\Sigma}_k^{(\gamma)}| - \frac{1}{2}(x-\hat{\mu}_k)^T(\hat{\Sigma}_k^{(\gamma)})^{-1}(x-\hat{\mu}_k) + \log\hat{\pi}_k
$$

Classify by $\arg\max_k \delta_k(x)$.

---

## Why It Works

The regularization directly controls covariance estimation variance:

1. **Small alpha (toward LDA)** pools information across classes, reducing estimation noise.
2. **Large alpha (toward QDA)** allows class-specific geometry when data supports it.
3. **Gamma shrinkage** improves conditioning and prevents unstable matrix inversions in high dimension.

So RDA trades flexibility for stability continuously instead of choosing only LDA or QDA.

---

## Limiting Cases (must know)

- $\alpha=0,\gamma=0$: **LDA**
- $\alpha=1,\gamma=0$: **QDA**
- $\gamma\to 1$: covariance approaches scaled identity (spherical regularization)

These limits are often asked to test conceptual understanding.

---

## Bias-Variance Interpretation

- Moving from LDA to QDA decreases bias but increases variance.
- RDA picks an interior point with lower expected prediction error:
  - enough flexibility to model heteroscedastic classes
  - enough shrinkage to avoid overfitting covariance noise

This is especially important when class sizes are modest relative to $p$.

---

## RDA vs LDA vs QDA

| Property | LDA | QDA | RDA |
|----------|-----|-----|-----|
| Covariance | Shared | Per-class | Shrunk per-class |
| Boundary | Linear | Quadratic | Usually quadratic |
| Variance | Low | High | Tunable |
| Hyperparameters | None | None | $\alpha$, $\gamma$ |
| Small-sample stability | Good | Poor | Good (with tuning) |

---

## RRDA (Robust RDA)

Classical covariance estimates are outlier-sensitive. RRDA replaces mean/covariance with robust estimators (for example robust location and scatter) before shrinkage. This improves classification when tails are heavy or outliers are present.

---

## Practical Tuning

Use nested CV over a grid, e.g.:

- $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$
- $\gamma \in \{0, 0.1, 0.25, 0.5\}$

Choose by validation error; apply a 1-SE preference toward simpler settings (more shrinkage) when tied.

---

## Limitations

1. Still relies on Gaussian class-conditional shape.
2. Two hyperparameters increase tuning cost.
3. If classes are very non-elliptical, discriminative methods (SVM, boosted trees) can perform better.
4. Severe class imbalance can destabilize covariance estimation for minority classes.

