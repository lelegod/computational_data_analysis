# Q21-AL — LDA vs QDA vs RDA
> Week 4. Could ask: compare the covariance assumptions behind LDA, QDA, and RDA, explain why the boundaries differ, and discuss the bias-variance tradeoff.

---

## The Shared Gaussian Framework

All three methods assume Gaussian class-conditional densities:
$$
P(x \mid C_k) = \mathcal{N}(x; \mu_k, \Sigma_k)
$$

They differ in how much covariance structure they allow.

---

## LDA

LDA imposes one shared covariance matrix:
$$
\Sigma_k = \Sigma \quad \text{for all } k
$$

This makes the quadratic terms cancel in the log-posterior ratio, so the decision boundary is linear.

### Consequences

- fewer parameters
- lower variance
- more bias if classes truly have different covariance shapes
- closed-form estimation

---

## QDA

QDA allows one covariance matrix per class:
$$
\Sigma_k \neq \Sigma_{k'} \quad \text{in general}
$$

Then the quadratic terms no longer cancel, so the boundary becomes quadratic.

### Consequences

- more flexibility
- lower bias
- higher variance
- requires much more data per class

---

## RDA

RDA interpolates between LDA and QDA by shrinking each class covariance toward the pooled covariance:
$$
\hat{\Sigma}_k(\alpha) = \alpha \hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}
$$

and often also toward a spherical target:
$$
\hat{\Sigma}_k(\alpha,\gamma)
=
(1-\gamma)\hat{\Sigma}_k(\alpha) + \gamma \frac{\operatorname{tr}(\hat{\Sigma}_k(\alpha))}{p}I
$$

### Consequences

- flexible continuum instead of all-or-nothing choice
- tunable bias-variance tradeoff
- more stable than QDA in small samples
- usually still gives quadratic boundaries

---

## Why the Boundaries Differ

For two classes $k$ and $k'$, the log-posterior ratio contains:
$$
-\frac{1}{2}x^T(\Sigma_k^{-1}-\Sigma_{k'}^{-1})x
$$

- In LDA, $\Sigma_k=\Sigma_{k'}$, so this term is zero
- In QDA, it is nonzero, so the boundary is quadratic
- In RDA, it depends on the shrinkage level, so the boundary moves continuously from linear-like to fully quadratic

This is the key derivation point in a compare question.

---

## Parameter Counting and Variance

Suppose there are $K$ classes and $p$ features.

- **LDA** estimates one covariance: roughly $p(p+1)/2$ covariance parameters
- **QDA** estimates $K$ covariances: roughly $Kp(p+1)/2$
- **RDA** still uses class-specific covariance structure, but shrinkage reduces effective complexity

So QDA can overfit badly when $p$ is not small relative to class sample size.

---

## Comparison Table

| Property | LDA | QDA | RDA |
|----------|-----|-----|-----|
| Covariance assumption | Shared | Per-class | Shrunk per-class |
| Boundary | Linear | Quadratic | Usually quadratic |
| Bias | Higher | Lower | Tunable |
| Variance | Lower | Higher | Tunable |
| Hyperparameters | None | None | $\alpha$, $\gamma$ |
| Small-sample stability | Good | Poor | Good if tuned |

---

## When to Use Which

**Use LDA when**:
- sample size is limited
- covariances are similar across classes
- interpretability and stability matter

**Use QDA when**:
- sample size per class is large
- classes clearly have different spreads or orientations
- a quadratic boundary is genuinely needed

**Use RDA when**:
- you suspect LDA is too restrictive but QDA is too unstable
- $p$ is moderate to large
- you want to tune the complexity by cross-validation

---

## Relation to Logistic Regression

LDA and QDA are **generative**:
- they model $P(x \mid C_k)$ and then apply Bayes' rule

Logistic regression is **discriminative**:
- it models $P(C_k \mid x)$ directly

So in a broader compare question:
- LDA vs logistic = generative linear vs discriminative linear
- QDA vs logistic = quadratic Gaussian vs linear discriminative
- RDA sits between the two Gaussian extremes

---

## Limitations

1. All three rely on Gaussian class-conditional reasoning.
2. QDA can fail when class covariance matrices are singular.
3. RDA introduces tuning complexity.
4. If class shapes are non-elliptical, SVM or trees may work better.

---

## Additional Possible Exam Questions

**Q: Under what condition does QDA reduce exactly to LDA?**
When all class covariance matrices are equal. Then the quadratic term in the discriminant cancels, leaving a linear boundary.

**Q: Why is RDA often preferable to QDA in practice?**
Because QDA has low bias but very high variance when each class covariance must be estimated from limited data. RDA preserves some class-specific structure while stabilizing those estimates by shrinkage.

**Q: Why does LDA often outperform QDA even when QDA is more flexible?**
Because expected test error depends on both bias and variance. If sample size is limited, the extra flexibility of QDA can cost more in estimation noise than it gains in lower bias.
