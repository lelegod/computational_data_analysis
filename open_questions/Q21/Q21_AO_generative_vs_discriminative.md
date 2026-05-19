# Q21-AO — Generative vs Discriminative Classifiers
> Weeks 4/7. Could ask: compare LDA, QDA, logistic regression, and SVM as generative vs discriminative approaches, and explain how assumptions trade against robustness.

---

## The Big Distinction

A **generative** classifier models the data distribution for each class:
$$
P(x,C_k)=P(x \mid C_k)P(C_k)
$$

A **discriminative** classifier models the boundary or the posterior directly:
$$
P(C_k \mid x)
$$
or even just the separating hyperplane.

This distinction is a classic compare-and-discuss exam question.

---

## LDA and QDA: Generative

LDA and QDA assume Gaussian class-conditionals.

### LDA
$$
P(x \mid C_k)=\mathcal{N}(x;\mu_k,\Sigma)
$$

- shared covariance
- linear boundary
- closed-form fitting

### QDA
$$
P(x \mid C_k)=\mathcal{N}(x;\mu_k,\Sigma_k)
$$

- class-specific covariance
- quadratic boundary
- more flexible, more variance

So both are generative, but with different covariance assumptions.

---

## Logistic Regression: Discriminative Probabilistic

Logistic regression models the posterior directly:
$$
\log \frac{P(C_1 \mid x)}{P(C_0 \mid x)} = \beta_0 + x^T \beta
$$

### Characteristics

- discriminative
- linear boundary
- no Gaussian assumption on $x$
- fitted iteratively by maximum likelihood
- outputs probabilities

So logistic regression is the discriminative counterpart to LDA.

---

## SVM: Discriminative Geometric

SVM focuses on the margin:
$$
\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 + C\sum_i \xi_i
$$

subject to margin constraints.

### Characteristics

- discriminative
- geometric rather than probabilistic
- emphasizes separation margin
- no direct probability model
- can become nonlinear with kernels

So SVM is boundary-based rather than density-based.

---

## Comparison Table

| Method | Type | Main assumption | Boundary | Output |
|--------|------|-----------------|----------|--------|
| LDA | Generative | Gaussian, shared covariance | Linear | Class/probability |
| QDA | Generative | Gaussian, class-specific covariance | Quadratic | Class/probability |
| Logistic | Discriminative | Linear log-odds | Linear | Probability |
| SVM | Discriminative | Margin-based separation | Linear or kernel nonlinear | Class / score |

---

## Efficiency vs Robustness

This is the core tradeoff.

### Generative methods

If the distributional assumptions are approximately correct:
- can be statistically efficient
- may learn well from smaller samples
- use more structure from the data

But if the assumptions are wrong:
- boundaries can be misspecified
- performance can degrade

### Discriminative methods

They make fewer assumptions about $x$:
- often more robust to model misspecification
- focus directly on classification

But:
- may need more data
- may lose efficiency when the generative assumptions are truly right

---

## Which Methods Are Linear and Why

- **LDA** is linear because equal covariance makes quadratic terms cancel
- **Logistic regression** is linear because linearity is imposed directly in the log-odds
- **SVM** is linear in the original feature space unless kernels are used
- **QDA** is nonlinear because class covariances differ

This is an easy but high-value comparison point.

---

## When to Use Which

**Use LDA when**:
- Gaussian structure is plausible
- sample size is limited
- shared covariance is reasonable

**Use QDA when**:
- class covariances differ substantially
- enough data exists per class

**Use logistic regression when**:
- probabilities matter
- linear boundary is okay
- you want fewer assumptions on $x$

**Use SVM when**:
- classification margin matters
- probabilities are not the main priority
- high-dimensional separation is important

---

## Limitations

1. LDA/QDA can fail badly under strong non-Gaussianity.
2. QDA is high-variance in small samples.
3. Logistic regression is limited to linear log-odds unless features are expanded.
4. SVM does not natively give calibrated probabilities.

---

## Additional Possible Exam Questions

**Q: Why can LDA outperform logistic regression on small datasets?**
If the Gaussian assumptions are approximately correct, LDA uses stronger structure and therefore estimates the boundary more efficiently from limited data.

**Q: Why is logistic regression considered more robust than LDA?**
Because it does not require modeling the class-conditional feature distribution. It directly fits the posterior boundary instead of relying on Gaussian assumptions.

**Q: Why is SVM often grouped with discriminative methods but separated from logistic regression?**
Because both are discriminative, but logistic regression is probabilistic while SVM is geometric and margin-based. Logistic optimizes log loss; SVM optimizes hinge loss.
