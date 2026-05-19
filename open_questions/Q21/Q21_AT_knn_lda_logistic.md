# Q21-AT — KNN vs LDA vs Logistic Regression
> Weeks 2/4. Could ask: compare local nonparametric classification to generative and discriminative linear classifiers, and explain when each is preferable.

---

## Three Different Classification Philosophies

These three methods represent very different ways of doing classification:

- **KNN**: local, nonparametric
- **LDA**: generative, parametric
- **Logistic regression**: discriminative, parametric

That makes this a very plausible compare-and-choose Q21.

---

## KNN

For a query point $x_0$, KNN looks at the $K$ nearest training observations and predicts by majority vote.

### Main idea

- no global model is fitted
- prediction is based on local neighborhood structure
- very flexible boundary

### Consequence

KNN can capture nonlinear decision boundaries, but suffers badly in high dimensions.

---

## LDA

LDA assumes Gaussian class-conditionals with shared covariance:
$$
P(x \mid C_k)=\mathcal{N}(x;\mu_k,\Sigma)
$$

### Main idea

- model each class distribution
- apply Bayes' rule
- boundary becomes linear

### Consequence

LDA can work very well with limited data if the assumptions are approximately right.

---

## Logistic Regression

Logistic regression models the posterior directly:
$$
\log\frac{P(C_1 \mid x)}{P(C_0 \mid x)}=\beta_0 + x^T\beta
$$

### Main idea

- no model for $P(x \mid C_k)$
- directly estimate the classification boundary
- linear boundary in feature space

### Consequence

It is usually more robust than LDA when Gaussian assumptions fail.

---

## Core Comparison

### KNN

- flexible, local, nonlinear
- low bias / high variance for small $K$
- vulnerable to curse of dimensionality

### LDA

- structured, generative, linear
- efficient under Gaussian assumptions
- lower variance than flexible methods

### Logistic Regression

- structured, discriminative, linear
- fewer assumptions on $x$
- probability output with iterative fitting

---

## Comparison Table

| Property | KNN | LDA | Logistic Regression |
|----------|-----|-----|---------------------|
| Parametric? | No | Yes | Yes |
| Local or global? | Local | Global | Global |
| Boundary | Potentially highly nonlinear | Linear | Linear |
| Uses class density model? | No | Yes | No |
| Works well in high dimension? | Usually no | Better | Better with regularization |
| Needs scaling? | Yes, critically | Less central | Often yes, but less crucial |

---

## Bias-Variance Perspective

- Small-$K$ KNN: low bias, high variance
- Large-$K$ KNN: higher bias, lower variance
- LDA: higher bias than KNN, usually much lower variance
- Logistic: similar boundary complexity to LDA, but different assumptions

So KNN is often strongest when there is lots of local data and a nonlinear boundary, while LDA or logistic are stronger when data are limited or high-dimensional.

---

## When to Use Which

**Use KNN when**:
- the boundary is nonlinear
- sample size is large
- dimension is modest

**Use LDA when**:
- Gaussian structure is plausible
- sample size is limited
- a linear boundary is acceptable

**Use logistic regression when**:
- you want probabilities
- a linear boundary is acceptable
- you want fewer assumptions on the feature distribution

---

## Limitations

1. KNN collapses in high dimensions.
2. LDA can fail if Gaussian assumptions are badly wrong.
3. Logistic regression is limited to linear log-odds unless features are engineered.
4. KNN prediction can be computationally expensive at test time.

---

## Additional Possible Exam Questions

**Q: Why can LDA beat KNN on small datasets even when the true boundary is not perfectly linear?**
Because LDA has lower variance. With small data, the stability gain from a simple parametric model can outweigh the bias from the linear assumption.

**Q: Why is KNN especially sensitive to feature scaling?**
Because it uses distances directly. A single high-scale variable can dominate neighbor search and distort the local geometry.

**Q: When would logistic regression be preferred over LDA?**
When Gaussian assumptions are doubtful but a linear decision boundary is still reasonable, especially if calibrated probabilities are needed.
