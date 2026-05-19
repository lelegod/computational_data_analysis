# Q21-AG — K-Nearest Neighbors (KNN): Theory and Model Selection
> Week 2. Could ask: derive KNN behavior as $K$ changes, explain curse-of-dimensionality effects, and compare with parametric classifiers.

---

## Model

For a query point $x_0$, find the set $\mathcal{N}_K(x_0)$ of its $K$ nearest training points.

- **Regression**:
  $$
  \hat{f}(x_0)=\frac{1}{K}\sum_{i\in\mathcal{N}_K(x_0)} y_i
  $$
- **Classification**:
  $$
  \hat{C}(x_0)=\arg\max_c\sum_{i\in\mathcal{N}_K(x_0)}\mathbf{1}(y_i=c)
  $$

KNN is nonparametric and instance-based (no explicit global model is fitted).

---

## Why K Matters: Bias-Variance

- **Small $K$ (e.g., 1)**:
  - Very flexible boundary
  - Low bias, high variance
  - Sensitive to noise and outliers
- **Large $K$**:
  - Smoother boundary
  - Higher bias, lower variance
  - Can underfit and move toward class prior / global mean

As $K \to N$, regression KNN predicts $\bar{y}$ and classification predicts majority class.

---

## Distance Metric and Feature Scaling

KNN depends entirely on the distance function:

- Euclidean for continuous features
- Manhattan for robust axis-aligned differences
- Gower/Hamming variants for mixed or binary features

Feature scaling is mandatory; otherwise one large-scale variable dominates neighbor search.

---

## Asymptotic Insight

Under regularity conditions:

- If $K \to \infty$ and $K/N \to 0$, KNN can be consistent.
- 1-NN has asymptotic error at most twice Bayes error (Cover-Hart bound).

So KNN can approach optimal classification with enough data and a proper $K$ schedule.

---

## Curse of Dimensionality (Critical)

In high dimension:

1. Distances concentrate (nearest and farthest points become similar).
2. Local neighborhoods become large in volume and stop being local.
3. Required sample size for fixed local resolution grows exponentially with dimension.

Hence plain KNN degrades strongly when $p$ is large unless dimensionality reduction or feature selection is used.

---

## Choosing K Properly

Select $K$ by cross-validation:

1. Candidate grid, e.g. odd $K \in \{1,3,\dots,31\}$ for binary classification.
2. Compute CV error for each $K$.
3. Choose minimum-CV $K$ or apply 1-SE rule to prefer smoother model.

For fair evaluation, tune $K$ in inner folds and report performance from outer folds (nested CV).

---

## KNN vs LDA / Logistic / Trees

| Property | KNN | LDA / Logistic | Trees |
|----------|-----|----------------|-------|
| Model form | Nonparametric local | Parametric global | Rule-based partitions |
| Interpretability | Low | Medium to high | High |
| Works with small $N$ | Usually poor | Often better | Moderate |
| High-dimensional behavior | Poor without reduction | Better with regularization | Variable |
| Decision boundary | Highly flexible | Linear (or specified link) | Piecewise constant |

KNN is strongest when the boundary is nonlinear and there is enough dense local data.

---

## Limitations

1. Prediction cost is high: neighbor search against training set each time.
2. Sensitive to irrelevant features and scaling.
3. Performance collapses in high dimensions without preprocessing.
4. No explicit coefficients, so explanation is limited.

