# Q21-BB — PCA vs CCA
> Week 8. Could ask: compare PCA and CCA, explain what each optimizes, and when correlation across two data blocks matters more than variance within one block.

---

## The Shared Setting

Both PCA and CCA construct linear combinations of variables.

But they answer very different questions:

- PCA asks: which directions explain variance in one matrix $X$?
- CCA asks: which paired directions maximize correlation between two matrices $X$ and $Y$?

---

## PCA

PCA solves:
$$
\max_v \operatorname{Var}(Xv)
\quad \text{s.t. } \|v\|=1
$$

### Main idea

- unsupervised
- uses only one data matrix
- finds dominant variance directions

### Key consequence

PCA can ignore directions that are low-variance in $X$ even if they are strongly related to an external outcome or second data block.

---

## CCA

CCA solves:
$$
\max_{u,v} \operatorname{Corr}(Xu,Yv)
$$

### Main idea

- two-block method
- seeks paired linear combinations
- focuses on cross-block association

### Key consequence

CCA may choose directions that have only modest variance individually but are strongly correlated across the two datasets.

---

## Core Difference

### PCA

- one-block variance explanation
- no response or second-view supervision

### CCA

- two-block dependence modeling
- explicitly supervised by the second data block

So PCA is about structure inside $X$, while CCA is about structure linking $X$ and $Y$.

---

## Comparison Table

| Property | PCA | CCA |
|----------|-----|-----|
| Number of data blocks | One | Two |
| Objective | Max variance | Max correlation |
| Supervision | None | Two-view supervised |
| Requires inverses? | No | Yes, often via covariance inverses |
| Works easily when $p>N$? | Yes | Not without regularization |

---

## High-Dimensional Issue

This is a major exam point.

CCA requires inversion of covariance matrices such as $\Sigma_{XX}$ and $\Sigma_{YY}$.

So if:
$$
p > N
$$
or there is strong collinearity, standard CCA can break down.

Then you need:
- regularized CCA
- sparse CCA

PCA does not have this same issue in the same way.

---

## When to Use Which

**Use PCA when**:
- you want compression or denoising of one dataset
- variance structure itself is the target

**Use CCA when**:
- two different data views are available
- the goal is to understand shared structure
- cross-block association matters more than within-block variance

Example:
- gene expression vs metabolomics
- brain signals vs behavior
- image features vs text features

---

## Relation to PLS

CCA is often contrasted with PLS:

- CCA maximizes correlation
- PLS maximizes covariance

So if the exam broadens the comparison, this is a natural extension to mention.

---

## Limitations

1. PCA can miss cross-block predictive structure.
2. CCA can overfit or fail in high dimensions without regularization.
3. CCA components may be harder to interpret because they come in paired directions.

---

## Additional Possible Exam Questions

**Q: Why might PCA miss an important relationship that CCA finds?**
Because PCA only cares about variance in $X$, not whether that variance is related to $Y$.

**Q: Why does standard CCA struggle when $p \gg n$?**
Because the covariance matrices become singular and cannot be inverted without regularization.

**Q: What is the main conceptual difference in one line?**
PCA explains variance within one dataset; CCA explains correlation between two datasets.
