# Q21-AF — Canonical Correlation Analysis (CCA)
> Week 8. Could ask: derive the CCA objective, explain generalized eigenproblem, and compare CCA to PLS/PCA.

---

## Problem Setup

Given two centered data matrices measured on the same $N$ samples:

- $X \in \mathbb{R}^{N \times p}$
- $Y \in \mathbb{R}^{N \times q}$

CCA finds linear combinations

$$
u = Xa,\qquad v = Yb
$$

that are maximally correlated.

---

## Objective

First canonical pair $(a_1,b_1)$ solves

$$
\max_{a,b}\ \operatorname{Corr}(Xa,Yb)
=
\max_{a,b}\ \frac{a^T\Sigma_{XY}b}
{\sqrt{a^T\Sigma_{XX}a}\sqrt{b^T\Sigma_{YY}b}}
$$

subject to normalization constraints:

$$
a^T\Sigma_{XX}a = 1,\qquad b^T\Sigma_{YY}b = 1.
$$

Later pairs are found with orthogonality constraints in the canonical variates.

---

## Derivation Sketch

Using Lagrange multipliers, optimal vectors satisfy:

$$
\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}a = \rho^2\Sigma_{XX}a
$$

and symmetrically for $b$:

$$
\Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY}b = \rho^2\Sigma_{YY}b.
$$

So CCA reduces to a generalized eigenvalue problem; eigenvalues are squared canonical correlations $\rho_k^2$.

---

## Interpretation

- $u_k = Xa_k$ and $v_k = Yb_k$ are the $k$th canonical variates.
- $\rho_k = \operatorname{Corr}(u_k, v_k)$ quantifies shared cross-block signal.
- CCA ranks modes of dependence between two views.

Unlike PCA, CCA is not about variance within one block; it is about association across blocks.

---

## CCA vs PCA vs PLS

| Method | Objective | Uses both blocks? | Needs matrix inversion? |
|--------|-----------|-------------------|-------------------------|
| PCA | Maximize variance in one block | No | No |
| PLS | Maximize covariance $\operatorname{Cov}(Xa, Yb)$ | Yes | Usually no direct inversion |
| CCA | Maximize correlation $\operatorname{Corr}(Xa, Yb)$ | Yes | Yes, $\Sigma_{XX}^{-1}, \Sigma_{YY}^{-1}$ |

Key distinction:

- PLS prefers components with large covariance (variance times correlation).
- CCA normalizes by variance and focuses purely on correlation strength.

---

## High-Dimensional Issue and Regularized CCA

If $p \ge N$ or $q \ge N$, covariance matrices are singular and classical CCA fails.

Regularized CCA replaces

$$
\Sigma_{XX}^{-1} \to (\Sigma_{XX} + \lambda_x I)^{-1},\quad
\Sigma_{YY}^{-1} \to (\Sigma_{YY} + \lambda_y I)^{-1}.
$$

Sparse CCA adds $L_1$ penalties to $a,b$ for interpretability in omics/high-dimensional settings.

---

## Statistical Testing

Typical null hypothesis for dimension $k$:

$$
H_0:\rho_k=\rho_{k+1}=\cdots=0
$$

Classical tests use Wilks' Lambda (with asymptotic approximations); permutation tests are often preferred in modern high-dimensional applications.

---

## Limitations

1. Sensitive to scaling and outliers (standardize and consider robust preprocessing).
2. Classical CCA is unstable in high dimensions without regularization.
3. Linear method: misses nonlinear dependencies unless kernel CCA is used.
4. Canonical vectors can be hard to interpret when many variables load weakly.

