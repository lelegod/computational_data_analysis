# Q21-W — Sparse PCA and Variable Selection in Dimensionality Reduction
> Week 8. Extension of PCA; could ask to explain sparse PCA, contrast with PCA, or describe PMD.

---

## Why Sparse PCA?

Standard PCA loadings $v_k$ are dense — every variable contributes to every component. This makes interpretation hard: a PC that is a linear combination of all 10,000 genes is not scientifically meaningful.

**Sparse PCA**: add a sparsity penalty to the loadings so each PC depends on only a few variables. Sacrifice some variance explained in exchange for interpretability.

---

## The Sparse PCA Objective

Standard PCA: $\max_v v^T \Sigma_{XX} v$ s.t. $\|v\|_2 = 1$

**Sparse PCA (Zou, Hastie & Tibshirani 2006)**: reformulate as a regression problem and add Lasso penalty:
$$\min_{A,B} \|X - XBA^T\|_F^2 + \lambda\|B\|_1 \quad \text{s.t.} \quad A^TA = I_K$$

- $A$: orthonormal loading matrix (rotation)
- $B$: sparse coefficient matrix (enforces sparsity via $L_1$)
- Alternating optimization: fix $A$, solve lasso for $B$; fix $B$, update $A$ via SVD

**Elastic net version**: replace $\|B\|_1$ with $\lambda_1\|B\|_1 + \lambda_2\|B\|_2^2$ — handles correlated variables better (groups correlated features into the same component).

---

## PMD — Penalized Matrix Decomposition (Witten, Tibshirani & Hastie 2009)

More general framework. For a single component:
$$\max_{u,v} u^T X v \quad \text{s.t.} \quad \|u\|_2=1, \|v\|_2=1, P_1(u)\leq c_1, P_2(v)\leq c_2$$

where $P_1, P_2$ are penalty functions (e.g., $L_1$).

- Standard SVD: no penalties ($c_1=\infty$, $c_2=\infty$) → dense $u,v$
- Sparse PCA: $P_2(v)=\|v\|_1 \leq c_2$, no penalty on $u$ → sparse loadings
- Sparse CCA: $L_1$ penalty on both $u$ and $v$ → sparse canonical variates

**Algorithm**: soft-threshold iteration:
1. $v \leftarrow S(X^Tu, c_2)/\|S(X^Tu, c_2)\|_2$
2. $u \leftarrow S(Xv, c_1)/\|S(Xv, c_1)\|_2$
3. Repeat until convergence

where $S(x,\lambda)_j = \text{sign}(x_j)\max(|x_j|-\lambda, 0)$ is soft-thresholding.

---

## Sparse PCA vs Standard PCA

| Property | PCA | Sparse PCA |
|----------|-----|-----------|
| Loadings | Dense (all vars) | Sparse (few vars) |
| Variance explained | Maximum | Reduced (sacrifice for sparsity) |
| Components orthogonal? | Yes (loadings + scores) | Approximately only |
| Interpretable? | No (all vars mix) | Yes (identify which vars drive component) |
| Unique? | Yes (up to sign) | Yes (given penalty) |
| Computationally | Eigendecomposition ($O(p^3)$) | Iterative, more expensive |
| High-dim ($p\gg n$)? | Yes | Yes |

---

## Connection to Lasso Regression

Sparse PCA via the regression formulation: each PC loading is the solution to a Lasso problem. The sparsity-inducing $L_1$ penalty shrinks small loadings to exactly zero, just as Lasso shrinks irrelevant regression coefficients to zero.

This connects sparse PCA to the broader theme of $L_1$ regularization as a variable selection tool — the same mechanism that gives Lasso sparsity in regression gives sparse PCA interpretable loadings.

---

## Sparse CCA and Sparse PLS

The PMD framework extends directly:
- **Sparse CCA**: find sparse canonical directions in both $X$ and $Y$ — relevant for identifying which genes (in $X$) associate with which clinical traits (in $Y$)
- **Sparse PLS**: add $L_1$ penalty to PLS weight vectors — variable selection while maximizing covariance

All follow the same principle: replace dense linear combinations with sparse ones via $L_1$ regularization.

---

## Additional Possible Exam Questions

**Q: What is the tradeoff when choosing the sparsity parameter $\lambda$ in sparse PCA?**
Large $\lambda$: very sparse loadings (few nonzero) → more interpretable but less variance explained. Small $\lambda$: dense loadings → more variance explained but harder to interpret. Cross-validation on variance explained or permutation testing (compare to null data) can guide selection, but there is no single correct answer — the choice depends on how much interpretability you are willing to trade for variance explained.

**Q: Why are sparse PCA components not exactly orthogonal?**
Standard PCA components are orthogonal because the eigendecomposition of a symmetric matrix always produces orthogonal eigenvectors. Sparse PCA adds a non-orthogonal constraint (sparsity) — the resulting loadings optimize a penalized objective that does not enforce orthogonality. In practice, sparse PCA components are approximately orthogonal if the sparsity patterns are non-overlapping (different variables in each component).

**Q: When would you use sparse PCA over standard PCA in genomics?**
Standard PCA on gene expression data produces components that are mixtures of thousands of genes — not biologically interpretable. Sparse PCA identifies a small set of genes (e.g., 20–50) that drive each component. These gene sets can be tested for enrichment in biological pathways. Sparse PCA is preferred whenever interpretability matters more than maximizing variance explained — which is almost always the case in biology.

**Q: What is the difference between sparse PCA and factor analysis?**
Factor analysis: $X = LF + \varepsilon$ where $L$ is a loading matrix and $F$ are latent factors. Factors are assumed to have a specific distribution (often Gaussian), and the model is generative — $\varepsilon$ captures observation-specific noise. Sparse PCA is purely discriminative (no noise model), finds deterministic loadings via optimization. Factor analysis estimates a full covariance model; sparse PCA is a pure dimensionality reduction/rotation tool.
