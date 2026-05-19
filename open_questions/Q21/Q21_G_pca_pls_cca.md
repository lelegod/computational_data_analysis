# Q21-G — PCA vs PLS vs CCA
> Related: 2022 Q17 — latent variable methods for dimensionality reduction

---

## The General Framework

All three find linear combinations (scores) of the input variables:
$$t = Xv \quad (\text{scores}), \quad v = \text{weight vector}$$

They differ in what they optimize and what data they use.

---

## PCA — Principal Component Analysis (Unsupervised)

**Objective**:
$$\max_v \text{Var}(Xv) = v^T \Sigma_{XX} v \quad \text{s.t.} \quad \|v\|=1$$

**Solution**: eigenvectors of $\Sigma_{XX} = X^TX/N$. First PC = direction of maximum variance in $X$.

**Properties**:
- Purely unsupervised — ignores $y$ entirely
- Components are orthogonal: $v_k^T v_{k'} = 0$ for $k \neq k'$
- Nested: the optimal $k$-component solution is always the first $k$ PCs
- Variance explained: $\sum_{j=1}^k \lambda_j / \sum_{j=1}^p \lambda_j$, where $\lambda_j$ are eigenvalues

**Limitation for regression**: the directions of maximum variance in $X$ may have zero correlation with $y$. PCR (PCA followed by regression) can miss the predictive signal.

---

## PLS — Partial Least Squares (Supervised)

**Objective**: maximize covariance between projected $X$ and projected $y$:
$$\max_{u,v} \text{Cov}(Xu, Yv) = \text{Var}(Xu)^{1/2} \cdot \text{Var}(Yv)^{1/2} \cdot \text{Corr}(Xu, Yv)$$

This balances two things simultaneously:
- Find high-variance directions in $X$ (predictors are informative)
- AND find directions correlated with $y$ (predictors are relevant)

**NIPALS algorithm** (most common PLS fitting):
1. Initialize $u = $ any column of $Y$
2. $v \leftarrow X^T u / \|X^Tu\|$; $t \leftarrow Xv$
3. $q \leftarrow Y^T t / \|Y^Tt\|$; $u \leftarrow Yq$
4. Repeat 2–3 until convergence
5. Deflate: remove the component from $X$ (and $Y$)

**Connection to ridge/OLS**: with $M=p$ PLS components, prediction = OLS; with $M<p$, PLS is a form of regularized regression. PLS shrinks small-variance directions more aggressively than PCR.

---

## CCA — Canonical Correlation Analysis (Two-Sided Supervised)

**Objective**: maximize correlation between projected $X$ and projected $Y$ (two separate matrices):
$$\max_{u,v} \text{Corr}(Xu, Yv) = \frac{u^T\Sigma_{XY}v}{\sqrt{u^T\Sigma_{XX}u \cdot v^T\Sigma_{YY}v}}$$

This maximizes ONLY correlation — it ignores internal variance of $X$ and $Y$.

**Solution**: generalized eigenvalue problem:
$$\Sigma_{XX}^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX} u = \rho^2 u$$

**Critical requirement**: $\Sigma_{XX}$ and $\Sigma_{YY}$ must be invertible → fails when $p > N$ or when variables are collinear.

**Solutions for high dimensions**:
- **Regularized CCA**: $\hat{\Sigma}_{XX} \leftarrow \hat{\Sigma}_{XX} + \lambda_x I$, similarly for $\hat{\Sigma}_{YY}$
- **Sparse CCA** (PMD): adds $L_1$ penalties to $u$ and $v$ — finds sparse canonical variates

---

## Comparison Table

| Property | PCA | PLS | CCA |
|----------|-----|-----|-----|
| Supervision | No | Yes ($Y$ guides $X$) | Two-sided ($X$ and $Y$) |
| Objective | Max Var($Xv$) | Max Cov($Xu$, $Yv$) | Max Corr($Xu$, $Yv$) |
| Requires $\Sigma^{-1}$? | No | No | Yes ($\Sigma_{XX}$, $\Sigma_{YY}$) |
| Works when $p > N$? | Yes | Yes | No (needs regularization) |
| Nested components? | Yes | No | No |
| Number of components | Up to $p$ | Up to $\min(p,q,N)$ | Up to $\min(p,q)$ |
| Ignores variance | No | No | Yes (corr only) |
| Missing signal risk | Yes (may miss $y$-corr) | No | No |

---

## Key Conceptual Distinctions

**PCA vs PLS**: PCA finds directions that explain $X$ well; PLS finds directions that predict $y$ well. When $X$ has noisy irrelevant features (e.g., spectroscopy with thousands of wavelengths), PLS finds the few predictive directions while PCA wastes components on high-variance noise.

**PLS vs CCA**: PLS maximizes covariance (variance × correlation); CCA maximizes correlation only. CCA is scale-invariant but ignores variance — it can find directions in $X$ with very small variance that happen to be correlated with $Y$, which may be noise-driven. PLS naturally balances both.

**All three can be made sparse**: Sparse PCA (elastic net penalty), sparse PLS (SPLS), sparse CCA (PMD) — add $L_1$ penalties to $v$ (and $u$) to perform variable selection simultaneously.

---

## Additional Possible Exam Questions

**Q: When would you use PCA for regression (PCR) vs PLS?**
PCR: when $X$ has interpretable variance structure and you want dimensionality reduction without supervision. PLS: when you have many correlated predictors and a response $y$ — PLS finds the latent structure in $X$ most relevant to $y$. In practice, PLS almost always outperforms PCR for prediction tasks because it doesn't waste components on irrelevant variance.

**Q: What does "deflation" mean in PLS?**
After extracting one PLS component (scores $t$, loadings $v$, weights $q$), subtract the component from the data matrices:
$$X \leftarrow X - tp^T, \quad Y \leftarrow Y - tq^T$$
This removes the variance explained by the first component so the next component is orthogonal to it (in score space, not loading space).

**Q: Why are PLS and PCA components orthogonal in different senses?**
PCA: loadings $v_k$ are orthogonal ($v_k^T v_{k'} = 0$) AND scores $t_k$ are orthogonal. PLS: scores $t_k$ are orthogonal (by deflation), but loadings $v_k$ are NOT necessarily orthogonal to each other. This matters for interpreting the weight vectors.

**Q: What is the connection between CCA and LDA?**
LDA can be formulated as a special case of CCA where $Y$ is the indicator matrix of class labels. The canonical variates in this case correspond to the LDA discriminant directions. Fisher's linear discriminant = first canonical variate from CCA($X$, class-indicator $Y$).

**Q: How many canonical variates does CCA produce?**
At most $\min(\text{rank}(X), \text{rank}(Y)) = \min(p, q, N)$ canonical pairs. Each subsequent pair $(u_k, v_k)$ is orthogonal (in the sense $u_k^T\Sigma_{XX}u_{k'}=0$) to all previous pairs and has lower canonical correlation $\rho_k \geq \rho_{k+1}$.
