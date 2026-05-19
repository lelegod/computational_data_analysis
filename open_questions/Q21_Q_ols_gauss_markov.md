# Q21-Q — OLS and the Gauss-Markov Theorem
> Week 1. Could be asked to prove unbiasedness, state Gauss-Markov, or compare OLS to Ridge.

---

## The OLS Estimator

**Model**: $y = X\beta + \varepsilon$, where $X\in\mathbb{R}^{N\times p}$, $\beta\in\mathbb{R}^p$, $\varepsilon\in\mathbb{R}^N$.

**Objective**: minimize residual sum of squares:
$$\hat{\beta}_\text{OLS} = \arg\min_\beta \|y - X\beta\|^2$$

**Closed-form solution** (assuming $X^TX$ is invertible):
$$\hat{\beta}_\text{OLS} = (X^TX)^{-1}X^Ty$$

**Derivation**: set gradient to zero:
$$\frac{\partial}{\partial\beta}\|y-X\beta\|^2 = -2X^T(y-X\beta) = 0 \;\Rightarrow\; X^TX\beta = X^Ty \;\Rightarrow\; \hat{\beta} = (X^TX)^{-1}X^Ty$$

---

## OLS is Unbiased

**Assumption 1**: correct specification — $E[y|X] = X\beta$ (true model is linear)
**Assumption 2**: strict exogeneity — $E[\varepsilon|X] = 0$ (errors uncorrelated with predictors)

**Proof**:
$$\hat{\beta} = (X^TX)^{-1}X^Ty = (X^TX)^{-1}X^T(X\beta+\varepsilon) = \beta + (X^TX)^{-1}X^T\varepsilon$$

Taking expectation:
$$E[\hat{\beta}|X] = \beta + (X^TX)^{-1}X^TE[\varepsilon|X] = \beta + 0 = \beta$$

Therefore $E[\hat{\beta}] = \beta$ — OLS is unbiased.

**Variance of OLS estimator** (assuming $\text{Var}(\varepsilon|X)=\sigma^2 I$):
$$\text{Var}(\hat{\beta}|X) = (X^TX)^{-1}X^T(\sigma^2 I)X(X^TX)^{-1} = \sigma^2(X^TX)^{-1}$$

---

## The Gauss-Markov Theorem

**Statement**: Under assumptions (1) correct specification, (2) exogeneity $E[\varepsilon|X]=0$, (3) homoscedasticity $\text{Var}(\varepsilon|X)=\sigma^2 I$, and (4) no multicollinearity ($X$ full rank):

**OLS is BLUE** — Best Linear Unbiased Estimator.

- **Linear**: $\hat{\beta} = Cy$ for some matrix $C$ (linear in $y$)
- **Unbiased**: $E[\hat{\beta}]=\beta$
- **Best**: minimum variance among all linear unbiased estimators

**Proof sketch**: let $\tilde{\beta} = Ay$ be any other linear unbiased estimator. Unbiasedness requires $AX=I$. Then:
$$\text{Var}(\tilde{\beta}) = \sigma^2 AA^T = \sigma^2(A-C+C)(A-C+C)^T = \text{Var}(\hat{\beta}) + \sigma^2(A-C)(A-C)^T$$

Since $(A-C)(A-C)^T \succeq 0$ (positive semi-definite), $\text{Var}(\tilde{\beta}) \succeq \text{Var}(\hat{\beta})$.

---

## OLS vs Ridge: Breaking Gauss-Markov Intentionally

Ridge introduces bias to reduce variance:
$$\hat{\beta}_\text{ridge} = (X^TX+\lambda I)^{-1}X^Ty = \beta + (X^TX+\lambda I)^{-1}X^T\varepsilon - \lambda(X^TX+\lambda I)^{-1}\beta$$

**Bias**: $E[\hat{\beta}_\text{ridge}]-\beta = -\lambda(X^TX+\lambda I)^{-1}\beta \neq 0$

**Variance**: $\text{Var}(\hat{\beta}_\text{ridge}) = \sigma^2(X^TX+\lambda I)^{-1}X^TX(X^TX+\lambda I)^{-1} \prec \sigma^2(X^TX)^{-1}$

Gauss-Markov says OLS is best among **unbiased** estimators. Ridge is biased — it escapes the constraint and can achieve lower EPE = Bias$^2$ + Var.

**When Ridge beats OLS**: when the variance reduction from $\lambda > 0$ exceeds the squared bias introduced. This is always possible when eigenvalues of $X^TX$ are small (near-singular, correlated predictors).

---

## Additional Possible Exam Questions

**Q: What happens to OLS when $X^TX$ is singular?**
$(X^TX)^{-1}$ does not exist. Occurs when $p > N$ or when columns of $X$ are linearly dependent (perfect multicollinearity). The normal equations $X^TX\beta = X^Ty$ still have solutions (infinitely many — any $\beta$ in the affine solution space). OLS is not unique. Fix: add regularization (Ridge: $(X^TX+\lambda I)^{-1}$ is always invertible for $\lambda>0$), use the Moore-Penrose pseudo-inverse, or reduce dimensionality first.

**Q: What does the Gauss-Markov theorem NOT guarantee?**
(1) OLS is not necessarily the best estimator among ALL estimators — biased estimators (Ridge) can have lower MSE. (2) It does not guarantee normality of $\hat{\beta}$ (that requires Gaussian $\varepsilon$). (3) It does not guarantee good prediction performance (EPE) — only minimum variance among linear unbiased estimators. (4) It breaks down under heteroscedasticity ($\text{Var}(\varepsilon_i)\neq\sigma^2$) or autocorrelation — then Weighted/Generalized Least Squares is BLUE.

**Q: Prove that the OLS residuals are orthogonal to the fitted values.**
$\hat{\varepsilon} = y - X\hat{\beta} = y - X(X^TX)^{-1}X^Ty = (I-H)y$ where $H=X(X^TX)^{-1}X^T$ is the hat matrix. Then $\hat{y}^T\hat{\varepsilon} = (Hy)^T(I-H)y = y^TH(I-H)y = y^T(H-H^2)y = 0$ since $H$ is idempotent ($H^2=H$). This decomposition $\|y\|^2 = \|\hat{y}\|^2 + \|\hat{\varepsilon}\|^2$ is the basis for $R^2$.

**Q: What is the leverage $h_{ii}$ and why does it matter?**
$h_{ii} = [X(X^TX)^{-1}X^T]_{ii} \in [1/N, 1]$ — the influence of observation $i$ on its own fitted value. High leverage: observation $i$ has unusual $x_i$ (far from the mean) and the model is pulled toward it. LOO-CV error for linear models: $\text{CV}_\text{LOO} = \frac{1}{N}\sum_i\left(\frac{y_i-\hat{y}_i}{1-h_{ii}}\right)^2$ — one formula, no refitting needed.
