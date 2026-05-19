# Q21-AC — Elastic Net
> Week 3. Could ask: derive the coordinate descent update, explain grouped selection, compare to Ridge and Lasso.

---

## The Model

Elastic Net adds both $L_1$ and $L_2$ penalties to OLS:
$$\hat{\beta}_\text{EN} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\bigl[\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2/2\bigr]$$

- $\alpha \in [0,1]$: mixing parameter. $\alpha=1$ recovers Lasso; $\alpha=0$ recovers Ridge (up to reparameterisation of $\lambda$).
- $\lambda > 0$: overall regularization strength.
- Two hyperparameters: the penalty contour is a **rounded diamond** — the $L_1$ corners are smoothed by the $L_2$ term.

---

## Mechanism: Coordinate Descent Update

Elastic Net has no closed-form solution (the $L_1$ term is non-differentiable at zero), but coordinate descent is efficient: cycle through each $\beta_j$ while holding the others fixed.

**Partial residual**: $r_j = y - X_{-j}\hat{\beta}_{-j}$ (residuals with predictor $j$ removed). The OLS estimate for $\beta_j$ given the others is:
$$z_j = \frac{1}{n}X_j^T r_j$$

**Elastic Net coordinate update**:
$$\hat{\beta}_j = \frac{S(z_j,\, \lambda\alpha)}{1 + \lambda(1-\alpha)}$$

where $S(z,\lambda) = \text{sign}(z)(|z|-\lambda)_+ = \text{sign}(z)\max(|z|-\lambda, 0)$ is the **soft-threshold operator**.

**Interpretation**:
- The numerator $S(z_j, \lambda\alpha)$ is the Lasso soft-threshold — sets $\hat{\beta}_j = 0$ if $|z_j| \leq \lambda\alpha$.
- The denominator $1 + \lambda(1-\alpha)$ is Ridge-like shrinkage — divides the surviving coefficient by a constant > 1.
- So Elastic Net first thresholds (via $L_1$) then shrinks (via $L_2$), in one closed-form step.

For Lasso ($\alpha=1$): denominator $= 1$, update reduces to plain soft-threshold.
For Ridge ($\alpha=0$): numerator has threshold $0$ (no zeroing), denominator $= 1+\lambda$, giving the Ridge analytical solution one coordinate at a time.

---

## Key Properties

### Grouped Selection
When predictors are highly correlated (e.g., $X_j \approx X_k$), Lasso arbitrarily picks one and zeros the rest. Elastic Net **tends to include or exclude correlated predictors together** (the "grouping effect"). Formally: if $X_j = X_k$ exactly, Elastic Net gives them equal coefficients; Lasso can assign arbitrary splits.

### Handles $p \gg n$
Lasso can select at most $n$ variables (once $n$ predictors enter the active set, the Gram matrix is saturated). Elastic Net has no such limitation — the $L_2$ component keeps the problem well-posed for $p \gg n$, allowing more than $n$ non-zero coefficients.

### Sparsity
Elastic Net still produces **exact zeros** (unlike pure Ridge), because the $L_1$ component provides a kink at zero. The rounded-diamond penalty shape still has corners (at the axes), just smoother ones.

### Bias-Variance
- More bias than Lasso (extra $L_2$ shrinkage on top of thresholding).
- Less variance than Lasso (Ridge component stabilizes the solution when predictors are correlated).
- More bias than Ridge (the $L_1$ component can zero out predictors entirely).
- Sparser than Ridge (produces exact zeros).

### Degrees of Freedom
Approximately equal to the number of non-zero coefficients, like Lasso. Ridge has fractional df; Elastic Net is closer to Lasso in its df behavior.

---

## Comparison Table

| Property | Ridge | Lasso | Elastic Net |
|----------|-------|-------|-------------|
| Penalty | $\|\beta\|_2^2$ | $\|\beta\|_1$ | $\alpha\|\beta\|_1+(1-\alpha)\|\beta\|_2^2/2$ |
| Penalty shape | Sphere | Diamond | Rounded diamond |
| Closed-form? | Yes | No | No |
| Exact zeros? | No | Yes | Yes |
| Works $p > n$? | Yes | $\leq n$ vars | Yes |
| Correlated predictors | Keeps all (proportional shrink) | Picks one arbitrarily | Groups together |
| Hyperparameters | $\lambda$ | $\lambda$ | $\lambda$, $\alpha$ (2D grid) |

---

## Connection to LARS

The **LARS-EN** algorithm (Zou & Hastie 2005) extends the LARS path algorithm to Elastic Net. It augments the design matrix $X$ with $\sqrt{\lambda(1-\alpha)} I$ rows, then runs LARS-Lasso on the augmented system. This computes the full Elastic Net path efficiently in $O(p^2 n)$ time.

---

## Tuning

Two hyperparameters require a 2D cross-validation grid:
- In practice, often fix $\alpha \in \{0.1, 0.5, 0.9, 1\}$ and tune $\lambda$ → reduces to 1D CV per $\alpha$ value.
- Or use a coarse 2D grid (e.g., $\alpha \in \{0.25, 0.5, 0.75, 1\}$, fine $\lambda$ grid).
- The `glmnet` package (R) parameterises as `alpha` (= $\alpha$) and `lambda` and computes the full path for each $\alpha$.

---

## Limitations

1. **Two hyperparameters**: 2D cross-validation is expensive; $\alpha$ must often be fixed heuristically.
2. **Not as interpretable as Lasso** when $\alpha < 1$: shrinkage is partly continuous (like Ridge), so coefficient magnitudes are harder to interpret.
3. **Bias**: when the true model is sparse and predictors are independent, Lasso may outperform Elastic Net (no benefit to the $L_2$ component, only extra bias).
4. **Grouped selection is approximate**: the grouping property holds exactly only for orthogonal/identical predictors; in practice it is an empirical tendency, not a guarantee.

---

## Additional Possible Exam Questions

**Q: When does Lasso fail and Elastic Net helps?**
Two main failure modes: (1) When $p > n$, Lasso selects at most $n$ predictors — the Gram matrix becomes saturated. Elastic Net has no such limit because the $L_2$ component ensures the problem is always well-posed. (2) When predictors are highly correlated (e.g., SNPs in linkage disequilibrium), Lasso picks one gene arbitrarily and zeros the rest, giving unstable and uninterpretable results. Elastic Net's grouping effect keeps correlated predictors together, giving more stable and biologically meaningful solutions.

**Q: What does the Elastic Net penalty look like geometrically?**
A **rounded diamond** (also called a "squircle" between the $L_1$ diamond and $L_2$ sphere). The corners of the $L_1$ diamond (on the coordinate axes) are smoothed by the $L_2$ term, but the shape still bulges toward the axes more than a sphere does. This means exact zeros are still likely (corners exist), but the sparsity pattern is more stable than pure $L_1$.

**Q: What is the grouped selection property precisely?**
If two predictors $X_j$ and $X_k$ are identical ($X_j = X_k$), then the Elastic Net solution always sets $\hat{\beta}_j = \hat{\beta}_k$ (they receive the same coefficient). Lasso gives degenerate solutions with $\hat{\beta}_j + \hat{\beta}_k = c$ for any constant $c$ — an entire line of solutions. Elastic Net resolves this degeneracy by picking the symmetric point. More generally, for correlated (not identical) predictors, Elastic Net encourages similar coefficient magnitudes.

**Q: Write the coordinate descent update for $\beta_j$ and identify the two terms.**
$\hat{\beta}_j = S(z_j, \lambda\alpha) / (1 + \lambda(1-\alpha))$ where $z_j = X_j^T r_j / n$ is the partial residual OLS estimate. The soft-threshold in the numerator is the $L_1$ term (creates sparsity). The division in the denominator is the $L_2$ term (shrinks the surviving coefficient toward zero). Both happen in one step — coordinate descent converges to the global minimum because the Elastic Net objective is strictly convex (the $L_2$ term makes it strongly convex).
