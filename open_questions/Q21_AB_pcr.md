# Q21-AB — Principal Component Regression (PCR) vs PLS
> Week 8. PCR uses PCA scores for regression — compare to PLS which is supervised.

---

## The Motivation: OLS Fails When $p \geq N$ or Features Are Correlated

OLS: $\hat{\beta} = (X^TX)^{-1}X^Ty$ — breaks when $X^TX$ is singular or near-singular:
- $p \geq N$: not enough observations to estimate $p$ coefficients
- High multicollinearity: $X^TX$ is nearly singular → unstable estimates with huge variance

**Solution**: reduce $X$ to $M \ll p$ uncorrelated components, then regress $y$ on those.

---

## Principal Component Regression (PCR)

**Step 1**: run PCA on $X$ → get $M$ principal components (PCs):
$$Z_m = Xv_m, \quad m=1,\ldots,M$$
where $v_m$ are the top eigenvectors of $X^TX$.

**Step 2**: regress $y$ on the PC scores $Z = [Z_1,\ldots,Z_M]$:
$$\hat{\beta}_\text{PCR} = V_M(Z^TZ)^{-1}Z^Ty = V_M\Lambda_M^{-1}V_M^TX^Ty$$

Since PCs are orthogonal ($Z^TZ = \Lambda_M$ diagonal), the regression is simply $M$ independent univariate regressions.

**Prediction**: $\hat{y} = Z\hat{\gamma} = Xv_m\hat{\gamma}_m$

**SVD view**: $X = UDV^T$, so the PCR estimator is:
$$\hat{\beta}_\text{PCR} = \sum_{m=1}^M \frac{d_m^2}{d_m^2} v_m \frac{u_m^Ty}{d_m} = \sum_{m=1}^M v_m\frac{u_m^Ty}{d_m}$$

Compare to Ridge: $\hat{\beta}_\text{ridge} = \sum_m \frac{d_m^2}{d_m^2+\lambda}v_m\frac{u_m^Ty}{d_m}$ — Ridge shrinks all directions continuously; PCR truncates small-variance directions entirely.

---

## PCR vs Ridge

| Property | PCR | Ridge |
|----------|-----|-------|
| Mechanism | Discard small-variance PCs entirely | Shrink all PCs continuously |
| Bias | Discrete (removes whole PCs) | Continuous (smooth shrinkage) |
| Variance | Reduced (fewer params) | Reduced (shrinkage) |
| Hyperparameter | $M$ (integer, number of PCs) | $\lambda$ (continuous) |
| Selects components by | Variance in $X$ only | Implicitly (regularization) |
| Ignores $y$ in feature selection | Yes | No (implicitly) |

**Key weakness of PCR**: PCA selects components that explain $X$ variance — these may not be the components that predict $y$. If the first few PCs of $X$ happen to be uncorrelated with $y$ (irrelevant noise has high variance), PCR keeps them and discards the low-variance predictive directions.

---

## PLS Fixes the PCR Weakness

PLS: $\max_{u,v} \text{Cov}(Xu, Yv)$ — finds directions that jointly explain $X$ variance AND correlate with $y$.

**PCR vs PLS in words**:
- PCR: "which directions in $X$ have maximum variance?" (ignore $y$)
- PLS: "which directions in $X$ most predict $y$?" (use $y$ to guide)

**When PCR ≈ PLS**: when the high-variance directions of $X$ are also the most predictive of $y$. This is common in practice when the signal is strong.

**When PLS beats PCR**: when predictive information is in low-variance directions of $X$ (e.g., a rare but important feature). PCR would miss this; PLS would find it.

---

## Full Comparison: OLS, Ridge, PCR, PLS

| Method | Uses $y$? | Closed-form? | Works $p>N$? | Zeros out directions? | Continuous shrinkage? |
|--------|----------|-------------|-------------|----------------------|----------------------|
| OLS | Yes | Yes | No | No | No |
| Ridge | Yes | Yes | Yes | No | Yes |
| PCR | No (step 1) | Yes | Yes | Yes (hard truncation) | No |
| PLS | Yes | Yes (NIPALS) | Yes | No | Implicit |
| Lasso | Yes | No | Yes ($\leq N$) | Yes (feature-level) | No |

---

## Choosing M in PCR

**Scree plot**: plot variance explained vs $M$; look for elbow.

**Cross-validation**: compute CV prediction error vs $M$; choose by 1-SE rule.

**Bias-variance**: small $M$ = high bias (throw away predictive info), low variance. Large $M$ = low bias, higher variance. Optimal $M$ via CV.

---

## Additional Possible Exam Questions

**Q: Why does PCR use the top $M$ PCs and not some other subset?**
PCA components are ordered by decreasing variance. The top $M$ PCs explain the most variance in $X$. Discarding the bottom PCs (low variance) is equivalent to removing noise directions — directions with little variability in $X$ are likely to be noise. However, this heuristic can fail when predictive information is in low-variance directions.

**Q: Prove that with $M=p$ PCR reduces to OLS.**
With $M=p$ (all PCs included): $V_M = V$ (all eigenvectors, orthonormal). PCR estimate: $\hat{\beta}_\text{PCR} = V\Lambda^{-1}V^TX^Ty = (V\Lambda V^T)^{-1}X^Ty$. Since $X^TX = V\Lambda V^T$, this gives $(X^TX)^{-1}X^Ty = \hat{\beta}_\text{OLS}$.

**Q: How does PCR compare to Lasso for variable selection?**
PCR: selects/discards entire principal components (dense linear combinations of all features). Lasso: selects individual features (sets specific $\hat{\beta}_j=0$). PCR gives no interpretability in terms of original variables; Lasso gives a sparse set of relevant original variables. For scientific interpretation, Lasso is usually preferred. For prediction with many correlated predictors, PCR and Lasso give similar performance.

**Q: What is the effective degrees of freedom of PCR?**
PCR with $M$ components: $\text{df}(M) = M$ (exactly $M$ free parameters in the projected space). Ridge with penalty $\lambda$: $\text{df}(\lambda) = \sum_m d_m^2/(d_m^2+\lambda) \in (0,p)$ (fractional). PCR has integer df (discrete steps); Ridge interpolates continuously. This means Ridge's model complexity can be tuned more finely than PCR.
