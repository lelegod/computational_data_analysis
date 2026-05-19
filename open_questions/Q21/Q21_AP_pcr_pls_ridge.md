# Q21-AP — PCR vs PLS vs Ridge
> Weeks 1/8. Could ask: compare PCR, PLS, and Ridge as ways to stabilize regression with many correlated predictors, and explain how each method treats low-variance directions.

---

## The Shared Problem

All three methods are responses to the same issue:

- many correlated predictors
- unstable OLS estimates
- risk of overfitting

They all regularize regression, but in different ways.

---

## PCR

Principal Component Regression first performs PCA on $X$, then regresses $y$ on the first $M$ principal components.

### Key idea

- unsupervised dimension reduction
- keep high-variance directions in $X$
- throw away the rest

In SVD form:
$$
X = UDV^T
$$
PCR keeps only the first $M$ singular directions.

### Consequence

PCR performs **hard truncation**:
- retained directions are fully kept
- discarded directions are completely removed

---

## PLS

Partial Least Squares also builds latent directions, but unlike PCR, it uses $y$ while constructing them.

### Key idea

PLS chooses directions in $X$ that have high covariance with $y$.

So it balances:
- variance in $X$
- relevance for predicting $y$

### Consequence

PLS is supervised latent-variable regression, while PCR is unsupervised latent-variable regression.

---

## Ridge

Ridge keeps all original predictor directions but shrinks them continuously:
$$
\hat{\beta}_{\text{ridge}} = (X^TX+\lambda I)^{-1}X^Ty
$$

In SVD form:
$$
\hat{\beta}_{\text{ridge}}
=
\sum_j \frac{d_j^2}{d_j^2+\lambda} v_j \frac{u_j^T y}{d_j}
$$

### Key idea

- no hard variable/direction removal
- all directions remain
- low-variance directions get shrunk more strongly

So Ridge is **continuous shrinkage**, unlike PCR.

---

## The Core Comparison

### PCR

- unsupervised
- hard truncation
- can discard predictive low-variance directions

### PLS

- supervised
- latent directions built using $y$
- usually better for prediction than PCR

### Ridge

- supervised through the regression fit
- no latent compression step
- smooth shrinkage instead of truncation

---

## Comparison Table

| Property | PCR | PLS | Ridge |
|----------|-----|-----|-------|
| Uses $y$ to build components? | No | Yes | Yes |
| Main mechanism | Keep top PCs | Covariance-driven latent factors | Shrink all coefficients |
| Shrinkage type | Hard truncation | Supervised low-rank regularization | Continuous shrinkage |
| Keeps all directions? | No | No | Yes |
| Risk | Miss predictive low-variance directions | More complex interpretation | No dimension reduction |

---

## Why PCR Can Fail

PCR assumes that directions with high variance in $X$ are also the important ones for predicting $y$.

That need not be true.

A low-variance direction can still have strong predictive power, and PCR may throw it away. PLS is designed to avoid exactly this problem.

---

## Why Ridge Is Different from Both

Ridge does not replace the predictors by latent components.

Instead, it leaves the regression in the original feature space and shrinks unstable directions smoothly. This often gives more stable prediction without the abrupt cutoff of PCR.

---

## When to Use Which

**Use PCR when**:
- you want dimension reduction and orthogonal latent variables
- the variance structure in $X$ itself is meaningful

**Use PLS when**:
- prediction is the priority
- predictors are many and highly collinear
- you want latent variables guided by the response

**Use Ridge when**:
- you want a simple stable regression baseline
- the signal is dense rather than sparse
- you do not need latent-component interpretation

---

## Limitations

1. PCR can discard predictive low-variance directions.
2. PLS is harder to interpret than PCR.
3. Ridge does not produce sparse solutions.
4. All three require tuning component count or penalty strength.

---

## Additional Possible Exam Questions

**Q: What is the key conceptual difference between PCR and PLS?**
PCR uses only the predictor covariance structure, while PLS uses the relationship between predictors and response during component construction.

**Q: Why is Ridge often described as smoother than PCR?**
Because PCR makes binary keep/drop decisions on principal directions, while Ridge shrinks each direction continuously by a factor depending on its singular value.

**Q: Which method is most likely to preserve a predictive low-variance direction?**
PLS or Ridge. PCR is most likely to discard it because PCR ranks directions only by variance in $X$.
