# Week 4 — Lecture Notes
## Computational Data Analysis (02582)

---

## LDA vs Logistic Regression

### Key Difference: Generative vs Discriminative

| | LDA | Logistic Regression |
|---|---|---|
| Model type | Generative | Discriminative |
| What it models | $p(x, y)$ | $p(y \mid x)$ |
| Assumptions | Gaussian features, equal covariance | None on $p(x)$ |
| Robustness | Less robust | More robust |

---

### LDA as a Generative Model

LDA models the **full joint distribution**:

$$p(x, y) = p(x \mid y)\, p(y)$$

It assumes:
- Features are **Gaussian within each class**: $p(x \mid y = k) = \mathcal{N}(\mu_k, \Sigma)$
- **Equal covariance** across all classes: $\Sigma_1 = \Sigma_2 = \cdots = \Sigma_K = \Sigma$

The resulting decision boundary is **linear** (hence *Linear* DA).

---

### Logistic Regression as a Discriminative Model

Logistic regression models only the **conditional distribution**:

$$p(y \mid x) = \frac{e^{\beta_0 + \beta^T x}}{1 + e^{\beta_0 + \beta^T x}}$$

It makes **no assumptions** about the distribution of $x$, making it more robust.

---

### When Fewer Assumptions is a Disadvantage

Logistic regression being more robust is **a weakness when LDA's assumptions actually hold**.

**Why:** Because LDA leverages the marginal distribution $p(x)$ to extract additional information. Logistic regression ignores this, leaving information unused.

This is measured by **statistical efficiency**. When the data truly follows $\mathcal{N}(\mu_k, \Sigma)$:

- LDA converges with **fewer samples**
- LDA has **lower variance** in its estimates
- LDA will **outperform** logistic regression, especially at small $n$

$$\text{Var}(\hat{\theta}_{\text{LDA}}) < \text{Var}(\hat{\theta}_{\text{LR}}) \quad \text{when assumptions hold}$$

**The tradeoff:**

$$\underbrace{\text{Logistic Regression}}_{\text{flexible, robust, needs more data}} \quad \text{vs} \quad \underbrace{\text{LDA}}_{\text{efficient, fast to converge, fragile to violations}}$$

> **Rule of thumb:** If you have a small dataset and have reason to believe features are approximately Gaussian with equal covariances, prefer LDA. If the distributional assumptions are violated or you have plenty of data, logistic regression is safer.

---

## Robustness to Outliers: The "Border Guard" Intuition

### Logistic Loss

For a correctly classified point, the logistic loss is:

$$\mathcal{L} = -\log\bigl(P(y \mid x)\bigr)$$

For a point far on the correct side of the boundary, say $P(y = \text{Legit} \mid x) = 0.9999$:

$$\mathcal{L} = -\log(0.9999) \approx 0.0001$$

If you move the outlier **even further** away, $P \to 0.99999$ and $\mathcal{L} \to 0.00001$. The loss is **saturated** — nearly flat. The gradient is essentially zero, so the point has almost no influence on the decision boundary.

### Why This Makes Logistic Regression Robust

The logistic model only "pays attention" to points **near the decision boundary** where $P \approx 0.5$. Outliers that are already confidently correct are ignored by the optimizer.

$$\frac{\partial \mathcal{L}}{\partial \beta} \approx 0 \quad \text{for points far from the boundary}$$

### Contrast with LDA

LDA estimates class means directly from the data:

$$\hat{\mu}_k = \frac{1}{n_k} \sum_{i \in C_k} x_i$$

An outlier on the correct side **directly shifts** $\hat{\mu}_k$, which moves the decision boundary. LDA has no mechanism to down-weight extreme points.

### Summary

| | Outlier far on correct side | Outlier near boundary |
|---|---|---|
| **Logistic Regression** | Near-zero gradient, ignored | Large gradient, influential |
| **LDA** | Pulls class mean, shifts boundary | Also influential |

---

## Basis Expansions

### Motivation

Linear models assume a **constant rate of change** — but real relationships are often non-linear (e.g. drug dose-response curves, human hearing sensitivity). The solution is to **transform** features into a space where linearity holds.

### The Idea

Replace (or augment) the data matrix $X$ with transformations $h(X)$:

$$y = \sum_{i=1}^{p} \beta_i x_i \quad \longrightarrow \quad y = \sum_{m=1}^{M} \beta'_m h_m(X)$$

The model remains **linear in the parameters** $\beta'_m$, even though it is non-linear in the original features $X$. This means the same linear solvers apply.

### Common Basis Functions

| Transformation | Formula | Use case |
|---|---|---|
| Polynomial | $h_m(X) = X_j^2$ or $X_j X_k$ | Quadratic/interaction effects |
| Log | $h_m(X) = \log(X_j)$ | Skewed, multiplicative relationships |
| Square root | $h_m(X) = \sqrt{X_j}$ | Count-like data |
| Standardization | $h_m(X) = \dfrac{X_j - \bar{X}_j}{s_{X_j}}$ | Always used with regularization |
| Rank | $h_m(X_{(i)}) = i$ | Robust to outliers, ordinal structure |

### Relation to Feature Engineering

Basis expansions are a **mathematically motivated subset** of feature engineering. The distinguishing property: the transformed features must keep the model **linear in** $\beta'$ so that standard linear fitting procedures apply unchanged.

$$\underbrace{\text{Basis Expansions}}_{\text{preserve linear-in-parameters structure}} \;\subset\; \underbrace{\text{Feature Engineering}}_{\text{any transformation of } X}$$

### Example: Capturing a Quadratic Relationship

Define $h_1(x) = x$ and $h_2(x) = x^2$. Then:

$$y = \beta_1 h_1(x) + \beta_2 h_2(x) = \beta_1 x + \beta_2 x^2$$

This is **non-linear in** $x$ but **linear in** $[\beta_1, \beta_2]$ — ordinary least squares or logistic regression solves it without modification.

---

## Appendix: Naïve Augmentation (Elastic Net via LASSO)

### The Goal

Elastic Net combines Ridge ($L_2$) and Lasso ($L_1$) penalties:

$$J(\beta) = \|y - X\beta\|_2^2 + \lambda_2\|\beta\|_2^2 + \lambda_1\|\beta\|_1$$

We have highly optimized solvers (LARS, coordinate descent) for pure LASSO. The idea: **trick a LASSO solver into solving Elastic Net** by manipulating the data.

### Step 1: Naïve Augmentation — Hide the Ridge in the Data

Append $p$ artificial rows to $X$ and zeros to $y$:

$$X_{aug} = \begin{pmatrix} X \\ \sqrt{\lambda_2}\, I_p \end{pmatrix}, \qquad y_{aug} = \begin{pmatrix} y \\ \mathbf{0}_p \end{pmatrix}$$

Now compute the RSS on the augmented data:

$$\|y_{aug} - X_{aug}\beta\|_2^2 = \|y - X\beta\|_2^2 + \underbrace{\lambda_2\|\beta\|_2^2}_{\text{Ridge penalty appears!}}$$

Feed $(X_{aug}, y_{aug})$ into a LASSO solver with penalty $\lambda_1$ — it thinks it's doing LASSO but is actually solving Elastic Net.

### Step 2: The Double Shrinkage Problem

The naïve solution is **biased**. The coefficients get shrunk **twice**:

1. The augmented rows apply $L_2$ (Ridge) shrinkage, pulling $\beta$ toward zero.
2. The LASSO solver then applies $L_1$ shrinkage on top of already-shrunk coefficients.

Result: coefficients are too small — effectively "taxed twice."

### Step 3: The Fix — Rescaled Augmentation

Introduce a scaling factor $c = (1 + \lambda_2)^{-1/2}$ and define:

$$X^* = \frac{1}{\sqrt{1+\lambda_2}} \begin{pmatrix} X \\ \sqrt{\lambda_2}\, I_p \end{pmatrix}, \qquad y^* = \begin{pmatrix} y \\ \mathbf{0}_p \end{pmatrix}$$

This scaling inflates the coefficients to undo the double shrinkage. The proof shows:

$$\frac{1}{\sqrt{1+\lambda_2}}\,\beta^* = (X^T X + \lambda_2 I_p)^{-1} X^T y = \hat{\beta}_{\text{ridge}}$$

$$\Rightarrow \quad \beta^* = \sqrt{1+\lambda_2}\;\hat{\beta}_{\text{ridge}}$$

The rescaled augmentation traces the **true Elastic Net path** — LARS can solve it exactly.

### Summary

| Version | Problem | Result |
|---|---|---|
| Naïve Augmentation | Double shrinkage | Biased, coefficients too small |
| Rescaled Augmentation | Fixed with factor $c = (1+\lambda_2)^{-1/2}$ | Exact Elastic Net solution |

---

## Splines

### The Problem with Naive Piecewise Polynomials

Split the domain at a knot $\xi$ and fit separate polynomials on each side with no constraints:

$$\text{Left: } f_1(x) = a_1 + b_1 x + c_1 x^2 + d_1 x^3 \qquad \text{Right: } f_2(x) = a_2 + b_2 x + c_2 x^2 + d_2 x^3$$

Without constraints at $\xi$, the pieces can **jump**, have **sharp corners**, or **bend abruptly** — the "step function problem". The curve is discontinuous and overfits noise at the boundary.

### The Three Smoothness Constraints

Impose constraints at every knot $\xi_k$ to enforce smoothness:

| Constraint | Requirement | Math |
|---|---|---|
| 1. Continuity | Pieces must touch — no jump | $f_1(\xi) = f_2(\xi)$ |
| 2. Smooth join | Slopes must match — no corner | $f_1'(\xi) = f_2'(\xi)$ |
| 3. Smooth bend | Curvature must match — no kink | $f_1''(\xi) = f_2''(\xi)$ |

### Why Cubic? Degrees of Freedom

A cubic polynomial has 4 parameters. With $K$ knots there are $K+1$ regions:

$$\text{Total parameters} = 4(K+1)$$

Each knot imposes 3 constraints:

$$\text{Total constraints} = 3K$$

$$\boxed{\text{Degrees of freedom} = 4(K+1) - 3K = K + 4}$$

Cubic is the **minimum degree** where all three constraints can be simultaneously enforced with a still-flexible curve.

### Cubic Spline as a Basis Expansion

A cubic spline with knots $\xi_1, \dots, \xi_K$ is written using the **truncated power basis**:

$$f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \sum_{k=1}^{K} \beta_{k+3}(x - \xi_k)_+^3$$

where the truncated power function is:

$$(x - \xi_k)_+^3 = \begin{cases} (x - \xi_k)^3 & \text{if } x > \xi_k \\ 0 & \text{otherwise} \end{cases}$$

Each term $(x-\xi_k)_+^3$ "switches on" after knot $k$, allowing the curve to change behaviour at that point. The model remains **linear in** $\beta$ — the same linear solvers apply.
