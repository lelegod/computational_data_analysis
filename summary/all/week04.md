# Week 4 — Linear and Regularized Classification: LDA, QDA, RDA, Logistic Regression

## Overview
Week 4 introduces supervised classification methods. The lecture begins with a brief recap of Week 3 (Curse of Dimensionality, Ridge, Lasso, Multiple Testing) and then covers two paradigms for classification: *generative* methods (LDA, QDA, RDA) that model the class-conditional data distribution and use Bayes' theorem, and *discriminative* methods (Logistic Regression) that model the posterior class probability directly. Both produce linear decision boundaries (LDA and Logistic Regression) or quadratic boundaries (QDA). Regularization (RDA) is introduced to handle high-dimensional settings where $p \gg N$.

---

## 1. Two Paradigms for Classification

### Generative Classifiers (LDA, QDA)
- Model the class-conditional density $f_k(x) = P(X=x|G=k)$ and prior $\pi_k = P(G=k)$.
- Apply Bayes' theorem to get the posterior $P(G=k|X=x)$.
- Analogy: a "forensic scientist" who models how each class generates data.

### Discriminative Classifiers (Logistic Regression)
- Model the posterior $P(G=k|X=x)$ directly without modelling $P(X|G)$.
- Analogy: a "border guard" who only cares what separates classes, not how they were generated.
- Fewer assumptions → more robust when class distributions are non-Gaussian.

---

## 2. Bayes' Theorem for Classification

For $K$ classes:
$$P(G=k|X=x) = \frac{f_k(x)\pi_k}{\sum_{l=1}^K f_l(x)\pi_l}$$

- $f_k(x) = P(X=x|G=k)$: class-conditional density.
- $\pi_k = P(G=k)$: prior probability.
- Classification rule: assign $x$ to $\hat{k} = \arg\max_k P(G=k|X=x)$.
- Equivalently: $\hat{k} = \arg\max_k [f_k(x)\pi_k]$ (denominator is same for all $k$).

---

## 3. The Multivariate Gaussian Assumption

The most common choice for $f_k(x)$ is the multivariate Gaussian:
$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}}\exp\!\left(-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)\right)$$

- $p$: number of features.
- $\mu_k \in \mathbb{R}^p$: mean vector for class $k$.
- $\Sigma_k \in \mathbb{R}^{p\times p}$: covariance matrix for class $k$.

The log-ratio (log-odds) of two classes is used to derive the decision boundary:
$$\log\frac{P(G=k|X=x)}{P(G=l|X=x)} = \log\frac{f_k(x)}{f_l(x)} + \log\frac{\pi_k}{\pi_l}$$

---

## 4. LDA — Linear Discriminant Analysis

### Key Assumption
All classes share the **same covariance matrix**: $\Sigma_k = \Sigma$ for all $k$.

### Why the Boundary is Linear
Plugging Gaussian densities with shared $\Sigma$ into the log-odds:
$$\log\frac{P(G=k|X=x)}{P(G=l|X=x)} = \log\frac{\pi_k}{\pi_l} - \frac{1}{2}(\mu_k+\mu_l)^T\Sigma^{-1}(\mu_k-\mu_l) + x^T\Sigma^{-1}(\mu_k-\mu_l)$$

The term $x^T\Sigma^{-1}x$ (quadratic in $x$) **cancels** from both classes because $\Sigma$ is shared. The remaining expression is **linear in $x$** → linear decision boundary.

### Linear Discriminant Function
$$\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k$$

- **Classification rule**: $\hat{G}(x) = \arg\max_k \delta_k(x)$
- Decision boundary between class $k$ and $l$: the hyperplane $\{x : \delta_k(x) = \delta_l(x)\}$

### Parameter Estimation
- **Class prior**: $\hat{\pi}_k = N_k/N$
- **Class mean**: $\hat{\mu}_k = \frac{1}{N_k}\sum_{i:\,g_i=k} x_i$
- **Pooled covariance** (shared $\Sigma$):
$$\hat{\Sigma} = \frac{1}{N-K}\sum_{k=1}^K\sum_{i:\,g_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T$$

---

## 5. QDA — Quadratic Discriminant Analysis

### Key Change
- Drops the equal covariance assumption: each class has its own $\Sigma_k$.
- The quadratic term $x^T\Sigma_k^{-1}x$ no longer cancels → decision boundary is **quadratic** in $x$.
- Decision boundaries are conic sections: ellipses, parabolas, hyperbolas.

### Discriminant Function (QDA)
$$\delta_k^Q(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) + \log\pi_k$$

### Drawback
- Requires $O(p^2)$ parameters **per class** (full $p\times p$ covariance matrix per class).
- For $p=100$ and $K=2$: need $\sim 10{,}000$ parameters per class.
- When $p \gg N$: covariance matrix is singular → cannot invert → QDA breaks down.

### LDA vs QDA

| Property | LDA | QDA |
|----------|-----|-----|
| Covariance | Shared $\Sigma$ | Per-class $\Sigma_k$ |
| Decision boundary | Linear | Quadratic (conic sections) |
| Parameters | $Kp + p(p+1)/2$ | $Kp + Kp(p+1)/2$ |
| Bias | Higher | Lower |
| Variance | Lower | Higher |
| Works when $p \gg N$? | With RDA | No (singular $\hat{\Sigma}_k$) |

---

## 6. The High-Dimensional Challenge

- Both LDA and QDA require inverting $\hat{\Sigma}$ (or $\hat{\Sigma}_k$).
- If $p \gg N$: the covariance matrix is $p\times p$ but estimated from $N$ observations → rank deficient (singular).
- Singular matrix cannot be inverted: LDA/QDA fail with a numerical error.
- **Solution**: regularize the covariance estimate (RDA).

---

## 7. RDA — Regularized Discriminant Analysis (Friedman 1989)

RDA creates a continuum between QDA (fully flexible), LDA (pooled), and diagonal covariance.

### Option 1: Shrink QDA towards LDA ($\alpha \in [0,1]$)
$$\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}$$
- $\alpha=1$: QDA (per-class covariance).
- $\alpha=0$: LDA (pooled covariance).

### Option 2: Shrink towards diagonal ($\gamma \in [0,1]$)
$$\hat{\Sigma}(\gamma) = \gamma\hat{\Sigma} + (1-\gamma)\,\text{diag}(\hat{\Sigma})$$
- $\gamma=1$: LDA covariance.
- $\gamma=0$: diagonal covariance (assumes independent features).

### Option 3: Shrink towards spherical ($\gamma \in [0,1]$)
$$\hat{\Sigma}(\gamma) = \gamma\hat{\Sigma} + (1-\gamma)\hat{\sigma}^2 I$$
- $\gamma=1$: LDA covariance.
- $\gamma=0$: isotropic covariance $\hat{\sigma}^2 I$ (all features equally varied).

### Tuning
- $\alpha$ and $\gamma$ are hyperparameters tuned by cross-validation.
- The two parameters together span a surface between QDA, LDA, diagonal, and spherical.

---

## 8. RRDA — Reduced Rank Discriminant Analysis

- Projects data into a $K-1$ dimensional subspace that **maximises class separation**.
- In the original $p$-dimensional space, boundaries are hyperplanes — hard to visualise.
- Projecting to $K-1$ dimensions allows plotting all class boundaries in a 2D canonical coordinate plot (for $K=3$).
- Very useful for visualisation and interpretability of classification structure.
- Example: $K$ tumour types in $p=10{,}000$ gene space → project to $K-1$ canonical coordinates to see clusters.

---

## 9. Logistic Regression

### The Discriminative Approach
- Unlike LDA/QDA which model $P(X|G=k)$ (generative), logistic regression models $P(G=k|X)$ directly.
- No modelling of how $X$ is generated → fewer assumptions, more robust to non-Gaussian data.

### Binary Classification Model ($Y \in \{0,1\}$)
$$P(Y=1|X=x) = \frac{e^{\beta_0+\beta^Tx}}{1+e^{\beta_0+\beta^Tx}}, \qquad P(Y=0|X=x) = \frac{1}{1+e^{\beta_0+\beta^Tx}}$$

- The sigmoid function maps $(-\infty,\infty)$ to $(0,1)$.
- Probabilities sum to 1 for all $x$.

### Log-Odds (Logit) Transformation
$$\text{logit}(p) = \log\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = \beta_0 + \beta^Tx$$

- The log-odds is **linear** in $x$.
- **Decision boundary**: $\beta_0 + \beta^Tx = 0$ — same linear form as LDA.
- **Coefficient interpretation**: $\beta_j$ = change in log-odds per unit increase in $x_j$.
- $e^{\beta_j}$ = multiplicative change in **odds** (NOT probability).

### Likelihood and MLE
Joint likelihood (assuming independence):
$$L(\beta_0, \beta) = \prod_{i=1}^n P(G=g_{x_i}|X=x_i)$$

Log-likelihood:
$$\ell(\beta) = \sum_{i=1}^N \left[y_i(\beta_0 + \beta^Tx_i) - \log(1+e^{\beta_0+\beta^Tx_i})\right]$$

- No closed-form solution for $\hat{\beta}$.
- Solved iteratively via **Newton-Raphson** (equivalently: IRLS — Iteratively Reweighted Least Squares).
- Newton update: $\beta^{new} = \beta^{old} - \left(\frac{\partial^2\ell}{\partial\beta\partial\beta^T}\right)^{-1}\frac{\partial\ell}{\partial\beta}$

### Regularized Logistic Regression
- Add $\ell_1$ or $\ell_2$ penalty to log-likelihood to prevent overfitting when $p$ is large:
$$\ell_\text{reg}(\beta) = \ell(\beta) - \lambda\|\beta\|_2^2 \quad (\text{Ridge}) \quad \text{or} \quad \ell(\beta) - \lambda\|\beta\|_1 \quad (\text{Lasso})$$

---

## 10. LDA vs Logistic Regression: Full Comparison

| Property | LDA | Logistic Regression |
|----------|-----|---------------------|
| Paradigm | Generative | Discriminative |
| Models | $P(X|G)$ and $\pi_k$ → infer $P(G|X)$ | $P(G|X)$ directly |
| Assumes Gaussian $X$ | Yes | No |
| Assumes equal $\Sigma$ | Yes | No |
| Decision boundary | Linear (from Gaussian cancellation) | Linear (by construction) |
| Closed-form fit | Yes | No (Newton-Raphson) |
| More efficient when? | Data truly Gaussian | Always (fewer assumptions wrong) |
| More robust when? | Gaussian assumptions hold | Non-Gaussian / messy data |
| Coefficient meaning | Class separation via $\Sigma^{-1}(\mu_k-\mu_l)$ | Change in log-odds per unit $x_j$ |

### When do they give the same boundary?
- LDA and logistic regression both produce linear boundaries, but estimated differently.
- They converge to the same boundary as $N \to \infty$ when data is truly Gaussian.
- In practice, logistic regression is usually preferred for robustness.

---

## 11. Basis Expansions and Splines (Appendix / Brief Coverage)

- Instead of fitting linear models in $x$, apply a transformation $h(x)$ first:
  $$f(x) = \sum_m \beta_m h_m(x)$$
- Common choices: polynomial basis, step functions, piecewise polynomials, splines.
- **Cubic splines**: piecewise cubic polynomials with continuous first and second derivatives at knots.
- Allows fitting non-linear class boundaries while keeping the linear algebra framework.
- Degrees of freedom: number of basis functions used.
- Splines with regularization = smoothing splines — tune via cross-validation.

---

## 12. Summary

| Method | Boundary | Covariance | Parameters | Use case |
|--------|---------|-----------|-----------|---------|
| LDA | Linear | Shared $\Sigma$ | $Kp + p(p+1)/2$ | Gaussian, $p < N$ |
| QDA | Quadratic | Per-class $\Sigma_k$ | $Kp + Kp(p+1)/2$ | Gaussian, flexible, $p \ll N$ |
| RDA | Continuous | Regularized | Tuned via CV | High-dim, unknown structure |
| RRDA | Linear in $K-1$ space | Shared, reduced | $K-1$ dimensions | Visualisation |
| Logistic Regression | Linear | None assumed | $p+1$ | Robust, non-Gaussian |
