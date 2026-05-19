# Week 4 — Linear and Regularized Classification: LDA, QDA, RDA, Logistic Regression (Exam Focus)

## Must-Know Facts

### Generative vs Discriminative Classification
- **Generative** (LDA, QDA): model the class-conditional density $P(X|G=k)$ and the prior $\pi_k$; use Bayes' theorem to get $P(G=k|X)$.
- **Discriminative** (Logistic Regression): model the posterior $P(G=k|X)$ directly without modelling how $X$ was generated.
- LDA makes stronger distributional assumptions (Gaussian, equal covariance) but is more efficient when those assumptions hold.
- Logistic Regression is more robust when the Gaussian assumption is violated.

### Bayes' Theorem for Classification
$$P(G=k|X=x) = \frac{f_k(x)\pi_k}{\sum_{l=1}^K f_l(x)\pi_l}$$
- $f_k(x) = P(X=x|G=k)$: class-conditional density.
- $\pi_k = P(G=k)$: prior probability (estimated by class frequency $N_k/N$).
- Classify to class $k$ with highest posterior.

### Multivariate Gaussian Assumption
$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}}\exp\!\left(-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)\right)$$

### LDA (Linear Discriminant Analysis)
- **Key assumption**: ALL classes share the same covariance matrix $\Sigma_k = \Sigma$.
- With equal $\Sigma$, the quadratic term $x^T\Sigma^{-1}x$ cancels in the log-odds → decision boundary is **linear** in $x$.
- **Log-odds** (class $k$ vs $l$):
$$\log\frac{P(G=k|X=x)}{P(G=l|X=x)} = \log\frac{\pi_k}{\pi_l} - \frac{1}{2}(\mu_k+\mu_l)^T\Sigma^{-1}(\mu_k-\mu_l) + x^T\Sigma^{-1}(\mu_k-\mu_l)$$
- **Linear Discriminant Function**:
$$\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k$$
- **Classify**: $\hat{G}(x) = \arg\max_k \delta_k(x)$

### Parameter Estimation in LDA
- $\hat{\pi}_k = N_k/N$
- $\hat{\mu}_k = \frac{1}{N_k}\sum_{g_i=k} x_i$
- **Pooled covariance**: $\hat{\Sigma} = \frac{1}{N-K}\sum_{k=1}^K\sum_{g_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T$

### QDA (Quadratic Discriminant Analysis)
- Drops the equal covariance assumption: each class has its own $\Sigma_k$.
- Log-odds are **quadratic** in $x$ → curved decision boundaries (ellipses, parabolas, hyperbolas).
- Drawback: requires $O(p^2)$ parameters per class — breaks down when $p \gg N$.
- More flexible than LDA but much higher variance in high dimensions.

### RDA — Regularized Discriminant Analysis (Friedman 1989)
- Bridges QDA and LDA via regularization to handle high dimensions.
- **Option 1 — Shrink QDA towards LDA** ($\alpha \in [0,1]$):
  $$\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}$$
  - $\alpha=1$ → QDA; $\alpha=0$ → LDA
- **Option 2 — Shrink towards diagonal** ($\gamma \in [0,1]$):
  $$\hat{\Sigma}(\gamma) = \gamma\hat{\Sigma} + (1-\gamma)\,\text{diag}(\hat{\Sigma})$$
- **Option 3 — Shrink towards spherical**:
  $$\hat{\Sigma}(\gamma) = \gamma\hat{\Sigma} + (1-\gamma)\hat{\sigma}^2 I$$
- Tune $\alpha$ and $\gamma$ by cross-validation.

### RRDA — Reduced Rank Discriminant Analysis
- Projects data into a $K-1$ dimensional subspace that maximises class separation.
- Excellent for visualization of multi-class separation.
- Example: 3-class problem projected to a 2D canonical coordinate plot.

### Why $p \gg N$ breaks LDA/QDA
- LDA/QDA need $\hat{\Sigma}^{-1}$ — if $p \gg N$, $\hat{\Sigma}$ is singular (not full rank) and cannot be inverted.
- Solution: regularize (RDA/RRDA) or use diagonal/spherical covariance approximation.

### Logistic Regression
- **Discriminative**: models $P(G=k|X)$ directly.
- For binary classification ($Y \in \{0,1\}$):
$$P(Y=1|X=x) = \frac{e^{\beta_0+\beta^Tx}}{1+e^{\beta_0+\beta^Tx}}, \qquad P(Y=0|X=x) = \frac{1}{1+e^{\beta_0+\beta^Tx}}$$
- **Log-odds (logit)**:
$$\log\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = \beta_0 + \beta^Tx$$
- Decision boundary: $\beta_0 + \beta^Tx = 0$ → **linear** in $x$ (same as LDA boundary).
- $\beta_j$ = change in log-odds for a one-unit increase in $x_j$.
- $e^{\beta_j}$ = multiplicative change in the **odds**.

### Fitting Logistic Regression: MLE
- Log-likelihood:
$$\ell(\beta) = \sum_{i=1}^N \left[y_i(\beta^Tx_i) - \log(1+e^{\beta^Tx_i})\right]$$
- **No closed-form solution** — maximised iteratively via **Newton-Raphson** (or IRLS: Iteratively Reweighted Least Squares).

### LDA vs Logistic Regression

| Property | LDA | Logistic Regression |
|----------|-----|---------------------|
| Approach | Generative | Discriminative |
| Assumes class distribution | Yes (Gaussian) | No |
| Equal covariance assumed | Yes | No |
| Decision boundary | Linear | Linear |
| Closed-form solution | Yes | No (Newton-Raphson) |
| Works well when | Gaussian classes | Messy/non-Gaussian data |
| Parameters estimated by | MLE (closed form) | MLE (iterative) |

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $P(G=k|X=x) = \frac{f_k(x)\pi_k}{\sum_l f_l(x)\pi_l}$ | Bayes' theorem | Generative classification |
| $\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k$ | LDA discriminant function | Classify with LDA |
| $\hat{G}(x) = \arg\max_k \delta_k(x)$ | LDA decision rule | Assign class |
| $\hat{\Sigma} = \frac{1}{N-K}\sum_k\sum_{g_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T$ | Pooled covariance | LDA estimation |
| $\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}$ | RDA option 1 | Shrink QDA to LDA |
| $P(Y=1|X=x) = \frac{e^{\beta_0+\beta^Tx}}{1+e^{\beta_0+\beta^Tx}}$ | Logistic sigmoid | Logistic regression prediction |
| $\log\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = \beta_0 + \beta^Tx$ | Log-odds (logit) | Logistic regression interpretation |
| $\ell(\beta) = \sum_i[y_i\beta^Tx_i - \log(1+e^{\beta^Tx_i})]$ | Log-likelihood | MLE for logistic regression |

---

## Common Traps (wrong answers in exams)

- ❌ LDA has per-class covariance matrices → ✓ LDA assumes ALL classes share ONE pooled $\Sigma$; per-class $\Sigma_k$ is QDA
- ❌ LDA has a quadratic decision boundary → ✓ Equal $\Sigma$ cancels the quadratic term → LDA boundary is STRICTLY LINEAR
- ❌ Logistic regression models $P(X|G)$ → ✓ Logistic regression is DISCRIMINATIVE: it models $P(G|X)$ directly
- ❌ Logistic regression has a closed-form solution → ✓ No closed form; requires iterative Newton-Raphson / IRLS
- ❌ QDA and LDA have the same number of parameters → ✓ QDA needs $O(p^2)$ per class; LDA pools into one $\Sigma$ — far fewer params
- ❌ If $p \gg N$, LDA still works → ✓ $p \gg N$ makes $\hat{\Sigma}$ singular → LDA/QDA cannot invert it; use RDA
- ❌ $e^{\beta_j}$ is the change in probability → ✓ $e^{\beta_j}$ is the change in ODDS (multiplicative); probability change is non-linear
- ❌ LDA and logistic regression have different decision boundaries → ✓ Both produce LINEAR decision boundaries in $x$; the difference is HOW they are estimated
- ❌ RDA with $\alpha=1$ gives LDA → ✓ $\alpha=1$ gives QDA (full per-class $\Sigma_k$); $\alpha=0$ gives LDA (pooled $\Sigma$)
- ❌ RRDA reduces to fewer features like PCA → ✓ RRDA reduces to $K-1$ dimensions maximising CLASS SEPARATION (not variance like PCA)

---

## Quick Decision Rules

- "Linear boundary, models class distributions as Gaussian" → LDA
- "Quadratic boundary, per-class covariance" → QDA
- "LDA/QDA breaks when $p \gg N$" → use RDA (regularize with $\alpha$/$\gamma$)
- "Discriminative, models $P(G|X)$ directly, no distributional assumption" → Logistic Regression
- "Coefficient $\beta_j$ in logistic regression: what does it mean?" → change in log-odds per unit $x_j$; $e^{\beta_j}$ = odds multiplier
- "Which method is more robust to non-Gaussian data?" → Logistic Regression (fewer assumptions)
- "Which method is more efficient when data IS Gaussian?" → LDA (uses more information)
- "Decision boundary location" → where $\delta_k(x) = \delta_l(x)$ for LDA; where $\beta_0 + \beta^Tx = 0$ for logistic
- "Fit logistic regression" → maximise log-likelihood via Newton-Raphson (iterative)
- "Visualise multi-class separation in 2D" → RRDA (project to K-1 canonical coordinates)
