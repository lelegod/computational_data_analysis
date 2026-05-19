# Week 4 — Group Discussion Questions

Topics: LDA, QDA, RDA, Logistic Regression, generative vs discriminative classifiers

---

## Q1: Derive Two-Class Logistic Regression Expressions

**Question (Think-Pair-Share, slide 22):** For the two-class logistic regression problem, derive the log-odds expression and the probability expression for class membership. Show how the linear decision boundary arises.

**Answer:**

Logistic regression models the posterior probability of class $y = 1$ given input $x$ directly (discriminative model). For two classes $\{0, 1\}$:

**Step 1 — Model the log-odds as linear:**

$$\log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = \beta_0 + \beta^T x$$

This is the **log-odds** (logit) of class 1 versus class 0. The choice of a linear function is the key modelling assumption.

**Step 2 — Solve for $P(y=1 \mid x)$:**

Let $\eta = \beta_0 + \beta^T x$. Then:

$$\frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)} = e^{\eta}$$

$$P(y=1 \mid x)(1 + e^{\eta}) = e^{\eta}$$

$$\boxed{P(y=1 \mid x) = \frac{e^{\eta}}{1 + e^{\eta}} = \frac{1}{1 + e^{-\eta}} = \sigma(\eta)}$$

where $\sigma(\cdot)$ is the **sigmoid (logistic) function**.

**Step 3 — Decision boundary:**

We classify $x$ as class 1 if $P(y=1 \mid x) > 0.5$, which happens when $\eta > 0$, i.e.:

$$\beta_0 + \beta^T x > 0$$

This is a **linear decision boundary** in feature space — a hyperplane.

**Why logistic regression is discriminative:** It directly models $P(y \mid x)$ without modelling the class-conditional densities $P(x \mid y)$ or the class priors $P(y)$ separately (contrast with LDA).

**Estimation:** Parameters $\beta_0, \beta$ are estimated by maximising the log-likelihood:

$$\ell(\beta) = \sum_{i=1}^{N} \left[ y_i \log P(y=1 \mid x_i) + (1-y_i) \log P(y=0 \mid x_i) \right]$$

No closed-form solution — solved iteratively (IRLS / Newton-Raphson).

---

## Q2: Does Doubling the Odds Double the Probability?

**Question (Think-Pair-Share, slide 25):** If the odds of an event double, does the probability of the event also double?

**Answer:**

**No.** Odds and probability are related nonlinearly, so doubling the odds does not double the probability (except as a rough approximation for very small probabilities).

**Formal argument:**

Let $p = P(\text{event})$. The odds are $\text{odds} = p / (1-p)$.

If odds double: $\text{odds}' = 2p/(1-p)$.

Solving for the new probability $p'$:

$$p' = \frac{\text{odds}'}{1 + \text{odds}'} = \frac{2p/(1-p)}{1 + 2p/(1-p)} = \frac{2p}{1-p+2p} = \frac{2p}{1+p}$$

**Comparison:**

| $p$ | $p' = 2p/(1+p)$ | $2p$ (would-be double) |
|-----|-----------------|------------------------|
| 0.1 | 0.182 | 0.200 |
| 0.3 | 0.462 | 0.600 |
| 0.5 | 0.667 | 1.000 (impossible) |
| 0.8 | 0.889 | 1.600 (impossible) |

**Key insight:** For $p \ll 1$ (rare events), $p' \approx 2p$ — doubling odds approximately doubles probability. But for larger $p$, the effect saturates because probabilities are bounded in $[0,1]$.

**Why this matters for logistic regression:** The model is linear on the **log-odds** scale, not the probability scale. A one-unit increase in $x_j$ multiplies the odds by $e^{\beta_j}$, but the effect on probability is nonlinear and depends on the current probability level.

---

## Q3: Outlier Effect on LDA vs Logistic Regression

**Question (Think-Pair-Share, slide 33):** Consider a two-class classification problem. An outlier observation is added far from the decision boundary. How does this affect (a) LDA's decision boundary and (b) Logistic Regression's decision boundary? Which is more robust?

**Answer:**

**LDA (Linear Discriminant Analysis) — generative model:**

LDA estimates the class means $\mu_k$ and the shared covariance $\Sigma$ from the data. The decision boundary is:

$$x^T \Sigma^{-1}(\mu_1 - \mu_2) = \frac{1}{2}(\mu_1^T \Sigma^{-1}\mu_1 - \mu_2^T \Sigma^{-1}\mu_2) - \log\frac{\pi_1}{\pi_2}$$

An outlier in class $k$ will:
1. Shift $\hat{\mu}_k$ toward the outlier
2. Inflate $\hat{\Sigma}$ (the pooled covariance)
3. Change class prior $\hat{\pi}_k$ if it adds an observation

All three effects directly move the decision boundary. LDA is **sensitive to outliers** because it fits moments to the full distribution of each class.

**Logistic Regression — discriminative model:**

LR only models the decision boundary directly via the log-likelihood. Observations far from the boundary (with a large correct margin) contribute almost **zero gradient** to the log-likelihood because $\sigma(\eta) \approx 1$ for large $\eta$ — the loss is essentially saturated.

An outlier far from the boundary has negligible influence on the estimated $\beta$. LR is **more robust to outliers** than LDA.

**Intuition:** LDA uses all data to estimate class distributions and then derives the boundary — outliers corrupt the distribution estimates. LR focuses on observations near the boundary — distant outliers are "already classified confidently" and ignored.

**Caveat:** LR can be sensitive to outliers on the *wrong side* of the boundary (badly mislabelled points or near-boundary outliers), since those produce large gradients.

**Summary table:**

| Property | LDA | Logistic Regression |
|----------|-----|---------------------|
| Model type | Generative | Discriminative |
| Assumption | Gaussian classes, equal $\Sigma$ | Linear log-odds |
| Outlier sensitivity | High (moments are global) | Low (far-margin points saturated) |
| Works well when | Gaussian assumption holds | Boundary is linear, classes non-Gaussian |

---

## Q4: The Knot Dilemma — Cubic Splines

**Question (Think-Pair-Share, slide ~52):** In fitting a smoothing spline or regression spline, how do we decide where to place the knots? What is the trade-off between too few and too many knots?

**Answer:**

A **regression spline** with $K$ knots fits a piecewise polynomial (typically cubic) that is continuous and smooth at each knot $\xi_k$. The model is a linear expansion of basis functions:

$$f(x) = \sum_{j} \beta_j h_j(x)$$

where the $h_j$ are the spline basis functions (e.g., truncated power basis or B-splines).

**The dilemma:**

- **Too few knots / knots in wrong positions:** The model underfits — it cannot capture local variation where the true function changes rapidly.
- **Too many knots:** The model overfits — it becomes overly flexible and noisy. Additionally, placing knots where there is no data is wasteful (no data to inform the fit there).

**Strategies for knot placement:**

1. **Fixed grid (uniform):** Place knots evenly across the data range. Simple but ignores data density.

2. **At data quantiles:** Place knots so that equal numbers of observations fall in each interval. Useful because regions with more data can support more complex fits.

3. **Smoothing splines (automatic):** Place a knot at every data point and penalise the second derivative:

$$\min_f \sum_{i=1}^N (y_i - f(x_i))^2 + \lambda \int [f''(x)]^2 \, dx$$

The penalty $\lambda$ controls smoothness — effectively the degrees of freedom. The optimal $\lambda$ is chosen by cross-validation. This sidesteps the knot placement problem entirely.

4. **Forward selection / MARS:** Adaptively add knots where they most reduce residuals.

**Key insight for exam:** In smoothing splines, the roughness penalty $\lambda$ plays the role of the knot selection — larger $\lambda$ shrinks toward a linear fit; $\lambda = 0$ interpolates every point. The bias-variance trade-off is controlled through $\lambda$ or equivalently the effective degrees of freedom $\text{df}_\lambda$.
