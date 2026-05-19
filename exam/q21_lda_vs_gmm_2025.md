# Q21 — LDA vs GMM (2025 Exam)

> **10 points** | Source: May 2025 exam (exam_vFinal.pdf), Q21
> **Question:** Discuss LDA and GMM in terms of:
> a) Explain and contrast the probabilistic assumptions between them
> b) Describe and contrast how model fitting is performed
> c) Highlight key differences in their goals, supervision, and use of labels
> d) Discuss how each model handles class overlap and latent structure

---

## a) Probabilistic Assumptions (~2.5 pts)

### LDA

Each class follows a Gaussian distribution with its **own mean** but **shared covariance** across all classes:

$$p(x \mid C_k) = \mathcal{N}(x \mid \mu_k, \Sigma)$$

Classification uses Bayes' theorem:

$$p(C_k \mid x) = \frac{p(x \mid C_k)\,\pi_k}{\sum_j p(x \mid C_j)\,\pi_j}$$

**Why does the shared $\Sigma$ produce a linear boundary?**

The log-posterior ratio for two classes is:

$$\log\frac{p(C_1 \mid x)}{p(C_2 \mid x)} = \log\frac{\pi_1}{\pi_2} + \log\frac{\mathcal{N}(x \mid \mu_1, \Sigma)}{\mathcal{N}(x \mid \mu_2, \Sigma)}$$

Expanding the Gaussians, the $x^T \Sigma^{-1} x$ quadratic term appears in both — and cancels because $\Sigma$ is shared. What remains is **linear in $x$**:

$$= \log\frac{\pi_1}{\pi_2} + (\mu_1 - \mu_2)^T \Sigma^{-1} x - \frac{1}{2}(\mu_1^T\Sigma^{-1}\mu_1 - \mu_2^T\Sigma^{-1}\mu_2)$$

If covariances differ (QDA), the quadratic terms do NOT cancel → quadratic boundary.

### GMM

Models the **marginal density** of $x$ as a mixture:

$$p(x) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Each component has its **own** $\mu_k$ and $\Sigma_k$ — covariances are not shared. This is strictly more flexible than LDA.

### Key contrast

| | LDA | GMM |
|--|-----|-----|
| Models | $p(x \mid C_k)$ — class-conditional | $p(x)$ — marginal density |
| Covariance | Shared $\Sigma$ across classes | Per-component $\Sigma_k$ |
| Boundary shape | Linear | Arbitrary (depends on components) |
| Assumes labels | Yes | No |

---

## b) Model Fitting (~2.5 pts)

### LDA — Supervised, Closed-Form MLE

Parameters are estimated directly from labeled data — no iteration:

| Parameter | Estimate |
|-----------|----------|
| Class means | $\hat{\mu}_k = \frac{1}{n_k}\sum_{i:\, y_i=k} x_i$ |
| Pooled covariance | $\hat{\Sigma} = \frac{1}{N-K}\sum_k \sum_{i:\, y_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$ |
| Class priors | $\hat{\pi}_k = n_k / N$ |

The pooled covariance uses $N - K$ (not $N$) to correct for estimating $K$ means.

### GMM — Unsupervised, EM Algorithm

No labels available — must discover structure. Uses the **Expectation-Maximisation (EM)** algorithm:

**E-step:** Compute soft responsibilities — how much does component $k$ "own" point $x_n$?

$$r_{nk} = \frac{\pi_k\,\mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K}\pi_j\,\mathcal{N}(x_n \mid \mu_j, \Sigma_j)}$$

**M-step:** Update all parameters using $r_{nk}$ as soft weights:

$$\hat{\mu}_k = \frac{\sum_n r_{nk}\,x_n}{\sum_n r_{nk}}, \qquad \hat{\Sigma}_k = \frac{\sum_n r_{nk}(x_n - \hat{\mu}_k)(x_n - \hat{\mu}_k)^T}{\sum_n r_{nk}}, \qquad \hat{\pi}_k = \frac{\sum_n r_{nk}}{N}$$

Repeat E → M → E → M until log-likelihood converges. EM is guaranteed to increase log-likelihood each iteration but may converge to a **local maximum** — multiple restarts are often used.

**Choosing K:** Use BIC to select the number of components — $\text{BIC} = -2\log L + p\log N$ penalises model complexity.

### Key contrast

| | LDA | GMM |
|--|-----|-----|
| Algorithm | Closed-form MLE | Iterative EM |
| Needs labels | Yes | No |
| Convergence | Exact, one pass | Local optimum, multiple passes |
| K selection | Fixed by number of classes | Must choose K (e.g. via BIC) |

---

## c) Goals, Supervision, and Labels (~2.5 pts)

| | LDA | GMM |
|--|-----|-----|
| **Type** | Supervised | Unsupervised |
| **Labels required?** | Yes — cannot run without them | No |
| **Primary goal** | Classify new points into known classes | Model density, discover latent groups |
| **Output** | Class label $\hat{C}$ or posterior $p(C_k \mid x)$ | Soft membership $r_{nk}$ for each component |
| **Secondary use** | Dimensionality reduction via Fisher's discriminant | Density estimation, anomaly detection, model selection via BIC |
| **Relation to K-means** | — | GMM is the **soft version** of K-means (K-means = GMM with equal spherical $\Sigma_k$ and hard assignments) |

LDA **cannot** run without labels — the class-conditional means and pooled covariance require knowing which class each point belongs to. GMM discovers structure entirely from unlabeled data; class labels can be assigned post-hoc by the highest-responsibility component.

---

## d) Class Overlap and Latent Structure (~2.5 pts)

### LDA

- Assigns each point to the class with highest posterior: $\hat{C} = \arg\max_k\, p(C_k \mid x)$
- **Hard linear boundary** — points near the boundary are assigned a class but the decision is uncertain
- Posterior probabilities $p(C_k \mid x)$ encode uncertainty — values near 0.5 indicate overlap
- Does **not** model latent structure — classes are pre-defined externally by labels
- Sensitive to outliers since $\hat{\mu}_k$ and $\hat{\Sigma}$ are estimated directly from data

### GMM

- Handles overlap naturally via **soft assignments** — $r_{nk}$ captures partial membership in multiple components simultaneously
- Can reveal **latent structure**: discovers hidden subgroups with no supervision required
- Per-component $\Sigma_k$ allows clusters of different shapes, sizes, and orientations (elliptical clusters)
- Richer model but **no guarantee** that discovered clusters align with meaningful ground-truth classes
- Can model multimodal distributions (multiple humps) that a single Gaussian cannot

### Summary

| | LDA | GMM |
|--|-----|-----|
| Overlap handling | Posterior probabilities, hard boundary | Soft assignments $r_{nk}$ |
| Boundary type | Linear (or quadratic with unequal $\Sigma$) | Arbitrary |
| Latent structure | No — labels define classes | Yes — discovers hidden groups |
| Cluster shape | Elliptical, shared across classes | Elliptical, per-component |
| Outlier sensitivity | Higher | Lower (soft assignments dilute outlier influence) |
