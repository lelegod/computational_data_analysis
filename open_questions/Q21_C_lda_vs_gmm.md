# Q21-C — LDA vs GMM Comparison
> Appeared: 2025 Q21

---

## Shared Foundation

Both model data as **Gaussian distributions**:
- LDA: each class $k$ has a Gaussian class-conditional $P(x|C_k) = \mathcal{N}(\mu_k, \Sigma)$
- GMM: each component $j$ has $P(x|Z=j) = \mathcal{N}(\mu_j, \Sigma_j)$

The key differences: supervision, covariance structure, fitting method, goal.

---

## Side-by-Side Comparison

| Property | LDA | GMM |
|----------|-----|-----|
| Supervision | Supervised (class labels required) | Unsupervised (no labels) |
| Covariance | Shared $\Sigma$ across all classes | Per-component $\Sigma_k$ |
| Fitting | Closed-form MLE | EM algorithm (iterative) |
| Goal | Classification (find decision boundary) | Density estimation / clustering |
| Decision boundary | Linear | Quadratic (or nonlinear) |
| Latent variables | None ($y_i$ observed) | $Z_i$ = unobserved cluster assignment |
| Converges to global? | Yes (closed-form) | Local maximum only |
| K selection | $K$ = number of classes (known) | $K$ chosen by BIC or CORCONDIA |

---

## Why LDA's Decision Boundary is Linear

Apply Bayes' rule for class $k$ vs class $k'$:
$$\log\frac{P(C_k|x)}{P(C_{k'}|x)} = \log\frac{\pi_k}{\pi_{k'}} + \log\frac{P(x|C_k)}{P(x|C_{k'})}$$

With Gaussian class-conditionals sharing $\Sigma$:
$$\log\frac{P(x|C_k)}{P(x|C_{k'})} = -\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k) + \frac{1}{2}(x-\mu_{k'})^T\Sigma^{-1}(x-\mu_{k'})$$

Expanding:
$$= x^T\Sigma^{-1}(\mu_k - \mu_{k'}) - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \frac{1}{2}\mu_{k'}^T\Sigma^{-1}\mu_{k'}$$

The quadratic term $-\frac{1}{2}x^T\Sigma^{-1}x$ appears in both class-conditionals and **cancels** because $\Sigma$ is shared. The result is **linear in $x$** → linear decision boundary.

**With unequal covariances (QDA)**: the quadratic terms do NOT cancel → quadratic boundary → QDA is strictly more flexible than LDA but needs more data to estimate $K$ separate $\Sigma_k$.

---

## How GMM is Fitted (EM Algorithm)

**E-step** (soft assignments): using current parameters $\{\mu_j, \Sigma_j, \pi_j\}$:
$$\gamma_{ij} = P(Z_i = j | x_i) = \frac{\pi_j\,\mathcal{N}(x_i|\mu_j,\Sigma_j)}{\sum_{j'}\pi_{j'}\,\mathcal{N}(x_i|\mu_{j'},\Sigma_{j'})}$$

**M-step** (update parameters): using soft assignments as weights:
$$\mu_j \leftarrow \frac{\sum_i \gamma_{ij} x_i}{\sum_i \gamma_{ij}}, \quad \Sigma_j \leftarrow \frac{\sum_i \gamma_{ij}(x_i-\mu_j)(x_i-\mu_j)^T}{\sum_i \gamma_{ij}}, \quad \pi_j \leftarrow \frac{\sum_i \gamma_{ij}}{N}$$

Iterate until convergence. Each iteration is guaranteed to increase the log-likelihood, but convergence is to a **local** maximum.

**Why EM, not closed-form?** Because the cluster assignments $Z_i$ are unobserved. If $Z_i$ were known, MLE would be trivial (separate sample means and covariances per cluster). The latent $Z_i$ makes the likelihood non-convex → EM handles this via soft imputation.

---

## Connection: GMM with Equal Covariances → K-Means

GMM with all $\Sigma_j = \sigma^2 I$ (spherical, equal) + hard assignments ($\gamma_{ij} \in \{0,1\}$):
- E-step becomes: assign each point to nearest centroid
- M-step becomes: update centroids as cluster means
- This is exactly **K-means**

K-means is a degenerate GMM with identity covariances and hard assignments.

---

## Choosing K for GMM

Use **BIC** (Bayesian Information Criterion):
$$\text{BIC} = -2\ell(\hat{\theta}) + p_\theta \log N$$

where $\ell$ is the log-likelihood and $p_\theta$ = number of free parameters (grows with $K$). BIC penalizes complexity more than AIC (penalty $p_\theta \log N$ vs $2p_\theta$). Choose $K$ that minimizes BIC.

---

## Additional Possible Exam Questions

**Q: When would LDA outperform QDA?**
When the true covariances are similar across classes (equal-covariance assumption approximately holds) and/or the sample size is small. LDA estimates $1$ covariance matrix ($p(p+1)/2$ parameters); QDA estimates $K$ separate ones ($Kp(p+1)/2$ parameters) — dramatically more parameters for large $p$ → QDA overfits with small $N$.

**Q: When would QDA outperform LDA?**
When classes have clearly different covariance structures (different shapes, orientations, or sizes of clusters). With sufficient data, QDA captures this and produces lower classification error.

**Q: What if $p > N$ in LDA?**
$\Sigma$ becomes singular (cannot be inverted). Solutions: (1) regularized LDA: $\hat{\Sigma}_\lambda = (1-\lambda)\hat{\Sigma} + \lambda I$; (2) use pseudo-inverse or PCA preprocessing; (3) diagonal LDA (naive Bayes assumption).

**Q: Is GMM always better than K-means?**
No. K-means is simpler, faster, and interpretable. GMM is better when: clusters have different sizes/shapes, or soft assignments are needed (uncertainty quantification). K-means is fine when clusters are roughly spherical and equal-sized.

**Q: Can LDA be used for dimensionality reduction?**
Yes — this is LDA's other interpretation. LDA finds the projection directions that maximize between-class variance relative to within-class variance:
$$\max_w \frac{w^T S_B w}{w^T S_W w}$$
where $S_B$ = between-class scatter, $S_W$ = within-class scatter. This is the **generalized eigenvalue problem** $S_B w = \lambda S_W w$. At most $K-1$ discriminant directions exist. Compare to PCA which finds directions of maximum total variance (ignores class labels).

**Q: What is the relationship between GMM and density estimation?**
A GMM with $K$ components is a parametric density estimator: $p(x) = \sum_k \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$. With enough components, it can approximate any continuous density (universal approximation for sufficiently large $K$). Contrast with kernel density estimation (KDE), which is nonparametric.
