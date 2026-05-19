# Q21-Z — Gaussian Mixture Models (GMM)
> Week 9. Standalone deep dive. Could ask to derive EM, explain soft clustering, or compare to K-means.

---

## The Model

A GMM models the data distribution as a weighted sum of $K$ Gaussian components:
$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x;\mu_k,\Sigma_k)$$

- $\pi_k \geq 0$, $\sum_k \pi_k = 1$: mixing weights (prior probabilities of each component)
- $\mu_k \in \mathbb{R}^p$: component mean
- $\Sigma_k \in \mathbb{R}^{p\times p}$: component covariance (positive definite)
- Parameters $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$

**Latent variable view**: introduce unobserved cluster assignment $Z_i \in \{1,\ldots,K\}$:
$$Z_i \sim \text{Categorical}(\pi_1,\ldots,\pi_K)$$
$$x_i | Z_i=k \sim \mathcal{N}(\mu_k,\Sigma_k)$$

The marginal $p(x_i) = \sum_k \pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)$ integrates out $Z_i$.

---

## Why Not Direct MLE?

The log-likelihood:
$$\ell(\theta) = \sum_{i=1}^N \log\left[\sum_{k=1}^K \pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)\right]$$

The $\log$ of a sum has no closed-form solution — the $k$ component assignments are unknown. If we knew which component generated each $x_i$, MLE would be trivial (separate Gaussians per group). The EM algorithm handles this by iterating between inferring soft assignments and updating parameters.

---

## The EM Algorithm for GMM

**Initialize**: $\mu_k$ (e.g., K-means++), $\Sigma_k = I$, $\pi_k = 1/K$

**E-step** — compute posterior (soft) assignment responsibilities:
$$\gamma_{ik} = P(Z_i=k|x_i,\theta) = \frac{\pi_k \, \mathcal{N}(x_i;\mu_k,\Sigma_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x_i;\mu_j,\Sigma_j)}$$

$\gamma_{ik} \in [0,1]$ is the probability that observation $i$ belongs to component $k$. Each row sums to 1: $\sum_k \gamma_{ik} = 1$.

**M-step** — update parameters using responsibilities as weights:
$$N_k = \sum_{i=1}^N \gamma_{ik} \quad \text{(effective number of points in component } k\text{)}$$
$$\mu_k \leftarrow \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik} x_i$$
$$\Sigma_k \leftarrow \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T$$
$$\pi_k \leftarrow \frac{N_k}{N}$$

**Iterate** E → M → E → M until log-likelihood converges (increases monotonically, guaranteed).

**Convergence**: always converges (log-likelihood never decreases), but to a **local maximum**. Run multiple times with different initializations; keep the run with highest final log-likelihood.

---

## Why EM Works: The Lower Bound View

EM maximizes a lower bound on the log-likelihood:
$$\ell(\theta) \geq \mathcal{L}(\theta, q) = \sum_i \sum_k q_{ik}\log\frac{\pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)}{q_{ik}}$$

- **E-step**: fix $\theta$, optimize $q_{ik}$ → optimal $q_{ik}^* = \gamma_{ik}$ (posterior), tightens the bound
- **M-step**: fix $q_{ik} = \gamma_{ik}$, optimize $\theta$ → closed-form updates, increases the bound

Each iteration increases $\ell$ or keeps it the same → convergence guaranteed.

---

## Soft vs Hard Clustering

| Property | GMM (soft) | K-means (hard) |
|----------|-----------|---------------|
| Assignment | $\gamma_{ik} \in [0,1]$ (probability) | $z_{ik} \in \{0,1\}$ (binary) |
| Cluster shape | Ellipsoidal (any $\Sigma_k$) | Spherical (isotropic) |
| Boundary | Probabilistic (soft) | Hard (Voronoi) |
| Uncertainty | Quantified ($\gamma_{ik}$) | Not modeled |
| Special case | K-means is GMM with $\Sigma_k=\sigma^2I$, hard assignments | — |

**GMM → K-means**: set $\Sigma_k = \sigma^2 I$ (spherical, equal) and take $\sigma\to 0$. The responsibilities $\gamma_{ik}$ become 0/1 (hardened to nearest mean). The M-step mean update becomes exactly K-means centroid update.

---

## Choosing K

**BIC** (preferred):
$$\text{BIC}(K) = -2\ell_K(\hat{\theta}) + p_K\log N$$

Number of free parameters for full GMM: $p_K = (K-1) + Kp + Kp(p+1)/2$
- $K-1$: mixing weights (sum to 1)
- $Kp$: means
- $Kp(p+1)/2$: covariance entries (symmetric)

Plot BIC vs $K$; choose $K$ at minimum.

**Practical covariance constraints** (to reduce parameters):
| Model | $\Sigma_k$ | Parameters |
|-------|-----------|-----------|
| Full | Each class: full $p\times p$ | Most flexible, most params |
| Diagonal | Only diagonal elements per class | Fewer params, assumes independent features |
| Spherical | $\sigma_k^2 I$ | Fewest params, isotropic |
| Tied | Shared $\Sigma$ (= LDA assumption) | Least flexible |

---

## GMM as Density Estimator

A GMM with $K$ components estimates the full data density $p(x)$, not just cluster assignments. This enables:
- Generating new samples: sample $k\sim\text{Categorical}(\pi)$, then $x\sim\mathcal{N}(\mu_k,\Sigma_k)$
- Anomaly detection: flag $x_i$ with low $p(x_i)$ as outliers
- With $K\to\infty$: GMM can approximate any continuous density (universal approximation)

---

## Additional Possible Exam Questions

**Q: Why does EM for GMM converge but not to the global maximum?**
The log-likelihood for GMM is non-convex — it has many local maxima corresponding to different labellings and orderings of the $K$ components. EM is guaranteed to find a local maximum (each step increases $\ell$) but the specific maximum found depends on initialization. Global optimization is NP-hard for GMM. In practice: run EM 10–20 times with different K-means++ initializations, keep the best.

**Q: What is the degenerate solution in GMM fitting and how do you avoid it?**
If a component's covariance collapses to zero ($\Sigma_k\to 0$), its density becomes a spike at one training point — infinite likelihood. This is a degenerate solution that makes $\ell\to\infty$ but is not useful. Solutions: (1) regularize covariance: $\Sigma_k \leftarrow \Sigma_k + \epsilon I$; (2) restart if any $N_k < \text{threshold}$; (3) use BIC to discourage overfitting.

**Q: How does increasing K affect GMM fit?**
More components → higher log-likelihood (more flexibility to fit the data). But BIC penalizes the growing parameter count $p_K\log N$. The optimal $K$ balances fit against complexity. For $K>K_\text{true}$ (true number of clusters): some components may split a true cluster or become redundant (small $\pi_k$). BIC detects this because the likelihood gain is small relative to the penalty.

**Q: When would you use GMM instead of K-means in practice?**
(1) When clusters have different shapes or sizes (different $\Sigma_k$). (2) When you need soft assignments — probabilistic membership $\gamma_{ik}$ for uncertainty quantification. (3) When you want to use the estimated density $p(x)$ for generation or anomaly detection. (4) When BIC model selection is important (GMM integrates naturally with BIC; K-means does not have a likelihood).

**Q: What is the relationship between GMM and LDA?**
Both model data as class-conditional Gaussians. LDA: supervised (class labels known), shared covariance $\Sigma_k=\Sigma$ (pooled), closed-form MLE. GMM: unsupervised (cluster assignments latent), per-component $\Sigma_k$, fitted by EM. If you fit a GMM to labeled data and pool the covariances, you recover LDA parameters. GMM fitted with tied (shared) covariance + hard assignments → equivalent to LDA in the large-sample limit.
