# Q21-BF — EM Algorithm
> Week 9. Could ask: derive the ELBO lower bound, explain E-step and M-step, instantiate for GMM, or connect to K-means and ICA.

---

## The Model

**Setting**: a latent variable model. We observe data $X = \{x_1, \ldots, x_N\}$ and posit unobserved (latent) variables $Z = \{z_1, \ldots, z_N\}$ along with model parameters $\theta$.

**Goal**: maximise the observed-data log-likelihood:

$$\log p(X;\theta) = \log \sum_Z p(X, Z;\theta)$$

**Problem**: the sum over all possible values of $Z$ is intractable in general. Even if each $z_i$ takes $K$ discrete values, summing over all $K^N$ joint configurations is infeasible. The EM algorithm circumvents this by iteratively optimising a tractable lower bound.

---

## The ELBO Lower Bound

Introduce an arbitrary distribution $q(Z)$ over the latent variables. Apply Jensen's inequality to the concave log function:

$$\log p(X;\theta) = \log \sum_Z q(Z) \frac{p(X,Z;\theta)}{q(Z)} \geq \sum_Z q(Z) \log \frac{p(X,Z;\theta)}{q(Z)} =: \mathcal{L}(q,\theta)$$

$\mathcal{L}(q,\theta)$ is the **Evidence Lower BOund (ELBO)**. The gap between the log-likelihood and the ELBO is the KL divergence from $q$ to the posterior:

$$\log p(X;\theta) - \mathcal{L}(q,\theta) = \mathrm{KL}(q(Z) \| p(Z|X;\theta)) \geq 0$$

The bound is **tight** ($= 0$) when $q(Z) = p(Z|X;\theta)$ (the true posterior).

---

## The E-Step and M-Step

EM alternates between two optimisation steps:

**E-Step** (Expectation): Fix $\theta$. Maximise $\mathcal{L}(q,\theta)$ over $q$.

Since $\log p(X;\theta)$ does not depend on $q$, maximising the ELBO over $q$ minimises $\mathrm{KL}(q \| p(Z|X;\theta))$. The unique minimiser is:

$$q^*(Z) = p(Z | X;\theta)$$

That is, set $q$ to the current posterior of the latent variables. After the E-step, the ELBO equals the log-likelihood.

**M-Step** (Maximisation): Fix $q = q^*(Z) = p(Z|X;\theta)$. Maximise $\mathcal{L}(q,\theta)$ over $\theta$.

$$\mathcal{L}(q^*,\theta) = \sum_Z p(Z|X;\theta_\text{old}) \log p(X,Z;\theta) - \underbrace{\sum_Z p(Z|X;\theta_\text{old})\log p(Z|X;\theta_\text{old})}_{\text{constant w.r.t. }\theta}$$

So the M-step maximises the **expected complete-data log-likelihood**:

$$\theta_\text{new} = \arg\max_\theta \; E_{Z|X;\theta_\text{old}}[\log p(X,Z;\theta)]$$

This is often tractable even when the original marginal is not, because $\log p(X,Z;\theta)$ typically factors over observations.

---

## Why the Likelihood Never Decreases

At each EM iteration:
1. **After E-step**: $\mathcal{L}(q^*,\theta_\text{old}) = \log p(X;\theta_\text{old})$ (ELBO is tight).
2. **After M-step**: $\mathcal{L}(q^*,\theta_\text{new}) \geq \mathcal{L}(q^*,\theta_\text{old})$ (M-step maximises over $\theta$).
3. **But**: $\log p(X;\theta_\text{new}) \geq \mathcal{L}(q^*,\theta_\text{new}) \geq \mathcal{L}(q^*,\theta_\text{old}) = \log p(X;\theta_\text{old})$.

Therefore $\log p(X;\theta_\text{new}) \geq \log p(X;\theta_\text{old})$ after every iteration. The observed-data log-likelihood is monotonically non-decreasing.

---

## GMM Instance

**Model**: Gaussian Mixture Model with $K$ components.

- Latent variable: $z_i \in \{1,\ldots,K\}$ = cluster assignment for observation $i$.
- Joint: $p(x_i,z_i=k;\theta) = \pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)$.
- Parameters: $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$.

**E-step**: compute **responsibilities** (soft cluster memberships):

$$\gamma_{ik} = p(z_i=k | x_i;\theta) = \frac{\pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i;\mu_j,\Sigma_j)}$$

**M-step**: update parameters using $\gamma_{ik}$ as weights (let $N_k = \sum_i \gamma_{ik}$):

$$\pi_k^{\text{new}} = \frac{N_k}{N}, \qquad \mu_k^{\text{new}} = \frac{\sum_i \gamma_{ik} x_i}{N_k}, \qquad \Sigma_k^{\text{new}} = \frac{\sum_i \gamma_{ik}(x_i - \mu_k^{\text{new}})(x_i-\mu_k^{\text{new}})^T}{N_k}$$

All M-step updates are **closed-form** weighted means and covariances — this is why EM is tractable for GMM despite the latent variables.

---

## Connection to K-Means

K-means is **hard-assignment EM** for a GMM with:
1. Spherical, equal covariances: $\Sigma_k = \sigma^2 I$ for all $k$.
2. Equal mixing weights: $\pi_k = 1/K$.
3. Hard assignments in the E-step: $\gamma_{ik} = \mathbf{1}(k = \arg\min_{k'} \|x_i - \mu_{k'}\|^2)$.

As $\sigma^2 \to 0$, the soft responsibilities $\gamma_{ik}$ converge to hard assignments (a point is entirely in the nearest cluster). GMM is the soft, probabilistic generalisation of K-means.

---

## Key Properties

**Convergence**: EM converges to a stationary point of $\log p(X;\theta)$ — but not necessarily the global maximum. The solution found depends on initialisation. Multiple random starts are recommended.

**Local optima**: common failure mode. For GMM, a component can collapse onto a single point ($\Sigma_k \to 0$, likelihood $\to \infty$) — a degenerate solution. Mitigation: regularise covariances (add $\epsilon I$), use robust initialisations (K-means++).

**Convergence speed**: EM can be slow to converge near saddle points and plateaus. The rate of convergence depends on the fraction of "missing information" — more latent uncertainty → slower convergence.

---

## Comparison to Alternatives

| Property | EM | K-means | Direct MLE (no latent) |
|----------|----|---------|------------------------|
| Assignment | Soft ($\gamma_{ik}\in[0,1]$) | Hard ($\in\{0,1\}$) | N/A |
| Objective | $\log p(X;\theta)$ (likelihood) | WCSS | $\log p(X;\theta)$ |
| Cluster shape | Ellipsoidal ($\Sigma_k$) | Spherical equal | N/A |
| Convergence | Local maximum | Local minimum | Closed-form (when feasible) |
| Probabilistic? | Yes | No | Yes |

---

## Limitations

- **Local optima only**: no convergence guarantee to global maximum.
- **Initialisation-sensitive**: poor initialisations can lead to degenerate solutions or slow convergence.
- **Choosing $K$**: EM requires specifying the number of components. Selection via BIC: $\text{BIC}(K) = -2\log\hat{L} + d_K\log N$.
- **Degenerate solutions**: a GMM component can collapse, causing $\log p(X;\theta) \to \infty$ without a meaningful fit.
- **Tractable E-step required**: EM only applies when $p(Z|X;\theta)$ can be computed or approximated. For complex posteriors, variational inference or MCMC is needed.

---

## Additional Possible Exam Questions

**Q: Why does Jensen's inequality give a lower bound here?**
Jensen's inequality states that for a concave function $f$ and a distribution $q$: $f(\mathbb{E}_q[X]) \geq \mathbb{E}_q[f(X)]$. Applied to $\log$ (concave): $\log\sum_Z q(Z)\frac{p(X,Z)}{q(Z)} \geq \sum_Z q(Z)\log\frac{p(X,Z)}{q(Z)}$. The gap is the KL divergence $\mathrm{KL}(q\|p(Z|X))$, which is always $\geq 0$.

**Q: Explain the E-step for GMM in words.**
In the E-step, we ask: "given our current estimates of cluster means, covariances, and weights, how likely is each observation to belong to each cluster?" The answer is the responsibility $\gamma_{ik}$, computed by Bayes' rule. It is a soft probability, not a hard assignment. Observations near cluster $k$'s centre get high $\gamma_{ik}$; those equidistant between two clusters get $\gamma_{ik} \approx 0.5$ for both.

**Q: What happens if you run EM for a GMM with $K$ too large?**
Several failure modes: (1) component collapse — one or more components shrink to zero variance and capture a single point ($\Sigma_k \to 0$, $\pi_k \to 0$), causing a degenerate likelihood spike; (2) many components may converge to nearly identical parameters (duplicated components), providing no additional information; (3) the remaining components are poorly estimated from the data split among too many parts. Select $K$ by minimising BIC to penalise overfitting.

**Q: Why can EM get stuck at saddle points?**
EM guarantees only that $\log p(X;\theta)$ does not decrease at each step. Near a saddle point, both E-step and M-step produce negligible improvements — the algorithm stalls. The likelihood surface for latent variable models is highly non-convex with many saddle points, especially in high dimensions. Multiple random restarts with different initialisations (e.g., from K-means outputs) help escape local saddle regions.

**Q: How is EM connected to coordinate ascent?**
EM alternates between optimising $\mathcal{L}(q,\theta)$ over $q$ (E-step) and over $\theta$ (M-step). This is coordinate ascent on the ELBO in the $(q, \theta)$ space. Like all coordinate ascent algorithms, each step is monotonically non-decreasing; the algorithm converges when neither coordinate can improve the objective.
