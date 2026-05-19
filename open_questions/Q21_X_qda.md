# Q21-X — QDA: Quadratic Discriminant Analysis
> Extension of LDA when covariance matrices differ per class. Natural follow-up to C (LDA vs GMM).

---

## From LDA to QDA

Both LDA and QDA assume class-conditional Gaussians:
$$P(x|C_k) = \mathcal{N}(x;\mu_k, \Sigma_k)$$

**LDA**: forces $\Sigma_k = \Sigma$ for all classes → linear boundary.
**QDA**: allows each class to have its own $\Sigma_k$ → quadratic boundary.

---

## Why QDA Gives a Quadratic Boundary

Apply Bayes' rule:
$$\log\frac{P(C_k|x)}{P(C_{k'}|x)} = \log\frac{\pi_k}{\pi_{k'}} + \log\frac{P(x|C_k)}{P(x|C_{k'})}$$

With class-specific covariances:
$$\log\frac{P(x|C_k)}{P(x|C_{k'})} = -\frac{1}{2}\log\frac{|\Sigma_k|}{|\Sigma_{k'}|} - \frac{1}{2}x^T(\Sigma_k^{-1}-\Sigma_{k'}^{-1})x + x^T(\Sigma_k^{-1}\mu_k - \Sigma_{k'}^{-1}\mu_{k'}) + \text{const}$$

The term $-\frac{1}{2}x^T(\Sigma_k^{-1}-\Sigma_{k'}^{-1})x$ is **quadratic in $x$** because $\Sigma_k \neq \Sigma_{k'}$ → the quadratic terms do NOT cancel → quadratic decision boundary.

In LDA with $\Sigma_k=\Sigma$: $\Sigma_k^{-1}-\Sigma_{k'}^{-1} = 0$ → quadratic term vanishes → linear boundary.

---

## Fitting QDA

**Closed-form MLE** (same as LDA but per class):
$$\hat{\mu}_k = \frac{1}{N_k}\sum_{i:y_i=k} x_i$$
$$\hat{\Sigma}_k = \frac{1}{N_k}\sum_{i:y_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T$$
$$\hat{\pi}_k = N_k/N$$

No iterative algorithm needed — compute sample means and covariances per class.

**Number of parameters**:
- LDA: $Kp + p(p+1)/2 + K$ (means + one shared covariance + priors)
- QDA: $Kp + Kp(p+1)/2 + K$ (means + $K$ covariance matrices + priors)
- QDA has $K$ times more covariance parameters than LDA

---

## LDA vs QDA: When to Use Each

| Scenario | Use |
|----------|-----|
| Small training set, $N/p$ moderate | LDA (fewer parameters, lower variance) |
| Large training set, enough data per class | QDA (more flexible, lower bias) |
| Classes have clearly different shapes/orientations | QDA |
| Classes have similar spread | LDA |
| $p$ close to $N_k$ | Neither — use regularization |
| Want interpretable boundary | LDA (hyperplane) |

**Bias-variance perspective**: LDA has more bias (imposes equal covariance) but lower variance (estimates one $\Sigma$ instead of $K$). QDA has less bias (fits class-specific covariances) but higher variance ($K\times$ more covariance parameters). For large $N$: QDA wins. For small $N$: LDA wins.

---

## Regularized Discriminant Analysis (RDA)

Interpolates between LDA and QDA:
$$\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}$$

- $\alpha=1$: QDA (class-specific covariances)
- $\alpha=0$: LDA (pooled covariance)
- $\alpha\in(0,1)$: shrinks each class covariance toward the pooled estimate

A second parameter $\gamma$ can further regularize toward $\hat{\sigma}^2 I$ (spherical):
$$\hat{\Sigma}_k(\alpha,\gamma) = (1-\gamma)[\alpha\hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}] + \gamma\hat{\sigma}^2 I$$

Choose $\alpha,\gamma$ by cross-validation.

---

## LDA vs QDA vs Logistic Regression

| Method | Assumption on $x$ | Boundary | Parameters | Works $p>N$? |
|--------|------------------|----------|-----------|-------------|
| LDA | Gaussian, shared $\Sigma$ | Linear | $O(Kp + p^2)$ | No ($\Sigma$ singular) |
| QDA | Gaussian, per-class $\Sigma_k$ | Quadratic | $O(Kp + Kp^2)$ | No |
| RDA | Gaussian, regularized | Quadratic | $O(Kp + p^2)$ + 2 hyperparams | Partially |
| Logistic | None | Linear | $O(Kp)$ | With regularization |

---

## Additional Possible Exam Questions

**Q: Under what exact condition does QDA reduce to LDA?**
When all class covariance matrices are equal: $\Sigma_1 = \Sigma_2 = \cdots = \Sigma_K = \Sigma$. Then $\Sigma_k^{-1}-\Sigma_{k'}^{-1} = 0$, the quadratic term vanishes, and the log-posterior ratio becomes linear in $x$.

**Q: Why is QDA more susceptible to overfitting than LDA?**
QDA estimates $K$ separate $p\times p$ covariance matrices — $Kp(p+1)/2$ parameters total. With small $N_k$, these estimates are noisy (high variance). A poorly estimated $\hat{\Sigma}_k$ can produce a wildly curved decision boundary that fits training data but generalizes poorly. LDA pools all classes to estimate one covariance — $p(p+1)/2$ parameters, much more stable.

**Q: What happens when $N_k < p$ in QDA?**
$\hat{\Sigma}_k$ is singular (rank at most $N_k-1 < p$) → cannot be inverted → QDA formula breaks down. Solutions: (1) regularize: $\hat{\Sigma}_k + \lambda I$; (2) use RDA with $\alpha < 1$ to borrow from pooled estimate; (3) use PCA preprocessing to reduce $p$ below $N_k$ before fitting QDA.

**Q: Is QDA a generative or discriminative model?**
Generative — like LDA, it models $P(x|C_k)$ (class-conditional density) and applies Bayes' rule to compute $P(C_k|x)$. This means it can also generate synthetic samples from each class. Logistic regression is discriminative (models $P(C_k|x)$ directly). Generative models are more efficient when assumptions hold; discriminative models are more robust when assumptions are violated.
