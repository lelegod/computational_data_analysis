# Practice Set 2 — CDA 02582 (Combined Questions + Solutions)

**Format:** 20 multiple-choice + 2 open questions
**Scoring:** MC: +1 (correct), −0.25 (incorrect), 0 (unanswered)
**Duration:** 4 hours

---

## Multiple Choice

---

**Question (1)** [Week 1]
Which of the following statements about the Expected Prediction Error (EPE) are correct?

A. The irreducible error $\sigma^2$ can be reduced by choosing a sufficiently complex model.
B. $\text{EPE} = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$, where the expectation is taken over both new observations $y$ and the training set $\mathcal{D}$.
C. As model complexity increases, training error decreases monotonically while test error forms a U-shape.
D. The variance component $E[(\hat{f} - E[\hat{f}])^2]$ measures how far the average prediction is from the true function.
E. None of the above.

#### Answer: **B, C**

- **A ✗** — $\sigma^2$ is irreducible — no model can eliminate it; it is inherent noise in the data-generating process.
- **B ✓** — $\text{EPE} = \text{Bias}^2 + \text{Var} + \sigma^2$, averaged over $y$ and $\mathcal{D}$ — this is the standard decomposition.
- **C ✓** — Training error always decreases with model complexity (degrees of freedom increase), while test error forms a U-shape (bias decreases, variance increases).
- **D ✗** — D describes the bias, not variance. Variance is $E[(\hat{f} - E[\hat{f}])^2]$ (fluctuation around the mean prediction), while bias is $(E[\hat{f}] - f)^2$ (deviation of mean prediction from truth).
- **E ✗** — B and C are correct.

---

**Question (2)** [Week 1]
For a linear smoother $\hat{Y} = SY$, the effective degrees of freedom is defined as $df(S) = \text{trace}(S)$. Which of the following are correct?

A. For OLS regression with $p$ predictors, $df = p$.
B. For Ridge regression, as $\lambda \to \infty$ the effective degrees of freedom approaches $p$.
C. For Ridge regression, as $\lambda \to 0$ the effective degrees of freedom approaches $p$.
D. AIC with $d = df(\lambda)$ for Ridge regression is asymptotically equivalent to leave-one-out cross-validation.
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — For OLS with $p$ predictors, $S = X(X^TX)^{-1}X^T$ and $\text{trace}(S) = p$.
- **B ✗** — As $\lambda \to \infty$, $df \to 0$ (not $p$) — extreme regularization shrinks all coefficients to zero, eliminating all effective parameters.
- **C ✓** — As $\lambda \to 0$, ridge approaches OLS, so $df(\lambda) = \text{trace}(S_\lambda) \to p$.
- **D ✓** — Stone (1977) proved AIC is asymptotically equivalent to LOO-CV; since AIC uses $d = df(\lambda)$ for ridge, this extends directly.
- **E ✗** — A, C, D are correct.

---

**Question (3)** [Week 2]
Consider the Benjamini-Hochberg (BH) procedure applied to $m = 6$ hypothesis tests with sorted p-values $p_{(1)} = 0.004$, $p_{(2)} = 0.012$, $p_{(3)} = 0.038$, $p_{(4)} = 0.080$, $p_{(5)} = 0.210$, $p_{(6)} = 0.430$, and a target FDR level $q = 0.10$. How many hypotheses are rejected?

A. 1
B. 2
C. 3
D. 4
E. None of the above.

#### Answer: **C**

BH rule: find the largest $k$ where $p_{(k)} \leq (k/m) \cdot q$, then reject all hypotheses $1, \ldots, k$.

Thresholds: $i=1$: $0.10/6 \approx 0.0167$; $i=2$: $0.0333$; $i=3$: $0.0500$; $i=4$: $0.0667$; $i=5$: $0.0833$; $i=6$: $0.100$.

- $p_{(1)} = 0.004 \leq 0.0167$ ✓
- $p_{(2)} = 0.012 \leq 0.0333$ ✓
- $p_{(3)} = 0.038 \leq 0.0500$ ✓
- $p_{(4)} = 0.080 > 0.0667$ ✗ — stop here

Largest satisfying $k = 3$ → reject $H_{(1)}, H_{(2)}, H_{(3)}$. Answer: **C (3 rejections)**.

- **A ✗** — Only 1 rejection would apply Bonferroni, not BH.
- **B ✗** — 2 is wrong; $p_{(3)} = 0.038 \leq 0.050$ passes.
- **C ✓** — 3 rejections, as shown above.
- **D ✗** — $p_{(4)} = 0.080 > 0.0667$ fails the BH threshold.
- **E ✗** — C is correct.

---

**Question (4)** [Week 3]
Which of the following correctly describe the curse of dimensionality?

A. As dimension $D$ grows, a fixed number of training points $N$ becomes exponentially sparse.
B. In high dimensions, most data points tend to reside near the boundary (corners) of the sample space rather than the interior.
C. Euclidean distances become more meaningful as dimension grows, because there are more directions to distinguish points.
D. When $p > N$, OLS can be computed but has high variance because $(X^TX)$ is near-singular.
E. None of the above.

#### Answer: **A, B**

- **A ✓** — A fixed $N$ becomes exponentially sparse as $D$ grows — this is the core statement of the curse.
- **B ✓** — In high dimensions, most data points cluster near the surface/corners of the hypercube (edge effect), not the interior.
- **C ✗** — Euclidean distances LOSE meaning in high dimensions — distances between all pairs of points become nearly equal (concentration of measure), making them uninformative, not more informative.
- **D ✗** — When $p > N$, $(X^TX)$ is not invertible (rank-deficient), so OLS cannot be computed at all — not merely "high variance."
- **E ✗** — A and B are correct.

---

**Question (5)** [Week 4]
Regarding CART and impurity measures, which statements are correct?

A. The Gini index for a node is zero when the node is pure (contains only one class).
B. The misclassification rate is preferred over the Gini index for growing classification trees because it is easier to compute.
C. For a binary classification problem with class probability $p$, the Gini index equals $2p(1-p)$.
D. Cost-complexity pruning generates a sequence of nested subtrees by increasing the complexity penalty $\alpha$.
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — Gini $G = \sum_k p_{mk}(1 - p_{mk})$. When a node is pure ($p_k=1$, all others 0), $G = 1\cdot0 + 0 + \ldots = 0$.
- **B ✗** — Misclassification rate is NOT preferred for growing trees — it is insensitive to probability shifts within the majority class. Gini and cross-entropy are preferred for growing; misclassification rate is used for pruning.
- **C ✓** — For $K=2$ with $p = p_{m1}$: $G = p(1-p) + (1-p)p = 2p(1-p)$.
- **D ✓** — Weakest-link pruning incrementally removes internal nodes as $\alpha$ increases, generating a nested sequence $T_0 \supset T_1 \supset \ldots \supset \text{root}$.
- **E ✗** — A, C, D are correct.

---

**Question (6)** [Week 5]
The variance of the bagged ensemble estimator (with $B$ trees, pairwise correlation $\rho$, and individual tree variance $\sigma^2$) is:

$$\text{Var}(\hat{y}_\text{bag}) = \rho\sigma^2 + \frac{(1-\rho)}{B}\sigma^2$$

Which of the following conclusions follow directly from this formula?

A. As $B \to \infty$, the ensemble variance approaches zero regardless of $\rho$.
B. If all trees are perfectly correlated ($\rho = 1$), bagging provides no variance reduction.
C. Bagging reduces bias as well as variance when $B$ is large.
D. The out-of-bag error estimate approximates leave-one-out cross-validation error.
E. None of the above.

#### Answer: **B, D**

- **A ✗** — As $B \to \infty$, $\text{Var} \to \rho\sigma^2$, not 0. The term $\rho\sigma^2$ remains as a floor regardless of $B$.
- **B ✓** — Setting $\rho=1$: $\text{Var} = \sigma^2 + 0 = \sigma^2$ = variance of a single tree. Bagging provides no reduction.
- **C ✗** — Bagging does NOT reduce bias — the bias of the bagged estimator equals the bias of any single tree (they are identically distributed): $E[(1/B)\sum(\hat{y}_b - y)] = E[\hat{y}_b - y]$.
- **D ✓** — Each observation's OOB prediction is formed from trees that never saw it during training — this exactly mimics LOO-CV's structure; empirically the errors are very similar.
- **E ✗** — B and D are correct.

---

**Question (7)** [Week 6]
Which of the following statements about AdaBoost.M1 are correct?

A. The classifier weight $\alpha_m = \log[(1 - \text{err}_m)/\text{err}_m]$ is negative when $\text{err}_m > 0.5$, effectively negating the weak classifier.
B. At each iteration, the weights of misclassified observations are decreased so future classifiers focus on easy cases.
C. AdaBoost is equivalent to forward stagewise additive modelling with exponential loss $L(y, f) = \exp(-y \cdot f(x))$.
D. Boosting primarily reduces variance rather than bias, which is why it uses shallow trees (stumps).
E. None of the above.

#### Answer: **A, C**

- **A ✓** — $\alpha_m = \log[(1-\text{err}_m)/\text{err}_m]$. When $\text{err}_m > 0.5$: $(1-\text{err}_m)/\text{err}_m < 1$, so $\log < 0 \Rightarrow \alpha_m < 0$. A negative weight negates the classifier's votes in the final sum, effectively reversing its predictions.
- **B ✗** — Weights of MISCLASSIFIED observations are INCREASED (they get more attention), not decreased. Correctly classified observations get relatively less weight.
- **C ✓** — This is the theoretical result of Friedman, Hastie & Tibshirani (2000) — AdaBoost minimizes the expected exponential loss via forward stagewise fitting.
- **D ✗** — Boosting primarily reduces BIAS (it uses weak learners with high bias like stumps and corrects their errors). Bagging reduces variance. The use of stumps is precisely because shallow trees are high-bias weak learners — boosting corrects the bias sequentially.
- **E ✗** — A and C are correct.

---

**Question (8)** [Week 7]
Consider the hard-margin SVM primal problem:

$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{subject to} \quad y_i(x_i^T\beta + \beta_0) \geq 1 \quad \forall i$$

Which of the following are correct?

A. The margin width in the canonical formulation equals $1/\|\beta\|$.
B. A point $x_i$ is a support vector if and only if its Lagrange multiplier $\alpha_i > 0$.
C. In the dual formulation, the training data appear only as pairwise dot products $\langle x_i, x_j \rangle$, enabling the kernel trick.
D. Strong duality (zero duality gap) holds for the SVM problem because the problem is convex and Slater's condition is satisfied.
E. None of the above.

#### Answer: **A, B, C, D**

- **A ✓** — With canonical scaling $|x_i^T\beta + \beta_0| = 1$ for support vectors, the signed distance for a support vector is $1/\|\beta\|$, so the total margin (from $+1$ to $-1$ boundary) is $2/\|\beta\|$, and the half-margin is $1/\|\beta\|$.
- **B ✓** — This follows directly from KKT complementary slackness: $\alpha_i[y_i(x_i^T\beta+\beta_0)-1]=0$. If a point is not on the margin (bracket $> 0$), $\alpha_i$ must be 0. Support vectors have $\alpha_i > 0$, so they sit exactly on the margin.
- **C ✓** — The dual objective $\sum\alpha_i - (1/2)\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$ contains $x_i$ only inside inner products — replacing $\langle x_i, x_j \rangle$ with $K(x_i, x_j)$ is the kernel trick.
- **D ✓** — The SVM primal is convex (quadratic objective, linear constraints). Slater's condition (existence of strictly feasible point) holds for separable data, so strong duality applies.
- **E ✗** — All of A, B, C, D are correct.

---

**Question (9)** [Week 8]
A researcher applies Principal Component Regression (PCR) to a dataset where the response $y$ is weakly correlated with $X$ but some features in $X$ have very high variance. Which method would most likely outperform PCR on this dataset, and why?

A. Ridge regression, because it shrinks all coefficients uniformly toward zero.
B. Partial Least Squares (PLS), because it finds components that maximize covariance with $y$ rather than variance in $X$.
C. Lasso regression, because it performs automatic variable selection.
D. CCA (Canonical Correlation Analysis), because it maximizes the correlation between X-scores and Y-scores, ignoring internal variance.
E. None of the above.

#### Answer: **B**

- **A ✗** — Ridge shrinks all coefficients but does not prioritize $y$-relevant directions.
- **B ✓** — When $y$ is weakly correlated with $X$ but some features have high variance, PCA selects directions of maximum variance in $X$ — these may have zero correlation with $y$. PLS explicitly maximizes $\text{Cov}(X\alpha, y) = \text{Var}(X\alpha) \cdot \text{Corr}^2(X\alpha, y)$, keeping directions that predict $y$. PCR keeps the high-variance but $y$-irrelevant directions; PLS elevates directions with correlation to $y$.
- **C ✗** — Lasso selects variables but still fits in original feature space, not a supervised subspace.
- **D ✗** — CCA maximizes correlation but requires $Y$ to also be a matrix; with scalar $y$, CCA reduces to a correlation maximization that ignores variance, which is less suitable than PLS.
- **E ✗** — B is correct.

---

**Question (10)** [Week 8]
Regarding the PLS algorithm, which of the following are correct?

A. The PLS weight for feature $j$ in component $m$ is $\hat{\phi}_{mj} = x_j^{(m-1)T} y$, assigning higher weight to features correlated with $y$.
B. PLS latent components $z_1, z_2, \ldots, z_m$ are mutually uncorrelated (orthogonal).
C. If $M = p$ (all components are kept), PLS predictions are identical to OLS predictions.
D. Canonical Correlation Analysis and PLS have identical objectives — both maximize covariance between X-scores and Y-scores.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — The weight step $\hat{\phi}_{mj} = x_j^{(m-1)T} y$ is the dot product of feature $j$ (after deflation) with $y$ — features more correlated with $y$ get higher weight.
- **B ✓** — This is a provable property of the PLS deflation step — the deflation removes variance explained by $z_m$ from all features, ensuring subsequent components are orthogonal to $z_m$.
- **C ✓** — When $M=p$ all components are extracted; the final PLS prediction equals OLS because the algorithm has extracted all possible directions in $X$.
- **D ✗** — PLS maximizes $\text{Cov}(Xu, Yv) = \sqrt{\text{Var}(Xu) \cdot \text{Var}(Yv)} \cdot \text{Corr}(Xu, Yv)$, balancing variance AND correlation. CCA maximizes only $\text{Corr}^2(Xu, Yv)$, ignoring internal variance. They have different objectives.
- **E ✗** — A, B, C are correct.

---

**Question (11)** [Week 9]
Consider running K-means clustering on a dataset with well-separated elliptical clusters of different sizes. Which statements are correct?

A. K-means will always fail on elliptical clusters because it minimizes squared Euclidean distance to spherical centroids.
B. K-medoids is more robust to outliers than K-means because cluster centers are constrained to be actual data points.
C. Gaussian Mixture Models with full per-cluster covariance matrices can better capture elliptical cluster shapes than K-means.
D. The gap statistic selects $K$ by comparing the log within-cluster dissimilarity of the data to that of uniformly distributed data.
E. None of the above.

#### Answer: **B, C, D**

- **A ✗** — "Always fail" is too strong. K-means can partition elliptical clusters if they are well-separated, though the partition may not match the elliptical structure. The deeper problem is that K-means assumes spherical clusters and equal cluster sizes.
- **B ✓** — K-medoids uses actual data points as cluster centers; outliers cannot become centroids, and pulling outliers cannot distort medoids as strongly as they distort means.
- **C ✓** — GMM with full (or diagonal) per-cluster covariance matrices can model elliptical, elongated, and differently-sized clusters that K-means (spherical assumption) misses.
- **D ✓** — $G(K) = \log(U_k) - \log(W_k)$, where $U_k$ is the expected within-cluster dissimilarity under uniform data (from Monte Carlo), and $W_k$ is the actual within-cluster dissimilarity. The selection rule picks the smallest $K$ where the gap is sufficiently large.
- **E ✗** — B, C, D are correct.

---

**Question (12)** [Week 9]
The silhouette score for observation $i$ is defined as:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

where $a(i)$ is the average distance to points in the same cluster and $b(i)$ is the average distance to points in the nearest other cluster. Which of the following are correct?

A. $s(i) = 1$ indicates observation $i$ is perfectly well-clustered (far from the neighboring cluster, close to its own).
B. $s(i) = 0$ indicates observation $i$ is equidistant between its assigned cluster and the next closest cluster.
C. $s(i) = -1$ indicates observation $i$ has been correctly assigned to its cluster.
D. The optimal $K$ is selected by minimizing the average silhouette score across all observations.
E. None of the above.

#### Answer: **A, B**

- **A ✓** — $s(i)=1$ requires $b(i) \gg a(i)$: very close to own cluster center and far from nearest other cluster. $s(i) = (b(i)-0)/b(i) = 1$. This is perfect clustering.
- **B ✓** — $s(i)=0$ means $b(i)=a(i)$ — the observation is equidistant between its cluster and the next closest. It sits on the decision boundary.
- **C ✗** — $s(i)=-1$ means $a(i) \gg b(i)$, i.e., the observation is much closer to the neighboring cluster than to its own — this indicates MIS-clustering (it should be in the neighboring cluster).
- **D ✗** — The optimal $K$ is selected by MAXIMIZING (not minimizing) the average silhouette width, since higher silhouette = better-defined clusters.
- **E ✗** — A and B are correct.

---

**Question (13)** [Week 10]
In training a multi-layer perceptron (MLP), the backpropagation algorithm uses the chain rule. The weight gradient for layer $\ell$ is:

$$\frac{\partial L}{\partial W^{(\ell)}} = \delta^{(\ell)} \cdot (a^{(\ell-1)})^T$$

where $\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$. Which of the following are correct?

A. $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ for the sigmoid activation, which is computationally convenient because $\sigma(z)$ is already computed in the forward pass.
B. The backward pass propagates blame (gradients) from output to input, while the forward pass propagates activations from input to output.
C. The gradient $\partial L / \partial W^{(\ell)}$ requires only the error signal at layer $\ell$ and the input activation from the previous layer — not any future layer's weights.
D. Binary cross-entropy loss is derived from the Gaussian likelihood, and is therefore the correct loss for binary classification.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Differentiating $\sigma(x) = 1/(1+e^{-x})$ gives $\sigma'(x) = \sigma(x)(1-\sigma(x))$. Once $\sigma(x)$ is computed in the forward pass, the derivative requires only two multiplications — no additional exponential evaluation.
- **B ✓** — This is the fundamental principle of backpropagation — activations flow forward, gradients flow backward.
- **C ✓** — $\partial L / \partial W^{(\ell)} = \delta^{(\ell)}(a^{(\ell-1)})^T$ — this product involves only the error signal $\delta^{(\ell)}$ at layer $\ell$ and the activation $a^{(\ell-1)}$ from the layer below; no weights from future (deeper) layers appear explicitly.
- **D ✗** — Binary cross-entropy is derived from the BERNOULLI likelihood (Bernoulli($p$) where $p = \sigma(f(x))$). MSE is derived from the Gaussian likelihood. Using BCE for classification and MSE for regression is the correct correspondence.
- **E ✗** — A, B, C are correct.

---

**Question (14)** [Week 10]
A neural network has the architecture: 5 inputs → 8 hidden units (sigmoid) → 4 hidden units (sigmoid) → 2 output units (linear), with biases at every layer. What is the total number of trainable parameters?

A. 78
B. 82
C. 86
D. 94
E. None of the above.

#### Answer: **D**

Layer-by-layer count:
- Layer 1 ($5 \to 8$): $5 \times 8$ weights + 8 biases $= 40 + 8 = 48$
- Layer 2 ($8 \to 4$): $8 \times 4$ weights + 4 biases $= 32 + 4 = 36$
- Layer 3 ($4 \to 2$): $4 \times 2$ weights + 2 biases $= 8 + 2 = 10$

**Total: $48 + 36 + 10 = 94$**

- **A ✗** — 78 results from omitting biases in two layers.
- **B ✗** — 82 is an off-by-one miscounting.
- **C ✗** — 86 miscomputes layer 2.
- **D ✓** — 94 is the correct total.
- **E ✗** — D is correct.

---

**Question (15)** [Week 11]
Regarding Non-negative Matrix Factorization (NMF) with $X \approx WH$, which statements are correct?

A. NMF enforces non-negativity on both $W$ and $H$, leading to parts-based, additive representations.
B. Unlike PCA, the NMF objective is jointly convex in $(W, H)$, guaranteeing convergence to the global optimum.
C. The Lee & Seung multiplicative update rule $H_{kj} \leftarrow H_{kj} \cdot (W^TX)_{kj} / (W^TWH)_{kj}$ guarantees non-negativity throughout optimization if initialized positively.
D. NMF solutions are generally not unique — for any invertible $Q$ with $WQ^{-1} \geq 0$ and $QH \geq 0$, $(WQ^{-1})(QH)$ is an equally valid factorization.
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — NMF requires $W \geq 0$ AND $H \geq 0$. Because both factors are non-negative, each data point is a purely additive combination of parts (no cancellation between components, unlike PCA).
- **B ✗** — The NMF objective is NOT jointly convex in $(W, H)$. It is separately convex in $W$ given $H$, and convex in $H$ given $W$ ("biconvex"), but not jointly — ALS converges to a local minimum, not necessarily the global one.
- **C ✓** — The multiplicative update $H \leftarrow H \odot (W^TX)/(W^TWH)$ multiplies $H$ element-wise by a non-negative ratio — if $H$ starts positive, every element stays positive throughout.
- **D ✓** — Any invertible $Q$ with $WQ^{-1} \geq 0$ and $QH \geq 0$ yields an equally valid NMF, so solutions are not unique without additional constraints (sparsity, geometric volume minimization).
- **E ✗** — A, C, D are correct.

---

**Question (16)** [Week 11]
Which of the following statements correctly distinguish Independent Component Analysis (ICA) from Principal Component Analysis (PCA)?

A. PCA finds directions of maximum variance; ICA finds directions of maximum statistical independence by maximizing non-Gaussianity.
B. ICA can separate statistically independent Gaussian sources, while PCA cannot.
C. After whitening (sphering), the ICA problem reduces to finding an orthogonal rotation matrix.
D. The Cocktail Party Problem (blind source separation) can be solved by PCA alone, since PCA decorrelates the signals.
E. None of the above.

#### Answer: **A, C**

- **A ✓** — PCA maximizes $\text{Var}(Xv)$ — purely a variance objective. ICA maximizes non-Gaussianity (kurtosis, negentropy) as a proxy for statistical independence. These are fundamentally different objectives.
- **B ✗** — ICA CANNOT separate Gaussian sources. By the Central Limit Theorem, all linear mixtures of Gaussians are also Gaussian — there is no way to distinguish sources from mixtures using non-Gaussianity. ICA requires non-Gaussian sources.
- **C ✓** — After whitening, the data's covariance is the identity. The mixing matrix $A$ becomes orthogonal after whitening, reducing the ICA search from arbitrary invertible matrices to orthogonal rotations — a much smaller search space.
- **D ✗** — PCA only decorrelates signals (removes second-order dependencies). Statistical independence requires eliminating ALL higher-order dependencies. Two signals can be uncorrelated but not independent.
- **E ✗** — A and C are correct.

---

**Question (17)** [Week 12]
For a 3-way tensor $\mathcal{X}^{I \times J \times K}$ decomposed using the PARAFAC model:

$$x_{ijk} \approx \sum_{r=1}^R a_{ir} b_{jr} c_{kr}$$

which of the following are correct?

A. PARAFAC is a special case of Tucker3 where the core tensor $\mathcal{G}$ is super-diagonal (identity-like tensor $\mathcal{I}^{R\times R\times R}$).
B. Tucker3 solutions are essentially unique (up to permutation and scaling), while PARAFAC solutions have rotational ambiguity.
C. CORCONDIA close to 100 indicates the PARAFAC model structure is appropriate for the chosen $R$.
D. In the PARAFAC ALS update, the mode-1 loading matrix is updated as $A \leftarrow X_{(1)} (C \odot B)(C^TC * B^TB)^{-1}$, where $\odot$ is the Khatri-Rao product.
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — PARAFAC assumes a super-diagonal core $\mathcal{G} = \mathcal{I}^{R\times R\times R}$ (ones on the main diagonal, zeros elsewhere). This eliminates cross-talk between components. Tucker3 with a full core tensor $\mathcal{G}^{P\times Q\times R}$ allows all pairwise interactions.
- **B ✗** — This is reversed. PARAFAC is essentially unique (up to permutation and scaling of components) under mild conditions (Kruskal's condition). Tucker3 has rotational freedom — you can insert any invertible rotation $Q$ between $\mathcal{G}$ and $A$ without changing the model fit.
- **C ✓** — $\text{CORCONDIA} = 100 \cdot (1 - \|\mathcal{I} - \mathcal{G}\|_F^2 / \|\mathcal{I}\|_F^2)$. When $\mathcal{G} \approx \mathcal{I}$ (super-diagonal), CORCONDIA $\approx 100$, indicating the PARAFAC structure is appropriate for this $R$.
- **D ✓** — This is exactly the ALS update for $A$ — it unfolds $\mathcal{X}$ along mode 1 and multiplies by the pseudoinverse of $Z_A = (C \odot B)^T$.
- **E ✗** — A, C, D are correct.

---

**Question (18)** [Week 12]
The split-half Factor Match Score (FMS) for PARAFAC model selection with $R$ components is defined as:

$$\text{FMS} = \sum_{r=1}^R \frac{a_r^T\hat{a}_r}{\|a_r\|\|\hat{a}_r\|} \cdot \frac{b_r^T\hat{b}_r}{\|b_r\|\|\hat{b}_r\|} \cdot \frac{c_r^T\hat{c}_r}{\|c_r\|\|\hat{c}_r\|}$$

Which statements are correct?

A. If $\text{FMS} \approx R$, the model is stable — both data halves recover the same components.
B. FMS is bounded between 0 and $R$, where $R$ is the number of PARAFAC components.
C. A low FMS ($\text{FMS} \ll R$) suggests $R$ is too small and more components should be added.
D. Unlike PCA components, PARAFAC components are not nested — fitting $R=3$ gives different components than fitting $R=4$ and taking the first 3.
E. None of the above.

#### Answer: **A, B, D**

- **A ✓** — Each product is at most 1, so $\text{FMS} \leq R$. If all corresponding loading vectors are identical between halves (cosine $= 1$ for all $r$ and all modes), $\text{FMS} = R$ — perfect stability.
- **B ✓** — Each term $(a_r^T\hat{a}_r)/(\|a_r\|\|\hat{a}_r\|)$ is a cosine similarity in $[-1,1]$. After resolving sign ambiguity, each factor is in $[0,1]$, making each component's contribution in $[0,1]$, so $\text{FMS} \in [0, R]$.
- **C ✗** — Low FMS ($\text{FMS} \ll R$) suggests $R$ is too LARGE — the model is fitting noise and the two halves find different spurious components. More components would make instability worse, not better.
- **D ✓** — PARAFAC components are not nested — the $R=3$ solution is computed independently from the $R=4$ solution. Component 1 in $R=3$ will generally differ from component 1 in $R=4$. This contrasts with PCA, where PC$_1$ at $R=3$ is identical to PC$_1$ at $R=5$.
- **E ✗** — A, B, D are correct.

---

**Question (19)** [Week 2 / Week 5]
A researcher tunes a Ridge regression regularization parameter $\lambda$ using 10-fold cross-validation and then reports the minimum CV error as the model's expected generalization error. Which of the following are correct?

A. The reported minimum CV error is an optimistically biased estimate of the true generalization error.
B. Nested cross-validation (double-loop CV) should be used to obtain an unbiased estimate of generalization error after hyperparameter selection.
C. The 1-SE rule would select a smaller $\lambda$ than the minimum-CV-error rule, producing a more complex model.
D. Reporting the minimum CV error as generalization error is valid because the validation folds were held out during training.
E. None of the above.

#### Answer: **A, B**

- **A ✓** — When you search over many $\lambda$ values and report the minimum CV error, you have "spent" the independence of the validation folds by selecting the configuration that performed best on them. The minimum over a grid is optimistically biased relative to the true generalization error.
- **B ✓** — Nested CV (outer loop = assessment, inner loop = selection) separates the selection procedure from the evaluation. The outer test folds have never been touched by the selection step, giving an unbiased estimate of the full pipeline's generalization error.
- **C ✗** — The 1-SE rule selects the LARGEST $\lambda$ (most regularized, simplest model) whose CV error is within 1 SE of the minimum — NOT a smaller $\lambda$. A larger $\lambda$ means fewer effective parameters, not more.
- **D ✗** — The fact that validation folds were held out during model training is true, but the problem is that the same validation folds were used to SELECT $\lambda$. The minimum over a search is biased even when individual fold errors are unbiased.
- **E ✗** — A and B are correct.

---

**Question (20)** [Week 6 / Week 8]
Which of the following statements about CCA (Canonical Correlation Analysis) are correct?

A. CCA maximizes the correlation between linear combinations $Xu$ and $Yv$, and is therefore purely focused on cross-covariance, not the internal variance of $X$ or $Y$.
B. When $p \gg n$, the within-group covariance matrix $\Sigma_{XX}$ is singular, making standard CCA ill-posed. Regularized CCA or Sparse CCA addresses this by adding a ridge penalty or $L_1$ sparsity.
C. PLS and CCA have identical objectives: both maximize $\text{Cov}(Xu, Yv)$ subject to unit-variance constraints.
D. CCA produces at most $\min(p, q)$ canonical variate pairs, where $p = \dim(X)$ and $q = \dim(Y)$.
E. None of the above.

#### Answer: **A, B, D**

- **A ✓** — CCA maximizes $(u^T\Sigma_{XY}v)/\sqrt{u^T\Sigma_{XX}u \cdot v^T\Sigma_{YY}v}$ — a ratio that normalizes out internal variance. Unlike PLS which balances variance and correlation, CCA is purely a correlation maximization.
- **B ✓** — CCA requires inverting $\Sigma_{XX}$ and $\Sigma_{YY}$. When $p > n$, $\Sigma_{XX}$ is rank-deficient (singular). Regularized CCA adds $\lambda I$ to make the matrix invertible; Sparse CCA (PMD) applies $L_1$ penalties instead.
- **C ✗** — PLS maximizes $\text{Cov}(Xu, Yv) = \sqrt{\text{Var}(Xu) \cdot \text{Var}(Yv)} \cdot \text{Corr}(Xu, Yv)$ — it balances internal variance and cross-correlation. CCA maximizes only the correlation (ignoring variance). They have different objectives, different solutions, and produce different components.
- **D ✓** — CCA finds canonical variate pairs $(u_m, v_m)$. The maximum number of pairs is $\min(\text{rank}(X), \text{rank}(Y)) \leq \min(p,q)$.
- **E ✗** — A, B, D are correct.

---

## Open Questions

---

**Question (21)** [Week 9 / Week 11] — 20 points

A pharmaceutical company has measured the transcriptome (gene expression) of $n = 120$ cancer cell lines across $p = 8{,}000$ genes ($X \in \mathbb{R}^{120 \times 8000}$). They have no class labels. They want to discover latent biological structure in the data.

**(a)** [6 points] They first apply PCA. Describe what PCA computes mathematically (objective function, solution via SVD, and how variance explained by each component is calculated). Explain why the company should scale the data before applying PCA.

**(b)** [6 points] After PCA, they consider K-means clustering on the first 20 principal component scores. Describe the K-means algorithm (initialization, assignment step, update step, convergence criterion). Explain why clustering on PC scores rather than the raw 8,000-gene matrix is preferable.

**(c)** [4 points] To select $K$ (number of clusters), they compute the gap statistic. Write the formula for $G(K)$ and explain the selection rule. What is a fundamental warning that applies to any clustering method, including K-means?

**(d)** [4 points] A colleague suggests using NMF (Non-negative Matrix Factorization) instead of PCA, arguing it gives more interpretable components. State one mathematical constraint that NMF imposes that PCA does not, and explain why this constraint leads to a parts-based (additive) representation rather than a subtractive one.

### Solution

**Part (a) — PCA: Objective, SVD, Variance Explained, Scaling**

**Mathematical objective of PCA:**

PCA finds unit-norm loading vectors $v$ that maximize the variance of the projected scores:

$$\max_v \; \text{Var}(Xv) = v^T \Sigma v \quad \text{subject to } \|v\| = 1$$

where $\Sigma = \frac{1}{n-1} X^TX$ is the sample covariance matrix (assuming $X$ is mean-centered). Subsequent components maximize residual variance subject to orthogonality to all previous components.

**Solution via SVD:**

The data matrix $X \in \mathbb{R}^{n \times p}$ (mean-centered) is decomposed as $X = U D V^T$, where:
- $U \in \mathbb{R}^{n \times n}$: left singular vectors
- $D \in \mathbb{R}^{n \times p}$: diagonal matrix of singular values $d_1 \geq d_2 \geq \ldots \geq 0$
- $V \in \mathbb{R}^{p \times p}$: right singular vectors = loading vectors (principal axes)

PC scores matrix: $S = XV = UD$. Eigenvalues of the covariance matrix: $\lambda_k = d_k^2/(n-1)$.

**Variance explained by component $k$:**

$$\text{VE}_k = \frac{\lambda_k}{\sum_j \lambda_j} = \frac{d_k^2}{\sum_j d_j^2}$$

**Why scale before PCA:** Gene expression measurements across 8,000 genes span vastly different ranges. PCA on unscaled data is dominated by high-variance genes regardless of biological relevance. Scaling to unit variance (using the correlation matrix) ensures every gene contributes equally to the principal components. Without scaling, PCA finds directions of high raw magnitude, not high informational content.

---

**Part (b) — K-means Algorithm and Motivation for PC Scores**

**K-means algorithm:**

**Initialization:** Randomly assign each of the $n$ observations to one of $K$ clusters (or choose $K$ random observations as initial centroids).

**Assignment step:** Assign each observation $x_i$ to the nearest centroid by Euclidean distance:

$$C_k \leftarrow \{i : \|x_i - \mu_k\|^2 \leq \|x_i - \mu_j\|^2 \text{ for all } j \neq k\}$$

**Update step:** Recompute each centroid as the mean of assigned observations:

$$\mu_k \leftarrow \frac{1}{|C_k|} \sum_{i \in C_k} x_i$$

**Convergence:** Repeat until cluster assignments do not change. **Objective:** $\min \sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2$. The algorithm is guaranteed to converge but may reach a local minimum — multiple random restarts recommended.

**Why use PC scores rather than raw 8,000-gene matrix:**
1. **Noise reduction:** Most of the 8,000 dimensions contain noise; the first 20 PCs capture dominant variance structure.
2. **Curse of dimensionality:** In 8,000 dimensions, Euclidean distances between all pairs of points become nearly equal (concentration of measure), making cluster assignment meaningless.
3. **Computational efficiency:** K-means in 20 dimensions is far faster than in 8,000 dimensions.
4. **Correlation removal:** Raw features are highly correlated; PC scores are uncorrelated by construction, so distance in PC-score space more cleanly separates orthogonal sources of variation.

---

**Part (c) — Gap Statistic Formula and Warning**

**Gap statistic formula:**

$$G(K) = \log(U_k) - \log(W_k)$$

where:
- $W_k = \sum_\ell \frac{1}{2N_\ell} D_\ell$ is the within-cluster dissimilarity for the actual data
- $U_k$ = average of $\log(W_k)$ over $B=20$ Monte Carlo samples drawn uniformly over the data's bounding box

**Selection rule:**

$$K^* = \text{smallest } K \text{ such that } G(K) \geq G(K+1) - s'_{K+1}$$

where $s'_{K+1} = \text{std}(\log U_{K+1}) \times \sqrt{1 + 1/B}$. Choose the smallest $K$ where the gap is large relative to the next gap.

**Fundamental warning:** Clustering algorithms ALWAYS produce a grouping, even on completely random data with no cluster structure. K-means applied to uniform noise will produce $K$ compact-looking clusters. The gap statistic (and other methods) may still suggest $K^* > 1$ by chance. Cluster validity must always be confirmed using domain knowledge, external validation, or replicated biological experiments — not statistical heuristics alone.

---

**Part (d) — NMF vs PCA: Non-negativity and Parts-based Representation**

**Mathematical constraint NMF imposes that PCA does not:**

NMF constrains both factor matrices to be non-negative:

$$W \geq 0 \quad \text{and} \quad H \geq 0 \quad \text{(element-wise)}$$

PCA places no sign constraints on loading vectors — eigenvectors can have positive and negative entries, so PCA components can cancel each other.

**Why non-negativity leads to parts-based representation:**

Because $W \geq 0$ and $H \geq 0$, the reconstruction of any sample is:

$$\hat{x}_j \approx W h_j = \sum_k w_k h_{kj}$$

where every $w_k \geq 0$ and every $h_{kj} \geq 0$ — a purely additive combination. No component can cancel another.

In gene expression terms: each basis vector $w_k$ represents a "gene program" (a pattern of activated genes), and each sample is a non-negative weighted sum of programs. This mirrors biological reality. PCA, by contrast, allows negative loadings — a gene can subtract from a principal component, leading to abstract factors without direct biological meaning.

---

**Question (22)** [Week 7 / Week 8] — 20 points

Consider the Support Vector Machine (SVM) dual problem for a linearly separable two-class problem (classes labeled $+1$ and $-1$):

$$\max_{\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$$

$$\text{subject to} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

**(a)** [4 points] State the primal SVM optimization problem and explain the canonical scaling convention. What is the geometric interpretation of the margin width in terms of $\|\beta\|$?

**(b)** [6 points] Derive the dual problem from the primal using the Lagrangian $L_P = \frac{1}{2}\|\beta\|^2 - \sum_i \alpha_i[y_i(x_i^T\beta + \beta_0) - 1]$. Show the stationarity conditions $\partial L_P/\partial \beta = 0$ and $\partial L_P/\partial \beta_0 = 0$ and use them to obtain the dual objective.

**(c)** [6 points] Explain the KKT complementary slackness condition $\alpha_i[y_i(x_i^T\beta + \beta_0) - 1] = 0$ and use it to explain why the SVM achieves informational sparsity. What distinguishes a support vector from a "safe" point?

**(d)** [4 points] The data are not linearly separable in the original feature space $\mathbb{R}^d$. Explain the kernel trick: how does replacing $\langle x_i, x_j \rangle$ with $K(x_i, x_j)$ in the dual allow the SVM to find non-linear decision boundaries without explicitly mapping the data to a high-dimensional space? Give one example of a kernel that implicitly maps data to an infinite-dimensional space.

### Solution

**Part (a) — Primal Problem and Margin Geometry**

**Primal SVM optimization problem:**

$$\min_{\beta, \beta_0} \; \frac{1}{2}\|\beta\|^2 \quad \text{subject to} \quad y_i(x_i^T\beta + \beta_0) \geq 1 \quad \text{for all } i = 1, \ldots, N$$

**Canonical scaling convention:** We fix the scale of $(\beta, \beta_0)$ so that the constraint is tight for the nearest points (support vectors): $|x^T\beta + \beta_0| = 1$ for support vectors. Since the decision boundary is defined up to positive scaling, we can always choose the scale so margin-touching points satisfy this equality. The factor $1/2$ in the objective is for mathematical convenience.

**Geometric interpretation of margin width:** The signed distance from any point $x$ to the hyperplane $\{x : x^T\beta + \beta_0 = 0\}$ is $d(x) = (x^T\beta + \beta_0)/\|\beta\|$. For a positive support vector ($y_i = +1$): $d = 1/\|\beta\|$. For a negative support vector ($y_i = -1$): $d = -1/\|\beta\|$. The total margin width is:

$$\text{Margin} = \frac{2}{\|\beta\|}$$

Minimizing $\frac{1}{2}\|\beta\|^2$ is equivalent to maximizing the margin $2/\|\beta\|$.

---

**Part (b) — Lagrangian Derivation of the Dual**

**Lagrangian:** $L_P = \frac{1}{2}\|\beta\|^2 - \sum_i \alpha_i[y_i(x_i^T\beta + \beta_0) - 1]$, with $\alpha_i \geq 0$.

**Stationarity with respect to $\beta$:**

$$\frac{\partial L_P}{\partial \beta} = \beta - \sum_i \alpha_i y_i x_i = 0 \implies \beta = \sum_i \alpha_i y_i x_i$$

**Stationarity with respect to $\beta_0$:**

$$\frac{\partial L_P}{\partial \beta_0} = -\sum_i \alpha_i y_i = 0 \implies \sum_i \alpha_i y_i = 0$$

**Substituting $\beta = \sum_i \alpha_i y_i x_i$ into $L_P$:**

$$\frac{1}{2}\|\beta\|^2 = \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$$

$$-\sum_i \alpha_i y_i (x_i^T\beta) + \sum_i\alpha_i = -\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle + \sum_i \alpha_i$$

(The $\beta_0$ term vanishes because $\sum_i \alpha_i y_i = 0$.)

**Combining:**

$$L_D = \sum_i \alpha_i - \frac{1}{2} \sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$$

**Dual problem:** Maximize $L_D$ subject to $\alpha_i \geq 0$, $\sum_i \alpha_i y_i = 0$. This is a concave quadratic program. The training data appear only through pairwise dot products $\langle x_i, x_j \rangle$.

---

**Part (c) — KKT Complementary Slackness and Informational Sparsity**

**KKT complementary slackness condition:** At the optimum, for every training point $i$:

$$\alpha_i \cdot [y_i(x_i^T\beta + \beta_0) - 1] = 0$$

This holds at the saddle point of the Lagrangian. For each $i$, either $\alpha_i = 0$ OR the constraint is tight ($y_i(x_i^T\beta+\beta_0)=1$) — or both.

**Case 1 — Safe point:** $y_i(x_i^T\beta+\beta_0) > 1$ (strictly inside the margin). The bracket is $> 0$, so KKT forces $\alpha_i = 0$. This point contributes nothing to the model.

**Case 2 — Support vector:** $y_i(x_i^T\beta+\beta_0) = 1$. Then $\alpha_i \geq 0$ (can be nonzero). These are the only points that matter.

**Informational sparsity:** Since $\beta = \sum_i \alpha_i y_i x_i$ and $\alpha_i = 0$ for all safe points, only support vectors contribute:

$$\beta = \sum_{i \in \text{SV}} \alpha_i y_i x_i, \qquad \hat{y} = \text{sign}\!\left(\beta_0 + \sum_{i \in \text{SV}} \alpha_i y_i \langle x, x_i \rangle\right)$$

A model trained on $N=10{,}000$ points may have only 50 support vectors. The other 9,950 "safe" points could be deleted, and the decision boundary would be identical.

**Distinguishing support vectors from safe points:**
- **Support vector**: $\alpha_i > 0$; sits exactly on the margin; removal changes the boundary.
- **Safe point**: $\alpha_i = 0$; strictly inside the feasible region; removal does not change the boundary.

---

**Part (d) — Kernel Trick and Non-linear Boundaries**

**The kernel trick:** In the dual, training data appear only as pairwise dot products $\langle x_i, x_j \rangle$. To work in a higher-dimensional feature space defined by $\phi: \mathbb{R}^d \to \mathcal{H}$, we would need $\langle \phi(x_i), \phi(x_j) \rangle_\mathcal{H}$ — expensive or impossible if $\dim(\mathcal{H}) = \infty$.

Instead, replace $\langle x_i, x_j \rangle$ with a kernel function $K(x_i, x_j)$ that implicitly computes $\langle \phi(x_i), \phi(x_j) \rangle_\mathcal{H}$:

$$K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_\mathcal{H}$$

The algorithm never computes $\phi(x)$ explicitly — only evaluating $K(x_i, x_j)$ is required, costing $O(d)$ rather than $O(\dim \mathcal{H})$. This allows the SVM to find non-linear decision boundaries in $\mathbb{R}^d$ that correspond to linear boundaries in $\mathcal{H}$, at no additional computational cost beyond evaluating the kernel.

**Example — RBF (Gaussian) kernel:**

$$K(x, x') = \exp(-\gamma\|x - x'\|^2)$$

This kernel implicitly corresponds to a dot product in an **infinite-dimensional** feature space — the Taylor expansion of the exponential generates all polynomial orders simultaneously. With the RBF kernel, the SVM can represent decision boundaries of arbitrary smoothness at the cost of evaluating a simple exponential.
