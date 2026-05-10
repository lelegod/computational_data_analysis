# Practice Set 2 — CDA 02582

**Format:** 20 multiple-choice + 2 open questions
**Scoring:** MC: +1 (all correct answers selected, none wrong), −1 (any wrong answer selected), 0 (unanswered)
**Duration:** 4 hours

---

## Multiple Choice

**Question (1)** [Week 1]
Which of the following statements about the Expected Prediction Error (EPE) are correct?

A. The irreducible error $\sigma^2$ can be reduced by choosing a sufficiently complex model.
B. $\text{EPE} = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$, where the expectation is taken over both new observations $y$ and the training set $\mathcal{D}$.
C. As model complexity increases, training error decreases monotonically while test error forms a U-shape.
D. The variance component $E[(\hat{f} - E[\hat{f}])^2]$ measures how far the average prediction is from the true function.
E. None of the above.

---

**Question (2)** [Week 1]
For a linear smoother $\hat{Y} = SY$, the effective degrees of freedom is defined as $df(S) = \text{trace}(S)$. Which of the following are correct?

A. For OLS regression with $p$ predictors, $df = p$.
B. For Ridge regression, as $\lambda \to \infty$ the effective degrees of freedom approaches $p$.
C. For Ridge regression, as $\lambda \to 0$ the effective degrees of freedom approaches $p$.
D. AIC with $d = df(\lambda)$ for Ridge regression is asymptotically equivalent to leave-one-out cross-validation.
E. None of the above.

---

**Question (3)** [Week 2]
Consider the Benjamini-Hochberg (BH) procedure applied to $m = 6$ hypothesis tests with sorted p-values $p_{(1)} = 0.004$, $p_{(2)} = 0.012$, $p_{(3)} = 0.038$, $p_{(4)} = 0.080$, $p_{(5)} = 0.210$, $p_{(6)} = 0.430$, and a target FDR level $q = 0.10$. How many hypotheses are rejected?

A. 1
B. 2
C. 3
D. 4
E. None of the above.

---

**Question (4)** [Week 3]
Which of the following correctly describe the curse of dimensionality?

A. As dimension $D$ grows, a fixed number of training points $N$ becomes exponentially sparse.
B. In high dimensions, most data points tend to reside near the boundary (corners) of the sample space rather than the interior.
C. Euclidean distances become more meaningful as dimension grows, because there are more directions to distinguish points.
D. When $p > N$, OLS can be computed but has high variance because $(X^TX)$ is near-singular.
E. None of the above.

---

**Question (5)** [Week 4]
Regarding CART and impurity measures, which statements are correct?

A. The Gini index for a node is zero when the node is pure (contains only one class).
B. The misclassification rate is preferred over the Gini index for growing classification trees because it is easier to compute.
C. For a binary classification problem with class probability $p$, the Gini index equals $2p(1-p)$.
D. Cost-complexity pruning generates a sequence of nested subtrees by increasing the complexity penalty $\alpha$.
E. None of the above.

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

---

**Question (7)** [Week 6]
Which of the following statements about AdaBoost.M1 are correct?

A. The classifier weight $\alpha_m = \log[(1 - \text{err}_m)/\text{err}_m]$ is negative when $\text{err}_m > 0.5$, effectively negating the weak classifier.
B. At each iteration, the weights of misclassified observations are decreased so future classifiers focus on easy cases.
C. AdaBoost is equivalent to forward stagewise additive modelling with exponential loss $L(y, f) = \exp(-y \cdot f(x))$.
D. Boosting primarily reduces variance rather than bias, which is why it uses shallow trees (stumps).
E. None of the above.

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

---

**Question (9)** [Week 8]
A researcher applies Principal Component Regression (PCR) to a dataset where the response $y$ is weakly correlated with $X$ but some features in $X$ have very high variance. Which method would most likely outperform PCR on this dataset, and why?

A. Ridge regression, because it shrinks all coefficients uniformly toward zero.
B. Partial Least Squares (PLS), because it finds components that maximize covariance with $y$ rather than variance in $X$.
C. Lasso regression, because it performs automatic variable selection.
D. CCA (Canonical Correlation Analysis), because it maximizes the correlation between X-scores and Y-scores, ignoring internal variance.
E. None of the above.

---

**Question (10)** [Week 8]
Regarding the PLS algorithm, which of the following are correct?

A. The PLS weight for feature $j$ in component $m$ is $\hat{\phi}_{mj} = x_j^{(m-1)T} y$, assigning higher weight to features correlated with $y$.
B. PLS latent components $z_1, z_2, \ldots, z_m$ are mutually uncorrelated (orthogonal).
C. If $M = p$ (all components are kept), PLS predictions are identical to OLS predictions.
D. Canonical Correlation Analysis and PLS have identical objectives — both maximize covariance between X-scores and Y-scores.
E. None of the above.

---

**Question (11)** [Week 9]
Consider running K-means clustering on a dataset with well-separated elliptical clusters of different sizes. Which statements are correct?

A. K-means will always fail on elliptical clusters because it minimizes squared Euclidean distance to spherical centroids.
B. K-medoids is more robust to outliers than K-means because cluster centers are constrained to be actual data points.
C. Gaussian Mixture Models with full per-cluster covariance matrices can better capture elliptical cluster shapes than K-means.
D. The gap statistic selects $K$ by comparing the log within-cluster dissimilarity of the data to that of uniformly distributed data.
E. None of the above.

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

---

**Question (14)** [Week 10]
A neural network has the architecture: 5 inputs → 8 hidden units (sigmoid) → 4 hidden units (sigmoid) → 2 output units (linear), with biases at every layer. What is the total number of trainable parameters?

A. 78
B. 82
C. 86
D. 94
E. None of the above.

---

**Question (15)** [Week 11]
Regarding Non-negative Matrix Factorization (NMF) with $X \approx WH$, which statements are correct?

A. NMF enforces non-negativity on both $W$ and $H$, leading to parts-based, additive representations.
B. Unlike PCA, the NMF objective is jointly convex in $(W, H)$, guaranteeing convergence to the global optimum.
C. The Lee & Seung multiplicative update rule $H_{kj} \leftarrow H_{kj} \cdot (W^TX)_{kj} / (W^TWH)_{kj}$ guarantees non-negativity throughout optimization if initialized positively.
D. NMF solutions are generally not unique — for any invertible $Q$ with $WQ^{-1} \geq 0$ and $QH \geq 0$, $(WQ^{-1})(QH)$ is an equally valid factorization.
E. None of the above.

---

**Question (16)** [Week 11]
Which of the following statements correctly distinguish Independent Component Analysis (ICA) from Principal Component Analysis (PCA)?

A. PCA finds directions of maximum variance; ICA finds directions of maximum statistical independence by maximizing non-Gaussianity.
B. ICA can separate statistically independent Gaussian sources, while PCA cannot.
C. After whitening (sphering), the ICA problem reduces to finding an orthogonal rotation matrix.
D. The Cocktail Party Problem (blind source separation) can be solved by PCA alone, since PCA decorrelates the signals.
E. None of the above.

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

---

**Question (19)** [Week 2 / Week 5]
A researcher tunes a Ridge regression regularization parameter $\lambda$ using 10-fold cross-validation and then reports the minimum CV error as the model's expected generalization error. Which of the following are correct?

A. The reported minimum CV error is an optimistically biased estimate of the true generalization error.
B. Nested cross-validation (double-loop CV) should be used to obtain an unbiased estimate of generalization error after hyperparameter selection.
C. The 1-SE rule would select a smaller $\lambda$ than the minimum-CV-error rule, producing a more complex model.
D. Reporting the minimum CV error as generalization error is valid because the validation folds were held out during training.
E. None of the above.

---

**Question (20)** [Week 6 / Week 8]
Which of the following statements about CCA (Canonical Correlation Analysis) are correct?

A. CCA maximizes the correlation between linear combinations $Xu$ and $Yv$, and is therefore purely focused on cross-covariance, not the internal variance of $X$ or $Y$.
B. When $p \gg n$, the within-group covariance matrix $\Sigma_{XX}$ is singular, making standard CCA ill-posed. Regularized CCA or Sparse CCA addresses this by adding a ridge penalty or $L_1$ sparsity.
C. PLS and CCA have identical objectives: both maximize $\text{Cov}(Xu, Yv)$ subject to unit-variance constraints.
D. CCA produces at most $\min(p, q)$ canonical variate pairs, where $p = \dim(X)$ and $q = \dim(Y)$.
E. None of the above.

---

## Open Questions

**Question (21)** [Week 9 / Week 11] — 10 points

A pharmaceutical company has measured the transcriptome (gene expression) of $n = 120$ cancer cell lines across $p = 8{,}000$ genes ($X \in \mathbb{R}^{120 \times 8000}$). They have no class labels. They want to discover latent biological structure in the data.

**(a)** [3 points] They first apply PCA. Describe what PCA computes mathematically (objective function, solution via SVD, and how variance explained by each component is calculated). Explain why the company should scale the data before applying PCA.

**(b)** [3 points] After PCA, they consider K-means clustering on the first 20 principal component scores. Describe the K-means algorithm (initialization, assignment step, update step, convergence criterion). Explain why clustering on PC scores rather than the raw 8,000-gene matrix is preferable.

**(c)** [2 points] To select $K$ (number of clusters), they compute the gap statistic. Write the formula for $G(K)$ and explain the selection rule. What is a fundamental warning that applies to any clustering method, including K-means?

**(d)** [2 points] A colleague suggests using NMF (Non-negative Matrix Factorization) instead of PCA, arguing it gives more interpretable components. State one mathematical constraint that NMF imposes that PCA does not, and explain why this constraint leads to a parts-based (additive) representation rather than a subtractive one.

---

**Question (22)** [Week 7 / Week 8] — 10 points

Consider the Support Vector Machine (SVM) dual problem for a linearly separable two-class problem (classes labeled $+1$ and $-1$):

$$\max_{\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$$

$$\text{subject to} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

**(a)** [2 points] State the primal SVM optimization problem and explain the canonical scaling convention. What is the geometric interpretation of the margin width in terms of $\|\beta\|$?

**(b)** [3 points] Derive the dual problem from the primal using the Lagrangian $L_P = \frac{1}{2}\|\beta\|^2 - \sum_i \alpha_i[y_i(x_i^T\beta + \beta_0) - 1]$. Show the stationarity conditions $\partial L_P/\partial \beta = 0$ and $\partial L_P/\partial \beta_0 = 0$ and use them to obtain the dual objective.

**(c)** [3 points] Explain the KKT complementary slackness condition $\alpha_i[y_i(x_i^T\beta + \beta_0) - 1] = 0$ and use it to explain why the SVM achieves informational sparsity. What distinguishes a support vector from a "safe" point?

**(d)** [2 points] The data are not linearly separable in the original feature space $\mathbb{R}^d$. Explain the kernel trick: how does replacing $\langle x_i, x_j \rangle$ with $K(x_i, x_j)$ in the dual allow the SVM to find non-linear decision boundaries without explicitly mapping the data to a high-dimensional space? Give one example of a kernel that implicitly maps data to an infinite-dimensional space.
