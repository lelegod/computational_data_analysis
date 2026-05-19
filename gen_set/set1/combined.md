# Practice Set 1 — CDA 02582 (Questions + Solutions)

**Format:** 20 multiple-choice + 2 open questions  
**Scoring:** MC: +1 (correct), −0.25 (wrong), 0 (unanswered)  
**Duration:** 4 hours

---

## Multiple Choice

---

**Question (1)** [Week 1]  
The Expected Prediction Error (EPE) at a new point $x_0$ can be written as:

$$\text{EPE} = E(y - \hat{f})^2 = \sigma^2 + \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f})$$

Which of the following statements about this decomposition are correct?

A. Increasing model complexity tends to decrease bias and increase variance simultaneously.  
B. The irreducible noise term $\sigma^2$ can be reduced by choosing a sufficiently flexible model.  
C. The variance component measures how much the fitted model fluctuates across different training datasets.  
D. For Ordinary Least Squares (OLS) with all p predictors included, the bias is exactly zero.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — Increasing complexity lowers bias (model can fit more patterns) and raises variance (model changes more with different training data) — this is the fundamental bias-variance tradeoff.
- **B ✗** — $\sigma^2$ is irreducible noise from the data-generating process; no model can eliminate it.
- **C ✓** — Variance is formally $E[(\hat{f} - E[\hat{f}])^2]$, measuring fluctuation of predictions across training sets.
- **D ✓** — OLS is unbiased: $E[\hat{\beta}_\text{OLS}] = \beta$, so $\text{Bias} = E[\hat{f}] - f = 0$ under correct model specification.
- **E ✗** — A, C, D are all correct.

---

**Question (2)** [Week 1]  
Which of the following correctly describes the Ridge regression estimator and its key properties?

A. The ridge estimator is $\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1} X^Ty$, which is always invertible for $\lambda > 0$.  
B. Ridge regression shrinks some coefficients to exactly zero, performing automatic variable selection.  
C. As $\lambda \to \infty$, the effective degrees of freedom $df(\lambda) = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$ decreases toward zero.  
D. Ridge regression is unbiased for any value of $\lambda > 0$.  
E. None of the above.

#### Answer: **A, C**

- **A ✓** — For any $\lambda > 0$, $X^TX + \lambda I$ is strictly positive definite → always invertible.
- **B ✗** — Ridge shrinks coefficients toward zero but never exactly to zero; exact zeros are Lasso's property ($L_1$ geometry has corners; $L_2$ sphere does not).
- **C ✓** — As $\lambda \to 0$: $df \to p$. As $\lambda \to \infty$: $df \to 0$. The trace decreases monotonically.
- **D ✗** — Ridge introduces bias for any $\lambda > 0$: $E[\hat{\beta}_\text{ridge}] \neq \beta$ in general.

---

**Question (3)** [Week 1]  
Consider the model selection criteria Cp, AIC, and BIC. Which of the following statements are correct?

A. For Gaussian errors, Cp and AIC are equivalent criteria.  
B. BIC uses a penalty of $\log(N)$ per parameter, whereas AIC uses a fixed penalty of 2 per parameter.  
C. AIC is asymptotically equivalent to leave-one-out cross-validation.  
D. For large N, BIC tends to select more complex models than AIC because its penalty grows with N.  
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — For Gaussian errors, $\text{AIC}(\lambda) = \overline{err}(\lambda) + 2(d(\lambda)/N)\hat{\sigma}_e^2$, which matches the Cp formula exactly.
- **B ✓** — BIC penalty = $\log(N) \cdot d/N$; AIC penalty = $2d/N$ — the BIC coefficient $\log(N)$ grows with N while AIC's coefficient 2 is fixed.
- **C ✓** — Stone (1977) showed AIC is asymptotically equivalent to leave-one-out cross-validation.
- **D ✗** — For large N, BIC penalizes MORE per parameter than AIC ($\log(N) > 2$ for $N \geq 8$), so BIC selects SIMPLER models than AIC, not more complex ones.

---

**Question (4)** [Week 2]  
Which of the following statements about the Lasso are correct?

A. The Lasso objective is $\min_\beta (Y - X\beta)^T(Y - X\beta) + \lambda\|\beta\|_1$, where $\|\beta\|_1 = \sum|\beta_j|$.  
B. The Lasso has a closed-form solution analogous to the Ridge estimator.  
C. The geometry of the $L_1$ constraint region (a diamond in 2D) explains why Lasso solutions are often sparse.  
D. In the $p > n$ setting, Lasso can select at most $n$ non-zero coefficients.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — Standard Lasso objective with $L_1$ norm $\|\beta\|_1 = \sum|\beta_j|$.
- **B ✗** — The $L_1$ norm is not differentiable at $\beta = 0$, so no closed-form solution exists; LARS or coordinate descent must be used.
- **C ✓** — The $L_1$ constraint region is a diamond in 2D with corners on coordinate axes; the RSS ellipsoid typically first contacts the diamond at a corner where one coordinate is zero → sparse solution.
- **D ✓** — Lasso selects at most $\min(n, p)$ variables; when $p > n$, at most $n$ variables can be non-zero.

---

**Question (5)** [Week 2]  
A researcher runs 50 independent hypothesis tests at individual significance level $\alpha = 0.05$. They apply the Bonferroni correction. Which of the following are correct?

A. The Bonferroni-corrected threshold for each individual test is $p < 0.001$.  
B. Without any correction, the Family-Wise Error Rate (FWER) is approximately $1 - (0.95)^{50} \approx 0.923$.  
C. The Bonferroni correction controls the FWER at level $\alpha = 0.05$ across all 50 tests.  
D. The Bonferroni correction has higher statistical power than the Benjamini-Hochberg procedure at the same overall $\alpha$ level.  
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Bonferroni threshold $= \alpha/M = 0.05/50 = 0.001$.
- **B ✓** — FWER without correction $= 1 - (1-0.05)^{50} = 1 - (0.95)^{50} \approx 0.923$.
- **C ✓** — Bonferroni controls FWER at level $\alpha$, ensuring $P(\text{any false rejection}) \leq 0.05$.
- **D ✗** — Bonferroni has LOWER power than BH. BH allows a controlled proportion of false discoveries and therefore rejects more hypotheses.

---

**Question (6)** [Week 2]  
In nested cross-validation (double-loop CV), which of the following statements are correct?

A. The outer loop is used for model selection (tuning hyperparameters), and the inner loop is used for model assessment.  
B. Nested CV audits the entire modelling pipeline including the hyperparameter selection step.  
C. A large gap between inner-loop error and outer-loop error suggests selection-induced overfitting.  
D. Nested CV is unnecessary when AIC or BIC is used for model selection.  
E. None of the above.

#### Answer: **B, C**

- **A ✗** — Reversed. The INNER loop handles model selection (hyperparameter tuning) and the OUTER loop handles model assessment (estimating generalization error).
- **B ✓** — Nested CV audits the full pipeline — including the selection step — providing an unbiased estimate of how well the "select-then-train" procedure generalizes.
- **C ✓** — A large gap between inner-loop (optimistic, selected) error and outer-loop (honest) error signals overfitting to hyperparameter selection noise.
- **D ✗** — AIC/BIC are not a substitute for nested CV; they also suffer from selection-induced bias if chosen from a set of models.

---

**Question (7)** [Week 3]  
Regarding the curse of dimensionality and regularization methods, which of the following are correct?

A. As the number of dimensions D increases, a fixed number of training points N becomes exponentially sparse in the feature space.  
B. In the elastic net, setting $\alpha = 1$ gives pure Ridge regression.  
C. The elastic net penalty combines an $L_1$ term and an $L_2$ term, allowing both variable selection and grouping of correlated predictors.  
D. Donoho (2000) identified that high-dimensional data often lies on a low-dimensional manifold as a "blessing" of dimensionality.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — The core statement of the curse of dimensionality: volume grows exponentially, fixed N becomes sparse.
- **B ✗** — $\alpha = 0$ gives pure Ridge; $\alpha = 1$ gives pure Lasso (not the other way around).
- **C ✓** — Elastic net penalty $\lambda[(1/2)(1-\alpha)\|\beta\|_2^2 + \alpha\|\beta\|_1]$ combines $L_2$ (grouping) with $L_1$ (sparsity).
- **D ✓** — Donoho (2000) listed the manifold hypothesis as one of the three "blessings" of dimensionality.

---

**Question (8)** [Week 4]  
A classification tree is being grown on training data. Which of the following statements about splitting criteria are correct?

A. The Gini index for a node is defined as $G = \sum_k p_{mk}(1 - p_{mk})$, and equals zero when the node is completely pure.  
B. The misclassification rate is the preferred criterion for growing classification trees because it is differentiable.  
C. Cross-entropy (deviance) and the Gini index are both more sensitive to changes in class probabilities than the misclassification rate.  
D. In CART, a regression tree prediction in each leaf region is the mean of the training responses in that region.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — Gini index $G = \sum_k p_{mk}(1-p_{mk}) = 0$ when all observations belong to one class.
- **B ✗** — Misclassification rate is NOT differentiable and NOT preferred for growing trees — it is insensitive to probability shifts within the majority class. Gini and cross-entropy are preferred.
- **C ✓** — Both Gini and cross-entropy respond to any shift in class probabilities; misclassification rate does not change as long as the majority class is unchanged.
- **D ✓** — In regression trees, the prediction in region $R_j$ is $\hat{c}_j = \text{mean}(y_i : x_i \in R_j)$.

---

**Question (9)** [Week 4 / Week 5]  
Cost-complexity pruning is applied to a fully grown CART regression tree. Which of the following are correct?

A. The cost-complexity criterion is $C_\alpha(T) = R(T) + \alpha|T|$, where $|T|$ is the number of terminal nodes.  
B. When $\alpha = 0$, the pruned tree is the root node (single-leaf tree) since no penalty is applied.  
C. As $\alpha$ increases, the selected subtree becomes smaller (fewer leaves).  
D. The pruning parameter $\alpha$ is typically chosen by minimizing cross-validation error over a sequence of candidate values.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — $C_\alpha(T) = R(T) + \alpha|T|$ where $R(T)$ = total node impurity, $|T|$ = number of terminal nodes.
- **B ✗** — When $\alpha = 0$, the FULL (unpruned) tree $T_0$ is selected (no penalty for complexity). The root is selected when $\alpha$ is very large.
- **C ✓** — Increasing $\alpha$ imposes a larger per-leaf penalty → fewer leaves → smaller trees.
- **D ✓** — Standard CART: grow $T_0$, find the sequence of subtrees via weakest-link pruning, use K-fold CV to select $\alpha^*$.

---

**Question (10)** [Week 5]  
Bagging (Bootstrap Aggregating) is applied to deep, unpruned CART regression trees. The variance of the bagged predictor is given by:

$$\text{Var}(\hat{y}_\text{bag}) = \rho\sigma^2 + \frac{1-\rho}{B} \cdot \sigma^2$$

where $\rho$ is the pairwise correlation between trees, $\sigma^2$ is the variance of a single tree, and $B$ is the number of bootstrap samples. Which of the following are correct?

A. As $B \to \infty$, the bagged variance approaches $\rho\sigma^2$, which is the irreducible floor determined by inter-tree correlation.  
B. Bagging reduces both the bias and the variance of individual trees.  
C. Each bootstrap sample of size $N$ contains on average approximately 63.2% of the unique training observations.  
D. Out-of-bag (OOB) error estimation is a free by-product of bagging that approximates leave-one-out cross-validation error.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — As $B \to \infty$, $(1-\rho)\sigma^2/B \to 0$, leaving $\rho\sigma^2$ as the irreducible floor.
- **B ✗** — Bagging does NOT reduce bias. Bias of the bagged predictor = bias of any single tree: $E[(1/B)\sum(\hat{y}_b - y)] = E[\hat{y}_b - y]$.
- **C ✓** — $P(\text{obs } i \text{ not in bootstrap}) = (1-1/N)^N \to 1/e \approx 0.368$, so ~63.2% are included.
- **D ✓** — For each training observation, predictions from trees where it was OOB give a free CV-like error estimate.

---

**Question (11)** [Week 6]  
Random Forests extend bagging by adding random feature subsampling at each split. Which of the following are correct?

A. The default heuristic for the number of candidate features per split in classification problems is $m = \lfloor\sqrt{p}\rfloor$.  
B. When $m = p$ (all features considered at each split), Random Forest reduces to standard bagging.  
C. Random Forests reduce variance compared to bagging by decorrelating the trees, which lowers the inter-tree correlation $\rho$.  
D. In gradient boosting, shallow trees (stumps) are preferred as base learners, while in Random Forests, deep trees are preferred.  
E. None of the above.

#### Answer: **A, B, C, D**

- **A ✓** — Default for classification: $m = \lfloor\sqrt{p}\rfloor$ (lecture stated default).
- **B ✓** — When all $p$ features are considered at every split, no random subsampling → reduces to standard bagging.
- **C ✓** — Random feature subsampling prevents trees from always splitting on the same dominant variable → lowers pairwise correlation $\rho$ → reduces the variance floor $\rho\sigma^2$.
- **D ✓** — RF: deep trees (low bias, high variance — bagging reduces variance). Gradient Boosting: stumps (high bias, low variance — boosting corrects bias sequentially).

---

**Question (12)** [Week 6]  
AdaBoost.M1 is applied to a binary classification problem with labels $y_i \in \{-1, +1\}$. The classifier weight at step $m$ is:

$$\alpha_m = \log\left[\frac{1 - \text{err}_m}{\text{err}_m}\right]$$

Which of the following statements are correct?

A. If $\text{err}_m = 0.5$ (random classifier), then $\alpha_m = 0$, meaning the $m$-th weak learner contributes nothing to the final vote.  
B. Boosting reduces bias (unlike bagging, which only reduces variance), which is why boosting uses shallow trees (stumps) as weak learners.  
C. The exponential loss used by AdaBoost is more robust to label noise than the binomial deviance loss because it penalizes misclassified observations less.  
D. In forward stagewise additive modelling, previously fitted trees are updated (their weights are adjusted) as each new tree is added.  
E. None of the above.

#### Answer: **A, B**

- **A ✓** — $\text{err}_m = 0.5$: $\alpha_m = \log[(1-0.5)/0.5] = \log(1) = 0$ → the weak learner is ignored.
- **B ✓** — Boosting corrects errors sequentially (reducing bias); using high-bias weak learners (stumps) ensures each step corrects a specific weakness without being already complex.
- **C ✗** — The exponential loss grows FASTER than binomial deviance for misclassified observations (negative margin) → AdaBoost is MORE sensitive to label noise, not less robust.
- **D ✗** — Forward stagewise additive modelling fixes previously fitted trees — once added, their coefficients are never adjusted; only new $(\beta_m, b_m)$ pairs are added.

---

**Question (13)** [Week 7]  
In the Support Vector Machine (SVM) with canonical scaling, the primal optimization problem is:

$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{subject to } y_i(x_i^T\beta + \beta_0) \geq 1 \text{ for all } i$$

Which of the following statements are correct?

A. The margin width in the canonical SVM is $C = 1/\|\beta\|$, so minimizing $\|\beta\|^2$ is equivalent to maximizing the margin.  
B. Non-support vectors (points far from the margin) have Lagrange multipliers $\alpha_i > 0$, and support vectors have $\alpha_i = 0$.  
C. The RBF (Gaussian) kernel $K(x, x') = \exp(-\gamma\|x - x'\|^2)$ mathematically corresponds to a dot product in an infinite-dimensional feature space.  
D. The SVM dual formulation expresses the problem purely in terms of inner products $\langle x_i, x_j \rangle$, enabling the kernel trick.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — Margin width $= 1/\|\beta\|$; minimizing $(1/2)\|\beta\|^2$ maximizes the margin.
- **B ✗** — Reversed. KKT: non-support vectors (bracket $> 0$) have $\alpha_i = 0$; support vectors (bracket $= 0$) have $\alpha_i > 0$.
- **C ✓** — The RBF kernel corresponds to a dot product in an infinite-dimensional RKHS.
- **D ✓** — Dual: $\max_\alpha \sum\alpha_i - (1/2)\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$ — data appear only as inner products.

---

**Question (14)** [Week 8]  
Principal Component Analysis (PCA) is applied to a centered data matrix $X \in \mathbb{R}^{100 \times 5}$. The singular value decomposition gives $X = UDV^T$. The singular values are $d_1 = 8, d_2 = 6, d_3 = 4, d_4 = 2, d_5 = 1$. Which of the following are correct?

A. The fraction of total variance explained by the first two principal components is $(64 + 36) / (64 + 36 + 16 + 4 + 1) = 100/121 \approx 82.6\%$.  
B. The loading vectors (principal axes) $V$ are the right singular vectors of $X$, and are identical to the eigenvectors of the covariance matrix $X^TX/(n-1)$.  
C. PCA applied to unscaled data (without standardizing features) may be dominated by high-variance features measured in large units.  
D. Partial Least Squares (PLS) differs from PCA in that PLS ignores the response variable $y$ and maximizes only the variance of $Xv$.  
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Squared singular values: $64, 36, 16, 4, 1$; total $= 121$; first two explain $100/121 \approx 82.6\%$.
- **B ✓** — Right singular vectors $V$ from $X = UDV^T$ are eigenvectors of $X^TX/(n-1)$ (sample covariance matrix).
- **C ✓** — PCA on unscaled data is dominated by features with large variance; standardizing ensures equal contribution.
- **D ✗** — Reversed. PLS uses the response $y$ to guide dimension reduction; PCA maximizes variance in $X$ alone (unsupervised).

---

**Question (15)** [Week 9]  
K-means clustering is applied to a dataset. Which of the following statements are correct?

A. K-means minimizes the objective $\sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2$, where $\mu_k$ is the centroid of cluster $k$.  
B. K-means is guaranteed to converge to the global optimum regardless of initialization.  
C. The silhouette coefficient $s(i) = (b(i) - a(i)) / \max\{a(i), b(i)\}$ takes values in $[-1, 1]$, with values near $+1$ indicating well-clustered points.  
D. The gap statistic selects the number of clusters $K$ by comparing the within-cluster dispersion of the data to that expected under a uniform reference distribution.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — Exact K-means within-cluster sum of squares objective.
- **B ✗** — K-means is NOT guaranteed to find the global optimum — it converges to a local minimum depending on initialization. Multiple restarts are recommended.
- **C ✓** — $s(i) \in [-1, 1]$; near $+1$ means $a(i) \ll b(i)$ (much closer to own cluster than to next nearest).
- **D ✓** — Gap statistic $G(K) = \log(U_k) - \log(W_k)$ compares actual within-cluster dispersion to that from simulated uniform reference data.

---

**Question (16)** [Week 9]  
The EM algorithm is used to fit a Gaussian Mixture Model (GMM) with $K$ components. Which of the following are correct?

A. In the E-step, the posterior probability $\gamma_{ij} = P(Z_i = j \mid x_i)$ is computed using Bayes' rule: $\gamma_{ij} = \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j) / \sum_{j'} \pi_{j'} \mathcal{N}(x_i; \mu_{j'}, \Sigma_{j'})$.  
B. In the M-step, the mean update is $\mu_j = \sum_i \gamma_{ij} x_i / \sum_i \gamma_{ij}$, a weighted average of data points with soft-assignment weights.  
C. K-means is a special case of GMM with equal, spherical covariances and hard (binary) assignments.  
D. GMM with full per-component covariance matrices always produces a unique global maximum of the likelihood.  
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Correct E-step formula: Bayes' rule combining prior $\pi_j$ with Gaussian likelihood.
- **B ✓** — M-step mean update is a weighted average with soft assignment weights $\gamma_{ij}$.
- **C ✓** — K-means = GMM with $\Sigma_j = \varepsilon^2 I \to 0$ (identical spherical covariances) and hard assignments ($\gamma_{ij} \to 0$ or $1$).
- **D ✗** — GMM likelihood is non-concave — EM finds a local maximum. Multiple restarts are needed.

---

**Question (17)** [Week 10]  
Consider a fully connected feedforward neural network: Input layer: 5 nodes; Hidden layer 1: 3 nodes (ReLU); Hidden layer 2: 3 nodes (ReLU); Output layer: 2 nodes (softmax). Each layer includes a bias term. How many scalar parameters does this network have, and which additional statements are correct?

A. The total number of parameters is $(5\times3 + 3) + (3\times3 + 3) + (3\times2 + 2) = 18 + 12 + 8 = 38$.  
B. Binary cross-entropy loss is derived from maximizing the Bernoulli likelihood: $L = -\sum_i[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$.  
C. In backpropagation, the error signal $\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$ propagates blame backwards through the network.  
D. Recurrent Neural Networks (RNNs) suffer from the vanishing gradient problem for long sequences, motivating LSTM and GRU architectures.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — $(5\times3+3) + (3\times3+3) + (3\times2+2) = 18 + 12 + 8 = 38$ parameters.
- **B** — Binary cross-entropy is correct for 2-class problems, derived from Bernoulli negative log-likelihood (correct formula). For the 2-output softmax here this corresponds to categorical CE. Mark as ✓ in spirit, but the network has 2 softmax outputs (multi-class setup), so BCE in the form shown applies to single-output binary problems.
- **C ✓** — Correct vectorized backpropagation: transpose weight matrix × upstream error, element-wise multiply with local activation derivative.
- **D ✓** — RNNs propagate error back through time; products of many Jacobians cause vanishing gradients for long sequences → LSTM and GRU use gating mechanisms.

---

**Question (18)** [Week 11]  
Non-negative Matrix Factorization (NMF) and Independent Component Analysis (ICA) are both unsupervised decomposition methods. Which of the following statements are correct?

A. NMF enforces non-negativity on both factor matrices $W$ and $H$, producing an additive, parts-based representation with no cancellation between components.  
B. ICA requires that the source components are statistically independent and non-Gaussian; it cannot separate Gaussian sources.  
C. NMF solutions are unique — there is only one valid factorization $X \approx WH$ for given $W \geq 0$, $H \geq 0$.  
D. ICA preprocessing involves centering and whitening the data so that subsequent optimization needs only to find an orthogonal rotation matrix.  
E. None of the above.

#### Answer: **A, B, D**

- **A ✓** — NMF forces $W \geq 0$ and $H \geq 0$ → purely additive combination, no cancellation → parts-based representation.
- **B ✓** — ICA requires non-Gaussian sources because the CLT says mixtures become more Gaussian; for Gaussian sources the mixing matrix is unidentifiable.
- **C ✗** — NMF solutions are NOT unique. For any invertible $Q$ with $WQ^{-1} \geq 0$ and $QH \geq 0$, $(WQ^{-1})(QH)$ is an equally valid factorization.
- **D ✓** — Whitening (sphering) reduces the ICA problem from finding an arbitrary matrix $W$ to finding an orthogonal rotation, which is far simpler.

---

**Question (19)** [Week 11]  
Archetypal Analysis (AA) approximates each data point as a convex mixture of $K$ archetypes, and each archetype is itself a convex combination of data points. The objective is:

$$\min_{S,H} \|X - XSH\|_F^2$$

Which of the following are correct?

A. Archetypes in AA are located on (or near) the convex hull of the data, representing extreme prototypes rather than average profiles.  
B. In Sparse Coding, the dictionary $W$ is overcomplete ($K > I$, more atoms than dimensions), and each data point is represented using a sparse coefficient vector $h$ with most entries equal to zero.  
C. The matrix $S$ in AA has columns that sum to 1 with non-negative entries, forcing each archetype to be a convex combination of real data points.  
D. AA and K-means find the same solution when the number of components is the same, since both represent data using a fixed number of prototypes.  
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Archetypes lie on or near the convex hull — they are extreme points, not averages. This distinguishes AA from K-means (centroids = interior) and PCA (variance directions).
- **B ✓** — Sparse coding uses an overcomplete dictionary $W$ with $K > I$ atoms, and represents each data point as $Wh$ where $h$ is sparse. This is the Lasso problem in the coding step.
- **C ✓** — $S$ has columns summing to 1 with $s_{ij} \geq 0$ → each archetype $Z = XS$ is a convex combination of real data points → archetypes cannot lie outside the data cloud.
- **D ✗** — K-means places centroids at interior cluster means; AA places archetypes on the convex hull (extreme boundary). Their solutions generally differ significantly.

---

**Question (20)** [Week 12]  
A 3-way tensor $\mathcal{X}$ of shape $I \times J \times K$ is decomposed using PARAFAC with $R$ components:

$$\mathcal{X} \approx \sum_r a_r \circ b_r \circ c_r$$

and separately using Tucker3 with ranks $(P, Q, R)$:

$$\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$

Which of the following are correct?

A. PARAFAC is a special case of Tucker3 where the core tensor $\mathcal{G}$ is super-diagonal (identity-like).  
B. Tucker3 solutions are essentially unique (up to sign and permutation), whereas PARAFAC solutions are not unique due to rotational freedom.  
C. CORCONDIA (Core Consistency Diagnostic) close to 100 indicates that the PARAFAC model has an appropriate number of components $R$, because the fitted core tensor is close to super-diagonal.  
D. In the Tucker3 matrix representation, $X_{(1)} \approx A G_{(1)} (C \otimes B)^T$ uses the Kronecker product, whereas PARAFAC uses the Khatri-Rao product.  
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — PARAFAC = Tucker3 with $\mathcal{G} = \mathcal{I}^{R\times R\times R}$ (super-diagonal), eliminating all cross-component interactions.
- **B ✗** — Reversed. PARAFAC IS essentially unique (up to sign/permutation, Kruskal's conditions). Tucker3 is NOT unique due to rotational freedom.
- **C ✓** — CORCONDIA $= 100(1 - \|\mathcal{I} - \mathcal{G}\|_F^2/\|\mathcal{I}\|_F^2)$; close to 100 means fitted core $\approx$ super-diagonal → PARAFAC structure is appropriate.
- **D ✓** — Tucker3 unfolded uses Kronecker product $\otimes$ (all outer products); PARAFAC unfolded uses Khatri-Rao product $\odot$ (column-wise Kronecker — only matching columns).

---

## Open Questions

---

**Question (21)** [Weeks 1–3] — 20 points

Ridge regression and the Lasso are both regularization methods for linear regression, but they have fundamentally different properties.

**(a)** [5 pts] Derive the closed-form solution for Ridge regression starting from the penalized least squares objective:

$$\min_\beta (Y - X\beta)^T(Y - X\beta) + \lambda\beta^T\beta$$

Show all steps including the derivative, setting it to zero, and solving for $\hat{\beta}_\text{ridge}$.

**(b)** [5 pts] Explain geometrically why the Lasso ($L_1$ penalty) produces sparse solutions (exact zeros) while Ridge ($L_2$ penalty) does not. Refer to the shape of the constraint regions and the RSS contours.

**(c)** [5 pts] The effective degrees of freedom for a ridge fit with regularization parameter $\lambda$ is:

$$df(\lambda) = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$$

Explain what happens to $df(\lambda)$ as $\lambda \to 0$ and as $\lambda \to \infty$, and interpret these limits.

**(d)** [5 pts] A data analyst applies 10-fold cross-validation to select $\lambda$ for ridge regression. She notices that the minimum CV error is achieved at $\lambda^* = 0.1$, but she selects $\lambda = 0.5$ instead. Explain what rule she might be applying and why this choice can be preferable.

### Solution

**Part (a) — Closed-form Ridge derivation**

Start from the penalized objective:
$$J(\beta) = (Y - X\beta)^T(Y - X\beta) + \lambda\beta^T\beta$$

Expand:
$$J(\beta) = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta + \lambda\beta^T\beta$$

Take the derivative with respect to $\beta$:
$$\frac{\partial J}{\partial \beta} = -2X^TY + 2X^TX\beta + 2\lambda I\beta = 0$$

Key derivative rules: $\frac{\partial}{\partial \beta}(\beta^TA\beta) = 2A\beta$ when $A$ is symmetric; $\frac{\partial}{\partial \beta}(b^T\beta) = b$.

Set derivative to zero:
$$(X^TX + \lambda I)\beta = X^TY$$

$$\boxed{\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1} X^TY}$$

This always exists: $X^TX$ is positive semi-definite, and adding $\lambda I$ ($\lambda > 0$) makes $X^TX + \lambda I$ strictly positive definite → invertible.

---

**Part (b) — Geometric explanation of sparsity**

Both penalties can be written as constrained problems:
- **Ridge ($L_2$):** minimize RSS subject to $\|\beta\|_2^2 \leq s$
- **Lasso ($L_1$):** minimize RSS subject to $\|\beta\|_1 \leq s$

The RSS contours are ellipses centered at the OLS solution. The solution is where the RSS ellipse first contacts the constraint region.

- **Lasso ($L_1$) constraint region:** In 2D, a **diamond** with corners on the coordinate axes. The RSS ellipse typically first contacts the diamond at a **corner**, where one coordinate is exactly zero → sparse solution.
- **Ridge ($L_2$) constraint region:** In 2D, a **circle** with no corners. The RSS ellipse contacts the sphere at a smooth curved point that is almost never exactly on an axis → coefficients are never exactly zero.

---

**Part (c) — Effective degrees of freedom**

$$df(\lambda) = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$$

**As $\lambda \to 0$:** The Ridge smoother matrix $S_\lambda \to X(X^TX)^{-1}X^T$ (the OLS hat matrix). $\text{trace}(\text{OLS hat}) = p$ → $df(\lambda) \to p$.

*Interpretation:* No regularization → model uses all $p$ degrees of freedom, equivalent to fitting $p$ free parameters.

**As $\lambda \to \infty$:** All coefficients shrink to zero; $S_\lambda \to 0$ → $\text{trace}(S_\lambda) \to 0$ → $df(\lambda) \to 0$.

*Interpretation:* Extreme regularization → effectively zero degrees of freedom — the model makes the same prediction regardless of $x$.

The effective $df(\lambda)$ provides a continuous measure of model complexity interpolating between 0 and $p$.

---

**Part (d) — The 1-SE rule**

The analyst is applying the **one-standard-error (1-SE) rule** (Breiman et al. 1984).

**Rule:** After CV, choose the **largest $\lambda$** (most regularized/simplest model) whose CV error is within one standard error of the minimum CV error.

**Why this is preferable:**
1. The minimum CV error estimate is subject to estimation noise — the true optimal $\lambda$ may be somewhat larger.
2. Models within 1 SE of the minimum are statistically indistinguishable from the optimal model.
3. The 1-SE rule selects a simpler, more regularized model that is more stable across repeated analyses.
4. In the example: $\lambda^* = 0.1$ minimizes CV error, but $\lambda = 0.5$ is within 1 SE and produces a simpler, more stable model.

---

**Question (22)** [Weeks 9–12] — 20 points

You are given a dataset of fluorescence excitation-emission spectra for 80 chemical solutions. Each solution is measured at 30 excitation wavelengths and 50 emission wavelengths, yielding a 3-way tensor $\mathcal{X}$ of shape $80 \times 30 \times 50$ (samples × excitations × emissions). Each solution is known to contain a mixture of two fluorescent compounds (compound A and compound B) in varying concentrations.

**(a)** [5 pts] Explain why PARAFAC is a particularly natural model for this spectroscopic dataset. What physical interpretation do the three loading matrices $A$, $B$, and $C$ carry when $R = 2$ components are used?

**(b)** [5 pts] How would you select the number of PARAFAC components $R$? Describe two complementary methods and explain what each assesses.

**(c)** [5 pts] Suppose you instead fit a K-means clustering model to the 80 sample spectra. Compare this approach to PARAFAC in terms of: (i) the type of structure recovered, (ii) physical interpretability, and (iii) the effect of the trilinear constraint in PARAFAC.

**(d)** [5 pts] After fitting PARAFAC with $R = 2$, you find that CORCONDIA = 87. What does this value tell you, and what would CORCONDIA $\approx 0$ indicate?

### Solution

**Part (a) — Why PARAFAC is natural**

Fluorescence is generated by a physically additive process: intensity at excitation $j$ and emission $k$ for sample $i$ is the sum of contributions from each compound:
$$x_{ijk} = \sum_r a_{ir} \cdot b_{jr} \cdot c_{kr} + \text{noise}$$

This is exactly the PARAFAC model. With $R = 2$:

- **$A \in \mathbb{R}^{80 \times 2}$ (sample mode):** $a_{i1}$ is proportional to the concentration of compound A in sample $i$; $a_{i2}$ to compound B.
- **$B \in \mathbb{R}^{30 \times 2}$ (excitation mode):** Column $b_r$ is the excitation spectrum of compound $r$ — how fluorescence varies with excitation wavelength.
- **$C \in \mathbb{R}^{50 \times 2}$ (emission mode):** Column $c_r$ is the emission spectrum of compound $r$.

The trilinear structure directly matches the physics: each compound contributes independently with its own spectral fingerprint, and the total signal is their additive mixture.

---

**Part (b) — Selecting R**

**Method 1: CORCONDIA** — Fit PARAFAC for several $R$ values. Compute:
$$\text{CORCONDIA} = 100\left(1 - \frac{\|\mathcal{I} - \mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$
CORCONDIA $\approx 100$: core $\mathcal{G}$ is nearly super-diagonal → PARAFAC structure fits → appropriate $R$. Drops sharply below $\sim 50$: $R$ too large. *Assesses: structural fit (does the trilinear model hold?).*

**Method 2: Split-Half FMS** — Split 80 samples into two halves of 40. Fit PARAFAC with rank $R$ to each half independently. Compute FMS (cosine similarity of matching loading vectors across halves). FMS $\approx R$: stable, reproducible components. FMS $\ll R$: $R$ too large, solutions are not reproducible. *Assesses: reproducibility and stability across independent subsets.*

Choose the largest $R$ where both CORCONDIA is high AND FMS is close to $R$.

---

**Part (c) — PARAFAC vs K-means**

**(i) Type of structure recovered:**
- **PARAFAC:** Decomposes the full 3-way tensor into additive rank-1 components. Recovers spectral profiles and concentration profiles simultaneously.
- **K-means:** Groups the 80 samples into $K$ clusters by similarity of vectorized spectra. Finds group membership but does not decompose spectral variation.

**(ii) Physical interpretability:**
- **PARAFAC:** Loading vectors $B$ and $C$ directly recover excitation and emission spectra of each compound — physically interpretable as pure-component spectra. $A$ gives concentration estimates.
- **K-means:** Cluster centroids are average spectra of grouped samples, not pure-component spectra. No direct physical interpretation in terms of underlying compounds.

**(iii) Trilinear constraint:**
- PARAFAC imposes $x_{ijk} = \sum_r a_{ir}b_{jr}c_{kr}$, encoding the physics of additive fluorescence. This acts as strong regularization, prevents overfitting, and ensures essential uniqueness (up to sign/permutation). K-means vectorizes spectra, destroying the 2D excitation-emission structure and the trilinear relationship.

---

**Part (d) — CORCONDIA = 87**

**CORCONDIA = 87:** Close to 100. The fitted core $\mathcal{G}$ is nearly super-diagonal — the PARAFAC model with $R = 2$ is a good fit. The trilinear structure is approximately satisfied; the two-component model captures the main variation. The small deviation from 100 may reflect minor noise or slight spectral overlap. Overall, $R = 2$ is likely appropriate.

**CORCONDIA $\approx 0$ (or negative):** The fitted core deviates severely from super-diagonal — dense off-diagonal entries mean components are interacting (cross-talk), contradicting PARAFAC's independence assumption. This signals $R$ is too large: the model is fitting noise and extra components are not physically meaningful. Action: reduce $R$ or switch to Tucker3 (which explicitly models cross-talk via a full core tensor).
