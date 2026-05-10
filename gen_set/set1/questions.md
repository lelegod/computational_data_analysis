# Practice Set 1 — CDA 02582

**Format:** 20 multiple-choice + 2 open questions  
**Scoring:** MC: +1 (all correct answers selected), -1 (any wrong selection), 0 (unanswered)  
**Duration:** 4 hours

---

## Multiple Choice

**Question (1)** [Week 1]  
The Expected Prediction Error (EPE) at a new point $x_0$ can be written as:

$$\text{EPE} = E(y - \hat{f})^2 = \sigma^2 + \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f})$$

Which of the following statements about this decomposition are correct?

A. Increasing model complexity tends to decrease bias and increase variance simultaneously.  
B. The irreducible noise term $\sigma^2$ can be reduced by choosing a sufficiently flexible model.  
C. The variance component measures how much the fitted model fluctuates across different training datasets.  
D. For Ordinary Least Squares (OLS) with all p predictors included, the bias is exactly zero.  
E. None of the above.

---

**Question (2)** [Week 1]  
Which of the following correctly describes the Ridge regression estimator and its key properties?

A. The ridge estimator is $\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1} X^Ty$, which is always invertible for $\lambda > 0$.  
B. Ridge regression shrinks some coefficients to exactly zero, performing automatic variable selection.  
C. As $\lambda \to \infty$, the effective degrees of freedom $df(\lambda) = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$ decreases toward zero.  
D. Ridge regression is unbiased for any value of $\lambda > 0$.  
E. None of the above.

---

**Question (3)** [Week 1]  
Consider the model selection criteria Cp, AIC, and BIC. Which of the following statements are correct?

A. For Gaussian errors, Cp and AIC are equivalent criteria.  
B. BIC uses a penalty of $\log(N)$ per parameter, whereas AIC uses a fixed penalty of 2 per parameter.  
C. AIC is asymptotically equivalent to leave-one-out cross-validation.  
D. For large N, BIC tends to select more complex models than AIC because its penalty grows with N.  
E. None of the above.

---

**Question (4)** [Week 2]  
Which of the following statements about the Lasso are correct?

A. The Lasso objective is $\min_\beta (Y - X\beta)^T(Y - X\beta) + \lambda\|\beta\|_1$, where $\|\beta\|_1 = \sum|\beta_j|$.  
B. The Lasso has a closed-form solution analogous to the Ridge estimator.  
C. The geometry of the $L_1$ constraint region (a diamond in 2D) explains why Lasso solutions are often sparse.  
D. In the $p > n$ setting, Lasso can select at most $n$ non-zero coefficients.  
E. None of the above.

---

**Question (5)** [Week 2]  
A researcher runs 50 independent hypothesis tests at individual significance level $\alpha = 0.05$. They apply the Bonferroni correction. Which of the following are correct?

A. The Bonferroni-corrected threshold for each individual test is $p < 0.001$.  
B. Without any correction, the Family-Wise Error Rate (FWER) is approximately $1 - (0.95)^{50} \approx 0.923$.  
C. The Bonferroni correction controls the FWER at level $\alpha = 0.05$ across all 50 tests.  
D. The Bonferroni correction has higher statistical power than the Benjamini-Hochberg procedure at the same overall $\alpha$ level.  
E. None of the above.

---

**Question (6)** [Week 2]  
In nested cross-validation (double-loop CV), which of the following statements are correct?

A. The outer loop is used for model selection (tuning hyperparameters), and the inner loop is used for model assessment.  
B. Nested CV audits the entire modelling pipeline including the hyperparameter selection step.  
C. A large gap between inner-loop error and outer-loop error suggests selection-induced overfitting.  
D. Nested CV is unnecessary when AIC or BIC is used for model selection.  
E. None of the above.

---

**Question (7)** [Week 3]  
Regarding the curse of dimensionality and regularization methods, which of the following are correct?

A. As the number of dimensions D increases, a fixed number of training points N becomes exponentially sparse in the feature space.  
B. In the elastic net, setting $\alpha = 1$ gives pure Ridge regression.  
C. The elastic net penalty combines an $L_1$ term and an $L_2$ term, allowing both variable selection and grouping of correlated predictors.  
D. Donoho (2000) identified that high-dimensional data often lies on a low-dimensional manifold as a "blessing" of dimensionality.  
E. None of the above.

---

**Question (8)** [Week 4]  
A classification tree is being grown on training data. Which of the following statements about splitting criteria are correct?

A. The Gini index for a node is defined as $G = \sum_k p_{mk}(1 - p_{mk})$, and equals zero when the node is completely pure.  
B. The misclassification rate is the preferred criterion for growing classification trees because it is differentiable.  
C. Cross-entropy (deviance) and the Gini index are both more sensitive to changes in class probabilities than the misclassification rate.  
D. In CART, a regression tree prediction in each leaf region is the mean of the training responses in that region.  
E. None of the above.

---

**Question (9)** [Week 4 / Week 5]  
Cost-complexity pruning is applied to a fully grown CART regression tree. Which of the following are correct?

A. The cost-complexity criterion is $C_\alpha(T) = R(T) + \alpha|T|$, where $|T|$ is the number of terminal nodes.  
B. When $\alpha = 0$, the pruned tree is the root node (single-leaf tree) since no penalty is applied.  
C. As $\alpha$ increases, the selected subtree becomes smaller (fewer leaves).  
D. The pruning parameter $\alpha$ is typically chosen by minimizing cross-validation error over a sequence of candidate values.  
E. None of the above.

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

---

**Question (11)** [Week 6]  
Random Forests extend bagging by adding random feature subsampling at each split. Which of the following are correct?

A. The default heuristic for the number of candidate features per split in classification problems is $m = \lfloor\sqrt{p}\rfloor$.  
B. When $m = p$ (all features considered at each split), Random Forest reduces to standard bagging.  
C. Random Forests reduce variance compared to bagging by decorrelating the trees, which lowers the inter-tree correlation $\rho$.  
D. In gradient boosting, shallow trees (stumps) are preferred as base learners, while in Random Forests, deep trees are preferred.  
E. None of the above.

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

---

**Question (14)** [Week 8]  
Principal Component Analysis (PCA) is applied to a centered data matrix $X \in \mathbb{R}^{100 \times 5}$. The singular value decomposition gives $X = UDV^T$. The singular values are $d_1 = 8, d_2 = 6, d_3 = 4, d_4 = 2, d_5 = 1$. Which of the following are correct?

A. The fraction of total variance explained by the first two principal components is $(64 + 36) / (64 + 36 + 16 + 4 + 1) = 100/121 \approx 82.6\%$.  
B. The loading vectors (principal axes) $V$ are the right singular vectors of $X$, and are identical to the eigenvectors of the covariance matrix $X^TX/(n-1)$.  
C. PCA applied to unscaled data (without standardizing features) may be dominated by high-variance features measured in large units.  
D. Partial Least Squares (PLS) differs from PCA in that PLS ignores the response variable $y$ and maximizes only the variance of $Xv$.  
E. None of the above.

---

**Question (15)** [Week 9]  
K-means clustering is applied to a dataset. Which of the following statements are correct?

A. K-means minimizes the objective $\sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2$, where $\mu_k$ is the centroid of cluster $k$.  
B. K-means is guaranteed to converge to the global optimum regardless of initialization.  
C. The silhouette coefficient $s(i) = (b(i) - a(i)) / \max\{a(i), b(i)\}$ takes values in $[-1, 1]$, with values near $+1$ indicating well-clustered points.  
D. The gap statistic selects the number of clusters $K$ by comparing the within-cluster dispersion of the data to that expected under a uniform reference distribution.  
E. None of the above.

---

**Question (16)** [Week 9]  
The EM algorithm is used to fit a Gaussian Mixture Model (GMM) with $K$ components. Which of the following are correct?

A. In the E-step, the posterior probability $\gamma_{ij} = P(Z_i = j \mid x_i)$ is computed using Bayes' rule: $\gamma_{ij} = \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j) / \sum_{j'} \pi_{j'} \mathcal{N}(x_i; \mu_{j'}, \Sigma_{j'})$.  
B. In the M-step, the mean update is $\mu_j = \sum_i \gamma_{ij} x_i / \sum_i \gamma_{ij}$, a weighted average of data points with soft-assignment weights.  
C. K-means is a special case of GMM with equal, spherical covariances and hard (binary) assignments.  
D. GMM with full per-component covariance matrices always produces a unique global maximum of the likelihood.  
E. None of the above.

---

**Question (17)** [Week 10]  
Consider a fully connected feedforward neural network: Input layer: 5 nodes; Hidden layer 1: 3 nodes (ReLU); Hidden layer 2: 3 nodes (ReLU); Output layer: 2 nodes (softmax). Each layer includes a bias term. How many scalar parameters does this network have, and which additional statements are correct?

A. The total number of parameters is $(5\times3 + 3) + (3\times3 + 3) + (3\times2 + 2) = 18 + 12 + 8 = 38$.  
B. Binary cross-entropy loss is derived from maximizing the Bernoulli likelihood: $L = -\sum_i[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$.  
C. In backpropagation, the error signal $\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$ propagates blame backwards through the network.  
D. Recurrent Neural Networks (RNNs) suffer from the vanishing gradient problem for long sequences, motivating LSTM and GRU architectures.  
E. None of the above.

---

**Question (18)** [Week 11]  
Non-negative Matrix Factorization (NMF) and Independent Component Analysis (ICA) are both unsupervised decomposition methods. Which of the following statements are correct?

A. NMF enforces non-negativity on both factor matrices $W$ and $H$, producing an additive, parts-based representation with no cancellation between components.  
B. ICA requires that the source components are statistically independent and non-Gaussian; it cannot separate Gaussian sources.  
C. NMF solutions are unique — there is only one valid factorization $X \approx WH$ for given $W \geq 0$, $H \geq 0$.  
D. ICA preprocessing involves centering and whitening the data so that subsequent optimization needs only to find an orthogonal rotation matrix.  
E. None of the above.

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

---

## Open Questions

**Question (21)** [Weeks 1–3] — 10 points

Ridge regression and the Lasso are both regularization methods for linear regression, but they have fundamentally different properties.

**(a)** [3 pts] Derive the closed-form solution for Ridge regression starting from the penalized least squares objective:

$$\min_\beta (Y - X\beta)^T(Y - X\beta) + \lambda\beta^T\beta$$

Show all steps including the derivative, setting it to zero, and solving for $\hat{\beta}_\text{ridge}$.

**(b)** [3 pts] Explain geometrically why the Lasso ($L_1$ penalty) produces sparse solutions (exact zeros) while Ridge ($L_2$ penalty) does not. Refer to the shape of the constraint regions and the RSS contours.

**(c)** [2 pts] The effective degrees of freedom for a ridge fit with regularization parameter $\lambda$ is:

$$df(\lambda) = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$$

Explain what happens to $df(\lambda)$ as $\lambda \to 0$ and as $\lambda \to \infty$, and interpret these limits.

**(d)** [2 pts] A data analyst applies 10-fold cross-validation to select $\lambda$ for ridge regression. She notices that the minimum CV error is achieved at $\lambda^* = 0.1$, but she selects $\lambda = 0.5$ instead. Explain what rule she might be applying and why this choice can be preferable.

---

**Question (22)** [Weeks 9–12] — 10 points

You are given a dataset of fluorescence excitation-emission spectra for 80 chemical solutions. Each solution is measured at 30 excitation wavelengths and 50 emission wavelengths, yielding a 3-way tensor $\mathcal{X}$ of shape $80 \times 30 \times 50$ (samples × excitations × emissions). Each solution is known to contain a mixture of two fluorescent compounds (compound A and compound B) in varying concentrations.

**(a)** [3 pts] Explain why PARAFAC is a particularly natural model for this spectroscopic dataset. What physical interpretation do the three loading matrices $A$ (sample mode), $B$ (excitation mode), and $C$ (emission mode) carry when $R = 2$ components are used?

**(b)** [2 pts] How would you select the number of PARAFAC components $R$? Describe two complementary methods and explain what each assesses.

**(c)** [3 pts] Suppose you instead fit a K-means clustering model to the 80 sample spectra (vectorizing each $30 \times 50$ spectrum into a 1500-dimensional vector). Compare this approach to PARAFAC in terms of: (i) the type of structure each method recovers, (ii) whether the result is physically interpretable, and (iii) the effect of the trilinear constraint in PARAFAC.

**(d)** [2 pts] After fitting PARAFAC with $R = 2$, you find that CORCONDIA = 87. What does this value tell you, and what would CORCONDIA $\approx 0$ indicate?
