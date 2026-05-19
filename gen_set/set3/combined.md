# Practice Set 3 — CDA 02582 (Combined Questions + Solutions)

**Format:** 20 multiple-choice + 2 open questions
**Scoring:** MC: +1 (correct), −0.25 (incorrect), 0 (unanswered)
**Duration:** 4 hours

---

## Multiple Choice

---

**Question (1)** [Week 1]
The Gauss-Markov theorem guarantees that the OLS estimator $\hat{\beta} = (X^TX)^{-1}X^Ty$ is BLUE (Best Linear Unbiased Estimator). Which of the following is the complete set of assumptions required for the theorem to hold?

A. The errors are normally distributed with mean zero and constant variance $\sigma^2$.
B. The errors have mean zero, constant variance $\sigma^2$, and are uncorrelated; $X$ has full column rank.
C. The errors are i.i.d. Gaussian; the design matrix $X$ must be orthogonal.
D. The errors have mean zero and are uncorrelated; no assumption on variance is needed.
E. None of the above.

#### Answer: **B**

- **A ✗** — Normality of errors is an additional assumption needed for inference (e.g. F-tests, t-tests), but is not required for the Gauss-Markov theorem itself. BLUE only requires zero mean, homoscedasticity, and uncorrelatedness.
- **B ✓** — These are exactly the Gauss-Markov conditions: $E[\varepsilon] = 0$, $\text{Var}(\varepsilon) = \sigma^2 I$ (constant variance and uncorrelated), and full column rank of $X$ (so that $(X^TX)^{-1}$ exists). Under these, OLS has the smallest variance among all linear unbiased estimators.
- **C ✗** — Orthogonality of $X$ is sufficient but far stronger than necessary; and Gaussianity is not required.
- **D ✗** — Constant (finite) variance is required. If variances differ across observations (heteroscedasticity), OLS is still unbiased but is no longer the minimum-variance linear unbiased estimator — GLS takes that role.
- **E ✗** — B is correct.

---

**Question (2)** [Week 1]
The hat matrix is defined as $H = X(X^TX)^{-1}X^T$. Which of the following statements about $H$ are correct? (Select all that apply.)

A. $H$ is idempotent: $H^2 = H$.
B. The rank of $H$ equals $p$, the number of predictors (columns of $X$), assuming $X$ has full column rank.
C. The diagonal entries $h_{ii}$ satisfy $0 \le h_{ii} \le 1$, and $\sum_i h_{ii} = p$.
D. $H$ is a symmetric matrix with all eigenvalues equal to 1.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — $H^2 = X(X^TX)^{-1}X^T \cdot X(X^TX)^{-1}X^T = X(X^TX)^{-1}X^T = H$. Idempotency is the defining property of a projection matrix.
- **B ✓** — The rank of a projection onto a $p$-dimensional column space equals $p$ (when $X$ is $n \times p$ with full column rank). Equivalently, $\text{rank}(H) = \text{tr}(H) = p$.
- **C ✓** — Because $H$ is idempotent and symmetric, its eigenvalues are 0 or 1, which forces $0 \le h_{ii} \le 1$. The trace equals the rank: $\sum_i h_{ii} = \text{tr}(H) = p$.
- **D ✗** — $H$ is symmetric, but its eigenvalues are 0 (with multiplicity $n-p$) and 1 (with multiplicity $p$), not all equal to 1.
- **E ✗** — A, B, and C are all correct.

---

**Question (3)** [Week 2]
A researcher preprocesses all 500 samples (standardizing features, then performing PCA to reduce to 20 dimensions), then performs 5-fold cross-validation to evaluate classifier accuracy. Which best describes the problem with this approach?

A. The sample size is too small to obtain reliable 5-fold CV estimates.
B. The preprocessing was applied to the entire dataset before splitting, so test-fold information leaked into training-fold preprocessing — the CV error estimate is optimistically biased.
C. PCA cannot be used as a preprocessing step for classification tasks.
D. 5-fold CV always underestimates true test error and should be replaced by LOOCV.
E. None of the above.

#### Answer: **B**

- **A ✗** — 500 samples is reasonable for 5-fold CV; sample size is not the issue here.
- **B ✓** — This is the classic data leakage error. When standardization and PCA are fit on all 500 samples, the transformation incorporates information from what will become test folds. The correct pipeline fits all preprocessing transformations on the training folds only, then applies them to the test fold. This inflates apparent performance because the test data is no longer truly unseen.
- **C ✗** — PCA is commonly used for dimensionality reduction before classification; it is entirely legitimate when applied correctly inside the CV loop.
- **D ✗** — 5-fold CV can have slightly higher bias than LOOCV but does not always underestimate test error; this is not the relevant issue here.
- **E ✗** — B is correct.

---

**Question (4)** [Week 2]
The in-sample optimism of a model fit to $N$ training observations is defined as $\omega = \text{Err}_{\text{in}} - \overline{\text{err}}$, where $\overline{\text{err}}$ is the training error. Which of the following best describes what drives larger optimism?

A. Using a larger training set $N$ always increases optimism because more data means more overfitting opportunities.
B. Optimism increases with model complexity (effective degrees of freedom $d$), and is approximated as $\omega \approx \frac{2d\sigma^2}{N}$ under squared-error loss.
C. Optimism is always zero for linear models, since they are unbiased estimators.
D. Optimism is independent of the number of parameters and depends only on the noise level $\sigma^2$.
E. None of the above.

#### Answer: **B**

- **A ✗** — Larger $N$ actually decreases optimism (it appears in the denominator of the formula $2d\sigma^2/N$). More data reduces overfitting for a fixed model complexity.
- **B ✓** — The covariance-based definition gives $\omega = (2/N)\sum_{i=1}^N \text{Cov}(\hat{y}_i, y_i)$. Under squared-error loss, this simplifies to $2d\sigma^2/N$ where $d$ is the effective degrees of freedom. Complexity increases optimism; larger $N$ reduces it.
- **C ✗** — Linear models are biased toward zero (Ridge) or have zero bias only for OLS, but optimism is nonzero for OLS because $\overline{\text{err}}$ systematically underestimates test error; $\omega = 2p\sigma^2/N > 0$.
- **D ✗** — Optimism grows linearly with effective degrees of freedom $d$, so it definitely depends on the number of parameters.
- **E ✗** — B is correct.

---

**Question (5)** [Week 3]
The Elastic Net minimizes:
$$\hat{\beta} = \arg\min_\beta \left\{ \|y - X\beta\|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2 \right\}$$
Which of the following statements about Elastic Net are correct? (Select all that apply.)

A. When $\lambda_1 = 0$, Elastic Net reduces to Ridge regression.
B. Elastic Net can select groups of correlated variables together, unlike pure Lasso which tends to select only one.
C. When $\lambda_2 = 0$, Elastic Net reduces to Lasso regression.
D. Elastic Net always produces sparser solutions than Lasso for any given $\lambda_1 > 0$.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Setting $\lambda_1 = 0$ removes the $\ell_1$ penalty, leaving only the $\ell_2$ ridge penalty: $\|y - X\beta\|^2 + \lambda_2\|\beta\|_2^2$, which is exactly Ridge.
- **B ✓** — The $\ell_2$ component creates a grouping effect: correlated predictors tend to receive similar, nonzero coefficients. Pure Lasso is indifferent among correlated variables and often picks one arbitrarily.
- **C ✓** — Setting $\lambda_2 = 0$ removes the ridge component and leaves only the Lasso objective.
- **D ✗** — Adding the $\ell_2$ penalty to Lasso makes the problem strictly convex and actually tends to produce less sparse solutions than pure Lasso at comparable $\lambda_1$, not more sparse.
- **E ✗** — A, B, and C are all correct.

---

**Question (6)** [Week 3]
In multiple hypothesis testing, suppose you test $m = 1000$ hypotheses simultaneously. Which of the following correctly describes the difference between Bonferroni correction and the Benjamini-Hochberg (BH) procedure?

A. Bonferroni controls the False Discovery Rate (FDR) at level $\alpha$; BH controls the Family-Wise Error Rate (FWER) at level $\alpha$.
B. Bonferroni controls the FWER (probability of at least one false positive) by using threshold $\alpha/m$; BH controls the FDR (expected proportion of false positives among rejections) and is less conservative.
C. Both Bonferroni and BH control the FWER, but BH is more powerful.
D. Bonferroni is always preferred because it has higher statistical power than BH.
E. None of the above.

#### Answer: **B**

- **A ✗** — The definitions are reversed. Bonferroni controls FWER; BH controls FDR.
- **B ✓** — Bonferroni uses the threshold $p_i \le \alpha/m$ for each test, controlling $P(\text{at least one false rejection}) \le \alpha$. This is very conservative when $m$ is large. BH orders p-values and rejects $H_{(1)}, \ldots, H_{(k)}$ where $k = \max\{i : p_{(i)} \le (i/m)\alpha\}$, controlling $E[\text{FDP}] \le \alpha$. BH is less conservative and has greater power when many true effects exist.
- **C ✗** — BH controls FDR, not FWER. The two criteria are fundamentally different.
- **D ✗** — BH generally has higher power than Bonferroni, especially when many true signals are present, because FDR control is a weaker (less strict) requirement than FWER control.
- **E ✗** — B is correct.

---

**Question (7)** [Week 4]
In Linear Discriminant Analysis (LDA) with $K = 2$ classes, the discriminant function is derived under the assumption that both classes share the same covariance matrix $\Sigma$. Which of the following best explains why the decision boundary is linear?

A. The decision boundary is linear because LDA uses a linear kernel internally.
B. When both classes share $\Sigma$, the quadratic terms $x^T\Sigma^{-1}x$ appear in both class log-likelihoods and cancel, leaving only the linear term $x^T\Sigma^{-1}(\mu_k - \mu_{k'})$ in the decision rule.
C. LDA assumes that all features are independent, which forces the boundary to be linear.
D. The decision boundary is linear because LDA uses only the first principal component of the data.
E. None of the above.

#### Answer: **B**

- **A ✗** — LDA does not use a kernel; the linearity is a direct consequence of the equal-covariance assumption in the Gaussian model.
- **B ✓** — Under $p(x|C_k) = \mathcal{N}(\mu_k, \Sigma)$ for both classes, the log-posterior ratio is $\log\frac{P(C_1|x)}{P(C_2|x)} \propto x^T\Sigma^{-1}\mu_1 - \frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1 - x^T\Sigma^{-1}\mu_2 + \frac{1}{2}\mu_2^T\Sigma^{-1}\mu_2$. The quadratic term $x^T\Sigma^{-1}x$ is identical for both classes and cancels, yielding a linear function of $x$. QDA retains the quadratic term because the covariances differ.
- **C ✗** — LDA does not assume feature independence; it assumes a full (shared) covariance structure. Naïve Bayes assumes independence.
- **D ✗** — LDA projects onto a linear discriminant direction, but this is not why the boundary is linear. The boundary linearity comes from the equal-covariance Gaussian derivation.
- **E ✗** — B is correct.

---

**Question (8)** [Week 4]
Logistic regression is typically fitted using the Newton-Raphson algorithm, also known as Iteratively Reweighted Least Squares (IRLS). Which of the following correctly describes the weight matrix $W$ in the IRLS update?

A. $W$ is a diagonal matrix with entries $w_{ii} = \hat{p}_i(1 - \hat{p}_i)$, where $\hat{p}_i$ is the current predicted probability for observation $i$.
B. $W$ is the identity matrix, making IRLS equivalent to ordinary least squares.
C. $W$ is a diagonal matrix with entries $w_{ii} = \hat{p}_i^2$, reflecting the squared predicted probabilities.
D. $W$ depends on the residuals $y_i - \hat{p}_i$ only, not on the predicted probabilities.
E. None of the above.

#### Answer: **A**

- **A ✓** — In IRLS for logistic regression, the working response is $z_i = \hat{\eta}_i + (y_i - \hat{p}_i)/(\hat{p}_i(1-\hat{p}_i))$ and the weight matrix is $W = \text{diag}(\hat{p}_i(1-\hat{p}_i))$. This comes from the second derivative (Fisher information) of the Bernoulli log-likelihood. When $\hat{p}_i \to 0$ or $\hat{p}_i \to 1$ (perfect separation), $w_{ii} \to 0$, which can cause numerical issues and slow convergence.
- **B ✗** — If $W = I$, IRLS would reduce to OLS, ignoring the Bernoulli variance structure.
- **C ✗** — $\hat{p}_i^2$ would be correct only for a different link function; for the logit link the correct weight is $\hat{p}_i(1-\hat{p}_i)$.
- **D ✗** — The weights depend on predicted probabilities, not on the residuals directly.
- **E ✗** — A is correct.

---

**Question (9)** [Week 5]
A regression tree splits the predictor space by minimizing residual sum of squares (RSS). At each node, the algorithm searches over all features $j$ and split points $s$. Which of the following best describes the prediction made in a leaf node?

A. The prediction is the median of the training observations falling in that region, because the median minimizes MAE.
B. The prediction is the mean of the training observations falling in that region, because the mean minimizes the within-region RSS.
C. The prediction is determined by a linear regression fit on the observations in that region.
D. The prediction is the mode of the training labels in that region, as in classification trees.
E. None of the above.

#### Answer: **B**

- **A ✗** — The median minimizes MAE (mean absolute error), not RSS. Regression trees use squared-error loss, so the optimal constant predictor is the mean.
- **B ✓** — For a leaf region $R_m$, the optimal constant prediction under squared-error loss is $\hat{c}_m = \text{mean}(y_i : x_i \in R_m)$. This directly minimizes $\sum_{x_i \in R_m}(y_i - c_m)^2$.
- **C ✗** — CART regression trees use constant (not linear) predictions in leaves. Model trees (e.g., M5) use linear leaf models, but that is not standard CART.
- **D ✗** — Mode is the correct prediction for classification trees (minimizing misclassification); regression trees use the mean.
- **E ✗** — B is correct.

---

**Question (10)** [Week 5]
In cost-complexity pruning of a regression tree, the weakest link $\alpha_t$ for subtree $T_t$ rooted at internal node $t$ is defined as:
$$\alpha_t = \frac{R(t) - R(T_t)}{|T_t| - 1}$$
where $R(\cdot)$ denotes the resubstitution error and $|T_t|$ is the number of leaves. What happens to the optimal tree as $\alpha$ increases from 0?

A. The tree grows larger, adding more splits to reduce training error.
B. The tree shrinks, pruning away subtrees with the smallest benefit-per-leaf-added, eventually collapsing to a single node (root prediction).
C. The tree structure is unchanged; only the leaf predictions are penalized.
D. The tree immediately collapses to the root at any $\alpha > 0$.
E. None of the above.

#### Answer: **B**

- **A ✗** — Larger $\alpha$ penalizes complexity more heavily, which discourages additional splits and leads to smaller trees, not larger.
- **B ✓** — As $\alpha$ increases from 0, subtrees whose training error reduction per additional leaf is less than $\alpha$ become unprofitable and are pruned. The weakest link at each step is the subtree with the smallest $\alpha_t$. This produces a sequence of nested trees from the full tree down to the root. Cross-validation selects the $\alpha$ (and hence the tree size) with best generalization.
- **C ✗** — The pruning modifies the tree structure by removing entire subtrees, not just penalizing leaf predictions.
- **D ✗** — For small positive $\alpha$, many subtrees still justify their complexity. The tree only collapses to the root when $\alpha$ is large enough to penalize all splits.
- **E ✗** — B is correct.

---

**Question (11)** [Week 6]
Random Forest differs from standard Bagging in one critical way. Which of the following correctly identifies this difference and explains its consequence for prediction variance?

A. Random Forest uses a different bootstrap scheme: it samples with replacement at the feature level rather than the observation level.
B. At each split, Random Forest considers only a random subset of $m \ll p$ features, which reduces the correlation between trees and thereby reduces the ensemble variance below what bagging achieves.
C. Random Forest grows shallower trees than bagging, introducing more bias but less variance.
D. Random Forest uses out-of-bag samples to debias individual trees before aggregating.
E. None of the above.

#### Answer: **B**

- **A ✗** — Both Random Forest and Bagging bootstrap observations (rows), not features. The distinction is about which features are considered at each split node.
- **B ✓** — In bagging, all $p$ features are candidates at every split, so trees trained on bootstrap samples from the same data tend to be similar (high correlation $\rho$). The variance of the ensemble is $\rho\sigma^2 + (1-\rho)\sigma^2/B$. By restricting each split to $m = \sqrt{p}$ (classification) or $m = p/3$ (regression) randomly chosen features, Random Forest decorrelates the trees (lowers $\rho$), reducing the first term and making the ensemble variance substantially smaller than bagging.
- **C ✗** — Random Forest typically grows fully deep trees (unpruned), just as in bagging. Depth restriction is not the distinguishing feature.
- **D ✗** — OOB samples are used for error estimation and variable importance, not for debiasing individual trees.
- **E ✗** — B is correct.

---

**Question (12)** [Week 6]
In AdaBoost, after fitting weak classifier $G_m(x)$, the weights of training observations are updated. Let $\text{err}_m = \frac{\sum_{i=1}^N w_i \mathbf{1}(y_i \ne G_m(x_i))}{\sum_{i=1}^N w_i}$ and $\alpha_m = \log\frac{1 - \text{err}_m}{\text{err}_m}$. What happens when $\text{err}_m = 0$ (perfect classifier)?

A. $\alpha_m = 0$, so the perfect classifier gets no weight in the final ensemble.
B. $\alpha_m \to \infty$, meaning the perfect classifier gets infinite weight; AdaBoost will terminate or the misclassified weights grow to dominate later rounds.
C. $\alpha_m = 1$, so the classifier gets a weight of 1 regardless of error.
D. The observation weights remain unchanged because all observations were classified correctly.
E. None of the above.

#### Answer: **B**

- **A ✗** — $\alpha_m = \log((1-0)/0) = \log(\infty) \to \infty$, not 0. $\alpha_m = 0$ occurs when $\text{err}_m = 0.5$ (random guessing).
- **B ✓** — When $\text{err}_m = 0$, the denominator in $\frac{1-\text{err}_m}{\text{err}_m}$ is zero, driving $\alpha_m \to \infty$. In practice, AdaBoost typically terminates early (algorithm converged after one round). The weight update $w_i \leftarrow w_i \exp(\alpha_m \cdot \mathbf{1}(y_i \ne G_m(x_i)))$ would not change misclassified weights here since there are none, but the classifier's contribution to $F(x) = \sum_m \alpha_m G_m(x)$ becomes infinite — the ensemble is simply this single perfect classifier.
- **C ✗** — There is no fixed value of 1 for $\alpha_m$; it is derived from the weighted error and ranges from $-\infty$ to $\infty$.
- **D ✗** — The update formula multiplies by $e^{\alpha_m \cdot \mathbf{1}(\text{misclassified})}$; since no observations are misclassified, weights indeed do not change in terms of relative ratios — but the algorithmic consequence is the infinite $\alpha_m$ and early stopping, as described in B.
- **E ✗** — B is correct.

---

**Question (13)** [Week 7]
In soft-margin SVM, slack variables $\xi_i \ge 0$ are introduced. The primal problem is:
$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^Tx_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
Which of the following correctly describes the role of $C$ and the effect of $C \to \infty$ vs $C \to 0$?

A. $C$ controls the kernel bandwidth; larger $C$ means a wider RBF kernel.
B. $C$ is the regularization parameter: large $C$ penalizes margin violations heavily (approaching hard-margin SVM, small margin possible); small $C$ allows more violations, yielding a larger but softer margin and potentially underfitting.
C. $C \to 0$ reduces to hard-margin SVM, since no slack is allowed.
D. $C$ has no effect on the decision boundary when the data is linearly separable.
E. None of the above.

#### Answer: **B**

- **A ✗** — $C$ is a regularization/cost parameter, not a kernel parameter. The bandwidth of an RBF kernel is controlled by $\gamma$ (or $\sigma$).
- **B ✓** — $C$ trades off margin width against training error. As $C \to \infty$, the penalty for any $\xi_i > 0$ becomes prohibitive and the SVM approaches the hard-margin solution (no violations tolerated). As $C \to 0$, the slack penalty vanishes and the classifier will tolerate many violations, creating a wide margin but potentially misclassifying many training points.
- **C ✗** — It is $C \to \infty$ (not $C \to 0$) that approaches hard-margin SVM. $C \to 0$ maximizes tolerance for violations.
- **D ✗** — Even for linearly separable data, $C$ affects the boundary when slack is technically allowed (the hard-margin solution is recovered only as $C \to \infty$).
- **E ✗** — B is correct.

---

**Question (14)** [Week 8]
PCA can be derived via the SVD of the (centered) data matrix $X \in \mathbb{R}^{n \times p}$: $X = UDV^T$. Which of the following statements are correct? (Select all that apply.)

A. The columns of $V$ (right singular vectors) are the principal component directions (loadings).
B. The principal component scores (projections of data onto PCs) are given by $XV = UD$.
C. The eigenvalues of $X^TX$ equal $d_j^2$, where $d_j$ are the singular values of $X$.
D. The left singular vectors $U$ have no interpretable role in PCA; only $V$ matters.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — The right singular vectors $V$ are the eigenvectors of $X^TX$ (the sample covariance matrix up to scaling), which are precisely the PCA loading vectors.
- **B ✓** — $XV = UDV^TV = UD$ (since $V^TV = I$). The matrix $UD$ contains the principal component scores — the coordinates of each observation in the PC space.
- **C ✓** — $X^TX = VD^TU^TUD V^T = VD^2V^T$. The eigenvalues of $X^TX$ are thus $d_j^2$, i.e., the squared singular values of $X$. The variance explained by PC $j$ is $d_j^2/(n-1)$.
- **D ✗** — The left singular vectors $U$ form the columns of $UD$, i.e., they encode the directions of the PC scores in observation space. They are interpretable (e.g., in image PCA, they are "eigenfaces"). Saying they have no role is incorrect.
- **E ✗** — A, B, and C are correct.

---

**Question (15)** [Week 8]
Sparse PCA via the Penalized Matrix Decomposition (PMD) solves:
$$\max_{u, v} \; u^T X v \quad \text{s.t.} \quad \|u\|_2 \le 1,\; \|v\|_2 \le 1,\; \|v\|_1 \le c_v$$
What is the role of the $\ell_1$ constraint on $v$ and what operation does it correspond to algorithmically?

A. The $\ell_1$ constraint on $v$ promotes sparsity in the loading vector; it is enforced via soft-thresholding: $v \leftarrow S_\lambda(X^Tu) / \|S_\lambda(X^Tu)\|_2$.
B. The $\ell_1$ constraint penalizes large loading values equally to the $\ell_2$ constraint, producing Ridge-like shrinkage.
C. The $\ell_1$ constraint is applied to $u$ (observation scores), not to $v$ (loadings), so it selects observations rather than features.
D. Soft-thresholding sets small entries of $v$ to a larger value, amplifying weak signals in the loading vector.
E. None of the above.

#### Answer: **A**

- **A ✓** — The $\ell_1$ ball constraint $\|v\|_1 \le c_v$ promotes sparsity in the loadings: only a subset of features have nonzero contributions to the sparse PC. In the alternating algorithm, given fixed $u$, the update for $v$ is obtained by computing $X^Tu$ and applying soft-thresholding $S_\lambda(\cdot)$ (setting entries with magnitude below $\lambda$ to zero), then normalizing. The threshold $\lambda$ is chosen to satisfy the $\ell_1$ constraint.
- **B ✗** — $\ell_1$ and $\ell_2$ constraints have very different effects: $\ell_2$ (Ridge) shrinks uniformly without sparsity; $\ell_1$ (Lasso) produces exact zeros in the solution.
- **C ✗** — The constraint is on $v$ (the $p$-dimensional loading vector), which selects features, not observations.
- **D ✗** — Soft-thresholding sets entries with small magnitude to exactly zero and shrinks the rest toward zero; it never amplifies entries.
- **E ✗** — A is correct.

---

**Question (16)** [Week 9]
In hierarchical clustering, three linkage methods are commonly used: single, complete, and average. Which of the following correctly pairs the linkage method with its definition and a characteristic behavior?

A. Single linkage uses the maximum distance between any pair of points in two clusters; it produces compact, globular clusters.
B. Complete linkage uses the minimum distance between any pair of points; it produces long, chained clusters (sensitive to outliers).
C. Average linkage computes the average distance between all pairs of points across two clusters and is a compromise between single and complete, less sensitive to outliers.
D. Complete linkage uses the maximum distance between any pair of points; it tends to produce tight, compact clusters but can split true clusters if they are elongated.
E. None of the above.

#### Answer: **C, D**

- **A ✗** — Single linkage uses the **minimum** distance (closest pair), not the maximum. It is prone to chaining, not compact clusters.
- **B ✗** — This describes single linkage (minimum), not complete linkage. The description of chaining and outlier sensitivity also fits single linkage.
- **C ✓** — Average linkage (UPGMA) uses the mean of all pairwise distances between clusters. It is a compromise that is less prone to chaining than single linkage and less sensitive to outliers than complete linkage.
- **D ✓** — Complete linkage uses the maximum pairwise distance (farthest neighbor). It ensures all points in a cluster are within a certain diameter of each other, producing compact clusters. However, it can struggle with non-globular (elongated) cluster shapes by splitting them.
- **E ✗** — C and D are correct.

---

**Question (17)** [Week 10]
Dropout is a regularization technique for neural networks. Which of the following statements about dropout are correct? (Select all that apply.)

A. During training, each neuron is independently set to zero with probability $p$ (drop probability), forcing the network to learn redundant representations.
B. During inference, dropout is turned off and neuron activations are scaled by $(1 - p)$ (or equivalently, training activations are scaled by $\frac{1}{1-p}$) to ensure consistent expected activations.
C. Dropout can be interpreted as training an exponentially large ensemble of sub-networks and averaging their predictions at test time.
D. Dropout reduces training speed but always improves test accuracy regardless of the dataset size.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — During each forward pass, neurons are randomly zeroed with probability $p$. This prevents co-adaptation: neurons cannot rely on any specific other neuron always being present, so they must learn more robust, distributed features.
- **B ✓** — At test time, dropout is disabled and all neurons are active. To match the expected activation at training time, outputs are multiplied by $(1-p)$. An equivalent implementation ("inverted dropout") scales activations by $1/(1-p)$ during training so no scaling is needed at test time.
- **C ✓** — Each dropout mask defines a different sub-network architecture. With $n$ neurons, there are $2^n$ possible sub-networks, and training with dropout approximately trains all of them with shared weights. Prediction at test time (using the full network with scaled weights) approximates the geometric mean of all sub-network predictions.
- **D ✗** — Dropout does increase training time (more epochs needed to converge) and does not always improve accuracy. For small datasets, dropout can actually hurt performance by being too aggressive a regularizer; its benefit is most pronounced in large, overparameterized networks.
- **E ✗** — A, B, and C are correct.

---

**Question (18)** [Week 11]
Archetypal Analysis (AA) represents data as convex combinations of archetypes, where each archetype is itself a convex combination of data points. Which of the following best describes the geometric interpretation and how AA differs from NMF and PCA?

A. Archetypes lie inside the data cloud (near the mean), minimizing reconstruction error like PCA; AA differs from PCA only by enforcing nonnegativity.
B. Archetypes are constrained to lie on the convex hull of the data (or near it), representing extreme profiles; unlike PCA (which finds directions of maximum variance) and NMF (which enforces nonnegativity), AA finds parts of the data that are genuinely extreme.
C. AA is equivalent to K-means clustering: each archetype is a cluster centroid.
D. NMF and AA are identical when the data matrix has all nonnegative entries; archetypes coincide with NMF basis vectors.
E. None of the above.

#### Answer: **B**

- **A ✗** — Archetypes are constrained to be convex combinations of data points and are pushed toward the extreme edges (convex hull) of the data cloud, not toward the center.
- **B ✓** — The AA constraint that archetypes are convex combinations of data points, and that data is reconstructed as convex combinations of archetypes, forces the archetypes toward the extremes of the data distribution (vertices of the convex hull or nearby). This gives each archetype an interpretable "extreme type" meaning. PCA directions are unconstrained and centered on the mean; NMF enforces nonnegativity but does not require parts-of-data convexity.
- **C ✗** — K-means centroids are means of cluster assignments and lie inside cluster mass; they are not extreme profiles. AA archetypes are geometrically very different from centroids.
- **D ✗** — NMF and AA have different constraints and objectives. NMF factorizes $X \approx WH$ with $W, H \ge 0$; AA specifically requires both that $X \approx Z\alpha$ (convex combination of archetypes) and that archetypes $Z = XB$ (convex combination of data points). These are not equivalent even for nonneg data.
- **E ✗** — B is correct.

---

**Question (19)** [Week 12]
The Tucker3 decomposition of a three-way tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$ is written as:
$$\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$
where $\mathcal{G} \in \mathbb{R}^{P \times Q \times R}$ is the core tensor. Which of the following statements are correct? (Select all that apply.)

A. Tucker3 generalizes PCA to three modes: applying $A$, $B$, $C$ to the three modes is analogous to projecting onto principal subspaces in each mode.
B. When $P = Q = R = 1$, Tucker3 reduces to a rank-1 tensor decomposition (equivalent to PARAFAC with one component).
C. Tucker3 has a unique solution (up to sign flips) when the factor matrices are constrained to be orthonormal.
D. The core tensor $\mathcal{G}$ describes the interaction between the subspaces spanned by $A$, $B$, and $C$.
E. None of the above.

#### Answer: **A, B, D**

- **A ✓** — The mode-$n$ products $\times_1 A$, $\times_2 B$, $\times_3 C$ each project the tensor onto lower-dimensional subspaces along that mode. With orthonormal factor matrices, this is exactly the multilinear generalization of PCA (HOSVD — Higher-Order SVD).
- **B ✓** — With $P = Q = R = 1$, the core becomes a scalar $g_{111}$, and $\mathcal{X} \approx g_{111} \cdot a_1 \circ b_1 \circ c_1$ — a rank-1 outer product, identical to a single PARAFAC component (up to the absorbed scalar).
- **C ✗** — Tucker3 with orthonormal factor matrices has rotational ambiguity: any orthogonal rotation $A \to AR_A$, $B \to BR_B$, $C \to CR_C$ with a corresponding transformation of $\mathcal{G}$ yields the same reconstruction. This is analogous to PCA's rotational ambiguity. PARAFAC (not Tucker3) achieves uniqueness under mild conditions.
- **D ✓** — The core tensor $\mathcal{G}_{pqr}$ encodes how strongly the $p$-th component of mode 1 interacts with the $q$-th component of mode 2 and the $r$-th component of mode 3. A diagonal core (as in PARAFAC) means modes interact only in matched components.
- **E ✗** — A, B, and D are correct.

---

**Question (20)** [Week 9 / Week 11]
Gaussian Mixture Models (GMM) use soft assignment of observations to clusters. Which of the following best describes what happens to the GMM as all covariance matrices $\Sigma_k \to 0$ (covariances shrink to zero)?

A. GMM becomes identical to PCA because the clusters collapse to line segments.
B. The soft assignments (posterior probabilities $r_{ik}$) approach hard 0/1 assignments, and GMM approaches K-means clustering with Euclidean distance.
C. GMM with shrinking covariances becomes equivalent to hierarchical clustering.
D. As $\Sigma_k \to 0$, the E-step of EM becomes numerically unstable but the final solution does not change.
E. None of the above.

#### Answer: **B**

- **A ✗** — Collapsing covariances does not produce line-like structures; clusters collapse to points (centroids), not lines.
- **B ✓** — When $\Sigma_k = \sigma^2 I$ and $\sigma^2 \to 0$, the Gaussian likelihood for cluster $k$ becomes sharply peaked around $\mu_k$. The posterior responsibility $r_{ik} \propto \pi_k \mathcal{N}(x_i; \mu_k, \sigma^2 I)$ approaches 1 for the nearest centroid and 0 for all others. The M-step for $\mu_k$ then becomes the mean of the (essentially hard-assigned) cluster members — exactly K-means. The GMM EM algorithm thus converges to the K-means algorithm in this limit.
- **C ✗** — Hierarchical clustering uses a different agglomerative or divisive procedure; the $\Sigma \to 0$ limit does not reproduce it.
- **D ✗** — Numerically, yes, shrinking covariances cause underflow problems. But the conceptual statement that the solution doesn't change is incorrect — the behavior does change qualitatively (soft to hard assignments), which is the important insight.
- **E ✗** — B is correct.

---

## Open Questions

---

### Question 21 (20 points) — Random Forest

**(a)** [5 pts] Describe the Random Forest algorithm. How does it differ from standard Bagging? What is the typical value of $m$ (the number of features considered at each split) for classification, and why is this choice motivated?

**(b)** [5 pts] The variance of a bagged ensemble of $B$ trees, each with variance $\sigma^2$ and pairwise correlation $\rho$, is:
$$\text{Var}(\hat{f}_{\text{bag}}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$
Explain how random feature subsampling in Random Forest reduces $\rho$, and what the consequence is as $B \to \infty$.

**(c)** [5 pts] Define permutation-based variable importance in Random Forest. How is it computed, and why is it a valid measure of feature relevance?

**(d)** [5 pts] Define the Out-of-Bag (OOB) error. Explain why OOB provides an approximately unbiased estimate of test error, and under what conditions it might differ from $K$-fold cross-validation.

---

### Solution

**Part (a) — Random Forest Algorithm and Difference from Bagging**

Both Bagging and Random Forest build an ensemble of $B$ decision trees by training each tree on a bootstrap sample of the training data (sampling $N$ observations with replacement). The prediction is the average (regression) or majority vote (classification) over all $B$ trees.

The single distinguishing change in Random Forest is **random feature subsampling at each split**: when constructing each node, instead of searching over all $p$ predictors for the best split, only a randomly chosen subset of $m < p$ features is considered. A new random subset of size $m$ is drawn independently at each node.

For classification, the standard recommendation is $m = \lfloor\sqrt{p}\rfloor$. For regression, $m = \lfloor p/3 \rfloor$ is typical. The motivation is that restricting to $m$ features at each split forces the trees to use different features, decorrelating them even when trained on similar bootstrap samples. This reduces the correlation $\rho$ between trees, which directly reduces the ensemble variance (see Part b). The choice $m = \sqrt{p}$ is empirically validated as a good default but can be tuned as a hyperparameter.

**Part (b) — Variance Reduction through Decorrelation**

The bagging variance formula is:
$$\text{Var}(\hat{f}_{\text{bag}}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

The first term $\rho\sigma^2$ is irreducible as $B \to \infty$ — adding more trees cannot reduce it. The second term $\frac{1-\rho}{B}\sigma^2$ vanishes as $B \to \infty$.

In standard Bagging, the trees are trained on bootstrap samples from the same data with all $p$ features available at every split. If a single strong predictor dominates, most trees will use it at the root and produce very similar structures. The correlation $\rho$ remains substantial, and the asymptotic variance floor $\rho\sigma^2$ is large.

In Random Forest, forcing each split to consider only $m$ features means that the dominant predictor is excluded from approximately $(1 - m/p)$ of all splits. Different trees are forced to rely on different predictors, so their structures and predictions are more diverse. This reduces $\rho$ substantially (often from $\sim 0.5$–$0.9$ in bagging to $\sim 0.05$–$0.3$ in Random Forest, depending on the data). The variance floor $\rho\sigma^2$ is therefore much lower.

As $B \to \infty$: the ensemble variance converges to the irreducible floor $\rho\sigma^2$. Random Forest reaches a lower floor than bagging due to smaller $\rho$. In practice, $B = 500$ is typically sufficient, as the improvement from additional trees becomes negligible.

**Part (c) — Permutation-Based Variable Importance**

For each tree $b$ and each feature $j$:
1. Predict on the OOB sample for tree $b$, recording the OOB error $\overline{\text{err}}^{(b)}$.
2. Randomly permute the values of feature $j$ across the OOB observations (breaking any association between $j$ and $y$).
3. Re-predict on the permuted OOB sample and record the permuted OOB error $\overline{\text{err}}^{(b)}_{\text{perm},j}$.
4. The importance of feature $j$ for tree $b$ is $\Delta^{(b)}_j = \overline{\text{err}}^{(b)}_{\text{perm},j} - \overline{\text{err}}^{(b)}$.

Variable importance for feature $j$ is averaged over all $B$ trees:
$$\text{VI}(j) = \frac{1}{B}\sum_{b=1}^B \Delta^{(b)}_j$$

A large $\text{VI}(j)$ means permuting feature $j$ substantially degrades OOB accuracy, indicating that $j$ carries important predictive information. If $j$ is irrelevant, permuting it does not change predictions, so $\text{VI}(j) \approx 0$.

This measure is valid because: (i) it uses held-out OOB data so it is not measuring memorization; (ii) it directly measures the functional contribution of each feature to predictive accuracy; (iii) it captures nonlinear and interaction effects that coefficient-based measures miss. One limitation: correlated features can split importance between them, understating the true importance of any individual correlated feature.

**Part (d) — OOB Error and Comparison to Cross-Validation**

For a bootstrap sample of size $N$ drawn with replacement, each observation is excluded from approximately $\frac{1}{e} \approx 36.8\%$ of bootstrap samples. For tree $b$, let $\mathcal{O}_b$ denote the set of observations not included in bootstrap sample $b$ (the out-of-bag set for tree $b$). The OOB prediction for observation $i$ uses only trees for which $i \in \mathcal{O}_b$:
$$\hat{f}^{\text{OOB}}(x_i) = \frac{1}{|\{b : i \in \mathcal{O}_b\}|} \sum_{b:\, i \in \mathcal{O}_b} T_b(x_i)$$

The OOB error is then:
$$\text{Err}^{\text{OOB}} = \frac{1}{N}\sum_{i=1}^N L\!\left(y_i,\, \hat{f}^{\text{OOB}}(x_i)\right)$$

**Why it is approximately unbiased:** Each OOB prediction is made by trees trained on data that did not include observation $i$, making the setup analogous to leave-one-out cross-validation (LOOCV). Asymptotically (large $B$), each observation is predicted by roughly $B/e \approx 0.368B$ trees, all trained without seeing that observation.

**Differences from $K$-fold CV:** (i) OOB is computationally free — no additional model fits are needed beyond the Random Forest itself. (ii) OOB ensemble size varies per observation (some observations appear in more bootstrap samples than others), which can add noise for small $B$. (iii) OOB assesses the full $B$-tree forest indirectly; $K$-fold CV trains on $(1 - 1/K) \cdot N$ observations per fold, which slightly underestimates the performance of the model trained on all $N$ observations. (iv) For small datasets with strong correlations in the data (violating IID), OOB shares the same limitation as any CV method that does not account for the correlation structure.

---

### Question 22 (20 points) — Cross-Validation Design for Wearables Dataset

**Dataset:** 16 subjects $\times$ 3 activities $\times$ 4 seasons = 192 observations total. Goal: predict physical activity from wearable biosignals.

**(a)** [6 pts] A researcher performs random 5-fold CV on all 192 observations to tune and evaluate a classifier. Explain precisely why this produces optimistically biased generalization error estimates. Which statistical principle is violated? What is the correct CV scheme for a generalized (person-independent) model?

**(b)** [6 pts] Describe Leave-One-Individual-Out (LOIO) CV in detail: number of folds, training/test observations per fold, and what generalization question it answers. Write the EPE estimator formula.

**(c)** [4 pts] Describe Leave-One-Season-Out (LOSO) CV for a **personalized** model evaluated on one subject. How many folds are used, and how many training observations are available per fold?

**(d)** [4 pts] If nested CV (double-loop) is used for simultaneous model selection and assessment, describe the two loops and explain why the inner loop must not use the outer test fold.

---

### Solution

**Part (a) — Why Random 5-Fold CV is Biased**

Random 5-fold CV randomly assigns observations to 5 equally-sized folds without regard to subject identity. With 192 observations and 16 subjects (12 observations per subject), each fold of size $\approx 38$ will contain observations from nearly all 16 subjects. Specifically, each subject contributes roughly $12/5 \approx 2.4$ observations to each fold. Consequently, the training fold and test fold for any split both contain data from the same individuals.

The violated principle is the **independence assumption** (IID): observations from the same person are highly correlated — they share the same physiology, gait, and biosignal characteristics. When the training set contains observations from a person who also appears in the test set, the model has effectively "seen" that person during training. The classifier can learn person-specific patterns (e.g., a particular heart rate baseline) and use them to classify test observations from that same person, inflating apparent accuracy.

This constitutes **data leakage through subject identity**: the model appears to generalize to new observations, but those observations are from people the model has already encountered. The result is optimistically biased generalization error — the reported error is lower than the true error on genuinely new, unseen individuals.

**Correct scheme:** For a person-independent (generalized) model, use **Leave-One-Individual-Out (LOIO) CV**, ensuring that no observations from the test subject appear anywhere in the training data.

**Part (b) — Leave-One-Individual-Out (LOIO) CV**

LOIO CV is structured as follows:

- **Number of folds:** $K = 16$ (one fold per subject).
- **Each fold:** The test set consists of all 12 observations from the held-out subject ($3 \text{ activities} \times 4 \text{ seasons}$). The training set consists of the remaining $192 - 12 = 180$ observations from the other 15 subjects.
- **What it answers:** Can the trained model accurately predict activity for a person it has never seen? This is the generalized (person-independent) error — the relevant quantity for deploying the classifier on a new user without any personalization.

The EPE (Expected Prediction Error) estimator under LOIO is:
$$\widehat{\text{EPE}}_{\text{LOIO}} = \frac{1}{16} \sum_{s=1}^{16} \frac{1}{12} \sum_{i \in \text{fold}_s} L\!\left(y_i,\, \hat{f}_{-s}(x_i)\right)$$

where $\hat{f}_{-s}$ is the model trained on all subjects except subject $s$, and $L(\cdot, \cdot)$ is the chosen loss (e.g., 0-1 loss for classification error rate). The outer average over subjects gives equal weight to each individual.

**Part (c) — Leave-One-Season-Out (LOSO) for Personalized Model**

For a personalized model, training and evaluation are performed within a single subject. One subject has $3 \times 4 = 12$ observations (3 activities $\times$ 4 seasons).

LOSO CV structure for one subject:

- **Number of folds:** $K = 4$ (one per season).
- **Test set per fold:** All 3 observations from the held-out season (one per activity).
- **Training set per fold:** The remaining $12 - 3 = 9$ observations from the 3 other seasons.
- **What it answers:** Can a personalized model (trained on this person's past data from other seasons) predict their activity in a new season?

This respects the temporal/seasonal structure: data from the test season is entirely held out, preventing leakage of season-specific patterns. This is the appropriate design because seasons may differ in ambient temperature, outdoor activity patterns, and other factors that affect biosignals.

**Part (d) — Nested CV for Model Selection and Assessment**

Nested (double-loop) CV separates two tasks:

**Outer loop (assessment):** Estimates the generalization error of the final model selection procedure. In the wearables context, use LOIO with $K_{\text{outer}} = 16$ folds. Subject $s$ is withheld entirely as the outer test fold. The remaining 15 subjects form the outer training pool.

**Inner loop (model selection / hyperparameter tuning):** Within the outer training pool (15 subjects), a second CV (e.g., LOIO over those 15 subjects, with $K_{\text{inner}} = 15$ folds) is used to tune hyperparameters (e.g., regularization strength $\lambda$, number of features, classifier type). The inner loop selects the best model configuration using only the 15-subject pool.

**Why the inner loop cannot use the outer test fold:** The outer test fold is held out precisely to provide an unbiased estimate of how the entire model-selection procedure generalizes to new subjects. If hyperparameter tuning (inner loop) used the outer test subject, the selected model would be adapted to that person's data — any "good" performance on the outer test fold would reflect this leakage, not true generalization. The outer test subject must remain completely invisible until the final evaluation step, after both model selection and training on all 15 outer-training subjects are complete.

The nested CV EPE estimator is:
$$\widehat{\text{EPE}}_{\text{nested}} = \frac{1}{K_{\text{outer}}} \sum_{s=1}^{K_{\text{outer}}} L\!\left(y_s,\, \hat{f}_{\hat{\lambda}_{-s}}(x_s)\right)$$

where $\hat{\lambda}_{-s}$ is the hyperparameter chosen by the inner loop using only the outer training pool (subjects excluding $s$), and $\hat{f}_{\hat{\lambda}_{-s}}$ is the model retrained on all outer training data with that hyperparameter.
