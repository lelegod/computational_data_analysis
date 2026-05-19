# Practice Set 5 — CDA 02582 (Combined Questions + Solutions)

**Format:** 20 multiple-choice + 2 open questions
**Scoring:** MC: +1 (correct), −0.25 (incorrect), 0 (unanswered)
**Duration:** 4 hours

---

## Multiple Choice

---

**Question (1)** [Week 1]

The ridge estimator $\hat{\beta}_\text{Ridge}(\lambda) = (X^TX + \lambda I)^{-1}X^Ty$ is biased but may still achieve lower MSE than OLS. Which of the following correctly states the condition under which Ridge strictly reduces MSE compared to OLS?

A. Ridge always has lower MSE than OLS regardless of $\lambda$.
B. Ridge has lower MSE than OLS whenever $\lambda > 0$, because shrinkage always reduces variance more than it increases bias.
C. Ridge has lower MSE than OLS when the variance reduction from shrinkage exceeds the squared bias introduced: $\text{Var}[\hat{\beta}_\text{OLS}] - \text{Var}[\hat{\beta}_\text{Ridge}] > \|\text{Bias}[\hat{\beta}_\text{Ridge}]\|^2$.
D. Ridge has lower MSE than OLS only when all true coefficients $\beta_j = 0$.
E. None of the above.

#### Answer: **C**

- **A ✗** — False. As $\lambda \to \infty$, Ridge shrinks all coefficients to zero, which can inflate MSE when true coefficients are large.
- **B ✗** — Partially true in direction but stated incorrectly; for large $\lambda$ the squared bias can dominate and MSE exceeds OLS.
- **C ✓** — This is exactly the bias-variance tradeoff condition. MSE(Ridge) = Var(Ridge) + Bias²(Ridge) and MSE(OLS) = Var(OLS) + 0. Ridge wins iff the variance reduction exceeds the bias penalty. There always exists a small $\lambda > 0$ for which this holds (by differentiating MSE w.r.t. $\lambda$ at $\lambda=0$).
- **D ✗** — If all true $\beta_j = 0$, OLS estimates are already zero in expectation, but this is a sufficient, not necessary condition.
- **E ✗** — C is correct.

---

**Question (2)** [Week 1]

For a linear model with $d$ parameters fit by OLS on $N$ training points, the optimism (expected difference between in-sample error and training error) is $2d\sigma^2/N$. If instead of OLS you use Ridge regression with shrinkage parameter $\lambda > 0$, which statement about the effective degrees of freedom $d_\text{eff}(\lambda)$ is correct?

A. $d_\text{eff}(\lambda) = d$ for all $\lambda > 0$ because Ridge still uses all $d$ predictors.
B. $d_\text{eff}(\lambda) = \sum_{j=1}^d \frac{d_j^2}{d_j^2 + \lambda}$ where $d_j$ are the singular values of $X$, and this quantity is strictly less than $d$ for $\lambda > 0$.
C. $d_\text{eff}(\lambda) = \sum_{j=1}^d \frac{d_j}{d_j + \lambda}$ where $d_j$ are the singular values of $X$, and this quantity is strictly less than $d$ for $\lambda > 0$.
D. $d_\text{eff}(\lambda) = 0$ for all $\lambda > 0$ because Ridge produces a biased estimator.
E. None of the above.

#### Answer: **B**

- **A ✗** — Ridge's effective degrees of freedom decrease with $\lambda$; using all predictors does not mean $d_\text{eff} = d$.
- **B ✓** — The hat matrix for Ridge is $H_\lambda = X(X^TX + \lambda I)^{-1}X^T$, whose trace gives $d_\text{eff}(\lambda) = \text{tr}(H_\lambda) = \sum_j d_j^2/(d_j^2 + \lambda)$, where $d_j$ are singular values of $X$. This is strictly less than $d$ for $\lambda > 0$, so the optimism $2\,d_\text{eff}(\lambda)\,\sigma^2/N$ is smaller than for OLS.
- **C ✗** — This formula uses $d_j$ (not $d_j^2$) in the numerator, which is incorrect; it corresponds to a different regularization scheme.
- **D ✗** — Bias does not collapse $d_\text{eff}$ to zero; $d_\text{eff} \to 0$ only as $\lambda \to \infty$.
- **E ✗** — B is correct.

---

**Question (3)** [Week 2]

A researcher builds a one-step-ahead forecast model on a financial time series with 500 daily observations. She randomly shuffles the data before applying 5-fold cross-validation. Which of the following best describes the problem with this approach?

A. 5-fold CV is too few folds for time series; she should use leave-one-out CV.
B. Random shuffling causes temporal leakage: future data ends up in training folds, making the model appear more accurate than it is on truly unseen future data.
C. Random shuffling is acceptable if the time series is stationary, because stationarity implies the observations are exchangeable.
D. The problem is that she should normalize the data before CV, not after.
E. None of the above.

#### Answer: **B**

- **A ✗** — The number of folds is not the primary issue; LOO would have the same leakage problem if done with random ordering.
- **B ✓** — When observations are temporally ordered, a future observation leaking into the training set allows the model to see the future, producing optimistically biased generalization estimates. The correct approach is rolling-window or expanding-window CV, where training always precedes validation in time.
- **C ✗** — Stationarity implies identically distributed marginals but does NOT imply independence or exchangeability; autocorrelation violates exchangeability.
- **D ✗** — Normalization leakage is a separate problem; the core issue here is temporal ordering.
- **E ✗** — B is correct.

---

**Question (4)** [Week 2]

A researcher compares 50 model configurations (different algorithms and hyperparameter combinations) using 10-fold CV on the same training set, then selects the best model and reports its CV error as the final performance estimate. Which statement is most accurate?

A. This procedure is unbiased because 10-fold CV is an unbiased estimator of generalization error.
B. This procedure is unbiased because cross-validation automatically corrects for the multiple-comparisons problem.
C. This procedure is biased downward (overly optimistic) because selecting the minimum CV error over many configurations is equivalent to overfitting to the validation folds.
D. This procedure is biased upward (overly pessimistic) because 10-fold CV underestimates performance on the full training set.
E. None of the above.

#### Answer: **C**

- **A ✗** — CV is an unbiased estimator for a *fixed* model, but after optimizing over 50 candidates the minimum is biased downward.
- **B ✗** — CV does not address multiple comparisons; it provides no correction for selection bias.
- **C ✓** — When the best model is chosen by minimizing CV error across many candidates, the selected error is an optimistic estimate due to the winner's curse (max-bias from taking the minimum of many noisy estimates). A separate, untouched test set is needed to get an unbiased final performance estimate.
- **D ✗** — 10-fold CV underestimates true performance slightly because it trains on fewer samples, but this is not the dominant bias here.
- **E ✗** — C is correct.

---

**Question (5)** [Week 3]

Consider the Lasso regularization path as $\lambda$ decreases from $\infty$ to $0$. Select ALL statements that are correct.

A. At $\lambda = \infty$, all coefficients are exactly zero.
B. As $\lambda$ decreases, coefficients enter the model one at a time (or possibly simultaneously), and the number of nonzero coefficients is non-decreasing.
C. Two highly correlated predictors can enter the model simultaneously along the Lasso path.
D. As $\lambda \to 0$, the Lasso solution converges to the OLS solution (assuming $N > p$).
E. None of the above.

#### Answer: **A, C, D**

- **A ✓** — At $\lambda = \infty$ the penalty overwhelms the data term and the solution is $\hat{\beta} = 0$.
- **B ✗** — The statement that coefficients are "non-decreasing" in count is not strictly guaranteed; a coefficient that entered can leave the model (become zero again) as $\lambda$ decreases further, especially with correlated predictors.
- **C ✓** — When two predictors are highly correlated, the Lasso may allow both to enter simultaneously (the LARS path has a "tie" at the same $\lambda$). This is a known property and motivates the Elastic Net.
- **D ✓** — At $\lambda = 0$ the $\ell_1$ penalty vanishes and Lasso reduces to OLS when the system is identified ($N > p$).
- **E ✗** — A, C, D are correct.

---

**Question (6)** [Week 3]

In a genomics study, researchers test $m = 10{,}000$ hypotheses simultaneously with a global significance level $\alpha = 0.05$. They apply Bonferroni correction and reject hypotheses where $p_i < \alpha/m = 5 \times 10^{-6}$. Which statement is correct?

A. Bonferroni correction controls the false discovery rate (FDR) at level $\alpha$.
B. Bonferroni correction controls the family-wise error rate (FWER), defined as the probability of making at least one false rejection, at level $\alpha$.
C. Bonferroni correction is exact (not conservative) when all tests are independent.
D. If the test statistics are positively correlated (as is common in genomics due to linkage disequilibrium), Bonferroni is anti-conservative and the actual FWER exceeds $\alpha$.
E. None of the above.

#### Answer: **B**

- **A ✗** — Bonferroni controls FWER, not FDR. The Benjamini-Hochberg procedure controls FDR.
- **B ✓** — By the union bound, $P(\text{any false rejection}) \leq \sum_{i: H_{0i} \text{ true}} P(p_i < \alpha/m) \leq m \cdot (\alpha/m) = \alpha$. This is the FWER.
- **C ✗** — When tests are independent, Bonferroni is slightly conservative (the actual FWER $= 1-(1-\alpha/m)^m < \alpha$, and $\alpha$ is an upper bound), but it is not exact — it is still conservative.
- **D ✗** — Positive correlation makes Bonferroni even more conservative (FWER $<\alpha$), not anti-conservative. The bound $P(\text{any false rejection}) \leq \sum P(p_i < \alpha/m)$ is valid regardless of dependence.
- **E ✗** — B is correct.

---

**Question (7)** [Week 4]

The LDA discriminant score for class $k$ is $\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k$. A new point $x$ is assigned to the class with the highest $\delta_k(x)$. What does each of the three terms represent?

A. The first term measures the distance from $x$ to class $k$; the second is a length correction; the third weights by prevalence.
B. The first term is the Mahalanobis inner product of $x$ with the class mean; the second is a quadratic normalization term ensuring the score is not artificially inflated for distant class means; the third adds the log prior so that rare classes require stronger evidence.
C. The second term $-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k$ is the squared Mahalanobis distance from the origin to $\mu_k$, which penalizes classes with means far from the origin.
D. The third term $\log\pi_k$ ensures that all $\delta_k$ values sum to one across classes.
E. None of the above.

#### Answer: **B, C**

- **A ✗** — The first term is not a distance; it is an inner product in the Mahalanobis metric, which increases as $x$ and $\mu_k$ align.
- **B ✓** — Correct interpretation. The score arises from the log-posterior: $\log P(Y=k|x) \propto \delta_k(x)$.
- **C ✓** — $-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k = -\frac{1}{2}\|\mu_k\|^2_{\Sigma^{-1}}$ is indeed the squared Mahalanobis distance from the origin to $\mu_k$; it plays the role of a normalization term so classes with far-away means do not unfairly dominate.
- **D ✗** — Log-probabilities do not sum to one; that would require probabilities (softmax form). The log prior merely shifts the threshold.
- **E ✗** — B and C are correct.

---

**Question (8)** [Week 4]

Logistic regression and LDA produce the same form of decision boundary (linear in $x$) but differ in how parameters are estimated. Which statement best describes a scenario where LDA is expected to outperform logistic regression?

A. LDA outperforms logistic regression whenever the training set is large.
B. LDA outperforms logistic regression when the data within each class is truly multivariate Gaussian with equal covariance matrices, because LDA exploits the full generative model and gains efficiency.
C. LDA always outperforms logistic regression in high-dimensional settings because it requires fewer parameters.
D. Logistic regression never outperforms LDA because LDA uses more information (the full joint distribution).
E. None of the above.

#### Answer: **B**

- **A ✗** — Large training sets favor logistic regression (less model misspecification risk); LDA's advantage lies in model correctness, not sample size.
- **B ✓** — LDA is the maximum-likelihood estimator under the Gaussian class-conditional model. When data truly follow this model, LDA is more statistically efficient than logistic regression (uses about 30% fewer training data to achieve the same error in the two-class case). When Gaussianity is violated, LDA's misspecification can hurt it.
- **C ✗** — Both models have $p+1$ effective parameters for the linear boundary; LDA additionally estimates $\Sigma$ and $\mu_k$, so it actually uses more parameters.
- **D ✗** — When Gaussian assumptions are wrong (e.g., heavy tails, mixed discrete/continuous features), logistic regression generally outperforms LDA.
- **E ✗** — B is correct.

---

**Question (9)** [Week 5]

Compare a single fully-grown CART tree to a bagged ensemble of 500 such trees. Select ALL correct statements.

A. A single fully-grown tree has low bias because it can fit the training data exactly; bagging does not change this bias.
B. Bagging primarily reduces variance by averaging over many bootstrapped trees, each of which has high variance.
C. A single tree is more interpretable than a bagged ensemble because it can be visualized as a single decision tree.
D. Bagging increases the risk of overfitting relative to a single tree because it fits more models.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Fully-grown trees have low bias (they approximate the training data well). Bagging averages trees trained on bootstrap samples, each with approximately the same bias; thus ensemble bias ≈ single-tree bias. The main gain is variance reduction.
- **B ✓** — This is the core mechanism of bagging: individual trees are high-variance (small changes in data lead to very different trees), and averaging $B$ weakly correlated estimators reduces variance by a factor approaching $1/B$ as correlation decreases.
- **C ✓** — A single decision tree can be printed or visualized and its rules can be inspected. A bagged ensemble of 500 trees has no single interpretable structure.
- **D ✗** — Bagging reduces overfitting risk compared to a single high-variance tree, because averaging over many trees cancels out noise-fitting; the ensemble has lower test error.
- **E ✗** — A, B, C are correct.

---

**Question (10)** [Week 5]

Out-of-bag (OOB) error is used as an internal error estimate in Random Forest. Which statements about OOB error are correct? Select ALL that apply.

A. For each training observation $x_i$, the OOB prediction averages only over trees for which $x_i$ was NOT in the bootstrap sample.
B. On average, about 36.8% of training observations are out-of-bag for any given tree.
C. OOB error approximates leave-one-out cross-validation (LOOCV) because each observation is predicted by a model not trained on it.
D. OOB error is always exactly equal to LOOCV error.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — This is the definition of OOB prediction: only trees whose bootstrap sample did not include $x_i$ contribute to the OOB vote for $x_i$.
- **B ✓** — Each bootstrap sample draws $N$ observations with replacement from $N$. The probability that a given observation is excluded from one draw is $(1-1/N)^N \to e^{-1} \approx 0.368$ as $N\to\infty$.
- **C ✓** — Each OOB prediction uses a model trained on approximately $N-1$ distinct data points (since roughly 63.2% of samples appear), analogous to LOOCV, but with a slightly smaller training set per fold.
- **D ✗** — OOB is an approximation; it is not exactly equal to LOOCV because (1) the training set for each OOB prediction is a bootstrap sample (not exactly $N-1$ points), and (2) the ensemble uses 500 trees whereas LOOCV uses $N$ separate models.
- **E ✗** — A, B, C are correct.

---

**Question (11)** [Week 6]

In gradient boosting with shrinkage (learning rate $\nu \in (0,1]$), the update rule is $f_m(x) = f_{m-1}(x) + \nu \cdot T(x; \Theta_m)$ where $T$ is a regression tree. Which statement best describes the role of $\nu$?

A. Smaller $\nu$ reduces the number of trees $M$ needed to achieve good performance because each tree contributes more.
B. Smaller $\nu$ acts as regularization: each tree makes a smaller contribution, requiring more trees $M$ but often yielding better generalization because the model explores the residual space more gradually.
C. Setting $\nu = 1$ is always optimal because it makes full use of each tree.
D. Shrinkage has no effect on bias; it only affects variance.
E. None of the above.

#### Answer: **B**

- **A ✗** — The opposite is true: smaller $\nu$ means each tree corrects less, so you need more trees to achieve the same training error.
- **B ✓** — Shrinkage is analogous to a small step size in gradient descent: it prevents the algorithm from over-correcting on noisy gradients. Empirically, small $\nu$ (e.g., $\nu = 0.01$–$0.1$) paired with large $M$ (found by early stopping via CV) consistently outperforms $\nu = 1$ with fewer trees.
- **C ✗** — $\nu = 1$ often overfits because the model rapidly "chases" noise in the residuals without regularization.
- **D ✗** — Shrinkage affects both bias (slows fitting) and variance (smoother trajectory through function space); the combined effect is typically a reduction in generalization error.
- **E ✗** — B is correct.

---

**Question (12)** [Week 6]

Match the boosting loss function to the correct gradient / pseudo-residual. Which set of matches is entirely correct?

- (i) Squared error loss $L(y,f) = \frac{1}{2}(y-f)^2$
- (ii) Exponential loss $L(y,f) = e^{-yf}$, $y \in \{-1,+1\}$
- (iii) Deviance (log-loss) $L(y,f) = \log(1 + e^{-2yf})$, $y \in \{-1,+1\}$

A. (i) residual $y-f$; (ii) $ye^{-yf}$; (iii) $\frac{2y}{1+e^{2yf}}$.
B. (i) residual $y-f$; (ii) sample-weight update recovers AdaBoost; (iii) leads to LogitBoost / gradient boosting for classification.
C. (i) leads to L2Boost where trees are fit to ordinary residuals; (ii) leads to AdaBoost where the pseudo-residual is $r_i = ye^{-yf}$; (iii) is more robust to outliers than exponential loss.
D. (ii) and (iii) are equivalent because both are used for binary classification.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — The negative gradient of each loss: $-\partial L/\partial f$ for (i) is $y-f$; for (ii) is $ye^{-yf}$; for (iii) is $2y/(1+e^{2yf})$. All correct.
- **B ✓** — These are the standard connections: (i) = L2Boost, (ii) = AdaBoost (forward stagewise with exponential loss), (iii) = logistic boosting.
- **C ✓** — Exponential loss assigns weight $e^{-y_i f_{m-1}(x_i)}$ to sample $i$; deviance loss has bounded influence function and is more robust to outliers than exponential loss.
- **D ✗** — Exponential and deviance losses are different; exponential loss is more sensitive to outliers (its influence function is unbounded), while deviance loss has a bounded influence function.
- **E ✗** — A, B, C are correct.

---

**Question (13)** [Week 7]

At test time, the SVM decision function is $\hat{y} = \text{sign}\!\left(\beta_0 + \sum_{i \in \mathcal{S}} \alpha_i y_i K(x_i, x)\right)$ where $\mathcal{S}$ is the set of support vectors. Which statements are correct? Select ALL that apply.

A. Non-support vectors (training points not on the margin) have $\alpha_i = 0$ and thus do not contribute to the prediction.
B. $\beta_0$ (the bias/intercept) is computed from any support vector on the margin by enforcing $y_i(\sum_{j \in \mathcal{S}} \alpha_j y_j K(x_j, x_i) + \beta_0) = 1$.
C. The kernel function $K(x_i, x)$ implicitly computes the inner product in the feature space $\phi(x_i)^T\phi(x)$ without explicitly mapping to that space.
D. The number of support vectors is always exactly 2 (one from each class).
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — By the KKT complementary slackness condition, $\alpha_i (y_i(\beta_0 + f(x_i)) - 1) = 0$. Points strictly inside the margin satisfy the constraint strictly, forcing $\alpha_i = 0$.
- **B ✓** — For any support vector $x_i$ on the margin, the constraint $y_i f(x_i) = 1$ allows solving for $\beta_0$. In practice, the mean over all margin support vectors is used for numerical stability.
- **C ✓** — This is the kernel trick: $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ for some (possibly infinite-dimensional) feature map $\phi$, and the dual formulation requires only inner products, never the explicit map.
- **D ✗** — The number of support vectors depends on the data and the margin; it is typically many more than 2.
- **E ✗** — A, B, C are correct.

---

**Question (14)** [Week 8]

A scree plot for a dataset with $p = 100$ features shows eigenvalues that drop sharply after component 3 and then level off into a near-flat "elbow." Which combination of selection rules is MOST consistent and appropriate?

A. Choose $K = 100$ components to retain all variance.
B. Choose $K = 3$ based on the elbow in the scree plot, and verify by checking that the cumulative variance explained exceeds 80–90%.
C. Always choose $K = 1$ because the first component explains the most variance.
D. Use $K$ such that each retained eigenvalue exceeds 1.0 (Kaiser criterion), regardless of the scree plot.
E. None of the above.

#### Answer: **B**

- **A ✗** — Retaining all 100 components keeps all noise; dimensionality reduction would serve no purpose.
- **B ✓** — The scree plot elbow is the most widely used visual heuristic for PCA component selection. Cross-referencing with a cumulative variance threshold (e.g., 80–90%) provides a quantitative check. Together these give a robust selection.
- **C ✗** — Selecting only 1 component discards potentially important structure; there is no justification for $K=1$ in general.
- **D ✗** — The Kaiser criterion (eigenvalue $> 1$) is an ad-hoc rule for correlation matrices (standardized data) and can give very different numbers of components from the elbow; using it "regardless of the scree plot" ignores the clear visual evidence.
- **E ✗** — B is correct.

---

**Question (15)** [Week 8]

PLS (Partial Least Squares) and Ridge regression are both shrinkage methods in the sense that they produce coefficient vectors with smaller norm than OLS. Which statement best identifies a scenario where Ridge outperforms PLS?

A. Ridge outperforms PLS when the predictors relevant to $y$ lie along directions of high variance in $X$.
B. Ridge outperforms PLS when the predictors relevant to $y$ lie along directions of low variance in $X$ (i.e., the important signal is in small-eigenvalue directions of $X^TX$).
C. Ridge always outperforms PLS on high-dimensional data.
D. PLS always outperforms Ridge because PLS additionally uses $y$ information when selecting directions.
E. None of the above.

#### Answer: **B**

- **A ✗** — When important directions have high variance, PLS will find them in its first components (it seeks directions of high covariance with $y$, which aligns with high-variance directions when those are also relevant). Both Ridge and PLS would do well, but PLS has the advantage here.
- **B ✓** — PLS constructs components that maximize covariance with $y$, but it inherits the principal component structure of $X$ if not corrected. Ridge shrinks all eigendirections of $X^TX$, giving less shrinkage to low-eigenvalue directions (which it still uses). When the true signal lies in small-eigenvalue directions of $X$ (e.g., weak but predictive features), Ridge can exploit them while PLS may miss them in early components.
- **C ✗** — There is no universal dominance; the relative performance depends on the alignment between signal and $X$ structure.
- **D ✗** — PLS uses $y$ information but this is advantageous only when the relevant $X$-directions are high-variance; when they are not, using $y$ in direction selection doesn't help.
- **E ✗** — B is correct.

---

**Question (16)** [Week 9]

A researcher tries to find the optimal number of clusters $K$ by minimizing within-cluster sum of squares (WCSS). Which statement is most accurate?

A. WCSS is a reliable criterion for choosing $K$ because it decreases monotonically and has a unique minimum at the true $K$.
B. WCSS always decreases as $K$ increases (reaching zero at $K = N$), so minimizing WCSS alone always selects $K = N$; other criteria like the gap statistic or silhouette width are needed.
C. The silhouette width criterion always agrees with the gap statistic when the true clusters are well-separated.
D. If class labels are available, internal validation measures (WCSS, silhouette) are always preferred over external measures like the Rand index.
E. None of the above.

#### Answer: **B**

- **A ✗** — WCSS has no unique interior minimum; it is strictly decreasing, so minimizing it blindly leads to $K = N$.
- **B ✓** — WCSS is monotonically decreasing in $K$ and reaches zero when every observation is its own cluster. To find a meaningful $K$, the gap statistic compares WCSS to a null reference distribution, and silhouette width measures cohesion vs. separation. Both provide genuine minima/maxima at the optimal $K$.
- **C ✗** — Silhouette and gap statistic can disagree, especially when clusters overlap, have different densities, or the data is noisy.
- **D ✗** — When true labels are available, external validation (Rand index, Fowlkes-Mallows, NMI) is generally superior because it directly measures alignment with ground truth, which is the true goal.
- **E ✗** — B is correct.

---

**Question (17)** [Week 10]

The universal approximation theorem states that a single hidden layer feedforward network with a sufficient number of neurons can approximate any continuous function on a compact domain to arbitrary accuracy. Which statement correctly identifies a limitation of this theorem and why depth (many layers) is preferred in practice?

A. The theorem is false; neural networks cannot approximate non-polynomial functions.
B. The theorem guarantees approximation but may require exponentially many neurons in the hidden layer; depth provides compositional structure that can represent certain function classes exponentially more efficiently with fewer parameters.
C. Depth only helps for image recognition; for tabular data, a single hidden layer is always optimal.
D. The theorem implies that adding more layers beyond one is redundant and only wastes computation.
E. None of the above.

#### Answer: **B**

- **A ✗** — The theorem is well-established (Cybenko 1989, Hornik 1991); it applies to continuous activation functions and continuous target functions on compact sets.
- **B ✓** — The theorem is existential, not constructive: the required width can be exponential in problem dimension. Deep networks can represent hierarchical compositional functions (e.g., image features: edges → textures → objects) with polynomially many parameters, which single hidden-layer networks cannot achieve without exponential width.
- **C ✗** — Depth helps across problem types; the advantage is not limited to image recognition.
- **D ✗** — This directly contradicts the theorem's limitation. The theorem says one layer suffices in principle but says nothing about efficiency; in practice deeper networks generalize better.
- **E ✗** — B is correct.

---

**Question (18)** [Week 11]

A document-term matrix $V \in \mathbb{R}^{n \times p}_{\geq 0}$ (rows = documents, columns = words, entries = word counts) is factored by NMF as $V \approx WH$ with $W \in \mathbb{R}^{n \times r}_{\geq 0}$, $H \in \mathbb{R}^{r \times p}_{\geq 0}$. Which statements are correct? Select ALL that apply.

A. NMF is appropriate here because word counts are non-negative, and the non-negativity constraints on $W$ and $H$ produce additive, parts-based components interpretable as topics.
B. PCA on the same matrix would produce components with both positive and negative loadings, which are harder to interpret as topics because negative word weights have no natural meaning in a count context.
C. NMF is unique (up to permutation and scaling of components) for any non-negative data matrix, just as PCA components are unique (up to sign).
D. The rank-$r$ NMF minimizes $\|V - WH\|_F^2$ subject to non-negativity, and the solution is found by closed-form eigendecomposition analogous to PCA.
E. None of the above.

#### Answer: **A, B**

- **A ✓** — Non-negativity ensures each document is a non-negative mixture of topics and each topic is a non-negative combination of words, yielding the "parts-based" interpretation (Lee & Seung, 1999).
- **B ✓** — PCA allows negative loadings (anti-topics), meaning it can represent a document as "full of topic 1 minus topic 2," which is uninterpretable for word count data.
- **C ✗** — NMF is generally NOT unique. For any invertible $Q$ with $WQ^{-1} \geq 0$ and $QH \geq 0$, the pair $(WQ^{-1})(QH)$ is an equally valid factorization. The Donoho–Stodden separability result is a highly restrictive sufficient condition, not a general property of NMF.
- **D ✗** — NMF has no closed-form solution; it is typically solved by alternating least squares (ALS) or multiplicative update rules, both iterative. The non-negativity constraint breaks the eigendecomposition structure.
- **E ✗** — A, B, C are correct.

---

**Question (19)** [Week 12]

The PARAFAC (CP) ALS algorithm updates factor matrix $A$ (mode-1) via $A \leftarrow X_{(1)}(C \odot B)\bigl((C^TC) \ast (B^TB)\bigr)^{-1}$, where $\odot$ denotes the Khatri-Rao product and $\ast$ denotes the Hadamard (element-wise) product. Which statements are correct? Select ALL that apply.

A. The Khatri-Rao product $C \odot B$ of matrices $C \in \mathbb{R}^{K \times R}$ and $B \in \mathbb{R}^{J \times R}$ is a $(KJ) \times R$ matrix formed by column-wise Kronecker products: the $r$-th column of $C \odot B$ is $c_r \otimes b_r$.
B. The term $(C^TC) \ast (B^TB)$ appears because it equals $(C \odot B)^T(C \odot B)$, making the update a least squares solution for $A$.
C. ALS is iterated until convergence because the update for each factor matrix is the global optimum of the full loss $\|X - \sum_r a_r \circ b_r \circ c_r\|^2$.
D. PARAFAC is unique (under mild conditions) even though the components are not constrained to be orthogonal, unlike PCA.
E. None of the above.

#### Answer: **A, B, D**

- **A ✓** — The Khatri-Rao product is exactly the column-wise Kronecker product: $(C \odot B)_{:,r} = c_r \otimes b_r$, giving a $(KJ) \times R$ matrix used to unfold the tensor update.
- **B ✓** — Since $(C \odot B)^T(C \odot B) = (C^TC) \ast (B^TB)$ (Hadamard product of Gram matrices), the mode-1 update is the least squares solution $A = X_{(1)}(C \odot B)[(C\odot B)^T(C\odot B)]^{-1}$.
- **C ✗** — Each ALS step minimizes the loss over one factor while fixing the others, which is a conditional optimum, not the global optimum. The full problem is non-convex and ALS only guarantees convergence to a stationary point (possibly a local minimum).
- **D ✓** — Kruskal's uniqueness theorem: PARAFAC is essentially unique (up to permutation and scaling of rank-1 components) when $k_A + k_B + k_C \geq 2R + 2$, where $k_X$ is the Kruskal rank of factor matrix $X$. PCA requires orthogonality for uniqueness; PARAFAC achieves uniqueness from the multi-linearity of the tensor structure alone.
- **E ✗** — A, B, D are correct.

---

**Question (20)** [Week 11 / Week 12]

A researcher applies three different unsupervised decomposition methods to a dataset and needs to choose the number of components $R$ (or clusters $K$) for each. Match each method to the correct model-selection criterion and its key limitation.

- (i) PARAFAC tensor decomposition
- (ii) Gaussian Mixture Model (GMM)
- (iii) PCA

A. (i) CORCONDIA (core consistency diagnostic): values near 100% indicate a Tucker-like residual core is nearly superdiagonal, supporting the PARAFAC model; may fail for noisy data or when components are correlated.
B. (ii) BIC: penalizes log-likelihood by $k\log N$ (where $k$ = number of free parameters); selects the GMM order minimizing BIC; may fail when the Gaussian assumption is badly violated.
C. (iii) Cumulative explained variance or cross-validated reconstruction error: straightforward but has no absolute threshold — the choice of 80% vs 95% is arbitrary.
D. (i) Split-half FMS (Faber-Meyers-Smilde): splits the data randomly and compares factor matrices between halves; high FMS ($\approx R$, where $R$ is the number of components) means stable components.
E. None of the above.

#### Answer: **A, B, C, D**

- **A ✓** — CORCONDIA computes $g = \|T - I_{\text{super}}\|^2 / \|I_{\text{super}}\|^2$ where $T$ is the Tucker core and $I_{\text{super}}$ is the superdiagonal; values near 100% indicate the PARAFAC model is appropriate at the chosen $R$. Limitation: noisy data or correlated components can cause CORCONDIA to drop even for the correct $R$.
- **B ✓** — BIC = $-2\ell + k\log N$ where $k$ is the parameter count of the GMM (means, covariance parameters, mixing weights). It selects the model with the lowest BIC. It fails when Gaussianity is violated (e.g., heavy-tailed or multi-modal clusters within one component).
- **C ✓** — PCA's explained-variance curve is purely data-driven and monotone in $R$; the threshold for "enough" variance is problem-dependent and arbitrary. Cross-validation on reconstruction error provides a less arbitrary alternative.
- **D ✓** — Split-half FMS is an alternative/complementary criterion for PARAFAC: it assesses reproducibility (stability) rather than model fit, providing independent evidence for the right $R$.
- **E ✗** — A, B, C, D are all correct.

---

## Open Questions

---

### Q21 (20 points) — Gradient Boosting and the Forward Stagewise Framework

**Part (a) — The General Forward Stagewise Algorithm [5 pts]**

State the general additive model and describe the forward stagewise fitting procedure.

**Part (b) — Squared Error Loss and L2Boost [5 pts]**

For squared error loss $L(y, f) = \frac{1}{2}(y-f)^2$, derive the pseudo-residual and show the connection to L2Boost.

**Part (c) — Exponential Loss and AdaBoost [5 pts]**

For exponential loss with $y \in \{-1,+1\}$, show that forward stagewise fitting recovers AdaBoost. Interpret $\alpha_m$ geometrically and explain the sample-weighting mechanism.

**Part (d) — Regularization in Gradient Boosting [5 pts]**

Describe three regularization strategies for gradient boosting: shrinkage, subsampling, and tree depth. Explain what each controls and why shrinkage $\nu < 1$ requires more trees but often generalizes better.

---

### Q22 (20 points) — Wearables CV Design: Feature Extraction and Leakage

**Dataset:** 16 subjects × 3 activities × 4 seasons = 192 observations. Each observation is a 5-minute wearable recording. Features (mean, variance, spectral entropy) are extracted from the raw signal before classification.

**Part (a) — Pre-split Normalization as Data Leakage [5 pts]**

The researcher normalizes all 192 feature vectors to zero mean and unit variance BEFORE splitting into folds. Explain precisely why this is data leakage and what should be done instead.

**Part (b) — LOIO CV: Counting and EPE Formula [5 pts]**

With Leave-One-Individual-Out (LOIO) CV and 16 folds: in fold $k$ (subject $k$ held out), how many training observations are there? How many test observations? Write the EPE estimate formula.

**Part (c) — Inner CV for Hyperparameter Tuning [5 pts]**

Inside each training fold (15 subjects × 12 observations = 180 observations), the researcher wants to tune a hyperparameter. Propose a correct inner CV scheme. Must the inner folds also respect the subject structure? Explain.

**Part (d) — Personalized vs. Generalized Models [5 pts]**

Compare LOSO CV (within one subject) and LOIO CV (across subjects). Why do these answer fundamentally different scientific questions? In which real-world application would you deploy each?

---

### Solution

---

**Part (a) — The General Forward Stagewise Algorithm**

The general additive model expresses the prediction function as a sum of $M$ base learners (weak learners):

$$f_M(x) = \sum_{m=1}^{M} \beta_m \, b(x;\, \gamma_m)$$

where $b(x;\gamma_m)$ is a basis function (e.g., a regression tree) parameterized by $\gamma_m$ (e.g., tree structure and leaf values), and $\beta_m \geq 0$ is a scalar weight. The goal is to minimize the empirical loss:

$$\min_{\{\beta_m,\gamma_m\}_{m=1}^M} \sum_{i=1}^N L\!\left(y_i,\, f_M(x_i)\right).$$

Jointly optimizing over all $M$ pairs $(\beta_m, \gamma_m)$ is computationally intractable. The **forward stagewise algorithm** is a greedy approximation: given the current approximation $f_{m-1}$, at step $m$ solve only:

$$(\hat\beta_m, \hat\gamma_m) = \arg\min_{\beta, \gamma} \sum_{i=1}^N L\!\left(y_i,\, f_{m-1}(x_i) + \beta \, b(x_i;\gamma)\right),$$

keeping all previously fitted terms $f_{m-1}$ fixed. After finding $(\hat\beta_m, \hat\gamma_m)$ we set:

$$f_m(x) = f_{m-1}(x) + \hat\beta_m \, b(x;\hat\gamma_m).$$

This greedy stage-by-stage approach trades global optimality for computational feasibility; importantly, the connection to functional gradient descent (Friedman 2001) allows it to be applied to any differentiable loss, giving rise to gradient boosting.

---

**Part (b) — Squared Error Loss and L2Boost**

For the squared error loss $L(y, f) = \tfrac{1}{2}(y-f)^2$, the step $m$ optimization becomes:

$$\min_{\beta,\gamma} \sum_{i=1}^N \tfrac{1}{2}\!\left(y_i - f_{m-1}(x_i) - \beta \, b(x_i;\gamma)\right)^2.$$

Define the **pseudo-residuals** (negative gradient of the loss with respect to the current fit):

$$r_{im} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = f_{m-1}} = y_i - f_{m-1}(x_i).$$

Under squared error loss, the pseudo-residual is exactly the ordinary residual $r_{im} = y_i - f_{m-1}(x_i)$.

The optimization over $\gamma$ thus reduces to fitting a tree $b(x;\gamma)$ to minimize $\sum_i (r_{im} - \beta \, b(x_i;\gamma))^2$, i.e., regressing the residuals on $x$. Once the tree structure $\hat\gamma_m$ is found, the optimal $\hat\beta_m$ is the least-squares scalar. This is exactly **L2Boost**: at each step, fit a weak learner to the current residuals and add it (with a scalar weight) to the ensemble. The result is that gradient boosting with squared error loss performs functional gradient descent in function space, taking a step of length $\hat\beta_m$ in the direction of the current residual vector.

---

**Part (c) — Exponential Loss and AdaBoost**

For binary classification with $y_i \in \{-1, +1\}$, the exponential loss is:

$$L(y, f) = e^{-yf}.$$

The pseudo-residual (negative gradient) is $r_{im} = y_i \, e^{-y_i f_{m-1}(x_i)}$.

**Recovering AdaBoost.** Define sample weights $w_i^{(m)} = e^{-y_i f_{m-1}(x_i)}$. The forward stagewise step becomes:

$$\min_{\beta, \gamma} \sum_{i=1}^N w_i^{(m)} \exp\!\left(-y_i \, \beta \, b(x_i;\gamma)\right),$$

which (after expanding and optimizing over a binary classifier $b \in \{-1, +1\}$) yields:

$$\hat\alpha_m = \frac{1}{2}\log\frac{1 - \text{err}_m}{\text{err}_m}, \qquad \text{err}_m = \frac{\sum_i w_i^{(m)} \mathbf{1}[y_i \neq b(x_i;\hat\gamma_m)]}{\sum_i w_i^{(m)}}.$$

This is exactly the AdaBoost update. **Geometric interpretation of $\alpha_m$:** $\alpha_m$ is the log-odds of a correct vote by the $m$-th weak learner. When $\text{err}_m \to 0$ (nearly perfect learner), $\alpha_m \to +\infty$, giving that learner an arbitrarily large vote. When $\text{err}_m = 0.5$ (random guessing), $\alpha_m = 0$: the learner contributes nothing. It measures the "confidence" or decision authority granted to the $m$-th weak learner.

**Sample weight concentration on hard examples.** After updating the classifier, weights are updated as $w_i^{(m+1)} \propto e^{-y_i f_m(x_i)}$. Points that have been consistently misclassified have small $y_i f_m(x_i)$ (or negative), so their weight increases exponentially. Points that have been correctly classified accumulate large $y_i f_m(x_i)$ and their weight decreases. The algorithm thus concentrates attention on the most difficult examples in successive rounds.

---

**Part (d) — Regularization in Gradient Boosting**

Three complementary regularization mechanisms:

**1. Shrinkage (learning rate $\nu$).** The update becomes $f_m = f_{m-1} + \nu \cdot T_m$, where $\nu \in (0, 1]$. Shrinkage reduces the contribution of each new tree, slowing down the learning and effectively truncating the functional gradient step. This is analogous to using a small step size in steepest descent: it prevents over-correction on noisy gradient estimates. The consequence is that more trees $M$ are needed to achieve the same training error, but the model explores a smoother path through function space and is less likely to overfit. Empirically, $\nu \in [0.01, 0.1]$ with large $M$ (chosen by early stopping via a held-out validation set) consistently outperforms $\nu = 1$ with small $M$.

**2. Subsampling (stochastic gradient boosting).** At each step, a random subsample (without replacement) of fraction $\eta \in (0.5, 1)$ of the training data is drawn, and $T_m$ is fit to only that subsample. This introduces randomness analogous to stochastic gradient descent: it reduces variance (each tree sees a different subset), can escape shallow local minima, and dramatically reduces computation. The random subsampling also decorrelates successive trees, analogous to the mechanism in Random Forests.

**3. Tree depth (interaction order).** The maximum depth $d$ of each tree controls the order of variable interactions. Depth-1 trees (stumps) fit only main effects; depth-2 trees fit pairwise interactions; depth $d$ fits up to $d$-way interactions. Smaller $d$ acts as a strong regularizer: it restricts the model class, reduces variance, and speeds training. For many practical datasets, $d \in \{3, 4, 5\}$ provides a good bias-variance tradeoff.

**Why small $\nu$ requires more trees but generalizes better.** Each tree corrects only a fraction $\nu$ of the residual signal. With $\nu = 1$, the model greedily makes full corrections and can overfit noise in early trees; later trees then attempt to correct errors introduced by the noise-fitting of earlier trees, amplifying instability. With $\nu < 1$, the residuals are corrected gradually, and the contribution of any single noisy tree is damped. The optimal $M$ is found via early stopping on a validation set: as $M$ grows the training error always decreases, but the validation error exhibits a U-shape; early stopping at the minimum of the validation error curve gives the optimal $M^*$. The smaller $\nu$ is, the larger $M^*$ will be, but the test error at $M^*$ is typically lower.

---

**Q22 Solution**

---

**Part (a) — Pre-split Normalization as Data Leakage**

**Why it is leakage.** When the researcher computes the global mean $\bar{\mu}_j = \frac{1}{192}\sum_{n=1}^{192} x_{nj}$ and global standard deviation $\hat\sigma_j$ over all 192 observations and then normalizes as $\tilde{x}_{nj} = (x_{nj} - \bar\mu_j)/\hat\sigma_j$, the normalized test-fold observations $\tilde{x}_{test}$ have been shifted and scaled using statistics that include the test-fold data itself. This means the classifier indirectly "sees" global distributional information about the test subjects before making predictions. In particular, if test subjects have systematically different signal amplitudes (due to sensor placement, body composition, seasonal variation), the global normalization removes this between-subject variance and makes the test distribution artificially more similar to training — a form of information leakage that produces optimistically biased generalization estimates.

**What to do instead.** Normalization must be performed inside each cross-validation fold:

1. For fold $k$ (subject $k$ held out), compute $\bar\mu_j^{(-k)}$ and $\hat\sigma_j^{(-k)}$ using only the 180 training observations (15 subjects × 12 observations).
2. Apply these training-set statistics to normalize both the training features and the held-out test features: $\tilde{x}_{nj} = (x_{nj} - \bar\mu_j^{(-k)})/\hat\sigma_j^{(-k)}$.
3. Fit the classifier on normalized training data; evaluate on normalized test data using training statistics.

This ensures the test-fold subject contributes no information to the normalization, preserving the integrity of the generalization estimate.

---

**Part (b) — LOIO CV: Counting and EPE Formula**

With 16 subjects × 12 observations each (= 192 total observations), in fold $k$ (subject $k$ held out):

- **Training observations:** $16 - 1 = 15$ subjects $\times$ 12 observations $= \mathbf{180}$ training observations.
- **Test observations:** 1 subject $\times$ 12 observations $= \mathbf{12}$ test observations (3 activities × 4 seasons).

The EPE (Expected Prediction Error) estimate is:

$$\widehat{\text{EPE}}_\text{LOIO} = \frac{1}{16}\sum_{k=1}^{16} \frac{1}{12}\sum_{i \in \mathcal{I}_k} L\!\left(y_i, \hat{f}^{(-k)}(x_i)\right)$$

where $\mathcal{I}_k$ is the index set of the 12 observations for subject $k$, and $\hat{f}^{(-k)}$ is the classifier trained on all subjects except $k$. Using 0-1 loss (misclassification rate):

$$\widehat{\text{EPE}}_\text{LOIO} = \frac{1}{16}\sum_{k=1}^{16} \frac{1}{12}\sum_{i \in \mathcal{I}_k} \mathbf{1}\!\left[\hat{f}^{(-k)}(x_i) \neq y_i\right].$$

---

**Part (c) — Inner CV for Hyperparameter Tuning**

**Proposed inner CV scheme.** Inside each outer training fold (15 subjects × 12 observations = 180 observations), use **Leave-One-Subject-Out inner CV** with 15 inner folds:

- Inner fold $j$ ($j = 1, \ldots, 15$): hold out subject $j$ from the 15 training subjects; train on the remaining 14 subjects (168 observations) with a candidate hyperparameter value $\lambda$; evaluate on subject $j$'s 12 observations.
- Select $\hat\lambda = \arg\min_\lambda \frac{1}{15}\sum_{j=1}^{15} \text{error}_j(\lambda)$.
- Refit the classifier on all 180 training observations using $\hat\lambda$; use this model for the outer test prediction.

**Must inner folds respect the subject structure? Yes.** The reason is the same as for the outer loop: the IID assumption is violated because observations from the same subject share physiological characteristics, sensor calibration, and behavioral patterns. If the inner CV randomly assigns observations to folds, a subject could appear in both inner training and inner validation, inflating estimated performance and leading to hyperparameter choices that are tuned to within-subject patterns rather than cross-subject generalization. The inner CV must therefore split at the subject level to correctly estimate the performance of the hyperparameter choice for the truly unseen subject in the outer fold.

---

**Part (d) — Personalized vs. Generalized Models**

**Personalized model (LOSO within one subject).**
Leave-One-Season-Out CV uses 4 folds within a single subject (3 activities × 4 seasons = 12 observations; hold out 1 season = 3 observations, train on 9). The EPE estimate answers: *"Given a labeled history for this specific person, how well can we predict their activity during a new season?"* This is a **subject-specific** or **personalized** generalization question. The model is calibrated to an individual's physiology and labeled observations.

**Generalized model (LOIO across subjects).**
The 16-fold LOIO CV answers: *"If we have labeled data from 15 subjects, how well does the model generalize to a completely new, unseen individual with no prior labeled data?"* This is a **cross-subject** or **generalized** generalization question. It is fundamentally harder: the model must capture population-level patterns robust to inter-individual variability.

**Fundamental difference.** The two estimands differ in what constitutes "new data":
- LOSO: new data = new time period for the same person (seasonal variation within individual).
- LOIO: new data = new person with no historical labels (individual variation across population).

These have different practical uses and different error magnitudes. LOIO error is almost always higher than LOSO error because it tests transfer to an unknown individual.

**Deployment scenarios:**
- **Personalized model (LOSO):** Deploy in wearable health apps where each user onboards with a calibration period (e.g., 9 labeled sessions). The model is retrained per user and used to classify their activity in future sessions. Example: a personal fitness tracker that adapts to an individual's gait and biosignal baseline.
- **Generalized model (LOIO):** Deploy in population screening tools or medical devices used on new patients without prior calibration. Example: a clinical wearable that classifies activity or detects anomalies immediately upon first use on any patient, without requiring labeled data collection for that individual first.
