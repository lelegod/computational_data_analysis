# Practice Set 4 — CDA 02582 (Combined Questions + Solutions)

**Format:** 20 multiple-choice + 2 open questions
**Scoring:** MC: +1 (correct), −0.25 (incorrect), 0 (unanswered)
**Duration:** 4 hours

---

## Multiple Choice

---

**Question (1)** [Week 1]
In K-nearest-neighbour regression, what happens to model bias and variance as $K$ increases from 1 to $N$?

A. Bias decreases; variance decreases.
B. Bias increases; variance decreases.
C. Bias decreases; variance increases.
D. Both bias and variance increase.
E. None of the above.

#### Answer: **B**

- **A ✗** — Increasing $K$ averages over more neighbours, smoothing the fit and pulling predictions toward the global mean, which *increases* bias rather than decreasing it.
- **B ✓** — Large $K$ produces a smoother, more biased fit (underfitting) but averages away idiosyncratic noise, reducing variance. At $K = N$ the model predicts the global mean for every point — maximum bias, minimum variance.
- **C ✗** — This would correspond to *decreasing* $K$, not increasing it.
- **D ✗** — Variance decreases as $K$ grows because more neighbours contribute to each estimate.
- **E ✗** — B is correct.

---

**Question (2)** [Week 1]
Which of the following correctly describes the geometric and Lagrangian duality connection for Ridge regression?

A. The Ridge constraint region is an $\ell_1$ ball; the penalty parameter $\lambda$ is unrelated to the constraint radius.
B. The Ridge constraint region is an $\ell_2$ ball of radius $t$; a larger $\lambda$ corresponds to a *smaller* $t$.
C. The Ridge constraint region is an $\ell_2$ ball of radius $t$; a larger $\lambda$ corresponds to a *larger* $t$.
D. Ridge has no closed-form solution because the constraint region lacks a corner.
E. None of the above.

#### Answer: **B**

- **A ✗** — The $\ell_1$ ball is the Lasso constraint. Ridge uses an $\ell_2$ (sphere) constraint, and $\lambda$ and $t$ are related via duality.
- **B ✓** — The constrained form $\min_\beta \|y - X\beta\|^2$ s.t. $\|\beta\|^2 \le t$ and the penalised form $\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|^2$ are equivalent by Lagrangian duality. A large penalty $\lambda$ forces a tight constraint, so $t$ is small.
- **C ✗** — The relationship is inverse: larger $\lambda$ → stronger shrinkage → smaller allowed $\|\beta\|^2$ → smaller $t$.
- **D ✗** — Ridge has the closed-form solution $\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1}X^Ty$; the smooth $\ell_2$ ball is precisely what makes this possible.
- **E ✗** — B is correct.

---

**Question (3)** [Week 2]
Which statement best explains why 5- or 10-fold CV is often preferred over Leave-One-Out CV (LOOCV), even though LOOCV has lower bias?

A. LOOCV is computationally cheaper than $K$-fold CV.
B. LOOCV training sets overlap almost entirely, causing high correlation between fold estimates and therefore high variance of the CV estimate.
C. LOOCV underestimates test error because each fold uses all $N$ observations.
D. $K$-fold CV with $K < N$ always gives a lower variance estimate regardless of data set size.
E. None of the above.

#### Answer: **B**

- **A ✗** — LOOCV requires fitting $N$ models vs. $K \ll N$ for $K$-fold; it is typically more expensive, not cheaper.
- **B ✓** — With LOOCV, each fold's training set differs from another by only one observation, so the $N$ fitted models are almost identical. Their errors are highly correlated, and the variance of a sum of correlated quantities can be large even when individual terms are unbiased.
- **C ✗** — LOOCV uses $N-1$ training observations per fold (not all $N$), and it is approximately unbiased, not pessimistically biased.
- **D ✗** — This is too strong: the variance advantage of $K$-fold holds specifically because the correlation between fold estimates is lower, not unconditionally.
- **E ✗** — B is correct.

---

**Question (4)** [Week 2]
The 1-SE rule in model selection via cross-validation selects:

A. The model with the absolute minimum cross-validated error.
B. The most complex model whose CV error lies within one standard error of the minimum.
C. The most regularized (simplest) model whose CV error lies within one standard error of the minimum.
D. The model whose CV error standard error is smallest.
E. None of the above.

#### Answer: **C**

- **A ✗** — That is pure minimum selection, ignoring the uncertainty in the CV estimate.
- **B ✗** — The 1-SE rule moves in the direction of *more* regularization (simpler models), not more complexity. Choosing the most complex model within the band would defeat the purpose.
- **C ✓** — We compute $\text{CV}^* + \hat{\text{SE}}$ and select the most regularized model still within this bound. The rationale: since models within one SE of the minimum are statistically indistinguishable in performance, we prefer the simplest (most parsimonious) one.
- **D ✗** — Minimizing the SE of the CV estimate is a different criterion and is not the 1-SE rule.
- **E ✗** — C is correct.

---

**Question (5)** [Week 3]
In coordinate descent for the Lasso, the soft-thresholding update for coefficient $j$ (holding all others fixed) is $\hat{\beta}_j = \text{sign}(z_j)\max(|z_j| - \lambda, 0)$, where $z_j$ is the OLS estimate of $\beta_j$ on the partial residual. Which of the following best describes the geometric effect of this update?

A. It shrinks all coefficients by the same absolute amount $\lambda$ toward zero, without setting any to exactly zero.
B. It shrinks coefficients toward zero by $\lambda$ and sets those with $|z_j| \le \lambda$ exactly to zero.
C. It shrinks coefficients proportionally (by factor $1/(1+\lambda)$), as in Ridge.
D. It applies a hard threshold: $\hat{\beta}_j = z_j$ if $|z_j| > \lambda$, else $\hat{\beta}_j = z_j/2$.
E. None of the above.

#### Answer: **B**

- **A ✗** — Close, but incomplete: the formula *does* set coefficients to zero when $|z_j| \le \lambda$, which A explicitly denies.
- **B ✓** — The soft-threshold operator translates the OLS partial estimate toward zero by $\lambda$. When $|z_j| \le \lambda$ the result is exactly zero, producing the sparsity property that distinguishes Lasso from Ridge.
- **C ✗** — Proportional shrinkage toward zero without exact zeros is the Ridge update, which arises from the $\ell_2$ penalty.
- **D ✗** — This describes a hard threshold with an ad-hoc modification, not soft thresholding.
- **E ✗** — B is correct.

---

**Question (6)** [Week 3]
Which of the following correctly describes a key difference between the LARS algorithm and greedy forward stepwise selection?

A. LARS adds the predictor most correlated with the residual and immediately fits its full OLS coefficient, whereas forward stepwise moves along the equiangular direction.
B. LARS moves the active-set coefficients in the equiangular direction (making equal angles with all active predictors' residual correlations), whereas forward stepwise steps the single most-correlated predictor to its full OLS fit at each stage.
C. LARS and forward stepwise are identical for orthogonal predictors.
D. LARS requires more computation than forward stepwise because it recomputes all OLS fits at each step.
E. None of the above.

#### Answer: **B**

- **A ✗** — This reverses the descriptions: LARS uses the equiangular direction; greedy forward stepwise fits full OLS to the selected predictor.
- **B ✓** — At each stage, LARS identifies which predictors are tied in correlation with the current residual, then advances the coefficient vector in the least-angle (equiangular) direction until another predictor ties. This is smoother than stepwise's discrete predictor-by-predictor jumps.
- **C ✗** — For orthogonal predictors LARS reduces to forward stepwise in terms of predictor ordering, but the *mechanism* (equiangular direction vs. full OLS step) is still different.
- **D ✗** — LARS is computationally efficient; its cost is comparable to a single OLS fit, making it cheaper than running $p$ OLS fits for forward stepwise.
- **E ✗** — B is correct.

---

**Question (7)** [Week 4]
Consider QDA vs. LDA on a classification problem. Select all true statements.

A. QDA estimates a separate covariance matrix $\Sigma_k$ per class, giving it lower bias but higher variance than LDA.
B. LDA pools all class covariance matrices into one $\Sigma$, which is appropriate when class covariances are approximately equal.
C. QDA is always preferred when sample size is large because it is a strictly more general model.
D. When $p$ is large relative to $n$, QDA's $p(p+1)/2$ parameters per class can make it unstable.
E. None of the above.

#### Answer: **A, B, D**

- **A ✓** — Estimating $K$ separate covariance matrices instead of one pooled matrix introduces $K$ times more free parameters, reducing bias on truly different-covariance problems but inflating variance (estimation error).
- **B ✓** — LDA's equal-covariance assumption is the defining constraint that allows pooling, and when it holds it reduces the number of parameters to estimate, lowering variance.
- **C ✗** — More general does not mean always preferred; if sample size is limited, the extra parameters of QDA may introduce more estimation variance than they reduce in bias, leading to worse generalization.
- **D ✓** — Each QDA class covariance matrix has $p(p+1)/2$ free parameters; with many classes and moderate $n$ this quickly leads to singular or near-singular estimates.
- **E ✗** — A, B, and D are all correct.

---

**Question (8)** [Week 4]
Regularized Discriminant Analysis (RDA) interpolates between QDA and a spherical model using two parameters $\alpha$ and $\gamma$. Which sequence of models does increasing $\alpha$ from 0 to 1 (with $\gamma = 0$) produce?

A. Spherical → diagonal → LDA → QDA.
B. QDA → LDA (by shrinking each class covariance toward the pooled covariance).
C. LDA → QDA (by expanding the pooled covariance to allow class-specific differences).
D. LDA → diagonal LDA → spherical LDA.
E. None of the above.

#### Answer: **B**

- **A ✗** — This describes the path along $\gamma$, not $\alpha$. The $\gamma$ parameter shrinks toward diagonal; $\alpha$ interpolates between QDA ($\alpha = 0$) and LDA ($\alpha = 1$).
- **B ✓** — RDA forms $\hat{\Sigma}_k(\alpha) = (1-\alpha)\hat{\Sigma}_k + \alpha\hat{\Sigma}$. At $\alpha = 0$ we have class-specific covariances (QDA); at $\alpha = 1$ all classes share the pooled covariance (LDA).
- **C ✗** — The direction is reversed; increasing $\alpha$ moves *toward* LDA, not away from it.
- **D ✗** — That describes the $\gamma$ interpolation, which shrinks the (already pooled) $\hat{\Sigma}$ toward its diagonal.
- **E ✗** — B is correct.

---

**Question (9)** [Week 5]
In binary classification trees, why are Gini impurity and cross-entropy preferred over misclassification rate as splitting criteria?

A. Gini and cross-entropy are differentiable and can be minimized by gradient descent, unlike misclassification rate.
B. Gini and cross-entropy are more sensitive to changes in class probabilities near 0.5 than misclassification rate, detecting splits that modestly improve probability estimates even when they do not change the predicted class.
C. Gini and cross-entropy are bounded between 0 and 1, whereas misclassification rate can exceed 1.
D. Gini and cross-entropy penalize tree depth, while misclassification rate does not.
E. None of the above.

#### Answer: **B**

- **A ✗** — Classification trees use exhaustive search over split points, not gradient descent; differentiability is not the reason for preferring Gini/entropy.
- **B ✓** — Consider a node with $p = 0.51$ in class A: misclassification rate = 0.49, Gini = $2 \times 0.51 \times 0.49 \approx 0.4998$, entropy ≈ 0.9997. A split that changes $p$ to 0.7 and 0.3 reduces Gini and entropy substantially, but misclassification changes less and can be flat around 0.5, leading to missed opportunities for informative splits.
- **C ✗** — Misclassification rate is always in $[0, 0.5]$ for binary problems (taking the majority class); it does not exceed 1.
- **D ✗** — None of these three criteria directly penalize tree depth; depth is controlled by stopping rules or pruning.
- **E ✗** — B is correct.

---

**Question (10)** [Week 5]
Why does bagging benefit from using fully grown (unpruned) trees as base learners?

A. Unpruned trees are faster to train because no pruning step is required.
B. Pruning reduces a tree's variance; bagging is most effective when base learners have high variance, so unpruned trees give bagging more variance to average away.
C. Pruned trees have higher bias, making them unsuitable as base learners for any ensemble.
D. Bagging cannot reduce bias, so unpruned trees with low bias ensure the ensemble also has low bias.
E. None of the above.

#### Answer: **B, D**

- **A ✗** — Speed is not the statistical motivation; and in practice full trees can be expensive to grow.
- **B ✓** — Bagging's variance-reduction mechanism is averaging: $\text{Var}(\bar{T}) = \rho\sigma^2 + (1-\rho)\sigma^2/B$. When individual trees have small variance (because they were pruned), there is little variance left to average away, so the ensemble gains less.
- **C ✗** — Pruned trees are actually useful in many ensembles (e.g., boosting uses shallow trees). The statement that they are unsuitable for *any* ensemble is too strong.
- **D ✓** — Bagging operates by averaging predictions, which cannot reduce the average bias of the base learners. Using low-bias base learners (unpruned trees overfit, but they are approximately unbiased on average across bootstrap samples) ensures the aggregate retains low bias.
- **E ✗** — B and D are both correct.

---

**Question (11)** [Week 6]
Which of the following best describes how gradient boosting reduces prediction error sequentially?

A. Each new tree is fit to the original labels $y$ but with updated observation weights, like AdaBoost.
B. Each new tree fits the negative gradient of the loss with respect to the current ensemble's predictions — the pseudo-residuals — thereby sequentially reducing the loss in a steepest-descent fashion.
C. Gradient boosting averages independently trained deep trees, reducing variance while keeping bias constant.
D. Gradient boosting is identical to AdaBoost for squared-error loss.
E. None of the above.

#### Answer: **B**

- **A ✗** — Reweighting observations is the AdaBoost procedure. Gradient boosting fits to pseudo-residuals (negative gradient of the loss), which is a different mechanism.
- **B ✓** — Gradient boosting performs functional gradient descent: given current predictions $F_m(x)$, the pseudo-residuals $r_{im} = -\partial L(y_i, F(x_i))/\partial F(x_i)$ are the targets for the next tree. For squared-error loss $L = (y - F)^2/2$, the pseudo-residuals equal the ordinary residuals $y_i - F_m(x_i)$.
- **C ✗** — That describes bagging. Gradient boosting trains trees *sequentially*, not independently, and its primary mechanism is bias reduction, not variance reduction.
- **D ✗** — For squared-error loss the pseudo-residuals equal ordinary residuals, but AdaBoost reweights observations via exponential loss and the update mechanism is fundamentally different.
- **E ✗** — B is correct.

---

**Question (12)** [Week 6]
In a Random Forest, what is the role of the parameter $m$ (number of randomly selected features at each split)?

A. $m$ controls the depth of each tree; $m = 1$ gives stumps.
B. $m$ controls the amount of randomization in feature selection at each node. When $m = p$ (all features), Random Forest reduces to standard bagging of trees.
C. $m$ controls the number of bootstrap replications; the recommended value is $m = \sqrt{B}$.
D. Setting $m = \sqrt{p}$ for regression and $m = p/3$ for classification is recommended because regression problems have more predictors.
E. None of the above.

#### Answer: **B**

- **A ✗** — $m$ governs feature subsampling per split, not tree depth. Tree depth is controlled by minimum node size or max depth parameters.
- **B ✓** — At each split, only $m$ of the $p$ features are candidates, decorrelating the trees in the ensemble. When $m = p$ no feature is excluded, so all trees use the same candidate set — equivalent to standard bagging. The decorrelation introduced by $m < p$ further reduces the ensemble's variance.
- **C ✗** — $m$ is a feature count, unrelated to the number of bootstrap samples $B$.
- **D ✗** — The conventional defaults are $m = \sqrt{p}$ for *classification* and $m = p/3$ for *regression* — this option has them swapped.
- **E ✗** — B is correct.

---

**Question (13)** [Week 7]
Consider the polynomial kernel $K(x, x') = (1 + x^Tx')^d$ used in an SVM. Which of the following are true? (Select all that apply.)

A. The polynomial kernel implicitly maps inputs to a feature space of dimension $\binom{p+d}{d}$, without explicitly computing the mapping.
B. Increasing $d$ increases model complexity, risking overfitting on small datasets.
C. The RBF kernel $K(x,x') = \exp(-\|x-x'\|^2/2\sigma^2)$ corresponds to $d \to \infty$ in the polynomial kernel.
D. The polynomial kernel cannot be used in the SVM dual objective because the Gram matrix may not be positive semi-definite.
E. None of the above.

#### Answer: **A, B**

- **A ✓** — The polynomial kernel implicitly defines inner products in the expanded feature space spanned by all degree-$\le d$ monomials, whose dimension grows as $\binom{p+d}{d}$. The kernel trick computes this inner product without enumerating the basis.
- **B ✓** — Higher $d$ corresponds to a richer hypothesis class; with few training points, high-degree polynomials can memorise training data, increasing test error.
- **C ✗** — The RBF kernel corresponds to an *infinite*-dimensional feature space (it can be related to an infinite Taylor expansion), but this is not the same as $d \to \infty$ in the polynomial kernel. They are distinct kernel families.
- **D ✗** — The polynomial kernel (with the $1+$ form) is a valid positive semi-definite kernel (it satisfies Mercer's condition), so the Gram matrix is PSD.
- **E ✗** — A and B are correct.

---

**Question (14)** [Week 8]
Which of the following best describes what Partial Least Squares (PLS) maximises when constructing its first component $z_1 = X\phi_1$?

A. The variance of $z_1$ in $X$, subject to $\|\phi_1\| = 1$ (same objective as the first PCA component).
B. The covariance between $z_1 = X\phi_1$ and $y$, subject to $\|\phi_1\| = 1$.
C. The correlation between $z_1$ and $y$ divided by the variance of $z_1$.
D. The $R^2$ of regressing $y$ on $z_1$.
E. None of the above.

#### Answer: **B**

- **A ✗** — Maximising variance in $X$ is the PCA criterion, which ignores the response $y$. PLS uses $y$ to direct the component.
- **B ✓** — PLS finds $\phi_1$ that maximises $\text{Cov}(X\phi_1, y) = \phi_1^T X^T y$, subject to $\|\phi_1\|=1$. The solution is $\phi_1 \propto X^Ty$, directly reflecting each predictor's covariance with the response.
- **C ✗** — Maximising the correlation is CCA's criterion, not PLS. CCA also normalises by the variance of the component, whereas PLS does not.
- **D ✗** — $R^2$ from a simple regression of $y$ on $z_1$ is related to the squared correlation, which again corresponds to CCA rather than PLS.
- **E ✗** — B is correct.

---

**Question (15)** [Week 8]
In high dimensions ($p > n$), standard CCA faces a fundamental problem. Which of the following correctly identifies the problem and a remedy?

A. CCA requires the covariance matrices $\Sigma_{XX}$ and $\Sigma_{YY}$ to be invertible; when $p > n$ these matrices are rank-deficient. Regularized CCA replaces them with $\hat{\Sigma}_{XX} + r_1 I$ and $\hat{\Sigma}_{YY} + r_2 I$.
B. CCA fails because the cross-covariance matrix $\Sigma_{XY}$ is not square; regularization makes it square.
C. CCA gives trivial solutions because all canonical correlations equal 1; the fix is to add an $\ell_1$ penalty.
D. When $p > n$, CCA reduces to PCA because the leading eigenvalues dominate.
E. None of the above.

#### Answer: **A**

- **A ✓** — The CCA solution involves $\Sigma_{XX}^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}$. When $p > n$, the sample covariance matrix $\hat{\Sigma}_{XX}$ is rank-deficient (at most rank $n$) and hence singular. Ridge-type regularization $\hat{\Sigma}_{XX} + r_1 I$ ensures invertibility.
- **B ✗** — The cross-covariance $\Sigma_{XY}$ does not need to be square for CCA; the problem is singularity of the within-set covariances, not non-squareness of the cross-covariance.
- **C ✗** — In the degenerate case (no regularization, $p > n$), all canonical correlations can indeed inflate toward 1 because the sample covariance is singular. However, an $\ell_1$ penalty (sparse CCA) is a different approach from standard regularized CCA, and the stated reason is incomplete.
- **D ✗** — CCA and PCA have different objectives and do not coincide even in high dimensions.
- **E ✗** — A is correct.

---

**Question (16)** [Week 9]
In the EM algorithm for a Gaussian Mixture Model with $K$ components, which of the following correctly states both the E-step and a property of convergence?

A. E-step: compute hard assignments $z_{ik} = \mathbf{1}[k = \arg\max_j \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)]$; convergence to global maximum is guaranteed.
B. E-step: compute responsibilities $\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i;\mu_j,\Sigma_j)}$; the log-likelihood is non-decreasing at each iteration.
C. E-step: compute responsibilities as above; convergence to global maximum is guaranteed because EM always increases the likelihood.
D. E-step computes sufficient statistics; M-step maximises the marginal likelihood directly; this guarantees a globally optimal solution.
E. None of the above.

#### Answer: **B**

- **A ✗** — Hard assignments are used in the K-means algorithm, which can be viewed as a limiting case of EM with infinitely tight Gaussians ($\Sigma_k \to 0$). Full EM uses soft responsibilities.
- **B ✓** — The E-step computes the posterior probability that observation $i$ belongs to component $k$. The log-likelihood $\sum_i \log \sum_k \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)$ is non-decreasing at each EM iteration by Jensen's inequality, but convergence may be to a local maximum.
- **C ✗** — Non-decreasing likelihood does *not* guarantee convergence to a global maximum. GMM EM can converge to local maxima; initialisation (e.g., via K-means) is important.
- **D ✗** — The M-step maximises the *expected complete-data* log-likelihood (the $Q$ function), not the marginal likelihood directly, and there is no global optimality guarantee.
- **E ✗** — B is correct.

---

**Question (17)** [Week 10]
Why do sigmoid activations cause vanishing gradients in deep networks, and how does ReLU mitigate this?

A. Sigmoid saturates in $[0,1]$; its derivative $\sigma'(z) = \sigma(z)(1-\sigma(z)) \le 0.25$, so backpropagated gradients are multiplied by small values at each layer and decay exponentially with depth. ReLU has derivative 1 for positive inputs, avoiding this multiplicative decay.
B. Sigmoid produces negative outputs, causing sign alternations in the gradient. ReLU restricts outputs to $[0, \infty)$, keeping gradients positive.
C. Sigmoid's derivative is unbounded, causing gradients to explode. ReLU clips gradients at 1.
D. ReLU eliminates the vanishing gradient problem entirely because its derivative is never less than 1.
E. None of the above.

#### Answer: **A**

- **A ✓** — The sigmoid derivative peaks at 0.25 (when $z=0$) and approaches zero for large $|z|$. In a network with $L$ layers, the gradient norm decays as $\le 0.25^L$, which vanishes rapidly. ReLU's derivative is 1 for $z > 0$ and 0 otherwise, so active neurons pass gradients without attenuation.
- **B ✗** — Sigmoid outputs are in $(0,1)$, not negative. Sign alternation is not the primary cause of vanishing gradients.
- **C ✗** — The sigmoid derivative is bounded above by 0.25; it is the *small* derivatives that vanish gradients, not large ones.
- **D ✗** — ReLU introduces the "dying ReLU" problem (neurons stuck at zero gradient when $z \le 0$), so it mitigates but does not eliminate gradient issues in deep networks.
- **E ✗** — A is correct.

---

**Question (18)** [Week 11]
Which of the following best describes the role of negentropy in ICA?

A. Negentropy $J(s) = H(z) - H(s)$ (where $z$ is Gaussian with the same variance as $s$) measures how non-Gaussian $s$ is; maximising negentropy across independent components identifies the mixing matrix $A$.
B. Negentropy is the KL divergence between the source distribution and a Laplace prior; minimising it recovers sparse components.
C. Negentropy equals zero when $s$ is Gaussian, confirming that Gaussian sources are identifiable in ICA.
D. FastICA maximises negentropy by gradient ascent on the full likelihood, requiring explicit density estimation at each step.
E. None of the above.

#### Answer: **A**

- **A ✓** — By the maximum-entropy principle, Gaussian random variables have the highest entropy among all distributions with the same variance. Hence $J(s) = H(z) - H(s) \ge 0$, with equality iff $s$ is Gaussian. Maximising $J(s)$ across all unit-variance projections finds the most non-Gaussian directions, which (under the ICA model) correspond to the independent sources.
- **B ✗** — Negentropy is not a KL divergence with a Laplace prior; sparse ICA methods (e.g., sparse coding) use sparsity priors, but that is a different framework.
- **C ✗** — Negentropy equalling zero for Gaussian sources means Gaussian sources are *not* identifiable (they cannot be distinguished from a rotation of other Gaussian sources), which is the fundamental ICA limitation, not a confirmation of identifiability.
- **D ✗** — FastICA avoids explicit density estimation by using a fixed non-polynomial contrast function $G$ (e.g., $\log\cosh$) to approximate negentropy, and it uses a Newton/fixed-point iteration, not plain gradient ascent.
- **E ✗** — A is correct.

---

**Question (19)** [Week 12]
The CORCONDIA diagnostic for PARAFAC is defined as $\text{CC} = 100\!\left(1 - \frac{\|\mathcal{I} - \mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$, where $\mathcal{I}$ is the identity super-diagonal core and $\mathcal{G}$ is the estimated core in Tucker form. What do values near 100 and near 0 indicate?

A. Near 100: the PARAFAC model with $R$ components fits the data well and the trilinear structure is appropriate. Near 0: the core deviates strongly from super-diagonal, suggesting $R$ is too large or the trilinear assumption is violated.
B. Near 100: overfitting because the core has been inflated to absorb noise. Near 0: the model is underfitting.
C. Near 100: model components are correlated, indicating degeneracy. Near 0: components are orthogonal, indicating a good fit.
D. CORCONDIA near 100 indicates that $R$ should be increased; near 0, $R$ should be decreased.
E. None of the above.

#### Answer: **A**

- **A ✓** — PARAFAC assumes $\mathcal{G} = \mathcal{I}$ (super-diagonal core with ones). When this holds, CC ≈ 100, validating the trilinear model and the chosen $R$. When $\mathcal{G}$ deviates substantially from $\mathcal{I}$, CC drops toward 0 (or even negative), indicating that the model is ill-specified — either $R$ is too high (components start fitting noise) or the data does not have genuine trilinear structure.
- **B ✗** — High CC reflects a good structural fit, not overfitting. Overfitting would manifest differently (e.g., degenerate components).
- **C ✗** — Correlated or degenerate components are a different problem (sign-flipping, slow convergence) not directly captured by CC approaching 100.
- **D ✗** — The guidance is the opposite: when CC drops sharply as $R$ increases, that signals $R$ is too large; the correct $R$ is where CC is still near 100.
- **E ✗** — A is correct.

---

**Question (20)** [Week 10 / Week 6]
Which of the following correctly characterises dropout, L2 regularization, and early stopping in neural networks? (Select all that apply.)

A. Dropout randomly zeros activations during training with probability $p$, acting as an implicit ensemble of $2^H$ sub-networks (where $H$ is the number of hidden units), reducing co-adaptation of features.
B. L2 regularization (weight decay) adds $\lambda \sum w^2$ to the loss, shrinking all weights toward zero and penalising large weights, analogous to a Gaussian prior on weights.
C. Early stopping halts training when validation error starts increasing, preventing the model from memorising training data; it implicitly regularises effective model capacity.
D. All three methods reduce bias rather than variance, because they constrain the hypothesis class.
E. None of the above.

#### Answer: **A, B, C**

- **A ✓** — Dropout with rate $p$ defines a stochastic sub-network at each forward pass; at test time weights are scaled by $(1-p)$. This prevents units from co-adapting and acts as approximate model averaging.
- **B ✓** — L2 adds $\lambda \|w\|^2$ to the loss, giving gradient update $w \leftarrow (1-2\lambda\eta)w - \eta \nabla_w L$; this is weight decay. It corresponds to placing a zero-mean Gaussian prior on weights under a Bayesian interpretation.
- **C ✓** — By stopping before the network converges to a training-set minimum, early stopping limits the effective number of parameters the optimiser has adjusted, acting as a form of capacity regularization analogous to L2 in some analyses.
- **D ✗** — All three reduce *variance* (overfitting), potentially at the cost of slight *increases* in bias. Constraining hypothesis class generally trades bias for variance in the bias-variance sense.
- **E ✗** — A, B, and C are all correct.

---

## Open Questions

---

**Question 21 (20 points) — ICA: Uniqueness, Distributions, and FastICA**

**(a)** [5 pts] State the ICA model $x = As$. What does the independence assumption require? Explain, using the Central Limit Theorem, why at most one source component can be Gaussian.

**(b)** [5 pts] Define negentropy $J(s)$ and explain why it measures non-Gaussianity. Describe two choices of contrast function $G$ used in practice and what properties make them suitable.

**(c)** [5 pts] Describe the whitening (sphering) step that precedes ICA estimation. What does it achieve mathematically, and why does it simplify the ICA optimisation problem?

**(d)** [5 pts] Compare ICA and PCA with respect to the order of statistical independence they achieve. Explain why PCA's decorrelation is insufficient for blind source separation (BSS).

---

### Solution

**Part (a) — ICA model and the Gaussian exception**

The ICA model assumes that observed signals $x \in \mathbb{R}^p$ are linear mixtures of $p$ statistically independent latent sources $s = (s_1, \ldots, s_p)^T$:

$$x = As$$

where $A \in \mathbb{R}^{p \times p}$ is the unknown mixing matrix. The goal is to estimate the demixing matrix $W = A^{-1}$ such that $\hat{s} = Wx$ recovers the original sources up to permutation and scaling.

**Independence assumption**: The sources $s_1, \ldots, s_p$ must be mutually statistically independent — their joint density factorises as $p(s) = \prod_j p_j(s_j)$. This is a much stronger requirement than the second-order (pairwise) uncorrelatedness that PCA achieves.

**Why at most one source can be Gaussian**: By the Central Limit Theorem, any mixture (convolution) of independent random variables converges toward Gaussian as the number of components grows. More precisely, the distribution of $w^T x = w^T A s = \sum_j a_j s_j$ (a linear mixture of independent sources) is *more Gaussian* than any individual $s_j$ unless $w^T A$ selects exactly one source. Therefore, when we search for directions $w$ that maximise non-Gaussianity, we move away from mixtures and toward individual sources. If *two or more* sources were Gaussian, their mixture would also be Gaussian (Gaussians are closed under convolution), making those sources indistinguishable from any rotation of each other — rendering them non-identifiable. Hence identifiability requires at most one Gaussian source.

---

**Part (b) — Negentropy as a measure of non-Gaussianity**

Negentropy is defined as the differential entropy gap between a Gaussian variable $z \sim \mathcal{N}(0, 1)$ and the variable of interest $s$ (assumed zero-mean, unit-variance):

$$J(s) = H(z) - H(s) \ge 0$$

By the maximum-entropy principle, the Gaussian distribution has the highest entropy among all distributions with a given mean and variance. Therefore $J(s) \ge 0$, with equality if and only if $s$ is Gaussian. Maximising $J(s)$ thus finds the *most non-Gaussian* projection of the data, which under the ICA model corresponds to an independent source.

Direct computation of differential entropy requires density estimation, which is expensive. FastICA uses an approximation:

$$J(s) \approx \left[E[G(s)] - E[G(z)]\right]^2$$

for a smooth nonlinear contrast function $G$. Two standard choices are:

1. $G(u) = \frac{1}{a}\log\cosh(au)$ (typically $a \approx 1$): this is a smooth approximation to the absolute value; it is robust to outliers and captures super-Gaussian (heavy-tailed) sources well.
2. $G(u) = -\exp(-u^2/2)$: related to the Gaussian kernel; it is particularly suited for super-Gaussian sources and provides good theoretical properties.

Both choices are non-polynomial and satisfy the conditions for valid ICA contrast functions (non-quadratic, even functions or functions capturing higher cumulants).

---

**Part (c) — Whitening (sphering) step**

Before running ICA, the observed data $x$ is whitened to produce $\tilde{x}$ satisfying:

$$E[\tilde{x}\tilde{x}^T] = I$$

**How it is done**: Compute the eigendecomposition of the covariance matrix $E[xx^T] = U\Lambda U^T$ and set $\tilde{x} = \Lambda^{-1/2}U^T x$. In practice this is done via PCA: the data is projected onto the principal components and scaled by the inverse square root of the eigenvalues.

**Mathematical effect**: The whitened data satisfies the identity covariance condition. If $x = As$, then after whitening the effective mixing matrix $\tilde{A} = \Lambda^{-1/2}U^T A$ satisfies $\tilde{A}\tilde{A}^T = I$, i.e., $\tilde{A}$ is an *orthogonal* matrix.

**Why it simplifies ICA**: Without whitening, the ICA search is over all invertible $p \times p$ matrices — a space with $p^2$ degrees of freedom. After whitening, the search reduces to orthogonal matrices, which have only $p(p-1)/2$ degrees of freedom. This is a major dimensionality reduction of the optimisation problem, and algorithms such as FastICA exploit the structure of the orthogonal group (Stiefel manifold) to converge rapidly via fixed-point iterations.

---

**Part (d) — ICA vs PCA: order of independence**

**PCA decorrelates** the components: the principal components $z = V^T x$ (where $V$ are the eigenvectors of $\hat{\Sigma}$) satisfy $E[z_i z_j] = 0$ for $i \ne j$. This is *second-order* independence — it eliminates linear dependencies, or equivalently, pairwise correlations.

**ICA achieves full statistical independence**: the recovered components $\hat{s} = Wx$ satisfy $p(\hat{s}) = \prod_j p_j(\hat{s}_j)$, meaning all joint moments factorise. This is *all-order* independence, eliminating not just linear but also nonlinear statistical dependencies.

**Why decorrelation is insufficient for BSS**: Consider two independent, identically distributed non-Gaussian sources $s_1, s_2$. Their covariance matrix after any orthogonal mixing $x = As$ remains diagonal (since orthogonal transformations preserve the covariance structure when sources are already uncorrelated). PCA, which seeks the eigenvectors of the covariance, cannot distinguish a rotation of uncorrelated sources from the original sources. Concretely: if $s_1$ and $s_2$ are i.i.d. uniform $[-1, 1]$, then both $s$ and $Rs$ (for any orthogonal $R$) have the same covariance matrix $I$ — PCA gives an arbitrary rotation, not the original sources. ICA resolves this by exploiting the higher-order structure (the non-Gaussianity) that a rotation destroys.

Formally: uncorrelatedness means $E[s_i s_j] = 0$ for $i \ne j$; full independence additionally requires $E[g(s_i)h(s_j)] = E[g(s_i)]E[h(s_j)]$ for all measurable $g, h$. Decorrelation satisfies the former but not the latter.

---

**Question 22 (20 points) — Nested CV for Wearables: Correct Procedure**

The dataset consists of 192 observations: 16 subjects $\times$ 3 activities $\times$ 4 seasons. The task is to classify activity from wearable biosignals. A regularized classifier with hyperparameter $\lambda$ is to be trained and evaluated.

**(a)** [5 pts] A colleague proposes: "Tune $\lambda$ using 10-fold CV on all 192 observations, then report the minimum CV error as the test error." Name and explain two statistical errors in this procedure.

**(b)** [5 pts] Propose a correct outer CV loop for unbiased assessment of a *generalized* model (one that should work for new subjects). How many outer folds? What is in each outer test fold? How many observations enter the outer training set in each iteration?

**(c)** [5 pts] Inside each outer fold's training set, an inner CV loop selects $\lambda$. Describe the inner loop. Why is it essential that the outer test fold is completely excluded from the inner loop?

**(d)** [5 pts] After nested CV completes, the researcher retrains a "final" model for deployment. Describe (i) what data to use and (ii) which $\lambda$ to choose. Explain why the nested CV error cannot be reported as this final model's training error.

---

### Solution

**Part (a) — Two errors in the naive procedure**

**Error 1 — Optimistic bias / data leakage in model selection**: Tuning $\lambda$ using all 192 observations and then reporting the CV error obtained during that same tuning process conflates model selection with model assessment. The minimum CV error over a grid of $\lambda$ values will be optimistically biased because we have selected $\lambda$ to minimize the very error we report. In effect, $\lambda$ has "seen" the test folds during its selection, so the reported error is not an unbiased estimate of the true generalization error for a new dataset.

**Error 2 — IID violation: random splits ignore subject identity**: The 192 observations are not independently and identically distributed. Each subject contributes 12 observations (3 activities $\times$ 4 seasons), and observations from the same subject are correlated. A random 10-fold split will likely place training and test observations from the same subject in different folds, but these share the same individual's physiological baseline. As a result, the model can exploit subject-specific patterns — it is being evaluated on how well it memorises subjects, not on whether it generalises to *new* subjects. This is a form of data leakage that leads to overly optimistic error estimates for deployment.

---

**Part (b) — Correct outer CV loop**

For a *generalized* model (one evaluated on new, unseen subjects), the outer loop must be **Leave-One-Individual-Out (LOIO) cross-validation**, with **16 outer folds** — one per subject.

**Structure of each outer fold**:
- **Outer test fold**: all 12 observations from subject $i$ (3 activities $\times$ 4 seasons).
- **Outer training set**: the remaining $192 - 12 = 180$ observations from subjects $\{1, \ldots, 16\} \setminus \{i\}$.

The LOIO structure ensures that the test fold contains data from a subject who contributed zero observations to training, correctly mimicking deployment on a new individual. This respects the non-IID structure of the data.

The outer test performance is computed 16 times, each time evaluating on a held-out subject. The overall assessment is the average (and standard deviation) of these 16 test errors.

---

**Part (c) — Inner CV loop and exclusion of the outer test fold**

Within outer fold $i$, the inner loop tunes $\lambda$ using only the 180 training observations. A practical inner loop is another LOIO CV on these 180 observations (15 inner folds, each holding out one of the remaining 15 subjects), or alternatively $K$-fold CV respecting subject boundaries:

1. Split the 180 training observations into inner folds (e.g., 15 inner folds of 12 observations each, one per remaining subject).
2. For each candidate $\lambda$ in a grid, train on the inner training set and evaluate on the inner validation fold.
3. Average the inner CV error across all inner folds; select $\hat{\lambda}_i = \arg\min_\lambda \overline{\text{CV}}_\text{inner}(\lambda)$.

**Why the outer test fold must be completely excluded**: The purpose of the outer test fold is to provide an unbiased estimate of generalization error for a *new* subject. If subject $i$'s data were used in the inner loop — even only to select $\lambda$ — the chosen $\hat{\lambda}_i$ would be tuned, at least partially, to subject $i$'s characteristics. The resulting error on subject $i$'s outer fold would then be optimistically biased. More formally, the nested CV error estimate is valid only when the outer test fold is never used, directly or indirectly, to fit any model parameter or hyperparameter. Leaking the outer test fold into the inner loop violates this condition.

---

**Part (d) — Final model retraining**

**(i) Data**: The final model should be trained on **all 192 observations** (all subjects, all seasons, all activities). Since the purpose of nested CV was assessment only — determining whether the modeling approach generalizes — the final deployed model should exploit all available data to maximize estimation quality.

**(ii) $\lambda$ selection**: The researcher should train the final model using a $\lambda$ value determined by running the inner CV on the full 192 observations (i.e., one additional, non-nested, inner-loop pass on all data). Common practice is to use the $\lambda$ that was selected most frequently across the 16 outer-fold inner loops, or equivalently to run a single LOIO inner CV on all 192 observations to select $\hat{\lambda}_\text{final}$.

**Why the nested CV error is not the final model's training error**: The nested CV error estimates the expected performance of a *model trained on 180 observations* (or $15/16$ of the data), not a model trained on all 192. Since the final model uses strictly more data, it will typically generalize at least as well as the cross-validated estimate suggests — so the nested CV error is a *conservative* (pessimistically biased) estimate of the final model's true generalization error, not its training error. The training error of the final model (evaluated on the 192 training points) would be far lower due to overfitting, and is not a valid estimate of generalization error. The nested CV error is the best available honest estimate of how the deployed model will perform on new subjects, despite being slightly pessimistic.
