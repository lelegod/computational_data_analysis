# Exam Solutions — CDA 02582 (2022/2023)

**Date:** May 17, 2022 (handed out for 2023 preparation)
**Format:** 20 multiple-choice questions + 2 open questions
**Scoring:** 2 points for a correct answer combination, 0 for wrong or unanswered. Multiple options can be part of the single correct combination.
**Note:** Q19 was removed from the 2023 curriculum and is omitted throughout.

---

## Answer Key Summary

| Q | Correct Options |
|---|----------------|
| 1 | C, D |
| 2 | A, C, D |
| 3 | C, D |
| 4 | A |
| 5 | B, D |
| 6 | C |
| 7 | C |
| 8 | B, D |
| 9 | A, D (see note) |
| 10 | A, C, D |
| 11 | B, C |
| 12 | B |
| 13 | D |
| 14 | A, D |
| 15 | B |
| 16 | E |
| 17 | A, B, D |
| 18 | C, D |
| 19 | Removed |
| 20 | C ⚠️ (official sheet appears corrupted — see note) |

---

## Multiple Choice Questions

---

### Question 1 — Probabilistic Models of Data

**Question:** Which method(s) are based on probabilistic models of data?

**Official Answer:** C and D

**Option A — Support Vector Machines:** ❌ Wrong
SVM is a purely geometric/margin-based method. It finds the maximum-margin hyperplane that separates classes using support vectors. There is no probability distribution over the data; SVM does not model $P(y \mid x)$ or $P(x \mid y)$ in any parametric sense. Although Platt scaling can post-hoc convert SVM outputs to probabilities, this is not part of the core method.

**Option B — K-means clustering:** ❌ Wrong
K-means is a hard-assignment iterative algorithm that minimises the within-cluster sum of squared distances $\sum_{k} \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$. There is no underlying probability distribution. Each observation is assigned to exactly one cluster with no notion of membership probability. K-means is sometimes described as a limiting case of the EM algorithm, but it is not itself probabilistic.

**Option C — Gaussian Mixture Models:** ✓ Correct
GMM is explicitly a probabilistic model. The data are assumed to come from a mixture of $K$ Gaussian distributions:
$$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
where $\pi_k$ are mixing proportions, $\mu_k$ are means, and $\Sigma_k$ are covariance matrices. Fitting uses the EM algorithm to maximise the log-likelihood. Cluster membership is given as a posterior probability $P(z=k \mid x)$, making it a fully probabilistic method.

**Option D — Logistic Regression:** ✓ Correct
Logistic regression is a discriminative probabilistic model. It directly models the posterior class probability:
$$P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$
This is derived from assuming that the log-odds are linear in $x$, which itself can be motivated by assuming class-conditional Gaussians with equal covariance (the same assumption as LDA). The model is fit by maximising the log-likelihood (cross-entropy), making it inherently probabilistic.

**Option E — None of the above:** ❌ Wrong
C and D are probabilistic methods, so this option is incorrect.

> **Key takeaway:** A method is probabilistic if it explicitly models a probability distribution over the data or class posteriors. GMM models the joint density; Logistic Regression models the class posterior. SVM and K-means are geometric/combinatorial and not probabilistic.

---

### Question 2 — Methods That Handle p > n (More Features Than Observations)

**Question:** Which method(s) can handle data with fewer observations than dimensions?

**Official Answer:** A, C, and D

**Option A — Support Vector Machines:** ✓ Correct
SVM works in the dual formulation where the optimisation problem scales with $n$ (number of observations), not $p$ (number of features). The dual problem involves an $n \times n$ kernel matrix rather than a $p \times p$ covariance matrix. Even with $p \gg n$, the dual form $\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$ remains well-posed. Additionally, the kernel trick allows implicit mapping to feature spaces of dimension far larger than $n$.

**Option B — Logistic regression without regularization:** ❌ Wrong
Without regularisation, logistic regression maximises the log-likelihood $\sum_i \log P(y_i \mid x_i)$. When $p \geq n$, the design matrix $X$ does not have full column rank, and the maximum likelihood estimator does not exist (the log-likelihood can be driven to $-\infty$ or to perfect separation). The problem is ill-posed, and coefficients diverge. Regularisation (L1 or L2) is required to make logistic regression work when $p \geq n$.

**Option C — Random Forest:** ✓ Correct
Random Forest selects a random subset of $m \ll p$ features at each split, where $m$ is typically $\sqrt{p}$ (classification) or $p/3$ (regression). This means each tree only ever looks at a small fraction of features at once, making high dimensionality manageable. Even when $p \gg n$, individual trees fit well because they only consider a few features per split. Bagging also means each tree sees a different bootstrap sample, adding further stability.

**Option D — Principal Component Analysis:** ✓ Correct
PCA reduces dimensionality by projecting data onto the directions of maximum variance (the eigenvectors of $X^TX$, or equivalently the right singular vectors of $X$). When $p \gg n$, $X^TX$ is $p \times p$ but only rank $n$, while $XX^T$ is $n \times n$ with full rank. PCA can use the $n \times n$ matrix instead, making it computationally tractable. The result is an embedding of $n$ observations into at most $n-1$ dimensions regardless of $p$.

**Option E — None of the above:** ❌ Wrong
Three of the listed methods handle $p \gg n$ scenarios, so this option is incorrect.

> **Key takeaway:** Methods that avoid inverting $X^TX$ directly (via dual formulations, random subspaces, or eigendecomposition tricks) can handle high-dimensional data. Logistic regression without regularisation fails because the system is underdetermined.

---

### Question 3 — Linear Discriminant Analysis (LDA)

**Question:** Which of the following statements are true for the Linear Discriminant Analysis (LDA) method?

**Official Answer:** C and D

**Option A — LDA is a linear method because we assume the separation function between the classes is linear:** ❌ Wrong
This statement has the causal direction backwards. LDA does not *assume* linearity of the boundary — the linearity of the boundary is a *consequence* of the Gaussian equal-covariance assumption. The decision boundary emerges from the log-posterior ratio:
$$\log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = x^T \Sigma^{-1}(\mu_1 - \mu_0) - \frac{1}{2}(\mu_1 + \mu_0)^T \Sigma^{-1}(\mu_1 - \mu_0) + \log \frac{\pi_1}{\pi_0}$$
The quadratic terms cancel when $\Sigma_1 = \Sigma_2 = \Sigma$, leaving a linear function of $x$. Linearity is derived, not assumed directly.

**Option B — LDA handles outliers better than logistic regression:** ❌ Wrong
The opposite is closer to the truth. LDA directly uses the sample mean $\hat{\mu}_k$ and the pooled sample covariance $\hat{\Sigma}$ to estimate class-conditional densities. Both of these statistics are sensitive to outliers (means and covariances are non-robust estimators). Logistic regression, by contrast, maximises the conditional likelihood $P(y \mid x)$ and is generally considered more robust to violations of distributional assumptions and to outliers in feature space.

**Option C — LDA is a probabilistic method:** ✓ Correct
LDA is explicitly probabilistic. It models class-conditional densities as Gaussians:
$$P(x \mid y=k) = \mathcal{N}(x \mid \mu_k, \Sigma)$$
and combines them with class priors $\pi_k = P(y=k)$ via Bayes' rule to compute posterior probabilities:
$$P(y=k \mid x) = \frac{P(x \mid y=k) \, \pi_k}{\sum_{j} P(x \mid y=j) \, \pi_j}$$
Classification is based on the maximum posterior probability, making LDA a generative probabilistic classifier.

**Option D — LDA is a linear method because we assume the classes in our data are Gaussian and have the same covariance matrices:** ✓ Correct
This is the correct mechanistic explanation. When all classes share the same covariance matrix $\Sigma$, the quadratic term $-\frac{1}{2}x^T \Sigma_k^{-1} x$ is identical for all classes $k$ and cancels in the log-posterior ratio. What remains is linear in $x$. If classes have *different* covariance matrices ($\Sigma_k \neq \Sigma_j$), the quadratic terms do not cancel and the boundary becomes quadratic — this is Quadratic Discriminant Analysis (QDA).

**Option E — None of the above:** ❌ Wrong
Both C and D are correct statements about LDA.

> **Key takeaway:** LDA's linearity is a *consequence* of equal class covariances (not an assumption in itself), and LDA is probabilistic because it applies Bayes' rule to Gaussian class-conditional densities.

---

### Question 4 — Ridge Regularisation — Effect of Too Large $\lambda$

**Question:** You happen to select a too large $\lambda$ in your ridge regularisation,
$$\arg\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$
How will that affect the estimated model?

**Official Answer:** A

**Option A — It will have high bias:** ✓ Correct
A large $\lambda$ heavily penalises the magnitude of $\beta$, shrinking all coefficients strongly toward zero. The ridge solution is $\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1}X^Ty$. As $\lambda \to \infty$, $\hat{\beta} \to 0$ regardless of the data. This creates a model that is systematically wrong (biased toward zero) — it underfits the true signal. High $\lambda$ = high bias, low variance. This is the classic overly-regularised end of the bias-variance tradeoff.

**Option B — It will have high variance:** ❌ Wrong
High variance corresponds to a model that is sensitive to small changes in the training data. With a large $\lambda$, the model is strongly constrained and changes very little across different training sets — variance is *low*, not high. High variance is the symptom of too *small* $\lambda$ (i.e., near-OLS behaviour).

**Option C — Not possible to say:** ❌ Wrong
The direction of the bias-variance tradeoff is completely determinate given the direction of $\lambda$. Too large $\lambda$ always causes high bias and low variance in any regularised linear model. The statement "not possible to say" is incorrect.

**Option D — It will have low bias:** ❌ Wrong
Low bias corresponds to a model that accurately captures the true signal on average. With large $\lambda$, coefficients are shrunk so far toward zero that the model systematically underestimates the true effect — this is precisely high bias. Low bias would correspond to small $\lambda$ (near-OLS), not large $\lambda$.

**Option E — None of the above:** ❌ Wrong
Option A is correct, so this option is incorrect.

> **Key takeaway:** In regularised models, increasing $\lambda$ increases bias (underfitting) and decreases variance. Too large $\lambda$ = high bias = the classic underfitting scenario.

---

### Question 5 — Lasso Regularisation — Effect of Too Small $\lambda$

**Question:** You happen to select a too small $\lambda$ in your lasso regularisation,
$$\arg\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_1$$
How will that affect the estimated model?

**Official Answer:** B and D

**Option A — It will have high bias:** ❌ Wrong
Bias measures the systematic error between the expected model predictions and the true function. With a very small $\lambda$, the lasso penalty has little effect and the solution approaches the ordinary least-squares (OLS) solution. OLS is unbiased (under standard assumptions), so a small $\lambda$ produces *low* bias, not high bias. High bias would require $\lambda$ to be large (strong shrinkage).

**Option B — It will have high variance:** ✓ Correct
With small $\lambda$, the lasso imposes little regularisation and the solution is close to OLS. OLS has high variance when the number of features is large relative to observations, or when features are correlated — small perturbations in the training data lead to large changes in $\hat{\beta}$. Insufficient regularisation means the model overfits the training data (high variance, low bias).

**Option C — Not possible to say:** ❌ Wrong
As with Q4, the direction is determinate. Too small $\lambda$ always moves the model toward high variance and low bias in any regularised linear model.

**Option D — It will have low bias:** ✓ Correct
With small $\lambda$, the penalty contributes negligibly to the objective, and $\hat{\beta}$ is close to the unpenalised OLS estimator, which is unbiased. Low bias and high variance simultaneously characterise the under-regularised (high-complexity) end of the bias-variance tradeoff. Both B and D are simultaneously true and are two sides of the same coin.

**Option E — None of the above:** ❌ Wrong
Both B and D are correct, so this option is incorrect.

> **Key takeaway:** Too small $\lambda$ in any regularised model $\approx$ OLS: low bias, high variance (overfitting). Too large $\lambda$: high bias, low variance (underfitting). Both B and D describe the same under-regularised state.

---

### Question 6 — How Lasso Estimates Are Calculated

**Question:** How are the estimates of a linear model with lasso regularisation calculated?

**Official Answer:** C

**Option A — $(X^TX + \lambda I)^{-1}X^TY$:** ❌ Wrong
This is the closed-form solution for *ridge* regression, not lasso. Ridge's L2 penalty $\lambda\|\beta\|_2^2$ is differentiable everywhere and produces a quadratic objective, which can be minimised analytically by setting the gradient to zero:
$$\nabla_\beta \left[\|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2 \right] = -2X^T(y - X\beta) + 2\lambda\beta = 0 \implies \hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$$

**Option B — $(X^TX)^{-1}X^TY$:** ❌ Wrong
This is the ordinary least-squares (OLS) solution with no regularisation at all ($\lambda = 0$). It is the analytical solution to $\min_\beta \|y - X\beta\|_2^2$.

**Option C — Solved numerically, no analytical solution:** ✓ Correct
The lasso penalty $\lambda\|\beta\|_1 = \lambda \sum_i |\beta_i|$ uses the L1 norm, which is *not differentiable* at $\beta_i = 0$. Because the subdifferential (not gradient) is required at zero, a standard closed-form solution via matrix algebra does not exist. The objective must be minimised numerically using algorithms such as coordinate descent (updating one coefficient at a time with a soft-thresholding operation) or LARS (Least Angle Regression), both of which trace the full regularisation path.

**Option D — $\arg\min_\beta \|y - X\beta\| + \lambda\|\beta\|_2^2$:** ❌ Wrong
This expression describes an elastic net-like problem (L1 loss on residuals + L2 penalty on coefficients), or alternatively a mis-specified lasso. The lasso objective uses the L2 norm *squared* on residuals and the L1 norm on coefficients: $\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$. The expression in option D has neither the correct penalty nor the squared residual term.

**Option E — None of the above:** ❌ Wrong
Option C is correct, so this option is incorrect.

> **Key takeaway:** The lasso's L1 penalty is non-differentiable at zero, preventing a closed-form solution. Ridge's L2 penalty is smooth everywhere, enabling the analytical formula $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$.

---

### Question 7 — Is SVM a Linear or Non-Linear Classifier?

**Question:** Is the Support Vector Machine a linear or a non-linear classifier?

**Official Answer:** C

**Option A — Linear:** ❌ Wrong
SVM with a linear kernel is indeed a linear classifier, but this is not universally true. The question asks about SVM in general, without specifying a kernel. Claiming SVM is always linear is incomplete and incorrect.

**Option B — Non-linear:** ❌ Wrong
SVM with a non-linear kernel (RBF, polynomial, sigmoid) is non-linear, but again this is not universally true. Claiming SVM is always non-linear ignores the common linear-kernel case.

**Option C — That depends on the chosen kernel:** ✓ Correct
The nature of the SVM decision boundary is entirely determined by the kernel function $K(x_i, x_j)$. With a linear kernel $K(x_i, x_j) = x_i^T x_j$, the boundary is a hyperplane (linear). With a Radial Basis Function (RBF) kernel $K(x_i, x_j) = \exp(-\gamma\|x_i - x_j\|^2)$ or a polynomial kernel $K(x_i, x_j) = (x_i^T x_j + c)^d$, the boundary in the original feature space is non-linear. The kernel trick implicitly maps data to a high-dimensional (possibly infinite-dimensional) feature space and finds a linear boundary *there*, which corresponds to a non-linear boundary in the original space.

**Option D — SVM is not a classifier:** ❌ Wrong
SVM can function as both a classifier (Support Vector Classification, SVC) and a regressor (Support Vector Regression, SVR). In its most common application it is explicitly a classifier. This statement is simply false.

**Option E — None of the above:** ❌ Wrong
Option C is correct, so this option is incorrect.

> **Key takeaway:** SVM's linearity depends entirely on the kernel. A linear kernel gives a linear boundary; non-linear kernels (RBF, polynomial) give non-linear boundaries via the kernel trick.

---

### Question 8 — Techniques for the Multiple Testing Problem

**Question:** Which techniques are developed to handle the multiple testing problem?

**Official Answer:** B and D

**Option A — Akaike Information Criteria (AIC):** ❌ Wrong
AIC is a model selection criterion, not a multiple testing correction. AIC $= -2\log\hat{L} + 2p$ penalises model complexity to balance fit vs. parsimony when comparing models. It has nothing to do with the problem of performing many hypothesis tests simultaneously and the resulting inflation of false-positive rates.

**Option B — Bonferroni correction:** ✓ Correct
The Bonferroni correction is specifically designed for the multiple testing problem. It controls the Family-Wise Error Rate (FWER) — the probability of making *at least one* false positive across all tests. Given $M$ tests and significance level $\alpha$, Bonferroni tests each hypothesis at threshold $\alpha/M$:
$$\text{Reject } H_0^{(i)} \text{ if } p_i < \frac{\alpha}{M}$$
This guarantees $P(\text{at least one false positive}) \leq \alpha$.

**Option C — Bootstrapping:** ❌ Wrong
Bootstrapping is a resampling technique used to estimate the sampling distribution of a statistic, compute standard errors, or construct confidence intervals. While it can be used in hypothesis testing, it is not a method specifically developed to address the multiple testing problem (the inflation of Type I error when many tests are performed simultaneously).

**Option D — Benjamini-Hochberg's algorithm:** ✓ Correct
The Benjamini-Hochberg (BH) procedure controls the False Discovery Rate (FDR) — the expected proportion of false positives among all rejected hypotheses. It is directly developed to address multiple testing. The procedure ranks $p$-values $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(M)}$ and rejects all hypotheses up to the largest $k$ such that $p_{(k)} \leq k\alpha/M$. BH is less conservative than Bonferroni and yields more power (more true discoveries).

**Option E — None of the above:** ❌ Wrong
Both B and D are valid multiple testing corrections.

> **Key takeaway:** Bonferroni controls FWER (probability of any false positive); BH controls FDR (expected fraction of false positives among discoveries). Both are specifically designed for multiple testing. AIC and bootstrapping serve different purposes.

---

### Question 9 — Why Not Penalise the Intercept in Regularisation?

**Question:** When dealing with regularisation we usually do not penalise the intercept. Why not (which statements are correct)?

**Official Answer:** A and D (the answer sheet marks A, B, C as "(x)" — meaning debated/partially credit — and D as the primary answer; based on standard teaching in this course, A and D are the defensible correct answers)

⚠️ **Note on the answer sheet:** The official sheet uses "(x)" notation for A, B, and C, indicating these were considered partially valid or debated. The pedagogically standard and defensible correct answer is **A**, with **D** being a correct but derivative consequence. **B is incorrect** (see below).

**Option A — Penalising the intercept would introduce bias without any reduction in variance:** ✓ Correct
The intercept $\beta_0$ simply shifts predictions by a constant; it does not control the complexity of the function being fitted. Penalising it shrinks predictions toward zero rather than toward the true mean of $y$, introducing bias. Because the intercept does not capture any pattern that could overfit, shrinking it does *not* reduce variance — there is no overfitting to suppress. The net effect is pure bias introduction with no variance benefit, which is never desirable.

**Option B — Penalising the intercept would introduce variance without any reduction in bias:** ❌ Wrong
This is incorrect. The intercept is a fixed constant term; penalising it does not add randomness to the estimation procedure in a way that increases variance. The primary effect of penalising the intercept is a systematic bias (the estimates are shifted), not an increase in variance across different training sets. This option has it backwards.

**Option C — Penalising the intercept would introduce both bias and variance:** ❌ Wrong (or very weakly argued)
While one could argue that any parameter penalisation has some variance effect, this is not the standard explanation and is not the primary concern. The dominant effect is the introduction of bias without meaningful variance reduction. Saying "both" without qualification is misleading and does not reflect the standard teaching of this concept.

**Option D — The model will get a lower Expected Prediction Error if we do not penalise the intercept:** ✓ Correct
Since penalising the intercept introduces bias without reducing variance (as stated in A), the Expected Prediction Error (EPE) = Bias$^2$ + Variance + $\sigma^2$ is higher with intercept penalisation than without. Therefore, not penalising the intercept gives a lower EPE. This is correct, but it is a *consequence* of A rather than an independent reason — D restates the conclusion of A in terms of EPE.

**Option E — None of the above:** ❌ Wrong
At least A and D are correct.

> **Key takeaway:** The intercept is not penalised because doing so introduces bias without reducing variance — the intercept controls the mean level of predictions, not model complexity. This always makes the EPE worse.

---

### Question 10 — Bonferroni Correction and FDR: True Statements

**Question:** Which of the following statements are correct?

**Official Answer:** A, C, and D

**Option A — Bonferroni correction reduces the risk of false positives:** ✓ Correct
This is the primary purpose of Bonferroni correction. By testing each of the $M$ hypotheses at threshold $\alpha/M$ instead of $\alpha$, the probability of making even a single false positive across all tests is controlled at level $\alpha$. It directly reduces false positives at the cost of reduced power (more false negatives).

**Option B — Bonferroni correction reduces the chance of accepting a null hypothesis:** ❌ Wrong
This is the opposite of what Bonferroni does. Bonferroni makes the rejection threshold *more stringent* ($\alpha/M$ instead of $\alpha$), which means it becomes *harder* to reject null hypotheses. Therefore, Bonferroni *increases* the chance of failing to reject (i.e., "accepting") null hypotheses, not decreasing it. This option has the direction completely wrong.

**Option C — Corrections based on a 5% False Discovery Rate is expected to give more significant findings compared to Bonferroni correction with a 5% significance level:** ✓ Correct
BH at 5% FDR is less conservative than Bonferroni at 5% FWER. Bonferroni uses threshold $\alpha/M$ per test, which becomes extremely stringent as $M$ grows. BH uses thresholds that grow proportionally with the rank of the $p$-value, allowing many more rejections while controlling the *proportion* of false discoveries (FDR) rather than the probability of *any* false discovery (FWER). For any reasonably large $M$, BH will find more significant results.

**Option D — Corrections based on a 5% False Discovery Rate is expected to give more false positives compared to Bonferroni Correction with a 5% significance level:** ✓ Correct
This is the direct statistical consequence of C. More total discoveries (C) with a higher per-test threshold means more false positives. BH guarantees that at most 5% of its *discoveries* are false positives, but because it makes more discoveries than Bonferroni, the absolute number of false positives is higher. Bonferroni guarantees that the probability of *any* false positive is at most 5% — a much stricter criterion.

**Option E — None of the above:** ❌ Wrong
A, C, and D are all correct.

> **Key takeaway:** Bonferroni controls FWER (probability of any false discovery) — very conservative. BH controls FDR (proportion of false discoveries) — less conservative, more powerful, more discoveries, but also more false positives in absolute terms.

---

### Question 11 — Definition of $\|\beta\|_2^2$

**Question:** What is $\|\beta\|_2^2$, with $\beta = (\beta_1, \ldots, \beta_n)^T$, equal to?

**Official Answer:** B and C

**Option A — $\sum_i |\beta_i|$:** ❌ Wrong
This is the L1 norm $\|\beta\|_1$, not the squared L2 norm. The L1 norm sums the absolute values of the components and is used in lasso regularisation. It is fundamentally different from the squared L2 norm.

**Option B — $\sum_i \beta_i^2$:** ✓ Correct
By definition, the L2 norm is $\|\beta\|_2 = \sqrt{\sum_i \beta_i^2}$, so the squared L2 norm is $\|\beta\|_2^2 = \sum_i \beta_i^2$. This is the sum of squares of the components — the ridge penalty.

**Option C — $\beta^T\beta$:** ✓ Correct
For a column vector $\beta$, the inner product $\beta^T\beta = \sum_i \beta_i^2$, which is exactly the sum of squares. This is an equivalent representation of $\|\beta\|_2^2$ using matrix/vector notation. Options B and C are two different notations for the same quantity.

**Option D — $\max_i \beta_i^2$:** ❌ Wrong
This would be the square of the L$\infty$ (infinity) norm $\|\beta\|_\infty = \max_i |\beta_i|$. The L$\infty$ norm takes the largest absolute component, not the sum of squares. It is not used in standard regularisation penalties for regression.

**Option E — None of the above:** ❌ Wrong
Both B and C are correct representations of $\|\beta\|_2^2$.

> **Key takeaway:** $\|\beta\|_2^2 = \sum_i \beta_i^2 = \beta^T\beta$. The L1 norm is $\|\beta\|_1 = \sum_i |\beta_i|$. The L2 norm squared (ridge penalty) and L1 norm (lasso penalty) produce fundamentally different shrinkage behaviour.

---

### Question 12 — Self-Organising Maps and Blessings of Dimensionality

**Question:** One could argue that Self-Organising Maps are especially good at illustrating one of the blessings of dimensionality (Donoho 2000). Which one?

**Official Answer:** B

**Option A — Features will be correlated:** ❌ Wrong
Feature correlation is a general property of real-world datasets but is not specifically a "blessing of dimensionality" in Donoho's framework. While SOMs can represent correlated structure, this is not their defining contribution to illustrating the blessings of dimensionality.

**Option B — Informative data will lie on a low-dimensional manifold:** ✓ Correct
Donoho's "blessing" here is the manifold hypothesis: even when data nominally lives in a high-dimensional space (e.g., images with thousands of pixels), the *informative* variation is often constrained to a low-dimensional manifold embedded in that space. SOMs explicitly learn a low-dimensional grid (typically 2D) that is organised to reflect the manifold structure of the data. The topology-preserving property of SOMs means nearby points on the map correspond to similar points in high-dimensional space, directly illustrating that meaningful variation is lower-dimensional.

**Option C — Underlying structure in data will give approximative finite dimensionality:** ❌ Wrong
This statement is related to B but is less precise. "Approximative finite dimensionality" refers to the idea that the intrinsic dimensionality of data is finite and much less than the ambient dimensionality. While true, SOMs specifically illustrate the manifold structure (the actual geometric shape of the data cloud), not just that dimensionality is finite. Option B is the more precise and direct connection.

**Option D — Curses and blessings of dimensionality is only an issue for supervised learning:** ❌ Wrong
This is factually incorrect. Both the curses (e.g., the concentration of measure, nearest-neighbours becoming equidistant) and blessings (manifold hypothesis, sparsity of natural data) of dimensionality apply to unsupervised learning (clustering, dimensionality reduction, density estimation) as well as supervised learning. SOMs are themselves an unsupervised method.

**Option E — None of the above:** ❌ Wrong
Option B is correct.

> **Key takeaway:** SOMs illustrate the manifold hypothesis — that high-dimensional data often lies on or near a low-dimensional manifold. This is one of Donoho's blessings of dimensionality because it makes learning tractable despite high ambient dimension.

---

### Question 13 — Covid-19 Test: Expected Positive Tests in a Population

**Question:** You are evaluating your new Covid-19 test and you have obtained the following confusion matrix:

|  | Predicted Covid | Predicted No Covid |
|---|---|---|
| **Actual Covid** | 99% | 1% |
| **Actual No Covid** | 2% | 98% |

Now we are testing in a population of 10,000 subjects where we expect 100 to have Covid. How many subjects do we expect to have a positive test?

**Official Answer:** D (297)

**Option A — 99:** ❌ Wrong
This counts only the true positives: $100 \times 0.99 = 99$ people who actually have Covid and test positive. But it ignores the false positives — people without Covid who also test positive. In a population of 9,900 non-Covid individuals, 2% will also test positive, contributing a large number of additional positives.

**Option B — 100:** ❌ Wrong
This would be the correct answer only if the test were perfect (100% sensitivity, 0% false positive rate). In reality, the test has a 2% false positive rate applied to 9,900 healthy individuals, generating many additional spurious positives.

**Option C — 200:** ❌ Wrong
This answer does not correspond to any straightforward calculation from the confusion matrix. It might arise from incorrectly applying the 2% false positive rate to the full population of 10,000 ($200$) and ignoring everything else, but this is not correct.

**Option D — 297:** ✓ Correct
The correct calculation applies both sensitivity and false positive rate to their respective subpopulations:
- **True positives:** $100 \times 0.99 = 99$ (Covid-positive individuals who test positive)
- **False positives:** $9{,}900 \times 0.02 = 198$ (Non-Covid individuals who incorrectly test positive)
- **Total positive tests:** $99 + 198 = \mathbf{297}$

This is a direct application of the law of total probability. Note that the majority of positive tests (198 out of 297) are false positives — a striking illustration of how low disease prevalence makes specificity critically important.

**Option E — 300:** ❌ Wrong
This is close to the correct answer but not exact. It might arise from rounding $9{,}900 \times 0.02 \approx 200$ and adding $100$, but the precise calculation gives 297, not 300.

> **Key takeaway:** Total positives = (prevalence × sensitivity) + (1 − prevalence) × FPR. When prevalence is low, false positives from the large healthy population can dominate total positives, making specificity critically important for test utility.

---

### Question 14 — Random Forest Bias and Variance (All Variables Informative, m=5)

**Question:** We simulate data with 50 variables and 100 observations, where each variable is simulated from a random standard normal Gaussian, and the response variable is given as a weighted sum of all 50 variables, with weights $1/\sqrt{50}$, and white noise added. For a Random Forest ensemble model with the hyperparameter of random selection of variables, $m = 5$ variables, what can you expect to hold for the bias and variance?

**Official Answer:** A and D

**Option A — The variance component is lower than the bias component in the expected prediction error of the ensemble:** ✓ Correct
In this specific setup, *all 50 variables are informative* (each contributes equally to the response). With $m = 5$ random features per split, each tree only ever considers 5 out of 50 relevant predictors at each node. This means each individual tree consistently misses 45 of the 50 signal-carrying variables at each split, introducing systematic underfitting. The bias of a Random Forest equals the bias of a single tree: $\text{Bias}_{RF} = \text{Bias}_{\text{single tree}}$. With most relevant variables excluded from each split, the bias is high. Meanwhile, averaging many trees in the ensemble substantially reduces variance compared to a single tree. Therefore, bias dominates variance in EPE.

**Option B — The variance component is higher than the bias component in the expected prediction error of the ensemble:** ❌ Wrong
Averaging in Random Forest aggressively reduces variance relative to a single tree. In this problem where all variables are informative, the primary limitation is bias (from random feature exclusion), not variance. Variance dominates in low-bias, high-variance settings (e.g., when only a few variables are relevant and m is large enough to often include them).

**Option C — The variance and bias components in the expected prediction error of the RF are of the same size:** ❌ Wrong
This is a specific quantitative claim that does not follow from the setup. Given $m = 5$ out of 50 fully-informative variables, the systematic bias from missing 45/50 variables at each split is substantial and clearly dominates the variance (which is suppressed by averaging).

**Option D — The variance of a single tree is larger than the variance of the ensemble:** ✓ Correct
This is a fundamental property of any ensemble via the bias-variance decomposition of averages. For $T$ independent trees with variance $\sigma^2$ and correlation $\rho$, the ensemble variance is:
$$\text{Var}(\hat{f}_{RF}) = \rho \sigma^2 + \frac{1-\rho}{T} \sigma^2$$
As $T \to \infty$, this approaches $\rho \sigma^2 < \sigma^2$ (since $\rho < 1$ due to random feature selection). Even for finite $T$, the ensemble variance is always less than or equal to the variance of a single tree. Random feature selection specifically reduces $\rho$ (decorrelates trees), further reducing ensemble variance.

**Option E — None of the above:** ❌ Wrong
Both A and D are correct.

> **Key takeaway:** RF bias = single-tree bias (unaffected by averaging). RF variance < single-tree variance (always reduced by averaging). When all features are informative and $m \ll p$, random feature exclusion introduces high bias that dominates the EPE.

---

### Question 15 — Archetypical Analysis and the Six Datasets

**Question:** Figure 1 illustrates 6 two-dimensional simulated datasets and their Archetypical Analysis solution with two archetypes (variance explained shown in each panel):
- Dataset a: VE = 0.91521
- Dataset b: VE = 0.92767
- Dataset c: VE = 0.87505
- Dataset d: VE = 0.8114
- Dataset e: VE = 0.9387
- Dataset f: VE = 0.97211

Which of the following statements are true?

**Official Answer:** B

⚠️ **Note:** The official answer B is technically correct but trivially so — it is a tautology for 2D data. See explanation below.

**Option A — Archetypical Analysis with two components describes more of the variance in dataset c, than in any of the other datasets:** ❌ Wrong
From the VE values: dataset c has VE = 0.875, while datasets a (0.915), b (0.928), e (0.939), and f (0.972) all have higher VE than c. Dataset c has the *second lowest* VE, only above d (0.811). This statement is directly refuted by the numbers in the figure.

**Option B — Singular Value Decomposition (SVD) with two components describes all the variance for all the datasets:** ✓ Correct
⚠️ This is trivially true: all six datasets are *two-dimensional* ($p=2$). SVD decomposes an $n \times p$ matrix into components, and with two SVD components applied to 2D data, all variance is always captured (since 2 components = full rank of a 2-column matrix). This is a mathematical tautology, not a meaningful statistical insight — but it is technically correct and is the official answer. The question tests whether students recognise that SVD with $p$ components in $p$ dimensions explains 100% of variance by definition.

**Option C — K-means clustering with three components is an appropriate choice for datasets d and f:** ❌ Wrong
Looking at the figures: dataset d has a triangular/wedge distribution that might suggest multiple clusters, but three clusters is not obviously motivated. Dataset f appears to have an elongated/banana-shaped single cluster with high VE (0.972). K-means with 3 components imposes spherical cluster shapes (minimises squared Euclidean distances), which is inappropriate for the elongated structure in these datasets. There is no visual evidence for 3 clear spherical clusters in either d or f.

**Option D — K-means clustering with two components is an appropriate choice for datasets a, b, and c:** ❌ Wrong
Datasets a, b, and c appear to have elongated, approximately linear distributions (the archetypes span the ends of the data cloud, and VE for AA is high). K-means with 2 clusters would attempt to find two spherical clusters, which is not well-suited to elongated or arc-shaped distributions. Datasets a, b, c look more like they have linear manifold structure rather than two distinct spherical groups.

**Option E — None of the above:** ❌ Wrong
Option B is correct.

> **Key takeaway:** SVD (PCA) with $k$ components on $k$-dimensional data explains 100% of the variance by definition — this is a tautology. Two-dimensional data always has its full variance explained by 2 SVD components. This question tests whether students catch this trivial fact.

---

### Question 16 — Core Consistency Diagnostic (CORCONDIA)

**Question:** Which of the following statements about the Core Consistency Diagnostic (CORCONDIA) given by:
$$\text{CORCONDIA} = 100 \cdot \left(1 - \frac{\|\mathcal{I} - \mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$
are true?

**Official Answer:** E (None of the above)

**Option A — $\mathcal{I}$ is a diagonal matrix:** ❌ Wrong
$\mathcal{I}$ in CORCONDIA is the *super-identity tensor* — a multi-way array (tensor) where the super-diagonal elements are 1 and all others are 0. For a 3-way tensor with $R$ components, $\mathcal{I}$ is an $R \times R \times R$ tensor. It is not a diagonal *matrix* (2-dimensional); the distinction between a super-diagonal tensor and a diagonal matrix is critical here.

**Option B — $\mathcal{G}$ is the core tensor obtained from a Tucker decomposition of the multiway data matrix $\mathcal{X}$ of interest:** ❌ Wrong
$\mathcal{G}$ in CORCONDIA is specifically the core tensor obtained from fitting a *PARAFAC* model and then extracting the Tucker core using the PARAFAC component matrices. It is not obtained from a direct Tucker decomposition. The CORCONDIA formula measures how close the PARAFAC-derived core is to the super-identity tensor $\mathcal{I}$ — if PARAFAC is a good fit, $\mathcal{G} \approx \mathcal{I}$ and CORCONDIA $\approx 100$.

**Option C — CORCONDIA basically measures whether we should choose a PARAFAC or a Tucker model for our data:** ❌ Wrong
While CORCONDIA can inform the choice between PARAFAC and Tucker in a loose sense, its primary purpose is to determine the *number of components* to use in a PARAFAC model. High CORCONDIA (close to 100) means the PARAFAC model with the chosen number of components is well-suited. CORCONDIA does not directly test whether Tucker is preferable to PARAFAC as a model family.

**Option D — We choose the minimum value of CORCONDIA to select the number of components to use in PARAFAC:** ❌ Wrong
This is the opposite of the correct rule. We choose the number of PARAFAC components by looking for the *maximum* CORCONDIA value (close to 100) while increasing $R$. A drop in CORCONDIA below approximately 90 indicates that the added component introduces non-super-diagonal structure, suggesting the chosen $R$ is too large. The rule is to select the *largest $R$ for which CORCONDIA remains high (close to 100)*, not the minimum.

**Option E — None of the above:** ✓ Correct
All four specific statements (A through D) contain factual errors. Therefore, the correct answer is E.

> **Key takeaway:** CORCONDIA measures how close the PARAFAC core tensor is to the super-identity tensor (not diagonal matrix). High CORCONDIA (near 100) indicates a good PARAFAC fit; we seek the highest $R$ still giving high CORCONDIA. $\mathcal{G}$ comes from the PARAFAC model, not a Tucker decomposition.

---

### Question 17 — Subspace Methods: PCA, PLS, and CCA

**Question:** Which of the following statements are true for the subspace methods using derived input directions/latent variables?

**Official Answer:** A, B, and D

⚠️ **Note on B:** Whether PLS and CCA impose orthogonality in the same sense as PCA is debatable. See explanation below.

**Option A — Principal component analysis, partial least squares, and canonical correlation analysis ALL result in latent variables constructed of linear combinations of the input variables:** ✓ Correct
All three methods produce latent variables (scores) that are linear projections of the original input variables:
- PCA: $z = Xw$ where $w$ is an eigenvector of $X^TX$
- PLS: $t = Xw$ where $w$ maximises $\text{Cov}(Xw, Yc)$
- CCA: $u = Xa$ where $a$ maximises $\text{Cor}(Xa, Yb)$

In all three cases, the latent variable is a weighted linear combination of the input columns.

**Option B — PCA, PLS, and CCA ALL impose orthogonality between the produced latent variables:** ✓ Correct (in the course framework, with caveats)
PCA guarantees orthogonal principal component scores. PLS ensures orthogonal X-scores ($t_i^T t_j = 0$ for $i \neq j$). CCA canonical variates are orthogonal in the sense that $u_i^T \Sigma_{XX} u_j = 0$ (orthogonal with respect to the metric $\Sigma_{XX}$). The course treats all three as imposing some form of orthogonality among components. ⚠️ The precise form of orthogonality differs: PCA uses standard Euclidean orthogonality, PLS uses orthogonal scores, and CCA uses orthogonality in the covariance-weighted inner product. Students should know that B is accepted as correct in this course.

**Option C — PCA, PLS, and CCA are ALL unsupervised methods:** ❌ Wrong
PCA is unsupervised — it only uses the input matrix $X$ and ignores any response $Y$. However, PLS and CCA are *supervised* (or more precisely, multi-view) methods: PLS seeks directions in $X$ that maximally covary with $Y$; CCA seeks directions in both $X$ and $Y$ that are maximally correlated. Both require a response or second variable set $Y$ and are not unsupervised.

**Option D — It is possible to use the elastic net to produce sparse versions of ALL the methods: PCA, PLS, and CCA:** ✓ Correct
Sparse versions of all three methods exist using the elastic net (L1+L2) penalty on the loading/weight vectors:
- Sparse PCA (Zou et al. 2006): penalises PCA loadings with elastic net
- Sparse PLS: penalises PLS weights
- Sparse CCA: penalises CCA canonical vectors

The elastic net penalty $\lambda_1 \|\cdot\|_1 + \lambda_2 \|\cdot\|_2^2$ combines sparsity-inducing (L1) and stability-inducing (L2) properties, making it applicable to all three methods.

**Option E — None of the above:** ❌ Wrong
A, B, and D are all correct.

> **Key takeaway:** PCA, PLS, and CCA all produce linear combinations of inputs and all impose some form of orthogonality. PLS and CCA are supervised (not PCA). Sparse versions using elastic net exist for all three.

---

### Question 18 — Cross-Validation for Wearable Activity Prediction

**Question:** We have measured accelerometer data from a wearable for 45 individuals between 30 and 55 (our target group) over 7 weeks. We want to develop a model that can predict the activity level for the included as well as new individuals next week. For simplicity we sum up the motion data over each day and use a boosting algorithm to predict the total activity level for a new day. As input features we use activity levels for the past 7 days (for each individual) as well as age, gender, job function and distance to work. As response we use the activity level for each day with a history of 7 previous recording days. Given our aim, which of the following method(s) can help us assess the prediction accuracy of our boosting model?

**Official Answer:** C and D

**Option A — A 5-fold cross-validation:** ❌ Wrong
Standard 5-fold CV randomly assigns individual daily observations to folds. This means data from the same person will appear in both training and test folds across different iterations. When the model is tested on a day from person $i$, it will have been trained on other days from person $i$ — giving an optimistic performance estimate that does not reflect generalisation to *new individuals*. The goal includes predicting for new individuals, so individual-level data leakage is a critical problem here.

**Option B — A leave-one-observation-out cross-validation:** ❌ Wrong
This is even more problematic than 5-fold CV. Leaving out a single daily observation while training on all other observations — including other days from the same person — maximises the data leakage between the held-out observation and the training data. Adjacent days from the same person are highly correlated (activity level is temporally autocorrelated), so this CV scheme greatly overstates generalisation performance.

**Option C — A leave-five-individuals-out cross-validation:** ✓ Correct
By holding out complete individuals (all their days), the training set contains no data from the held-out individuals, properly simulating the "new individual" scenario. Each fold tests the model on people the model has never seen, which matches the stated goal of predicting for "new individuals." With 45 individuals, leaving out 5 at a time creates 9 folds.

**Option D — A leave-one-week-out test set:** ✓ Correct
The model is also meant to predict "next week" for all individuals (including those in the training set). Holding out all data from one week while training on the preceding weeks tests the model's ability to generalise to a future time point — directly matching the "predict next week" goal. This also respects the temporal ordering of the data, preventing future data from leaking into training.

**Option E — None of the above:** ❌ Wrong
Both C and D are appropriate methods for the stated goal.

> **Key takeaway:** When the goal is to generalise to both new individuals and future time points, cross-validation must respect both the individual structure (leave whole individuals out) and the temporal structure (leave future weeks out). Standard CV that randomly shuffles observations violates both and produces overly optimistic estimates.

---

### Question 19 — Removed from 2023 Curriculum

**Not applicable.** This question was part of the 2022 exam but is explicitly removed from the 2023 preparation materials as it is no longer part of the curriculum.

---

### Question 20 — Suitable Models for Boosting Ensembles

**Question:** Which of the following methods are suitable to use as the individual models in a boosting ensemble?

**Official Answer:** C (Any classification or regression tree)

⚠️ **Warning — Answer sheet anomaly:** The official answer sheet appears to mark both A and E for Q20, which is internally contradictory (A = "KNN with high K"; E = "None of the above"). This is almost certainly a grid transcription or rendering error in the official document. The pedagogically correct answer based on the theory of boosting is **C**.

**Option A — K-Nearest Neighbors with a high number of K:** ❌ Wrong
KNN with high $K$ is a low-variance, high-bias model (it averages over many neighbours, producing smooth predictions). Boosting works by sequentially correcting the *errors* of previous learners by reweighting observations. KNN is not naturally amenable to gradient boosting because: (1) it has no gradient-based update mechanism, (2) high-K KNN has too much bias to be an effective weak learner, and (3) KNN cannot easily be implemented as part of the forward stagewise additive framework used by gradient boosting.

**Option B — K-Nearest Neighbors with a low number of K:** ❌ Wrong
Low-K KNN is high-variance and low-bias, which is closer to the desired profile of a weak learner in terms of bias, but KNN still lacks the tree-splitting mechanism that makes decision trees ideal for boosting. More importantly, KNN cannot be efficiently fit as part of the residual-fitting framework of gradient boosting (where each weak learner fits the pseudo-residuals of the previous ensemble).

**Option C — Any kind of classification or regression tree:** ✓ Correct
Decision trees (especially shallow trees called "stumps" — trees with a single split) are the canonical weak learners in boosting. They are ideal because: (1) they can model non-linear interactions; (2) shallow trees have high bias and low variance (appropriate for weak learners); (3) they naturally decompose the feature space in ways that allow gradient boosting to fit residuals; (4) tree depth is an interpretable hyperparameter that controls the bias-variance tradeoff of the weak learner; (5) the forward stagewise additive model framework (AdaBoost, Gradient Boosting, XGBoost) is built around trees.

**Option D — Any kind of linear model:** ❌ Wrong
Linear models can theoretically be used as base learners in boosting, but they are not "suitable" in the practical sense. Boosting of linear models does not capture non-linear interactions without explicit feature engineering, and the ensemble of many linear models is still linear. The power of boosting comes from combining non-linear base learners (trees) to capture complex patterns.

**Option E — None of the above:** ❌ Wrong (despite the apparent marking on the answer sheet)
Option C is the correct answer. The simultaneous marking of A and E on the answer sheet is contradictory and is flagged as a transcription error.

> **Key takeaway:** Boosting uses weak learners that have high bias and low variance — shallow decision trees (stumps) are the canonical choice. KNN cannot be efficiently integrated into the gradient boosting framework, and linear models lose the non-linear advantage of boosting.

---

## Open Questions

---

### Question 21 — The Random Forest Algorithm

**Question:** Describe the steps in the Random Forest Algorithm and how they contribute to the performance of the algorithm.

---

**Model Answer (targeting full 10 marks):**

Random Forest is a supervised ensemble learning method that builds multiple decision trees and aggregates their predictions. The algorithm has four core steps, each contributing to performance in a specific way.

#### Step 1: Bootstrap Sampling (Bagging)

For each of the $B$ trees, draw a bootstrap sample $\mathcal{D}_b^*$ of size $n$ with replacement from the training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$. Approximately $1 - (1 - 1/n)^n \approx 63.2\%$ of observations appear in each bootstrap sample; the remaining $\approx 36.8\%$ form the out-of-bag (OOB) set for that tree.

**Contribution:** Bootstrap sampling creates diversity among the trees. Because each tree is trained on a different sample, the trees learn different aspects of the data. When predictions from these diverse trees are averaged, the variance of the ensemble is reduced relative to a single tree:
$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B \hat{f}_b(x)\right) = \frac{\sigma^2}{B}$$
for independent trees with variance $\sigma^2$.

#### Step 2: Random Feature Selection at Each Split

At each node of each tree, select a random subset of $m$ features from the full $p$ features (typically $m = \sqrt{p}$ for classification, $m = p/3$ for regression). The best split among only these $m$ features is used.

**Contribution:** This step decorrelates the trees. Without random feature selection, all trees would tend to use the same dominant features at the top splits, making them highly correlated. Correlated trees do not reduce variance when averaged: for $B$ trees with pairwise correlation $\rho$, the ensemble variance is:
$$\text{Var}_{RF} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$
Reducing $\rho$ (by randomly excluding features) makes the second term dominant and drives variance toward zero as $B \to \infty$.

#### Step 3: Grow Deep, Unpruned Trees

Each tree is grown to full depth (no pruning), so each leaf contains the minimum number of observations (typically 1 for regression, 1 for classification).

**Contribution:** Deep trees have low bias but high variance. Because averaging in the ensemble handles the variance, we want each individual tree to be as accurate as possible (low bias). RF bias = individual tree bias, so we grow deep trees to minimise bias. This is the opposite of boosting, where we deliberately use shallow, high-bias weak learners.

#### Step 4: Aggregate Predictions

For regression, average the predictions of all $B$ trees:
$$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x)$$
For classification, use majority vote:
$$\hat{y}(x) = \text{mode}\left(\hat{y}_1(x), \hat{y}_2(x), \ldots, \hat{y}_B(x)\right)$$

**Contribution:** Averaging reduces variance without changing bias (since $E[\bar{f}] = E[f_b] = f_b$ for unbiased estimators). The law of large numbers ensures that as $B \to \infty$, the ensemble converges to the expected prediction of a single tree, with variance approaching $\rho \sigma^2$.

#### Additional Features

**Out-of-Bag Error Estimation:** For each observation $i$, predictions from trees for which $i$ was OOB provide a free cross-validation-like estimate of generalisation error — no separate test set is required:
$$\hat{\epsilon}_{OOB} = \frac{1}{n}\sum_{i=1}^n L\left(y_i, \hat{f}_{OOB}^{(-i)}(x_i)\right)$$

**Variable Importance:** Two main measures:
1. *Gini importance:* Sum the decrease in Gini impurity at all splits involving variable $j$, averaged across all trees. Higher sum = more important variable.
2. *Permutation importance:* For variable $j$, permute its values in the OOB data, record the increase in OOB error. Larger increase = higher importance. Permutation importance is more reliable because it directly measures prediction degradation.

#### Performance Summary

Random Forest achieves strong performance by combining:
- **Low variance:** via averaging (bagging) and decorrelation (random features)
- **Low bias:** via deep individual trees
- **Robustness:** to irrelevant features and missing data
- **No overfitting** as $B$ increases: the OOB error converges

The key trade-off is that random feature selection introduces bias (by sometimes missing relevant features) — this effect is pronounced when all features are informative.

---

### Question 22 — Identifying Unique People from Passport Control Face Images

**Question:** You are given a dataset consisting of images of faces that have been taken in the passport control in Copenhagen Airport. Security would like to know how many unique people have entered Denmark and compare it to the unique passport numbers that entered in the same time period, as they fear a systematic fraud is happening. Which methodology, and why, would you use to analyse the data and find the number of unique people from the available face images to help security?

---

**Model Answer (targeting full 10 marks):**

This is an **unsupervised learning problem**: we have a collection of face images with no labels (no known identities), and our goal is to determine the number of unique individuals. The core challenge is (1) extracting meaningful features from images and (2) grouping images by identity without knowing identities in advance.

#### Step 1: Feature Extraction

Raw pixel values are high-dimensional and contain irrelevant variation (lighting, pose, expression). We need a compact, discriminative representation:

**Option A — PCA / Eigenfaces (classical approach):**
Flatten each image into a vector $x_i \in \mathbb{R}^p$ (e.g., $100 \times 100$ pixels = 10,000 dimensions). Apply PCA (SVD) to find the principal directions of variation (eigenfaces):
$$X \approx U_k S_k V_k^T$$
Project each image onto the first $k$ principal components to obtain feature vectors $z_i = V_k^T x_i \in \mathbb{R}^k$ where $k \ll p$. This dramatically reduces dimensionality while retaining the dominant facial variation.

**Option B — Deep Learning Embeddings (modern approach):**
Use a pre-trained deep convolutional neural network (e.g., FaceNet, ArcFace) to extract a compact embedding vector (e.g., 128-dimensional) for each face image. These embeddings are specifically trained to map faces of the same person close together and faces of different people far apart. This is more powerful than PCA for this task but requires pre-trained models.

For an exam answer in this course, PCA/SVD is the primary recommended approach.

#### Step 2: Clustering to Find Unique Identities

Once feature vectors $z_1, z_2, \ldots, z_N \in \mathbb{R}^k$ are extracted (one per image), cluster them to group images of the same person together:

**Gaussian Mixture Models (GMM):**
$$p(z) = \sum_{j=1}^{K} \pi_j \mathcal{N}(z \mid \mu_j, \Sigma_j)$$
Each component represents one unique individual. The number of components $K$ equals the number of unique people. GMM is probabilistic and allows soft assignment (an ambiguous image may have non-zero probability of belonging to multiple clusters). Fit using the EM algorithm; select $K$ using BIC:
$$\text{BIC} = -2\log\hat{L} + K \cdot d \cdot \log(N)$$
where $d$ is the number of parameters per component. The optimal $K$ minimises BIC.

**K-means:**
Minimise $\sum_{k=1}^K \sum_{z_i \in C_k} \|z_i - \mu_k\|_2^2$. More computationally efficient but assumes spherical clusters. Select $K$ via the elbow method (plot within-cluster sum of squares vs. $K$) or silhouette score. The silhouette coefficient for observation $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance. Maximise the average silhouette over $K$.

**Hierarchical Clustering:**
Build a dendrogram using, e.g., Ward linkage on Euclidean distances in feature space. Cut the dendrogram at the appropriate level to identify clusters. The optimal cut can be identified by looking for the largest gap in merge distances.

#### Step 3: Determining the Number of Unique People ($K$)

- **For GMM:** minimise BIC over $K$
- **For K-means:** maximise silhouette score, or use the elbow method
- **For hierarchical clustering:** identify the largest gap in dendrogram merge heights

The optimal $K$ from clustering gives the estimated number of unique individuals in the airport images.

#### Step 4: Comparing to Passport Numbers

Once $K$ (unique faces) is estimated, compare it to the number of unique passport numbers recorded in the same period. If $K_{\text{faces}} < K_{\text{passports}}$, this suggests multiple passport numbers are associated with the same physical person — i.e., potential fraud (one person using multiple passports). If $K_{\text{faces}} \approx K_{\text{passports}}$, the data is consistent with no fraud.

#### Recommended Methodology and Justification

**Recommended pipeline:** PCA for feature extraction + GMM for clustering with BIC for model selection.

**Why PCA?** Face images are high-dimensional and collinear (pixels in a face are spatially correlated). PCA finds the principal axes of variation (eigenfaces), reducing dimensionality from thousands to tens of components while retaining most variance. This makes subsequent clustering tractable and removes noise.

**Why GMM?** Face images of the same person vary in lighting, pose, and expression — this produces an approximately elliptical cluster in feature space, which GMM can model with full covariance matrices (unlike K-means which assumes spherical clusters). Soft assignment is natural since borderline images may be ambiguous. BIC provides a principled criterion for selecting $K$.

**Why unsupervised?** We have no labels (no known identities). This is inherently an unsupervised problem. No amount of supervised classification can help here because the classes are unknown. The goal is *discovery* of natural groupings, not prediction against known categories.

#### Limitations to Acknowledge

1. **Image quality:** Low-quality images (blurry, occluded) may not cluster correctly.
2. **Non-Gaussian clusters:** Extreme pose variation may produce non-Gaussian distributions in feature space.
3. **Determining $K$:** All methods give a heuristic estimate; none is definitive.
4. **Rare individuals:** People with very few images may not form clear clusters and could be merged with others.

---

## Errors Found in Official Solutions

### Q9 — Intercept Penalisation (Answer Sheet Notation Ambiguity)
**Issue:** The answer sheet marks A, B, and C with "(x)" notation, suggesting debate or partial credit. Option B ("penalising the intercept would introduce variance without any reduction in bias") is incorrect — the primary effect is bias, not variance. The standard, taught explanation is A. D is correct as a consequence of A.
**Correct answer:** A (and D as a derived consequence). B is wrong.

### Q15 — SVD Variance Statement (Trivially True)
**Issue:** The official answer B ("SVD with two components describes all the variance for all the datasets") is technically correct but is a mathematical tautology for 2D data. Any full-rank decomposition of a 2-column matrix with 2 components explains 100% of variance. This tests recognition of the tautology rather than conceptual understanding of SVD.
**Note:** The answer is technically correct; students should know it is trivially true for 2D data.

### Q20 — Boosting Model Answer (Grid Transcription Error)
**Issue:** The official answer sheet appears to mark both A ("KNN with high K") and E ("None of the above") for Q20. These are internally contradictory — if A is correct, E cannot be correct, and vice versa. This is almost certainly a grid rendering or transcription error.
**Correct answer:** C (Any classification or regression tree). Shallow decision trees (stumps) are the canonical weak learner in boosting. KNN is not suitable for boosting because it lacks a gradient-based update mechanism compatible with the forward stagewise additive framework.
