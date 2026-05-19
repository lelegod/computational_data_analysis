# Exam Solutions — CDA 02582 (2025)

**Date:** May 19, 2025
**Format:** 20 MC questions (single correct answer, a–e) + 2 open questions
**Scoring:** Single correct answer per MC question

---

## Multiple Choice Questions

---

### Question 1 — Bias-Variance Decomposition

**Question:** In the bias-variance decomposition of prediction error, which component is not directly affected by model complexity?

**Official Answer:** (c) Irreducible error

**Option A — Variance:** ❌ Wrong
Variance is directly and strongly affected by model complexity. A more complex model (e.g., high-degree polynomial, small $k$ in KNN) fits each training dataset more closely, causing predictions to vary widely across different training sets. The formal definition is $\text{Var}[\hat{f}(x_0)] = E[(\hat{f}(x_0) - E[\hat{f}(x_0)])^2]$, which grows as complexity increases.

**Option B — Bias:** ❌ Wrong
Bias is directly affected by model complexity — in the opposite direction from variance. Simpler models (e.g., linear fit on nonlinear data) have high bias because they systematically miss the true function: $\text{Bias}^2 = (E[\hat{f}(x_0)] - f(x_0))^2$. As complexity increases, bias decreases.

**Option C — Irreducible error:** ✓ Correct
The irreducible error $\sigma^2 = \text{Var}[\varepsilon]$ is the variance of the noise in the data-generating process $y = f(x) + \varepsilon$. No matter how complex or simple the model, this component cannot be reduced — it is a fixed property of the problem. The full decomposition is:
$$\text{EPE}(x_0) = \text{Bias}^2[\hat{f}(x_0)] + \text{Var}[\hat{f}(x_0)] + \sigma^2$$

**Option D — Training error:** ❌ Wrong
Training error is very much affected by model complexity — it strictly decreases as complexity increases (a complex enough model can memorize the training data, achieving near-zero training error). This is precisely why training error is a poor proxy for generalization error.

**Option E — Expected prediction error:** ❌ Wrong
EPE is the sum of bias$^2$, variance, and irreducible error. Since both bias and variance change with complexity, EPE changes with complexity. The classic U-shaped EPE curve (high for very simple models, high again for very complex models, with a sweet spot in between) demonstrates this directly.

> **Key takeaway:** The irreducible error $\sigma^2$ is a fixed property of the data, not the model — it sets a hard lower bound on any model's expected prediction error regardless of complexity.

---

### Question 2 — Kernel Trick in SVM

**Question:** The kernel trick in SVM allows:

**Official Answer:** (b) Computation in high-dimensional feature space without explicit transformation

**Option A — Automatic feature selection:** ❌ Wrong
The kernel trick performs no feature selection whatsoever. It maps data into a (possibly infinite-dimensional) feature space and computes inner products there implicitly, but all features are used and none are eliminated. Feature selection is a separate concern handled by methods like Lasso or wrapper methods.

**Option B — Computation in high-dimensional feature space without explicit transformation:** ✓ Correct
The kernel trick relies on Mercer's theorem: if $K(x, x') = \langle \phi(x), \phi(x') \rangle$ for some mapping $\phi$, then the SVM dual objective only requires computing $K(x_i, x_j)$ in the original input space, never materializing $\phi(x)$ explicitly. This allows implicit operation in feature spaces of arbitrarily high (or infinite) dimension, as with the RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$.

**Option C — Better regularization:** ❌ Wrong
The kernel trick does not inherently improve regularization. Regularization in SVM is controlled by the margin parameter $C$ (or equivalently the slack variables). The kernel choice determines the geometry of the decision boundary, not the regularization strength.

**Option D — Classification without using margin-defining data samples:** ❌ Wrong
This is the opposite of how SVM works. The SVM decision boundary is defined entirely by the support vectors — the training points that lie on or inside the margin. The kernel trick does not remove the need for support vectors; the dual formulation still relies on them via the coefficients $\alpha_i \neq 0$ for support vectors.

**Option E — Faster training:** ❌ Wrong
The kernel trick generally does not speed up training. In fact, using a nonlinear kernel can be computationally more expensive because the kernel matrix $K \in \mathbb{R}^{n \times n}$ must be computed and stored, with training complexity scaling as $O(n^2)$ to $O(n^3)$.

> **Key takeaway:** The kernel trick $K(x, x') = \phi(x)^T\phi(x')$ lets SVM work in high-dimensional (even infinite-dimensional) feature spaces by computing inner products in the original space, avoiding the explicit computation of $\phi(x)$.

---

### Question 3 — Matrix Factorization Methods

**Question:** Which of the following methods is not matrix factorization-based?

**Official Answer:** (d) K-means

**Option A — NMF:** ❌ Wrong (NMF IS matrix factorization-based)
Non-negative Matrix Factorization explicitly decomposes $X \approx WH$ where $W \in \mathbb{R}^{n \times r}_{\geq 0}$ and $H \in \mathbb{R}^{r \times p}_{\geq 0}$. This is definitionally a matrix factorization — it factors the data matrix into two non-negative factor matrices.

**Option B — PCA:** ❌ Wrong (PCA IS matrix factorization-based)
PCA is founded on the Singular Value Decomposition $X = U\Sigma V^T$, which is one of the most fundamental matrix factorizations. The principal components are the columns of $V$ (right singular vectors), and the scores are $XV = U\Sigma$.

**Option C — ICA:** ❌ Wrong (ICA IS matrix factorization-based)
ICA assumes $X = AS$ where $A$ is an unknown mixing matrix and $S$ contains independent source signals. The goal is to recover $A^{-1}$ (the unmixing matrix) and the sources $S$. This is a matrix factorization under statistical independence constraints.

**Option D — K-means:** ✓ Correct
K-means is a clustering algorithm, not a matrix factorization. It partitions $n$ observations into $k$ clusters by minimizing the within-cluster sum of squared Euclidean distances: $\sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$. There is no factorization of the data matrix; instead, each point is assigned a hard cluster label. (Note: while a loose connection to NMF with binary indicators can be constructed, K-means is not natively expressed or typically understood as matrix factorization in this course.)

**Option E — Archetypal Analysis:** ❌ Wrong (AA IS matrix factorization-based)
Archetypal Analysis decomposes $X \approx ZB X$ where $Z \in \mathbb{R}^{n \times r}$ gives the mixture weights (each row sums to 1, non-negative) and $B \in \mathbb{R}^{r \times n}$ defines the archetypes as convex combinations of data points. This is a constrained matrix factorization.

> **Key takeaway:** K-means is the only listed method that works by iterative cluster assignment (distance-to-centroid minimization) rather than factorizing the data matrix into a product of two or more matrices.

---

### Question 4 — Cross-Validation Assumptions

**Question:** Which of the following is an implicit assumption behind the validity of standard cross-validation estimate?

**Official Answer:** (c) The data are independently and identically distributed (IID)

**Option A — The training and test sets must be of equal size:** ❌ Wrong
Standard CV makes no requirement about equal-sized splits. Leave-one-out CV (LOOCV) has a test set of size 1 and a training set of size $n-1$. $k$-fold CV produces folds of approximately equal size but this is a computational choice, not a validity assumption. The estimate remains valid for unequal splits.

**Option B — The response variable must be normally distributed:** ❌ Wrong
CV is a completely nonparametric procedure — it only requires the ability to compute a loss function (e.g., MSE, misclassification rate). It imposes no distributional assumption on $y$. It works equally well for binary outcomes, counts, continuous responses, etc.

**Option C — The data are independently and identically distributed (IID):** ✓ Correct
CV randomly shuffles and partitions observations into folds, implicitly treating each observation as exchangeable. If observations are IID, then any partition into train/test is a valid simulation of the train-on-new-data-test-on-new-data scenario. If observations are NOT IID (e.g., time series with autocorrelation, repeated measures from the same individual, spatially correlated samples), this exchangeability breaks — the CV error estimate becomes optimistically biased because the test fold is not truly independent of training data.

**Option D — The model must be linear in parameters:** ❌ Wrong
CV is completely model-agnostic. It can be applied to linear models, decision trees, neural networks, SVMs, k-NN, or any other method. The procedure makes no assumption about the functional form of the model.

**Option E — The folds must have the same mean response value:** ❌ Wrong
While stratified CV (which attempts to maintain class balance across folds) is a useful practical technique for imbalanced classification problems, it is not a validity requirement for CV in general. Standard CV with random fold assignment is valid without this constraint.

> **Key takeaway:** Standard CV assumes observations are exchangeable (IID) — violating this (time series, repeated measures, clustered data) requires structured CV strategies such as time-based splits or leave-one-group-out.

---

### Question 5 — AIC/BIC vs Cross-Validation Assumptions

**Question:** Which of the following best distinguishes information criterion-based model selection (e.g., AIC/BIC) from cross-validation, with respect to their assumptions?

**Official Answer:** (a) AIC/BIC assumes a correctly specified likelihood model, while cross-validation makes fewer assumptions about the data-generating process

**Option A — AIC/BIC assumes a correctly specified likelihood model, while cross-validation makes fewer assumptions about the data-generating process:** ✓ Correct
AIC ($= -2\log\hat{L} + 2p$) and BIC ($= -2\log\hat{L} + p\log N$) both require a parametric likelihood $\hat{L}$ to be specified and maximized. If the likelihood model is misspecified, AIC/BIC lose their theoretical justification. Cross-validation only requires a loss function (e.g., MSE) and makes no parametric assumption about the data-generating distribution — it is model-agnostic.

**Option B — Cross-validation assumes the model is linear and errors are normally distributed, while AIC/BIC does not:** ❌ Wrong
This is exactly backwards. CV makes no assumption of linearity or normality — it is a nonparametric procedure. AIC/BIC, by contrast, require a parametric likelihood (often Gaussian errors are assumed when using AIC/BIC for regression).

**Option C — AIC/BIC provides unbiased estimates of out-of-sample error, while cross-validation estimates training error:** ❌ Wrong
This is backwards on both counts. AIC/BIC estimate a penalized in-sample fit that approximates out-of-sample performance (with a complexity penalty to correct for the in-sample optimism). CV directly estimates out-of-sample (generalization) error by evaluating on held-out data. CV does NOT estimate training error.

**Option D — Cross-validation requires large samples to be valid, while AIC/BIC can be used on any data size without issue:** ❌ Wrong
The opposite is closer to truth. CV can be applied even with small samples (LOOCV is often used for small $n$). AIC was derived under the assumption $p \ll n$, and BIC requires asymptotic justification as $N \to \infty$. Both criteria can behave poorly in small samples with many parameters.

**Option E — AIC/BIC rely on data splitting like cross-validation, but are computationally more expensive:** ❌ Wrong
AIC/BIC do NOT rely on data splitting at all. They are computed once on the full training set using the fitted likelihood value and a complexity penalty term. CV splits data into folds and refits the model multiple times. AIC/BIC are generally computationally cheaper than CV, not more expensive.

> **Key takeaway:** AIC/BIC require a correctly specified parametric likelihood model; cross-validation is model-agnostic and only requires a computable loss function, making it applicable under weaker assumptions.

---

### Question 6 — Nested Cross-Validation

**Question:** When is nested cross-validation preferred over standard (non-nested) cross-validation?

**Official Answer:** (c) When you want an unbiased estimate of generalization error after hyperparameter tuning

**Option A — When you have a very large dataset and want to reduce computation time:** ❌ Wrong
Nested CV is computationally MORE expensive than standard CV, not less. If the inner loop has $k_{\text{inner}}$ folds and the outer loop has $k_{\text{outer}}$ folds, you perform $k_{\text{outer}} \times k_{\text{inner}}$ model fits during the inner loop alone. Large datasets make nested CV even more expensive.

**Option B — When the model has no tunable parameters:** ❌ Wrong
If there are no hyperparameters to tune, standard CV suffices. Nested CV exists precisely to address the problem that standard CV is biased when the same data is used for both selecting hyperparameters and evaluating the model. If there is nothing to select, there is no bias to correct.

**Option C — When you want an unbiased estimate of generalization error after hyperparameter tuning:** ✓ Correct
The problem with standard CV for model evaluation after hyperparameter tuning: the hyperparameters are chosen to minimize CV error on the full dataset, so the reported CV error is optimistically biased (it has been "optimized over"). Nested CV separates the two tasks: the inner loop selects optimal hyperparameters, and the outer loop evaluates generalization error on data that was never used for selection — providing an unbiased estimate.

**Option D — When you're testing whether the response variable is normally distributed:** ❌ Wrong
Normality testing is a completely unrelated task involving statistical tests (Shapiro-Wilk, Kolmogorov-Smirnov, Q-Q plots), not cross-validation. CV is a model evaluation procedure, not a distributional test.

**Option E — When cross-validation folds are randomly sampled with replacement:** ❌ Wrong
Sampling with replacement describes bootstrap resampling, not nested CV. The nesting structure of nested CV refers to having two nested loops (inner for model selection, outer for performance evaluation), both using standard without-replacement folds.

> **Key takeaway:** Nested CV is used when hyperparameter tuning is performed — the outer loop provides an unbiased estimate of generalization error, while the inner loop handles hyperparameter selection, preventing information leakage between the two tasks.

---

### Question 7 — Ridge Regression and Feature Selection

**Question:** Which of the following best explains why Ridge regression tends not to perform feature selection?

**Official Answer:** (c) It shrinks coefficients but doesn't set them to zero

**Option A — It uses L1 regularization:** ❌ Wrong
Ridge uses L2 regularization, not L1. The Ridge objective is $\hat{\beta}_{\text{ridge}} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2$. It is LASSO that uses L1 regularization ($\lambda\|\beta\|_1$), and this L1 penalty is precisely what enables exact sparsity. This statement is factually incorrect.

**Option B — It removes collinear features before fitting the model:** ❌ Wrong
Ridge does not remove any features before fitting. In fact, Ridge is specifically useful for handling collinearity (the penalty on $\|\beta\|_2^2$ stabilizes the solution when $X^TX$ is near-singular), but it handles collinearity by shrinking all coefficients jointly, not by discarding any.

**Option C — It shrinks coefficients but doesn't set them to zero:** ✓ Correct
The closed-form Ridge solution $\hat{\beta}_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^Ty$ multiplies each OLS coefficient by a shrinkage factor $< 1$, but the result is never identically zero for finite $\lambda$. Geometrically, the L2 constraint region $\|\beta\|_2^2 \leq t$ is a smooth sphere with no corners — the constrained optimum almost never occurs exactly on a coordinate axis, so coefficients are shrunk but not zeroed. Contrast with Lasso's L1 ball (a diamond/rhombus) which has corners precisely on the coordinate axes.

**Option D — It standardizes features before modelling:** ❌ Wrong
Feature standardization is a preprocessing step that should be applied before running any regularized regression (Ridge or Lasso alike, since both are scale-dependent). It is not a property or explanation for Ridge's behavior — standardization does not cause or prevent feature selection.

**Option E — It excludes features with low variance by default:** ❌ Wrong
Ridge excludes no features by default. Low-variance features receive smaller Ridge coefficients because they explain less variance, but they are never set to exactly zero. This is not a deliberate exclusion but a consequence of the shrinkage.

> **Key takeaway:** Ridge's L2 penalty shrinks coefficients proportionally toward zero but can never reach exactly zero for finite $\lambda$ because the circular constraint region has no corners on the coordinate axes — this is the geometric reason Ridge does not perform feature selection.

---

### Question 8 — Model Selection Criterion as $N \to \infty$

**Question:** Which model selection criterion penalizes model complexity the most as $N \to \infty$? ($N$ is the sample size.)

**Official Answer:** (b) BIC

**Option A — AIC:** ❌ Wrong
The AIC penalty is $2p$ (twice the number of parameters), which is a constant independent of $N$. As $N \to \infty$, the AIC penalty does not grow, so AIC becomes relatively less conservative for large samples. AIC is known to be inconsistent (tends to overfit as $N \to \infty$ for nested model families).

**Option B — BIC:** ✓ Correct
The BIC formula is $\text{BIC} = -2\log\hat{L} + p\log N$. The complexity penalty $p\log N$ grows without bound as $N \to \infty$. For even moderate $N$, $\log N > 2$, meaning BIC penalizes each parameter more heavily than AIC does. This makes BIC increasingly conservative as sample size grows, and it is a consistent model selector (asymptotically selects the true model if it is in the candidate set).

**Option C — $C_p$:** ❌ Wrong
Mallows' $C_p = \frac{\text{RSS}}{s^2} + 2p - N$ has a penalty of $2p/s^2$ (effectively proportional to $2p$, similar to AIC), which does not grow with $N$. It is closely related to AIC for Gaussian models.

**Option D — Cross-validation:** ❌ Wrong
Cross-validation has no explicit complexity penalty term. It estimates generalization error empirically through data splitting. Its behavior as $N \to \infty$ depends on the fold structure and model class — there is no formula with an $N$-dependent penalty.

**Option E — Adjusted $R^2$:** ❌ Wrong
Adjusted $R^2 = 1 - \frac{\text{RSS}/(N-p-1)}{\text{TSS}/(N-1)}$ penalizes model complexity implicitly through the degrees-of-freedom correction, but this penalty does not grow as $\log N$ — it is a ratio-based correction that does not diverge comparably to BIC.

> **Key takeaway:** BIC's penalty $p\log N$ grows with sample size, making it increasingly conservative and causing it to favor simpler models more aggressively than AIC (penalty $2p$) as $N \to \infty$. This is why BIC is a consistent model selector while AIC is not.

---

### Question 9 — LDA Decision Boundary Linearity

**Question:** In LDA, the decision boundary is linear because:

**Official Answer:** (e) It assumes equal class covariances

**Option A — It uses polynomial basis functions:** ❌ Wrong
LDA does not use polynomial basis functions. It is a linear method in the original feature space. Using polynomial basis functions would create a nonlinear boundary (it would be equivalent to a form of kernel method or QDA with polynomial features). This option confuses LDA with basis-expansion methods.

**Option B — It assumes equal class priors:** ❌ Wrong
Equal class priors $P(Y=k) = 1/K$ simplify the decision rule but do NOT cause linearity. With unequal priors, LDA's boundary is still linear — the prior $\log P(Y=k)$ enters as an additive constant in the log-posterior, not as a quadratic term. Linearity is unaffected by the prior assumption.

**Option C — It minimizes squared loss:** ❌ Wrong
LDA is derived from Bayes' theorem applied to Gaussian class-conditional distributions, not from minimizing squared loss. Minimizing squared loss gives ordinary least squares regression. The connection between LDA and regression exists in the two-class case but squared loss minimization is not the mechanism that produces linearity.

**Option D — It fits a linear regression:** ❌ Wrong
LDA is a generative classification model — it models $P(X \mid Y=k) = \mathcal{N}(\mu_k, \Sigma)$ and applies Bayes' rule to get $P(Y=k \mid X)$. It does not fit a linear regression. The connection to regression exists (in the two-class case, LDA is equivalent to linear regression of a 0/1 indicator), but the reason for the linear boundary is not that it "fits linear regression."

**Option E — It assumes equal class covariances:** ✓ Correct
The log-posterior ratio for LDA (comparing class $k$ to class $l$) involves computing:
$$\log\frac{P(Y=k|x)}{P(Y=l|x)} = \log\frac{P(X=x|Y=k)}{P(X=x|Y=l)} + \log\frac{\pi_k}{\pi_l}$$
With $P(X|Y=k) = \mathcal{N}(\mu_k, \Sigma_k)$, the log-ratio of two Gaussians contains a quadratic term $x^T(\Sigma_k^{-1} - \Sigma_l^{-1})x$. When $\Sigma_k = \Sigma_l = \Sigma$ (the LDA assumption), this quadratic term cancels exactly, leaving only a linear function of $x$: $x^T\Sigma^{-1}(\mu_k - \mu_l) + \text{const}$. QDA (which allows class-specific $\Sigma_k$) produces a quadratic boundary precisely because this cancellation does not occur.

> **Key takeaway:** LDA's equal-covariance assumption $\Sigma_k = \Sigma$ for all classes causes the quadratic terms in the log-posterior ratio to cancel, leaving a linear discriminant function of $x$ — this is the mechanistic reason for the linear boundary.

---

### Question 10 — BH vs Bonferroni

**Question:** Which of the following statements best distinguishes the Benjamini–Hochberg (BH) procedure from the Bonferroni correction in multiple hypothesis testing?

**Official Answer:** (a) BH controls the expected proportion of false discoveries, while Bonferroni controls the probability of at least one false discovery

**Option A — BH controls the expected proportion of false discoveries, while Bonferroni controls the probability of at least one false discovery:** ✓ Correct
This is the defining theoretical distinction. BH controls the False Discovery Rate: $\text{FDR} = E\left[\frac{V}{R}\right]$, where $V$ is the number of false rejections and $R$ is the total number of rejections (with $V/R = 0$ when $R = 0$). Bonferroni controls the Family-Wise Error Rate: $\text{FWER} = P(V \geq 1)$, the probability of making at least one false rejection. FWER control is stricter, leading to fewer discoveries but stronger error guarantees. BH is more powerful (more discoveries) but allows a controlled proportion of those discoveries to be false.

**Option B — BH controls the family-wise error rate, while Bonferroni controls the false discovery rate:** ❌ Wrong
This exactly reverses the truth. Bonferroni controls FWER; BH controls FDR. Confusing these two is a common trap.

**Option C — BH adjusts p-values to be more conservative than Bonferroni:** ❌ Wrong
BH is LESS conservative than Bonferroni. Bonferroni uses the threshold $\alpha/M$ (where $M$ is the number of tests), which becomes very stringent for large $M$. BH uses a ranked p-value procedure with threshold $\alpha \cdot k/M$ for the $k$-th ranked p-value, rejecting more hypotheses. At the same nominal $\alpha$, BH always rejects at least as many hypotheses as Bonferroni.

**Option D — BH is only valid when all null hypotheses are true, while Bonferroni is valid regardless:** ❌ Wrong
The original Benjamini-Hochberg (1995) procedure is valid for independent tests or tests with positive dependence (PRDS condition). Bonferroni is valid regardless of the dependence structure. However, the claim that BH is "only valid when all null hypotheses are true" is incorrect — BH is valid even when some nulls are false (the partial null case), which is precisely the realistic scenario it is designed for.

**Option E — BH requires all p-values to be normally distributed:** ❌ Wrong
Neither BH nor Bonferroni requires p-values to be normally distributed. P-values are defined as probabilities (uniform under the null hypothesis for continuous test statistics), not as Gaussian random variables. Both procedures work with any p-values from any test.

> **Key takeaway:** BH controls FDR = $E[V/R]$ (expected proportion of false discoveries among all discoveries); Bonferroni controls FWER = $P(V \geq 1)$ (probability of any false discovery). BH is less conservative and has higher power, at the cost of allowing a small proportion of false discoveries.

---

### Question 11 — ICA vs PCA

**Question:** Which of the following best explains why ICA can recover sources that PCA cannot?

**Official Answer:** (c) ICA maximizes non-Gaussianity to find statistically independent components, while PCA finds uncorrelated directions

**Option A — ICA assumes orthogonal components, while PCA assumes independence:** ❌ Wrong
This exactly reverses the assumptions. PCA finds orthogonal components (uncorrelated, zero second-order cross-covariance). ICA seeks statistically independent components (zero cross-dependence at all orders, not just second-order). Orthogonality is a weaker condition than independence.

**Option B — ICA identifies components with maximal variance, while PCA minimizes kurtosis:** ❌ Wrong
Both claims are backwards. PCA identifies directions of maximal variance (the eigenvectors of the covariance matrix corresponding to the largest eigenvalues). ICA maximizes non-Gaussianity (often measured by kurtosis or negentropy), not PCA.

**Option C — ICA maximizes non-Gaussianity to find statistically independent components, while PCA finds uncorrelated directions:** ✓ Correct
PCA finds uncorrelated directions by diagonalizing the covariance matrix — for Gaussian data, uncorrelated equals independent, and PCA suffices. For non-Gaussian data, uncorrelated does NOT imply independent: $\text{Cov}(X_1, X_2) = 0 \not\Rightarrow X_1 \perp\!\!\!\perp X_2$ in general. ICA exploits the Central Limit Theorem in reverse: linear mixtures of independent non-Gaussian sources are more Gaussian than the original sources. By maximizing non-Gaussianity (kurtosis, negentropy), ICA finds the original independent sources. This is why ICA can separate audio signals (cocktail party problem) or find independent brain activation maps in fMRI data — tasks that PCA cannot solve.

**Option D — ICA removes noise using eigenvalue shrinkage, which PCA cannot do:** ❌ Wrong
Eigenvalue shrinkage is actually a technique used in the context of PCA/covariance estimation (e.g., Ledoit-Wolf shrinkage), not ICA. ICA has no eigenvalue shrinkage step. ICA typically includes a PCA whitening step as preprocessing, but the ICA-specific step is the rotation to maximize independence, not eigenvalue shrinkage.

**Option E — ICA rotates the principal components to align with the class labels:** ❌ Wrong
ICA does perform a rotation in the whitened PCA space, but the rotation is guided by maximizing statistical independence (non-Gaussianity), NOT by class labels. ICA is an unsupervised method — it has no access to class labels. The supervised analog (using class labels to guide rotation) would be LDA or related methods.

> **Key takeaway:** PCA finds uncorrelated directions (zero covariance = second-order independence); ICA finds statistically independent directions by maximizing non-Gaussianity via higher-order statistics (kurtosis, negentropy). For non-Gaussian sources, uncorrelated $\neq$ independent, so ICA succeeds where PCA fails.

---

### Question 12 — Neural Network Parameter Count

**Question:** Consider a fully connected feedforward neural network: Input layer: 3 nodes; Hidden layer 1: 4 nodes (ReLU); Hidden layer 2: 2 nodes (ReLU); Output layer: 1 node (Linear). Each layer includes a bias term. How many total scalar parameters does the network have?

**Official Answer:** (d) 29

**Calculation:**

| Connection | Weights | Biases | Subtotal |
|---|---|---|---|
| Input (3) $\to$ Hidden 1 (4) | $3 \times 4 = 12$ | $4$ | $16$ |
| Hidden 1 (4) $\to$ Hidden 2 (2) | $4 \times 2 = 8$ | $2$ | $10$ |
| Hidden 2 (2) $\to$ Output (1) | $2 \times 1 = 2$ | $1$ | $3$ |
| **Total** | | | $\mathbf{29}$ |

The formula for each layer is: $(\text{inputs to layer} \times \text{units in layer}) + \text{units in layer (biases)}$.

**Option A — 21:** ❌ Wrong
This is the weight count alone ($12 + 8 + 2 - 1 = 21$) or a miscount that omits some biases. Correct counting requires adding all biases ($4 + 2 + 1 = 7$) to all weights ($12 + 8 + 2 = 22$), giving $29$.

**Option B — 25:** ❌ Wrong
This could arise from missing the bias of the output layer or the first hidden layer, or from a $3 \times 3$ miscounting of the first layer ($9 + 4 + 8 + 2 + 2 = 25$). Neither gives a consistent derivation.

**Option C — 27:** ❌ Wrong
A possible source: $12 + 8 + 2 = 22$ weights $+ 4 + 1 = 5$ biases $= 27$, omitting the 2 biases for hidden layer 2. Every layer, including hidden layer 2, contributes biases.

**Option D — 29:** ✓ Correct
$16 + 10 + 3 = 29$ as computed above.

**Option E — 31:** ❌ Wrong
31 was the answer for a different architecture in the 2024 exam (10 inputs $\to$ 2 $\to$ 2 $\to$ 1: $20 + 2 + 4 + 2 + 2 + 1 = 31$). For this 2025 architecture (3 $\to$ 4 $\to$ 2 $\to$ 1), the correct count is 29. A common error is using the wrong architecture's numbers.

> **Key takeaway:** Parameter count per layer = (inputs $\times$ units) + units. For this network: $(3 \times 4 + 4) + (4 \times 2 + 2) + (2 \times 1 + 1) = 16 + 10 + 3 = \mathbf{29}$.

---

### Question 13 — K-means vs GMM

**Question:** Which of the following statements most correctly highlights a key difference between K-means clustering and Gaussian Mixture Models (GMMs)?

**Official Answer:** (c) GMM models the data distribution probabilistically, allowing for elliptical clusters, while K-means minimizes squared Euclidean distance to centroids

**Option A — K-means allows clusters to have different covariance structures, while GMM assumes identical spherical clusters:** ❌ Wrong
This is completely backwards. K-means implicitly assumes all clusters are spherical and equal-variance (because it uses Euclidean distance to a single centroid per cluster, making it equivalent to fitting a GMM with equal spherical covariances $\Sigma_k = \sigma^2 I$). GMM allows each component to have its own mean $\mu_k$ and covariance $\Sigma_k$, enabling elliptical, rotated, and differently-sized clusters.

**Option B — K-means assigns soft cluster memberships, while GMM assigns hard labels only:** ❌ Wrong
This reverses the truth. K-means assigns hard labels: each point belongs to exactly one cluster (the nearest centroid). GMM assigns soft memberships: each point has a posterior probability $r_{ik} = P(Z_i = k \mid x_i)$ for each cluster, computed via Bayes' rule. This is the E-step of the EM algorithm.

**Option C — GMM models the data distribution probabilistically, allowing for elliptical clusters, while K-means minimizes squared Euclidean distance to centroids:** ✓ Correct
GMM defines $p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$ and fits it by maximizing the log-likelihood via EM. This probabilistic framework allows each component to have a full covariance matrix, enabling elliptical shapes of any orientation. K-means minimizes $\sum_{k=1}^K \sum_{i \in C_k} \|x_i - \mu_k\|^2$, which implicitly assumes spherical clusters and assigns membership deterministically.

**Option D — GMM is more robust to outliers than K-means because it uses medians instead of means:** ❌ Wrong
GMM does NOT use medians — it uses means (the $\mu_k$ parameters in the Gaussian components). In fact, both K-means and standard GMM are sensitive to outliers because they are both based on Euclidean distance / squared error. Neither is inherently robust to outliers; both can have their cluster centers pulled toward outliers.

**Option E — K-means uses likelihood maximization for parameter estimation, while GMM does not:** ❌ Wrong
This is backwards. GMM uses likelihood maximization (via the EM algorithm) as its fitting criterion: it maximizes $\sum_i \log \sum_k \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)$. K-means uses a geometric objective (minimize within-cluster sum of squares) and has no probabilistic likelihood interpretation in the standard formulation.

> **Key takeaway:** The two core differences: (1) K-means = hard assignment + spherical clusters via Euclidean distance; GMM = soft probabilistic membership + flexible elliptical clusters via full covariance matrices. (2) K-means minimizes a geometric objective; GMM maximizes a likelihood via EM.

---

### Question 14 — PCA Variance Explained

**Question:** Given a centered data matrix $X \in \mathbb{R}^{100 \times 2}$, and its empirical covariance matrix has eigenvalues $\lambda_1 = 6$, $\lambda_2 = 2$. What fraction of total variance is explained by the first principal component?

**Official Answer:** (d) 75%

**Calculation:**
$$\text{Proportion of variance} = \frac{\lambda_1}{\sum_j \lambda_j} = \frac{6}{6 + 2} = \frac{6}{8} = 0.75 = 75\%$$

**Option A — 25%:** ❌ Wrong
$25\% = \lambda_2 / (\lambda_1 + \lambda_2) = 2/8$ — this is the fraction explained by the SECOND principal component, not the first.

**Option B — 50%:** ❌ Wrong
$50\%$ would result from equal eigenvalues (e.g., $\lambda_1 = \lambda_2 = 4$). With $\lambda_1 = 6 > \lambda_2 = 2$, the first PC explains more than half the variance.

**Option C — 60%:** ❌ Wrong
$60\% = 6/10$ — this would arise from incorrectly using the sum as 10 rather than 8, perhaps from confusing the number of observations (100) or dimensions (2) with the eigenvalues. The total variance is $\lambda_1 + \lambda_2 = 8$.

**Option D — 75%:** ✓ Correct
$6/(6+2) = 6/8 = 0.75 = 75\%$.

**Option E — 80%:** ❌ Wrong
$80\% = 4/5$ — no straightforward misapplication of $\lambda_1 = 6$, $\lambda_2 = 2$ produces this result. Possibly from confusing the 2D data with a 5-dimensional case.

> **Key takeaway:** Fraction of variance explained by the $j$-th PC $= \lambda_j / \sum_i \lambda_i$. Here: $6/(6+2) = 75\%$.

---

### Question 15 — K-medoids vs K-means

**Question:** Which of the following correctly characterizes a fundamental difference between K-medoids and K-means clustering?

**Official Answer:** (d) K-medoids is more robust to outliers than K-means

**Option A — K-medoids minimizes the sum of squared distances, while K-means minimizes absolute distances:** ❌ Wrong
This reverses the objectives. K-means minimizes the sum of squared Euclidean distances: $\sum_{k} \sum_{i \in C_k} \|x_i - \mu_k\|^2$. K-medoids minimizes the sum of dissimilarities (often absolute/Manhattan distances or any dissimilarity measure): $\sum_{k} \sum_{i \in C_k} d(x_i, m_k)$ where $m_k$ is an actual data point (the medoid).

**Option B — K-medoids selects centroids that may lie outside the data cloud, while K-means does not:** ❌ Wrong
This reverses the truth. K-means computes centroids as the arithmetic mean of cluster members — these means may lie outside the data cloud (e.g., for non-convex clusters). K-medoids selects the cluster representative from among the actual data points, so it is always within the data cloud by construction.

**Option C — K-medoids is less robust to outliers than K-means:** ❌ Wrong
This is the opposite of D (the correct answer). K-medoids is MORE robust because its cluster center is constrained to be an actual data point, making it harder for a single outlier to pull the center far from the true cluster.

**Option D — K-medoids is more robust to outliers than K-means:** ✓ Correct
K-means centroids are computed as the arithmetic mean: $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$. A single outlier with extreme values can pull $\mu_k$ far from the rest of the cluster, similar to how the sample mean is sensitive to outliers. K-medoids selects the medoid — the data point that minimizes total dissimilarity to other cluster members — which is inherently resistant to extreme observations because the medoid must be an actual (non-outlier-dominated) data point.

**Option E — K-medoids assumes normally distributed features, unlike K-means:** ❌ Wrong
Neither K-medoids nor K-means assumes normally distributed features. Both are nonparametric clustering algorithms based on distance/dissimilarity measures, with no distributional assumptions. GMM (not K-means/medoids) is the clustering method that assumes Gaussian distributions.

> **Key takeaway:** K-medoids centers are actual data points (not computed means), making them inherently resistant to outliers — one extreme observation cannot distort the medoid as it can pull a K-means centroid.

---

### Question 16 — NMF Defining Characteristic

**Question:** Which of the following is a defining characteristic of Non-negative Matrix Factorization (NMF) when applied to a data matrix $X \in \mathbb{R}^{n \times p}$?

**Official Answer:** (b) NMF seeks matrices $W$ and $H$ such that $X \approx WH$, with all entries of $W$ and $H$ constrained to be non-negative

**Option A — NMF guarantees a unique factorization for all datasets:** ❌ Wrong
NMF solutions are generally NOT unique. Unlike PCA (which has a unique solution for given dimensions via SVD), NMF is a non-convex optimization problem with multiple local minima. Different initializations may yield different factorizations with the same or similar reconstruction error. Uniqueness holds only under special conditions (e.g., sufficiently sparse factors).

**Option B — NMF seeks matrices $W$ and $H$ such that $X \approx WH$, with all entries of $W$ and $H$ constrained to be non-negative:** ✓ Correct
This is the complete and correct definition. NMF solves: $\min_{W \geq 0,\ H \geq 0} \|X - WH\|_F^2$ where $W \in \mathbb{R}^{n \times r}_{\geq 0}$ (basis/dictionary matrix) and $H \in \mathbb{R}^{r \times p}_{\geq 0}$ (encoding/coefficient matrix). The non-negativity constraint on BOTH $W$ and $H$ is the key distinguishing feature — it enables part-based representations (e.g., decomposing face images into facial features that add together, never cancel).

**Option C — NMF seeks matrices $W$ and $H$ such that $X \approx WH$, with orthogonality constraints on $W$:** ❌ Wrong
Orthogonality constraints on $W$ are the PCA/SVD setting, not NMF. NMF imposes non-negativity, not orthogonality. Adding orthogonality to NMF is possible (Orthogonal NMF) but is a specialized variant, not the standard definition.

**Option D — NMF seeks matrices $W$ and $H$ such that $X \approx WH$, with the only assumption that all entries of $X$ be non-negative:** ❌ Wrong
This is subtly wrong in a critical way: the non-negativity constraint is on $W$ AND $H$, not just on $X$. While in practice NMF is applied to non-negative $X$ (the non-negativity of $WH$ implies non-negative $X$), the defining constraint is on the factor matrices, not merely on the input. Option D incorrectly locates the constraint solely on $X$.

**Option E — NMF requires the input data matrix $X$ to be square and symmetric:** ❌ Wrong
There is no such requirement. NMF is applied to arbitrary rectangular matrices $X \in \mathbb{R}^{n \times p}$ with $n \neq p$ being perfectly standard (e.g., $n$ = documents, $p$ = words in text mining). Square symmetric matrices are the domain of eigendecompositions (like covariance matrices in PCA), not NMF.

> **Key takeaway:** NMF's defining constraint is non-negativity of BOTH factor matrices $W \geq 0$ and $H \geq 0$, enabling parts-based, additive representations — this is distinct from requiring non-negative $X$ (the input) or requiring orthogonality (PCA).

---

### Question 17 — Archetypal Analysis

**Question:** Which of the following best describes a defining characteristic of Archetypal Analysis (AA) when applied to a dataset $X \in \mathbb{R}^{n \times p}$?

**Official Answer:** (c) Archetypes are constructed as weighted combinations of extreme observations, and each data point is approximated by a weighted mixture of these archetypes

**Option A — Archetypes are orthogonal directions that capture the maximum variance in the data, and each data point is approximated by a weighted mixture of these archetypes:** ❌ Wrong
This describes PCA, not AA. PCA principal components are orthogonal directions of maximum variance, and data points are projected onto them. AA archetypes do NOT maximize variance and are NOT orthogonal — they lie on the convex hull of the data (extreme points), not along directions of maximum spread.

**Option B — Archetypes are cluster centroids that minimize squared distances to data points:** ❌ Wrong
This describes K-means, not AA. K-means centroids are the means of assigned clusters and are positioned to minimize within-cluster sum of squares. AA archetypes are positioned at (or near) the boundary of the data — they are the "extreme" prototypes, not central tendencies.

**Option C — Archetypes are constructed as weighted combinations of extreme observations, and each data point is approximated by a weighted mixture of these archetypes:** ✓ Correct
Archetypal Analysis (Cutler & Breiman 1994) solves for archetypes $Z^* = XB$ where $B \in \mathbb{R}^{n \times r}$ has non-negative rows summing to 1 (each archetype is a convex combination of data points), constrained to place archetypes on the convex hull of $X$. Each data point is reconstructed as $\hat{x}_i = Z^* s_i = XBs_i$ where $s_i \geq 0$, $\sum_k s_{ik} = 1$. The key insight: archetypes represent the most "extreme" or "pure" types in the data (e.g., the most athletic, the most sedentary), and all other observations are blends of these extremes.

**Option D — Archetypal Analysis assumes features follow a multivariate Gaussian distribution:** ❌ Wrong
AA makes no distributional assumption about the features. It is a geometric method: it finds extreme points in feature space and represents all data as convex combinations of those extremes. There is no Gaussian likelihood or covariance modeling involved.

**Option E — Archetypal Analysis decomposes the data using sparse binary encodings:** ❌ Wrong
AA uses convex (non-negative, sum-to-one) continuous weights, not binary encodings. Sparse binary representations are associated with methods like sparse coding or dictionary learning. AA's weights are continuous probability-like coefficients, not binary indicators.

> **Key takeaway:** AA archetypes live on or near the convex hull (extremes) of the data and are themselves convex combinations of data points; every observation is then approximated as a convex combination of these archetypes — this "extreme prototype" interpretation distinguishes AA from PCA (variance directions), K-means (cluster centers), and NMF.

---

### Question 18 — Standardization Within CV Folds

**Question:** Why is it important to apply feature standardization within each fold of a cross-validation procedure, rather than using statistics from the entire dataset?

**Official Answer:** (b) Using full-data statistics can cause information leakage from the test fold into the training process

**Option A — Standardizing within each fold reduces the variance of cross-validation error estimates:** ❌ Wrong
Standardization within folds does not directly reduce the variance of the CV error estimate. The variance of CV error depends on the number of folds, data variability, and model stability — not on whether standardization is done inside or outside the fold loop. In fact, within-fold standardization slightly increases variance (each fold uses slightly different mean/std) compared to using one fixed standardization.

**Option B — Using full-data statistics can cause information leakage from the test fold into the training process:** ✓ Correct
When you compute the mean and standard deviation across the entire dataset (including the test fold) and then use those statistics to standardize all data before CV, the test fold's values influence the transformation applied to the training fold. This constitutes data leakage: the training process has implicitly "seen" the test fold's distribution. To simulate truly unseen test data, standardization parameters (mean, std) must be estimated only from the training fold and then applied (without re-estimation) to the test fold. This mirrors the deployment scenario where new data must be standardized using statistics from training data only.

**Option C — Standardization is unnecessary if the model is linear:** ❌ Wrong
Standardization matters most precisely for regularized linear models (Ridge, Lasso, Elastic Net), because the penalty $\lambda\|\beta\|^2$ treats all coefficients symmetrically only if features are on the same scale. Without standardization, features with large magnitudes dominate the penalty. Standardization is arguably more important for linear regularized models than for trees or distance-based methods.

**Option D — Performing standardization within each cross-validation fold speeds up model training:** ❌ Wrong
Standardization within each fold adds computational overhead (computing mean/std per fold, $k$ times). For most models, this is negligible, but it certainly does not speed up training. The reason for within-fold standardization is methodological correctness, not computational efficiency.

**Option E — Standardizing within folds ensures features are normally distributed in each fold:** ❌ Wrong
Standardization (subtracting mean, dividing by standard deviation) produces features with zero mean and unit variance — it does NOT transform arbitrary distributions to Gaussian. A bimodal feature remains bimodal after standardization; it just shifts and scales. Normalization and normality are fundamentally different concepts.

> **Key takeaway:** Computing standardization statistics from the full dataset (including the test fold) lets the test fold's distribution influence training preprocessing — this is data leakage. Correct procedure: fit the scaler on the training fold only, then apply it to the test fold.

---

### Question 19 — One-Standard-Error Rule

**Question:** You are shown a plot of Mean Squared Error (MSE) against model complexity. Using the one-standard error rule, which model should be selected?

**Official Answer:** (c) Model with complexity = 4

**Option A — Model with complexity = 5:** ❌ Wrong
Complexity = 5 is the model with the minimum CV-MSE (the "best" model by the naive criterion). The one-SE rule explicitly rejects this choice in favor of a simpler model — it is the reference point from which we move leftward, not the selection.

**Option B — Model with complexity = 6:** ❌ Wrong
Complexity = 6 would be more complex than the minimum-error model. The one-SE rule always selects a simpler model than the minimum, never a more complex one. Moving to complexity = 6 would increase variance without benefit.

**Option C — Model with complexity = 4:** ✓ Correct
The one-SE rule procedure: (1) Find the model with minimum CV error — here complexity = 5. (2) Compute the standard error of that minimum CV error estimate. (3) Draw a horizontal line at $\text{MSE}_{\min} + 1 \cdot \text{SE}$. (4) Select the simplest model whose CV error falls at or below this line. Since complexity = 4 lies within one standard error of the minimum at complexity = 5, and it is simpler, the one-SE rule selects it. The rationale is parsimony under uncertainty: if a simpler model is statistically indistinguishable from the best model (within one SE), prefer the simpler one.

**Option D — Model with complexity = 1:** ❌ Wrong
Complexity = 1 (the simplest possible model) would only be selected if all models up to complexity = 1 have CV error within one SE of the minimum — this would require the minimum-error model to have very high variance in its estimate and that the simplest model performs nearly as well, which is not the scenario implied by the standard textbook version of this question.

**Option E — Model with complexity = 10:** ❌ Wrong
Complexity = 10 is the most complex model available. The one-SE rule never selects a more complex model than the CV minimum — the direction of the 1-SE selection is always toward greater simplicity (lower complexity), never toward greater complexity.

> **Key takeaway:** The one-SE rule: find the minimum-CV-error model (complexity = 5), add one standard error to its CV error, then select the SIMPLEST model whose error falls within that band (complexity = 4). This encodes a preference for parsimony under statistical uncertainty.

---

### Question 20 — Optimal Clusters from Silhouette Plot

**Question:** What is the optimal number of clusters, from the silhouette plot?

**Official Answer:** (b) 4

**Option A — 3:** ❌ Wrong
From the silhouette plot, $k = 3$ does not achieve the highest average silhouette width. While it may be a reasonable solution visually, it is not the peak of the silhouette criterion in this plot.

**Option B — 4:** ✓ Correct
The silhouette coefficient for observation $i$ in cluster $C_k$ is:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-cluster distance. Values near +1 indicate well-separated, compact clusters. The average silhouette width across all observations peaks at $k = 4$ in this plot, indicating that 4 clusters provide the best balance of cluster cohesion and separation.

**Option C — 5:** ❌ Wrong
$k = 5$ does not achieve the maximum average silhouette width in this plot. Increasing $k$ beyond the true number of clusters fragments true clusters, reducing within-cluster cohesion and thus the silhouette score.

**Option D — 6:** ❌ Wrong
Similarly, $k = 6$ over-partitions the data and does not achieve the maximum silhouette.

**Option E — 7:** ❌ Wrong
$k = 7$ further over-fragments the data. The silhouette method typically shows a clear peak at the true $k$ and declining values for larger $k$.

> **Key takeaway:** The optimal $k$ in a silhouette analysis is the value that maximizes the average silhouette width — here, $k = 4$ gives the best separation/cohesion tradeoff. Always read the plot to find the peak, not just the largest $k$ considered.

---

## Open Questions

---

### Question 21 — LDA vs GMM

**Question:** Discuss LDA and GMM in terms of the following: a) Explain and contrast the probabilistic assumptions between them. b) Describe and contrast how model fitting is performed. c) Highlight key differences in their goals, supervision, and use of labels. d) Discuss how each model handles class overlap and latent structure.

---

#### Part a) Probabilistic Assumptions

**Shared foundation:** Both LDA and GMM model data using Gaussian distributions. Both assume that observations within a class (or component) follow a multivariate Gaussian distribution:

$$P(X \mid Y = k) = \mathcal{N}(x; \mu_k, \Sigma_k)$$

**Key distinction — covariance structure:**

**LDA** imposes the additional constraint that all classes share the same covariance matrix:
$$\Sigma_1 = \Sigma_2 = \cdots = \Sigma_K = \Sigma \quad \text{(pooled within-class covariance)}$$

This is a strong parametric assumption. It reduces the number of parameters to estimate (one shared $\Sigma$ instead of $K$ class-specific $\Sigma_k$) but may be violated in practice when different classes genuinely have different spread or orientation.

**GMM** allows each component to have its own mean and covariance:
$$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x; \mu_k, \Sigma_k)$$

Each component has a free $\mu_k \in \mathbb{R}^p$ and $\Sigma_k \in \mathbb{R}^{p \times p}$ (positive definite). This is more flexible — GMM can represent clusters of different shapes, sizes, and orientations. The mixing proportions $\pi_k = P(Z = k)$ are free parameters satisfying $\sum_k \pi_k = 1$.

**Summary:** LDA assumes shared $\Sigma$; GMM assumes per-component $\Sigma_k$. Both are multivariate Gaussian; GMM is strictly more flexible.

---

#### Part b) Model Fitting

**LDA fitting (closed-form MLE):**

LDA uses labeled data and computes closed-form maximum likelihood estimates:
1. Class priors: $\hat{\pi}_k = n_k / n$ (fraction of training points in class $k$)
2. Class means: $\hat{\mu}_k = \frac{1}{n_k}\sum_{i: y_i = k} x_i$
3. Pooled within-class covariance:
$$\hat{\Sigma} = \frac{1}{n - K} \sum_{k=1}^{K} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$$

This is a single-pass estimation — no iterations are required. The discriminant function is then applied to classify new points via Bayes' rule.

**GMM fitting (EM algorithm — iterative):**

GMM uses the Expectation-Maximization (EM) algorithm because the component assignments $Z_i$ are latent (unobserved):

- **E-step (Expectation):** Compute soft responsibilities (posterior probabilities of cluster membership) using current parameter estimates:
$$r_{ik} = P(Z_i = k \mid x_i) = \frac{\pi_k \, \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{l=1}^K \pi_l \, \mathcal{N}(x_i; \mu_l, \Sigma_l)}$$

- **M-step (Maximization):** Update parameters using weighted sufficient statistics:
$$\hat{\pi}_k = \frac{\sum_i r_{ik}}{n}, \quad \hat{\mu}_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \quad \hat{\Sigma}_k = \frac{\sum_i r_{ik}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T}{\sum_i r_{ik}}$$

EM iterates between E and M steps until convergence (log-likelihood increase falls below a threshold). It converges to a local maximum — initialization matters.

**Key contrast:** LDA = closed-form, one-shot, supervised. GMM = iterative EM, convergence to local optimum, typically unsupervised.

---

#### Part c) Goals, Supervision, and Use of Labels

| Dimension | LDA | GMM |
|---|---|---|
| **Primary goal** | Supervised classification | Density estimation / clustering |
| **Supervision** | Supervised (requires class labels $y_i$) | Unsupervised (no labels needed) |
| **Use of labels** | Labels determine class means and pooled covariance | No labels used; component assignment is latent |
| **Output** | Class prediction + posterior probability $P(Y=k \mid x)$ | Posterior probability of component membership $P(Z=k \mid x)$; soft cluster assignments |
| **Use case** | Classify new observations into known classes | Discover structure / clusters in unlabeled data; generative model for density estimation |

**Important nuance:** GMM can be used as a supervised generative classifier by fitting a separate GMM per class and applying Bayes' rule — in this mode, it resembles LDA but with class-specific $\Sigma_k$. This is equivalent to QDA (Quadratic Discriminant Analysis). However, the typical and default use of GMM is unsupervised.

**Dimensionality reduction:** LDA also produces a linear subspace of dimension $K-1$ that maximally separates the classes (Fisher's Linear Discriminant) — a byproduct that enables supervised dimensionality reduction. GMM produces no explicit dimensionality reduction.

---

#### Part d) Class Overlap and Latent Structure

**Class overlap:**

Both models can represent overlapping classes through their probabilistic posteriors $P(Y=k \mid x)$. In the overlap region, neither model gives a confident prediction.

- **LDA:** The linear decision boundary is a hyperplane $x^T w + w_0 = 0$. In overlap regions, both sides have positive posterior probability. LDA handles overlap gracefully probabilistically, but the linear boundary may not match the true overlap structure if the true boundary is nonlinear or if class covariances differ.

- **GMM:** With class-specific $\Sigma_k$, GMM can capture more complex overlap geometries. If class 1 is a long thin ellipse and class 2 is a sphere, LDA (forced to use a shared $\Sigma$) will draw a suboptimal boundary, while GMM (or QDA) captures the true geometry. GMM's soft assignments naturally express uncertainty in overlap regions through fractional $r_{ik}$ values.

**Latent structure:**

- **LDA:** Has no latent variables. The class labels $Y$ are observed, and the model directly estimates $P(X \mid Y=k)$. There is no hidden or unobserved structure beyond the known classes.

- **GMM:** Has explicit latent variables $Z_i \in \{1, \ldots, K\}$ — the component assignments. These are never observed; they are inferred from data via the E-step. This latent structure enables GMM to discover hidden groupings in unlabeled data. GMM can be seen as a soft version of K-means where cluster uncertainty is explicitly modeled. The latent $Z_i$ can also reveal subpopulations within data that have no known label structure.

**Practical implications:**
- Use LDA when you have labeled data and want a fast, interpretable, linear classifier — especially when sample sizes per class are small (LDA estimates fewer parameters due to shared $\Sigma$).
- Use GMM when data is unlabeled and you want to discover structure, or when a probabilistic generative model is needed, or when class boundaries are nonlinear/elliptical (use GMM as generative classifier = QDA).

---

### Question 22 — Cross-Validation Design for Wearable Biosignals

**Question:** You are given a dataset of time series biosignals (BVP, skin temperature, HR) from a wearable device. Data was collected from 16 individuals under three conditions (rest, running, social media) at four time points throughout the year (seasonal variation). This gives 192 observations (16 individuals × 3 activities × 4 seasons). Design training/validation/test sets to estimate expected prediction error for: a) A personalized model predicting stress for a specific individual. b) A generalized model predicting stress for a new (unseen) individual. Also discuss trade-offs and clinical deployment considerations.

---

#### Dataset Structure

The 192 observations have a hierarchical, non-IID structure:
- **Level 1 (individuals):** 16 subjects — data from the same person is correlated (inter-individual variation)
- **Level 2 (seasons within individual):** 4 time points — repeated measures on the same person, potentially autocorrelated
- **Level 3 (conditions within session):** 3 conditions — within a session, conditions may be ordered or have carry-over effects

Standard random CV would be invalid here because it would mix observations from the same individual across folds, leading to optimistically biased error estimates (the model effectively "sees" the test individual during training).

---

#### Part a) Personalized Model (within-individual)

**Goal:** Predict stress for a specific known individual using only that person's data. The model will be deployed on the same individual — future observations from the same person.

**Data available for this individual:** $1 \times 3 \times 4 = 12$ observations (3 conditions × 4 seasons).

**Recommended CV design: Leave-One-Season-Out (LOSO) Cross-Validation**

- **Why:** Seasons represent temporal structure. The most realistic deployment scenario for a personalized model (e.g., "my smartwatch should predict my stress next month") is predicting future time points from past data. Leave-one-season-out respects this temporal ordering.
- **Procedure:** Use 3 seasons for training, hold out 1 season for testing. Repeat for all 4 seasons (4 folds). Report mean test error across folds.
- **Training set:** 3 seasons × 3 conditions = 9 observations
- **Test set:** 1 season × 3 conditions = 3 observations

**Alternative:** Leave-one-condition-out if the goal is to generalize across activity types within that individual, but leave-one-season-out is more appropriate for temporal generalization.

**Key requirements:**
- Standardization must be performed within each fold (using only the 9 training observations to compute mean/std)
- All preprocessing must be done after the fold split
- If hyperparameter tuning is needed (e.g., regularization in a Ridge classifier), use nested CV: inner leave-one-season-out for model selection, outer leave-one-season-out for assessment

---

#### Part b) Generalized Model (new individual)

**Goal:** Predict stress for an entirely new individual who was not in the training data. Deployment scenario: a clinical patient whose biosignals have never been seen.

**Recommended CV design: Leave-One-Individual-Out (LOIO) Cross-Validation**

- **Why:** The generalization target is a new, unseen individual. To estimate how well the model performs on such individuals, we must simulate this: train on data from $N-1$ individuals, test on the held-out individual's data. All observations from a given individual must be in the same fold to prevent leakage.
- **Procedure:** For each of the 16 individuals: train on data from the other 15 (15 × 3 × 4 = 180 observations), test on the held-out individual's data (1 × 3 × 4 = 12 observations). Repeat for all 16 individuals (16 folds). Report mean test error across all 16 held-out individuals.
- **Training set:** 15 individuals × 3 conditions × 4 seasons = 180 observations
- **Test set:** 1 individual × 3 conditions × 4 seasons = 12 observations

**Key requirements:**
- Standardization (mean/std of features) computed from 180 training observations only, applied to test individual
- If hyperparameter tuning is needed, add an inner LOIO loop within the training set of 15 individuals (nested LOIO-CV)
- This design ensures the test estimate reflects true generalization to new individuals with no prior data

---

#### Trade-offs Between Personalized and Generalized Models

| Dimension | Personalized Model | Generalized Model |
|---|---|---|
| **Training data** | Small (12 obs from one person) | Large (up to 180 obs from 15 people) |
| **Training set diversity** | Low — only one individual | High — captures inter-individual variation |
| **Prediction accuracy** | Potentially very high for that individual (tailored) | Moderate — must generalize across people |
| **Deployment scenario** | Existing user with historical data | New patient/user with no prior data |
| **Scalability** | Must retrain per individual | One model serves all new users |
| **Data collection burden** | Requires extensive calibration per user | Only population-level data needed upfront |
| **Overfitting risk** | High (very few training samples, risk of overfitting to individual quirks) | Lower (more data, regularization helps) |
| **Inter-individual variation** | Ignored (by design) | Must be handled (key challenge) |

**Personalized model strengths:** Captures individual-specific physiological baselines (e.g., one person's resting HR is 55, another's is 80). Biosignals are highly individual — a personalized model avoids the problem of between-subject variability overwhelming within-subject signal.

**Personalized model weaknesses:** Only 12 training observations is very little. Requires a calibration period before the model is useful. Cannot be deployed immediately for a new user.

**Generalized model strengths:** Immediately deployable for new users with no prior data. More data for training. Clinically scalable.

**Generalized model weaknesses:** Must bridge inter-individual physiological differences, which are substantial. Average performance across individuals may be moderate.

---

#### Clinical Deployment Recommendation

**For clinical deployment to aid mental health experts with new patients: the generalized model is more appropriate.**

**Reasoning:**

1. **New patients are unseen individuals.** A clinical setting involves new patients for whom no baseline wearable data exists. A personalized model cannot be applied until substantial calibration data is collected — which may take weeks or months, defeating the purpose of real-time stress monitoring.

2. **Clinical scalability.** A hospital cannot afford to train a new model for each patient. One generalized model deployed across the clinic is operationally feasible.

3. **Ethical and practical considerations.** A model that requires extensive individual calibration before deployment creates an access barrier. The generalized model can be used from day one.

4. **Risk of overfitting in personalized settings.** With only 12 data points per individual, a personalized model risks overfitting to noise or condition-specific artifacts from that individual's calibration period.

**However**, a hybrid approach can be considered: start with a generalized model and, as data accumulates for a specific patient, fine-tune (adapt) the model using that patient's data — combining the scalability of a generalized model with the accuracy of a personalized one. This is the basis of federated learning and personalization strategies in modern clinical AI.

**Bottom line:** Deploy the generalized (leave-one-individual-out validated) model clinically, with the option to personalize over time as individual data accumulates.

---

*End of Solutions — CDA 02582 (2025)*
