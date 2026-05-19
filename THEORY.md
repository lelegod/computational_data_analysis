# CDA 02582 — Comprehensive Theory Guide
> Week-by-week. Intuition + formulas + key traps. Use Ctrl+F to find any topic.

---

## QUICK NAVIGATION
- [Week 1 — OLS, Ridge, Bias-Variance, Cp, AIC, BIC](#week-1--ols-ridge-bias-variance-cp-aic-bic)
- [Week 2 — KNN, Model Selection vs Assessment, K-fold CV, 1-SE Rule](#week-2--knn-model-selection-vs-assessment-k-fold-cv-1-se-rule)
- [Week 3 — Lasso, Elastic Net, LARS, Curse of Dimensionality, Multiple Testing](#week-3--lasso-elastic-net-lars-curse-of-dimensionality-multiple-testing)
- [Week 4 — LDA, QDA, RDA, Logistic Regression](#week-4--lda-qda-rda-logistic-regression)
- [Week 5 — CART, Bootstrap, Bagging](#week-5--cart-bootstrap-bagging)
- [Week 6 — Random Forests, Boosting, AdaBoost, Gradient Boosting](#week-6--random-forests-boosting-adaboost-gradient-boosting)
- [Week 7 — Support Vector Machines](#week-7--support-vector-machines)
- [Week 8 — PCA, Sparse PCA, PLS, CCA](#week-8--pca-sparse-pca-pls-cca)
- [Week 9 — K-means, K-medoids, Hierarchical, GMM, Silhouette, Gap](#week-9--k-means-k-medoids-hierarchical-gmm-silhouette-gap)
- [Week 10 — Neural Networks, Backpropagation](#week-10--neural-networks-backpropagation)
- [Week 11 — NMF, ICA, Archetypal Analysis, Sparse Coding](#week-11--nmf-ica-archetypal-analysis-sparse-coding)
- [Week 12 — Tucker3, PARAFAC, CORCONDIA, Split-Half FMS](#week-12--tucker3-parafac-corcondia-split-half-fms)
- [Q22 — CV Design for Wearables Dataset](#q22--cv-design-for-wearables-dataset)
- [APPENDIX A — Neural Network Parameter Counting](#appendix-a--neural-network-parameter-counting)
- [APPENDIX B — Confusion Matrix, Sensitivity, Specificity](#appendix-b--confusion-matrix-sensitivity-specificity)
- [APPENDIX C — Nested CV](#appendix-c--nested-cv)
- [APPENDIX D — ICA Uniqueness Indeterminacies](#appendix-d--ica-uniqueness-indeterminacies)
- [APPENDIX E — Known Errors in Official Exam Solutions](#appendix-e--known-errors-in-official-exam-solutions)
- [APPENDIX F — Recurring Exam Patterns](#appendix-f--recurring-exam-patterns)

---

## Week 1 — OLS, Ridge, Bias-Variance, Cp, AIC, BIC

### The Bias-Variance Tradeoff

Every prediction model makes two kinds of errors: **bias** (systematic mismatch between the model family and reality) and **variance** (sensitivity to the particular training set you happened to draw). The key insight is that these trade off: a complex model fits the training data closely (low bias) but varies wildly across different datasets (high variance). A simple model is stable (low variance) but consistently wrong (high bias).

The **Expected Prediction Error (EPE)** quantifies this precisely. For a given test point $x_0$:

$$\text{EPE}(x_0) = \sigma^2 + \text{Bias}^2(\hat{f}(x_0)) + \text{Var}(\hat{f}(x_0))$$

Three terms, always. $\sigma^2$ is **irreducible noise** — the fundamental randomness in the data that no model can remove. Training error always decreases as complexity increases; test error forms a U-shape because variance eventually dominates.

### OLS (Ordinary Least Squares)

OLS minimizes the sum of squared residuals $\|y - X\beta\|^2$ and has a closed-form solution:

$$\hat{\beta}_\text{OLS} = (X^TX)^{-1}X^Ty$$

This requires $X^TX$ to be invertible — fails when $p \geq n$ or features are perfectly collinear. OLS is **unbiased**: $E[\hat{\beta}] = \beta$. By the **Gauss-Markov theorem**, OLS has the minimum variance among all linear unbiased estimators (BLUE). However, biased estimators like Ridge can have lower mean squared error by trading a little bias for much lower variance.

The **hat matrix** is $S = X(X^TX)^{-1}X^T$, so $\hat{Y} = SY$. The effective degrees of freedom of any linear smoother is $\text{df}(S) = \text{trace}(S)$.

### Ridge Regression

Ridge adds an $L_2$ penalty to prevent overfitting when features are correlated or $p$ is large relative to $n$:

$$\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1}X^Ty$$

The $\lambda I$ term makes the matrix always invertible — Ridge works even when $p > n$. The penalty shrinks all coefficients toward zero but **never sets any to exactly zero** — Ridge does not perform variable selection, only shrinkage. As $\lambda \to 0$, ridge → OLS. As $\lambda \to \infty$, all $\hat{\beta} \to 0$.

Effective degrees of freedom decrease as $\lambda$ increases:
$$\text{df}(\lambda) = \text{trace}\!\left(X(X^TX + \lambda I)^{-1}X^T\right)$$

At $\lambda = 0$: df = $p$. At $\lambda \to \infty$: df → 0.

### Cp Statistic

Training error $\overline{\text{err}}$ is optimistically biased because the model was fit on the same data. The optimism (how much training error underestimates test error) is proportional to complexity. The **Cp statistic** corrects for this:

$$C_p = \overline{\text{err}} + 2\frac{d}{N}\hat{\sigma}^2_e$$

$d$ = number of free parameters; $\hat{\sigma}^2_e$ = noise floor estimated from a low-bias (full) model — it is **fixed**, not from the current model. Minimize $C_p$ to select the best model.

### AIC and BIC

AIC generalizes Cp to any likelihood-based model:
$$\text{AIC} = -\frac{2}{N}\log L + \frac{2d}{N}$$

For Gaussian models, AIC and Cp are **identical**. AIC is asymptotically equivalent to **leave-one-out cross-validation** (Stone, 1977).

BIC penalizes complexity more aggressively:
$$\text{BIC} = -2\log L + d\log N$$

For $N > e^2 \approx 7$: BIC penalizes more than AIC → selects **simpler** models. BIC is **consistent** (recovers the true model as $N \to \infty$); AIC is **not** (over-selects complex models asymptotically). Use **AIC for prediction**, **BIC for model identification**.

### Key Traps W1
- Ridge has no closed form → **WRONG**. Ridge has closed form: $(X^TX + \lambda I)^{-1}X^Ty$
- Ridge performs variable selection → **WRONG**. Ridge only shrinks; Lasso sets to zero
- AIC is more conservative than BIC → **WRONG**. BIC penalizes more for large $N$
- AIC is asymptotically equivalent to k-fold CV → **WRONG**. It's equivalent to LOO-CV
- $\hat{\sigma}^2_e$ in Cp comes from the current model → **WRONG**. It's from a low-bias full model
- Adding more variables always improves Cp → **WRONG**. The penalty may increase Cp even if training error drops

---

## Week 2 — KNN, Model Selection vs Assessment, K-fold CV, 1-SE Rule

### KNN (K-Nearest Neighbours)

KNN predicts by looking at the $K$ closest training points (by Euclidean distance) and averaging (regression) or voting (classification). Standardize features first. The complexity is controlled entirely by $K$: **small $K$ = low bias, high variance** (jagged/overfit); **large $K$ = high bias, low variance** (smooth/underfit). Choose $K$ by cross-validation.

### Model Selection vs Model Assessment

These are two distinct problems that must be kept separate:
- **Model selection**: picking $\lambda$ or the model class — uses a validation set or CV.
- **Model assessment**: estimating the generalisation error of the *chosen* model — uses a test set.

The test set must be used **exactly once** at the very end. If you inspect the test error to make decisions, you have contaminated the assessment — the reported error will be optimistically biased.

### K-Fold Cross-Validation

Split data into $K$ equal folds. For each fold $k$: fit on the other $K-1$ folds, evaluate on fold $k$. Average the errors:

$$CV(\lambda) = \frac{1}{K}\sum_{k=1}^{K} \text{Err}_k(\lambda)$$

The SE of this estimate is:
$$\text{SE}(\lambda) = \frac{1}{\sqrt{K}}\sqrt{\frac{1}{K}\sum_k (\text{Err}_k(\lambda) - CV(\lambda))^2}$$

This SE is **biased downward** (underestimated) because fold errors are correlated — the $K-1$ training sets overlap heavily. Typical choices: $K = 5$ or $K = 10$. LOOCV ($K = N$) is essentially unbiased but gives very similar folds, making the SE estimate nearly meaningless.

**Critical rule**: All pre-processing (normalization, imputation) must happen **inside the CV loop** using only the training fold's statistics. Normalizing before splitting causes **data leakage**.

### 1-SE Rule

After CV, don't automatically pick the $\lambda$ with minimum CV error. Instead, the **1-SE rule** selects the **most regularized** model whose CV error is within 1 SE of the minimum. This yields a simpler, more stable model at almost no cost in accuracy. Source: Breiman et al. (1984) CART monograph.

### Optimism of Training Error

For a linear model with $d$ parameters:
$$E[\text{Err}_\text{in}] = E[\overline{\text{err}}] + \frac{2d}{N}\sigma^2_\varepsilon$$

The optimism ($2d/N)\sigma^2_\varepsilon$ grows with complexity $d$ — this is where Cp/AIC come from.

### Key Traps W2
- Test set can be used to select models → **WRONG**. Test set = assessment only
- LOOCV is always best → **WRONG**. Folds are highly correlated; $K=5$ or $K=10$ preferred
- 1-SE rule selects the minimum CV error model → **WRONG**. It selects the most regularized model within 1 SE
- CV SE estimate is unbiased → **WRONG**. Biased downward due to fold correlation
- Normalize before CV folds → **WRONG**. Normalize inside each fold separately
- Small $K$ in KNN = smooth boundaries → **WRONG**. Small $K$ = jagged; large $K$ = smooth

---

## Week 3 — Lasso, Elastic Net, LARS, Curse of Dimensionality, Multiple Testing

### Curse of Dimensionality

As dimension $p$ grows, the number of regions needed to cover the space grows exponentially. Most data points concentrate at the **boundaries/corners** of the hypercube rather than the interior. Euclidean distances become nearly equal for all point pairs — "all points are equidistant." Local neighborhoods become empty, destroying KNN and similar methods. When $p > N$, OLS can perfectly fit noise (non-invertible $X^TX$), making regularization mandatory.

Three "blessings" (Donoho 2000): features are correlated (can average), data lies on a low-dimensional manifold, continuous processes have approximate finite dimensionality.

### Lasso

Lasso adds an $L_1$ penalty:
$$\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1$$

The $L_1$ constraint forms a **diamond** in coefficient space. The RSS ellipsoid tends to hit the diamond at its **corners**, where some $\beta_j = 0$ exactly. This is why Lasso performs **variable selection** — it produces sparse solutions. The number of non-zero coefficients approximates the effective df. When $p > n$, Lasso selects at most $n$ variables. **No closed form** — must be solved iteratively.

### Ridge vs Lasso

| Property | Ridge | Lasso |
|----------|-------|-------|
| Penalty | $L_2$: $\|\beta\|_2^2$ | $L_1$: $\|\beta\|_1$ |
| Constraint shape | Sphere (smooth) | Diamond (corners) |
| Exact zeros | Never | Yes (sparse) |
| Variable selection | No | Yes |
| Closed form | Yes | No |
| Works when $p > n$ | Yes | Yes (selects $\leq n$ vars) |
| Correlated groups | Shrinks together | Picks one, ignores others |

### Elastic Net

Elastic Net combines both penalties:
$$\lambda\!\left[\frac{1-\alpha}{2}\|\beta\|_2^2 + \alpha\|\beta\|_1\right]$$

$\alpha = 1$ → Lasso; $\alpha = 0$ → Ridge; $0 < \alpha < 1$ → Elastic Net. Solves Lasso's three limitations: (1) $p \gg n$, (2) grouped correlated features (Lasso picks one arbitrarily; EN selects the group), (3) predictive accuracy when features are correlated. Implemented by augmenting $X$ with $\sqrt{\lambda_2} \cdot I$ rows and $y$ with zeros, then solving standard Lasso.

### LARS (Least Angle Regression)

LARS computes the entire Lasso regularization path at the cost of one OLS fit. It moves in the **equiangular direction** — bisecting the angle between the current residual and the active predictors. At each step the active predictor with the highest correlation to the residual is brought in. For Lasso-LARS: if a coefficient crosses zero during the path, drop it from the active set. Data must be centered and normalized.

### Coordinate Descent

Cyclically updates one $\beta_j$ at a time (all others fixed). The update applies **soft thresholding**:
$$\text{sign}(x)(|x| - \lambda)_+$$
Values with $|x| \leq \lambda$ are zeroed; larger values are shrunk by $\lambda$. This produces exact zeros and is how Lasso is computed in practice.

### Multiple Testing

Testing $M$ independent hypotheses at level $\alpha$, the probability of at least one false positive (Family-Wise Error Rate) is:
$$\text{FWER} = 1 - (1-\alpha)^M$$

For $M = 20$, $\alpha = 0.05$: FWER $\approx 64\%$ — you almost certainly make at least one mistake.

**Bonferroni**: reject if $p$-value $< \alpha/M$. Strict, controls FWER, but very low power (misses real effects).

**BH (Benjamini-Hochberg)**: controls **FDR** = expected fraction of false discoveries among all discoveries. Procedure:
1. Sort $p$-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find the **largest** $k$ such that $p_{(k)} \leq \frac{k}{m}q$
3. Reject hypotheses $1, \ldots, k$

BH has more power than Bonferroni. Typical $q = 0.1$ or $0.2$.

### Key Traps W3
- Lasso always beats Ridge with correlated predictors → **WRONG**. Ridge often outperforms when predictors are correlated and $n > p$
- Elastic Net with $\alpha = 0$ is Lasso → **WRONG**. $\alpha = 0$ is Ridge; $\alpha = 1$ is Lasso
- LARS adds one full variable at each step → **WRONG**. LARS moves equiangularly, not by full variable jumps
- BH uses a fixed threshold → **WRONG**. BH threshold is adaptive: $i \cdot q / m$ increases with rank
- FWER and FDR control the same thing → **WRONG**. FWER = probability of any false positive; FDR = fraction of discoveries that are false
- After BH: reject only the hypotheses that individually pass their threshold → **WRONG**. Find the largest $k$ that passes, then reject all $1, \ldots, k$

---

## Week 4 — LDA, QDA, RDA, Logistic Regression

### Generative vs Discriminative

Two philosophies for classification:
- **Generative** (LDA, QDA): model how each class generates data — $P(X | G=k)$ — then apply Bayes' theorem to get $P(G=k|X)$.
- **Discriminative** (Logistic Regression): model $P(G=k|X)$ directly, never bothering to model the data distribution.

Generative models are more efficient when assumptions hold; discriminative models are more robust when they don't.

### Bayes' Theorem for Classification

$$P(G=k|X=x) = \frac{f_k(x)\pi_k}{\sum_{l=1}^K f_l(x)\pi_l}$$

$f_k(x) = P(X=x|G=k)$ is the class-conditional density; $\pi_k = N_k/N$ is the prior. Classify to the class with highest posterior.

### LDA (Linear Discriminant Analysis)

LDA assumes all classes share **one pooled covariance matrix** $\Sigma$. With equal $\Sigma$, the quadratic term $x^T\Sigma^{-1}x$ is the same for all classes and cancels in the log-odds — leaving a **linear** decision boundary in $x$.

**Discriminant function:**
$$\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k$$

Classify: $\hat{G}(x) = \arg\max_k \delta_k(x)$.

**Parameter estimates:**
- $\hat{\pi}_k = N_k/N$
- $\hat{\mu}_k = \frac{1}{N_k}\sum_{g_i=k} x_i$
- Pooled covariance: $\hat{\Sigma} = \frac{1}{N-K}\sum_k\sum_{g_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$

LDA has a closed-form solution. Fails when $p \gg N$ ($\hat{\Sigma}$ becomes singular).

### QDA (Quadratic Discriminant Analysis)

QDA drops the equal-covariance assumption — each class gets its own $\Sigma_k$. The quadratic term no longer cancels → **quadratic** decision boundaries (ellipses, parabolas, hyperbolas). Much more flexible than LDA but requires $O(p^2)$ parameters per class — breaks down in high dimensions.

### RDA — Regularized Discriminant Analysis

Bridges QDA and LDA by interpolating between per-class and pooled covariance:

$$\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}$$

$\alpha = 1$ → QDA (per-class); $\alpha = 0$ → LDA (pooled). A second parameter $\gamma$ shrinks toward diagonal or spherical covariance. Tune $\alpha$, $\gamma$ by cross-validation. Solves the $p \gg N$ singularity problem.

**RRDA (Reduced Rank DA)**: projects into a $K-1$ dimensional subspace that maximizes class separation. Excellent for 2D visualization of multi-class data.

### Logistic Regression

For binary classification, logistic regression models the log-odds of class membership as linear in $x$:
$$\log\frac{P(Y=1|x)}{P(Y=0|x)} = \beta_0 + \beta^Tx$$

The probability is:
$$P(Y=1|x) = \frac{e^{\beta_0+\beta^Tx}}{1+e^{\beta_0+\beta^Tx}}$$

Decision boundary: $\beta_0 + \beta^Tx = 0$ — **linear** in $x$ (same position as LDA, different estimation). **No closed form** — fit by maximizing log-likelihood iteratively via **Newton-Raphson** (IRLS). Coefficient $\beta_j$ = change in log-odds per unit $x_j$; $e^{\beta_j}$ = multiplicative change in **odds** (not probability).

Log-likelihood:
$$\ell(\beta) = \sum_{i=1}^N \left[y_i(\beta^Tx_i) - \log(1 + e^{\beta^Tx_i})\right]$$

| Property | LDA | Logistic Regression |
|----------|-----|---------------------|
| Type | Generative | Discriminative |
| Assumes class distribution | Yes (Gaussian) | No |
| Equal covariance | Yes | No |
| Decision boundary | Linear | Linear |
| Closed form | Yes | No |
| Robust to non-Gaussian | No | Yes |

### Key Traps W4
- LDA has per-class covariance → **WRONG**. LDA uses ONE pooled $\Sigma$; QDA uses per-class $\Sigma_k$
- LDA has a quadratic boundary → **WRONG**. Equal $\Sigma$ cancels the quadratic term → strictly linear
- Logistic regression models $P(X|G)$ → **WRONG**. Logistic regression is discriminative; it models $P(G|X)$
- Logistic regression has a closed form → **WRONG**. Requires iterative Newton-Raphson
- $e^{\beta_j}$ is the change in probability → **WRONG**. $e^{\beta_j}$ is the change in ODDS (multiplicative)
- RDA with $\alpha = 1$ gives LDA → **WRONG**. $\alpha = 1$ gives QDA; $\alpha = 0$ gives LDA

---

## Week 5 — CART, Bootstrap, Bagging

### Decision Trees (CART)

A decision tree recursively splits the feature space into rectangles. At each split, choose the feature and threshold that most reduces node impurity. For classification:

| Measure | Formula | Use |
|---------|---------|-----|
| Gini | $G = \sum_k \hat{p}_{mk}(1-\hat{p}_{mk}) = 2p(1-p)$ (binary) | **Growing** trees |
| Cross-entropy | $D = -\sum_k \hat{p}_{mk}\log\hat{p}_{mk} = -p\log p - (1-p)\log(1-p)$ | **Growing** trees |
| Misclassification | $E = 1 - \max_k \hat{p}_{mk} = \min(p, 1-p)$ | **Pruning / evaluation** |

Gini and cross-entropy are more sensitive to probability changes than misclassification rate — they detect improvements that misclassification rate misses when growing.

**Pruning** uses cost-complexity: $C_\alpha(T) = R(T) + \alpha|T|$, where $|T|$ is the number of terminal nodes. Tune $\alpha$ by cross-validation. At $\alpha = 0$: full tree. As $\alpha$ grows: tree shrinks to a stump.

### Bootstrap

A bootstrap sample draws $N$ observations **with replacement** from $N$ training observations. On average, $\approx 63.2\%$ of unique observations appear in each bootstrap sample:
$$P(\text{obs NOT in sample}) = \left(1 - \frac{1}{N}\right)^N \to \frac{1}{e} \approx 0.368$$

The remaining $\approx 36.8\%$ of observations that were NOT selected are the **out-of-bag (OOB)** sample — a free validation set.

### Bagging (Bootstrap Aggregating)

Bagging trains $B$ models on $B$ bootstrap samples and averages (regression) or majority-votes (classification). Why does averaging help? Because the **bias of the average equals the bias of one tree** (identically distributed trees), but the **variance decreases**:

$$\text{Var}(\hat{y}_\text{bag}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$, the second term → 0, leaving a floor of $\rho\sigma^2$ where $\rho$ is the pairwise inter-tree correlation. Bagging:
- **Reduces variance** — yes
- **Reduces bias** — NO (bias of average = bias of one tree)
- Works best for **high-variance, low-bias** methods (deep unpruned trees)
- Never causes overfitting as $B$ increases
- Loses interpretability (ensemble, not a single tree)

**OOB error** $\approx$ LOOCV error — unbiased, free by-product of bagging.

### Key Traps W5
- Bagging reduces bias → **WRONG**. Bagging reduces VARIANCE only; bias unchanged
- More trees causes overfitting → **WRONG**. Bagging never overfits with more trees
- OOB error is optimistic like training error → **WRONG**. OOB error is UNBIASED (like CV)
- Variance reduction is complete ($\sigma^2/B$) → **WRONG**. Floor at $\rho\sigma^2$ due to tree correlation
- Bagging trees should be pruned → **WRONG**. Use UNPRUNED trees; pruning makes variance reduction less effective
- Misclassification rate is used to grow trees → **WRONG**. Gini/entropy are used for growing; misclassification for pruning

---

## Week 6 — Random Forests, Boosting, AdaBoost, Gradient Boosting

### Random Forests

RF = Bagging + random feature subsampling. At each split, only $m < p$ randomly chosen features are considered. This **decorrelates** the trees (lowers $\rho$ in the variance formula), breaking through the bagging floor:

$$\text{Var}_\text{RF} = \rho_\text{low}\sigma^2 + \frac{1-\rho_\text{low}}{B}\sigma^2 < \text{Var}_\text{bag}$$

Default $m$: classification $= \lfloor\sqrt{p}\rfloor$; regression $= \lfloor p/3 \rfloor$. When $m = p$ → RF reduces to plain Bagging. RF trees are grown deep (no pruning). RF does **NOT reduce bias**. RF can handle $p > n$. RF trees are independent → **parallelizable**.

**Variable importance**: (1) Gini importance — total impurity reduction across all splits on feature $j$; (2) OOB permutation — randomly permute feature $j$ for OOB samples, measure accuracy drop. Both give similar rankings; Gini concentrates on top features more.

**Proximity matrix**: $n \times n$ matrix where $P(i,j)$ is incremented when OOB observations $i$ and $j$ land in the same terminal node. Large $P(i,j)$ → similar observations. Visualized with MDS.

### Boosting

Boosting trains trees **sequentially** — each tree corrects the errors of the previous ensemble. Uses **weak learners** (shallow trees, stumps). Unlike bagging, boosting primarily **reduces bias**, making it suitable for high-bias, low-complexity models.

Key differences from bagging:
| | Bagging / RF | Boosting |
|--|-------------|---------|
| Tree dependence | Independent (parallel) | Sequential (dependent) |
| Target | High-variance methods | High-bias methods (stumps) |
| Effect | Reduces variance | Reduces bias (and variance) |
| Overfitting risk | None | Can overfit noisy data |

### AdaBoost.M1

Binary classification ($y \in \{-1, +1\}$). Algorithm:
1. Initialize: $w_i = 1/N$ for all $i$
2. For $m = 1, \ldots, M$:
   a. Fit tree $G_m(x)$ using weights $w_i$
   b. Compute weighted error: $\text{err}_m = \frac{\sum_i w_i \cdot \mathbf{I}(y_i \neq G_m(x_i))}{\sum_i w_i}$
   c. Classifier weight: $\alpha_m = \log\!\left[\frac{1-\text{err}_m}{\text{err}_m}\right]$
   d. Update weights: $w_i \leftarrow w_i \cdot \exp[\alpha_m \cdot \mathbf{I}(y_i \neq G_m(x_i))]$, renormalize
3. Final: $G(x) = \text{sign}\!\left[\sum_m \alpha_m G_m(x)\right]$

$\text{err}_m = 0.5$ → $\alpha_m = 0$ (classifier contributes nothing). $\text{err}_m = 0$ → $\alpha_m = \infty$. AdaBoost is equivalent to forward stagewise additive modeling with **exponential loss** — sensitive to noise/outliers (exponential growth of misclassification penalty).

### Gradient Boosting

At each step, fit a tree to the **negative gradient** of the loss function (pseudo-residuals). For squared-error loss: pseudo-residuals = ordinary residuals $y_i - F_{m-1}(x_i)$. For general loss: use the gradient.

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

**Shrinkage** (learning rate $\nu$): scale each tree's contribution by $0 < \nu < 1$. Slower convergence but better generalization. Tree depth determines interaction order: stumps = additive (no interactions); $J$-leaf tree = up to $(J-1)$-way interactions.

**Loss functions**: Exponential (AdaBoost) — sensitive to noise. Binomial deviance — more robust for noisy/mislabeled data.

### Key Traps W6
- RF reduces bias → **WRONG**. RF reduces VARIANCE (by decorrelating trees); bias unchanged
- $m = p$ in RF gives a stronger model → **WRONG**. $m = p$ → same as Bagging (no decorrelation)
- Boosting reduces variance only → **WRONG**. Boosting primarily reduces BIAS
- Boosting uses deep trees → **WRONG**. Boosting uses SHALLOW trees/stumps
- Boosting never overfits → **WRONG**. Can overfit with noisy data or mislabeled observations
- Gradient boosting fits to residuals in all cases → **WRONG**. Only for squared-error loss; general loss uses negative gradient (pseudo-residuals)
- RF proximity matrix is based on training data → **WRONG**. Based on OOB samples

---

## Week 7 — Support Vector Machines

### Core Idea

SVM finds the hyperplane that **maximizes the margin** — the perpendicular distance between the boundary and the nearest training points of each class (support vectors). Maximum margin = most robust boundary. Labels are $\{-1, +1\}$ (not $\{0, 1\}$).

**Key geometry**: $\beta$ is **orthogonal** to the hyperplane. The total margin width is $2/\|\beta\|$ (between the two support hyperplanes $x^T\beta + \beta_0 = \pm 1$). Maximizing the margin = **minimizing** $\|\beta\|$.

### Primal SVM

$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{subject to} \quad y_i(x_i^T\beta + \beta_0) \geq 1 \;\; \forall i$$

The constraint ensures each point is at least 1 canonical unit from the boundary (after normalizing to the canonical hyperplane). Solved by **Quadratic Programming**.

### Dual SVM and the Kernel Trick

The Lagrangian formulation leads to the dual:
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle \quad \text{s.t.} \quad \alpha_i \geq 0$$

Data only appears as **dot products** $\langle x_i, x_j \rangle$ in the dual. Replace any dot product with a kernel function $K(x_i, x_j)$ and you implicitly map to a higher-dimensional feature space — this is the **kernel trick**. The RBF kernel:
$$K(x, x') = \exp(-\gamma\|x-x'\|^2)$$
corresponds to an **infinite-dimensional** feature space. Never need to compute the actual high-dimensional coordinates.

### Support Vectors and Sparsity

**KKT complementary slackness**: $\alpha_i[y_i(x_i^T\beta + \beta_0) - 1] = 0$ for every $i$.

- **Support vectors**: points ON the margin ($|x_i^T\beta + \beta_0| = 1$); have $\alpha_i > 0$
- **Safe points**: points far from the margin (bracket > 0); **must** have $\alpha_i = 0$

You can delete all non-support vectors and the boundary does not move. Typically only a small fraction of training points are support vectors.

**Weak vs Strong Duality**: Weak duality: $d^* \leq p^*$ always. Strong duality: $d^* = p^*$ — holds for SVM via Slater's condition (convex + strictly feasible).

SVM has **no probabilistic model** — purely geometric.

### Soft Margin SVM

The standard SVM above requires linear separability. Soft-margin SVM adds slack variables $\xi_i \geq 0$ to allow some violations:
$$\min \frac{1}{2}\|\beta\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(x_i^T\beta + \beta_0) \geq 1 - \xi_i$$

$C$ controls the trade-off: large $C$ = strict margin (few violations), small $C$ = wider margin (more violations allowed).

### Key Traps W7
- SVM uses a probabilistic model → **WRONG**. SVM is purely geometric
- All training points define the boundary → **WRONG**. Only support vectors define it
- Removing non-support vectors changes the boundary → **WRONG**. Boundary is identical
- $\alpha_i > 0$ for all points → **WRONG**. $\alpha_i = 0$ for safe points
- The kernel maps data explicitly → **WRONG**. The mapping is implicit; only compute $K(x_i, x_j)$
- RBF maps to finite-dimensional space → **WRONG**. RBF = infinite-dimensional feature space
- Weak duality: $d^* = p^*$ → **WRONG**. Weak duality means $d^* \leq p^*$
- $\beta$ is parallel to the hyperplane → **WRONG**. $\beta$ is ORTHOGONAL to the hyperplane
- Labels are 0 and 1 in SVM → **WRONG**. Labels are $-1$ and $+1$

---

## Week 8 — PCA, Sparse PCA, PLS, CCA

### PCA (Principal Component Analysis)

PCA finds directions of maximum **variance** in $X$. Objective:
$$\max_v \text{Var}(Xv) = \max_v v^T\Sigma v \quad \text{s.t.} \quad \|v\|=1$$

PCA is **unsupervised** — it ignores any response $y$. The $k$-th eigenvalue $\lambda_k$ = variance explained by PC $k$; proportion explained = $\lambda_k / \sum_j \lambda_j$.

**EVD vs SVD**: EVD is computed on the $p \times p$ covariance/correlation matrix; SVD is computed on the $n \times p$ data matrix $X$ directly. Both give the **same loading vectors $V$**. SVD standard deviation: $\sigma_l = d_l / \sqrt{n-1}$ where $d_l$ is the $l$-th singular value.

Mode of variation plot: $\mu \pm 2.5\sigma_l v_l$ shows how data varies along the $l$-th PC.

**Scaling matters**: unscaled PCA is dominated by high-variance features. Use the correlation matrix (standardized data) for equal weighting.

Principal components are orthogonal; scores are uncorrelated.

### Sparse PCA

Standard PCA loadings use all $p$ features — hard to interpret when $p$ is large. Sparse PCA zeros out many loadings for interpretability. Three methods:
1. **Thresholding**: zero out loadings below a threshold; scores must be **recomputed** after
2. **Varimax rotation**: orthogonal rotation to maximize loading simplicity
3. **Elastic Net**: $L_1 + L_2$ penalty, most principled sparse solution

After any of these three methods: **uncorrelatedness of scores is NOT guaranteed**. All three destroy PCA's orthogonality.

### PLS (Partial Least Squares)

PLS is **supervised** — uses $y$ to find the subspace of $X$ most relevant for prediction. Objective:
$$\max_{u,v} \text{Cov}(Xu, Yv)$$

Covariance $= \sqrt{\text{Var}(Xu) \cdot \text{Var}(Yv)} \cdot \text{Corr}(Xu, Yv)$ — PLS balances both variance AND correlation. PCR flaw: PCA may discard the X-directions most predictive of $y$ (highest variance ≠ most predictive of $y$). PLS automatically focuses on predictive directions.

PLS latent components are **uncorrelated** ($z_i^Tz_j = 0$). With $M = p$ components: PLS = OLS (no regularization). Choose $M < p$ by cross-validation.

### CCA (Canonical Correlation Analysis)

CCA finds pairs of directions $(u, v)$ that maximize the **correlation** between two matrices $X$ and $Y$:
$$(u^T\Sigma_{XY}v)^2 / (u^T\Sigma_{XX}u \cdot v^T\Sigma_{YY}v)$$

CCA ignores internal variance of $X$ and $Y$ — pure correlation. Requires inverting $\Sigma_{XX}$ and $\Sigma_{YY}$ → fails when $p > n$. Solutions: **Regularized CCA** (Ridge: add $\lambda I$) or **Sparse CCA** (PMD: $L_1$ penalty on $u$ and $v$).

| Method | Objective | Supervised | Works when $p > n$ |
|--------|-----------|------------|-------------------|
| PCA | Max variance of $Xv$ | No | Yes (use SVD) |
| PLS | Max $\text{Cov}(Xu, Yv)$ | Yes | Yes |
| CCA | Max $\text{Corr}(Xu, Yv)$ | Two-matrix | No (need regularization) |

### Key Traps W8
- PCA maximizes correlation with $y$ → **WRONG**. PCA maximizes variance; knows nothing about $y$
- PCR is always better than PLS → **WRONG**. PCR can fail if high-variance X directions have zero correlation with $y$
- EVD and SVD give different loadings → **WRONG**. Both give the same loading vectors $V$
- PLS components are correlated → **WRONG**. PLS components are orthogonal by the deflation step
- PLS with $M = p$ is regularized → **WRONG**. $M = p$ → equivalent to OLS
- CCA maximizes variance → **WRONG**. CCA maximizes CORRELATION only
- Sparse PCA scores remain uncorrelated after thresholding → **WRONG**. Thresholding destroys orthogonality

---

## Week 9 — K-means, K-medoids, Hierarchical, GMM, Silhouette, Gap

### General Clustering

Clustering is **unsupervised** — no response variable. **Critical warning**: clustering algorithms will ALWAYS produce a grouping, even on completely random data. You need domain knowledge to assess whether discovered clusters are meaningful. Standard cross-validation does not apply to clustering.

### K-means

Partitions $n$ observations into $K$ clusters minimizing total within-cluster squared Euclidean distance:
$$\min_{C_1,\ldots,C_K} \sum_{k=1}^K \sum_{i \in C_k} \|x_i - \mu_k\|^2$$

Algorithm: assign each point to its nearest centroid → update centroids as cluster means → repeat until convergence. **Hard** assignments. $K$ must be pre-specified. **Not guaranteed global optimum** — use multiple random restarts. Only uses **Euclidean distance**. Favors convex, spherical, similarly-sized clusters. Sensitive to outliers.

### K-medoids

Like K-means but cluster centers must be **actual data points** (medoids). More robust to outliers. Works with **any distance measure** (not just Euclidean).

### Hierarchical Clustering

Produces a **dendrogram** — no need to pre-specify $K$. Bottom-up (agglomerative): start with $n$ singleton clusters, iteratively merge the closest pair. Cut at any height to get any number of clusters.

Linkage methods:
- **Single**: distance = closest pair → tends to chain (elongated clusters)
- **Complete**: distance = farthest pair → compact, balanced clusters
- **Ward**: minimize increase in total within-cluster variance → requires **Euclidean distance**

### Gaussian Mixture Models (GMM)

Probabilistic clustering: $X_i \sim \mathcal{N}(\mu_j, \Sigma_j)$ with probability $\pi_j$ (mixing proportion). Gives **soft assignments** $\gamma_{ij} = P(Z_i = j | x_i) \in [0, 1]$ instead of hard cluster labels.

Fitted by the **EM algorithm**:
- **E-step**: compute posterior probabilities $\gamma_{ij}$ using Bayes' rule (given current parameters)
- **M-step**: update parameters as weighted averages using $\gamma_{ij}$ as weights

$$\gamma_{ij} = \frac{\pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}{\sum_{j'} \pi_{j'} \mathcal{N}(x_i; \mu_{j'}, \Sigma_{j'})}$$

$$\mu_j^{(\text{new})} = \frac{\sum_i \gamma_{ij} x_i}{\sum_i \gamma_{ij}}, \quad \Sigma_j^{(\text{new})} = \frac{\sum_i \gamma_{ij}(x_i - \mu_j)(x_i - \mu_j)^T}{\sum_i \gamma_{ij}}, \quad \pi_j^{(\text{new})} = \frac{1}{n}\sum_i \gamma_{ij}$$

K-means is a special case of GMM with hard assignments and equal spherical covariances. Validate GMM using **AIC or BIC** (not silhouette or gap).

### Silhouette Method

For each observation $i$:
$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

$a(i)$ = average distance to points in **same cluster** (cohesion — lower is better); $b(i)$ = average distance to points in **neighboring cluster** (the nearest cluster $i$ is NOT in). $s(i) \in [-1, 1]$: 1 = perfect, 0 = on boundary, negative = likely misclassified.

Choose $K$ with maximum average silhouette, or the smallest $K$ where all clusters have observations above average. Favors **convex, spherical clusters**; unreliable for other shapes.

### Gap Statistic

Compares actual within-cluster dissimilarity to expected dissimilarity for **uniformly distributed random data** (no structure):

$$G(K) = \log(U_K) - \log(W_K)$$

$W_K$ = actual within-cluster dissimilarity; $U_K$ = average over 20 simulations of uniform data. Large $G(K)$ means the actual clustering is much tighter than random → real structure. Choose:
$$K^* = \arg\min_K \{K \mid G(K) \geq G(K+1) - s'_{K+1}\}$$

where $s'_{K+1} = \text{std}(\log U_K) \cdot \sqrt{1 + 1/20}$. Works for K-means, K-medoids, and hierarchical. More statistically principled than silhouette.

### Key Traps W9
- K-means guarantees the global optimum → **WRONG**. Local optimum only; use multiple restarts
- Clustering only works when real clusters exist → **WRONG**. Always produces a grouping, even on random data
- K-medoids uses cluster means → **WRONG**. K-medoids uses actual data points (medoids)
- Ward linkage works with any distance → **WRONG**. Ward requires EUCLIDEAN distance specifically
- Single linkage produces compact clusters → **WRONG**. Single = chaining; complete = compact
- GMM gives hard assignments → **WRONG**. GMM gives soft (probabilistic) assignments $\gamma_{ij}$
- $\pi_j$ in GMM is the cluster mean → **WRONG**. $\pi_j$ is the mixing proportion (prior probability)
- E-step in EM updates parameters → **WRONG**. E-step computes soft assignments; M-step updates parameters
- K-means works with Manhattan distance → **WRONG**. K-means uses Euclidean only; K-medoids can use any distance

---

## Week 10 — Neural Networks, Backpropagation

### The Four Ingredients of Deep Learning

1. **Data** — $(x_i, y_i)$ training pairs
2. **Objective/Loss** — derived from negative log-likelihood (not arbitrary): $w^* = \arg\min_w -\log \ell(D; w)$
3. **Engine** — gradient descent + backpropagation
4. **Architecture** — MLP, CNN, RNN, Transformer, Autoencoder

Loss functions are not design choices — they follow from the likelihood assumption:
- **MSE** ↔ Gaussian likelihood (regression)
- **Binary Cross-Entropy** ↔ Bernoulli likelihood (binary classification)

BCE: $-\sum_i [y_i \ln \hat{y}_i + (1-y_i)\ln(1-\hat{y}_i)]$

### The Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}, \qquad \sigma'(x) = \sigma(x)(1-\sigma(x))$$

The derivative only needs the sigmoid output — no need to recompute $e^{-x}$. Output used as $P(y=1|x)$ in binary classification. Vanishing gradient problem: $\sigma'(x) \leq 0.25$ — multiplied across many layers → exponential decay of gradients. ReLU ($\max(0,x)$) avoids this.

### Forward and Backward Pass

**Forward pass**: compute activations layer by layer.
$$z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}, \qquad a^{(\ell)} = \sigma(z^{(\ell)})$$

All intermediate values $z^{(\ell)}$ and $a^{(\ell)}$ must be **stored** — they are required for the backward pass.

**Backward pass**: propagate error signals (gradients) backward using the chain rule.
$$\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$$

$$\frac{\partial L}{\partial W^{(\ell)}} = \delta^{(\ell)} \times (a^{(\ell-1)})^T$$

The $(W^T\delta)$ term arises because each activation in layer $\ell$ connects to ALL neurons in layer $\ell+1$ — the gradient must sum over all those paths (multivariate chain rule).

**Gradient descent**: $w \leftarrow w - \eta \cdot \nabla_w L$. Too large $\eta$ → diverge; too small → slow convergence. Deep networks have non-convex loss landscapes — no global minimum guarantee.

### Architecture Comparison

| Architecture | Key property | Best for |
|---|---|---|
| MLP | Fully connected | Tabular/fixed-size data |
| CNN | Weight sharing, translation equivariance | Images/grids |
| RNN | Sequential hidden state | Sequences (short) — vanishing gradient issue |
| LSTM/GRU | Gated memory | Sequences (long-range) |
| Autoencoder | Encoder → bottleneck → decoder | Unsupervised representation learning |
| Transformer | Self-attention, parallelizable | Long-range sequences, LLMs |

### Key Traps W10
- BCE is an arbitrary design choice → **WRONG**. BCE is the exact negative log-likelihood under Bernoulli
- $\sigma'(x)$ requires recomputing $e^{-x}$ → **WRONG**. $\sigma'(x) = \sigma(x)(1-\sigma(x))$
- Backprop stores nothing during forward pass → **WRONG**. Must store all $z^{(\ell)}$ and $a^{(\ell)}$
- Gradients flow forward → **WRONG**. Activations go forward; gradients go backward
- Vanishing gradient is a CNN problem → **WRONG**. Vanishing gradient is the specific problem of RNNs on long sequences
- Autoencoders are supervised → **WRONG**. Autoencoders are UNSUPERVISED; reconstruction uses the input as its own target
- Transformers use sequential RNN-style processing → **WRONG**. Self-attention is fully parallelizable

---

## Week 11 — NMF, ICA, Archetypal Analysis, Sparse Coding

### The Unifying Framework

All four methods approximate $X \approx WH$ but with different structural constraints:

| Method | Constraint | Interpretation |
|--------|-----------|---------------|
| NMF | $W \geq 0$, $H \geq 0$ | Parts-based, additive |
| ICA | rows of $H$ independent and non-Gaussian | Independent source signals |
| AA | archetypes on convex hull, convex mixtures | Extreme prototypes |
| Sparse Coding | $H$ sparse (mostly zeros) | Few active atoms at a time |

### NMF (Non-negative Matrix Factorization)

Objective: $\min_{W,H \geq 0} \frac{1}{2}\|X - WH\|_F^2$

Non-negativity enforces **parts-based** additive representation — no cancellation between basis vectors. Think face images: each component is a part (eyes, nose), and the face is a sum of parts.

NMF is **NOT jointly convex** in $(W, H)$ — only convex in one given the other fixed. This justifies alternating least squares (ALS). Multiplicative updates guarantee non-negativity without projection:
$$H_{kj} \leftarrow H_{kj} \cdot \frac{(W^T X)_{kj}}{(W^T WH)_{kj}}, \quad W_{ik} \leftarrow W_{ik} \cdot \frac{(XH^T)_{ik}}{(WHH^T)_{ik}}$$

These are gradient descent with a spatially-varying learning rate $\eta_H = H/(W^TWH)$.

NMF solutions are **NOT unique**: $WH = (WQ^{-1})(QH)$ for any invertible $Q$ (as long as non-negativity holds). Disambiguation: geometric constraints or sparsity penalties.

### ICA (Independent Component Analysis)

ICA assumes a **linear mixing model**: $x = As$, where $s$ are unknown independent source signals. Goal: find $W \approx A^{-1}$ such that $\hat{s} = Wx$ recovers the sources.

ICA requires sources to be (1) statistically independent AND (2) **non-Gaussian**. It CANNOT separate Gaussian sources. Why? The CLT says mixing non-Gaussian signals makes them more Gaussian. ICA works in reverse: find the $W$ that maximizes non-Gaussianity of the output — the maximally non-Gaussian direction is the unmixed source. For Gaussians, all rotations are equally Gaussian → no unique solution.

**Non-Gaussianity measures**: excess kurtosis $= \mu_4/\sigma^4 - 3$ (Gaussian = 0; Laplace = 3; Uniform ≈ -1.2); negentropy.

**Whitening** (pre-processing): transform data so $E[\tilde{x}\tilde{x}^T] = I$ — reduces the problem to finding rotations only.

**FastICA** update: $w_\text{new} \leftarrow E[\tilde{x}\, g(w^T\tilde{x})] - E[g'(w^T\tilde{x})]\, w$; then normalize $w \leftarrow w/\|w\|$. Cubic/quadratic convergence. For multiple components: deflationary approach — project out found components.

**PCA vs ICA**: PCA finds uncorrelated components (second-order); ICA finds statistically independent components (all orders). Independence is stricter than uncorrelatedness.

### Archetypal Analysis (AA)

AA finds extreme prototypes on the **boundary** (convex hull) of the data, not centroids. Objective:
$$\min_{S,H} \|X - XSH\|_F^2$$

Constraints: $S_{ij} \geq 0$, $\sum_i S_{ij} = 1$ (archetypes = convex combinations of data); $H_{ij} \geq 0$, $\sum_j H_{ij} = 1$ (data = convex combinations of archetypes). Archetypes: $Z = XS$ (must be built from actual data points).

**AA vs K-means**: AA places prototypes at extremes (convex hull); K-means places centroids in the interior. **AA vs PCA**: PCA finds average profile; AA finds extreme profiles. **AA vs NMF**: NMF has $W$ arbitrary; AA anchors archetypes to the data ($Z = XS$).

### Sparse Coding

Uses an **overcomplete dictionary** $W$ (more columns than input dimensions: $K > I$) where each observation is represented by a **sparse** $h$ (few non-zero entries):
$$L(W,H) = \frac{1}{2}\|X - WH\|_F^2 + \lambda\sum_j \|h_j\|_1$$

Step 1 (fix $W$, solve for $h$): reduces to **Lasso** — solve with Coordinate Descent or LARS. Step 2 (fix $H$, solve for $W$): standard least squares with unit norm constraint $\|w_k\|_2 \leq 1$. The unit norm constraint is **required** — without it, scaling $W \to \infty$ and $H \to 0$ trivially drives $L_1$ to zero.

**Cross-validation for matrix methods**: standard row-holdout fails (cannot learn $H$ for held-out sample). Use **Speckled CV** — randomly mask individual matrix entries; train ignoring masked entries; evaluate on masked entries only.

### Key Traps W11
- NMF is convex → **WRONG**. NOT jointly convex in $(W,H)$
- NMF solutions are unique → **WRONG**. $Q$-ambiguity exists for any invertible $Q$
- ICA requires Gaussian sources → **WRONG**. ICA REQUIRES NON-Gaussian sources
- PCA and ICA find the same components → **WRONG**. PCA = uncorrelated; ICA = statistically independent
- Whitening is optional in ICA → **WRONG**. Whitening is required preprocessing
- AA archetypes can be any point → **WRONG**. Archetypes must be convex combinations of real data ($Z = XS$)
- Skip unit norm constraint on $W$ in sparse coding → **WRONG**. Without it, trivially drives $L_1$ to zero
- Row-holdout CV works for NMF → **WRONG**. Use Speckled CV (mask individual entries)
- $L_2$ regularization causes sparsity → **WRONG**. $L_1$ causes exact zeros; $L_2$ only shrinks

---

## Week 12 — Tucker3, PARAFAC, CORCONDIA, Split-Half FMS

### Tensor Notation

A 3-way tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$ has three modes. Frobenius norm: $\|\mathcal{X}\|_F = \sqrt{\sum_{ijk} x_{ijk}^2}$.

**N-mode multiplication**: $[\mathcal{X} \times_n M]_{(n)} = M X_{(n)}$ — unfold along mode $n$, multiply by $M$, fold back.

**Matricization (unfolding)**: mode-1 unfolding $X_{(1)} \in \mathbb{R}^{I \times JK}$ — equivalent to reshape. No information lost.

### Tucker3

Tucker3 decomposes a 3-way tensor as:
$$\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$

- $\mathcal{G} \in \mathbb{R}^{P \times Q \times R}$: **core tensor** — defines cross-talk between components
- $A \in \mathbb{R}^{I \times P}$, $B \in \mathbb{R}^{J \times Q}$, $C \in \mathbb{R}^{K \times R}$: loading matrices for each mode
- Ranks $P$, $Q$, $R$ can be **different** for each mode (key advantage)

Scalar form: $x_{ijk} \approx \sum_p\sum_q\sum_r g_{pqr}\, a_{ip}\, b_{jq}\, c_{kr}$

Matrix form: $X_{(1)} \approx A\, G_{(1)}\, (C \otimes B)^T$ (Kronecker product $\otimes$)

Tucker3 is **NOT unique**: any rotation $Q$ can be applied to $\mathcal{G}$ with compensating change in loading matrices. Good for **data compression**.

### PARAFAC (CP Decomposition)

PARAFAC decomposes as a sum of $R$ rank-one tensors (outer products):
$$\mathcal{X} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r$$

Scalar form: $x_{ijk} \approx \sum_{r=1}^R a_{ir}\, b_{jr}\, c_{kr}$

Matrix form: $X_{(1)} \approx A\,(C \odot B)^T$ (Khatri-Rao product $\odot$, NOT Kronecker)

**ALS update for $A$**: $A \leftarrow X_{(1)}(C \odot B)(C^TC * B^TB)^{-1}$ (Hadamard $*$ in denominator)

PARAFAC is a **special case of Tucker3** where the core $\mathcal{G}$ is **super-diagonal** (ones on main diagonal, zeros elsewhere). No cross-talk between components.

PARAFAC is **essentially unique** — the super-diagonal constraint prevents arbitrary rotations. PARAFAC components are **NOT nested** — changing $R$ changes all components. Good for **resolving physically interpretable additive profiles** (spectra, kinetics).

| | Tucker3 | PARAFAC |
|--|---------|---------|
| Core tensor | Full $P \times Q \times R$ | Super-diagonal $R \times R \times R$ |
| Ranks per mode | Different ($P, Q, R$) | Same ($R, R, R$) |
| Cross-talk | Yes | No |
| Unique? | NO (rotation ambiguity) | YES (essentially) |
| Use for | Compression | Physical profiles |
| Matrix product used | Kronecker $\otimes$ | Khatri-Rao $\odot$ |

### CORCONDIA (Core Consistency Diagnostic)

Used to choose $R$ for PARAFAC. Fits loading matrices, then computes the actual core $\mathcal{G}$ and compares it to the ideal super-diagonal tensor $\mathcal{I}$:

$$\text{CORCONDIA} = 100 \cdot \left(1 - \frac{\|\mathcal{I} - \mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$

- CORCONDIA $\approx 100$: core is nearly super-diagonal → $R$ is appropriate
- CORCONDIA $\approx 0$ or negative: core deviates strongly → $R$ is too large

Choose the **largest $R$ before CORCONDIA drops sharply**. CORCONDIA is specific to PARAFAC (not Tucker3).

### Split-Half Analysis (FMS)

Randomly split data in two halves along the sample mode. Fit PARAFAC independently to each half. Compute the Factor Match Score:

$$\text{FMS} = \sum_{r=1}^R \frac{a_r^T\hat{a}_r \cdot b_r^T\hat{b}_r \cdot c_r^T\hat{c}_r}{\|a_r\|\|\hat{a}_r\|\|b_r\|\|\hat{b}_r\|\|c_r\|\|\hat{c}_r\|}$$

Each factor is a product of cosine similarities between corresponding half-model loadings. FMS close to $R$ → stable solution, good choice of $R$. FMS $\ll R$ → unstable, $R$ too large. Best practice: use CORCONDIA and FMS together — they should agree.

### Key Traps W12
- Tucker3 and PARAFAC are unrelated → **WRONG**. PARAFAC is a SPECIAL CASE of Tucker3 with super-diagonal core
- Tucker3 is unique → **WRONG**. Tucker has rotation ambiguity
- PARAFAC is not unique → **WRONG**. PARAFAC is essentially unique
- Tucker uses Khatri-Rao product → **WRONG**. Tucker uses Kronecker $\otimes$; PARAFAC uses Khatri-Rao $\odot$
- PARAFAC components are nested → **WRONG**. Changing $R$ changes ALL components
- CORCONDIA close to 0 means good fit → **WRONG**. Close to 100 is good; close to 0 means $R$ too large
- CORCONDIA can be used for Tucker3 → **WRONG**. CORCONDIA is specifically for PARAFAC
- Tucker ranks must be equal → **WRONG**. Tucker ranks $P$, $Q$, $R$ can differ per mode
- PARAFAC is better for compression → **WRONG**. Tucker is better for compression; PARAFAC is better for physical profiles
- In split-half analysis, fit one model and split it → **WRONG**. Split DATA first, fit SEPARATE models to each half

---

## Q22 — CV Design for Wearables Dataset

### Dataset
**16 subjects × 3 activities × 4 seasons = 192 observations** (12 obs/subject). Features: BVP, skin temperature, HR. Target: stress/activity level.

### Task and Model (answer this first in the exam)
This is a **3-class supervised classification problem** (rest / running / social media). Extract features from the raw time series (mean HR, HRV/RMSSD, BVP amplitude, temperature slope), then apply:
- **Regularized Logistic Regression (L1/L2)** — handles correlated features; regularization essential since LOSO training folds have only 9 obs
- **LDA** — fast, interpretable, works well with small $n$
- **Random Forest** — if non-linear patterns expected

Hyperparameter tuning ($\lambda$) must use a **nested inner CV loop** — never on the full 192 observations.

### Why Standard CV Fails
Observations from the same person share physiology (resting HR, signal amplitudes). Random splits let the model "see" a test subject's data during training → learns their personal baseline → **data leakage** → EPE is optimistically biased. The IID assumption fails: observations within an individual are correlated.

### Part a) Personalized Model — predict new season for KNOWN individual

**CV: Leave-One-Season-Out** (LOSO-CV, within one subject, 4 folds)

| Fold | Train | Test |
|------|-------|------|
| 1 | Spring, Summer, Autumn (9 obs) | Winter (3 obs) |
| 2 | Winter, Summer, Autumn (9 obs) | Spring (3 obs) |
| 3 | Winter, Spring, Autumn (9 obs) | Summer (3 obs) |
| 4 | Winter, Spring, Summer (9 obs) | Autumn (3 obs) |

$$\text{EPE}_\text{pers} = E_{x,y \mid i_\text{fixed}}\!\left[\mathcal{L}(y, \hat{f}_i(x))\right]$$

Only intra-individual variation. EPE is **lower**. Limitation: only 9 training obs/fold → high-variance estimate.

### Part b) Generalized Model — predict for NEW individual

**CV: Leave-One-Individual-Out** (LOIO-CV, 16 folds)

Each fold: train on 15 subjects (180 obs), test on 1 subject (12 obs). All 12 observations from a subject must stay together in the same fold — never split a person across train/test.

$$\text{EPE}_\text{gen} = E_{i_\text{new}}\!\left[E_{x,y \mid i_\text{new}}\!\left[\mathcal{L}(y, \hat{f}(x))\right]\right]$$

Captures inter-individual generalization. EPE is **higher** — must handle between-individual variance.

### Key Comparison

| Property | Personalized (LOSO) | Generalized (LOIO) |
|----------|--------------------|--------------------|
| Folds | 4 | 16 |
| Training size | 9 obs | 180 obs |
| Test size | 3 obs | 12 obs |
| Captures | Intra-individual variation | Inter-individual variation |
| Typical EPE | Lower | Higher |
| Clinical use | Monitor known patient | Screen new patient |

**$\text{EPE}_\text{gen} > \text{EPE}_\text{pers}$ always** — generalized integrates over between-individual variance; personalized does not.

**Clinical recommendation**: Deploy generalized for new patients. Fine-tune with personalized as data accumulates (transfer learning).

### Q22 Variants (Recognition Table)

| Signal in the question | Variant | CV answer | Model answer |
|------------------------|---------|-----------|-------------|
| "predict for new patient" | Supervised, generalized | LOIO-CV | Reg. LR or LDA |
| "predict for same individual" | Supervised, personalized | LOSO-CV | Reg. LR or LDA |
| "multiple measurements per person" | IID violation → grouped CV | Group K-fold by person | Reg. LR / RF |
| "how many unique clusters" | Unsupervised | — | PCA/NMF + GMM + BIC |
| "tensor / multi-way, how many components" | PARAFAC | — | CORCONDIA + split-half FMS |
| "time series, predict next week" | Temporal leakage | Forward-chaining | Ridge / Gradient Boosting |
| "genomics, $p \gg n$" | High-dimensional | Stratified K-fold + nested | Elastic Net |

---

## APPENDIX A — Neural Network Parameter Counting

Tested in **2024 Q12** and **2025 Q12**. Always the same formula:

$$\text{params per layer} = (\text{inputs to layer} \times \text{units in layer}) + \text{units in layer (biases)}$$

**Worked examples:**

**2025: 3 → 4 → 2 → 1**
- Layer 1: $3 \times 4 + 4 = 16$
- Layer 2: $4 \times 2 + 2 = 10$
- Layer 3: $2 \times 1 + 1 = 3$
- **Total = 29** ✓

**2024: 10 → 2 → 2 → 1**
- Layer 1: $10 \times 2 + 2 = 22$
- Layer 2: $2 \times 2 + 2 = 6$
- Layer 3: $2 \times 1 + 1 = 3$
- **Total = 31** ✓

**Rule**: For any architecture $n_0 \to n_1 \to n_2 \to \cdots \to n_L$:
$$\text{total} = \sum_{\ell=1}^{L} (n_{\ell-1} \times n_\ell + n_\ell)$$

Common trap: forgetting to add biases (one per node in each layer, except the input layer).

---

## APPENDIX B — Confusion Matrix, Sensitivity, Specificity

Tested in **2022 Q13**. Know how to apply Bayes-style arithmetic to populations.

### Definitions

| Term | Formula | Meaning |
|------|---------|---------|
| Sensitivity (TPR, Recall) | $\text{TP}/(\text{TP}+\text{FN})$ | Fraction of positives correctly detected |
| Specificity (TNR) | $\text{TN}/(\text{TN}+\text{FP})$ | Fraction of negatives correctly rejected |
| False Positive Rate | $\text{FP}/(\text{TN}+\text{FP}) = 1-\text{Specificity}$ | Fraction of negatives incorrectly called positive |
| Precision (PPV) | $\text{TP}/(\text{TP}+\text{FP})$ | Fraction of positive predictions that are correct |
| Accuracy | $(\text{TP}+\text{TN})/N$ | Overall fraction correct |

### Population-Level Calculation (2022 Q13 style)

Given: 10,000 subjects. 100 have COVID (prevalence = 1%). Test: sensitivity = 99%, FPR = 2%.

$$\text{True Positives} = 100 \times 0.99 = 99$$
$$\text{False Positives} = 9900 \times 0.02 = 198$$
$$\text{Total Positives} = 99 + 198 = \mathbf{297}$$

**Template**: Always compute separately from the positive group and negative group:
- TP = (# true positives) × sensitivity
- FP = (# true negatives) × FPR

Common trap: only counting true positives (99) and forgetting false positives (198).

---

## APPENDIX C — Nested CV

Tested in **2025 Q6**, mentioned in **2024 Q7**.

### Why Nested CV?

Standard CV selects the best $\lambda$ and reports the CV error at that $\lambda$. Problem: the reported error is **optimistically biased** because you used the same data for both selection AND assessment.

### The Two Loops

**Outer loop** (assessment): $K_\text{out}$ folds. Estimates generalization error of the whole pipeline.

**Inner loop** (selection): For each outer training set, run another $K_\text{in}$-fold CV to choose the best $\lambda$.

```
For each outer fold k = 1..K_out:
    Training_outer = all data except fold k
    Test_outer     = fold k
    
    → Run inner CV on Training_outer to pick best λ_k
    → Fit model with λ_k on all of Training_outer
    → Evaluate on Test_outer
    
Report: average error over all outer folds
```

**When to use it**: Whenever you tune a hyperparameter and also want an honest estimate of how well your model will perform on new data. Without the outer loop, the reported test error is biased downward.

---

## APPENDIX D — ICA Uniqueness Indeterminacies

Tested in **2024 Q21 (open question)**. ICA is NOT fully unique — it has two fundamental indeterminacies:

1. **Permutation ambiguity**: the order of independent components can be permuted in any way ($A$ can be permuted by any permutation matrix $P$: $x = As = APP^{-1}s$).
2. **Scaling/sign ambiguity**: each component can be multiplied by any nonzero scalar (the sign and scale of each source is unidentifiable). Convention: normalize to unit variance.

These two indeterminacies are **fundamental** — they cannot be resolved without additional information. They are NOT bugs; they are properties of the model.

**What ICA CAN uniquely recover** (up to permutation and scaling): the actual independent components themselves, as long as at most one source is Gaussian.

**The Gaussian failure case**: If sources are Gaussian, any rotation of the Gaussian mixture is also Gaussian → no way to distinguish the mixing matrix $A$ from $AR$ for any orthogonal $R$. ICA is unidentifiable for Gaussian sources.

---

## APPENDIX E — Known Errors in Official Exam Solutions

Three confirmed errors. If these exact questions appear again, know the correct answer.

### 2022 Q20 — Boosting Suitable Individual Models
- **Official answer**: A (KNN high K) + E (None of the above) — contradictory, likely grid error
- **Correct answer**: **C (Any classification or regression tree)**. Boosting uses weak learners — shallow decision trees (stumps) are canonical. KNN has no natural fit into the gradient boosting framework.

### 2024 Q11 — RF Proximity Plots
- **Official answer includes C**: "Proximity plots measure closeness of variables"
- **C is WRONG**: Proximity plots measure closeness of **observations** (data points), not variables. Two observations have high proximity if they frequently end up in the same terminal node across trees.
- **Correct answers**: A (Gini VI = aggregation of Gini impurity decrease at splits for variable $j$) and D (deep trees are good because RF bias = individual tree bias).

### 2022 Q9 — Why Not Penalize the Intercept
- **Official answer appears uncertain** (multiple options marked with "(x)")
- **Most defensible answer**: **A** — "Penalizing the intercept introduces bias without any variance reduction." The intercept shifts predictions globally; penalizing it shrinks them toward zero rather than toward the true mean. D ("lower EPE if we don't penalize") is a consequence of A, not an independent reason.

---

## APPENDIX F — Recurring Exam Patterns

These patterns appear across 2022, 2024, 2025. Know them cold.

### Patterns That Appear Every Year

| Pattern | What to know |
|---------|-------------|
| Lasso λ direction | Too small λ = low bias, high variance. Too large λ = high bias, low variance. |
| Ridge vs Lasso | Ridge: closed form, no zeros, L2. Lasso: no closed form, exact zeros, L1. |
| Bias-Variance / EPE | Three terms always: $\sigma^2 + \text{Bias}^2 + \text{Var}$. $\sigma^2$ never changes. |
| CV design | Never normalize before CV. Dependent obs must stay in same fold. |
| BH vs Bonferroni | BH = FDR control, more power. Bonferroni = FWER control, more conservative. |
| LDA linearity reason | Equal covariance → quadratic terms cancel → linear boundary. |
| Boosting vs Bagging tree depth | Boosting: SHALLOW trees (stumps). Bagging/RF: DEEP trees. |
| BIC vs AIC penalty | BIC penalty = $p \log N$ (grows with N). AIC penalty = $2p$ (constant). BIC more conservative. |

### Option E (None of the Above) — Don't Reflexively Reject It

E was the correct answer in: 2022 Q16 (CORCONDIA), 2024 Q1 (supervised methods), 2024 Q13 (LARS vs CD). Never assume E is a trap answer.

### Neural Network Parameter Count — Formula

$(n_\text{in} \times n_\text{layer} + n_\text{layer})$ per layer. Always count biases. Appeared in 2024 and 2025.

### "Which Methods Work When $p \gg n$?"

Always: Elastic Net, PCA, Random Forest, SVM (dual), Ridge.
Never: OLS, standard logistic regression (without regularization), LDA/QDA (without regularization).

### Confusion Matrix / Bayes' Theorem Arithmetic

TP = (# actual positives) × sensitivity. FP = (# actual negatives) × FPR. Total positives = TP + FP. Appeared 2022 Q13.

### K-means Cannot Use AIC/BIC

K-means has no likelihood → cannot compute AIC/BIC. Use: silhouette score, gap statistic, or elbow method. GMM CAN use AIC/BIC. This is a common trap in clustering questions.

### SVM: Dual ≠ Nonlinear

The dual formulation of SVM alone does NOT make it nonlinear. Nonlinearity comes from the **kernel choice** (RBF, polynomial). A linear kernel in the dual is still a linear boundary.
