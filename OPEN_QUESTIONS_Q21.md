# CDA 02582 — Q21 CHEAT SHEET
> Q21 = 20 points. Explain / compare / derive a key algorithm in depth.
> Full model answers in `open_questions/Q21/Q21_*.md`. This file = fast review before exam.

---

## Contents

| # | Topic | Exam Status |
|---|-------|-------------|
| [A](#a--random-forest) | Random Forest | Appeared 2022 |
| [B](#b--ica) | ICA — Non-Gaussianity & Uniqueness | Appeared 2024 |
| [C](#c--lda-vs-gmm) | LDA vs GMM | Appeared 2025 |
| [D](#d--svm) | SVM — Dual & Kernel Trick | **Not yet appeared** |
| [E](#e--boosting) | Boosting — AdaBoost & Gradient Boosting | **Not yet appeared** |
| [F](#f--parafac-vs-tucker) | PARAFAC vs Tucker | Not yet appeared |
| [G](#g--pca-vs-pls-vs-cca) | PCA vs PLS vs CCA | Not yet appeared |
| [H](#h--nmf--ica--aa--pca) | NMF / ICA / AA / PCA | Not yet appeared |
| [I](#i--ridge-vs-lasso-vs-elastic-net) | Ridge vs Lasso vs Elastic Net | **Not yet appeared** |
| [J](#j--k-means-vs-hierarchical-clustering) | K-means vs Hierarchical Clustering | **Not yet appeared** |
| [K](#k--multiple-testing-bonferroni-vs-bh) | Multiple Testing — Bonferroni vs BH | Not yet appeared |
| [L](#l--neural-networks-and-backpropagation) | Neural Networks & Backpropagation | Not yet appeared |
| [M](#m--epe-decomposition--bias-variance-tradeoff) | EPE Decomposition — Bias-Variance | **High — most fundamental** |
| [N](#n--cart--decision-trees) | CART / Decision Trees | Not yet appeared as Q21 |
| [O](#o--cross-validation-and-model-selection) | Cross-Validation & 1-SE Rule | Not yet appeared as Q21 |
| [P](#p--logistic-regression-vs-lda) | Logistic Regression vs LDA | Classic comparison |
| [Q](#q--ols--gauss-markov-theorem) | OLS & Gauss-Markov Theorem | Week 1, derivation |
| [R](#r--the-bootstrap) | Bootstrap — CIs & Sampling Distributions | Week 2 |
| [S](#s--curse-of-dimensionality) | Curse of Dimensionality | Week 3, conceptual |
| [T](#t--aic--bic--model-selection-criteria) | AIC / BIC — Model Selection Criteria | Weeks 1/2, derivation |
| [U](#u--bagging-and-variance-reduction) | Bagging — Variance Formula & OOB | Week 5, standalone |
| [V](#v--cluster-validation) | Cluster Validation — Silhouette, Gap, BIC | Week 9, choosing K |
| [W](#w--sparse-pca) | Sparse PCA — PMD, interpretable loadings | Week 8 |
| [X](#x--qda--quadratic-discriminant-analysis) | QDA — quadratic boundary, per-class covariance | LDA extension |
| [Y](#y--k-medoids-pam-vs-k-means) | K-medoids vs K-means — robustness, any distance | Week 9 |
| [Z](#z--gaussian-mixture-models-gmm) | GMM — EM derivation, soft clustering, BIC for K | Week 9, deep dive |
| [AA](#aa--split-half-analysis--fms-for-parafac) | Split-Half FMS — PARAFAC reproducibility validation | Week 12 |
| [AB](#ab--principal-component-regression-pcr) | PCR vs PLS — SVD view, Ridge comparison, weakness | Week 8 |
| [AC](#ac--elastic-net-regression) | Elastic Net — grouped selection, coordinate descent update | Week 3 |
| [AD](#ad--gradient-boosting) | Gradient Boosting — pseudo-residuals, shrinkage, AdaBoost link | Week 6 |
| [AE](#ae--regularized-discriminant-analysis-rda) | RDA — shrinkage between LDA and QDA | Week 4 |
| [AF](#af--canonical-correlation-analysis-cca) | CCA — maximize cross-block correlation | Week 8 |
| [AG](#ag--k-nearest-neighbors-knn) | KNN — local averaging, K tuning, high-dimensional limits | Week 2 |

---

## Past Exams

| Year | Q21 Topic |
|------|-----------|
| 2022 | Random Forest algorithm |
| 2024 | ICA uniqueness and distributions |
| 2025 | LDA vs GMM comparison |

**Likely for 2026**: SVM, Boosting, Ridge/Lasso, K-means/Hierarchical, Multiple Testing, Neural Networks, PARAFAC/Tucker, PCA/PLS/CCA

---

## WRITING TEMPLATE (every Q21)

1. **State the model** — formula, one sentence
2. **Explain mechanism** — why does each step work?
3. **Key properties** — bias/variance, uniqueness, complexity
4. **Compare to alternatives** — name the distinguishing property
5. **Limitations** — when does it fail?

**Marks come from**: correct objective function · explaining WHY · comparison · formula · edge case behavior

**Common mistakes to avoid**:
- "It minimizes the error" — say WHICH loss and HOW
- Listing steps without explaining what each achieves
- "More complex = better" — always frame with bias-variance tradeoff
- Forgetting assumptions (equal covariance for LDA, non-Gaussian for ICA)
- Confusing variance reduction (bagging) with bias reduction (boosting)

---

## A — RANDOM FOREST

**Algorithm**: Bagging + random feature subsampling on deep unpruned trees.

**Step 1 — Bootstrap**: Draw $Z^{*b}$ size $N$ with replacement. ~36.8% OOB.

**Step 2 — Random features**: At each node, consider only $m=\lfloor\sqrt{p}\rfloor$ features (classification). Decorrelates trees.

**Step 3 — Deep trees**: Grow unpruned. Low bias, high variance individually — averaging removes variance.

**Step 4 — Aggregate**: $\hat{f}_{RF}(x) = \frac{1}{B}\sum_b T_b(x)$ (regression) / majority vote (classification).

**Variance formula**:
$$\text{Var}(\text{avg}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2 \xrightarrow{B\to\infty} \rho\sigma^2$$

Random features reduce $\rho$ (decorrelation) → lower variance floor than pure bagging.

**OOB error**: predict each $x_i$ using only trees that didn't bootstrap it → ≈ LOO-CV, free.

**Variable importance**: OOB permutation (permute feature $j$, measure accuracy drop).

**Key**: RF does NOT increase bias vs single tree. RF does NOT overfit as $B\to\infty$.

---

## B — ICA

**Model**: $x = As$, $x$ observed, $A$ unknown mixing matrix, $s$ unknown independent sources. Goal: find $W \approx A^{-1}$.

**Why non-Gaussianity required**: CLT says mixtures become MORE Gaussian. Least-Gaussian direction = original source. Gaussian sources: all rotations equally Gaussian → **completely unidentifiable**.

**Measuring non-Gaussianity**:
- Kurtosis: $\kappa_4 = \mu_4/\sigma^4 - 3$ (Gaussian=0; super-Gaussian>0; sub-Gaussian<0). Sensitive to outliers.
- Negentropy: $J(y) = H(y_\text{Gauss}) - H(y) \geq 0$. Zero iff Gaussian. More robust.

**FastICA**: whiten first ($E[\tilde{x}\tilde{x}^T]=I$), then fixed-point:
$$w \leftarrow E[\tilde{x}\,g(w^T\tilde{x})] - E[g'(w^T\tilde{x})]w, \quad w \leftarrow w/\|w\|$$

**Uniqueness**: unique up to (1) permutation of components, (2) sign and scale per component ($s$ and $-s$ produce same distribution; variance absorbed into $A$). NOT unique beyond this.

**ICA vs PCA**: PCA = uncorrelated (2nd-order). ICA = statistically independent (all orders). Independence is strictly stronger. PCA works on Gaussians; ICA does not.

---

## C — LDA vs GMM

**Shared**: both assume Gaussian class-conditionals.

| Property | LDA | GMM |
|----------|-----|-----|
| Supervision | Supervised | Unsupervised |
| Covariance | Shared $\Sigma$ | Per-component $\Sigma_k$ |
| Fitting | Closed-form MLE | EM (iterative, local max) |
| Boundary | Linear | Nonlinear |
| Latent vars | None | $Z_i$ = unobserved cluster |

**Why LDA boundary is linear**: log-ratio $\log[P(C_k|x)/P(C_{k'}|x)] = \log(\pi_k/\pi_{k'}) + x^T\Sigma^{-1}(\mu_k-\mu_{k'}) - \frac{1}{2}(\mu_k^T\Sigma^{-1}\mu_k - \mu_{k'}^T\Sigma^{-1}\mu_{k'})$ — the quadratic $-\frac{1}{2}x^T\Sigma^{-1}x$ term appears in both numerator and denominator and **cancels** when $\Sigma$ is shared → linear in $x$. (The prior term $\log(\pi_k/\pi_{k'})$ only vanishes if classes are equally frequent.)

With unequal $\Sigma_k$ (QDA): quadratic terms don't cancel → quadratic boundary.

**GMM EM**:
- E-step: $\gamma_{ij} = \pi_j\mathcal{N}(x_i|\mu_j,\Sigma_j)/\sum_{j'}\pi_{j'}\mathcal{N}(x_i|\mu_{j'},\Sigma_{j'})$
- M-step: update $\mu_j, \Sigma_j, \pi_j$ using $\gamma_{ij}$ as soft weights

**GMM → K-means**: spherical equal $\Sigma_j=\sigma^2 I$ + hard assignments → exactly K-means.

**Choosing K for GMM**: BIC = $-2\ell + p_\theta\log N$. Larger penalty than AIC ($2p_\theta$).

---

## D — SVM

**Setup**: maximize margin $2/\|\beta\|$ between classes $y_i\in\{-1,+1\}$.

**Primal**:
$$\min_{\beta,\beta_0}\frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i^T\beta+\beta_0)\geq1 \;\forall i$$

**Dual** (after Lagrangian + KKT):
$$\max_\alpha \sum_i\alpha_i - \frac{1}{2}\sum_{ij}\alpha_i\alpha_jy_iy_j K(x_i,x_j) \quad \text{s.t.} \quad 0\leq\alpha_i\leq C,\;\sum_i\alpha_iy_i=0$$

**Support vectors**: points with $\alpha_i>0$; solution $\beta=\sum_i\alpha_iy_ix_i$ depends only on these.

**Soft margin**: slack $\xi_i\geq0$, $C\sum_i\xi_i$ penalty. Large $C$ = hard boundary = high variance.

**Kernel trick**: dual uses only $\langle x_i,x_j\rangle$ → replace with $K(x_i,x_j) = \phi(x_i)^T\phi(x_j)$. Compute infinite-dimensional dot product without $\phi$ explicitly.
- RBF: $K(x,x')=\exp(-\gamma\|x-x'\|^2)$ → infinite-dimensional space.
- Prediction: $\hat{y}=\text{sign}(\sum_i\alpha_iy_iK(x_i,x)+\beta_0)$

**Key**: SVM is geometric (no probabilities). $N$ dual parameters, not $p$ → works when $p\gg n$.

---

## E — BOOSTING

**Core idea**: sequential ensemble, each learner fits errors of current ensemble. Reduces **bias**.

**AdaBoost** ($y_i\in\{-1,+1\}$):

Initialize $w_i=1/N$. For $m=1,\ldots,M$:
1. Fit $G_m(x)$ with weights $w_i$
2. $\text{err}_m = \sum_i w_i\mathbf{I}(y_i\neq G_m(x_i))/\sum_iw_i$
3. $\alpha_m = \log\frac{1-\text{err}_m}{\text{err}_m}$
4. $w_i \leftarrow w_i\exp[\alpha_m\cdot\mathbf{I}(y_i\neq G_m(x_i))]$, normalize

Final: $G(x) = \text{sign}[\sum_m\alpha_mG_m(x)]$

**Theoretical connection**: AdaBoost = forward stagewise additive model minimizing exponential loss $\exp(-yF(x))$.

**Weight update = exponential loss**: misclassified point weight $\times e^{\alpha_m}$ (grows to ~20 for good classifier). Plain misclassification loss: flat +1. Exponential: grows fast → much more aggressive upweighting.

**Gradient Boosting** (general):
- Pseudo-residuals: $r_{im} = -[\partial L(y_i,F(x_i))/\partial F(x_i)]$
- For squared error: $r_{im} = y_i - F_{m-1}(x_i)$ (ordinary residuals)
- Fit tree to $r_{im}$; update $F_m = F_{m-1} + \nu h_m$

**Why stumps**: boosting reduces bias → use high-bias (simple) base learners. Bagging reduces variance → use low-bias (deep) trees.

---

## F — PARAFAC vs TUCKER

**Tucker3**: $\mathcal{X}\approx\mathcal{G}\times_1A\times_2B\times_3C$, core $\mathcal{G}\in\mathbb{R}^{P\times Q\times R}$ encodes all cross-mode interactions. Ranks $(P,Q,R)$ per mode. NOT unique (rotation ambiguity). Mode-1: $X_{(1)}\approx AG_{(1)}(C\otimes B)^T$.

**PARAFAC**: $\mathcal{X}\approx\sum_{r=1}^R a_r\circ b_r\circ c_r$. Special case of Tucker3 with super-diagonal core (no cross-talk between components). Single rank $R$ all modes. **Essentially unique** (up to permutation/scale). Mode-1: $X_{(1)}\approx A(C\odot B)^T$.

**CORCONDIA**:
$$\text{CORCONDIA} = 100\left(1 - \frac{\|\mathcal{I}-\tilde{\mathcal{G}}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$
$\approx100$: good $R$. $<50$ or negative: $R$ too large. Drop CORCONDIA sharply before this $R$.

**Key**: Tucker = compression (different ranks per mode). PARAFAC = physical interpretation (unique components). Kronecker $\otimes$ vs Khatri-Rao $\odot$.

---

## G — PCA vs PLS vs CCA

| Method | Supervised? | Objective | Works $p>N$? |
|--------|------------|-----------|-------------|
| PCA | No | $\max\text{Var}(Xv)$ | Yes |
| PLS | Yes | $\max\text{Cov}(Xu,Yv)$ | Yes |
| CCA | Two-sided | $\max\text{Corr}(Xu,Yv)$ | No (needs $\Sigma^{-1}$) |

**PCA**: eigenvectors of $\Sigma_{XX}$. Nested components. Ignores $y$ → may miss predictive directions.

**PLS**: maximizes covariance = variance × correlation. Balance of both. With $M=p$: reduces to OLS.

**CCA**: maximizes correlation only. Ignores internal variance. Requires $\Sigma_{XX}^{-1}$, $\Sigma_{YY}^{-1}$ → fails if $p>N$. Fix: regularized CCA ($+\lambda I$) or sparse CCA (PMD).

---

## H — NMF / ICA / AA / PCA

| Method | Constraint | Unique? | Prototypes at |
|--------|-----------|---------|---------------|
| PCA | Orthogonal | Yes (±sign) | Interior (mean directions) |
| NMF | $W\geq0$, $H\geq0$ | No ($Q$-ambiguity) | Interior (additive parts) |
| ICA | Independent, non-Gaussian | Yes (±perm/sign/scale) | Non-Gaussian directions |
| AA | Convex hull + convex mix | Partially | Boundary (extremes) |
| K-means | Hard cluster assignments | No (local optima) | Interior (centroids) |

**NMF**: parts-based (no cancellation). Multiplicative updates. Fitted by alternating updates.

**ICA**: see Candidate B. Non-Gaussianity = identifiability. Whitening first, then rotation.

**AA**: archetypes $= XS$ (convex combinations of data). Lie on convex hull. Doubly-convex: $S$ and $H$ both have rows summing to 1. Best for extreme phenotypes.

**Sparse coding**: $\min\|X-WH\|_F^2 + \lambda\|H\|_1$. Equivalent to ICA with Laplacian (super-Gaussian) prior.

---

## I — RIDGE vs LASSO vs ELASTIC NET

All solve: $\min_\beta \|y-X\beta\|^2 + \lambda\cdot\text{Penalty}(\beta)$

| Method | Penalty | Closed-form? | Exact zeros? | Works $p>n$? |
|--------|---------|-------------|-------------|-------------|
| OLS | None | Yes | No | No |
| Ridge | $\|\beta\|_2^2$ | **Yes**: $(X^TX+\lambda I)^{-1}X^Ty$ | No | Yes |
| Lasso | $\|\beta\|_1$ | No (coordinate descent / LARS) | **Yes** | Selects $\leq n$ |
| Elastic Net | $\alpha\|\beta\|_1+(1-\alpha)\|\beta\|_2^2$ | No | Yes | Yes |

**Why Ridge never zeros**: $L_2$ penalty has gradient $2\lambda\beta_j$ → smooth, asymptotes to 0. Geometric: $L_2$ ball is a sphere — solution almost never on an axis.

**Why Lasso zeros**: $L_1$ subgradient $= \lambda\cdot\text{sign}(\beta_j)$. If $|X_j^Tr| < \lambda$, coordinate set to 0. Geometric: $L_1$ diamond has corners on axes — solution likely to land there.

**Lasso solution path** (LARS): as $\lambda\downarrow 0$, coefficients enter one at a time. At $\lambda_\text{max}=\max_j|X_j^Ty|/N$, all $\hat{\beta}_j=0$.

**Elastic Net**: groups correlated predictors (keeps/drops together). Best when $p\gg n$ with correlated features (genomics).

**Bias-variance tradeoff**: increasing $\lambda$ → more bias, less variance. Optimal $\lambda$ via CV + 1-SE rule (pick most regularized model within 1 SE of minimum CV error).

---

## J — K-MEANS vs HIERARCHICAL CLUSTERING

**K-means** (Lloyd's algorithm):
1. Assign each $x_i$ to nearest centroid: $C(i)=\arg\min_k\|x_i-\mu_k\|^2$
2. Update centroids: $\mu_k=\frac{1}{|C_k|}\sum_{i:C(i)=k}x_i$
3. Repeat until stable

Converges to **local minimum** (not global). Run multiple times with different inits. K-means++ init: place centroids proportional to $d(x_i,\text{nearest centroid})^2$ → better coverage.

**Connection to GMM**: K-means = GMM with $\Sigma_k=\sigma^2I$ (spherical equal) + hard assignments.

**Hierarchical (agglomerative)**: start with $N$ singletons, merge two closest clusters at each step. Output = dendrogram — choose $K$ after by cutting at desired height. Deterministic. $O(N^2\log N)$.

**Linkage methods**:
- Single: $\min$ distance (chaining effect)
- Complete: $\max$ distance (compact clusters)
- Ward: minimize increase in within-cluster variance (equivalent to K-means criterion)

**Choosing K**: Elbow (WCSS vs $K$), Silhouette $s(i)=(b(i)-a(i))/\max(a,b)\in[-1,1]$, Gap statistic.

| | K-means | Hierarchical |
|--|---------|-------------|
| $K$ upfront? | Yes | No |
| Reproducible? | No (random init) | Yes |
| Scalability | Fast $O(NKd)$ | Slow $O(N^2\log N)$ |

---

## K — MULTIPLE TESTING: BONFERRONI vs BH

**Problem**: testing $m$ hypotheses — probability of $\geq 1$ false positive = $1-(1-\alpha)^m\to1$.

**Two error rates**:
- **FWER** (Family-Wise Error Rate): $P(\geq1$ false positive$)\leq\alpha$ — very strict
- **FDR** (False Discovery Rate): $E[\text{FP}/R]\leq\alpha$ where $R$ = total rejections — controls proportion

**Bonferroni** (controls FWER): reject $H_i$ if $p_i\leq\alpha/m$.
- Works under any dependence (union bound)
- Very low power for large $m$ (threshold shrinks to $5\times10^{-6}$ for $m=10000$)

**Benjamini-Hochberg** (controls FDR):
1. Sort: $p_{(1)}\leq\cdots\leq p_{(m)}$
2. Find largest $k^*$ such that $p_{(k^*)}\leq k^*\alpha/m$
3. Reject all $H_{(1)},\ldots,H_{(k^*)}$

BH controls $\text{FDR}\leq(m_0/m)\alpha\leq\alpha$ under independence ($m_0$ = true nulls).

**Key**: Bonferroni = "no false positives allowed." BH = "5% of discoveries can be wrong." BH is much more powerful for large $m$. Use Bonferroni for critical decisions (small $m$); BH for exploratory studies (large $m$, e.g., genomics).

---

## L — NEURAL NETWORKS AND BACKPROPAGATION

**MLP forward pass**: $z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$, $a^{(l)}=g(z^{(l)})$, with $a^{(0)}=x$, $\hat{y}=a^{(L)}$.

**Activation functions**:
- Sigmoid: $\sigma(z)=1/(1+e^{-z})$, derivative $\sigma(1-\sigma)$. Vanishing gradient problem for large $|z|$.
- ReLU: $\max(0,z)$, derivative $\mathbf{I}(z>0)$. No vanishing gradient → preferred in deep networks.
- Softmax (output): $e^{z_k}/\sum_j e^{z_j}$. Multi-class probabilities.

**Backpropagation** (chain rule applied layer by layer):

Output error: $\delta^{(L)} = \partial L/\partial a^{(L)} \odot g'(z^{(L)})$

Backpropagate: $\delta^{(l)} = [(W^{(l+1)})^T\delta^{(l+1)}]\odot g'(z^{(l)})$

Gradients: $\partial L/\partial W^{(l)} = \delta^{(l)}(a^{(l-1)})^T$

**Key**: store activations during forward pass → backward pass costs same as one forward pass. Without storage: $O(p\times\text{forward})$.

**Regularization**: $L_2$ weight decay · Dropout (zero units with prob $p$, scale by $(1-p)$ at test) · Early stopping.

**Vanishing gradient**: sigmoid/tanh gradients $<1$ → shrink exponentially through layers. Fix: ReLU, batch norm, residual connections.

---

## M — EPE DECOMPOSITION / BIAS-VARIANCE TRADEOFF

$$\text{EPE}(x_0) = \underbrace{\sigma^2}_{\text{irreducible}} + \underbrace{\text{Bias}^2[\hat{f}(x_0)]}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}(x_0)]}_{\text{sensitivity to data}}$$

**Derivation sketch**: write $y_0-\hat{f}(x_0) = \varepsilon + (f-E[\hat{f}]) + (E[\hat{f}]-\hat{f})$. Square and take expectation. Three cross-terms vanish because:
- $E[\varepsilon]=0$ (noise is zero-mean)
- $\varepsilon$ independent of $\hat{f}$ (test noise independent of training)
- $E[E[\hat{f}]-\hat{f}]=0$ (by definition of expectation)

**Bias**: error from wrong assumptions (over-regularization, wrong model class). High-bias: ridge large $\lambda$, shallow trees, KNN large $K$.

**Variance**: sensitivity to training data (too complex model). High-variance: deep unpruned trees, KNN $K=1$, unregularized OLS with many predictors.

**Bagging reduces variance** (not bias): $\text{Var}(\text{avg})=\rho\sigma^2+(1-\rho)\sigma^2/B$. RF reduces $\rho$ via random features → lower floor.

**Training error vs EPE**: training error underestimates EPE by $\approx 2p\sigma^2/N$ (optimism). Basis for AIC penalty.

---

## N — CART / DECISION TREES

**Idea**: recursively partition feature space into axis-aligned regions. Each leaf predicts:
- Regression: mean of responses in region
- Classification: majority class

**Growing**: at each node, find split $(j,s)$ minimizing child impurity:
- Regression: total RSS = $\sum_{R_1}\!(y_i-\bar{y}_{R_1})^2+\sum_{R_2}\!(y_i-\bar{y}_{R_2})^2$
- Classification — Gini: $\sum_k\hat{p}_{mk}(1-\hat{p}_{mk})$ (preferred for splitting)
- Classification — Entropy: $-\sum_k\hat{p}_{mk}\log\hat{p}_{mk}$
- Misclassification: $1-\max_k\hat{p}_{mk}$ (preferred for pruning)

**Cost-complexity pruning**: grow full tree, then prune by minimizing $C_\alpha(T)=\sum_m N_mQ_m+\alpha|T|$. Produces nested sequence of trees; choose $\alpha$ by CV.

**Key properties**: scale invariant · handles mixed types · high variance (small data change → different tree) · that instability is WHY bagging works so well on trees.

---

## O — CROSS-VALIDATION AND MODEL SELECTION

**K-fold CV**: partition into $K$ folds, train on $K-1$, test on held-out fold, average. Standard: 10-fold.

**1-SE rule**:
1. Find $\lambda^*=\arg\min\text{CV}(\lambda)$
2. Threshold $=\text{CV}(\lambda^*)+1\cdot\text{SE}(\lambda^*)$
3. Pick largest $\lambda$ (simplest model) below threshold
Rationale: models within 1 SE are statistically indistinguishable — prefer simpler.

**Nested CV** (for hyperparameter selection + evaluation):
- Outer loop: $K_\text{out}$ folds → unbiased EPE estimate
- Inner loop: $K_\text{in}$ folds within each outer training set → select $\lambda^*$
- Never let test fold influence $\lambda$ selection → avoids optimistic bias

**IID requirement**: observations must be exchangeable. Violated by repeated measures (use subject-level splits), time series (use temporal splits), spatial data (use block splits).

**AIC vs CV**: AIC = $-2\ell+2p$ (closed-form, requires Gaussian errors). BIC = $-2\ell+p\log N$ (stronger penalty, consistent). CV is model-agnostic but costs $K$ refits.

---

## P — LOGISTIC REGRESSION vs LDA

Both produce **linear decision boundaries** — from different assumptions.

| Property | Logistic Regression | LDA |
|----------|--------------------|----|
| What's modeled | $P(C_k\|x)$ directly | $P(x\|C_k)$ + Bayes rule |
| Type | Discriminative | Generative |
| Distributional assumption | None on $x$ | $x\|C_k\sim\mathcal{N}(\mu_k,\Sigma)$ |
| Fitting | IRLS (iterative, concave log-likelihood) | Closed-form MLE |
| Gaussian assumption holds | Less efficient | More efficient |
| Assumption violated | More robust | Less robust |

**Logistic regression model**: $\log[P(C_1|x)/P(C_0|x)] = \beta_0+x^T\beta$

**Fitting**: maximize $\ell(\beta)=\sum_i[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)]$ via Newton-Raphson (IRLS). Log-likelihood is concave → unique global maximum.

**Regularized logistic**: L1 logistic → sparse (variable selection). L2 logistic → shrinks all coefficients.

**Complete separation**: if classes are linearly separable, $\hat{\beta}\to\infty$. Fix: add L2 penalty.

---

## Q — OLS & GAUSS-MARKOV THEOREM

**OLS solution**: $\hat{\beta}_\text{OLS} = (X^TX)^{-1}X^Ty$

**Unbiasedness proof**: $\hat{\beta} = \beta + (X^TX)^{-1}X^T\varepsilon \Rightarrow E[\hat{\beta}|X] = \beta + (X^TX)^{-1}X^TE[\varepsilon|X] = \beta$. Requires: (1) $E[y|X]=X\beta$ (correct spec), (2) $E[\varepsilon|X]=0$ (exogeneity).

**Variance**: $\text{Var}(\hat{\beta}|X) = \sigma^2(X^TX)^{-1}$

**Gauss-Markov**: OLS is **BLUE** (Best Linear Unbiased Estimator) under correct spec + exogeneity + homoscedasticity + full rank. "Best" = minimum variance among all linear unbiased estimators.

**Why Ridge can beat OLS**: Gauss-Markov restricts to unbiased estimators. Ridge introduces bias → escapes the restriction → can lower EPE = Bias$^2$ + Var. Ridge bias: $-\lambda(X^TX+\lambda I)^{-1}\beta$. Ridge variance: smaller than $\sigma^2(X^TX)^{-1}$.

**When $X^TX$ singular**: $p>N$ or multicollinearity → no unique OLS solution. Ridge adds $\lambda I$ → always invertible.

---

## R — THE BOOTSTRAP

**Algorithm**: draw $B$ samples of size $N$ **with replacement** from data. Each sample contains ~63.2% unique observations (~36.8% OOB).

**Estimate variance**: $\widehat{\text{Var}}(\hat{\theta}) = \frac{1}{B-1}\sum_b(\hat{\theta}^{*b}-\bar{\theta}^*)^2$

**Confidence intervals**:
- Percentile: $[\hat{\theta}^*_{(0.025)}, \hat{\theta}^*_{(0.975)}]$
- Basic (pivot): $[2\hat{\theta}-\hat{\theta}^*_{(0.975)}, 2\hat{\theta}-\hat{\theta}^*_{(0.025)}]$

**Why 63.2%**: $P(\text{obs } i \text{ not selected}) = (1-1/N)^N \to e^{-1} \approx 0.368$.

**Bootstrap vs CV**:
- Bootstrap: WITH replacement, estimates variability of $\hat{\theta}$, naive error estimate is optimistic (training overlaps test)
- CV: WITHOUT replacement, disjoint folds, estimates EPE directly, nearly unbiased

**.632 estimator**: $\hat{\text{Err}}^{.632} = 0.368\cdot\overline{\text{err}} + 0.632\cdot\hat{\text{Err}}^1$ — corrects bootstrap optimism by weighting training error (0.368) and OOB error (0.632).

**RF connection**: OOB observations in Random Forest are exactly the bootstrap's "left-out" set — free validation without extra computation.

---

## S — CURSE OF DIMENSIONALITY

**Volume in shell**: in $p$ dimensions, fraction of unit sphere volume within $\epsilon$ of surface $= 1-(1-\epsilon)^p \to 1$. Nearly all points concentrate near the boundary.

**Neighborhood size**: to capture fraction $r$ of data in $p$ dimensions, edge length $l = r^{1/p}$. For $r=0.01$, $p=10$: $l\approx 0.63$ — no longer local.

**Distances converge**: $(d_\text{max}-d_\text{min})/d_\text{min} \to 0$ as $p\to\infty$. All points equidistant → nearest-neighbor queries lose meaning.

**Consequences**: KNN becomes global mean (no locality). Density estimation needs exponentially more data. OLS breaks ($p>N$). All distance-based methods degrade.

**Blessing of dimensionality** (Donoho 2000): high dimensions can help when data has low intrinsic dimensionality (manifold hypothesis) or sparsity structure:
- Lasso recovers sparse signals with $N\sim s\log p$ (far less than $p$)
- Classes often linearly separable in high $p$ (SVM benefits)
- Johnson-Lindenstrauss: $N$ points projected to $k\sim\log N$ dims preserve pairwise distances

**Key insight**: the curse is real for dense/global methods (KNN, KDE). Structured methods (sparse regression, PCA, SVM) exploit the blessing.

---

## T — AIC / BIC / MODEL SELECTION CRITERIA

**Problem**: training error underestimates EPE — penalize for complexity.

$$\text{AIC} = -2\ell(\hat{\theta}) + 2p \qquad \text{BIC} = -2\ell(\hat{\theta}) + p\log N$$

**AIC origin**: minimizes expected KL divergence from true distribution. Asymptotically equivalent to LOO-CV. Bias correction for using training log-likelihood: $+2p$ (one parameter adds bias of 1 to log-likelihood estimate).

**BIC origin**: Bayesian marginal likelihood via Laplace approximation. The $p\log N$ penalty comes from integrating over parameter space.

**Key difference**: BIC penalty grows with $N$ → selects sparser models for large datasets. For $N>7$: $\log N > 2$ so BIC is always stricter.

| | AIC | BIC |
|--|-----|-----|
| Penalty | $2p$ (fixed) | $p\log N$ (grows) |
| Goal | Best predictive model | Identify true model |
| Consistent? | No (over-selects) | Yes (as $N\to\infty$) |
| Use when | Prediction, many candidates | Model identification |

**Mallow's $C_p$**: $\text{RSS}_p/\hat{\sigma}^2 - N + 2p$ — equivalent to AIC for linear Gaussian models.

**AIC vs CV**: AIC = one fit, assumes Gaussian errors. CV = $K$ fits, model-agnostic. Use CV when: non-Gaussian loss, complex preprocessing, nonparametric models.

---

## U — BAGGING AND VARIANCE REDUCTION

**Algorithm**: draw $B$ bootstrap samples, fit one model per sample, average predictions.

**Variance formula** for $B$ models with individual variance $\sigma^2$, pairwise correlation $\rho$:
$$\text{Var}(\text{avg}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B\to\infty$: variance $\to \rho\sigma^2$ (irreducible floor from shared bootstrap structure).

**Bagging reduces variance, NOT bias**: $E[\hat{f}_\text{bag}] = E[\hat{f}^{*b}]$ (same as one model). Helps only when base learner has high variance — deep trees, low-K KNN. Does NOT help linear regression (already low variance) or stumps (high bias dominates).

**Best base learners for bagging**: deep unpruned trees (canonical), low-$K$ KNN.

**OOB error**: predict each $x_i$ using only models that didn't include it in bootstrap $\approx$ LOO-CV, free.

**Bagging vs RF**: RF additionally subsamples $m=\lfloor\sqrt{p}\rfloor$ features per split → decorrelates trees → reduces $\rho$ → lower variance floor.

**Bagging vs Boosting**: bagging = parallel, reduces variance. Boosting = sequential, reduces bias. Use deep trees for bagging; use stumps for boosting.

---

## V — CLUSTER VALIDATION: CHOOSING K

**Elbow method**: plot WCSS vs $K$. Choose $K$ at the "elbow" — diminishing returns. Subjective; often ambiguous.

**Silhouette score**: for each point $i$ in cluster $k$:
$$s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))} \in [-1,1]$$
where $a(i)$ = mean distance to own cluster, $b(i)$ = mean distance to nearest other cluster.
- $s\approx1$: well-clustered. $s\approx0$: on boundary. $s<0$: misassigned.
- Plot mean $\bar{s}(K)$ vs $K$; pick maximum. More principled than elbow.

**Gap statistic**: compare $\log\text{WCSS}(K)$ to expected $\log\text{WCSS}$ under uniform random data. Choose smallest $K$ where $\text{Gap}(K)\geq\text{Gap}(K+1)-s_{K+1}$ (1-SE rule analog). Most principled; computationally expensive.

**BIC for GMM**: $\text{BIC}(K) = -2\ell_K + p_K\log N$. Choose $K$ at minimum. Penalizes extra components. Equivalent to formal model selection.

| Method | Measures | Objective? | Cost |
|--------|---------|-----------|------|
| Elbow | Cohesion only | No | Low |
| Silhouette | Cohesion + separation | Partially | Medium |
| Gap statistic | vs random null | Yes | High |
| BIC (GMM) | Probabilistic fit | Yes | Medium |

---

## W — SPARSE PCA

**Problem with standard PCA**: every variable loads onto every component — not interpretable when $p$ is large (e.g., 10,000 genes).

**Sparse PCA**: add $L_1$ penalty to loadings → each PC depends on only a few variables.

**Regression formulation** (Zou et al.):
$$\min_{A,B}\|X-XBA^T\|_F^2 + \lambda\|B\|_1 \quad \text{s.t.} \quad A^TA=I_K$$
Alternating updates: fix $A$ → Lasso for $B$; fix $B$ → SVD for $A$.

**PMD framework** (Witten et al.): $\max_{u,v} u^TXv$ with $L_1$ constraints on $u$ and/or $v$. Solved by soft-threshold iteration. Generalizes to sparse CCA (both $u$ and $v$ penalized).

| | PCA | Sparse PCA |
|--|-----|-----------|
| Loadings | Dense | Sparse (few vars per PC) |
| Variance explained | Maximum | Reduced |
| Interpretable? | No | Yes |

**Key**: same sparsity mechanism as Lasso — $L_1$ penalty sets small loadings to exactly zero. Use when scientific interpretation of which variables drive each component matters.

---

## X — QDA (QUADRATIC DISCRIMINANT ANALYSIS)

**Model**: Gaussian class-conditionals with **per-class** covariance $\Sigma_k$ (unlike LDA's shared $\Sigma$).

**Why quadratic boundary**: log-ratio contains $-\frac{1}{2}x^T(\Sigma_k^{-1}-\Sigma_{k'}^{-1})x$. With unequal $\Sigma_k$: this term is nonzero → **quadratic in $x$** → curved decision boundary.

With equal $\Sigma_k=\Sigma$: $\Sigma_k^{-1}-\Sigma_{k'}^{-1}=0$ → term vanishes → LDA.

**Fitting**: closed-form — compute $\hat{\mu}_k$, $\hat{\Sigma}_k$, $\hat{\pi}_k$ separately per class.

**Parameters**: LDA = $Kp + p(p+1)/2$. QDA = $Kp + Kp(p+1)/2$ (K times more covariance params).

**Bias-variance**: LDA has more bias (equal covariance), lower variance. QDA has less bias, higher variance. Use QDA only when $N_k \gg p$ (enough data per class to estimate $\hat{\Sigma}_k$ reliably).

**Regularized DA (RDA)**: $\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k + (1-\alpha)\hat{\Sigma}$. Interpolates between QDA ($\alpha=1$) and LDA ($\alpha=0$). Choose $\alpha$ by CV.

---

## Y — K-MEDOIDS (PAM) vs K-MEANS

**K-medoids**: cluster centers = actual training observations (medoids), not means. Medoid of cluster $k$: $m_k = \arg\min_{x_j\in C_k}\sum_{x_i\in C_k}d(x_i,x_j)$.

**Objective**: minimize total dissimilarity (not squared Euclidean): $\min\sum_k\sum_{x_i\in C_k}d(x_i,m_k)$.

**Why more robust to outliers**: medoid is the most central real point — a single extreme outlier has large total distance to others, so it is never chosen as medoid. K-means centroid is pulled by the outlier's coordinates.

**Works with any dissimilarity**: edit distance, Hamming, DTW, precomputed matrix. K-means requires Euclidean (to compute means).

**Cost**: PAM is $O(K(N-K)^2)$ per iteration — slow for large $N$. K-means is $O(NKd)$ — fast.

| | K-means | K-medoids |
|--|---------|-----------|
| Centers | Means (any point) | Actual data points |
| Distance | Euclidean only | Any dissimilarity |
| Outlier robust? | No | Yes |
| Speed | Fast | Slow |

K-means = K-medoids with unrestricted centers + squared Euclidean distance (mean minimizes squared distance).

---

## Z — GAUSSIAN MIXTURE MODELS (GMM)

**Model**: $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x;\mu_k,\Sigma_k)$. Latent assignment $Z_i\sim\text{Categorical}(\pi)$; $x_i|Z_i=k\sim\mathcal{N}(\mu_k,\Sigma_k)$.

**Why not direct MLE**: $\ell(\theta)=\sum_i\log[\sum_k\pi_k\mathcal{N}]$ — log of sum has no closed form. EM handles unknown assignments.

**E-step** (soft assignments):
$$\gamma_{ik} = \frac{\pi_k\mathcal{N}(x_i;\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_i;\mu_j,\Sigma_j)}$$

**M-step** (weighted MLE): $N_k=\sum_i\gamma_{ik}$; $\mu_k\leftarrow\frac{1}{N_k}\sum_i\gamma_{ik}x_i$; $\Sigma_k\leftarrow\frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T$; $\pi_k\leftarrow N_k/N$.

**Convergence**: log-likelihood increases every iteration → local max. Run 10-20 times with different K-means++ init.

**GMM → K-means**: set $\Sigma_k=\sigma^2I$, $\sigma\to0$ → responsibilities harden to 0/1, M-step becomes centroid update.

**BIC for K**: $\text{BIC}(K)=-2\ell_K+p_K\log N$; $p_K=(K-1)+Kp+Kp(p+1)/2$. Choose $K$ at BIC minimum.

**Degenerate solution**: $\Sigma_k\to0$ gives infinite likelihood. Fix: add $\epsilon I$ regularization.

---

## AA — SPLIT-HALF ANALYSIS & FMS FOR PARAFAC

**Why validate**: PARAFAC ALS may find local minima; rank $R$ unknown; overfitting if $R$ too large.

**Split-half procedure**: split $I$ samples into two halves $\mathcal{X}^{(1)},\mathcal{X}^{(2)}$. Fit PARAFAC rank $R$ to each independently → $(A^{(1)},B^{(1)},C^{(1)})$ and $(A^{(2)},B^{(2)},C^{(2)})$. Compare shared modes 2 and 3 only (mode 1 is the split mode — different samples, not comparable).

**FMS for one component pair** (normalized):
$$\text{fms}_r = |b_r^{(1)T}b_r^{(2)}| \in [0,1]$$

**Overall FMS** (after optimal permutation matching):
$$\text{FMS} = \frac{1}{R}\sum_{r=1}^R \text{fms}_r^{(B)}\cdot\text{fms}_r^{(C)}$$

**Threshold**: FMS $>0.95$ = reproducible. FMS $<0.7$ = $R$ too large or no real structure.

**Repeat**: run 10–50 random splits, report median FMS (one split may be lucky/unlucky).

| | CORCONDIA | Split-Half FMS |
|--|-----------|---------------|
| Measures | Core = super-diagonal? | Reproducibility across splits |
| Detects | Wrong structure | Overfitting |
| Decision | $\geq 95$: good | $\geq 0.95$: reproducible |

**Choose $R$**: largest $R$ where both CORCONDIA $\geq95$ AND FMS $\geq0.95$.

---

## AB — PRINCIPAL COMPONENT REGRESSION (PCR)

**Why PCR**: OLS breaks when $p\geq N$ or features are collinear ($X^TX$ singular). Solution: reduce to $M\ll p$ orthogonal components.

**Step 1**: run PCA → $Z_m=Xv_m$ (top $M$ eigenvectors of $X^TX$).

**Step 2**: regress $y$ on scores $Z$ (orthogonal → $M$ independent univariate regressions):
$$\hat{\beta}_\text{PCR} = V_M(Z^TZ)^{-1}Z^Ty = V_M\Lambda_M^{-1}V_M^TX^Ty$$

**SVD view**: $\hat{\beta}_\text{PCR}=\sum_{m=1}^M v_m\frac{u_m^Ty}{d_m}$ (keep top $M$ singular directions).

**PCR vs Ridge**: both shrink small-variance directions. Ridge: $\hat{\beta}_\text{ridge}=\sum_m\frac{d_m^2}{d_m^2+\lambda}v_m\frac{u_m^Ty}{d_m}$ (continuous shrinkage). PCR: hard truncation (keeps or drops entire PC). Ridge has fractional df; PCR has integer df.

**Key weakness**: PCA selects PCs by $X$-variance, ignoring $y$. If predictive signal lives in low-variance directions of $X$, PCR misses it.

**PLS fixes this**: $\max_{u,v}\text{Cov}(Xu,Yv)$ — finds directions in $X$ that jointly explain $X$ variance AND correlate with $y$. PLS beats PCR when signal is in low-variance directions of $X$.

| | PCR | Ridge | PLS |
|--|-----|-------|-----|
| Uses $y$ in step 1? | No | No (implicitly) | Yes |
| Shrinkage | Hard truncation | Continuous | Implicit |
| df | Integer $M$ | Fractional | — |

---

## AC — ELASTIC NET REGRESSION

**Model**:
$$
\hat{\beta}=\arg\min_\beta \|y-X\beta\|^2+\lambda\left[\alpha\|\beta\|_1+(1-\alpha)\|\beta\|_2^2/2\right]
$$

**Coordinate update**:
$$
\hat{\beta}_j=\frac{S(z_j,\lambda\alpha)}{1+\lambda(1-\alpha)}
$$
Soft-thresholding ($L_1$) creates sparsity; denominator ($L_2$) stabilizes correlated predictors.

**Why it matters**: combines Lasso variable selection with Ridge stability. Strong when $p\gg n$ and predictors are highly correlated.

---

## AD — GRADIENT BOOSTING

**View**: forward stagewise additive modeling + gradient descent in function space.

At step $m$, fit weak learner to pseudo-residuals:
$$
r_{im}=-\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}
$$

Update:
$$
F_m(x)=F_{m-1}(x)+\nu\gamma_m h_m(x)
$$

**Key**: boosting mainly reduces bias (sequential error correction). Smaller learning rate $\nu$ + larger $M$ generally improves generalization.

---

## AE — REGULARIZED DISCRIMINANT ANALYSIS (RDA)

RDA interpolates between LDA and QDA by shrinking class-specific covariance:
$$
\hat{\Sigma}_k(\alpha)=\alpha\hat{\Sigma}_k+(1-\alpha)\hat{\Sigma}
$$

Optional identity shrinkage improves conditioning:
$$
\hat{\Sigma}_k^{(\gamma)}=(1-\gamma)\hat{\Sigma}_k(\alpha)+\gamma\frac{\mathrm{tr}(\hat{\Sigma}_k(\alpha))}{p}I
$$

**Limits**: $\alpha=0\Rightarrow$ LDA, $\alpha=1\Rightarrow$ QDA.  
**Purpose**: lower covariance-estimation variance while keeping more flexibility than LDA.

---

## AF — CANONICAL CORRELATION ANALYSIS (CCA)

Given two blocks $X$ and $Y$, CCA finds projections $u=Xa$, $v=Yb$ maximizing
$$
\operatorname{Corr}(u,v)=\frac{a^T\Sigma_{XY}b}{\sqrt{a^T\Sigma_{XX}a}\sqrt{b^T\Sigma_{YY}b}}
$$

Leads to generalized eigenproblem:
$$
\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}a=\rho^2\Sigma_{XX}a
$$

**Contrast with PLS**: CCA maximizes correlation; PLS maximizes covariance.

---

## AG — K-NEAREST NEIGHBORS (KNN)

**Classifier**: majority vote among $K$ nearest points.  
**Regressor**: local average among $K$ nearest responses.

Bias-variance control via $K$:
- Small $K$: low bias, high variance
- Large $K$: high bias, low variance

As $K\to N$, predictions approach global mean/majority class.

**High-dimensional caveat**: distance concentration and sparse local neighborhoods make vanilla KNN deteriorate quickly without feature scaling/reduction.
