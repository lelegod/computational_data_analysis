# CDA 02582 — DEEP DIVE REFERENCE
> Full technical depth for hard questions. Ctrl+F by topic, method name, or formula.
> Use MASTER.md first for quick lookup. Come here only when you need depth.

---

## TABLE OF CONTENTS
1. [WEEK 1 — EPE / OLS / Ridge / AIC / BIC](#week-1--epe--ols--ridge--aic--bic)
2. [WEEK 2 — Lasso / LARS / Coordinate Descent / Elastic Net / Bootstrap / Confusion Matrix](#week-2--lasso--lars--coordinate-descent--elastic-net--bootstrap--confusion-matrix)
3. [WEEK 3 — Curse of Dimensionality / Multiple Testing / FWER / FDR / BH](#week-3--curse-of-dimensionality--multiple-testing--fwer--fdr--bh)
4. [WEEK 4 — CART: Regression & Classification Trees / Pruning](#week-4--cart-regression--classification-trees--pruning)
5. [WEEK 5 — Bagging / Bootstrap Aggregating](#week-5--bagging--bootstrap-aggregating)
6. [WEEK 6 — Random Forests / Boosting / AdaBoost / Gradient Boosting](#week-6--random-forests--boosting--adaboost--gradient-boosting)
7. [WEEK 7 — SVM / Kernel Trick / Duality](#week-7--svm--kernel-trick--duality)
8. [WEEK 8 — PCA / Sparse PCA / PLS / CCA](#week-8--pca--sparse-pca--pls--cca)
9. [WEEK 9 — K-means / K-medoids / Hierarchical / GMM / Silhouette / Gap](#week-9--k-means--k-medoids--hierarchical--gmm--silhouette--gap)
10. [WEEK 10 — Neural Networks / Backprop / Autoencoders](#week-10--neural-networks--backprop--autoencoders)
11. [WEEK 11 — NMF / ICA / Archetypal Analysis / Sparse Coding](#week-11--nmf--ica--archetypal-analysis--sparse-coding)
12. [WEEK 12 — PARAFAC / Tucker / CORCONDIA / Split-Half](#week-12--parafac--tucker--corcondia--split-half)
13. [CROSS-CUTTING: Methods Comparison Tables](#cross-cutting-methods-comparison-tables)

---

## WEEK 1 — EPE / OLS / Ridge / AIC / BIC

### EPE Decomposition — Full Derivation Logic
- Model: $y = f(x) + \varepsilon$, with $E[\varepsilon]=0$, $E[\varepsilon^2]=\sigma^2$
- $\text{EPE} = E(y-\hat{f})^2 = \sigma^2 + (E[\hat{f}]-f)^2 + E[(\hat{f}-E[\hat{f}])^2]$
- Three terms: **irreducible noise** + **Bias²** + **Variance**
- Cross-terms vanish because: (1) $E[\varepsilon]=0$, (2) linearity of $E$, (3) test noise $\varepsilon$ is independent of training data in $\hat{f}$
- The decomposition is at a **specific point $x_0$**, averaged over $y$ AND over training sets $D$

| Term | Formula | Changes with complexity? |
|------|---------|--------------------------|
| Irreducible noise | $\sigma^2$ | Never — property of the data |
| Bias² | $(E[\hat{f}]-f)^2$ | Decreases as model gets more complex |
| Variance | $E[(\hat{f}-E[\hat{f}])^2]$ | Increases as model gets more complex |

### OLS — Key Properties
- Estimator: $\hat{\beta} = (X^TX)^{-1}X^Ty$
- **Unbiased**: $E[\hat{\beta}] = \beta$ (proof: substitute $y=X\beta+\varepsilon$, take expectation, use $E[\varepsilon]=0$)
- **Gauss-Markov**: minimum variance among ALL linear unbiased estimators
- **Fails** when $p > n$ ($X^TX$ not invertible) or with multicollinearity (near-singular → high variance)
- Hat matrix: $\hat{Y} = SY$ where $S = X(X^TX)^{-1}X^T$; df = trace($S$) = $p$

### Ridge — Closed Form Derivation
- Objective: $\min_\beta \|Y-X\beta\|^2 + \lambda\|\beta\|^2$
- Differentiate, set to zero: $-2X^Ty + 2X^TX\beta + 2\lambda I\beta = 0$
- Solution: $\hat{\beta}_\text{ridge} = (X^TX+\lambda I)^{-1}X^Ty$
- Adding $\lambda I$ makes the matrix **positive definite** → always invertible
- Ridge **shrinks toward zero** but never reaches it (sphere constraint has no corners)
- As $\lambda \uparrow$: bias↑, variance↓, df↓ (df = trace of $X(X^TX+\lambda I)^{-1}X^T$)

### AIC vs BIC — Deep Comparison
| Property | AIC | BIC |
|----------|-----|-----|
| Penalty | $2d$ | $\log(N)\cdot d$ |
| Motivation | Prediction accuracy | Bayesian marginal likelihood |
| Asymptotic | Equivalent to LOO-CV | Consistent (selects true model) |
| Large $n$ | Picks too complex | Penalizes more → simpler |
| Small $n$ | Reasonable | Too simple |
| When equal | $N = e^2 \approx 7.4$ | Same penalty at this sample size |
- AIC ≡ LOO-CV asymptotically (Stone 1977)
- BIC ≡ log Bayes factor comparison → comparing posterior odds
- Both minimize: AIC = $-2\log L + 2d$; BIC = $-2\log L + \log(N)d$
- For Gaussian: Cp = AIC (identical formulas)
- $\hat{\sigma}^2_e$ in Cp comes from the **full (low-bias) model**, not the current model

### CV Design Rules
- **Normalize within each fold** — never before splitting (data leakage)
- **Dependent observations** (repeated measures, time series) → keep them in same fold
- **Nested CV**: outer loop = assessment, inner loop = selection; gap (inner 5%, outer 12%) = overfitting
- **1-SE rule**: choose largest $\lambda$ whose CV error ≤ min + 1SE → simpler, more stable

---

## WEEK 2 — Lasso / LARS / Coordinate Descent / Elastic Net / Bootstrap / Confusion Matrix

### Lasso — Geometry
- L1 constraint: diamond with corners on axes
- RSS ellipsoid typically hits a corner → one coordinate = exactly 0
- Contrast Ridge: sphere → no corners → never exactly 0
- This is the fundamental geometric reason for sparsity

### Lasso Properties
- df = **number of non-zero coefficients** (at most $n$ when $p > n$)
- No closed form (L1 non-differentiable at 0)
- With correlated predictors: picks one from group arbitrarily (Elastic Net fixes this)
- Path: as $\lambda$ increases, coefficients hit 0 one by one (kinks in the path)

### LARS Algorithm — Step by Step
1. Start: $\beta=0$, residual $r=y$
2. Find variable $x_j$ with maximum $|x_j^Tr|$ (correlation with residual)
3. Move $\beta_j$ toward its OLS value until another variable $x_k$ becomes equally correlated
4. Step size: $\gamma = (c_j - c_k)/(1-\rho_{jk})$ where $c_j,c_k$ = correlations, $\rho_{jk}$ = correlation between features
5. Add $x_k$; now move in **equiangular direction** (bisects angle between $x_j$ and $x_k$)
6. Repeat; LASSO modification: if any $\beta$ crosses 0, drop it and recompute direction
- Entire path computed at cost of **one OLS fit**
- Data must be centered and normalized

### Coordinate Descent — Step by Step
For fixed $\lambda$, update each $\beta_j$ cyclically:
1. Partial residual: $r_i^{(j)} = y_i - \sum_{k\neq j} x_{ik}\tilde{\beta}_k$
2. OLS: $\tilde{\beta}_j^\text{OLS} = \frac{1}{n}\sum_i x_{ij}r_i^{(j)}$
3. Soft threshold: $\tilde{\beta}_j(\lambda) = \text{sign}(\tilde{\beta}_j^\text{OLS})(|\tilde{\beta}_j^\text{OLS}|-\lambda)_+$
4. Cycle until convergence
- If $|\tilde{\beta}_j^\text{OLS}| \leq \lambda$ → coefficient = 0
- If $> \lambda$ → shrink by exactly $\lambda$

### Elastic Net — Implementation via Augmentation
- $\alpha=1$: Lasso; $\alpha=0$: Ridge; middle: Elastic Net
- Augment: $X^* = [X; \sqrt{\lambda_2}I]$, $y^* = [y; 0]$
- Then solve Lasso on $(X^*, y^*)$ — the $L_2$ penalty is absorbed into residuals
- Handles: grouping effect (correlated variables in/out together), $p>n$, predictive power

### Bootstrap
- Sample **with replacement**, size $N$, $B$ times
- ~63.2% unique observations per sample; ~36.8% OOB
- Variance: $\widehat{\text{Var}}[S] = \frac{1}{B-1}\sum_b(S(Z^{*b})-\bar{S}^*)^2$
- Use for: SEs, CIs, bias estimation
- Do NOT use for model selection (Tibshirani's warning)
- 100-200 replications for SD; 1000-2000 for CIs

### Confusion Matrix — Full Metrics
| | Pred + | Pred - |
|--|--------|--------|
| **Actual +** | TP | FN |
| **Actual -** | FP | TN |

- Sensitivity (TPR/Recall) = TP/(TP+FN) — "of positives, what fraction found?"
- Specificity (TNR) = TN/(TN+FP) — "of negatives, what fraction correctly identified?"
- Precision (PPV) = TP/(TP+FP) — "of predicted positives, what fraction correct?"
- FPR = FP/(FP+TN) = 1 - Specificity
- F1 = 2TP/(2TP+FP+FN) — harmonic mean of precision and recall
- ROC: plots TPR vs FPR as threshold varies; AUC=1 perfect; AUC=0.5 random
- Bayes theorem style calculation: TP = prevalence × sensitivity; FP = (1-prevalence) × FPR

**Example (2022 Q13)**: 10,000 subjects, 100 with Covid, sensitivity=99%, FPR=2%
- TP = 100 × 0.99 = 99
- FP = 9900 × 0.02 = 198
- Total positives = **297**

---

## WEEK 3 — Curse of Dimensionality / Multiple Testing / FWER / FDR / BH

### Curse of Dimensionality — 5 Manifestations
1. **Sparsity**: neighborhoods become empty; KNN breaks down
2. **Distances**: all points become roughly equidistant (Euclidean meaningless)
3. **Overfitting**: $p>n$ → perfect fit to noise
4. **Edge effect**: most data at boundaries/corners (not interior)
5. **Computational**: search algorithms slow down

### Blessings (Donoho 2000)
1. Correlated features → can average (Ridge exploits this)
2. Data lies on low-dimensional manifold (PCA, latent variables)
3. Approximate finite dimensionality (continuous processes)

### FWER and Bonferroni — Deep
- Testing $M$ hypotheses at $\alpha$: $\text{FWER} = 1-(1-\alpha)^M$
- $M=20$, $\alpha=0.05$: FWER $\approx 64\%$ — massive inflation
- **Bonferroni**: reject if $p < \alpha/M$ → controls FWER at $\alpha$ but low power
- Most conservative correction; appropriate when cost of any false positive is very high

### BH Algorithm — Step by Step with Example
Goal: control $\text{FDR} = E[\text{FP}/(\text{FP+TP})] \leq q$

Given $m$ p-values, target FDR level $q$:
1. Sort p-values ascending: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Compute adaptive threshold for each rank: threshold$(i) = \frac{i}{m}q$
3. Find largest $k$ where $p_{(k)} \leq \frac{k}{m}q$
4. Reject ALL hypotheses $H_{(1)}, \ldots, H_{(k)}$ (not just the ones that passed)

**Example** ($m=5$, $q=0.20$, p-values: 0.01, 0.03, 0.15, 0.40, 0.50):
- Thresholds: 0.04, 0.08, 0.12, 0.16, 0.20
- $i=1$: $0.01 \leq 0.04$ ✓
- $i=2$: $0.03 \leq 0.08$ ✓
- $i=3$: $0.15 \leq 0.12$ ✗ → STOP
- $k=2$: reject $H_{(1)}$ and $H_{(2)}$

Key: the threshold is **adaptive** (increases with rank); it is NOT a fixed cutoff.

---

## WEEK 4 — CART: Regression & Classification Trees / Pruning

### Splitting Criteria — Full Detail
**Regression**: minimize RSS = $\sum_{i\in R_1}(y_i-c_1)^2 + \sum_{i\in R_2}(y_i-c_2)^2$

**Classification impurity measures** (all = 0 at pure node):
- Gini: $G = \sum_k \hat{p}_{mk}(1-\hat{p}_{mk}) = 1-\sum_k\hat{p}_{mk}^2$
- Cross-entropy: $D = -\sum_k \hat{p}_{mk}\log(\hat{p}_{mk})$
- Misclassification: $E = 1-\max_k(\hat{p}_{mk})$

**Binary case** ($p$ = proportion class 1):
- Misclassification = $\min(p, 1-p)$; maximum at $p=0.5$ = 0.5
- Gini = $2p(1-p)$; maximum at $p=0.5$ = 0.5
- Entropy = $-p\log p - (1-p)\log(1-p)$; maximum at $p=0.5$ = $\log 2$

**Why not misclassification for growing?** It's insensitive to probability changes within a class — Gini/entropy detect improvement in class probabilities even when majority class doesn't change.

**Why misclassification for pruning?** It directly measures what we care about at prediction time.

### Cost-Complexity Pruning
$C_\alpha(T) = \sum_m N_m Q_m(T) + \alpha|T|$
- $\alpha=0$: full tree; large $\alpha$: tree → root
- Cross-validation over $\alpha$ to find optimal subtree
- Post-pruning is better than pre-pruning (greedy early stopping misses good future splits)

### Variable Importance
$VI_j = \sum_{\text{splits on }j} N_t \cdot \Delta I_t$ — accumulated impurity reduction across all splits on feature $j$

### Key Properties
- CART is **greedy** (no lookahead)
- Deep trees: low bias, HIGH variance (tiny data changes → completely different tree)
- Handles: categorical variables (ordering trick), missing data (surrogate splits), no scaling needed

---

## WEEK 5 — Bagging / Bootstrap Aggregating

### Bagging Variance Formula — Derivation Logic
For regression with $B$ identically distributed trees, each with variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}\!\left(\frac{1}{B}\sum_b T_b(x)\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

- First term $\rho\sigma^2$: irreducible floor from tree correlation (doesn't go to zero with $B$)
- Second term $\frac{1-\rho}{B}\sigma^2$: vanishes as $B\to\infty$
- To minimize: decrease $\rho$ (RF's strategy: random feature subsampling)
- For independent trees ($\rho=0$): variance $\to 0$ as $B\to\infty$
- For perfect correlation ($\rho=1$): no reduction at all

### OOB Error
- $P(\text{obs not in bootstrap}) = (1-1/N)^N \to 1/e \approx 0.368$
- Each obs is OOB in ~36.8% of trees; predict using only those trees
- OOB error $\approx$ LOO-CV error — **unbiased** estimate
- Free by-product of bagging

### When Bagging Helps
- Best for: high variance, low bias methods (deep trees, low-K KNN)
- Not helpful for: high bias methods (stumps) — bias stays the same
- Not helpful for: low variance methods (ridge, smoothed models)

---

## WEEK 6 — Random Forests / Boosting / AdaBoost / Gradient Boosting

### Random Forests — Full Detail
- RF = Bagging + random $m < p$ features at each split
- Default: $m = \lfloor\sqrt{p}\rfloor$ (classification), $m = \lfloor p/3\rfloor$ (regression)
- Effect of $m$: smaller $m$ → lower $\rho$ → lower variance, but higher bias per tree
- RF bias = single tree bias (use deep unpruned trees)
- RF variance = $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ (same formula, but lower $\rho$ than bagging)
- When $m=p$: RF = Bagging; when $m=1$: maximum decorrelation

**Variable Importance**:
- Gini: sum impurity reductions at splits on feature $j$ across all trees
- OOB permutation: permute feature $j$'s values for OOB data, measure accuracy drop
- Both give similar rankings; Gini concentrates in top few, OOB spreads more uniformly

**Proximity Matrix**: $n\times n$ matrix counting how often obs $i$ and $j$ end up in same leaf (OOB); visualized with MDS

### Bagging vs RF vs Boosting
| Property | Bagging | RF | Boosting |
|----------|---------|-----|----------|
| Tree type | Deep | Deep | Shallow/stumps |
| Parallelizable | Yes | Yes | No (sequential) |
| Reduces bias | No | No | Yes |
| Reduces variance | Yes | More than bagging | Yes |
| Can overfit | No | No | Yes (with noise) |
| Tree dependence | Independent | Independent | Dependent |

### AdaBoost — Step by Step
Initialize weights $w_i = 1/N$ for all $i$.

For $m = 1, 2, \ldots, M$:
1. Fit classifier $G_m(x)$ to training data with weights $w_i$
2. Compute weighted error: $\text{err}_m = \sum_i w_i \mathbf{I}(y_i \neq G_m(x_i)) / \sum_i w_i$
3. Compute classifier weight: $\alpha_m = \log\!\left(\frac{1-\text{err}_m}{\text{err}_m}\right)$
4. Update observation weights: $w_i \leftarrow w_i \cdot \exp[\alpha_m \cdot \mathbf{I}(y_i \neq G_m(x_i))]$
5. Normalize weights to sum to 1

Final: $G(x) = \text{sign}\!\left[\sum_m \alpha_m G_m(x)\right]$

**Key properties**:
- $\text{err}_m = 0.5$: $\alpha_m = 0$ (useless classifier)
- $\text{err}_m = 0$: $\alpha_m = \infty$ (perfect classifier)
- $\text{err}_m > 0.5$: $\alpha_m < 0$ (classifier votes against itself)
- AdaBoost ≡ forward stagewise additive modelling with **exponential loss** $L(y,F) = \exp(-yF(x))$
- Sensitive to noise: exponential loss puts huge weight on misclassified points

### Gradient Boosting — Core Idea
At each step, fit a tree to the **negative gradient** of the loss (pseudo-residuals):
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

- For squared error loss: $r_{im} = y_i - F_{m-1}(x_i)$ (ordinary residuals)
- For exponential loss: recovers AdaBoost
- Tree depth $J$: stumps = additive model; $J$-leaf tree = $(J-1)$-way interactions
- Shrinkage $\nu$ (learning rate): $F_m = F_{m-1} + \nu \cdot h_m$ — smaller $\nu$ = better generalization, more trees needed

---

## WEEK 7 — SVM / Kernel Trick / Duality

### SVM Geometry
- Find hyperplane $\{x: x^T\beta + \beta_0 = 0\}$ with maximum margin
- Margin = $2/\|\beta\|$ (canonical scaling: support vectors satisfy $|x^T\beta+\beta_0|=1$)
- $\beta$ is **orthogonal to the hyperplane**
- Classes: $y_i \in \{-1, +1\}$
- Distance from point $x_i$ to hyperplane: $(x_i^T\beta+\beta_0)/\|\beta\|$

### Primal Problem
$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i^T\beta+\beta_0) \geq 1 \quad \forall i$$

### Lagrangian and Dual
Lagrangian: $L_P = \frac{1}{2}\|\beta\|^2 - \sum_i\alpha_i[y_i(x_i^T\beta+\beta_0)-1]$

Taking derivatives and setting to zero:
- $\partial L/\partial\beta = 0$: $\beta = \sum_i\alpha_i y_i x_i$
- $\partial L/\partial\beta_0 = 0$: $\sum_i\alpha_i y_i = 0$

Dual problem (substitute back):
$$\max_\alpha \sum_i\alpha_i - \frac{1}{2}\sum_{ij}\alpha_i\alpha_j y_i y_j\langle x_i,x_j\rangle \quad \text{s.t.} \quad \alpha_i \geq 0, \sum_i\alpha_i y_i = 0$$

**KKT complementary slackness**: $\alpha_i[y_i(x_i^T\beta+\beta_0)-1] = 0$
- Support vectors: on margin → bracket = 0 → $\alpha_i > 0$
- Safe points: beyond margin → bracket > 0 → $\alpha_i = 0$ (by KKT, one factor must be zero)

### Kernel Trick
The dual only involves $\langle x_i, x_j\rangle$ — replace with $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$:
- Linear kernel: $K(x,x') = x^Tx'$ → linear boundary
- Polynomial: $K(x,x') = (1+x^Tx')^d$ → degree-$d$ boundary
- RBF: $K(x,x') = \exp(-\gamma\|x-x'\|^2)$ → **infinite-dimensional** space, highly nonlinear
- Prediction: $\hat{y} = \text{sign}\!\left(\sum_i\alpha_i y_i K(x_i, x) + \beta_0\right)$
- The mapping $\phi$ is **implicit** — never computed explicitly

### Weak vs Strong Duality
- Weak duality: $d^* \leq p^*$ always holds
- Strong duality: $d^* = p^*$ — holds for SVM via Slater's condition (convex problem + feasible point)
- Strong duality means: solving the dual gives the exact same solution as the primal

---

## WEEK 8 — PCA / Sparse PCA / PLS / CCA

### PCA — Full Detail
- Objective: $\max_v \text{Var}(Xv)$ subject to $\|v\|=1$
- Solution: eigenvectors of the covariance matrix $\Sigma = X^TX/(n-1)$
- Eigenvalue $\lambda_k$ = variance explained by $k$-th PC
- Fraction explained: $\lambda_k/\sum_j\lambda_j$ (DO NOT square eigenvalues)
- SVD of $X$: $X = UDV^T$ → loadings = $V$, scores = $UD$, $\text{SD}_l = d_l/\sqrt{n-1}$
- EVD on covariance matrix and SVD on X give **same loadings $V$**

**Scaling matters**: PCA on unscaled data dominated by high-variance features → use correlation matrix for equal weighting

**Mode of variation**: $\mu \pm 2.5\sigma_l v_l$ — shows what varying along PC $l$ looks like

### PCA Variance Example
Given covariance matrix eigenvalues $\lambda_1=6$, $\lambda_2=2$:
- Fraction by PC1 = $6/(6+2) = 0.75 = 75\%$
- **Do NOT compute** $36/40$ — that would be for singular values (standard deviations), not eigenvalues (variances)

### Sparse PCA
- Standard PCA uses all $p$ features → hard to interpret
- Three sparsification methods: (1) Thresholding, (2) Varimax rotation, (3) Elastic Net
- After thresholding or varimax: scores **must be recomputed** and may be **correlated** (orthogonality lost)
- Elastic Net gives most principled sparse solution

### PLS — Partial Least Squares
- **Supervised**: uses $y$ to find relevant X-subspace
- Objective: $\max \text{Cov}(Xu, Yv)$ — balances variance AND correlation
- PCR flaw: highest-variance X directions may have zero correlation with $y$ → PLS avoids this
- PLS components are **orthogonal** by construction (deflation step)
- PLS with $M=p$: equivalent to OLS (no regularization)
- PLS with $M<p$: regularized regression

**PLS Algorithm** (one direction):
1. $\hat{\phi}_{mj} = x_j^{(m-1)^T}y$ (weight each feature by its covariance with y)
2. $z_m = \sum_j\hat{\phi}_{mj}x_j^{(m-1)}$ (latent component)
3. $\hat{\theta}_m = z_m^Ty/z_m^Tz_m$ (regression coefficient)
4. $\hat{y}^{(m)} = \hat{y}^{(m-1)} + \hat{\theta}_mz_m$ (prediction update)
5. Deflate: $x_j^{(m)} = x_j^{(m-1)} - (z_m^Tx_j^{(m-1)}/z_m^Tz_m)z_m$

### CCA — Canonical Correlation Analysis
- Finds associations between two matrices $X$ and $Y$
- Objective: $\max \text{Corr}^2(Xu, Yv)$ — pure correlation (ignores variance unlike PLS)
- Requires inverting $\Sigma_{XX}$ and $\Sigma_{YY}$ → **fails when $p>n$**
- Solution for $p>n$: Regularized CCA (add $\lambda I$) or Sparse CCA (PMD with L1 penalty)
- At most $\min(p,q)$ canonical variate pairs

| Method | Supervised? | Objective | High-dim? |
|--------|-------------|-----------|-----------|
| PCA | No | Max variance of $Xv$ | Yes (dimensionality reduction) |
| PCR | Yes (indirect) | PCA then regress | Yes |
| PLS | Yes | Max Cov$(Xu, Yv)$ | Yes |
| CCA | Yes (two-sided) | Max Corr$(Xu, Yv)$ | No (needs invertible $\Sigma$) |

---

## WEEK 9 — K-means / K-medoids / Hierarchical / GMM / Silhouette / Gap

### K-means — Full Detail
- Objective: $\min\sum_k\sum_{i\in C_k}\|x_i-\mu_k\|^2$
- Algorithm: (1) Assign each point to nearest centroid; (2) Update centroid = mean of cluster; (3) Repeat
- Uses **Euclidean distance only**
- Favors: convex, spherical clusters of similar size
- Sensitive to: outliers (means can be pulled), initialization (local optima)
- Multiple restarts recommended
- $K$ must be specified in advance

### K-medoids
- Centers = actual data points (medoids), not computed means
- More robust to outliers
- Works with **any distance measure** (not just Euclidean)

### Hierarchical Clustering — Linkages
- Single: min distance between any two points in clusters → **chaining**
- Complete: max distance between any two points → **compact clusters**
- Average: average pairwise distance → compromise
- Ward: minimize increase in total within-cluster variance → **requires Euclidean**
- Produces dendrogram: cut at height $h$ → specific number of clusters

### GMM — Full Detail
- $X_i \sim \mathcal{N}(\mu_j, \Sigma_j)$ if $Z_i=j$ (latent cluster assignment)
- $\pi_j = P(Z_i=j)$, $\sum_j\pi_j=1$ (mixing proportions)
- Solved by EM algorithm

**EM for GMM**:
- **E-step** (compute soft assignments):
$$\gamma_{ij} = \frac{\pi_j\mathcal{N}(x_i;\mu_j,\Sigma_j)}{\sum_{j'}\pi_{j'}\mathcal{N}(x_i;\mu_{j'},\Sigma_{j'})}$$
- **M-step** (update parameters using soft assignments):
$$\mu_j^{(\text{new})} = \frac{\sum_i\gamma_{ij}x_i}{\sum_i\gamma_{ij}}, \quad \Sigma_j^{(\text{new})} = \frac{\sum_i\gamma_{ij}(x_i-\mu_j)(x_i-\mu_j)^T}{\sum_i\gamma_{ij}}, \quad \pi_j^{(\text{new})} = \frac{1}{n}\sum_i\gamma_{ij}$$

- K-means = GMM with hard assignments + equal spherical covariances ($\Sigma_j = \sigma^2 I$)
- Model selection: AIC or BIC (NOT silhouette/gap — those are for K-means)
- High-dim tricks: shared covariance, diagonal $\Sigma$, regularize $\Sigma=\Sigma+\lambda I$, PCA first

### Silhouette Score
$$s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$$
- $a(i)$ = avg distance to **same cluster** (cohesion; want small)
- $b(i)$ = avg distance to **nearest other cluster** (separation; want large)
- $s(i) \in [-1,1]$: 1=perfect, 0=boundary, negative=misclassified
- Favors convex spherical clusters; unreliable for non-spherical shapes

### Gap Statistic
$$G(K) = \log(U_K) - \log(W_K)$$
- $W_K$ = actual within-cluster dissimilarity; $U_K$ = expected for uniform random data (20 simulations)
- Choose $K^* = \arg\min_k\{K: G(K) \geq G(K+1)-s'_{K+1}\}$
- More principled than silhouette; works for K-means, K-medoids, hierarchical

---

## WEEK 10 — Neural Networks / Backprop / Autoencoders

### MLP Architecture
- Layer $\ell$: pre-activation $z^{(\ell)} = W^{(\ell)}a^{(\ell-1)} + b^{(\ell)}$; activation $a^{(\ell)} = \sigma(z^{(\ell)})$
- Output: no activation (regression); sigmoid (binary); softmax (multiclass)
- Parameters per layer: (inputs × units) + units (biases)

**Parameter count formula**: for layer $i\to j$: $i\times j + j$ parameters

**Examples**:
- $3\to4\to2\to1$ with biases: $(3\times4+4)+(4\times2+2)+(2\times1+1) = 16+10+3 = \mathbf{29}$
- $10\to2\to2\to1$ with biases: $(10\times2+2)+(2\times2+2)+(2\times1+1) = 22+6+3 = \mathbf{31}$

### Loss Functions — Derivation
- **Regression** → Gaussian likelihood → MSE: $-\log P(y|x) \propto (y-\hat{y})^2$
- **Binary classification** → Bernoulli likelihood → BCE: $-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$
- Neither is arbitrary — both derived from negative log-likelihood

### Sigmoid and Derivative
- $\sigma(x) = 1/(1+e^{-x})$; maps $\mathbb{R}\to(0,1)$
- $\sigma'(x) = \sigma(x)(1-\sigma(x))$ — **computable from output alone** (no recomputing $e^{-x}$)
- Max value = 0.25 at $x=0$ → vanishing gradient when multiplied through many layers

### Backpropagation — Full Algorithm
1. **Forward pass**: compute all $z^{(\ell)}, a^{(\ell)}$; STORE ALL intermediate values (needed for backward)
2. **Backward pass**: propagate error signals $\delta^{(\ell)}$ from output to input
   - Output layer: $\delta^{(L)} = \nabla_a L \odot \sigma'(z^{(L)})$
   - Hidden layers: $\delta^{(\ell)} = (W^{(\ell+1)})^T\delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$
3. **Gradient**: $\partial L/\partial W^{(\ell)} = \delta^{(\ell)}(a^{(\ell-1)})^T$
4. **Update**: $W^{(\ell)} \leftarrow W^{(\ell)} - \eta \cdot \partial L/\partial W^{(\ell)}$

Why $(W^T\delta)$: each $a^{(\ell)}_i$ connects to ALL neurons in layer $\ell+1$ → multivariate chain rule sums over all paths → becomes $W^T\delta$

### Architecture Guide
| Architecture | Use case | Key property |
|-------------|----------|--------------|
| MLP | Tabular/fixed-size data | Fully connected |
| CNN | Images/grids | Weight sharing, translation equivariance |
| RNN/LSTM | Sequences | Hidden state; LSTM fixes vanishing gradient |
| Autoencoder | Unsupervised representation | Encoder → bottleneck → decoder; reconstructs input |
| Transformer | Long-range sequences | Self-attention; parallelizable |

---

## WEEK 11 — NMF / ICA / Archetypal Analysis / Sparse Coding

### All Methods Compared
| Method | Model | Key constraint | Unique? | Use case |
|--------|-------|----------------|---------|----------|
| PCA | $X\approx WH$ | Orthogonality | Yes | Variance explanation |
| NMF | $X\approx WH$ | $W\geq0, H\geq0$ | No ($Q$-ambiguity) | Parts-based (faces, spectra) |
| ICA | $X=AS$ | Non-Gaussianity, independence | Yes (up to perm/scale) | Source separation |
| AA | $X\approx XSH$ | Archetypes on convex hull | Partially | Extreme profiles |
| Sparse Coding | $X\approx WH$ | $H$ sparse ($L_1$) | No | Overcomplete dictionary |

### NMF — Full Detail
- Both $W\geq0$ AND $H\geq0$ (not just one)
- Non-negativity → parts-based additive representation (no cancellation)
- NOT jointly convex in $(W,H)$; only convex in one given the other → alternating minimization
- Multiplicative updates: $H_{kj}\leftarrow H_{kj}\cdot\frac{(W^TX)_{kj}}{(W^TWH)_{kj}}$; $W_{ik}\leftarrow W_{ik}\cdot\frac{(XH^T)_{ik}}{(WHH^T)_{ik}}$
- Multiplicative updates = GD with adaptive learning rate; preserve non-negativity if initialized positive
- Non-unique: $WH = (WQ^{-1})(QH)$ for any invertible $Q$ with non-neg sides
- Disambiguation: geometric constraints or $L_1$ sparsity penalties

### ICA — Full Detail
- Model: $x = As$; goal: find $W \approx A^{-1}$ so $\hat{s}=Wx$ has statistically independent components
- Requirements: (1) sources are **non-Gaussian** AND (2) sources are **statistically independent**
- Cannot separate Gaussian sources (CLT: mixtures are more Gaussian → no signal to exploit)
- Strategy: find $W$ maximizing non-Gaussianity (kurtosis or negentropy)
- Excess kurtosis: Gaussian=0; Laplace=3; Uniform=-1.2
- Whitening (required preprocessing): transform data so $E[\tilde{x}\tilde{x}^T]=I$ → reduces problem to finding rotations only
- FastICA: $w_\text{new} \leftarrow E[\tilde{x}g(w^T\tilde{x})] - E[g'(w^T\tilde{x})]w$; normalize after each step
- Indeterminacies: permutation and sign of components (ICA is unique up to these)
- PCA finds uncorrelated components; ICA finds **statistically independent** components (strictly stronger)

### Archetypal Analysis — Full Detail
- Objective: $\min_{S,H}\|X-XSH\|_F^2$
- $S$: $s_{ij}\geq0$, $\sum_i s_{ij}=1$ → archetypes = convex combinations of DATA POINTS ($Z=XS$)
- $H$: $h_{ij}\geq0$, $\sum_i h_{ij}=1$ → data = convex combinations of archetypes
- Archetypes lie on the **convex hull** (extreme points), NOT interior like k-means centroids
- The $XS$ constraint anchors archetypes to real data (not arbitrary points in space)
- AA vs k-means: AA → extremes; k-means → centroids
- AA vs NMF: NMF: $W$ arbitrary; AA: archetypes must be $XS$ (data-grounded)
- AA vs PCA: PCA → average profile; AA → extreme profiles

### Sparse Coding
- Overcomplete dictionary: $K > I$ atoms (more basis vectors than dimensions)
- $L = \frac{1}{2}\|X-WH\|_F^2 + \lambda\sum_j\|h_j\|_1$
- Step 1 (fix $W$, update $h$): = Lasso problem → solve with Coordinate Descent or LARS
- Step 2 (fix $H$, update $W$): standard LS with unit norm constraint $\|w_k\|_2\leq1$
- Unit norm on $W$ required: without it, $W\to\infty$, $H\to0$ trivially minimizes $L_1$
- CV: **Speckled CV** (mask individual entries, NOT rows) — row holdout fails (can't learn $H$ for new rows)

---

## WEEK 12 — PARAFAC / Tucker / CORCONDIA / Split-Half

### Tensor Basics
- N-way tensor $\mathcal{X}\in\mathbb{R}^{I_1\times I_2\times\cdots\times I_N}$
- 3-way slices: horizontal $X(i,:,:)$, lateral $X(:,j,:)$, frontal $X(:,:,k)$
- 3-way fibers: column $X(:,j,k)$, row $X(i,:,k)$, tube $X(i,j,:)$
- Frobenius norm: $\|\mathcal{A}\|_F = \sqrt{\sum_{ijk}a_{ijk}^2}$
- Mode-$n$ unfolding: $X_{(n)}\in\mathbb{R}^{I_n\times \prod_{m\neq n}I_m}$
- N-mode multiplication: $[\mathcal{X}\times_n M]_{(n)} = MX_{(n)}$

### Tucker3 — Full Detail
- Decompose: $\mathcal{X}\approx\mathcal{G}\times_1 A\times_2 B\times_3 C$
- $\mathcal{G}\in\mathbb{R}^{P\times Q\times R}$ = core tensor (defines cross-talk between components)
- $A\in\mathbb{R}^{I\times P}$, $B\in\mathbb{R}^{J\times Q}$, $C\in\mathbb{R}^{K\times R}$ = mode loading matrices
- Ranks $P,Q,R$ can be **different** per mode
- Scalar form: $x_{ijk}\approx\sum_p\sum_q\sum_r g_{pqr}a_{ip}b_{jq}c_{kr}$
- Matrix form (mode 1): $X_{(1)}\approx A\,G_{(1)}(C\otimes B)^T$ → uses **Kronecker product** $\otimes$
- ALS update for $A$: define $Z_A = G_{(1)}(C\otimes B)^T$; then $A\leftarrow X_{(1)}Z_A^T(Z_AZ_A^T)^{-1}$
- After fitting: $\mathcal{G}\leftarrow\mathcal{X}\times_1 A^{-1}\times_2 B^{-1}\times_3 C^{-1}$
- **NOT unique**: can rotate $\mathcal{G}$ by any $Q$ and compensate in loading matrices
- Best for: **data compression** (flexible ranks per mode)

### PARAFAC (CP) — Full Detail
- Decompose: $\mathcal{X}\approx\sum_{r=1}^R a_r\circ b_r\circ c_r$
- Special case of Tucker3 with **super-diagonal core** (identity-like tensor $\mathcal{I}^{R\times R\times R}$)
- Scalar form: $x_{ijk}\approx\sum_r a_{ir}b_{jr}c_{kr}$
- Matrix form (mode 1): $X_{(1)}\approx A(C\odot B)^T$ → uses **Khatri-Rao product** $\odot$ (NOT Kronecker)
- ALS update for $A$: $A\leftarrow X_{(1)}(C\odot B)(C^TC*B^TB)^{-1}$ where $*$ = Hadamard (elementwise)
- **Essentially unique** → super-diagonal constraint prevents arbitrary rotation
- Components are **NOT nested**: changing $R$ changes all components (unlike PCA)
- Best for: **resolving physical/spectral profiles** (interpretable, additive, unique)

### Tucker vs PARAFAC Summary
| Property | Tucker3 | PARAFAC |
|----------|---------|---------|
| Core | Full $P\times Q\times R$ | Super-diagonal $R\times R\times R$ |
| Ranks | Different per mode | Single $R$ |
| Product | Kronecker $\otimes$ | Khatri-Rao $\odot$ |
| Unique | NO | YES |
| Best for | Compression | Physical profiles |
| Nested components | Yes | No |

### CORCONDIA
$$\text{CORCONDIA} = 100\cdot\left(1 - \frac{\|\mathcal{I}-\mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$
- $\mathcal{I}$ = perfect super-diagonal tensor (what PARAFAC assumes the core to be)
- $\mathcal{G}$ = actual core fitted from data (computed from PARAFAC loadings)
- CORCONDIA ≈ 100: $\mathcal{G}$ is nearly super-diagonal → $R$ is appropriate
- CORCONDIA ≈ 0 or negative: $\mathcal{G}$ deviates strongly → $R$ too large
- Choose the **largest $R$ before CORCONDIA drops sharply**
- Specifically for PARAFAC — NOT for Tucker model selection

### Split-Half FMS
$$\text{FMS} = \sum_{r=1}^R\frac{a_r^T\hat{a}_r\cdot b_r^T\hat{b}_r\cdot c_r^T\hat{c}_r}{\|a_r\|\|\hat{a}_r\|\|b_r\|\|\hat{b}_r\|\|c_r\|\|\hat{c}_r\|}$$
- Split DATA first; fit separate PARAFAC models to each half independently
- FMS close to $R$: stable (good $R$); FMS $\ll R$: unstable ($R$ too large)
- Use CORCONDIA + FMS together for confirmation

---

## CROSS-CUTTING: Methods Comparison Tables

### Supervised vs Unsupervised
| Supervised | Unsupervised |
|-----------|--------------|
| OLS, Ridge, Lasso, EN | PCA, Sparse PCA |
| LDA, Logistic Regression | GMM, K-means, K-medoids |
| SVM, CART, RF, Boosting | Hierarchical Clustering |
| Neural Networks | NMF, ICA, AA, Sparse Coding |
| PLS (uses y) | CCA (two-sided, uses X and Y but no single y) |
| | Tucker, PARAFAC |
| | Autoencoder |

### Methods for p >> n
| Works well | Fails |
|-----------|-------|
| SVM (dual formulation) | OLS (singular $X^TX$) |
| Ridge (adds $\lambda I$) | Standard Logistic Regression |
| Lasso (selects ≤ $n$ vars) | CCA (singular $\Sigma_{XX}$) |
| Elastic Net | |
| RF (random feature subsets) | |
| PCA (dimensionality reduction) | |

### Regularization / Penalization Methods
| Method | Penalty | Zeros? | Closed form? |
|--------|---------|--------|-------------|
| OLS | None | No | Yes |
| Ridge | $\lambda\|\beta\|_2^2$ | No | Yes: $(X^TX+\lambda I)^{-1}X^Ty$ |
| Lasso | $\lambda\|\beta\|_1$ | Yes | No (LARS/CD) |
| Elastic Net | $\lambda[\frac{1-\alpha}{2}\|\beta\|_2^2+\alpha\|\beta\|_1]$ | Yes | No |

### Kernel Methods
| Method | Uses kernel? | Effect |
|--------|------------|--------|
| SVM | Yes (dual formulation) | Nonlinear boundary |
| Kernel PCA | Yes | Nonlinear dimensionality reduction |
| Ridge regression | Not standard | Can be kernelized |
| Random Forest | No | |
| Boosting | No | |

### Choosing $K$ or Number of Components
| Method | Use |
|--------|-----|
| K-means | Silhouette, gap statistic, elbow |
| K-medoids | Silhouette, gap statistic |
| Hierarchical | Dendrogram (visual), gap statistic |
| GMM | AIC or BIC (likelihood-based) |
| PCA | Scree plot (elbow), cumulative variance threshold |
| PARAFAC | CORCONDIA + Split-half FMS |
| NMF | Speckled CV |
| Ridge/Lasso | Cross-validation, AIC, BIC, 1-SE rule |
