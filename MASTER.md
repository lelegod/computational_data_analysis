# CDA 02582 — MASTER EXAM CHEAT SHEET
> Ctrl+F is your friend. Every topic, every trap, every formula.
> Format: 1+ correct options per question. Evaluate each option independently as True/False.

---

## QUICK TOPIC INDEX
- [W1: Bias-Variance / EPE / OLS / Ridge](#w1-bias-variance--epe--ols--ridge)
- [W2: Lasso / Elastic Net / LARS / Coordinate Descent](#w2-lasso--elastic-net--lars--coordinate-descent)
- [W2: AIC / BIC / Cp / Bootstrap / CV](#w2-aic--bic--cp--bootstrap--cv)
- [W3: Multiple Testing — Bonferroni / BH / FDR](#w3-multiple-testing--bonferroni--bh--fdr)
- [W4: CART — Decision Trees](#w4-cart--decision-trees)
- [W5: Bagging](#w5-bagging)
- [W6: Random Forests / Boosting / AdaBoost](#w6-random-forests--boosting--adaboost)
- [W7: SVM / Kernel Trick](#w7-svm--kernel-trick)
- [W8: PCA / PLS / CCA / Subspace Methods](#w8-pca--pls--cca--subspace-methods)
- [W9: Clustering — K-means / K-medoids / Hierarchical / GMM](#w9-clustering--k-means--k-medoids--hierarchical--gmm)
- [W10: Neural Networks / Autoencoders](#w10-neural-networks--autoencoders)
- [W11: NMF / ICA / Archetypal Analysis / Sparse Coding](#w11-nmf--ica--archetypal-analysis--sparse-coding)
- [W12: Multiway Models — PARAFAC / Tucker / CORCONDIA](#w12-multiway-models--parafac--tucker--corcondia)
- [CROSS-TOPIC TRAPS](#cross-topic-traps)
- [OPEN QUESTION Q22 TEMPLATE](#open-question-q22-template)

---

## W1: Bias-Variance / EPE / OLS / Ridge

### Key Formulas
| Formula | What |
|---------|------|
| $\text{EPE} = \sigma^2 + \text{Bias}^2 + \text{Variance}$ | Always 3 terms |
| $\hat{\beta}_\text{OLS} = (X^TX)^{-1}X^Ty$ | OLS — requires invertible $X^TX$ |
| $\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1}X^Ty$ | Ridge — always invertible |
| $\text{df}(\lambda) = \text{trace}(X(X^TX+\lambda I)^{-1}X^T)$ | Ridge effective df |
| $C_p = \widehat{\text{err}} + 2\frac{d}{N}\hat{\sigma}^2_e$ | In-sample criterion |

### Traps
- $\sigma^2$ (irreducible noise) is **NOT** affected by model complexity — no model can reduce it
- Ridge **DOES** have closed form; Lasso does NOT
- Ridge **DOES NOT** set coefficients to zero — only shrinks
- Large $\lambda$ → high bias, low variance; small $\lambda$ → low bias, high variance
- OLS fails when $p > n$ (singular); Ridge still works (adding $\lambda I$ makes it invertible)
- $C_p$ and AIC are **identical** for Gaussian models
- AIC is asymptotically equivalent to **leave-one-out CV** (not k-fold)
- BIC penalizes more than AIC for $n > e^2 \approx 7.4$ (BIC penalty: $\log(N)d$ vs AIC's $2d$)
- $\hat{\sigma}^2_e$ in $C_p$ comes from the **low-bias (full OLS) model**, not the current model
- Increasing $\lambda$ → df **decreases** (not increases)

---

## W2: Lasso / Elastic Net / LARS / Coordinate Descent

### Key Formulas
| Formula | What |
|---------|------|
| Lasso: $\min \|Y-X\beta\|^2 + \lambda\|\beta\|_1$ | L1 penalty — sparsity |
| Soft threshold: $\text{sign}(x)(|x|-\lambda)_+$ | Coordinate descent update |
| EN penalty: $\lambda[\frac{1-\alpha}{2}\|\beta\|_2^2 + \alpha\|\beta\|_1]$ | $\alpha=1$→Lasso, $\alpha=0$→Ridge |
| LARS step: $\gamma = (c_j-c_k)/(1-\rho_{jk})$ | Equiangular step size |

### Traps
- Lasso: **no closed form** (L1 non-differentiable at 0); uses LARS or coordinate descent
- Lasso **df** = number of non-zero coefficients (not $p$, not $n$)
- Lasso selects **at most $n$ variables** when $p > n$
- With highly correlated predictors: Lasso picks **one arbitrarily** — use Elastic Net instead
- LARS computes the **entire path** (all $\lambda$) at speed of one OLS fit
- LASSO = LARS **with modification** (drop variables when they cross zero)
- Elastic Net: $\alpha=1$ is Lasso; $\alpha=0$ is Ridge (not the other way around)
- Nested CV needed for **unbiased assessment** of a full pipeline that includes hyperparameter tuning
- 1-SE rule: choose the **most regularized (largest $\lambda$)** model within 1 SE of minimum CV error
- Bootstrap is for **statistical accuracy** (SEs, CIs) — NOT for model selection

---

## W2: AIC / BIC / Cp / Bootstrap / CV

### Traps
- AIC: "more data → can afford more complexity" — penalty constant ($2d$)
- BIC: "more data → stricter" — penalty grows as $\log(N) \cdot d$
- BIC → consistent model selection (right model probability → 1 as $n \to \infty$)
- AIC tends to pick **too complex** a model asymptotically
- BIC tends to pick **too simple** for small $n$
- AIC/BIC require a **parametric likelihood** — cannot directly apply to K-means
- Normalize **within** each CV fold — NOT before splitting (causes data leakage)
- Dependent observations must stay in the **same fold** (time series, repeated measures)

---

## W3: Multiple Testing — Bonferroni / BH / FDR

### Key Formulas
| Formula | What |
|---------|------|
| $\text{FWER} = 1-(1-\alpha)^M$ | M independent tests at level $\alpha$ |
| Bonferroni: reject if $p < \alpha/M$ | Controls FWER |
| BH: find max $k$ where $p_{(k)} \leq \frac{k}{m}q$, reject top $k$ | Controls FDR |
| $\text{FDR} = E[\text{FP}/(\text{FP+TP})]$ | Expected false discovery proportion |

### Traps
- **Bonferroni** controls FWER (probability of ANY false positive)
- **BH** controls FDR (expected PROPORTION of false discoveries among all discoveries)
- BH has **more power** (more discoveries); Bonferroni is more conservative
- BH at same $\alpha$ → more significant findings AND more false positives than Bonferroni
- BH threshold is **adaptive**: $\frac{i}{m}q$ — not a fixed cutoff
- BH: stop at first failure; reject **all hypotheses 1 through $k$** (not just the ones that passed)
- Example: 20 tests at $\alpha=0.05$ → FWER $= 1-(0.95)^{20} \approx 64\%$

---

## W4: CART — Decision Trees

### Key Formulas
| Formula | What |
|---------|------|
| $G = \sum_k \hat{p}_{mk}(1-\hat{p}_{mk})$ | Gini index (growing) |
| $D = -\sum_k \hat{p}_{mk}\log(\hat{p}_{mk})$ | Cross-entropy (growing) |
| $E = 1 - \max_k(\hat{p}_{mk})$ | Misclassification rate (pruning) |
| $C_\alpha(T) = R(T) + \alpha|T|$ | Cost-complexity pruning |

### Traps
- **Growing**: use Gini or cross-entropy (NOT misclassification rate — it's insensitive to probability changes)
- **Pruning/evaluation**: use misclassification rate
- Deep trees = low bias, **high variance**; shallow trees = high bias, low variance
- $\alpha=0$ → full (largest) tree; large $\alpha$ → small (pruned) tree
- CART handles missing data via **surrogate splits**
- CART does NOT require feature scaling (splits based on ordering, not distances)
- Gini and cross-entropy give **similar trees** in practice

---

## W5: Bagging

### Key Formula
$$\text{Var}_\text{bag} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

### Traps
- Bagging reduces **variance only** — bias stays the same (= single tree bias)
- More trees ($B \to \infty$): variance → $\rho\sigma^2$ (floor, not zero)
- The **floor** $\rho\sigma^2$ is limited by inter-tree correlation $\rho$
- Bagging uses **deep (unpruned)** trees — pruned trees have less variance, making bagging less effective
- OOB error ≈ leave-one-out CV error — **unbiased** (not optimistic like training error)
- Bootstrap sample contains ~**63.2% unique** observations; ~36.8% are OOB
- More trees do NOT cause overfitting in bagging
- Bagging reduces interpretability

---

## W6: Random Forests / Boosting / AdaBoost

### Key Formulas
| Formula | What |
|---------|------|
| RF default $m$: $\lfloor\sqrt{p}\rfloor$ (classification), $\lfloor p/3\rfloor$ (regression) | Feature subsampling |
| $\alpha_m = \log\!\left[\frac{1-\text{err}_m}{\text{err}_m}\right]$ | AdaBoost classifier weight |
| $G(x) = \text{sign}[\sum_m \alpha_m G_m(x)]$ | AdaBoost final prediction |
| $r_{im} = -[\partial L/\partial F]_{F=F_{m-1}}$ | Gradient boosting pseudo-residual |

### Random Forest Traps
- RF lowers $\rho$ (inter-tree correlation) → lower variance than Bagging
- RF bias = **single tree bias** → use DEEP trees (not stumps)
- When $m = p$: RF = plain Bagging (no decorrelation benefit)
- RF **CAN** handle $p > n$
- RF trees are **independent → parallelizable**
- Gini variable importance ≠ OOB permutation importance — rankings are usually similar but differ
- Cannot directly compare OOB error with CV error from a different model

### Boosting Traps
- Boosting reduces **bias** (primarily), not just variance
- Boosting uses **shallow trees / stumps** (not deep trees — opposite of RF)
- Boosting trees are **sequential and dependent** — NOT parallelizable
- Boosting **CAN overfit**, especially with noisy data
- AdaBoost = forward stagewise additive modelling with **exponential loss**
- If $\text{err}_m = 0.5$: $\alpha_m = 0$ (classifier contributes nothing)
- If $\text{err}_m > 0.5$: $\alpha_m < 0$ (votes against its own prediction)
- Exponential loss penalizes misclassifications **much more** than binomial deviance → noisy data → prefer binomial deviance
- Smaller learning rate $\nu$ → better generalization but more trees needed

---

## W7: SVM / Kernel Trick

### Key Formulas
| Formula | What |
|---------|------|
| Primal: $\min \frac{1}{2}\|\beta\|^2$ s.t. $y_i(x_i^T\beta+\beta_0)\geq 1$ | SVM optimization |
| Dual: $\max \sum_i\alpha_i - \frac{1}{2}\sum_{ij}\alpha_i\alpha_j y_i y_j\langle x_i,x_j\rangle$ | Entry point for kernel trick |
| KKT: $\alpha_i[y_i(x_i^T\beta+\beta_0)-1]=0$ | Complementary slackness |
| RBF kernel: $K(x,x')=\exp(-\gamma\|x-x'\|^2)$ | Maps to infinite-dimensional space |
| Margin $= 1/\|\beta\|$ | Maximizing margin = minimizing $\|\beta\|$ |

### Traps
- SVM has **NO probabilistic model** — purely geometric
- $\beta$ is **orthogonal** (perpendicular) to the hyperplane (not parallel)
- Labels are **−1 and +1** (not 0 and 1)
- Only **support vectors** ($\alpha_i > 0$) define the boundary; safe points have $\alpha_i = 0$
- Kernel trick: mapping is **implicit** — we never compute high-dimensional coordinates
- RBF kernel → **infinite-dimensional** feature space
- Linearity depends on **kernel choice**, NOT on the dual formulation
- Weak duality: $d^* \leq p^*$; Strong duality: $d^* = p^*$ (holds for SVM via Slater's condition)

---

## W8: PCA / PLS / CCA / Subspace Methods

### Key Formulas
| Formula | What |
|---------|------|
| PCA objective: $\max_v \text{Var}(Xv)$ | Unsupervised variance maximization |
| PLS objective: $\max \text{Cov}(Xu, Yv)$ | Supervised covariance maximization |
| CCA objective: $\max \text{Corr}^2(Xu, Yv)$ | Pure correlation (ignores variance) |
| Variance explained by PC $k$: $\lambda_k/\sum_j\lambda_j$ | Eigenvalue ratio (NOT squared) |

### Traps
- PCA is **unsupervised** — ignores y completely
- PCA on unscaled data is dominated by high-variance features — use correlation matrix for equal weighting
- EVD on covariance matrix and SVD on X give the **same loading vectors $V$**
- **PLS** uses y; PCR does not — PCR can discard X-directions most predictive of y
- PLS with $M = p$ → **equivalent to OLS** (not regularized)
- **CCA** maximizes correlation only — ignores variance of X and Y (PLS does not)
- CCA requires inverting $\Sigma_{XX}$ → **fails when $p > n$** (use Regularized CCA or Sparse CCA)
- Sparse PCA: thresholding/varimax destroys orthogonality — scores must be recomputed and may be correlated
- Variance explained: use eigenvalue ratio $\lambda_1/(\lambda_1+\lambda_2)$ — **do NOT square the eigenvalues**
  - Given eigenvalues of covariance matrix → ratio directly
  - Given singular values of X → square first, then ratio

---

## W9: Clustering — K-means / K-medoids / Hierarchical / GMM

### Key Formulas
| Formula | What |
|---------|------|
| K-means objective: $\min\sum_k\sum_{i\in C_k}\|x_i-\mu_k\|^2$ | Hard assignment, Euclidean |
| Silhouette: $s(i)=(b(i)-a(i))/\max\{a(i),b(i)\}$ | Cluster quality ($\in[-1,1]$) |
| GMM E-step: $\gamma_{ij}=\pi_j\mathcal{N}(x_i;\mu_j,\Sigma_j)/\sum_{j'}\ldots$ | Soft assignments |
| GMM mixing: $\pi_j = (1/n)\sum_i\gamma_{ij}$ | M-step proportion update |

### Traps
- Clustering **ALWAYS** produces a grouping — even on random data
- K-means finds a **local** optimum — use multiple random restarts
- K-means uses **Euclidean distance only** — K-medoids can use any distance
- **K-medoids** is more robust to outliers (centers are actual data points)
- **Ward linkage** requires **Euclidean distance** specifically
- Single linkage → chaining; complete linkage → compact clusters
- GMM gives **soft (probabilistic)** assignments; K-means gives hard assignments
- K-means is a special case of GMM with hard assignments + equal spherical covariances
- $\pi_j$ = mixing proportion (prior probability), NOT the mean of cluster $j$
- **E-step** computes $\gamma_{ij}$; **M-step** updates parameters (not the other way around)
- AIC/BIC for model selection → only valid for **GMM** (has a likelihood); NOT for K-means
- Silhouette and gap statistic for K-means; AIC/BIC for GMM
- Silhouette: $s(i) \approx 1$ = well clustered; negative = possibly misclassified
- **Cross-validation does NOT directly apply** to unsupervised clustering

---

## W10: Neural Networks / Autoencoders

### Key Formulas
| Formula | What |
|---------|------|
| Parameters per layer: (inputs × units) + units (biases) | Parameter counting |
| BCE: $-\sum_i[y_i\ln\hat{y}_i + (1-y_i)\ln(1-\hat{y}_i)]$ | Binary classification loss |
| $\sigma(x)=1/(1+e^{-x})$; $\sigma'(x)=\sigma(x)(1-\sigma(x))$ | Sigmoid and derivative |
| Backprop: $\delta^{(\ell)} = (W^{(\ell+1)})^T\delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$ | Error signal propagation |

### Parameter Count Example
- 3→4→2→1 with biases: $(3\times4+4)+(4\times2+2)+(2\times1+1) = 16+10+3 = \mathbf{29}$
- 10→2→2→1 with biases: $(10\times2+2)+(2\times2+2)+(2\times1+1) = 22+6+3 = \mathbf{31}$

### Traps
- BCE and MSE are NOT arbitrary — they come from negative log-likelihood (Bernoulli and Gaussian)
- Autoencoder is **unsupervised** — uses input as its own target (reconstruction loss)
- Backprop stores **all intermediate activations** $z^{(\ell)}$, $a^{(\ell)}$ during forward pass
- Activations flow **forward**; gradients flow **backward**
- Vanishing gradient is a problem for **RNNs** (not CNNs)
- Transformers use **self-attention** — fully parallelizable (not sequential like RNNs)
- Neural networks **tend to overfit** — not robust against it by default
- $\sigma'(x) = \sigma(x)(1-\sigma(x))$ — no need to recompute $e^{-x}$

---

## W11: NMF / ICA / Archetypal Analysis / Sparse Coding

### Key Formulas
| Formula | What |
|---------|------|
| NMF: $\min_{W,H\geq0}\frac{1}{2}\|X-WH\|_F^2$ | Both W and H non-negative |
| NMF update $H$: $H_{kj}\leftarrow H_{kj}\cdot\frac{(W^TX)_{kj}}{(W^TWH)_{kj}}$ | Multiplicative (preserves non-neg) |
| ICA: $x = As$; find $W\approx A^{-1}$ s.t. $\hat{s}=Wx$ | Source separation |
| AA objective: $\min_{S,H}\|X-XSH\|_F^2$ | Archetypes = data combinations |
| AA constraints: $s_{ij}\geq0,\sum_i s_{ij}=1$; $h_{ij}\geq0,\sum_i h_{ij}=1$ | Doubly convex |

### Traps
- NMF: **BOTH** W and H must be non-negative (not just one)
- NMF is **NOT unique** — $Q$-ambiguity: $WH = (WQ^{-1})(QH)$ for any invertible $Q$ with non-neg sides
- NMF is NOT jointly convex — only convex in W given H fixed (justifies alternating minimization)
- ICA **CANNOT** separate Gaussian sources (CLT makes mixtures more Gaussian — no signal)
- ICA requires **non-Gaussian, statistically independent** sources
- PCA finds uncorrelated components; ICA finds **statistically independent** components (stricter)
- Whitening is a **required preprocessing** step for ICA (not optional)
- AA archetypes: on the **convex hull** (extremes), NOT interior centroids like k-means
- AA archetypes MUST be convex combinations of real data points ($Z = XS$, not arbitrary)
- AA vs k-means: AA → boundary/extremes; k-means → interior/centroids
- In sparse coding: $L_1$ causes **exact zeros** (sparsity); $L_2$ only shrinks
- Sparse coding Step 1 (fix W, update h) = **Lasso problem** (even though unsupervised)
- Must use **Speckled CV** (mask individual entries) for matrix methods — row holdout fails

---

## W12: Multiway Models — PARAFAC / Tucker / CORCONDIA

### Key Formulas
| Formula | What |
|---------|------|
| Tucker: $\mathcal{X} \approx \mathcal{G}\times_1 A\times_2 B\times_3 C$ | Full core tensor — cross-talk allowed |
| PARAFAC: $x_{ijk}\approx\sum_r a_{ir}b_{jr}c_{kr}$ | Sum of rank-1 tensors |
| Tucker matrix form: $X_{(1)}\approx A\,G_{(1)}(C\otimes B)^T$ | Uses **Kronecker** $\otimes$ |
| PARAFAC matrix form: $X_{(1)}\approx A(C\odot B)^T$ | Uses **Khatri-Rao** $\odot$ |
| $[\mathcal{X}\times_n M]_{(n)} = MX_{(n)}$ | N-mode multiplication |
| CORCONDIA $= 100(1-\|\mathcal{I}-\mathcal{G}\|_F^2/\|\mathcal{I}\|_F^2)$ | Core consistency diagnostic |
| FMS $= \sum_r \cos(a_r,\hat{a}_r)\cdot\cos(b_r,\hat{b}_r)\cdot\cos(c_r,\hat{c}_r)$ | Split-half stability |

### Traps
- PARAFAC is a **special case** of Tucker3 (Tucker with super-diagonal core)
- Tucker is **NOT unique** (can rotate core by any $Q$); PARAFAC **IS** essentially unique
- Tucker uses **Kronecker** $\otimes$; PARAFAC uses **Khatri-Rao** $\odot$ (not the same)
- Tucker ranks $P, Q, R$ can be **different** per mode; PARAFAC uses one rank $R$ for all modes
- Tucker → **data compression**; PARAFAC → **resolving physical/spectral profiles**
- PARAFAC components are **NOT nested** — changing $R$ changes ALL components (unlike PCA)
- CORCONDIA $\approx 100$ → good (core is nearly super-diagonal)
- CORCONDIA $\approx 0$ or negative → $R$ **too large** (NOT too small)
- CORCONDIA is specifically for **PARAFAC** model selection — not Tucker
- High $R$ can worsen CORCONDIA even if reconstruction improves
- Split-half: split DATA first → fit **separate models** to each half → compare via FMS
- FMS close to $R$ → stable; FMS $\ll R$ → $R$ too large

### Exam history
- 2022 Q16: CORCONDIA — answer was **E (None of the above)** — all options had subtle errors
- 2024 Q17: True statements about multiway models — answer was **A and D**

---

## CROSS-TOPIC TRAPS (from old exams)

### "Which methods are supervised?"
- Supervised: Ridge, Lasso, OLS, LDA, SVM, Logistic Regression, RF, Boosting, PLS
- **Unsupervised**: GMM, Autoencoder, K-means, Tucker, PCA, NMF, ICA, AA, PARAFAC
- 2024 Q1 answer was **E (None of the above)** — GMM, Autoencoder, K-means, Tucker all unsupervised

### "Which methods handle p >> n?"
- Work well: SVM (dual formulation), RF (random feature subsets), PCA (dimensionality reduction), Elastic Net, Lasso, Ridge
- Fails: OLS (singular $X^TX$), standard Logistic Regression (no regularization), CCA

### "Which methods can use the kernel trick?"
- Yes: SVM (naturally), Kernel PCA
- No: Boosting, RF (no natural kernel formulation)

### "Which methods are matrix factorization-based?"
- Yes: PCA (SVD), NMF, ICA ($X=AS$), AA ($X \approx XSH$)
- **NO**: K-means (distance-based, not factorization)

### "Which method is suitable as a base learner in bagging vs boosting?"
- Bagging/RF: **deep trees** (high variance, low bias → bagging reduces variance)
- Boosting: **stumps** (high bias, low variance → boosting reduces bias)
- KNN large K: low variance, low bias → bagging barely helps
- KNN small K: high variance, low bias → **good for bagging**

### "None of the above" alert
- Appeared correct in: 2022 Q16, 2024 Q1, 2024 Q13 — do NOT reflexively reject it

### LDA — Why is the boundary linear?
- Because **equal covariance assumption** cancels quadratic terms in log-posterior ratio
- NOT because of equal priors (that alone doesn't cause linearity)

### GMM vs LDA
- Both assume Gaussian class-conditionals
- LDA: **equal** covariance across classes; GMM: each component has its **own** covariance
- LDA: **supervised** (uses labels); GMM: typically **unsupervised**
- LDA: closed-form MLE; GMM: EM algorithm

---

## OPEN QUESTION Q22 TEMPLATE
*(Same dataset used in 2024 and 2025 — very likely to appear again)*

**Dataset**: 16 subjects × 3 conditions × 4 seasons = 192 observations. Predict activity from biosignals.

### a) Personalized model (predict for same individual)
- Use only that individual's data (12 observations: 3 conditions × 4 seasons)
- CV: leave-one-season-out (or leave-one-condition-out) within that individual
- Temporal structure: train on past seasons, test on future season

### b) Generalized model (predict for new individual)
- Use **leave-one-individual-out CV** — train on 15, test on held-out 16th
- Repeat for all 16 individuals
- Rationale: individual's data must NEVER appear in both training and test set

### Key talking points
- Mixing individuals across folds = **data leakage** (repeated measures)
- Personalized: more accurate for known individuals, limited training data
- Generalized: appropriate for clinical deployment (new patients never seen before)
- Both Q22 in 2024 and 2025 asked this exact setup

---

## FORMULA QUICK REFERENCE

| Need | Formula |
|------|---------|
| Fraction of variance by PC $k$ | $\lambda_k / \sum_j \lambda_j$ (eigenvalues of cov matrix — no squaring) |
| Fraction of variance from singular values | $\sigma_k^2 / \sum_j \sigma_j^2$ (square first) |
| NN parameter count | $(\text{inputs}\times\text{units})+\text{units}$ per layer, sum all layers |
| Bonferroni threshold | $\alpha / M$ |
| BH threshold for rank $i$ | $\frac{i}{m}q$ |
| FWER for $M$ tests | $1-(1-\alpha)^M$ |
| Ridge closed form | $(X^TX+\lambda I)^{-1}X^Ty$ |
| AdaBoost weight | $\alpha_m = \log\frac{1-\text{err}_m}{\text{err}_m}$ |
| Silhouette | $(b(i)-a(i))/\max\{a(i),b(i)\}$ |
| CORCONDIA | $100(1-\|\mathcal{I}-\mathcal{G}\|_F^2/\|\mathcal{I}\|_F^2)$ |
| Bagging variance floor | $\rho\sigma^2$ as $B\to\infty$ |
| Lasso soft threshold | $\text{sign}(x)(|x|-\lambda)_+$ |
| GMM mixing proportion | $\pi_j = (1/n)\sum_i\gamma_{ij}$ |
| Tucker (mode-1) | $X_{(1)}\approx A\,G_{(1)}(C\otimes B)^T$ |
| PARAFAC (mode-1) | $X_{(1)}\approx A(C\odot B)^T$ |
| OLS | $(X^TX)^{-1}X^Ty$ |
| EPE decomposition | $\sigma^2 + \text{Bias}^2 + \text{Variance}$ |
