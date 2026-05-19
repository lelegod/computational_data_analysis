# CDA 02582 — Exam Tips & Tricks
> Based on 2022, 2024, 2025 exam patterns. Speed-optimized: read an option → know instantly if it's right or wrong.

---

## QUICK NAVIGATION
- [1. What Will Likely Appear (2026 Prediction)](#1-what-will-likely-appear-2026-prediction)
- [2. The Meta-Strategy for MCQ](#2-the-meta-strategy-for-mcq)
- [3. Instant True/False Option Patterns](#3-instant-truefalse-option-patterns)
- [4. Cross-Topic Quick Filters](#4-cross-topic-quick-filters)
- [5. Topic-by-Topic Speed Rules](#5-topic-by-topic-speed-rules)
- [6. The "None of the Above" Signal](#6-the-none-of-the-above-signal)
- [7. Known Official Errors](#7-known-official-errors)
- [8. Q21 Open Question — Writing Template](#8-q21-open-question--writing-template)
- [9. Q22 Open Question — Instant Answer](#9-q22-open-question--instant-answer)

---

## 1. What Will Likely Appear (2026 Prediction)

### Topics tested in ALL 3 past exams (3/3) — almost certain:
| Topic | What to know cold |
|-------|------------------|
| Bias-Variance / EPE | 3 terms; $\sigma^2$ never changes; training error always goes down with complexity |
| Ridge vs Lasso | Ridge: closed form, no zeros, L2. Lasso: no closed form, exact zeros, L1 |
| Cross-validation design | IID assumption; normalize inside folds; nested CV for unbiased assessment |
| Multiple testing (BH / Bonferroni) | BH = FDR control (more power). Bonferroni = FWER control (stricter) |
| LDA | Linear because equal covariance cancels quadratic terms. Probabilistic |
| GMM / Clustering | Unsupervised; EM algorithm; soft assignments; K-means is special case of GMM |
| SVM / Kernel | Linearity depends on kernel, not on dual formulation. RBF = infinite-dim |
| Archetypal Analysis / NMF | AA: extremes on convex hull. NMF: W,H ≥ 0, not unique |

### Topics tested in 2/3 — likely:
| Topic | Last seen |
|-------|----------|
| Random Forest / Bagging | 2022 + 2024 — not 2025 |
| Neural Networks (parameter count) | 2024 + 2025 — **count the parameters** |
| ICA | 2024 + 2025 |
| Multiway models (Tucker/PARAFAC/CORCONDIA) | 2022 + 2024 — not 2025 |

### Q21 open question prediction:
- 2022: Random Forest | 2024: ICA | 2025: LDA vs GMM
- Topics NOT yet used as Q21: SVM, Boosting, PCA/PLS/CCA, PARAFAC, Clustering, Neural Networks
- **High probability**: SVM, Gradient Boosting, or PCA/PLS/CCA

### Q22 open question prediction:
- **Near certain: same wearables dataset** (used 2024 + 2025 unchanged)
- Know cold: LOSO-CV (personalized) vs LOIO-CV (generalized)

---

## 2. The Meta-Strategy for MCQ

### Scoring
2025 format: single correct answer. No stated penalty for wrong — but check the paper. If no penalty: answer everything. If penalty: only answer if confidence > ~20%.

### Reading options
1. **Read the question stem first** — identify the topic (1–2 seconds)
2. **Apply topic-specific rules** — not general reasoning
3. **Eliminate reversed options first** — many wrong options are exact reversals of the truth
4. **Watch for "always", "never", "only"** — these often signal a wrong answer
5. **Watch for "None of the above"** — it's correct surprisingly often (2022 Q16, 2024 Q1, 2024 Q13)

### The reversal trap (most common wrong option type)
Exam writers love flipping true statements. Examples that appear across exams:
- "Ridge performs variable selection" → reversed (Lasso does)
- "Boosting uses deep trees" → reversed (Bagging does)
- "GMM is supervised" → reversed (unsupervised)
- "Large K in KNN = high variance" → reversed (small K = high variance)
- "E-step in EM updates parameters" → reversed (M-step does)
- "K-medoids uses cluster means" → reversed (K-means does)
- "CORCONDIA close to 0 is good" → reversed (close to 100 is good)
- "PARAFAC is not unique" → reversed (PARAFAC IS essentially unique; Tucker is NOT)

---

## 3. Instant True/False Option Patterns

Read the option → immediately flag as ✓ or ✗.

### Always TRUE
- "Irreducible error is not affected by model complexity" → ✓
- "Ridge has a closed-form solution" → ✓
- "Lasso can set coefficients exactly to zero" → ✓
- "OLS is unbiased (Gauss-Markov)" → ✓
- "BIC penalizes more than AIC for large N" → ✓
- "LDA decision boundary is linear" → ✓
- "LDA assumes equal covariance across classes" → ✓
- "Logistic regression is discriminative (models P(G|X) directly)" → ✓
- "GMM gives soft (probabilistic) cluster assignments" → ✓
- "K-medoids is more robust to outliers than K-means" → ✓
- "PARAFAC is essentially unique" → ✓
- "Tucker3 is NOT unique" → ✓
- "PARAFAC is a special case of Tucker3 with super-diagonal core" → ✓
- "SVM has no probabilistic model" → ✓
- "RBF kernel maps to infinite-dimensional space" → ✓
- "Bagging reduces variance but not bias" → ✓
- "RF bias = individual tree bias (deep trees preferred)" → ✓
- "Boosting can overfit (especially with noisy data)" → ✓
- "OOB error ≈ LOOCV error (unbiased)" → ✓
- "ICA cannot separate Gaussian sources" → ✓
- "NMF solutions are not unique ($Q$-ambiguity)" → ✓
- "AA archetypes lie on the convex hull of the data" → ✓
- "Ward linkage requires Euclidean distance" → ✓
- "AIC is asymptotically equivalent to LOOCV (not k-fold CV)" → ✓
- "Preprocessing must happen inside the CV loop" → ✓

### Always FALSE
- "Ridge sets some coefficients exactly to zero" → ✗
- "Lasso has a closed-form solution" → ✗
- "Ridge performs variable selection" → ✗
- "Increasing λ in Ridge increases df(λ)" → ✗ (df DECREASES)
- "Bagging reduces bias" → ✗
- "Adding more trees in RF causes overfitting" → ✗
- "RF reduces bias compared to a single tree" → ✗
- "Boosting uses deep/unpruned trees" → ✗ (stumps)
- "Boosting trees are independent / parallelizable" → ✗ (sequential)
- "LDA has per-class covariance matrices" → ✗ (QDA does)
- "LDA has a quadratic decision boundary" → ✗
- "Logistic regression has a closed-form solution" → ✗
- "GMM is supervised" → ✗
- "Autoencoder is supervised" → ✗
- "K-means guarantees the global optimum" → ✗
- "Ward linkage works with any distance metric" → ✗ (Euclidean only)
- "K-medoids uses cluster means (not actual data points)" → ✗
- "Single linkage produces compact clusters" → ✗ (complete linkage does)
- "E-step in EM updates parameters" → ✗ (M-step does)
- "Tucker3 is unique" → ✗
- "PARAFAC is not unique" → ✗
- "CORCONDIA close to 0 means good fit" → ✗ (means R too large)
- "CORCONDIA can be used for Tucker3" → ✗ (only for PARAFAC)
- "PCA maximizes correlation with y" → ✗ (maximizes variance)
- "PLS with M=p is regularized" → ✗ (M=p → same as OLS)
- "CCA maximizes variance" → ✗ (maximizes correlation only)
- "CCA works fine when p > n" → ✗ (singular covariance)
- "Sparse PCA scores remain uncorrelated after thresholding" → ✗
- "The kernel trick makes SVM training faster" → ✗
- "The dual formulation of SVM makes it nonlinear" → ✗ (kernel choice does)
- "Vanishing gradient is a problem for CNNs" → ✗ (it's for RNNs)
- "Transformers process sequences one-by-one (sequential)" → ✗ (self-attention is parallel)
- "ICA and PCA find the same components" → ✗
- "Row-holdout CV works for NMF/sparse coding" → ✗ (use Speckled CV)
- "L2 regularization causes sparsity (exact zeros)" → ✗ (L1 does)
- "Normalizing data before CV folds is correct" → ✗ (leakage)
- "AIC is asymptotically equivalent to k-fold CV" → ✗ (LOO-CV, not k-fold)
- "BIC is better than AIC for small N" → ✗ (BIC over-simplifies at small N)
- "Clustering only works when real clusters exist" → ✗ (always produces groups)
- "AIC/BIC can be used to choose K in K-means" → ✗ (no likelihood; use silhouette/gap)
- "Proximity plots in RF measure closeness of variables" → ✗ (closeness of OBSERVATIONS)

---

## 4. Cross-Topic Quick Filters

When the question asks "which of these methods does X", apply these tables instantly.

### Supervised vs Unsupervised
| Supervised (needs labels y) | Unsupervised (no labels) |
|-----------------------------|--------------------------|
| OLS, Ridge, Lasso, Elastic Net | PCA, Sparse PCA |
| LDA, QDA, RDA, Logistic Regression | K-means, K-medoids, Hierarchical |
| SVM | GMM |
| Random Forest, Bagging, Boosting | NMF, ICA, Archetypal Analysis |
| Neural Networks (MLP, CNN) | Autoencoder |
| PLS | Tucker3, PARAFAC |
| | CCA (two-matrix, no label) |

**Trap**: GMM is unsupervised. Autoencoder is unsupervised. Tucker/PARAFAC are unsupervised.

### Works when $p \gg n$
| Works | Fails |
|-------|-------|
| SVM (dual uses N, not p) | OLS (singular $X^TX$) |
| Random Forest (random feature subsets) | Logistic Regression (unregularized) |
| PCA (SVD-based, always works) | CCA (needs to invert $\Sigma_{XX}$) |
| Ridge (adds $\lambda I$ → always invertible) | LDA/QDA (singular $\hat{\Sigma}$ without regularization) |
| Lasso, Elastic Net | |
| RDA (regularized) | |

### Kernel trick applicable
| Yes | No |
|-----|----|
| SVM (natural) | Boosting |
| Kernel PCA | Random Forest |
| | Neural Networks (use feature learning instead) |

### Has a closed-form solution
| Yes | No |
|-----|-----|
| OLS: $(X^TX)^{-1}X^Ty$ | Lasso (L1 non-differentiable at 0) |
| Ridge: $(X^TX+\lambda I)^{-1}X^Ty$ | Logistic Regression (Newton-Raphson) |
| LDA (class means + pooled covariance) | GMM (EM algorithm) |
| PCA (eigendecomposition of $\Sigma$) | Neural Networks (gradient descent) |

### Matrix factorization methods
Yes: PCA (SVD), NMF ($X\approx WH$), ICA ($X=AS$), AA ($X\approx XSH$), Tucker, PARAFAC
**No: K-means** (distance-based clustering, not factorization)

---

## 5. Topic-by-Topic Speed Rules

### Bias-Variance / EPE
- Three terms always: Bias² + Variance + $\sigma^2$ (irreducible)
- Large λ → high bias, low variance, low df
- Small λ → low bias, high variance, high df
- Training error always decreases with complexity; test error U-shapes
- $\sigma^2$ = property of data, not model. Nothing can reduce it.

### Ridge vs Lasso
| You see | Answer |
|---------|--------|
| "closed form" | Ridge |
| "exact zeros / variable selection / sparse" | Lasso |
| "shrinks but doesn't zero" | Ridge |
| "path algorithm / LARS" | Lasso |
| "correlated predictors — which is better?" | Ridge (or Elastic Net) |
| "p > n, need variable selection" | Lasso or Elastic Net |

### Cross-Validation
- **Normalize INSIDE folds** (not before) — outside = leakage
- **Grouped/dependent obs**: keep groups in same fold (never split a person/time-series across train/test)
- **1-SE rule**: choose most regularized model within 1 SE of minimum — NOT the minimum itself
- **Nested CV**: outer loop = assessment, inner loop = selection → unbiased after tuning
- **IID violation** = dependent observations = standard CV invalid → use group/structured CV

### Multiple Testing
| You see | Answer |
|---------|--------|
| "controls any false positive" | Bonferroni (FWER) |
| "allows some false positives, maximizes discoveries" | BH (FDR) |
| "more conservative" | Bonferroni |
| "more powerful / more discoveries" | BH |
| FWER formula | $1-(1-\alpha)^M$ |
| BH threshold for rank $i$ | $\frac{i}{m}q$ |

### LDA
- Linear because equal covariance → quadratic terms cancel in log-posterior ratio
- Equal priors alone ≠ linear boundary
- Sensitive to outliers (uses means/covariance directly)
- Probabilistic (Gaussian class-conditionals + Bayes)
- $\alpha=0$ in RDA = LDA; $\alpha=1$ in RDA = QDA

### Clustering
| You see | Answer |
|---------|--------|
| "uses actual data points as centers" | K-medoids |
| "uses means as centers" | K-means |
| "dendrogram, no need to specify K" | Hierarchical |
| "soft probabilistic assignments" | GMM |
| "hard assignments" | K-means or K-medoids |
| "robust to outliers" | K-medoids |
| "requires Euclidean distance" | K-means, Ward linkage |
| "any distance metric" | K-medoids |
| "choose K for GMM" | AIC or BIC |
| "choose K for K-means" | Silhouette or Gap statistic (NOT AIC/BIC) |
| "chains/elongated clusters" | Single linkage |
| "compact clusters" | Complete linkage |
| "E-step" | Computes soft assignments $\gamma_{ij}$ (not parameters) |
| "M-step" | Updates parameters $\mu_j, \Sigma_j, \pi_j$ |

### RF vs Boosting
| Feature | Random Forest | Boosting |
|---------|--------------|---------|
| Tree type | Deep (unpruned) | Shallow (stumps) |
| Trees independent? | Yes → parallelizable | No → sequential |
| What it reduces | Variance | Bias (primarily) |
| Can overfit? | No (more trees = better) | Yes (noisy data) |
| KNN good base learner? | Small K (high variance) | No |
| Stump good base learner? | No (too high bias) | Yes |

### SVM
- Margin width = $2/\|\beta\|$ — maximizing margin = minimizing $\|\beta\|$
- $\beta$ is **orthogonal** to the hyperplane (not parallel)
- Labels are ±1, not 0/1
- Support vectors: on the margin, $\alpha_i > 0$
- Safe points: beyond margin, $\alpha_i = 0$ (delete them → boundary unchanged)
- Dual formulation alone ≠ nonlinear (kernel choice makes it nonlinear)
- No probabilistic model
- RBF kernel = infinite-dimensional space

### PCA / PLS / CCA
| You see | Answer |
|---------|--------|
| "maximizes variance, ignores y" | PCA |
| "maximizes covariance with y, supervised" | PLS |
| "maximizes correlation between two matrices" | CCA |
| "fails when p > n" | CCA |
| "PLS with M=p" | Same as OLS (no regularization) |
| "EVD vs SVD give different loadings?" | No — same V |
| "uncorrelated components" | PCA or PLS (both) |
| "Sparse PCA scores uncorrelated after thresholding?" | No — destroyed |

### PARAFAC / Tucker
| You see | Answer |
|---------|--------|
| "which is a special case of the other?" | PARAFAC is special case of Tucker (super-diagonal core) |
| "which is unique?" | PARAFAC (Tucker is NOT) |
| "which for physical profiles / spectra?" | PARAFAC |
| "which for compression?" | Tucker |
| "product used in matrix form" | Tucker: Kronecker ⊗ / PARAFAC: Khatri-Rao ⊙ |
| "CORCONDIA = 100" | Core ≈ super-diagonal → R appropriate |
| "CORCONDIA = 0 or negative" | R too large |
| "CORCONDIA applies to Tucker?" | No — only PARAFAC |

### Neural Networks
- Parameter count: $(n_\text{in} \times n_\text{units} + n_\text{units})$ per layer
- MSE loss ↔ Gaussian assumption. BCE loss ↔ Bernoulli assumption
- Vanishing gradient: sigmoid multiplied across layers → exponential decay → problem for RNNs, not CNNs
- Autoencoder: UNSUPERVISED. Reconstruction loss = $\|x - \hat{x}\|^2$
- Transformers: self-attention = fully parallelizable (not sequential like RNN)
- Gradients go BACKWARD. Activations go FORWARD.

### NMF / ICA / AA
| You see | Answer |
|---------|--------|
| "parts-based additive representation" | NMF |
| "extremes / boundary prototypes" | AA |
| "separates statistically independent signals" | ICA |
| "cocktail party problem" | ICA |
| "kurtosis = 0 → Gaussian → ICA?" | ICA FAILS for Gaussian sources |
| "NMF unique?" | NOT unique ($Q$-ambiguity) |
| "NMF convex?" | NOT jointly convex — only in one factor given the other |
| "AA vs K-means: where are the prototypes?" | AA: convex hull (extremes). K-means: interior |
| "sparse coding CV" | Speckled CV (mask entries) — NOT row holdout |

---

## 6. The "None of the Above" Signal

**E (None of the above) was the correct answer in 3 past questions:**
- 2022 Q16 — CORCONDIA: none of options A–D correctly described CORCONDIA
- 2024 Q1 — Supervised methods: GMM, Autoencoder, K-means, Tucker — all unsupervised
- 2024 Q13 — LARS vs Coordinate Descent: no option cleanly described both correctly

**When to pick E**: If you've applied your rules and every option A–D has a clear flaw, trust E.
**Never skip E** just because "there's always a real answer". This exam deliberately uses it.

---

## 7. Known Official Errors

These exact questions may not reappear, but knowing the correct answer builds confidence.

### 2022 Q20 — Base learners in boosting
- **Official answer (grid error)**: A + E (contradictory)
- **Correct answer**: **C — any classification or regression tree**
- Shallow trees (stumps) are the canonical boosting base learner

### 2024 Q11 — RF proximity plots
- **Official answer includes C**: "proximity plots measure closeness of variables"
- **C is WRONG**: proximity plots measure closeness of **observations**, not variables
- Correct: **A** (Gini VI = aggregated Gini impurity reduction) + **D** (deep trees preferred)

### 2022 Q9 — Why not penalize the intercept
- **Official answer is ambiguous**
- **Most defensible**: **A** — "Penalizing the intercept introduces bias without variance reduction"
- D ("lower EPE") is a consequence of A, not an independent reason

---

## 8. Q21 Open Question — Writing Template

20 points. Every answer needs this structure:

1. **State the model** — one equation, one sentence
2. **Explain the mechanism** — WHY does each step work, not just WHAT it does
3. **Key properties** — bias/variance/uniqueness/complexity
4. **Compare to alternatives** — name 1–2 competitors and the distinguishing property
5. **Limitations** — when does it fail, edge cases

**Where marks come from**: correct objective function · explaining WHY · comparison · formula · edge case behavior

**Common mark losses**:
- Saying "minimizes error" without specifying which loss
- Listing algorithm steps without explaining what each achieves
- Forgetting model assumptions (e.g., equal covariance for LDA)
- Confusing variance reduction (bagging) with bias reduction (boosting)

**Topics not yet tested as Q21** (higher probability for 2026):
SVM, Gradient Boosting, PCA/PLS/CCA, PARAFAC/Tucker, K-means/GMM, Neural Networks

---

## 9. Q22 Open Question — Instant Answer

**Near-certain**: same 16 subjects × 3 activities × 4 seasons = 192 observations wearables dataset.

### Part a) Personalized model — predict new season for KNOWN individual
- **CV: Leave-One-Season-Out** within one subject (4 folds)
- Train: 3 seasons (9 obs) → Test: 1 season (3 obs)
- Captures intra-individual variation only
- EPE is LOWER

### Part b) Generalized model — predict for NEW individual
- **CV: Leave-One-Individual-Out** (16 folds)
- Train: 15 subjects (180 obs) → Test: 1 subject (12 obs)
- Never split one person across train/test (IID violation → leakage)
- EPE is HIGHER (must handle between-individual variance)

### Why standard CV fails
Random splits let the model see the test subject's data during training → learns their personal physiology → data leakage → EPE is optimistically biased.

### Formula to write
$$\text{EPE}_\text{pers} = E_{x,y|i_\text{fixed}}[\mathcal{L}(y,\hat{f}_i(x))]$$
$$\text{EPE}_\text{gen} = E_{i_\text{new}}[E_{x,y|i_\text{new}}[\mathcal{L}(y,\hat{f}(x))]]$$
$$\text{EPE}_\text{gen} > \text{EPE}_\text{pers} \text{ always}$$

### 30-second recognition table
| Signal | Variant | CV Method |
|--------|---------|-----------|
| "predict for new patient" | Generalized | LOIO-CV (16 folds) |
| "predict for same individual" | Personalized | LOSO-CV (4 folds) |
| "multiple measurements per person" | IID violation | Group K-fold by person |
| "tensor / how many components" | PARAFAC | CORCONDIA + split-half FMS |
