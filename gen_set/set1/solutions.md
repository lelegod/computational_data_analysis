# Practice Set 1 — Solutions

## Multiple Choice Answers

| Q | Correct | Explanation |
|---|---------|-------------|
| 1 | A, C, D | **A** is correct: increasing complexity lowers bias and raises variance — this is the fundamental bias-variance tradeoff (Week 1 EPE decomposition). **B** is wrong: σ² is the irreducible noise from the data-generating process; no model can eliminate it. **C** is correct: variance is formally Var(f̂) = E[(f̂ − E[f̂])²], measuring fluctuation across training sets. **D** is correct: OLS is unbiased — E[β̂_OLS] = β — so Bias = E[f̂] − f = 0 under correct model specification. **E** is wrong because A, C, D are all correct. |
| 2 | A, C | **A** is correct: the Ridge estimator β̂_ridge = (XᵀX + λI)⁻¹Xᵀy is always invertible for λ > 0 because XᵀX + λI is positive definite. **B** is wrong: Ridge shrinks coefficients toward zero but never exactly to zero — that is Lasso's property (L₁ geometry has corners; L₂ sphere does not). **C** is correct: as λ → 0, df → p (OLS); as λ → ∞, df → 0; the trace decreases monotonically. **D** is wrong: Ridge introduces bias for any λ > 0 — E[β̂_ridge] ≠ β in general. |
| 3 | A, B, C | **A** is correct: for Gaussian errors AIC(λ) = err(λ) + 2(d(λ)/N)σ̂²_e, which matches the Cp formula exactly. **B** is correct: BIC penalty = log(N)·d/N; AIC penalty = 2d/N — the BIC coefficient log(N) grows with N while AIC's coefficient 2 is fixed. **C** is correct: Stone (1977) showed AIC is asymptotically equivalent to leave-one-out cross-validation. **D** is wrong: for large N, BIC penalizes MORE per parameter than AIC (log(N) > 2 for N ≥ 8), so BIC selects SIMPLER models than AIC, not more complex ones. |
| 4 | A, C, D | **A** is correct: this is the standard Lasso objective with the L₁ norm ‖β‖₁ = Σ|βⱼ|. **B** is wrong: the L₁ norm is not differentiable at β = 0, so no closed-form solution exists — LARS or coordinate descent must be used. **C** is correct: the L₁ constraint region is a diamond in 2D with corners on the coordinate axes; the RSS ellipsoid typically first contacts the diamond at a corner where one coordinate is zero. **D** is correct: Lasso selects at most min(n, p) variables; when p > n, at most n variables can be non-zero. |
| 5 | A, B, C | **A** is correct: Bonferroni threshold = α/M = 0.05/50 = 0.001. **B** is correct: FWER without correction = 1 − (1−0.05)⁵⁰ = 1 − (0.95)⁵⁰ ≈ 0.923 — a 92.3% chance of at least one false rejection. **C** is correct: Bonferroni controls FWER at level α, ensuring P(any false rejection) ≤ 0.05. **D** is wrong: Bonferroni has LOWER power than BH (Benjamini-Hochberg). BH allows a controlled proportion of false discoveries and therefore rejects more hypotheses (more power). |
| 6 | B, C | **A** is wrong: the description is reversed. In nested CV, the INNER loop handles model selection (hyperparameter tuning) and the OUTER loop handles model assessment (estimating generalization error). **B** is correct: nested CV audits the full pipeline — including the selection step — providing an unbiased estimate of how well the "select-then-train" procedure generalizes. **C** is correct: a large gap between inner-loop (optimistic, selected) error and outer-loop (honest) error signals that the model is overfitting to the hyperparameter selection noise. **D** is wrong: nested CV is especially important when hyperparameters are tuned; AIC/BIC are not a substitute because they also suffer from selection-induced bias if chosen from a set of models. |
| 7 | A, C, D | **A** is correct: this is the core statement of the curse of dimensionality — the volume of the space grows exponentially, so fixed N becomes sparse. **B** is wrong: in the elastic net, α = 0 gives pure Ridge and α = 1 gives pure Lasso (not the other way around). **C** is correct: the elastic net penalty λ[(1/2)(1−α)‖β‖²₂ + α‖β‖₁] combines L₂ (grouping correlated predictors) with L₁ (sparsity/variable selection). **D** is correct: Donoho (2000) listed the manifold hypothesis — that real data often lies on a low-dimensional manifold embedded in high-dimensional space — as one of the three "blessings" of dimensionality. |
| 8 | A, C, D | **A** is correct: Gini index G = Σₖ p_mk(1−p_mk) = 0 when all observations in the node belong to a single class (p_mk = 1 for one class). **B** is wrong: misclassification rate is NOT differentiable and is NOT the preferred criterion for growing trees — it is insensitive to probability shifts within the majority class. Gini index and cross-entropy are preferred for growing. **C** is correct: both Gini and cross-entropy respond to any shift in class probabilities, whereas misclassification rate does not change as long as the majority class is unchanged. **D** is correct: in regression trees, the prediction in region R_j is c_j = mean(yᵢ : xᵢ ∈ R_j). |
| 9 | A, C, D | **A** is correct: C_α(T) = R(T) + α|T| where R(T) = total node impurity and |T| = number of terminal nodes. **B** is wrong: when α = 0, the full (unpruned) tree T₀ is selected because there is no penalty for complexity — not the root node. The root is selected when α is very large. **C** is correct: increasing α imposes a larger per-leaf penalty, causing the algorithm to prefer fewer leaves (smaller trees). **D** is correct: the standard CART procedure grows T₀, finds the sequence of subtrees via weakest-link pruning, then uses K-fold CV to select α*. |
| 10 | A, C, D | **A** is correct: the limiting variance as B → ∞ is ρσ², since the term (1−ρ)σ²/B → 0. This is the irreducible floor set by inter-tree correlation. **B** is wrong: bagging does NOT reduce bias. The bias of the bagged predictor equals the bias of any single tree, since E[(1/B)Σŷ_b − y] = E[ŷ_b − y]. **C** is correct: P(observation i not in a bootstrap sample) = (1 − 1/N)^N → 1/e ≈ 0.368, so approximately 63.2% are included. **D** is correct: for each training observation, predictions are made only by trees for which that observation was OOB; these OOB predictions give a free CV-like error estimate. |
| 11 | A, B, C, D | **A** is correct: the default for classification is m = ⌊√p⌋, as stated in the lecture. **B** is correct: when all p features are considered at every split, no random subsampling occurs, which reduces to standard bagging. **C** is correct: by excluding strong predictors from some splits, random feature subsampling prevents trees from always splitting on the same variable, lowering pairwise correlation ρ and thus reducing the variance floor ρσ². **D** is correct: RF uses deep trees (low bias, high variance — bagging then reduces variance) while gradient boosting uses shallow trees/stumps (high bias, low variance — boosting then reduces bias sequentially). |
| 12 | A, B | **A** is correct: when err_m = 0.5, α_m = log[(1−0.5)/0.5] = log(1) = 0, so the weak learner is ignored. **B** is correct: boosting reduces bias by sequentially targeting errors (hard cases); using high-bias weak learners (stumps) ensures each step corrects a specific weakness without already being complex. **C** is wrong: the exponential loss grows FASTER than binomial deviance for misclassified observations (negative margin), making AdaBoost MORE sensitive to label noise, not less robust. **D** is wrong: forward stagewise additive modelling fixes previously fitted trees — once added, their weights are never adjusted; only new (β_m, b_m) pairs are added. |
| 13 | A, C, D | **A** is correct: margin C = 1/‖β‖, so minimizing (1/2)‖β‖² maximizes C — this is exactly the canonical SVM formulation. **B** is wrong: the relationship is reversed. KKT complementary slackness states αᵢ[yᵢ(xᵢᵀβ + β₀) − 1] = 0. Points with a large margin (far from boundary) have bracket > 0, so αᵢ = 0 (non-support vectors). Support vectors are ON the margin (bracket = 0) and have αᵢ > 0. **C** is correct: the RBF kernel corresponds to the dot product in an infinite-dimensional RKHS — this is the fundamental mathematical fact underlying the kernel trick. **D** is correct: in the dual problem, max_α Σαᵢ − (1/2)ΣΣ αᵢαⱼyᵢyⱼ⟨xᵢ,xⱼ⟩, data appear only as inner products ⟨xᵢ,xⱼ⟩, enabling substitution with any kernel K(xᵢ,xⱼ). |
| 14 | A, B, C | **A** is correct: eigenvalues of the covariance matrix are λₖ = dₖ²/(n−1). So the squared singular values are d₁²=64, d₂²=36, d₃²=16, d₄²=4, d₅²=1; total = 121; first two PCs explain (64+36)/121 = 100/121 ≈ 82.6%. **B** is correct: the right singular vectors V from X = UDVᵀ are identical to the eigenvectors of XᵀX/(n−1), which is the sample covariance matrix — this is the standard relationship between SVD and EVD of the covariance. **C** is correct: PCA on unscaled data is dominated by features with large variance (e.g., features in kg vs. mm); standardizing to unit variance ensures all features contribute equally. **D** is wrong: the description is reversed. PLS uses the response y to guide dimension reduction (it maximizes covariance between Xv and y), while PCA maximizes variance in X alone (unsupervised). |
| 15 | A, C, D | **A** is correct: the K-means objective function is exactly this within-cluster sum of squares. **B** is wrong: K-means is NOT guaranteed to find the global optimum — it converges to a local minimum that depends on initialization. Multiple random restarts are recommended. **C** is correct: s(i) ∈ [−1, 1] by definition (since a(i) and b(i) are non-negative distances); s(i) near +1 means a(i) << b(i), so the point is much closer to its own cluster than to the next nearest. **D** is correct: the gap statistic G(K) = log(U_k) − log(W_k) compares actual within-cluster dispersion W_k to U_k from simulated uniform reference data; a large gap signals real cluster structure. |
| 16 | A, B, C | **A** is correct: this is the E-step formula — Bayes' rule applied to compute the posterior cluster membership probability, combining prior πⱼ with Gaussian likelihood N(xᵢ; μⱼ, Σⱼ). **B** is correct: the M-step mean update is a weighted average of data points, weighted by soft assignments γᵢⱼ — identical to standard MLE but with fractional counts. **C** is correct: K-means can be derived as a limiting case of GMM where all Σⱼ = ε²I → 0 (identical spherical covariances) and assignments are hard (γᵢⱼ → 0 or 1). **D** is wrong: GMM likelihood is not concave — the EM algorithm finds a local maximum, not the guaranteed global maximum. Multiple restarts are needed. |
| 17 | A, C, D | **A** is correct: (5×3+3) + (3×3+3) + (3×2+2) = 18 + 12 + 8 = 38 parameters total. **B** is correct (but check — this is binary CE, applicable for 2-class problems): binary cross-entropy −Σ[y log ŷ + (1−y)log(1−ŷ)] is derived from the Bernoulli negative log-likelihood. For the 2-output softmax here this corresponds to categorical CE, but the formula is derived the same way. **C** is correct: vectorized backpropagation computes δ^(ℓ) by transposing the weight matrix and multiplying with the upstream error, then element-wise multiplying with the local activation derivative. **D** is correct: RNNs propagate error back through time, and gradients of distant time steps involve products of many Jacobians, causing vanishing (or exploding) gradients — LSTM and GRU use gating mechanisms to address this. |
| 18 | A, B, D | **A** is correct: NMF forces W ≥ 0 and H ≥ 0, so all components are additive — no negative entries means no cancellation, producing a parts-based representation (e.g., face parts, topic words). **B** is correct: ICA requires non-Gaussian sources because the Central Limit Theorem states that mixtures become more Gaussian; ICA reverses this by maximizing non-Gaussianity. For Gaussian sources, the mixing matrix A is unidentifiable. **C** is wrong: NMF solutions are NOT unique. For any invertible Q with WQ⁻¹ ≥ 0 and QH ≥ 0, (WQ⁻¹)(QH) is an equally valid factorization. **D** is correct: whitening (sphering) transforms the data so its covariance is the identity matrix, reducing the ICA problem from finding arbitrary W to finding an orthogonal matrix (rotation), which is far simpler. |
| 19 | A, B, C | **A** is correct: archetypes lie on or near the convex hull — they are extreme points, not averages. This is the defining feature that distinguishes AA from K-means (centroids = interior points) and PCA (directions of maximum variance, not extreme profiles). **B** is correct: sparse coding uses an overcomplete dictionary W with K > I atoms, and represents each data point as Wh where h is sparse (most entries zero). This is exactly the Lasso problem in the coding step. **C** is correct: the S matrix has columns summing to 1 with s_{ij} ≥ 0, ensuring each archetype Z = XS is a convex combination (weighted average) of real data points, so archetypes cannot lie outside the data cloud. **D** is wrong: AA and K-means find fundamentally different solutions. K-means places centroids at interior cluster means (inside the data cloud), while AA places archetypes on the convex hull (extreme boundary). Their solutions generally differ significantly. |
| 20 | A, C, D | **A** is correct: PARAFAC = Tucker3 with G = super-diagonal identity tensor I^{R×R×R}, where g_{rrr} = 1 and all other entries are 0. This eliminates all cross-component interactions. **B** is wrong: the uniqueness relationship is the opposite. PARAFAC IS essentially unique (up to sign and permutation of components), under Kruskal's conditions — this is a major practical advantage. Tucker3 is NOT unique due to rotational freedom: G ×₁ A can be rewritten as (G ×₁ Q) ×₁ (AQ⁻¹) for any invertible Q. **C** is correct: CORCONDIA = 100 · (1 − ‖I − G‖²_F / ‖I‖²_F). Close to 100 means the fitted core G is nearly super-diagonal, confirming the PARAFAC model structure is appropriate for the chosen R. **D** is correct: Tucker3 unfolded form X_{(1)} ≈ A G_{(1)} (C ⊗ B)ᵀ uses the Kronecker product ⊗ (all outer products between columns), while PARAFAC X_{(1)} ≈ A (C ⊙ B)ᵀ uses the Khatri-Rao product ⊙ (column-wise Kronecker — only matching columns). |

---

## Open Question Solutions

---

### Q21 Solution: Ridge Regression — Derivation, Geometry, and Model Selection

**Part (a) — Closed-form Ridge derivation [3 pts]**

Start from the penalized objective:

`J(β) = (Y − Xβ)ᵀ(Y − Xβ) + λβᵀβ`

Expand:

`J(β) = YᵀY − 2βᵀXᵀY + βᵀXᵀXβ + λβᵀβ`

Take the derivative with respect to β:

`∂J/∂β = −2XᵀY + 2XᵀXβ + 2λIβ = 0`

Key derivative rules used:
- ∂/∂β (βᵀAβ) = (A + Aᵀ)β = 2Aβ when A is symmetric (XᵀX is symmetric)
- ∂/∂β (bᵀβ) = b

Set derivative to zero and rearrange:

`2XᵀXβ + 2λIβ = 2XᵀY`  
`(XᵀX + λI)β = XᵀY`

**Ridge estimator:**

`β̂_ridge = (XᵀX + λI)⁻¹ XᵀY`

This always exists because XᵀX is positive semi-definite, and adding λI (with λ > 0) makes XᵀX + λI strictly positive definite, hence invertible.

---

**Part (b) — Geometric explanation of sparsity [3 pts]**

The penalized form of Ridge can be rewritten as a constrained problem:

- **Ridge (L₂):** min_β RSS subject to ‖β‖²₂ ≤ s
- **Lasso (L₁):** min_β RSS subject to ‖β‖₁ ≤ s

The RSS contours form ellipses centered at the OLS solution. The solution is found where the RSS ellipse first contacts the constraint region.

- **Lasso constraint region (L₁):** In 2D, this is a **diamond** with corners on the coordinate axes. The RSS ellipse will typically first contact the diamond at a **corner**, where one coordinate equals exactly zero → sparse solution.
- **Ridge constraint region (L₂):** In 2D, this is a **circle** (sphere in higher dimensions). The circle has no corners. The RSS ellipse contacts the sphere at a smooth point that is almost never exactly on an axis → coefficients are never exactly zero.

This geometric difference is why Lasso performs variable selection (sets coefficients to zero) while Ridge only shrinks them.

---

**Part (c) — Effective degrees of freedom [2 pts]**

`df(λ) = trace(X(XᵀX + λI)⁻¹Xᵀ)`

**As λ → 0:** The ridge smoother matrix S_λ = X(XᵀX + λI)⁻¹Xᵀ approaches the OLS hat matrix S = X(XᵀX)⁻¹Xᵀ. The trace of the OLS hat matrix equals p (the number of predictors). Therefore df(λ) → p.

**Interpretation:** With no regularization, the model uses all p degrees of freedom — equivalent to fitting p free parameters.

**As λ → ∞:** The penalty dominates; all coefficients are shrunk toward zero. S_λ → 0 (the zero matrix), so trace(S_λ) → 0. Therefore df(λ) → 0.

**Interpretation:** With extreme regularization, the model has effectively zero degrees of freedom — it makes the same constant prediction regardless of x.

The effective df(λ) provides a continuous, interpretable measure of model complexity that interpolates between 0 and p as λ decreases from ∞ to 0.

---

**Part (d) — The 1-SE rule [2 pts]**

The analyst is applying the **one-standard-error rule** (1-SE rule), introduced by Breiman et al. (1984) in the CART monograph.

**Rule:** After cross-validation, do not choose the λ that minimizes CV error. Instead, choose the **largest λ** (most regularized, simplest model) whose CV error is within one standard error of the minimum CV error.

**Why this is preferable:**
1. The minimum CV error estimate is itself subject to estimation noise — the true optimal λ may be somewhat larger.
2. Models with CV errors within 1 SE of the minimum are statistically indistinguishable from the optimal model.
3. The 1-SE rule selects a simpler, more regularized model that is more stable across repeated analyses and generalizes more reliably.
4. In the example: λ* = 0.1 minimizes CV error, but λ = 0.5 is within 1 SE and produces a simpler model. The analyst prefers parsimony without sacrificing predictive accuracy.

---

### Q22 Solution: PARAFAC for Fluorescence Spectroscopy and Comparison with K-means

**Part (a) — Why PARAFAC is natural here [3 pts]**

Fluorescence spectroscopic data is generated by a physically additive process: the measured fluorescence intensity at excitation wavelength j and emission wavelength k for sample i is the sum of contributions from each fluorescent compound:

`x_{ijk} = Σᵣ aᵢᵣ · bⱼᵣ · cₖᵣ + noise`

This is exactly the PARAFAC model: `X ≈ Σᵣ aᵣ ∘ bᵣ ∘ cᵣ`

**Physical interpretation with R = 2 components:**

- **A ∈ ℝ^{80×2} (sample mode):** Each row i gives the scores [aᵢ₁, aᵢ₂] for sample i. aᵢᵣ is proportional to the concentration of compound r in sample i. If the model works correctly, aᵢ₁ correlates linearly with the known concentration of compound A, and aᵢ₂ with compound B.

- **B ∈ ℝ^{30×2} (excitation mode):** Column bᵣ (length 30) is the excitation spectrum (profile) of compound r — how fluorescence intensity varies with the excitation wavelength.

- **C ∈ ℝ^{50×2} (emission mode):** Column cᵣ (length 50) is the emission spectrum of compound r — how fluorescence intensity varies with the emission wavelength.

The trilinear structure of PARAFAC directly matches the physics of fluorescence: each compound contributes independently with its own spectral fingerprint, and the total signal is their additive mixture.

---

**Part (b) — Selecting R [2 pts]**

**Method 1: CORCONDIA (Core Consistency Diagnostic)**

Fit PARAFAC for several values of R. For each R, compute the Tucker core tensor G from the PARAFAC loading matrices (G = X ×₁ A⁻¹ ×₂ B⁻¹ ×₃ C⁻¹) and calculate:

`CORCONDIA = 100 · (1 − ‖I − G‖²_F / ‖I‖²_F)`

- CORCONDIA ≈ 100: G is nearly super-diagonal → PARAFAC structure is appropriate → this R is suitable.
- CORCONDIA drops sharply below ~50: model is strained → R is too large.
- Select the largest R before CORCONDIA drops.

**What it assesses:** Whether the trilinear PARAFAC structure truly fits the data at the chosen R — a structural diagnostic.

**Method 2: Split-Half Analysis (FMS)**

1. Randomly split the 80 samples into two halves of 40.
2. Fit PARAFAC with R components to each half independently.
3. Compute the Factor Match Score (FMS) = Σᵣ (cosine similarity of aᵣ) × (cosine similarity of bᵣ) × (cosine similarity of cᵣ).
4. FMS close to R: both halves find the same components → stable, reliable solution.
5. FMS << R: the two halves disagree → R too large, solutions are not reproducible.

**What it assesses:** Reproducibility and stability of the solution across data subsets.

Use both methods together: choose R where CORCONDIA is high AND FMS is close to R.

---

**Part (c) — PARAFAC vs K-means [3 pts]**

**(i) Type of structure recovered:**

- **PARAFAC** decomposes the full 3-way tensor into R additive rank-one components. It recovers the underlying spectral profiles and concentration profiles simultaneously. The result is a physically meaningful decomposition reflecting the two fluorescent compounds.
- **K-means** groups the 80 samples into K clusters based on the Euclidean distance between vectorized spectra. It finds which samples have similar spectra but does not decompose or interpret the spectral variation — it only assigns group membership.

**(ii) Physical interpretability:**

- **PARAFAC:** The loading vectors B and C directly recover the excitation and emission spectra of each compound — these are physically interpretable as the pure-component spectra. The score vector A gives concentration estimates. This is directly usable by a chemist.
- **K-means:** The cluster centroids are average spectra of samples in each cluster. They do not correspond to pure-component spectra and have no direct physical interpretation in terms of the underlying compounds.

**(iii) Trilinear constraint:**

PARAFAC imposes the trilinear structure `x_{ijk} = Σᵣ aᵢᵣ bⱼᵣ cₖᵣ`, which directly encodes the physics of how fluorescence signals combine. This constraint:
- Acts as strong regularization, preventing overfitting by restricting the model to physically plausible solutions.
- Ensures that PARAFAC is essentially unique (up to sign and permutation of components), meaning the recovered spectra are the true spectra, not an arbitrary rotation.
- K-means imposes no such structure — it vectorizes the spectra and treats each of the 1500 entries as an independent feature, destroying the 2D excitation-emission structure and the trilinear relationship.

---

**Part (d) — Interpreting CORCONDIA = 87 [2 pts]**

**CORCONDIA = 87** (close to 100, but not perfect):

This indicates that the fitted core tensor G is close to super-diagonal but not exactly so. The PARAFAC model with R = 2 is a good fit — the trilinear structure is approximately satisfied, meaning the two-component model captures the main variation in the data. The small deviation from 100 may reflect minor model misfit, measurement noise, or small violations of the pure trilinear assumption (e.g., slight instrumental noise or overlapping spectra). Overall, R = 2 is likely appropriate and the solution is physically interpretable.

**CORCONDIA ≈ 0 (or negative):**

This would indicate that G deviates severely from the super-diagonal identity tensor — the fitted core is dense (many non-zero off-diagonal elements). This means the PARAFAC model is strained: the trilinear structure does not adequately describe the data at the chosen R. The components are interacting (cross-talk), which contradicts PARAFAC's independence assumption. This signals that R is too large — the model is fitting noise and the extra components are not physically meaningful. In this case, reduce R or switch to Tucker3 (which explicitly models cross-talk via a full core tensor G).
