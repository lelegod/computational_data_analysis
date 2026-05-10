# Practice Set 2 — Solutions

## Multiple Choice Answers

| Q | Correct | Explanation |
|---|---------|-------------|
| 1 | B, C | **B is correct**: EPE = Bias² + Var + σ², averaged over y and D — this is the standard decomposition. **C is correct**: Training error always decreases with model complexity (degrees of freedom increase), while test error forms a U-shape (bias decreases, variance increases). **A is wrong**: σ² is irreducible — no model can eliminate it; it is inherent noise in the data-generating process. **D is wrong**: D describes the bias, not variance. Variance is E[(f̂ − E[f̂])²] (fluctuation around the mean prediction), while bias is (E[f̂] − f)² (deviation of mean prediction from truth). |
| 2 | A, C, D | **A is correct**: For OLS with p predictors, S = X(XᵀX)⁻¹Xᵀ and trace(S) = p. **C is correct**: As λ → 0, ridge approaches OLS, so df(λ) = trace(S_λ) → p. **D is correct**: Stone (1977) proved AIC is asymptotically equivalent to LOO-CV; since AIC uses d = df(λ) for ridge, this extends directly. **B is wrong**: As λ → ∞, df → 0 (not p) — extreme regularization shrinks all coefficients to zero, eliminating all effective parameters. |
| 3 | C | Apply the BH rule: sort p-values (already sorted). For each i, check p_(i) ≤ (i/m)q = (i/6)(0.10). Thresholds: i=1: 0.10/6≈0.0167; i=2: 0.20/6≈0.0333; i=3: 0.30/6=0.050; i=4: 0.40/6≈0.0667; i=5: 0.50/6≈0.0833; i=6: 0.60/6=0.10. Check: p_(1)=0.004 ≤ 0.0167 ✓; p_(2)=0.012 ≤ 0.0333 ✓; p_(3)=0.038 ≤ 0.050 ✓; p_(4)=0.080 > 0.0667 ✗. The largest k satisfying the condition is k=3. Reject H_(1), H_(2), H_(3). Answer: **C (3 rejections)**. The BH rule finds the largest k where p_(k) ≤ (k/m)q and rejects all hypotheses 1 through k simultaneously. |
| 4 | A, B | **A is correct**: A fixed N becomes exponentially sparse as D grows — this is the core statement of the curse. **B is correct**: In high dimensions, most data points cluster near the surface/corners of the hypercube (edge effect), not the interior. **C is wrong**: Euclidean distances LOSE meaning in high dimensions — distances between all pairs of points become nearly equal (concentration of measure), making them uninformative, not more informative. **D is wrong**: When p > N, (XᵀX) is not invertible (rank-deficient), so OLS cannot be computed at all — not merely "high variance." |
| 5 | A, C, D | **A is correct**: Gini G = Σₖ p_{mk}(1 − p_{mk}). When a node is pure (one class has pₖ=1, all others 0), G = 1·0 + 0+... = 0. **C is correct**: For K=2 with p = p_{m1}: G = p(1−p) + (1−p)p = 2p(1−p). **D is correct**: Weakest-link pruning incrementally removes internal nodes as α increases, generating a nested sequence T₀ > T₁ > ... > root. **B is wrong**: Misclassification rate is NOT preferred for growing trees — it is insensitive to probability shifts within the majority class. Gini and cross-entropy are preferred for growing; misclassification rate is used for pruning. |
| 6 | B, D | **B is correct**: Setting ρ=1: Var = σ² + 0 = σ² = variance of a single tree. Bagging provides no reduction. **D is correct**: Each observation's OOB prediction is formed from trees that never saw it during training — this exactly mimics LOO-CV's "train on everything except one" structure; empirically the errors are very similar. **A is wrong**: As B → ∞, Var → ρσ², not 0. The term ρσ² remains as a floor regardless of B. **C is wrong**: Bagging does NOT reduce bias — the bias of the bagged estimator equals the bias of any single tree (they are identically distributed), as shown by E[(1/B)Σ(ŷ_b − y)] = E[ŷ_b − y]. |
| 7 | A, C | **A is correct**: αₘ = log[(1−errₘ)/errₘ]. When errₘ > 0.5: (1−errₘ)/errₘ < 1, so log < 0 → αₘ < 0. A negative weight negates the classifier's votes in the final sum, effectively reversing its predictions. **C is correct**: This is the theoretical result of Friedman, Hastie & Tibshirani (2000) — AdaBoost minimizes the expected exponential loss via forward stagewise fitting. **B is wrong**: Weights of MISCLASSIFIED observations are INCREASED (they get more attention), not decreased. Correctly classified observations get relatively less weight. **D is wrong**: Boosting primarily reduces BIAS (it uses weak learners with high bias like stumps and corrects their errors). Bagging reduces variance. The use of stumps is precisely because shallow trees are high-bias weak learners — boosting corrects the bias sequentially. |
| 8 | A, B, C, D | **A is correct**: With canonical scaling |xᵢᵀβ + β₀| = 1 for support vectors, the signed distance for a support vector is 1/‖β‖, so the total margin (from +1 to −1 boundary) is 2/‖β‖, and the half-margin is 1/‖β‖. **B is correct**: This follows directly from KKT complementary slackness: αᵢ[yᵢ(xᵢᵀβ+β₀)−1]=0. If a point is not on the margin (bracket>0), αᵢ must be 0. Support vectors have αᵢ>0, so the bracket=0 (they sit exactly on the margin). **C is correct**: The dual objective Σαᵢ − (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ⟨xᵢ,xⱼ⟩ contains xᵢ only inside inner products — replacing ⟨xᵢ,xⱼ⟩ with K(xᵢ,xⱼ) is the kernel trick. **D is correct**: The SVM primal is convex (quadratic objective, linear constraints). Slater's condition (existence of strictly feasible point) holds for separable data, so strong duality applies. |
| 9 | B | **B is correct**: When y is weakly correlated with X but some features have high variance, PCA selects directions of maximum variance in X — these may have zero correlation with y. PLS explicitly maximizes Cov(Xα, y) = Var(Xα)·Corr²(Xα,y), keeping directions that predict y. In the audio example, PCR keeps the loud hum (high variance, irrelevant); PLS elevates the vocal quiver (low variance, high correlation). **A**: Ridge shrinks all coefficients but does not prioritize y-relevant directions. **C**: Lasso selects variables but still fits in original feature space, not a supervised subspace. **D**: CCA maximizes correlation but requires Y to also be a matrix; with scalar y, CCA reduces to a correlation maximization that ignores variance, which is less suitable than PLS. |
| 10 | A, B, C | **A is correct**: The weight step φ̂ₘⱼ = xⱼ^(m−1)ᵀy is the dot product of feature j (after deflation) with y — this is the covariance (since y is standardized), so features more correlated with y get higher weight. **B is correct**: This is a provable property of the PLS deflation step — the deflation removes variance explained by zₘ from all features, ensuring subsequent components are orthogonal to zₘ. **C is correct**: When M=p all components are extracted; the final PLS prediction equals OLS because the algorithm has extracted all possible directions in X. **D is wrong**: PLS maximizes Cov(Xu, Yv) = Var(Xu)·Var(Yv)·Corr²(Xu,Yv), balancing variance AND correlation. CCA maximizes only Corr²(Xu,Yv), ignoring internal variance. They have different objectives. |
| 11 | B, C, D | **B is correct**: K-medoids uses actual data points as cluster centers; outliers cannot become centroids, and pulling outliers cannot distort medoids as strongly as they distort means. **C is correct**: GMM with full (or diagonal) per-cluster covariance matrices can model elliptical, elongated, and differently-sized clusters that K-means (spherical assumption) misses. **D is correct**: G(K) = log(U_k) − log(W_k), where U_k is the expected within-cluster dissimilarity under uniform data (from Monte Carlo), and W_k is the actual within-cluster dissimilarity. The selection rule picks the smallest K where the gap is sufficiently large. **A is wrong**: "Always fail" is too strong. K-means can partition elliptical clusters if they are well-separated, though the partition may not match the elliptical structure. The deeper problem is that K-means assumes spherical clusters and equal cluster sizes. |
| 12 | A, B | **A is correct**: s(i)=1 requires b(i)>>a(i). If a(i)→0 (very close to own cluster center) and b(i)>>0 (far from nearest other cluster), max is b(i) and s(i)=(b(i)−0)/b(i)=1. This is perfect clustering. **B is correct**: s(i)=0 means b(i)=a(i) — the observation is equidistant between its cluster and the next closest. It sits on the decision boundary. **C is wrong**: s(i)=−1 means a(i)>>b(i), i.e., the observation is much closer to the neighboring cluster than to its own — this indicates MIS-clustering (it should be in the neighboring cluster). **D is wrong**: The optimal K is selected by MAXIMIZING (not minimizing) the average silhouette width, since higher silhouette = better-defined clusters. |
| 13 | A, B, C | **A is correct**: Differentiating σ(x) = 1/(1+e^{−x}) gives σ'(x) = σ(x)(1−σ(x)). This is computationally efficient: once σ(x) is computed in the forward pass, the derivative requires only two multiplications — no additional exponential evaluation. **B is correct**: This is the fundamental principle of backpropagation — activations flow forward, gradients flow backward. **C is correct**: ∂L/∂W^(ℓ) = δ^(ℓ)(a^(ℓ−1))ᵀ — this product involves only the error signal δ^(ℓ) at layer ℓ and the activation a^(ℓ−1) from the layer below; no weights from future (deeper) layers appear explicitly in this product. **D is wrong**: Binary cross-entropy is derived from the BERNOULLI likelihood (Bernoulli(p) where p = σ(f(x))). MSE is derived from the Gaussian likelihood. Using BCE for classification and MSE for regression is the correct correspondence. |
| 14 | D | Calculation: Layer 1 (5→8): 5×8 weights + 8 biases = 40+8 = 48. Layer 2 (8→4): 8×4 weights + 4 biases = 32+4 = 36. Layer 3 (4→2): 4×2 weights + 2 biases = 8+2 = 10. **Total: 48+36+10 = 94.** A=78 would result from omitting biases in two layers; B=82 is off; C=86 miscomputes layer 2. The correct count is 94, which is option D. |
| 15 | A, C, D | **A is correct**: NMF requires W ≥ 0 AND H ≥ 0. Because both factors are non-negative, each data point is represented as an additive combination of parts (no cancellation between positive and negative components, unlike PCA). **C is correct**: The multiplicative update H ← H ⊙ (WᵀX)/(WᵀWH) multiplies H element-wise by a non-negative ratio — if H starts positive, every element stays positive throughout. **D is correct**: Any invertible Q with WQ⁻¹ ≥ 0 and QH ≥ 0 yields an equally valid NMF, so solutions are not unique without additional constraints (sparsity, geometric volume minimization). **B is wrong**: The NMF objective is NOT jointly convex in (W,H). It is separately convex in W given H, and convex in H given W ("biconvex"), but not jointly — this is why ALS converges to a local minimum, not necessarily the global one. |
| 16 | A, C | **A is correct**: PCA maximizes Var(Xv) — purely a variance objective, no supervision, no independence. ICA maximizes non-Gaussianity (kurtosis, negentropy) as a proxy for statistical independence. These are fundamentally different objectives. **C is correct**: After whitening, the data's covariance is the identity. The mixing matrix A (from x=As) becomes orthogonal after whitening, reducing the ICA search from arbitrary invertible matrices to orthogonal rotations — a much smaller search space. **B is wrong**: ICA CANNOT separate Gaussian sources. By the Central Limit Theorem, all linear mixtures of Gaussians are also Gaussian — there is no way to distinguish sources from mixtures using non-Gaussianity. ICA requires non-Gaussian sources. **D is wrong**: PCA only decorrelates signals (removes second-order dependencies). Statistical independence requires eliminating ALL higher-order dependencies (kurtosis, etc.). Two signals can be uncorrelated but not independent (e.g., if one is a nonlinear function of the other). |
| 17 | A, C, D | **A is correct**: PARAFAC assumes a super-diagonal core G = I^{R×R×R} (ones on the main diagonal, zeros elsewhere). This eliminates cross-talk between components — each component r interacts only with itself across all three modes. Tucker3 with a full core tensor G^{P×Q×R} allows all pairwise interactions. **C is correct**: CORCONDIA = 100·(1 − ‖I − G‖²_F / ‖I‖²_F). When G ≈ I (super-diagonal), ‖I−G‖²_F ≈ 0, so CORCONDIA ≈ 100. This indicates the PARAFAC model structure is appropriate for this R. **D is correct**: This is exactly equation (15) in the Tucker/PARAFAC derivations — the ALS update for A unfolds X along mode 1 and multiplies by the pseudoinverse of Z_A = (C ⊙ B)ᵀ. **B is wrong**: This is reversed. PARAFAC is essentially unique (up to permutation and scaling of components) under mild conditions (Kruskal's condition). Tucker3 has rotational freedom — you can insert any invertible rotation Q between G and A without changing the model fit. |
| 18 | A, B, D | **A is correct**: FMS = Σᵣ [cosine similarity of a_r] × [cosine similarity of b_r] × [cosine similarity of c_r]. Each product is at most 1, so FMS ≤ R. If all corresponding loading vectors are identical between halves (cosine = 1 for all r and all modes), FMS = R — perfect stability. **B is correct**: Each term (a_rᵀâ_r)/(‖a_r‖‖â_r‖) is a cosine similarity in [−1,1]. In practice, after resolving sign ambiguity, each factor is in [0,1], making each component's contribution in [0,1], so FMS ∈ [0, R]. **D is correct**: PARAFAC components are not nested — the R=3 solution is computed independently from the R=4 solution. Component 1 in R=3 will generally differ from component 1 in R=4 because the overall factorization changes. This contrasts with PCA, where PC₁ at R=3 is identical to PC₁ at R=5. **C is wrong**: Low FMS (FMS << R) suggests R is too LARGE — the model is fitting noise and the two halves find different spurious components. More components would make instability worse, not better. |
| 19 | A, B | **A is correct**: When you search over many λ values and report the minimum CV error, you have "spent" the independence of the validation folds by selecting the configuration that performed best on them. The minimum over a grid of CV errors is optimistically biased — it will tend to be lower than the true generalization error. **B is correct**: Nested CV (outer loop = assessment, inner loop = selection) separates the selection procedure from the evaluation. The outer test folds have never been touched by the selection step, giving an unbiased estimate of the full pipeline's generalization error. **C is wrong**: The 1-SE rule selects the LARGEST λ (most regularized, simplest model) whose CV error is within 1 SE of the minimum — NOT a smaller λ. A larger λ means fewer effective parameters, not more. **D is wrong**: The fact that validation folds were held out during model training is true, but the problem is that the same validation folds were used to SELECT λ. The minimum over a search is biased even when individual fold errors are unbiased. |
| 20 | A, B, D | **A is correct**: CCA maximizes (uᵀΣ_{XY}v)/√(uᵀΣ_{XX}u · vᵀΣ_{YY}v) — a ratio that normalizes out internal variance. Unlike PLS which balances variance and correlation, CCA is purely a correlation maximization and ignores how much variance X or Y has in any direction. **B is correct**: CCA requires inverting Σ_{XX} and Σ_{YY}. When p > n, Σ_{XX} is rank-deficient (singular) — the inverse does not exist. Regularized CCA adds λI to make the matrix invertible; Sparse CCA (PMD) applies L₁ penalties instead. **D is correct**: CCA finds canonical variate pairs (u_m, v_m). The maximum number of pairs is min(rank(X), rank(Y)) ≤ min(p,q). **C is wrong**: PLS maximizes Cov(Xu, Yv) = √(Var(Xu)·Var(Yv))·Corr(Xu,Yv) — it balances internal variance and cross-correlation. CCA maximizes only the correlation (ignoring variance). They have different objectives, different solutions, and produce different components. |

**Note for Q14:** Correct answer is D (94). Step-by-step: (5×8+8) + (8×4+4) + (4×2+2) = 48 + 36 + 10 = 94. Common errors: forgetting biases (gives 80), or miscounting one layer.

---

## Open Question Solutions

---

### Q21 Solution: Unsupervised Analysis of Gene Expression Data

#### Part (a) — PCA: Objective, SVD, Variance Explained, Scaling [3 points]

**Mathematical objective of PCA:**

PCA finds unit-norm loading vectors v that maximize the variance of the projected scores:

```
max_v  Var(Xv) = vᵀ Σ v     subject to ‖v‖ = 1
```

where Σ = (1/(n−1)) XᵀX is the sample covariance matrix (assuming X is mean-centered). Subsequent components maximize residual variance subject to orthogonality to all previous components.

**Solution via SVD:**

The data matrix X ∈ R^{n×p} (mean-centered) is decomposed as:

```
X = U D Vᵀ
```

- U ∈ R^{n×n}: left singular vectors (score directions, up to scaling)
- D ∈ R^{n×p}: diagonal matrix of singular values d₁ ≥ d₂ ≥ ... ≥ 0
- V ∈ R^{p×p}: right singular vectors = loading vectors (principal axes)

The k-th principal component loadings are the k-th column of V. The PC scores matrix is S = XV = UD.

Eigenvalues of the covariance matrix: λₖ = dₖ²/(n−1).

**Variance explained by component k:**

```
VE_k = λₖ / Σⱼ λⱼ = dₖ² / Σⱼ dⱼ²
```

**Why scale before PCA:**

Gene expression measurements across 8,000 genes span vastly different ranges (some genes vary 100-fold, others 2-fold). PCA on unscaled data is dominated by high-variance genes (effectively those with large absolute measurements), regardless of biological relevance. Scaling to unit variance (using the correlation matrix) ensures every gene contributes equally to the principal components. Without scaling, PCA finds directions of high raw magnitude, not high informational content.

---

#### Part (b) — K-means Algorithm and Motivation for PC Scores [3 points]

**K-means algorithm:**

**Initialization:** Randomly assign each of the n observations to one of K clusters (or choose K random observations as initial centroids).

**Assignment step (E-step analog):** Assign each observation xᵢ to the nearest centroid by Euclidean distance:

```
Cₖ ← {i : ‖xᵢ − μₖ‖² ≤ ‖xᵢ − μⱼ‖² for all j ≠ k}
```

**Update step (M-step analog):** Recompute each centroid as the mean of its assigned observations:

```
μₖ ← (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
```

**Convergence:** Repeat assignment and update until cluster assignments do not change (or until the objective Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖² stops decreasing meaningfully).

**Objective function:** min Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖²

The algorithm is guaranteed to converge (the objective is non-increasing) but may reach a local minimum. Multiple random restarts are recommended.

**Why use PC scores rather than raw 8,000-gene matrix:**

1. **Noise reduction:** Most of the 8,000 dimensions contain noise. The first 20 PCs capture the dominant variance structure; the remaining dimensions add noise that distorts Euclidean distances.

2. **Computational efficiency:** K-means in 20 dimensions is far faster than in 8,000 dimensions.

3. **Curse of dimensionality:** In 8,000 dimensions, Euclidean distances between all pairs of points become nearly equal (concentration of measure), making cluster assignment meaningless. PC scores live in a much lower-dimensional space where distances are informative.

4. **Correlation removal:** Raw gene expression features are highly correlated. PC scores are uncorrelated by construction, which means distance in PC-score space more cleanly separates orthogonal sources of variation.

---

#### Part (c) — Gap Statistic Formula and Warning [2 points]

**Gap statistic formula:**

```
G(K) = log(U_k) − log(W_k)
```

where:
- W_k = Σₗ (1/2Nₗ) Dₗ is the within-cluster dissimilarity for the actual data (Dₗ = Nₗ Σᵢ∈Cₗ ‖xᵢ − x̄ₗ‖²)
- U_k = average of log(W_k) computed on B=20 Monte Carlo samples drawn from a uniform distribution over the data's bounding box

**Selection rule:**

```
K* = smallest K such that G(K) ≥ G(K+1) − s'_{K+1}
```

where s'_{K+1} = std(log U_{K+1}) × √(1 + 1/B) accounts for simulation variability. Choose the smallest K where the gap is large relative to the next gap.

**Fundamental warning:**

Clustering algorithms ALWAYS produce a grouping, even on completely random data with no cluster structure. K-means applied to uniform noise will produce K compact-looking clusters. The gap statistic (and other methods) may still suggest K* > 1 by chance. Cluster validity must always be confirmed using domain knowledge, external validation, or replicated biological experiments — not statistical heuristics alone.

---

#### Part (d) — NMF vs PCA: Non-negativity and Parts-based Representation [2 points]

**Mathematical constraint NMF imposes that PCA does not:**

NMF constrains both factor matrices to be non-negative:

```
W ≥ 0   and   H ≥ 0   (element-wise)
```

PCA places no sign constraints on loading vectors (eigenvectors can have positive and negative entries). PCA components can cancel each other — one component can subtract from another.

**Why non-negativity leads to parts-based representation:**

Because W ≥ 0 and H ≥ 0, the reconstruction of any sample xⱼ is:

```
x̂ⱼ ≈ W hⱼ = Σₖ wₖ hₖⱼ
```

where every wₖ ≥ 0 and every hₖⱼ ≥ 0. This is a sum of non-negative vectors with non-negative coefficients — purely additive combination. No component can cancel another.

In gene expression terms: each basis vector wₖ represents a "gene program" (a pattern of activated genes — all positive), and each sample is a weighted sum of programs (no negative weights). This mirrors biological reality: a cell simultaneously activates multiple gene programs, and these add together. PCA, by contrast, allows negative loadings — a gene can have a negative component in PC₁, which means it "subtracts" from that principal component, leading to abstract factors without biological meaning.

---

### Q22 Solution: SVM — Primal, Dual, KKT, Kernel Trick

#### Part (a) — Primal Problem and Margin Geometry [2 points]

**Primal SVM optimization problem:**

```
min_{β, β₀}   (1/2)‖β‖²

subject to    yᵢ(xᵢᵀβ + β₀) ≥ 1   for all i = 1, ..., N
```

**Canonical scaling convention:** We fix the scale of (β, β₀) so that the constraint is tight for the nearest points (support vectors): |xᵀβ + β₀| = 1 for support vectors. This is a normalization — since the decision boundary is defined up to positive scaling, we can always choose the scale so the margin-touching points satisfy this equality. The factor 1/2 in the objective is for mathematical convenience (it cancels the coefficient 2 from differentiation).

**Geometric interpretation of margin width:**

The signed distance from any point x to the hyperplane {x : xᵀβ + β₀ = 0} is:

```
d(x) = (xᵀβ + β₀) / ‖β‖
```

For a positive support vector (yᵢ = +1 on the margin): d = 1/‖β‖.
For a negative support vector (yᵢ = −1 on the margin): d = −1/‖β‖.

The total margin width (gap between the two margin hyperplanes) is:

```
Margin = 2 / ‖β‖
```

Minimizing (1/2)‖β‖² is equivalent to maximizing the margin 2/‖β‖.

---

#### Part (b) — Lagrangian Derivation of the Dual [3 points]

**Lagrangian:**

```
L_P = (1/2)‖β‖² − Σᵢ αᵢ[yᵢ(xᵢᵀβ + β₀) − 1]
```

with αᵢ ≥ 0 (Lagrange multipliers, one per training point).

**Stationarity condition with respect to β:**

```
∂L_P/∂β = β − Σᵢ αᵢyᵢxᵢ = 0
          ⟹  β = Σᵢ αᵢyᵢxᵢ
```

**Stationarity condition with respect to β₀:**

```
∂L_P/∂β₀ = −Σᵢ αᵢyᵢ = 0
            ⟹  Σᵢ αᵢyᵢ = 0
```

**Substituting β = Σᵢ αᵢyᵢxᵢ into L_P:**

Step 1 — Expand ‖β‖²:

```
(1/2)‖β‖² = (1/2)(Σᵢ αᵢyᵢxᵢ)ᵀ(Σⱼ αⱼyⱼxⱼ) = (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ⟨xᵢ, xⱼ⟩
```

Step 2 — Expand the constraint term:

```
−Σᵢ αᵢyᵢ(xᵢᵀβ) + Σᵢ αᵢ = −Σᵢ αᵢyᵢxᵢᵀ(Σⱼ αⱼyⱼxⱼ) + Σᵢ αᵢ
                           = −ΣᵢΣⱼ αᵢαⱼyᵢyⱼ⟨xᵢ,xⱼ⟩ + Σᵢ αᵢ
```

Note: the β₀ term vanishes because Σᵢ αᵢyᵢ = 0.

Step 3 — Combine:

```
L_D = Σᵢ αᵢ − (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ⟨xᵢ, xⱼ⟩
```

**Dual problem:**

```
max_α   Σᵢ αᵢ − (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ⟨xᵢ, xⱼ⟩

subject to   αᵢ ≥ 0,   Σᵢ αᵢyᵢ = 0
```

This is a concave quadratic program in α, solvable efficiently. The training data appear only through pairwise dot products ⟨xᵢ, xⱼ⟩.

---

#### Part (c) — KKT Complementary Slackness and Informational Sparsity [3 points]

**KKT complementary slackness condition:**

At the optimum, for every training point i:

```
αᵢ · [yᵢ(xᵢᵀβ + β₀) − 1] = 0
```

This holds because we are at a saddle point of the Lagrangian (primal minimum, dual maximum). The condition says: for each i, either αᵢ = 0 OR the constraint is tight (yᵢ(xᵢᵀβ+β₀)=1) — or both.

**Two cases:**

**Case 1 — Safe point (not on the margin):** yᵢ(xᵢᵀβ+β₀) > 1 (strictly inside the margin). Then the bracket [yᵢ(xᵢᵀβ+β₀)−1] > 0, so KKT forces αᵢ = 0. This point contributes nothing to the model.

**Case 2 — Support vector (on the margin):** yᵢ(xᵢᵀβ+β₀) = 1. Then αᵢ ≥ 0 (can be nonzero). These are the only points that "matter."

**Informational sparsity:**

The primal solution is β = Σᵢ αᵢyᵢxᵢ. Since αᵢ = 0 for all safe points, only support vectors (αᵢ > 0) contribute:

```
β = Σ_{i ∈ SV} αᵢyᵢxᵢ
```

The decision function for a new point x is:

```
ŷ = sign(β₀ + xᵀβ) = sign(β₀ + Σ_{i ∈ SV} αᵢyᵢ⟨x, xᵢ⟩)
```

A model trained on N=10,000 points may have only 50 support vectors. The other 9,950 "safe" points could be deleted from the training set, and the decision boundary would be identical. The support vectors are the "difficult" observations that lie on or near the margin boundary.

**Distinguishing support vectors from safe points:**

- **Support vector**: αᵢ > 0; sits exactly on the margin (yᵢ(xᵢᵀβ+β₀)=1); removal would change the boundary.
- **Safe point**: αᵢ = 0; strictly inside the feasible region (yᵢ(xᵢᵀβ+β₀)>1); removal would not change the boundary.

---

#### Part (d) — Kernel Trick and Non-linear Boundaries [2 points]

**The kernel trick:**

In the dual problem, training data appear only as pairwise dot products ⟨xᵢ, xⱼ⟩. Suppose we want to work in a higher-dimensional feature space defined by a mapping φ: R^d → H, where H may be infinite-dimensional. The dual with explicit mapping would require computing ⟨φ(xᵢ), φ(xⱼ)⟩_H — expensive or impossible if dim(H) = ∞.

The kernel trick: replace ⟨xᵢ, xⱼ⟩ with a kernel function K(xᵢ, xⱼ) that implicitly computes ⟨φ(xᵢ), φ(xⱼ)⟩_H:

```
K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩_H
```

The dual becomes:

```
max_α   Σᵢ αᵢ − (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ K(xᵢ, xⱼ)
```

The algorithm never computes φ(x) explicitly — only evaluating K(xᵢ, xⱼ) is required, which costs O(d) (original dimension) rather than O(dim H). This allows the SVM to find non-linear decision boundaries in R^d that correspond to linear boundaries in H, at no additional computational cost beyond evaluating the kernel.

**Example: RBF (Gaussian) kernel:**

```
K(x, x') = exp(−γ‖x − x'‖²)
```

This kernel implicitly corresponds to a dot product in an **infinite-dimensional** feature space — the Taylor expansion of the exponential generates all polynomial orders simultaneously. With the RBF kernel, the SVM can represent decision boundaries of arbitrary smoothness at the cost of evaluating a simple exponential — representing infinite representational power for the price of one exponent.
