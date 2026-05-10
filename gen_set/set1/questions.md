# Practice Set 1 — CDA 02582

**Format:** 20 multiple-choice + 2 open questions  
**Scoring:** MC: +1 (all correct answers selected), -1 (any wrong selection), 0 (unanswered)  
**Duration:** 4 hours

---

## Multiple Choice

**Question (1)** [Week 1]  
The Expected Prediction Error (EPE) at a new point x₀ can be written as:

`EPE = E(y − f̂)² = σ² + Bias²(f̂) + Var(f̂)`

Which of the following statements about this decomposition are correct?

A. Increasing model complexity tends to decrease bias and increase variance simultaneously.  
B. The irreducible noise term σ² can be reduced by choosing a sufficiently flexible model.  
C. The variance component measures how much the fitted model fluctuates across different training datasets.  
D. For Ordinary Least Squares (OLS) with all p predictors included, the bias is exactly zero.  
E. None of the above.

---

**Question (2)** [Week 1]  
Which of the following correctly describes the Ridge regression estimator and its key properties?

A. The ridge estimator is β̂_ridge = (XᵀX + λI)⁻¹ Xᵀy, which is always invertible for λ > 0.  
B. Ridge regression shrinks some coefficients to exactly zero, performing automatic variable selection.  
C. As λ → ∞, the effective degrees of freedom df(λ) = trace(X(XᵀX + λI)⁻¹Xᵀ) decreases toward zero.  
D. Ridge regression is unbiased for any value of λ > 0.  
E. None of the above.

---

**Question (3)** [Week 1]  
Consider the model selection criteria Cp, AIC, and BIC. Which of the following statements are correct?

A. For Gaussian errors, Cp and AIC are equivalent criteria.  
B. BIC uses a penalty of log(N) per parameter, whereas AIC uses a fixed penalty of 2 per parameter.  
C. AIC is asymptotically equivalent to leave-one-out cross-validation.  
D. For large N, BIC tends to select more complex models than AIC because its penalty grows with N.  
E. None of the above.

---

**Question (4)** [Week 2]  
Which of the following statements about the Lasso are correct?

A. The Lasso objective is min_β (Y − Xβ)ᵀ(Y − Xβ) + λ‖β‖₁, where ‖β‖₁ = Σ|βⱼ|.  
B. The Lasso has a closed-form solution analogous to the Ridge estimator.  
C. The geometry of the L₁ constraint region (a diamond in 2D) explains why Lasso solutions are often sparse.  
D. In the p > n setting, Lasso can select at most n non-zero coefficients.  
E. None of the above.

---

**Question (5)** [Week 2]  
A researcher runs 50 independent hypothesis tests at individual significance level α = 0.05. They apply the Bonferroni correction. Which of the following are correct?

A. The Bonferroni-corrected threshold for each individual test is p < 0.001.  
B. Without any correction, the Family-Wise Error Rate (FWER) is approximately 1 − (0.95)⁵⁰ ≈ 0.923.  
C. The Bonferroni correction controls the FWER at level α = 0.05 across all 50 tests.  
D. The Bonferroni correction has higher statistical power than the Benjamini-Hochberg procedure at the same overall α level.  
E. None of the above.

---

**Question (6)** [Week 2]  
In nested cross-validation (double-loop CV), which of the following statements are correct?

A. The outer loop is used for model selection (tuning hyperparameters), and the inner loop is used for model assessment.  
B. Nested CV audits the entire modelling pipeline including the hyperparameter selection step.  
C. A large gap between inner-loop error and outer-loop error suggests selection-induced overfitting.  
D. Nested CV is unnecessary when AIC or BIC is used for model selection.  
E. None of the above.

---

**Question (7)** [Week 3]  
Regarding the curse of dimensionality and regularization methods, which of the following are correct?

A. As the number of dimensions D increases, a fixed number of training points N becomes exponentially sparse in the feature space.  
B. In the elastic net, setting α = 1 gives pure Ridge regression.  
C. The elastic net penalty combines an L₁ term and an L₂ term, allowing both variable selection and grouping of correlated predictors.  
D. Donoho (2000) identified that high-dimensional data often lies on a low-dimensional manifold as a "blessing" of dimensionality.  
E. None of the above.

---

**Question (8)** [Week 4]  
A classification tree is being grown on training data. Which of the following statements about splitting criteria are correct?

A. The Gini index for a node is defined as G = Σₖ p_mk(1 − p_mk), and equals zero when the node is completely pure.  
B. The misclassification rate is the preferred criterion for growing classification trees because it is differentiable.  
C. Cross-entropy (deviance) and the Gini index are both more sensitive to changes in class probabilities than the misclassification rate.  
D. In CART, a regression tree prediction in each leaf region is the mean of the training responses in that region.  
E. None of the above.

---

**Question (9)** [Week 4 / Week 5]  
Cost-complexity pruning is applied to a fully grown CART regression tree. Which of the following are correct?

A. The cost-complexity criterion is C_α(T) = R(T) + α|T|, where |T| is the number of terminal nodes.  
B. When α = 0, the pruned tree is the root node (single-leaf tree) since no penalty is applied.  
C. As α increases, the selected subtree becomes smaller (fewer leaves).  
D. The pruning parameter α is typically chosen by minimizing cross-validation error over a sequence of candidate values.  
E. None of the above.

---

**Question (10)** [Week 5]  
Bagging (Bootstrap Aggregating) is applied to deep, unpruned CART regression trees. The variance of the bagged predictor is given by:

`Var(ŷ_bag) = ρσ² + (1−ρ)/B · σ²`

where ρ is the pairwise correlation between trees, σ² is the variance of a single tree, and B is the number of bootstrap samples. Which of the following are correct?

A. As B → ∞, the bagged variance approaches ρσ², which is the irreducible floor determined by inter-tree correlation.  
B. Bagging reduces both the bias and the variance of individual trees.  
C. Each bootstrap sample of size N contains on average approximately 63.2% of the unique training observations.  
D. Out-of-bag (OOB) error estimation is a free by-product of bagging that approximates leave-one-out cross-validation error.  
E. None of the above.

---

**Question (11)** [Week 6]  
Random Forests extend bagging by adding random feature subsampling at each split. Which of the following are correct?

A. The default heuristic for the number of candidate features per split in classification problems is m = ⌊√p⌋.  
B. When m = p (all features considered at each split), Random Forest reduces to standard bagging.  
C. Random Forests reduce variance compared to bagging by decorrelating the trees, which lowers the inter-tree correlation ρ.  
D. In gradient boosting, shallow trees (stumps) are preferred as base learners, while in Random Forests, deep trees are preferred.  
E. None of the above.

---

**Question (12)** [Week 6]  
AdaBoost.M1 is applied to a binary classification problem with labels yᵢ ∈ {−1, +1}. The classifier weight at step m is:

`α_m = log[(1 − err_m) / err_m]`

Which of the following statements are correct?

A. If err_m = 0.5 (random classifier), then α_m = 0, meaning the m-th weak learner contributes nothing to the final vote.  
B. Boosting reduces bias (unlike bagging, which only reduces variance), which is why boosting uses shallow trees (stumps) as weak learners.  
C. The exponential loss used by AdaBoost is more robust to label noise than the binomial deviance loss because it penalizes misclassified observations less.  
D. In forward stagewise additive modelling, previously fitted trees are updated (their weights are adjusted) as each new tree is added.  
E. None of the above.

---

**Question (13)** [Week 7]  
In the Support Vector Machine (SVM) with canonical scaling, the primal optimization problem is:

`min_{β,β₀} (1/2)‖β‖²`  
`subject to yᵢ(xᵢᵀβ + β₀) ≥ 1 for all i`

Which of the following statements are correct?

A. The margin width in the canonical SVM is C = 1/‖β‖, so minimizing ‖β‖² is equivalent to maximizing the margin.  
B. Non-support vectors (points far from the margin) have Lagrange multipliers αᵢ > 0, and support vectors have αᵢ = 0.  
C. The RBF (Gaussian) kernel K(x, x') = exp(−γ‖x − x'‖²) mathematically corresponds to a dot product in an infinite-dimensional feature space.  
D. The SVM dual formulation expresses the problem purely in terms of inner products ⟨xᵢ, xⱼ⟩, enabling the kernel trick.  
E. None of the above.

---

**Question (14)** [Week 8]  
Principal Component Analysis (PCA) is applied to a centered data matrix X ∈ ℝ¹⁰⁰×⁵. The singular value decomposition gives X = UDVᵀ. The singular values are d₁ = 8, d₂ = 6, d₃ = 4, d₄ = 2, d₅ = 1. Which of the following are correct?

A. The fraction of total variance explained by the first two principal components is (64 + 36) / (64 + 36 + 16 + 4 + 1) = 100/121 ≈ 82.6%.  
B. The loading vectors (principal axes) V are the right singular vectors of X, and are identical to the eigenvectors of the covariance matrix XᵀX/(n−1).  
C. PCA applied to unscaled data (without standardizing features) may be dominated by high-variance features measured in large units.  
D. Partial Least Squares (PLS) differs from PCA in that PLS ignores the response variable y and maximizes only the variance of Xv.  
E. None of the above.

---

**Question (15)** [Week 9]  
K-means clustering is applied to a dataset. Which of the following statements are correct?

A. K-means minimizes the objective Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖², where μₖ is the centroid of cluster k.  
B. K-means is guaranteed to converge to the global optimum regardless of initialization.  
C. The silhouette coefficient s(i) = (b(i) − a(i)) / max{a(i), b(i)} takes values in [−1, 1], with values near +1 indicating well-clustered points.  
D. The gap statistic selects the number of clusters K by comparing the within-cluster dispersion of the data to that expected under a uniform reference distribution.  
E. None of the above.

---

**Question (16)** [Week 9]  
The EM algorithm is used to fit a Gaussian Mixture Model (GMM) with K components. Which of the following are correct?

A. In the E-step, the posterior probability γᵢⱼ = P(Zᵢ = j | xᵢ) is computed using Bayes' rule: γᵢⱼ = πⱼN(xᵢ; μⱼ, Σⱼ) / Σⱼ' πⱼ'N(xᵢ; μⱼ', Σⱼ').  
B. In the M-step, the mean update is μⱼ = Σᵢ γᵢⱼ xᵢ / Σᵢ γᵢⱼ, a weighted average of data points with soft-assignment weights.  
C. K-means is a special case of GMM with equal, spherical covariances and hard (binary) assignments.  
D. GMM with full per-component covariance matrices always produces a unique global maximum of the likelihood.  
E. None of the above.

---

**Question (17)** [Week 10]  
Consider a fully connected feedforward neural network: Input layer: 5 nodes; Hidden layer 1: 3 nodes (ReLU); Hidden layer 2: 3 nodes (ReLU); Output layer: 2 nodes (softmax). Each layer includes a bias term. How many scalar parameters does this network have, and which additional statements are correct?

A. The total number of parameters is (5×3 + 3) + (3×3 + 3) + (3×2 + 2) = 18 + 12 + 8 = 38.  
B. Binary cross-entropy loss is derived from maximizing the Bernoulli likelihood: L = −Σᵢ[yᵢ log(ŷᵢ) + (1−yᵢ)log(1−ŷᵢ)].  
C. In backpropagation, the error signal δ^(ℓ) = (W^(ℓ+1))ᵀ δ^(ℓ+1) ⊙ σ'(z^(ℓ)) propagates blame backwards through the network.  
D. Recurrent Neural Networks (RNNs) suffer from the vanishing gradient problem for long sequences, motivating LSTM and GRU architectures.  
E. None of the above.

---

**Question (18)** [Week 11]  
Non-negative Matrix Factorization (NMF) and Independent Component Analysis (ICA) are both unsupervised decomposition methods. Which of the following statements are correct?

A. NMF enforces non-negativity on both factor matrices W and H, producing an additive, parts-based representation with no cancellation between components.  
B. ICA requires that the source components are statistically independent and non-Gaussian; it cannot separate Gaussian sources.  
C. NMF solutions are unique — there is only one valid factorization X ≈ WH for given W ≥ 0, H ≥ 0.  
D. ICA preprocessing involves centering and whitening the data so that subsequent optimization needs only to find an orthogonal rotation matrix.  
E. None of the above.

---

**Question (19)** [Week 11]  
Archetypal Analysis (AA) approximates each data point as a convex mixture of K archetypes, and each archetype is itself a convex combination of data points. The objective is:

`min_{S,H} ‖X − XSH‖²_F`

Which of the following are correct?

A. Archetypes in AA are located on (or near) the convex hull of the data, representing extreme prototypes rather than average profiles.  
B. In Sparse Coding, the dictionary W is overcomplete (K > I, more atoms than dimensions), and each data point is represented using a sparse coefficient vector h with most entries equal to zero.  
C. The matrix S in AA has columns that sum to 1 with non-negative entries, forcing each archetype to be a convex combination of real data points.  
D. AA and K-means find the same solution when the number of components is the same, since both represent data using a fixed number of prototypes.  
E. None of the above.

---

**Question (20)** [Week 12]  
A 3-way tensor X of shape I × J × K is decomposed using PARAFAC with R components:

`X ≈ Σᵣ aᵣ ∘ bᵣ ∘ cᵣ`

and separately using Tucker3 with ranks (P, Q, R):

`X ≈ G ×₁ A ×₂ B ×₃ C`

Which of the following are correct?

A. PARAFAC is a special case of Tucker3 where the core tensor G is super-diagonal (identity-like).  
B. Tucker3 solutions are essentially unique (up to sign and permutation), whereas PARAFAC solutions are not unique due to rotational freedom.  
C. CORCONDIA (Core Consistency Diagnostic) close to 100 indicates that the PARAFAC model has an appropriate number of components R, because the fitted core tensor is close to super-diagonal.  
D. In the Tucker3 matrix representation, X_{(1)} ≈ A G_{(1)} (C ⊗ B)ᵀ uses the Kronecker product, whereas PARAFAC uses the Khatri-Rao product.  
E. None of the above.

---

## Open Questions

**Question (21)** [Weeks 1–3] — 10 points

Ridge regression and the Lasso are both regularization methods for linear regression, but they have fundamentally different properties.

**(a)** [3 pts] Derive the closed-form solution for Ridge regression starting from the penalized least squares objective:

`min_β (Y − Xβ)ᵀ(Y − Xβ) + λβᵀβ`

Show all steps including the derivative, setting it to zero, and solving for β̂_ridge.

**(b)** [3 pts] Explain geometrically why the Lasso (L₁ penalty) produces sparse solutions (exact zeros) while Ridge (L₂ penalty) does not. Refer to the shape of the constraint regions and the RSS contours.

**(c)** [2 pts] The effective degrees of freedom for a ridge fit with regularization parameter λ is:

`df(λ) = trace(X(XᵀX + λI)⁻¹Xᵀ)`

Explain what happens to df(λ) as λ → 0 and as λ → ∞, and interpret these limits.

**(d)** [2 pts] A data analyst applies 10-fold cross-validation to select λ for ridge regression. She notices that the minimum CV error is achieved at λ* = 0.1, but she selects λ = 0.5 instead. Explain what rule she might be applying and why this choice can be preferable.

---

**Question (22)** [Weeks 9–12] — 10 points

You are given a dataset of fluorescence excitation-emission spectra for 80 chemical solutions. Each solution is measured at 30 excitation wavelengths and 50 emission wavelengths, yielding a 3-way tensor X of shape 80 × 30 × 50 (samples × excitations × emissions). Each solution is known to contain a mixture of two fluorescent compounds (compound A and compound B) in varying concentrations.

**(a)** [3 pts] Explain why PARAFAC is a particularly natural model for this spectroscopic dataset. What physical interpretation do the three loading matrices A (sample mode), B (excitation mode), and C (emission mode) carry when R = 2 components are used?

**(b)** [2 pts] How would you select the number of PARAFAC components R? Describe two complementary methods and explain what each assesses.

**(c)** [3 pts] Suppose you instead fit a K-means clustering model to the 80 sample spectra (vectorizing each 30×50 spectrum into a 1500-dimensional vector). Compare this approach to PARAFAC in terms of: (i) the type of structure each method recovers, (ii) whether the result is physically interpretable, and (iii) the effect of the trilinear constraint in PARAFAC.

**(d)** [2 pts] After fitting PARAFAC with R = 2, you find that CORCONDIA = 87. What does this value tell you, and what would CORCONDIA ≈ 0 indicate?
