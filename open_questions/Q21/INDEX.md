# Q21 — Index
> **20 points.** Q21 asks you to describe and explain a specific method in depth.
> Q22 files are in `open_questions/Q22/` — see `Q22/INDEX.md`.

---

## Past Exam Pattern

| Year | Q21 Topic |
|------|-----------|
| 2022 | Random Forest algorithm |
| 2024 | ICA uniqueness and distributions |
| 2025 | LDA vs GMM comparison |

**Prediction**: Q21 will be a topic not yet appeared — SVM, Boosting, EPE/Bias-Variance, and Ridge/Lasso are highest risk.

---

## File Index

### Q21 Candidates

| File | Topic | Exam Likelihood |
|------|-------|----------------|
| [Q21-A — Random Forest](Q21_A_random_forest.md) | Bootstrap + random features + OOB | Appeared 2022 |
| [Q21-B — ICA](Q21_B_ica.md) | Non-Gaussianity, uniqueness, FastICA | Appeared 2024 |
| [Q21-C — LDA vs GMM](Q21_C_lda_vs_gmm.md) | Shared vs per-class covariance, EM, linear boundary | Appeared 2025 |
| [Q21-D — SVM](Q21_D_svm.md) | Max-margin, Lagrangian dual, kernel trick | **High — not yet appeared** |
| [Q21-E — Boosting](Q21_E_boosting.md) | AdaBoost, exponential loss, gradient boosting | **High — not yet appeared as Q21** |
| [Q21-F — PARAFAC/Tucker](Q21_F_parafac_tucker.md) | Core tensor, uniqueness, CORCONDIA | Related MC appeared 2022/2024 |
| [Q21-G — PCA vs PLS vs CCA](Q21_G_pca_pls_cca.md) | Objectives, supervision levels, high-dim | Classic comparison topic |
| [Q21-H — NMF/ICA/AA](Q21_H_nmf_ica_aa.md) | Constraints, uniqueness, parts vs extremes | Comparison-type Q21 |
| [Q21-I — Ridge/Lasso/Elastic Net](Q21_I_ridge_lasso.md) | Regularization, sparsity, solution paths | **High — fundamental Week 1/2** |
| [Q21-J — K-means vs Hierarchical](Q21_J_clustering.md) | Lloyd's algorithm, linkage, silhouette | **High — not yet appeared** |
| [Q21-K — Multiple Testing](Q21_K_multiple_testing.md) | Bonferroni (FWER) vs BH (FDR) | Unique topic, Week 3 |
| [Q21-L — Neural Networks](Q21_L_neural_networks.md) | Backpropagation, activations, regularization | Week 10, derivation question |
| [Q21-M — EPE / Bias-Variance](Q21_M_epe_bias_variance.md) | Decomposition derivation, cross-terms, tradeoff | **High — most fundamental theorem** |
| [Q21-N — CART / Decision Trees](Q21_N_cart.md) | Splitting criteria, pruning, impurity measures | Week 4, base of RF/Boosting |
| [Q21-O — Cross-Validation](Q21_O_cross_validation.md) | K-fold, 1-SE rule, nested CV, AIC vs CV | Weeks 2/3/5 |
| [Q21-P — Logistic Regression vs LDA](Q21_P_logistic_regression.md) | Discriminative vs generative, IRLS, regularized | Classic comparison question |
| [Q21-Q — OLS & Gauss-Markov](Q21_Q_ols_gauss_markov.md) | Unbiasedness proof, BLUE theorem, Ridge comparison | Week 1, derivation question |
| [Q21-R — Bootstrap](Q21_R_bootstrap.md) | Algorithm, CIs, .632 estimator, vs CV | Week 2 |
| [Q21-S — Curse of Dimensionality](Q21_S_curse_dimensionality.md) | Volume shell, distance concentration, blessings | Week 3, conceptual + geometric |
| [Q21-T — AIC / BIC](Q21_T_aic_bic.md) | KL derivation, AIC vs BIC, Cp, vs CV | Weeks 1/2, derivation question |
| [Q21-U — Bagging](Q21_U_bagging.md) | Variance formula derivation, OOB, vs RF/Boosting | Week 5, standalone treatment |
| [Q21-V — Cluster Validation](Q21_V_cluster_validation.md) | Silhouette, Gap statistic, BIC for GMM | Week 9, choosing K |
| [Q21-W — Sparse PCA](Q21_W_sparse_pca.md) | PMD, elastic net loadings, vs dense PCA | Week 8, interpretability |
| [Q21-X — QDA](Q21_X_qda.md) | Quadratic boundary, per-class covariance, RDA | Natural LDA extension |
| [Q21-Y — K-medoids (PAM)](Q21_Y_kmedoids.md) | Medoids, any dissimilarity, outlier robustness | Week 9, vs K-means |
| [Q21-Z — Gaussian Mixture Models](Q21_Z_gmm.md) | EM derivation, soft clustering, BIC for K, degenerate solutions | Week 9, deep dive |
| [Q21-AA — Split-Half FMS](Q21_AA_split_half_fms.md) | PARAFAC validation, reproducibility, FMS formula, CORCONDIA vs FMS | Week 12, PARAFAC validation |
| [Q21-AB — PCR vs PLS](Q21_AB_pcr.md) | PCR formula, SVD view, Ridge vs PCR, PCR weakness vs PLS | Week 8, dimension reduction for regression |
| [Q21-AC — Elastic Net](Q21_AC_elastic_net.md) | Coordinate descent, grouping effect, $L_1+L_2$ regularization | Week 3, high-yield extension of Ridge/Lasso |
| [Q21-AD — Gradient Boosting](Q21_AD_gradient_boosting.md) | FSAM, pseudo-residuals, shrinkage, AdaBoost link | Week 6, high-yield ensemble topic |
| [Q21-AE — RDA](Q21_AE_rda.md) | Covariance shrinkage between LDA and QDA, RRDA | Week 4, bias-variance discriminant analysis |
| [Q21-AF — CCA](Q21_AF_cca.md) | Canonical variates, generalized eigenproblem, regularized CCA | Week 8, theory-heavy comparison topic |
| [Q21-AG — KNN](Q21_AG_knn.md) | Local averaging, $K$ tuning, distance concentration, consistency | Week 2, classic nonparametric method |
| [Q21-AH — Sparse Coding](Q21_AH_sparse_coding.md) | Dictionary learning, sparse coefficients, relation to ICA/NMF | Week 11, high-yield representation learning topic |
| [Q21-AI — NMF](Q21_AI_nmf.md) | Non-negativity, multiplicative updates, parts-based interpretation | Week 11, strong standalone decomposition topic |
| [Q21-AJ — Archetypal Analysis](Q21_AJ_archetypal_analysis.md) | Convex hull archetypes, extreme profiles, $XSH$ formulation | Week 11, high-yield interpretability topic |
| [Q21-AK — N-mode Multiplication & Tensor Unfolding](Q21_AK_nmode_tensor_ops.md) | Unfolding, mode products, bridge from matrices to Tucker/PARAFAC | Week 12, technical derivation topic |
| [Q21-AL — LDA vs QDA vs RDA](Q21_AL_lda_qda_rda.md) | Covariance assumptions, boundary shape, bias-variance continuum | Week 4, classic compare-three question |
| [Q21-AM — Bagging vs Random Forest vs Boosting](Q21_AM_bagging_rf_boosting.md) | Variance vs bias reduction, tree correlation, sequential vs parallel | Weeks 5/6, very exam-plausible comparison |
| [Q21-AN — K-means vs GMM](Q21_AN_kmeans_vs_gmm.md) | Hard vs soft clustering, WCSS vs likelihood, EM connection | Week 9, strong clustering comparison |
| [Q21-AO — Generative vs Discriminative Classifiers](Q21_AO_generative_vs_discriminative.md) | LDA/QDA vs logistic vs SVM, assumptions vs robustness | Weeks 4/7, broad compare-and-choose question |
| [Q21-AP — PCR vs PLS vs Ridge](Q21_AP_pcr_pls_ridge.md) | Shrinkage in latent directions, supervision, bias-variance tradeoff | Weeks 1/8, very exam-plausible compare-three question |
| [Q21-AQ — NMF vs ICA vs Sparse Coding](Q21_AQ_nmf_ica_sparse_coding.md) | Parts, independence, sparsity, identifiability | Week 11, strong decomposition comparison |
| [Q21-AR — AIC vs BIC vs Cross-Validation](Q21_AR_aic_bic_cv.md) | Likelihood penalties vs model-agnostic risk estimation | Weeks 1/2, classic model-selection comparison |
| [Q21-AS — Bootstrap vs Cross-Validation vs OOB Error](Q21_AS_bootstrap_cv_oob.md) | Uncertainty vs prediction error vs bagging-based validation | Week 2/5, practical assessment comparison |
| [Q21-AT — KNN vs LDA vs Logistic Regression](Q21_AT_knn_lda_logistic.md) | Local vs generative vs discriminative classification | Weeks 2/4, exam-style compare-and-choose question |
| [Q21-AU — PCA vs Sparse PCA vs NMF](Q21_AU_pca_sparsepca_nmf.md) | Variance directions vs sparsity vs parts-based decomposition | Weeks 8/11, interpretable latent-factor comparison |

---

## Writing Strategy for 20-Point Questions

### Structure Every Answer As:
1. **State the model** — one sentence, formula if relevant
2. **Explain the mechanism** — why does each step work?
3. **State key properties** — bias/variance, uniqueness, complexity
4. **Compare to alternatives** — articulate the key distinction
5. **Limitations** — when does it fail, edge cases

### Where Marks Come From:
- Identifying the correct objective function (not just "minimizes error")
- Explaining WHY each step achieves something
- Comparing two methods and naming the distinguishing property
- Concrete formula or example that supports a claim
- Edge case behavior ($\lambda\to\infty$, $R$ too large, Gaussian sources in ICA, etc.)

### Common Mistakes to Avoid:
- "It minimizes the error" — be specific about which loss and how
- Listing steps without explaining what each achieves
- Forgetting to state assumptions (equal covariance for LDA, non-Gaussian for ICA)
- Confusing variance reduction (bagging) with bias reduction (boosting)
- Saying uniqueness without stating the exceptions (permutation/sign for ICA, $Q$-ambiguity for NMF)

---

## Quick-Reference: Key Formulas to Memorize

| Topic | Formula |
|-------|---------|
| RF variance | $\rho\sigma^2 + (1-\rho)\sigma^2/B$ |
| AdaBoost $\alpha_m$ | $\log\frac{1-\text{err}_m}{\text{err}_m}$ |
| LDA log-ratio | $x^T\Sigma^{-1}(\mu_k-\mu_{k'})-\frac{1}{2}(\mu_k^T\Sigma^{-1}\mu_k-\mu_{k'}^T\Sigma^{-1}\mu_{k'})$ |
| SVM dual | $\max_\alpha\sum_i\alpha_i-\frac{1}{2}\sum_{ij}\alpha_i\alpha_jy_iy_j K(x_i,x_j)$ |
| CORCONDIA | $100(1-\|\mathcal{I}-\tilde{\mathcal{G}}\|_F^2/\|\mathcal{I}\|_F^2)$ |
| ICA negentropy | $J(y)=H(y_\text{Gauss})-H(y)\geq0$ |
| GMM E-step | $\gamma_{ij}=\pi_j\mathcal{N}(x_i|\mu_j,\Sigma_j)/\sum_{j'}\pi_{j'}\mathcal{N}(x_i|\mu_{j'},\Sigma_{j'})$ |
| PARAFAC mode-1 | $X_{(1)}\approx A(C\odot B)^T$ |
| Tucker mode-1 | $X_{(1)}\approx AG_{(1)}(C\otimes B)^T$ |
