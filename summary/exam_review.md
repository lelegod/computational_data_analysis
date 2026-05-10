# Exam Review — CDA 02582

> Covers three exams: May 2022 (handout 2023), May 2024, May 2025 (vFinal).
> Solutions verified against official answer sheets and first-principles reasoning.
> Q19 of the 2022 exam was removed from the 2023 curriculum and is omitted throughout.

---

## Topic Frequency Analysis

| Topic | Week(s) | 2022 | 2024 | 2025 | Total | Notes |
|-------|---------|------|------|------|-------|-------|
| Bias-Variance / EPE decomposition | 1 | Q4, Q5, Q9, Q14 | Q4, Q16 | Q1 | 3/3 | Core concept — always tested |
| Ridge / Lasso / Regularization | 2 | Q4, Q5, Q6, Q9 | Q4, Q5 | Q7 | 3/3 | Formulas, effects of lambda |
| Cross-validation design | 1–2 | Q18 | Q7 | Q4, Q6, Q18, Q19 | 3/3 | Watch for data leakage, dependent obs |
| Multiple testing (BH / Bonferroni) | 3 | Q8, Q10 | Q8 | Q10 | 3/3 | BH vs Bonferroni distinctions |
| Random Forest / Bagging / Ensembles | 5–6 | Q14, Q20, Q21 | Q3, Q11, Q19 | — | 2/3 | Bias = individual tree bias |
| LDA / Classification | 4 | Q3 | Q9 | Q9, Q21 | 3/3 | Gaussian assumptions, linearity reason |
| SVM / Kernel | 7 | Q7 | Q20 | Q2 | 3/3 | Linear vs nonlinear, kernel trick |
| PCA / Subspace methods | 8–9 | Q17 | — | Q14 | 2/3 | Latent variables, variance explained |
| GMM / Probabilistic models | 4 | Q1 | Q9, Q18 | Q13, Q21 | 3/3 | Probabilistic vs hard clustering |
| K-means / Clustering | 9 | Q15 | Q14, Q15 | Q13, Q15, Q20 | 3/3 | Distance metrics, number of clusters |
| Boosting | 6 | Q20 | Q10 | — | 2/3 | Stumps preferred, AdaBoost |
| Neural Networks | 10–11 | — | Q12 | Q12 | 2/3 | Parameter counting, autoencoder |
| ICA | 11 | — | Q21 | Q11 | 2/3 | Non-Gaussianity, uniqueness |
| Archetypal Analysis / NMF | 9–10 | Q15 | Q15 | Q16, Q17 | 3/3 | Extremes, non-negativity |
| Multiway models (PARAFAC/Tucker/CORCONDIA) | 12 | Q16 | Q17 | — | 2/3 | Core tensor, CORCONDIA |
| Model selection (AIC/BIC/Information criteria) | 2 | — | Q6 | Q5, Q8 | 2/3 | BIC penalizes complexity more |
| High-dimensional data (p >> n) | 1–2 | Q2 | Q2 | — | 2/3 | SVM, Elastic Net, RF, PCA |
| Norm definitions (L1, L2) | 2 | Q11 | — | — | 1/3 | $\|\|\beta\|\|_2^2 = \beta^T\beta$ |
| Confusion matrix / Sensitivity/Specificity | 4 | Q13 | — | — | 1/3 | Bayes theorem application |
| Self-Organizing Maps / Manifold | 8 | Q12 | — | — | 1/3 | Blessings of dimensionality |
| K-medoids | 9 | — | — | Q15 | 1/3 | Robustness to outliers |
| Nested CV | 2 | — | — | Q6 | 1/3 | Unbiased after hyperparameter tuning |
| Open Q: Random Forest algorithm | 5–6 | Q21 | — | — | 1/3 | |
| Open Q: Clustering for face data | 9 | Q22 | — | — | 1/3 | |
| Open Q: ICA uniqueness | 11 | — | Q21 | — | 1/3 | |
| Open Q: Cross-validation design for wearables | 1–2 | Q18 | Q22 | Q22 | 3/3 | Personalized vs generalized model |

---

## Exam 2022/2023 — Question-by-Question Review

**Format note:** The 2022 exam uses a "one correct combination" rule (2 pts correct, 0 pts wrong). The answer sheet marks multiple letters — all marked letters together form the correct answer combination.

### Q1: Probabilistic Models of Data
- **Question**: Which methods are based on probabilistic models of data?
- **Official Answer**: C (GMM), D (Logistic Regression)
- **Verification**: ✓ Correct
- **Notes**: GMM explicitly models class-conditional Gaussian densities. Logistic regression is derived from a probabilistic generative model (exponential family). SVM is a geometric/margin method — not probabilistic. K-means is a hard-assignment iterative algorithm, not probabilistic in itself.

### Q2: Methods Handling p >> n
- **Question**: Which methods can handle data with fewer observations than dimensions?
- **Official Answer**: A (SVM), C (Random Forest), D (PCA)
- **Verification**: ✓ Correct
- **Notes**: SVM works via the dual formulation (depends on n, not p). Random Forest uses random subsets of features, so p >> n is manageable. PCA reduces dimensionality. Logistic regression WITHOUT regularization cannot handle p > n (non-invertible $X^TX$).

### Q3: LDA Statements
- **Question**: Which statements are true for LDA?
- **Official Answer**: C (LDA is a probabilistic method), D (linear because Gaussian + equal covariance assumption)
- **Verification**: ✓ Correct
- **Notes**: D is the key mechanical reason for linearity — equal covariance matrices cause the quadratic terms in the log-ratio of class-conditional densities to cancel, leaving a linear boundary. C is true because LDA models class-conditional Gaussians and uses Bayes' rule. A is wrong — the decision boundary being linear is a CONSEQUENCE of the Gaussian equal-covariance assumption, not an assumption itself. B is wrong — LDA is actually sensitive to outliers because it uses the mean and covariance directly.

### Q4: Ridge — Too Large Lambda
- **Question**: Too large lambda in ridge regularization — effect on model?
- **Official Answer**: A (High bias)
- **Verification**: ✓ Correct
- **Notes**: Large $\lambda$ over-shrinks coefficients toward zero → underfitting → high bias, low variance. This is a fundamental bias-variance tradeoff question.

### Q5: Lasso — Too Small Lambda
- **Question**: Too small lambda in lasso regularization — effect on model?
- **Official Answer**: B (High variance), D (Low bias)
- **Verification**: ✓ Correct
- **Notes**: Small $\lambda$ means little regularization → model approximates OLS → low bias but high variance. Both B and D are simultaneously correct (they are two sides of the same coin at small lambda). This question correctly identifies that both can be marked.

### Q6: Lasso Estimation
- **Question**: How are lasso estimates calculated?
- **Official Answer**: C (Solved numerically, no analytical solution)
- **Verification**: ✓ Correct
- **Notes**: The L1 norm ($\|\beta\|_1$) is not differentiable at zero, so no closed-form analytical solution exists. Algorithms like coordinate descent or LARS are used. Option A is the ridge solution. Option B is OLS. Option D describes ridge, not lasso (wrong norm).

### Q7: SVM — Linear or Non-linear?
- **Question**: Is SVM linear or non-linear?
- **Official Answer**: C (Depends on the chosen kernel)
- **Verification**: ✓ Correct
- **Notes**: With a linear kernel, SVM gives a linear boundary. With RBF, polynomial, or other kernels, boundaries are non-linear. The kernel trick allows implicit mapping to high-dimensional spaces.

### Q8: Multiple Testing Techniques
- **Question**: Which techniques handle the multiple testing problem?
- **Official Answer**: B (Bonferroni), D (Benjamini-Hochberg)
- **Verification**: ✓ Correct
- **Notes**: AIC is for model selection, not multiple testing. Bootstrapping is a resampling technique for estimation/inference but not specifically for multiple testing. Both Bonferroni (FWER control) and BH (FDR control) are the canonical multiple testing corrections.

### Q9: Why Not Penalize the Intercept?
- **Question**: Why do we not penalize the intercept in regularization?
- **Official Answer**: A(x), B(x), C(x), D — The (x) notation indicates these are debated/all could be partially right; D is the primary answer: "The model will get a lower Expected Prediction Error if we do not penalize the intercept."
- **Verification**: ⚠️ Questionable presentation
- **Notes**: The canonical answer taught in most ML courses is A — penalizing the intercept introduces bias without any variance reduction, because the intercept is not associated with input complexity. The intercept merely shifts predictions and its penalization shrinks predictions toward zero rather than toward the true mean. D is a consequence of A, not an independent reason. The (x) marks on A, B, C suggest the examiners considered these contentious. The clearest and most defensible answer is **A**: penalizing the intercept would introduce bias without variance reduction. D is correct as a consequence but circular (it restates the conclusion). B is incorrect — penalizing intercept does not introduce variance. C is partially true but less precise. **Recommended answer: A** (or A and D if multiple are allowed).

### Q10: Bonferroni vs. FDR Statements
- **Question**: Which statements about Bonferroni and FDR corrections are correct?
- **Official Answer**: A (Bonferroni reduces risk of false positives), C (FDR 5% gives more significant findings than Bonferroni 5%), D (FDR 5% gives more false positives than Bonferroni 5%)
- **Verification**: ✓ Correct
- **Notes**: A is the primary purpose of Bonferroni. C is correct — Bonferroni is more conservative, so BH at same alpha level will call more findings significant. D is the direct consequence of C — more findings means more false positives (by design, BH allows up to 5% of discoveries to be false). B is wrong: Bonferroni REDUCES the chance of accepting a false positive (it raises the bar to reject H0), so it actually reduces the chance of rejecting the null, not accepting it.

### Q11: L2 Norm Squared
- **Question**: What is $\|\beta\|_2^2$ equal to?
- **Official Answer**: B ($\sum \beta_i^2$), C ($\beta^T \beta$)
- **Verification**: ✓ Correct
- **Notes**: Both B and C are equivalent definitions of the squared L2 norm. A is the L1 norm. D is the max norm squared (L∞ type). The question asks for the squared L2 norm which is the sum of squares = $\beta^T \beta$.

### Q12: Self-Organizing Maps and Blessings of Dimensionality
- **Question**: SOMs illustrate which blessing of dimensionality?
- **Official Answer**: B (Informative data will lie on a low-dimensional manifold)
- **Verification**: ✓ Correct
- **Notes**: SOMs learn a low-dimensional grid that maps onto the manifold structure of high-dimensional data. This directly illustrates the manifold hypothesis. C (approximative finite dimensionality) is related but B is the more specific and direct connection to SOMs.

### Q13: Covid-19 Test — Expected Positive Tests
- **Question**: In 10,000 subjects (100 with Covid, 9900 without), how many expected positive tests?
- **Official Answer**: D (297)
- **Verification**: ✓ Correct
- **Calculation**:
  - True positives: $100 \times 0.99 = 99$
  - False positives: $9900 \times 0.02 = 198$
  - Total positives: $99 + 198 = \mathbf{297}$
- **Notes**: This requires applying both sensitivity (99%) and false positive rate (2%) to their respective populations. A common trap is to only count true positives (99) or only round to 100.

### Q14: Random Forest Bias-Variance
- **Question**: With m=5, 50 variables, 100 obs (all relevant), what holds for RF bias/variance?
- **Official Answer**: A (variance < bias), D (variance of single tree > variance of ensemble)
- **Verification**: ✓ Correct
- **Notes**: D is certainly true — averaging in RF always reduces variance vs a single tree. A is correct in this setting because all 50 variables are informative but m=5 means we miss 45 each split → high bias from random feature subsetting on a problem where all features matter. The ensemble reduces variance but the bias from missing variables dominates.

### Q15: Archetypical Analysis — Which Dataset Fits
- **Question**: True statements about the 6 datasets and Archetypal Analysis/K-means?
- **Official Answer**: B (SVD with two components describes all variance for all datasets)
- **Verification**: ⚠️ Questionable
- **Notes**: The datasets are 2-dimensional, so SVD with 2 components will always capture 100% of variance — this is trivially true for any 2D dataset. However, option D states "K-means with two components is appropriate for datasets a, b, and c" — from the figures, datasets a, b, c appear to have elongated/linear structure that doesn't naturally form 2 clusters. Option B is technically true (2 SVD components = full variance in 2D) but it is a trivial statement. **The official answer B is technically correct but pedagogically weak.** The question tests whether students know SVD captures all variance with sufficient components.

### Q16: CORCONDIA
- **Question**: True statements about CORCONDIA?
- **Official Answer**: E (None of the above)
- **Verification**: ✓ Correct
- **Notes**: Detailed analysis: A is wrong — I (the super-identity tensor) is super-diagonal, not a plain diagonal matrix. B is wrong — G is the core tensor from a PARAFAC decomposition approximated as a Tucker core, not from Tucker itself. C is partially right in spirit but CORCONDIA specifically helps select the number of PARAFAC components, not choose between PARAFAC and Tucker. D is wrong — we choose the MAXIMUM (close to 100), not minimum. Hence E (none of the above) is correct.

### Q17: Subspace Methods — Latent Variables
- **Question**: True statements about PCA, PLS, and CCA?
- **Official Answer**: A (all result in linear combinations of input), B (all impose orthogonality), D (elastic net can produce sparse versions of all)
- **Verification**: ⚠️ Questionable — B is debatable
- **Notes**: A is correct — PCA, PLS, and CCA all form latent variables as linear combinations of inputs. D is correct — sparse PCA (elastic net penalty on loadings), sparse PLS, and sparse CCA all exist. B is partially correct for PCA (PC scores are orthogonal), but for PLS and CCA, the X-scores are orthogonal in PLS, however CCA components are not necessarily orthogonal in the input space — they are orthogonal in a different sense. **B may be considered questionable for CCA.** C is wrong — PLS and CCA are supervised (they use the Y matrix). The official answer of A, B, D appears broadly accepted in this course's framework.

### Q18: Cross-Validation for Wearable Activity Prediction
- **Question**: Which CV method helps assess prediction accuracy for new individuals and new weeks?
- **Official Answer**: C (Leave-five-individuals-out), D (Leave-one-week-out)
- **Verification**: ✓ Correct
- **Notes**: The goal is to predict for BOTH new individuals AND next week's data. Options A and B (standard and leave-one-observation-out CV) would mix individuals' data across folds — inflating performance by allowing the same individual's data in both training and test. C ensures whole individuals are held out. D ensures future weeks are held out. Both structural considerations are needed for the stated aim.

### Q19: Removed from 2023 curriculum
- (Not applicable)

### Q20: Boosting — Suitable Individual Models
- **Question**: Which methods are suitable as individual models in a boosting ensemble?
- **Official Answer**: A (KNN high K), E (None — wait, let me re-check)
- **Re-reading the grid**: Row A: col 20 = x; Row E: col 20 = x → Answer is A and E? That doesn't make sense. Re-reading: Answer E for Q20 has an x. But "None of the above" + "KNN high K" is contradictory.
- **Correction**: Looking at the grid again: A has x at Q20, E has x at Q20. This is likely a grid rendering issue. The pedagogically correct answer is **C (Any classification or regression tree)**. Boosting uses weak learners — shallow trees (stumps) are standard. KNN with high K underfits but is not differentiable (no gradient for gradient boosting). KNN with low K is high-variance but not a standard weak learner for boosting. The intended answer is almost certainly **C**.
- **Verification**: ⚠️ Grid ambiguity — official answer appears to be A+E which is internally contradictory; **likely a grid transcription error**. Correct answer is C.

### Q21 (Open): Random Forest Algorithm
- **Question**: Describe the steps of the Random Forest algorithm and how they contribute to performance.
- **Key points for full marks**:
  1. Bootstrap sampling (bagging) — creates diverse training sets, reduces variance
  2. Random feature selection at each split ($m \ll p$) — decorrelates trees, further reduces variance
  3. Grow deep (unpruned) trees — low bias per tree
  4. Average predictions (regression) or majority vote (classification) — reduces variance without increasing bias
  5. OOB (out-of-bag) error estimate — free internal validation
  6. Variable importance via permutation or Gini impurity decrease

### Q22 (Open): Face Image Clustering for Unique People Count
- **Question**: Which methodology to find number of unique people from face images?
- **Key points for full marks**:
  1. Extract features from images (e.g., PCA/SVD on face images, or deep embeddings)
  2. Cluster the feature vectors — K-means or hierarchical clustering, or GMM
  3. Determine optimal K (number of unique people) using silhouette scores, elbow method, or BIC (for GMM)
  4. This is unsupervised — no labels available
  5. Archetypical analysis could also be mentioned
  6. Compare K (unique face clusters) to unique passport numbers to detect fraud

---

## Exam 2024 — Question-by-Question Review

**Format note:** The 2024 exam uses +1/-1 scoring per answer option (multiple correct answers per question possible). The answer sheet marks which options are correct.

### Q1: Supervised Methods
- **Question**: Which methods are supervised?
- **Official Answer**: E (None of the above)
- **Verification**: ✓ Correct
- **Notes**: GMM is unsupervised (fits to unlabeled data). Autoencoder is unsupervised (reconstructs input). K-means is unsupervised. Tucker decomposition is unsupervised (tensor factorization). None of the listed methods uses labeled output for training.

### Q2: Methods for p >> n
- **Question**: Which methods handle more variables than observations well?
- **Official Answer**: A (Elastic Net), B (PCA), D (Random Forest)
- **Verification**: ✓ Correct
- **Notes**: Elastic Net (L1+L2 regularization) handles high-dimensional data. PCA reduces dimensionality first. RF uses random subsets of features. OLS (C) fails when p > n (singular $X^TX$). Note: All three correct answers (A, B, D) are valid.

### Q3: Random Forest True Statements
- **Question**: True statements about Random Forest?
- **Official Answer**: D (Fully grown trees are more suitable than stumps as individual models in RF)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — fewer random variables per split leads to LESS correlated trees (greater variance reduction through averaging), but each individual tree has higher variance. The net effect on the ensemble variance compared to more variables is nuanced, but the statement says "smaller reduction in variance" which is wrong — fewer variables leads to more variance reduction in the ensemble. B is wrong — RF CAN be parallelized because each tree is grown independently on a bootstrap sample. C is wrong — RF specifically uses trees (the random feature selection is tree-specific); KNN cannot perform the same random-split procedure. D is correct — RF bias = individual tree bias, so deep trees (low bias) are preferred.

### Q4: Lasso — Too Small Lambda
- **Question**: Too small lambda in lasso — effect?
- **Official Answer**: B (High variance), D (Low bias)
- **Verification**: ✓ Correct
- **Notes**: Same logic as 2022 Q5. Small $\lambda$ = weak regularization = near-OLS solution = low bias, high variance.

### Q5: Regularized Regression Algorithms
- **Question**: True statements about regularized regression?
- **Official Answer**: B (Lasso is a path algorithm), C (Path algorithm provides solutions for all relevant lambda values)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — Lasso has no closed-form solution (L1 non-differentiable). B is correct — LARS/LASSO is a path algorithm. C is correct — path algorithms efficiently trace the full regularization path. D is wrong — closed-form solutions (like ridge: $(X^TX + \lambda I)^{-1}X^TY$) do not require asymptotic theory; they are exact for finite n.

### Q6: Information Criteria Statements
- **Question**: True statements about information criteria?
- **Official Answer**: D (BIC is comparing models according to their posterior odds)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — in-sample error is defined as error on the TRAINING data (not a new training set). B is wrong — AIC is derived assuming p << n; it breaks down when p is comparable to or larger than n. C is wrong — BIC requires a correctly specified parametric model (likelihood); it does not work for all supervised models broadly. D is correct — BIC approximates the log marginal likelihood, and comparing two BIC values approximates log Bayes factors (posterior odds).

### Q7: Cross-Validation True Statements
- **Question**: True statements for running cross-validation?
- **Official Answer**: B (Double-loop CV assists in both model selection and assessment), C (Dependent observations should stay in same fold), D (CV provides reasonable EPE estimates)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — normalization MUST be performed WITHIN each fold (using only training fold statistics), NOT before running CV. Normalizing before CV introduces data leakage. B is correct — outer loop = assessment, inner loop = selection. C is correct — this prevents information leakage between dependent observations (e.g., time series, repeated measures). D is correct — CV is the standard method for EPE estimation.

### Q8: Multiple Testing Statements
- **Question**: True statements about multiple testing?
- **Official Answer**: A (BH controls upper bound of FDR)
- **Verification**: ✓ Correct
- **Notes**: A is the defining property of BH. B is wrong — family-wise error rate is the probability of AT LEAST ONE false rejection (not "at most $\alpha/M$"). C is wrong — Bonferroni is most useful when there are FEWER tests (more tests make it too conservative). D is wrong — the plug-in FDR estimate IS a consistent estimate under certain conditions (specifically, it tends to overestimate FDR, making it conservative, which is actually desirable).

### Q9: Classification Statements
- **Question**: True statements about classification?
- **Official Answer**: C (RF, Sparse DA, Regularized Logistic Regression all suitable for p > n)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — GMM is UNSUPERVISED; in the discriminant analysis context, it fits Gaussians per class, but the statement misdescribes it. B is wrong — CART handles categorical variables natively through split rules, NOT by one-hot encoding. C is correct — all three are regularized and suitable for high-dimensional data. D is wrong — SVM's nonlinearity comes from the kernel, NOT simply from using the dual formulation. The dual formulation is just an equivalent mathematical reformulation.

### Q10: Boosting Statements
- **Question**: True statements about boosting?
- **Official Answer**: B (Forward stagewise: consecutively adding trees), C (AdaBoost.M1 approximates exponential loss in forward stagewise)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — forward stagewise does NOT update weights of previous trees; once added, a tree's weight is fixed. B is correct — new tree+weight pairs are added sequentially without changing existing ones. C is correct — this is a key theoretical result (Friedman et al. 2000). D is wrong — the exponential loss gives HIGHER weight to misclassified observations ($e^{-y_i f(x_i)}$ grows large when f and y disagree), compared to misclassification loss which gives equal weight to all misclassified.

### Q11: RF Variable Importance
- **Question**: True statements about RF variable importance?
- **Official Answer**: A (Gini VI = aggregation of gini index at splits containing variable j), C (Proximity plots measure closeness of observations), D (Deep trees are good because RF bias = individual tree bias)
- **Verification**: ⚠️ C is questionable
- **Notes**: A is correct — Gini importance sums the Gini decrease at each split involving variable j across all trees. D is correct — since RF bias = single tree bias, using deeper trees (lower bias) leads to lower RF bias. C is problematic: proximity plots measure closeness of OBSERVATIONS (how often two observations end up in the same terminal node), NOT variables. The statement says "closeness of variables" which is wrong. Additionally the comparison to Ridge regression is not standard. **C appears to be an error in the official solution.** B is wrong — OOB importance uses permutation on OOB samples within trees that do NOT contain those OOB samples (correct), but the description about "aggregates the gini index only of trees that do not contain the OOB samples" is inaccurate.

### Q12: Neural Networks Statements
- **Question**: True statements about neural networks?
- **Official Answer**: A (Autoencoder loss = difference between input and output), C (parameter count = 31)
- **Verification**: ⚠️ Parameter count needs checking
- **Parameter count calculation (2024 Q12)**:
  - 10 inputs → 2 hidden units (layer 1): $10 \times 2$ weights + 2 biases = 22
  - 2 hidden (layer 1) → 2 hidden units (layer 2): $2 \times 2$ weights + 2 biases = 6
  - 2 hidden (layer 2) → 1 output: $2 \times 1$ weights + 1 bias = 3
  - Total: $22 + 6 + 3 = \mathbf{31}$ ✓
- **Notes**: A is correct — autoencoder minimizes reconstruction error $\|x - \hat{x}\|^2$. B is wrong — neural networks are notorious for overfitting. C is correct. D is wrong (40 is simply wrong arithmetic).

### Q13: Least Angle Regression vs Coordinate Descent for Lasso
- **Question**: Comparing LARS and Coordinate Descent for Lasso?
- **Official Answer**: B (Both update one variable at a time), E (None of the above — wait, re-checking)
- **Re-reading 2024 grid**: Q13: B=x, E=x → Answer B and E. But E="None of the above" contradicts B being correct.
- **Correction**: The grid shows Q13 has Answer E marked. Looking again at the 2024 solution table for Q13: Answer E has an x. This would mean "None of the above." But answer B also has no x in the 2024 table for Q13... Let me re-read the 2024 table: Row B, col 13 is blank. Row E, col 13 has x.
- **Re-decoded 2024 answer for Q13: E (None of the above)**
- **Verification**: ⚠️ Questionable
- **Notes**: A is partially correct — LARS is a path algorithm; Coordinate Descent is also a path-type algorithm but not strictly called one. B is partially right for Coordinate Descent (updates one variable at a time) but LARS updates equiangularly and doesn't strictly update one variable. C is wrong — LARS gives the Lasso solution WITH modification (the LARS-Lasso modification); standard LARS gives a related but different solution. D is wrong — neither makes more assumptions about data than the other. **E (None of the above) is defensible** because none of A–D is cleanly correct for both algorithms simultaneously.

### Q14: Clustering True Statements
- **Question**: True statements about clustering?
- **Official Answer**: B (Can use different distance metrics for hierarchical clustering)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — Manhattan distance measures absolute distances between CONTINUOUS variables; for categorical variables, Hamming distance or Gower distance is used. B is correct — hierarchical clustering supports Euclidean, Manhattan, correlation-based distances, etc. C is wrong — AIC/BIC require a likelihood model; K-means doesn't have one. You can use elbow method, gap statistic, or silhouette score instead. D is wrong — linkage (single, complete, average, Ward) decides how distances between clusters are computed; agglomerative vs divisive is a separate choice about the direction of clustering.

### Q15: K-means Clustering Plots — True Statements
- **Question**: True statements about the 6 datasets with K-means (2 clusters)?
- **Official Answer**: B (K-means with two components is appropriate for datasets a and c)
- **Verification**: ⚠️ Questionable
- **Notes**: From the figure, dataset a has a triangular/wedge shape (VE=0.80) and dataset c appears scattered (VE=0.75). Dataset f has a clear two-cluster structure (VE=0.93). Dataset b is elongated. The claim that a and c are appropriate for 2-cluster K-means while f is not is counterintuitive from the VE values. However, "appropriate" in the context of this course likely refers to whether the data has clearly separable globular clusters matching K-means' spherical assumption. This answer depends heavily on visual interpretation of figures not fully reproducible here. Official answer B is accepted but visually debatable.

### Q16: EPE, Bias, Variance Statements
- **Question**: True statements about EPE, bias, variance?
- **Official Answer**: C (EPE is expectation over data samples; generalization error is conditioned on fixed training set)
- **Verification**: ✓ Correct
- **Notes**: A is wrong — variance of a model is not estimated by MSE on a test set; test MSE estimates EPE, not variance alone. B is wrong — bias is the difference between the EXPECTED model estimate (over training datasets) and the true value, not the estimate on one training set. D is wrong for the same reason as B, but using test data instead. C is correct and matches the formal definition (ESL Chapter 7).

### Q17: Multiway Models (CONCORDIA)
- **Question**: True statements about multiway models?
- **Official Answer**: A (CONCORDIA close to 100 → PARAFAC has suitable components because core tensor is close to diagonal), D (Dimensionality of Tucker core defines ranks in modes)
- **Verification**: ✓ Correct
- **Notes**: A is the defining interpretation of CORCONDIA. B is wrong — CORCONDIA is defined relative to PARAFAC (not Tucker). C is wrong — the Tucker core is NOT super-diagonal; the super-diagonal core is the PARAFAC model's ideal case. D is correct — the Tucker core has dimensionality $R_1 \times R_2 \times R_3$ for a 3-mode tensor, and these define the rank in each mode.

### Q18: DNA Microarray — Cancer Type Clustering
- **Question**: 1000 features, 820 individuals with known cancer types; investigate if similar DNA expression shares same cancer type?
- **Official Answer**: B (Hierarchical clustering)
- **Verification**: ⚠️ Questionable
- **Notes**: The question says we KNOW the cancer type and want to INVESTIGATE if similar expressions share cancer type. This sounds like a supervised question (classification/discriminant analysis) OR a visualization question (clustering then comparing to labels). A (RF classification) would also work to predict and assess whether expression separates cancer types. B (hierarchical clustering) with color-coded cancer labels on a dendrogram is a classic bioinformatics approach. D (logistic regression without regularization) is impossible with p=1000 > n=820. C (RF regression) doesn't fit a classification problem. **Both A and B could be argued, but B is the official answer** and fits the "investigate if similar" framing (exploration vs. prediction).

### Q19: Bagging — Methods That Benefit Most
- **Question**: Which methods achieve significant EPE reduction when used as individual models in bagging?
- **Official Answer**: C (KNN with small K)
- **Verification**: ✓ Correct
- **Notes**: Bagging reduces variance without changing bias. To benefit, the base model must have HIGH variance (so averaging helps) and LOW bias (so the average is accurate). KNN with small K is high variance, low bias — prime candidate for bagging. A (CART stump) is low variance, high bias — bagging won't help much (bias stays high). B (KNN high K) is low variance, low bias — variance is already low, bagging helps little. D (the mean) has zero variance — bagging provides no improvement.

### Q20: Kernel Trick — Which Methods Support It?
- **Question**: Which methods can be formulated to apply the kernel trick?
- **Official Answer**: A (SVM), C (PCA — kernel PCA)
- **Verification**: ✓ Correct
- **Notes**: SVM naturally uses kernels in its dual formulation. Kernel PCA exists and is well-established. Boosting (B) is a stagewise additive model — no natural kernel formulation. Random Forest (D) uses random splits on original features — no kernel formulation. The kernel trick requires expressing the algorithm purely in terms of inner products, which both SVM and PCA (via the Gram matrix) support.

### Q21 (Open): ICA — Uniqueness and Distributions
- **Question**: Demonstrate whether ICA is unique, describe independent components, and favored distributions.
- **Key points for full marks**:
  1. ICA is NOT unique up to: (a) permutation of components, (b) scaling/sign of components. These are fundamental indeterminacies.
  2. Independent components: components are statistically independent (joint pdf = product of marginals): $p(s_1, s_2) = p(s_1) \cdot p(s_2)$
  3. ICA requires non-Gaussian sources — Central Limit Theorem causes mixtures to be MORE Gaussian; ICA exploits this by finding the "least Gaussian" directions
  4. Favored distributions: super-Gaussian (heavy-tailed, high kurtosis) or sub-Gaussian distributions. Standard Gaussian cannot be used as a source distribution because ICA is unidentifiable for Gaussian sources.
  5. FastICA uses negentropy or kurtosis as the contrast function.

### Q22 (Open): Cross-Validation Design for Wearable Biosignals
- **Question**: Predict activity from biosignals (16 persons, 3 conditions, 4 seasons). Design CV for a) one individual, b) new individual.
- **Key points for full marks**:
  - a) Personalized model: use only that individual's data (48 observations: 3×4×... seasons×conditions). Use leave-one-season-out or leave-one-condition-out CV within that individual.
  - b) Generalized model: use leave-one-individual-out CV — train on 15 individuals, test on the held-out individual. Repeat for all 16.
  - Note the dependency structure: multiple observations per individual (repeated measures) — mixing individuals across folds would be a data leakage error.
  - Generalized model is more appropriate for clinical deployment (new patients are unknown individuals).

---

## Exam vFinal (2025) — Question-by-Question Review

**Format note:** The 2025 exam has single correct answers (a)–(e) per question. Official answers are highlighted in the solution PDF.

### Q1: Bias-Variance — Component Not Affected by Model Complexity
- **Question**: Which component is NOT affected by model complexity?
- **Official Answer**: (c) Irreducible error
- **Verification**: ✓ Correct
- **Notes**: The irreducible error ($\sigma^2$ — noise in the data) is a property of the data-generating process and cannot be reduced by any model. Bias decreases with complexity, variance increases with complexity, EPE is their sum plus irreducible error. Training error also changes with complexity (decreases).

### Q2: Kernel Trick in SVM
- **Question**: The kernel trick in SVM allows?
- **Official Answer**: (b) Computation in high-dimensional feature space without explicit transformation
- **Verification**: ✓ Correct
- **Notes**: The kernel trick $K(x,x') = \phi(x) \cdot \phi(x')$ allows implicit computation in a (possibly infinite-dimensional) feature space using only dot products in the original space. No feature selection, no automatic regularization, no speed increase in general.

### Q3: Not Matrix Factorization-Based
- **Question**: Which method is NOT matrix factorization-based?
- **Official Answer**: (d) K-means
- **Verification**: ✓ Correct
- **Notes**: NMF ($X \approx WH$), PCA (SVD), ICA ($X = AS$, matrix mixing model), Archetypal Analysis ($X \approx XZS$, matrix formulation) are all matrix factorization methods. K-means is a clustering algorithm based on distance to centroids — it minimizes within-cluster sum of squares; it is NOT a matrix factorization (though it can be loosely related to a special case of NMF, it is not natively expressed as one).

### Q4: IID Assumption for Cross-Validation
- **Question**: Implicit assumption behind standard CV validity?
- **Official Answer**: (c) The data are independently and identically distributed (IID)
- **Verification**: ✓ Correct
- **Notes**: Standard CV randomly assigns observations to folds, which is only valid if observations are exchangeable (IID). Time series, spatial data, or repeated measures violate this, requiring structured CV (time-based splits, group-based folds).

### Q5: AIC/BIC vs Cross-Validation Assumptions
- **Question**: Best distinction between information criteria and CV in terms of assumptions?
- **Official Answer**: (a) AIC/BIC assumes correctly specified likelihood model; CV makes fewer assumptions
- **Verification**: ✓ Correct
- **Notes**: AIC $= -2 \log L + 2p$ requires you to specify and maximize a likelihood. BIC is similar. CV is model-agnostic and only requires the ability to compute a prediction error. B is wrong — CV makes no normality or linearity assumption. C is wrong — AIC/BIC estimate in-sample corrected error; CV estimates out-of-sample error.

### Q6: When is Nested CV Preferred?
- **Question**: When is nested CV preferred?
- **Official Answer**: (c) When you want an unbiased estimate of generalization error after hyperparameter tuning
- **Verification**: ✓ Correct
- **Notes**: Without nested CV, using the same data for hyperparameter tuning and performance estimation introduces optimistic bias. The outer loop of nested CV provides an unbiased generalization error estimate; the inner loop handles hyperparameter selection.

### Q7: Why Ridge Does Not Perform Feature Selection
- **Question**: Why does Ridge not perform feature selection?
- **Official Answer**: (c) It shrinks coefficients but doesn't set them to zero
- **Verification**: ✓ Correct
- **Notes**: Ridge's L2 penalty shrinks all coefficients toward zero proportionally but never exactly to zero (the circular constraint region touches the axes only at infinity). Lasso's L1 penalty has corners at the axes, enabling exact zeros. This is a fundamental geometric distinction.

### Q8: Model Criterion That Penalizes Complexity Most as N → ∞
- **Question**: Which criterion penalizes complexity most as $N \to \infty$?
- **Official Answer**: (b) BIC
- **Verification**: ✓ Correct
- **Notes**: AIC penalty $= 2p$ (constant in N). BIC penalty $= p \cdot \log(N)$, which grows without bound as N increases. Cp is related to AIC. Cross-validation has no explicit penalty term. Therefore BIC penalizes complexity increasingly harshly as sample size grows → consistent model selection.

### Q9: Why LDA Boundary is Linear
- **Question**: LDA decision boundary is linear because?
- **Official Answer**: (e) It assumes equal class covariances
- **Verification**: ✓ Correct
- **Notes**: When class covariances are equal ($\Sigma_k = \Sigma$ for all k), the quadratic terms in the log-posterior ratio cancel, leaving a linear function of x. With unequal covariances (QDA), the boundary is quadratic. Equal priors (b) alone do not cause linearity.

### Q10: BH vs Bonferroni
- **Question**: Best distinction between BH and Bonferroni?
- **Official Answer**: (a) BH controls expected proportion of false discoveries (FDR); Bonferroni controls probability of at least one false discovery (FWER)
- **Verification**: ✓ Correct
- **Notes**: This is the defining theoretical distinction. BH controls $\text{FDR} = E[V/R]$, where V = false discoveries, R = total discoveries. Bonferroni controls $\text{FWER} = P(V \geq 1)$. BH is less conservative than Bonferroni, resulting in more rejections.

### Q11: Why ICA Can Recover Sources PCA Cannot
- **Question**: Why can ICA recover sources PCA cannot?
- **Official Answer**: (c) ICA maximizes non-Gaussianity to find statistically independent components; PCA finds uncorrelated directions
- **Verification**: ✓ Correct
- **Notes**: PCA finds uncorrelated components (zero second-order covariance) — uncorrelated ≠ independent for non-Gaussian data. ICA seeks full statistical independence by exploiting higher-order statistics (kurtosis, negentropy). For Gaussian sources, ICA and PCA are equivalent (and ICA is unidentifiable). A is wrong (the assumptions are reversed). B is wrong (PCA maximizes variance, ICA minimizes Gaussianity).

### Q12: Neural Network Parameter Count
- **Question**: 3 inputs → 4 nodes (ReLU) → 2 nodes (ReLU) → 1 output (linear), with biases. Total parameters?
- **Official Answer**: (d) 29
- **Verification**: ✓ Correct
- **Calculation**:
  - Input (3) → Hidden 1 (4): $3 \times 4$ weights + 4 biases = $12 + 4 = 16$
  - Hidden 1 (4) → Hidden 2 (2): $4 \times 2$ weights + 2 biases = $8 + 2 = 10$
  - Hidden 2 (2) → Output (1): $2 \times 1$ weights + 1 bias = $2 + 1 = 3$
  - Total: $16 + 10 + 3 = \mathbf{29}$ ✓
- **Notes**: Contrast with 2024 Q12 (10→2→2→1 = 31). Always count: (inputs_to_layer × units_in_layer) + units_in_layer (for biases).

### Q13: K-means vs GMM Key Difference
- **Question**: Key difference between K-means and GMM?
- **Official Answer**: (c) GMM models data distribution probabilistically allowing elliptical clusters; K-means minimizes squared Euclidean distance to centroids
- **Verification**: ✓ Correct
- **Notes**: A is wrong (reversed — GMM allows different covariances; K-means implicitly assumes spherical). B is wrong (reversed — K-means assigns hard labels; GMM assigns soft memberships via posterior probabilities). D is wrong — GMM uses means, not medians. E is wrong — K-means uses iterative assignment, not likelihood; GMM uses EM (likelihood maximization).

### Q14: PCA Variance Explained
- **Question**: Eigenvalues $\lambda_1 = 6$, $\lambda_2 = 2$. Fraction explained by first PC?
- **Official Answer**: (d) 75%
- **Verification**: ✓ Correct
- **Calculation**: $6/(6+2) = 6/8 = 0.75 = \mathbf{75\%}$ ✓

### Q15: K-medoids vs K-means Fundamental Difference
- **Question**: Fundamental difference between K-medoids and K-means?
- **Official Answer**: (d) K-medoids is more robust to outliers than K-means
- **Verification**: ✓ Correct
- **Notes**: K-medoids selects actual data points as cluster centers (medoids), not computed means. Because medoids are constrained to be data points, they are less influenced by extreme outliers compared to means. A is wrong (reversed — K-means minimizes squared distances; K-medoids minimizes absolute/Manhattan distances). B is wrong (reversed — K-means centroids can lie outside data cloud; K-medoids are always data points). C is wrong (opposite of D).

### Q16: NMF Defining Characteristic
- **Question**: Defining characteristic of NMF?
- **Official Answer**: (b) NMF seeks W and H such that $X \approx WH$, with all entries of W and H constrained to be non-negative
- **Verification**: ✓ Correct
- **Notes**: The key constraint is non-negativity of BOTH factor matrices W and H. This is NOT the same as requiring X to be non-negative (though in practice X is usually non-negative). A is wrong — NMF solutions are generally not unique. C is wrong — no orthogonality constraints in standard NMF. D is wrong — the constraint is on W and H, not only X. E is wrong — X need not be square.

### Q17: Archetypal Analysis Defining Characteristic
- **Question**: Defining characteristic of Archetypal Analysis?
- **Official Answer**: (c) Archetypes are constructed as weighted combinations of extreme observations, and each data point is approximated by a weighted mixture of these archetypes
- **Verification**: ✓ Correct
- **Notes**: The key features: (1) archetypes lie on the convex hull of the data (extreme observations), (2) archetypes are themselves convex combinations of data points, (3) each data point is reconstructed as a convex combination of archetypes. A describes PCA. B describes K-means. D and E are fabricated.

### Q18: Standardization Within CV Folds
- **Question**: Why standardize within each fold rather than using full-dataset statistics?
- **Official Answer**: (b) Using full-data statistics can cause information leakage from the test fold into the training process
- **Verification**: ✓ Correct
- **Notes**: If you compute mean/std from the entire dataset before splitting, the test fold's statistics influence the normalization applied to training data — this is data leakage. Within-fold standardization ensures the test fold is truly unseen during preprocessing. A is wrong — this doesn't reduce variance of CV estimates per se. C is wrong — standardization matters for regularized linear models (Ridge, Lasso) regardless. D is wrong — it may actually slow training (adds overhead). E is wrong — standardization doesn't guarantee normality.

### Q19: One-Standard-Error Rule
- **Question**: Using 1-SE rule on MSE vs complexity plot, which model to select?
- **Official Answer**: (c) Model with complexity = 4
- **Verification**: ✓ Correct (assuming the plot shows minimum at complexity=5)
- **Notes**: The 1-SE rule: find the model with minimum CV error (complexity=5), compute its standard error, then select the SIMPLEST model whose CV error is within 1 SE of the minimum. If the minimum is at complexity=5, the 1-SE rule selects the simplest model (complexity=4) that is still within one standard error of the minimum. This favors parsimony.

### Q20: Optimal Number of Clusters from Silhouette Plot
- **Question**: What is the optimal number of clusters from the silhouette plot?
- **Official Answer**: (b) 4
- **Verification**: ✓ Correct (assuming the silhouette plot peaks at k=4)
- **Notes**: The silhouette coefficient ranges from -1 to 1. The optimal k maximizes the average silhouette width. Without seeing the actual plot, the official answer of 4 is accepted. The silhouette method is a standard tool for determining k in clustering.

### Q21 (Open): LDA vs GMM
- **Question**: Compare LDA and GMM in terms of assumptions, fitting, goals, supervision, class overlap, and latent structure.
- **Key points for full marks**:
  - **Assumptions**: Both assume class-conditional Gaussian distributions. LDA additionally assumes equal covariance matrices across classes; GMM allows each component to have its own mean and covariance.
  - **Fitting**: LDA uses closed-form MLE (pooled within-class covariance, class means, priors). GMM uses EM algorithm (iterative, maximizes marginal likelihood).
  - **Goals & Supervision**: LDA is SUPERVISED (uses class labels) for classification. GMM is typically UNSUPERVISED (clusters without labels), but can be used in a supervised manner as a generative classifier.
  - **Class overlap**: Both can model overlapping classes probabilistically. GMM handles complex, non-spherical overlap better due to per-component covariances. LDA's linear boundary is less flexible.
  - **Latent structure**: GMM has explicit latent variables (cluster assignments) modeled via EM. LDA does not have latent variables — it directly models class-conditional distributions.

### Q22 (Open): CV Design for Wearable Data (Personalized vs Generalized)
- **Question**: Same wearable biosignal dataset (16×3×4=192 observations). Design datasets for a) personalized model and b) generalized model.
- **Key points for full marks**:
  - **a) Personalized model**: Use only data from the target individual (3 conditions × 4 seasons = 12 observations). Use leave-one-season-out CV (temporal structure) or leave-one-condition-out. Training on past seasons, testing on future seasons is most realistic.
  - **b) Generalized model**: Leave-one-individual-out CV — use 15 individuals for training, test on the held-out 16th. Repeat for all 16 individuals. This directly estimates generalization to a new, unseen individual.
  - **Trade-offs**: Personalized models have limited training data but may be highly accurate for that individual. Generalized models train on more data but must capture inter-individual variation. For clinical deployment (new patients), the generalized model is more appropriate as the "new patient" has never been seen by the model.

---

## Errors Found in Official Solutions

### 2022 Q9 (Intercept Penalization)
- **Issue**: The (x) notation around A, B, C suggests uncertainty. The official primary answer appears to be D, but A is the more precise and commonly taught rationale.
- **Correct explanation**: A ("Penalizing the intercept would introduce bias without any variance reduction") is the standard answer. D is a consequence, not an independent reason. B is incorrect — intercept penalization does not introduce variance.

### 2022 Q15 (SVD Variance Statement)
- **Issue**: The official answer B ("SVD with two components describes all the variance for all the datasets") is trivially true for 2D data — it's a tautology. The question appears to use this as a "trick" to catch students who look for a meaningful statement. While technically correct, it tests tautological reasoning rather than conceptual understanding.

### 2022 Q20 (Boosting Models)
- **Issue**: The answer sheet appears to mark A and E simultaneously for Q20, which is internally contradictory (A = "KNN with high K"; E = "None of the above"). This is almost certainly a grid transcription/rendering error.
- **Correct answer**: C (Any classification or regression tree). Boosting uses weak learners; shallow decision trees (stumps or small trees) are the canonical choice. KNN is not suitable because (a) it doesn't naturally fit into the gradient boosting framework, and (b) high-K KNN has high bias, making it a poor weak learner.

### 2024 Q11 (RF Proximity Plots)
- **Issue**: Answer C states "Proximity plots measure the closeness of variables." This is WRONG — proximity plots in Random Forest measure the closeness of OBSERVATIONS (data points), not variables. Two observations have high proximity if they frequently end up in the same terminal node across trees.
- **Correct statement**: Proximity plots measure similarity between observations. Variable importance measures assess variables. The official solution including C appears to contain an error; D and A should be the accepted answers.

### 2024 Q8 (Bonferroni — Number of Tests)
- **Issue**: Official answer A says "Bonferroni correction is useful when the number of tests is large but can be too conservative for small number of tests." The reality is the opposite: Bonferroni is most conservative (and appropriate) when the number of tests M is LARGE (it divides $\alpha$ by M). For SMALL M, it is barely conservative. For LARGE M, it becomes extremely conservative, losing power. Official answer A states it backwards.
- **Correct statement**: Bonferroni is useful when M is SMALL; it becomes overly conservative when M is LARGE. The official answer to Q8 should be A (BH controls FDR upper bound), which it is — so the overall answer A is correct, but the specific text of option C about Bonferroni and number of tests is misleading in the question. This is a question wording issue, not a solution error.

---

## High-Yield Topics (most frequently tested)

1. **Bias-Variance Tradeoff / EPE decomposition** — Week 1 — appears in 3/3 exams — Know $\text{EPE} = \text{Bias}^2 + \text{Variance} + \sigma^2$. Know which direction $\lambda$ pushes each.
2. **Lasso vs Ridge** — Week 2 — appears in 3/3 exams — Lasso: no closed form, L1, sparsity; Ridge: closed form $(X^TX + \lambda I)^{-1}X^TY$, L2, no sparsity. Too large $\lambda$ = high bias. Too small $\lambda$ = high variance.
3. **Cross-Validation Design** — Week 1–2 — appears in 3/3 exams — Never normalize before CV. Nested CV for unbiased post-tuning error. Dependent observations must stay in same fold. Personalized vs generalized model = within-subject vs leave-subject-out.
4. **Multiple Testing (Bonferroni / BH)** — Week 3 — appears in 3/3 exams — Bonferroni: FWER control ($\alpha/M$ threshold). BH: FDR control (more powerful, more discoveries, more false positives than Bonferroni at same $\alpha$).
5. **LDA** — Week 4 — appears in 3/3 exams — Linear because equal covariance assumption cancels quadratic terms. Probabilistic method (Gaussian class-conditionals + Bayes). More sensitive to outliers than logistic regression.
6. **GMM / Clustering** — Weeks 4, 9 — appears in 3/3 exams — Unsupervised. EM algorithm. Soft assignments. Allows elliptical clusters (vs spherical K-means).
7. **SVM / Kernel Trick** — Week 7 — appears in 3/3 exams — Linearity depends on kernel. Kernel trick = implicit high-dim computation. Dual formulation enables kernels. KernelPCA also uses kernel trick.
8. **Random Forest** — Weeks 5–6 — appears in 2/3 exams — Bias = individual tree bias → use deep trees. Variance reduced by averaging + random feature selection. Cannot parallelize? WRONG — trees are independent. OOB error is free validation.
9. **Archetypal Analysis / NMF** — Weeks 9–10 — appears in 3/3 exams — AA: archetypes = convex combinations of extreme data points; data = convex combinations of archetypes. NMF: $W, H \geq 0$; not unique; requires non-negative data.
10. **Neural Networks** — Weeks 10–11 — appears in 2/3 exams — Parameter count = (inputs×units + units) per layer. Autoencoders minimize reconstruction loss. Tend to overfit.

---

## Pattern Notes

- **Lasso lambda direction is tested every year.** Both 2022 and 2024 ask the SAME question (too small lambda in lasso). Know: too small $\lambda$ = low bias, high variance; too large $\lambda$ = high bias, low variance.

- **The cross-validation design question (Q22 in 2022 and 2024, Q22 in 2025) involves the same wearable biosignal dataset.** The 2024 and 2025 Q22 are nearly identical in setup. Master the personalized vs. generalized model distinction and the leave-one-individual-out vs within-individual CV design.

- **Confusion matrix arithmetic appears in 2022 Q13.** Practice applying sensitivity/specificity to population-level calculations (Bayes theorem style): TP = prevalence × sensitivity; FP = (1-prevalence) × FPR.

- **Neural network parameter counting appeared in both 2024 and 2025.** Formula: for each layer, parameters = (inputs_to_layer × nodes_in_layer) + nodes_in_layer (biases). Practice this arithmetic.

- **The "None of the above" option (E) is correct surprisingly often** — 2022 Q16 (CORCONDIA), 2024 Q1 (supervised methods), 2024 Q13 (LARS vs CD). Do not reflexively reject E.

- **Boosting uses SHALLOW trees (stumps), not deep trees.** Bagging/RF uses DEEP trees. This distinction is a common trap and appears in multiple questions.

- **BIC grows as $\log(N)$ → penalizes complexity more than AIC (constant $2p$) as N increases.** This is tested in 2025 Q8.

- **The 2022 exam is the only one with multiple answers per question AND a 2-point scoring system.** The 2024 exam uses +1/-1 per option. The 2025 exam uses single correct answers. Exam format changes have real strategy implications.

- **Question 17 (subspace methods / multiway models) appeared in the same slot in both 2022 and 2024.** The 2022 version tested PCA/PLS/CCA; the 2024 version tested PARAFAC/Tucker. Expect one of these two topic clusters at Q17.

- **Open question Q22 has used the exact same experimental dataset description (wearables, 16 subjects, 3 conditions, 4 seasons) in both 2024 and 2025.** This is extremely likely to appear again. Prepare a thorough answer on CV design for structured/hierarchical data.

- **Common trap in clustering questions**: AIC/BIC cannot be directly applied to K-means (no likelihood). Use elbow method, gap statistic, or silhouette score for K-means; use BIC for GMM.

- **Common trap in SVM questions**: The dual formulation alone does NOT make SVM nonlinear — the kernel choice does. A linear kernel in the dual still gives a linear boundary.
