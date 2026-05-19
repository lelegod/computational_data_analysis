# CDA 02582 — OPEN QUESTIONS MASTER GUIDE
> Q21 and Q22 are open-ended. These require structured written answers with justification.
> This file: model answers for every open question topic that has appeared + likely new ones.
> Format each answer: (1) Key claim, (2) Mechanism/why, (3) Comparison if asked.

---

## PATTERN FROM PAST EXAMS

| Year | Q21 | Q22 |
|------|-----|-----|
| 2022 | Random Forest algorithm | Clustering for face images |
| 2024 | ICA uniqueness and distributions | CV design for wearables |
| 2025 | LDA vs GMM comparison | CV design for wearables (same dataset!) |

**Q22 will almost certainly be the wearables CV design again.** Prepare it first, deepest.

**Q21 pattern**: Asks you to explain, compare, or derive a key algorithm or concept in depth.
Likely candidates for 2026: SVM, Boosting, PARAFAC/Tucker, NMF/ICA/AA comparison, PCA vs PLS, GMM, LDA

---

## Q22 TEMPLATE — CV DESIGN FOR WEARABLES
*(Used in 2024 and 2025 — prepare this cold)*

**Dataset**: 16 subjects × 3 conditions × 4 seasons = 192 observations total. Predict activity from biosignals. Each subject has 3×4=12 observations.

---

### Part a) Personalized Model — Predict for SAME individual seen during training

**Goal**: Estimate how well the model predicts for a specific person using only their own data.

**CV Design**:
- Use only that individual's 12 observations (3 conditions × 4 seasons)
- Apply **leave-one-season-out CV**: train on 3 seasons (9 obs), test on 4th season
- Repeat 4 times (one for each held-out season)
- Alternative: leave-one-condition-out CV

**Why this works**:
- Only that person's data is used → model is tailored to individual physiology
- Seasons are the natural temporal unit → mimics real prediction scenario (future season from past)
- No mixing across individuals → no data leakage between people

**Key considerations**:
- Small training set (9 observations per fold) → high variance estimates
- Model learns individual-specific patterns (e.g., that person's resting heart rate baseline)
- NOT appropriate for a new, unseen patient

---

### Part b) Generalized Model — Predict for a NEW, UNSEEN individual

**Goal**: Estimate how well the model generalizes to a person it has never seen.

**CV Design**:
- Apply **leave-one-individual-out CV** (LOIO-CV)
- Fold 1: train on subjects 2–16 (15×12=180 obs), test on subject 1 (12 obs)
- Fold 2: train on subjects 1, 3–16, test on subject 2
- Repeat for all 16 subjects → 16 folds
- Report average performance across the 16 folds

**Why this works**:
- The held-out individual's data is NEVER seen during training
- Directly simulates the deployment scenario: new patient arrives, model must predict
- Captures between-subject variability (inter-individual differences in signal)

**Why NOT standard random CV?**
- Random splitting would put some observations from person $i$ in training AND testing
- The model would "recognize" person $i$'s individual patterns → artificially inflated performance
- This is **data leakage** due to the repeated-measures structure (observations from same person are not independent)

---

### Part c) Trade-offs

| Property | Personalized | Generalized |
|----------|-------------|-------------|
| Training data | 9 obs (tiny) | 180 obs (large) |
| Accuracy | High for known individual | Lower but applicable to new people |
| Clinical use | Good for monitoring known patient | Essential for new patients |
| CV fold count | 4 folds | 16 folds |
| Captures | Intra-individual variation | Inter-individual variation |

**Which to use for clinical deployment?**
→ Generalized model. New patients are unknown individuals — you cannot use their personal training data.

**Can you combine both?**
→ Yes: train a generalized model first, then fine-tune with a few sessions from the new individual (transfer learning / personalization strategy). This is the clinical gold standard.

---

### Full written answer (exam-ready):

*"For a personalized model, we restrict training and evaluation to a single individual's data. Using leave-one-season-out cross-validation within that person's 12 observations, we train on 9 observations and test on the held-out season, repeating for all 4 seasons. This design respects the temporal structure and estimates how well the model predicts for a known individual in a future time period.*

*For a generalized model, we use leave-one-individual-out cross-validation across all 16 subjects. In each fold, one complete individual (all 12 observations) is held out as the test set while the model is trained on the remaining 15 individuals. This ensures the test individual is entirely unseen during training, directly simulating prediction for a new patient.*

*The key distinction is the source of variation: personalized CV measures intra-individual prediction performance; generalized CV measures inter-individual generalization. For clinical applications where the model will be deployed on new patients, the generalized CV estimate is the appropriate measure of real-world performance. Mixing individuals across folds would constitute data leakage, as repeated observations from the same person share physiological structure, leading to over-optimistic performance estimates."*

---

## Q21 CANDIDATE A — RANDOM FOREST ALGORITHM
*(Appeared 2022 Q21)*

### Full Model Answer

**Step 1: Bootstrap sampling (Bagging)**
For each tree $b = 1, \ldots, B$:
- Draw a bootstrap sample $Z^{*b}$ of size $N$ with replacement from the training data
- ~63.2% of unique observations appear; ~36.8% are left out (out-of-bag, OOB)
- **Contribution**: creates $B$ diverse training sets → reduces variance through averaging

**Step 2: Random feature subsampling (the "Random" in RF)**
At each node when growing tree $b$:
- Randomly select $m < p$ features from the $p$ total (default: $m=\lfloor\sqrt{p}\rfloor$ for classification)
- The split is chosen only from these $m$ features
- **Contribution**: decorrelates trees (lowers $\rho$ in variance formula), reducing variance beyond plain bagging

**Step 3: Grow unpruned trees**
- Grow each tree to minimum node size (no pruning)
- Deep trees have low bias, high variance — averaging removes the variance
- **Contribution**: keeps bias low (RF bias = individual tree bias)

**Step 4: Aggregate predictions**
- Regression: $\hat{f}_{RF}(x) = \frac{1}{B}\sum_{b=1}^B T_b(x)$ (average)
- Classification: majority vote across all trees
- **Contribution**: averaging $B$ trees reduces variance by factor up to $\rho + (1-\rho)/B$

**Step 5: OOB error (free validation)**
- For observation $i$, predict using only trees that did NOT include $i$ in their bootstrap sample
- OOB error ≈ LOO-CV error — unbiased estimate of generalization error, no extra runs needed

**Step 6: Variable importance**
- Gini importance: accumulate impurity reduction across all splits on feature $j$
- OOB permutation: permute feature $j$'s values in OOB data, measure accuracy drop → more unbiased

**Performance summary**:
- Lower variance than bagging (through decorrelation)
- Same bias as individual tree (use deep trees to keep bias low)
- Scales well to high-dimensional data ($p > n$ is fine)
- Parallelizable (trees are independent)

---

## Q21 CANDIDATE B — ICA: UNIQUENESS AND DISTRIBUTIONS
*(Appeared 2024 Q21)*

### Full Model Answer

**The ICA model**: $x = As$, where $x$ = observed mixtures, $A$ = unknown mixing matrix, $s$ = independent sources. Goal: find $W \approx A^{-1}$ so that $\hat{s} = Wx$ recovers $s$.

**Why non-Gaussianity is required**:
The Central Limit Theorem states that linear mixtures of independent variables become MORE Gaussian. Therefore, if we find a linear combination of $x$ that is LEAST Gaussian, we are moving toward the original sources. If sources were Gaussian, all rotations would be equally Gaussian — ICA would be completely unidentifiable.

**Favored distributions** (for ICA sources):
- Super-Gaussian (leptokurtic, heavy-tailed): excess kurtosis > 0. Example: Laplace (kurtosis=3), speech signals
- Sub-Gaussian (platykurtic, flat): excess kurtosis < 0. Example: Uniform (kurtosis=-1.2)
- Gaussian sources: kurtosis = 0 → CANNOT be separated by ICA

**Measuring non-Gaussianity**:
- Excess kurtosis: $\kappa_4 = \mu_4/\sigma^4 - 3$ (Gaussian=0; simple but sensitive to outliers)
- Negentropy: $J(y) = H(y_\text{Gauss}) - H(y)$ (always ≥ 0; 0 for Gaussian; more robust)
- FastICA uses negentropy or kurtosis as contrast function

**Uniqueness — what ICA CAN and CANNOT determine**:
ICA is unique UP TO:
1. **Permutation of components**: cannot order the recovered sources
2. **Sign and scale of components**: $s$ and $-s$ produce the same distribution; variance is absorbed into $A$

ICA is NOT unique with respect to these — they are fundamental indeterminacies.

Contrast with PCA: PCA finds uncorrelated components (zero second-order covariance). Uncorrelated ≠ independent for non-Gaussian distributions. ICA finds full statistical independence (all higher-order statistics), which is a strictly stronger condition.

**FastICA algorithm (sketch)**:
1. Whiten data: $E[\tilde{x}\tilde{x}^T]=I$ (reduces search to rotations only)
2. Fixed-point iteration: $w_\text{new} \leftarrow E[\tilde{x}g(w^T\tilde{x})] - E[g'(w^T\tilde{x})]w$
3. Normalize: $w \leftarrow w/\|w\|$
4. For multiple components: use deflationary orthogonalization $w_2 \leftarrow w_2 - (w_2^Tw_1)w_1$
- Converges cubically/quadratically (much faster than gradient descent)

---

## Q21 CANDIDATE C — LDA vs GMM COMPARISON
*(Appeared 2025 Q21)*

### Full Model Answer

**Shared foundation**: Both assume class-conditional Gaussian distributions. The key differences are in supervision, covariance assumptions, fitting procedure, and goals.

| Property | LDA | GMM |
|----------|-----|-----|
| Supervision | Supervised (uses class labels) | Typically unsupervised |
| Covariance | Equal across all classes ($\Sigma_k=\Sigma$) | Each component has own $\Sigma_k$ |
| Fitting | Closed-form MLE | EM algorithm (iterative) |
| Goal | Classification (find decision boundary) | Density estimation / clustering |
| Boundary | Linear (due to equal covariance) | Can be nonlinear/complex |
| Latent variables | None | $Z_i$ = unobserved cluster assignment |
| Scalable to $p>n$ | Problematic (singular $\Sigma$) | Also problematic (use regularization) |

**Why LDA boundary is linear**:
In Bayes' rule: $\log[P(C_k|x)/P(C_{k'}|x)] = \log[\pi_k/\pi_{k'}] + \log[P(x|C_k)/P(x|C_{k'})]$

With Gaussian class-conditionals and EQUAL covariance $\Sigma_k=\Sigma$:
$$\log[P(x|C_k)/P(x|C_{k'})] = (x^T\Sigma^{-1})(\mu_k-\mu_{k'}) - \frac{1}{2}(\mu_k^T\Sigma^{-1}\mu_k - \mu_{k'}^T\Sigma^{-1}\mu_{k'})$$

The quadratic terms $x^T\Sigma^{-1}x$ cancel (same $\Sigma$ for both classes) → linear function of $x$ → linear boundary.

With unequal covariances (QDA): quadratic terms don't cancel → quadratic boundary.

**How GMM is fitted (EM)**:
1. E-step: compute soft assignments $\gamma_{ij} = P(Z_i=j|x_i)$ using current parameters
2. M-step: update $\mu_j, \Sigma_j, \pi_j$ using $\gamma_{ij}$ as weights
3. Iterate to convergence (maximizes log-likelihood)

**Key distinctions**:
- LDA: **closed-form** (compute pooled within-class covariance, class means, priors directly)
- GMM: **iterative EM** (converges to local maximum of likelihood)
- LDA class labels are observed → no latent variables
- GMM cluster assignments are latent → EM needed to handle unobserved $Z_i$
- LDA: cannot model overlapping clusters with different shapes (equal covariance constraint)
- GMM: each component has its own $\Sigma_k$ → can model elliptical, differently-sized clusters

---

## Q21 CANDIDATE D — SVM: DERIVATION AND KERNEL TRICK
*(High probability given week 7 is always tested)*

### Full Model Answer

**Problem Setup**:
Binary classification, labels $y_i \in \{-1,+1\}$, data $x_i \in \mathbb{R}^p$.
Find hyperplane $\{x: x^T\beta+\beta_0=0\}$ maximizing margin to nearest points.

**Primal Problem**:
$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i^T\beta+\beta_0)\geq 1 \quad \forall i$$
- Margin = $2/\|\beta\|$ → maximizing margin = minimizing $\|\beta\|^2$
- Constraint ensures all points are correctly classified with at least 1 canonical unit of margin

**Lagrangian and Dual Derivation**:
$$L_P = \frac{1}{2}\|\beta\|^2 - \sum_i\alpha_i[y_i(x_i^T\beta+\beta_0)-1]$$

Setting derivatives to zero:
- $\partial L/\partial\beta=0$: $\beta = \sum_i\alpha_iy_ix_i$ (solution expressed as weighted sum of training points)
- $\partial L/\partial\beta_0=0$: $\sum_i\alpha_iy_i=0$

Substituting back → **Dual**:
$$\max_\alpha \sum_i\alpha_i - \frac{1}{2}\sum_{ij}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle \quad \text{s.t.} \quad \alpha_i\geq0, \sum_i\alpha_iy_i=0$$

**KKT conditions**: $\alpha_i[y_i(x_i^T\beta+\beta_0)-1]=0$
- Support vectors: on margin (bracket=0) → $\alpha_i>0$
- Safe points: beyond margin (bracket>0) → $\alpha_i=0$

**Kernel Trick**:
The dual ONLY involves dot products $\langle x_i,x_j\rangle$. Replace with $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$:
- This implicitly maps to high (possibly infinite) dimensional space WITHOUT computing $\phi$ explicitly
- RBF: $K(x,x')=\exp(-\gamma\|x-x'\|^2)$ → infinite-dimensional feature space → highly nonlinear boundaries
- Prediction: $\hat{y}=\text{sign}(\sum_i\alpha_iy_iK(x_i,x)+\beta_0)$

**Comparison**: SVM has NO probabilistic model (unlike LDA/logistic regression). It is purely geometric (maximize margin). It naturally handles $p>>n$ through the dual (number of parameters = $N$, not $p$).

---

## Q21 CANDIDATE E — BOOSTING ALGORITHM
*(Appeared 2022 Q20-21, 2024 Q10)*

### Full Model Answer

**Core idea**: Build an ensemble of weak learners SEQUENTIALLY, where each new learner focuses on the errors of the current ensemble.

**AdaBoost.M1 Algorithm** (binary classification, $y\in\{-1,+1\}$):

Initialize: $w_i = 1/N$ for all $i$ (equal weights)

For $m = 1, \ldots, M$:
1. Fit classifier $G_m(x)$ to training data WITH weights $w_i$
2. Compute weighted error: $\text{err}_m = \sum_i w_i\mathbf{I}(y_i\neq G_m(x_i))/\sum_i w_i$
3. Compute classifier weight: $\alpha_m = \log[(1-\text{err}_m)/\text{err}_m]$
   - Good classifier ($\text{err}_m\to0$): $\alpha_m\to\infty$
   - Random classifier ($\text{err}_m=0.5$): $\alpha_m=0$ (contributes nothing)
   - Worse than random ($\text{err}_m>0.5$): $\alpha_m<0$ (flip predictions)
4. Update weights: $w_i \leftarrow w_i\cdot\exp[\alpha_m\cdot\mathbf{I}(y_i\neq G_m(x_i))]$
   - Misclassified points get HIGHER weights → next classifier focuses on them
5. Normalize: $w_i \leftarrow w_i/\sum_j w_j$

Final: $G(x) = \text{sign}[\sum_m\alpha_mG_m(x)]$

**Theoretical connection**: AdaBoost = forward stagewise additive modelling minimizing EXPONENTIAL loss $L(y,F)=\exp(-yF(x))$. Each step adds the best weak learner+weight pair to the current model.

**Why stumps?**
- Stumps (depth-1 trees) are high-bias, low-variance weak learners
- Boosting reduces BIAS by sequentially fitting residuals → use weak learners with high bias
- Contrast with RF: uses deep trees (low bias) and reduces variance through averaging

**Gradient Boosting** (general extension):
- At step $m$: compute pseudo-residuals $r_{im} = -[\partial L(y_i,F(x_i))/\partial F(x_i)]_{F=F_{m-1}}$
- For squared error: $r_{im} = y_i - F_{m-1}(x_i)$ (ordinary residuals)
- Fit tree to $r_{im}$; add with shrinkage: $F_m = F_{m-1} + \nu\cdot h_m$
- Works for any differentiable loss (exponential, binomial deviance, etc.)

**Bagging vs Boosting**:
| | Bagging/RF | Boosting |
|--|-----------|----------|
| Tree type | Deep (low bias) | Shallow/stumps (high bias) |
| Sequential? | No (parallel) | Yes |
| What it reduces | Variance | Bias |
| Can overfit? | No | Yes (noisy data) |

---

## Q21 CANDIDATE F — PARAFAC vs TUCKER
*(Appeared 2022 Q16, 2024 Q17)*

### Full Model Answer

**Tucker3 Model**:
$\mathcal{X}\approx\mathcal{G}\times_1 A\times_2 B\times_3 C$ where $\mathcal{G}\in\mathbb{R}^{P\times Q\times R}$ is the core tensor.

The core tensor $\mathcal{G}$ encodes ALL possible cross-talk between the $P$ components in mode 1, $Q$ in mode 2, and $R$ in mode 3. Ranks $P,Q,R$ can be different.

Matrix form (mode 1): $X_{(1)}\approx A\,G_{(1)}(C\otimes B)^T$ (Kronecker product $\otimes$)

Tucker is NOT unique: for any invertible $Q$, $\mathcal{G}\times_1 Q$ and $A\times_1 Q^{-1}$ give the same reconstruction. This rotation ambiguity is a fundamental limitation.

Tucker is best for **data compression**: flexible ranks per mode compress each independently.

**PARAFAC (CP) Model**:
$\mathcal{X}\approx\sum_{r=1}^R a_r\circ b_r\circ c_r$ — sum of $R$ rank-1 tensors.

PARAFAC is a **special case of Tucker3** where the core tensor $\mathcal{G}$ is super-diagonal (identity-like). The super-diagonal constraint means each component in mode 1 interacts ONLY with the corresponding component in modes 2 and 3 (no cross-talk).

Matrix form (mode 1): $X_{(1)}\approx A(C\odot B)^T$ (Khatri-Rao product $\odot$, NOT Kronecker)

PARAFAC IS essentially unique: the super-diagonal constraint prevents arbitrary rotations. This is a major advantage — components are physically interpretable.

PARAFAC components are NOT nested (unlike PCA): changing $R$ changes all components.

PARAFAC is best for **resolving physical profiles**: spectroscopy (excitation × emission × samples), EEG (channels × frequencies × subjects).

**CORCONDIA** measures how well the fitted PARAFAC model's core matches the ideal super-diagonal core:
$$\text{CORCONDIA} = 100\left(1 - \frac{\|\mathcal{I}-\mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$
- ≈100: $R$ is appropriate (core is nearly super-diagonal)
- ≈0 or negative: $R$ too large (forcing data into too many components degrades the super-diagonal structure)

**Choosing $R$**: Use CORCONDIA + Split-half FMS. Both should agree on the appropriate $R$.

---

## Q21 CANDIDATE G — PCA vs PLS vs CCA
*(Appeared 2022 Q17)*

### Full Model Answer

All three find linear combinations of variables, but with different objectives and supervision levels.

**PCA** (unsupervised):
- Objective: $\max_v \text{Var}(Xv)$, $\|v\|=1$
- Finds directions of maximum variance in $X$, ignoring $y$ entirely
- Components are orthogonal; eigenvalues of covariance matrix
- Limitation: highest-variance directions may have zero correlation with $y$

**PLS** (supervised):
- Objective: $\max \text{Cov}(Xu, Yv) = \text{Var}(Xu)^{1/2}\cdot\text{Var}(Yv)^{1/2}\cdot\text{Corr}(Xu,Yv)$
- Maximizes COVARIANCE between projected $X$ and projected $y$ → balances variance AND correlation
- Uses $y$ to guide which X-directions matter → avoids PCR's limitation
- With $M=p$ components: reduces to OLS; with $M<p$: regularized regression
- Components are orthogonal (by deflation)

**CCA** (two-sided supervised):
- Objective: $\max_{u,v} \text{Corr}(Xu, Yv)$
- Maximizes CORRELATION ONLY — ignores internal variance of $X$ and $Y$
- Two separate data matrices $X$ and $Y$
- Requires inverting $\Sigma_{XX}$ and $\Sigma_{YY}$ → fails when $p>n$
- Solution: Regularized CCA (add $\lambda I$) or Sparse CCA (PMD with $L_1$)

| Method | Supervised? | Objective | High-dim? | Orthogonal? |
|--------|-------------|-----------|-----------|-------------|
| PCA | No | Max Var($Xv$) | Yes | Yes |
| PLS | Yes | Max Cov($Xu$,$Yv$) | Yes | Yes |
| CCA | Two-sided | Max Corr($Xu$,$Yv$) | No (needs invertible $\Sigma$) | In transformed space |

All three: linear combinations of inputs, can produce sparse versions (sparse PCA, sparse PLS, sparse CCA using Elastic Net / PMD).

---

## Q21 CANDIDATE H — NMF / ICA / AA — COMPARING DECOMPOSITION METHODS
*(Could be asked to compare these)*

### Model Answer

All approximate $X\approx WH$ but differ in constraints:

**NMF**: $W\geq0, H\geq0$
- Non-negativity forces ADDITIVE parts-based representation (no cancellation)
- Solution not unique ($Q$-ambiguity)
- Interpretable for non-negative data (images, text counts, spectra)
- Fitted by alternating multiplicative updates or ALS

**ICA**: rows of $H$ are statistically independent AND non-Gaussian
- Uniqueness (up to permutation and sign) because non-Gaussianity identifies directions
- Requires non-Gaussian sources — fails completely for Gaussians
- Pre-processing: whiten data first
- Best for: cocktail party problem, EEG source separation

**Archetypal Analysis**: archetypes = $XS$ (convex combinations of data points), data = $XSH$ (convex combinations of archetypes)
- Archetypes lie on the convex HULL (extremes), not interior
- Doubly constrained: both $S$ and $H$ are convex weight matrices
- Good for: finding extreme phenotypes, patient profiles

**PCA**: orthogonal components, maximize variance
- Unique, nested components (first $k$ components are optimal for $k$-dimensional reduction)
- Does not enforce non-negativity, sparsity, or independence

**Summary table**:
| Method | Constraint | Unique? | Prototypes at |
|--------|-----------|---------|---------------|
| PCA | Orthogonal | Yes | Interior (mean) |
| NMF | Non-negative | No | Interior (parts) |
| ICA | Independent, non-Gaussian | Yes (±perm/sign) | Directions |
| AA | Convex hull + convex mix | Partially | Boundary (extremes) |
| K-means | Hard assignments | No (local opt) | Interior (centroids) |

---

## GENERAL OPEN QUESTION WRITING STRATEGY

### Structure every answer as:
1. **State the model/algorithm** (one sentence, formula if relevant)
2. **Explain the mechanism** (why does it work? what does each step do?)
3. **State key properties** (bias, variance, uniqueness, complexity)
4. **Compare to alternatives** (what would you use instead and when?)
5. **Limitations** (when does it fail?)

### Common marks are given for:
- Correctly identifying the objective function
- Explaining the role of each component (not just listing them)
- Comparing two methods and articulating the key distinction
- Giving a concrete example or calculation to support a claim
- Mentioning what happens at edge cases ($\lambda\to\infty$, $R$ too large, etc.)

### Common mistakes to avoid:
- Writing "it works because it minimizes the error" — be specific about WHICH error and HOW
- Listing steps without explaining what each achieves
- Saying "more complex = better" — always frame with bias-variance tradeoff
- Forgetting to state assumptions (e.g., "assuming equal covariance matrices, LDA...")
- Writing about variance reduction when asked about bias, or vice versa
