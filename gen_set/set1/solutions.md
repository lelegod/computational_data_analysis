# Practice Set 1 — Solutions

## Multiple Choice Answers

| Q | Correct | Explanation |
|---|---------|-------------|
| 1 | A, C, D | **A** is correct: increasing complexity lowers bias and raises variance — this is the fundamental bias-variance tradeoff (Week 1 EPE decomposition). **B** is wrong: $\sigma^2$ is the irreducible noise from the data-generating process; no model can eliminate it. **C** is correct: variance is formally $\text{Var}(\hat{f}) = E[(\hat{f} - E[\hat{f}])^2]$, measuring fluctuation across training sets. **D** is correct: OLS is unbiased — $E[\hat{\beta}_\text{OLS}] = \beta$ — so $\text{Bias} = E[\hat{f}] - f = 0$ under correct model specification. **E** is wrong because A, C, D are all correct. |
| 2 | A, C | **A** is correct: the Ridge estimator $\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1}X^Ty$ is always invertible for $\lambda > 0$ because $X^TX + \lambda I$ is positive definite. **B** is wrong: Ridge shrinks coefficients toward zero but never exactly to zero — that is Lasso's property ($L_1$ geometry has corners; $L_2$ sphere does not). **C** is correct: as $\lambda \to 0$, $df \to p$ (OLS); as $\lambda \to \infty$, $df \to 0$; the trace decreases monotonically. **D** is wrong: Ridge introduces bias for any $\lambda > 0$ — $E[\hat{\beta}_\text{ridge}] \neq \beta$ in general. |
| 3 | A, B, C | **A** is correct: for Gaussian errors $\text{AIC}(\lambda) = \text{err}(\lambda) + 2(d(\lambda)/N)\hat{\sigma}^2_e$, which matches the Cp formula exactly. **B** is correct: BIC penalty $= \log(N) \cdot d/N$; AIC penalty $= 2d/N$ — the BIC coefficient $\log(N)$ grows with N while AIC's coefficient 2 is fixed. **C** is correct: Stone (1977) showed AIC is asymptotically equivalent to leave-one-out cross-validation. **D** is wrong: for large N, BIC penalizes MORE per parameter than AIC ($\log(N) > 2$ for $N \geq 8$), so BIC selects SIMPLER models than AIC, not more complex ones. |
| 4 | A, C, D | **A** is correct: this is the standard Lasso objective with the $L_1$ norm $\|\beta\|_1 = \sum|\beta_j|$. **B** is wrong: the $L_1$ norm is not differentiable at $\beta = 0$, so no closed-form solution exists — LARS or coordinate descent must be used. **C** is correct: the $L_1$ constraint region is a diamond in 2D with corners on the coordinate axes; the RSS ellipsoid typically first contacts the diamond at a corner where one coordinate is zero. **D** is correct: Lasso selects at most $\min(n, p)$ variables; when $p > n$, at most $n$ variables can be non-zero. |
| 5 | A, B, C | **A** is correct: Bonferroni threshold $= \alpha/M = 0.05/50 = 0.001$. **B** is correct: FWER without correction $= 1 - (1-0.05)^{50} = 1 - (0.95)^{50} \approx 0.923$ — a 92.3% chance of at least one false rejection. **C** is correct: Bonferroni controls FWER at level $\alpha$, ensuring $P(\text{any false rejection}) \leq 0.05$. **D** is wrong: Bonferroni has LOWER power than BH (Benjamini-Hochberg). BH allows a controlled proportion of false discoveries and therefore rejects more hypotheses (more power). |
| 6 | B, C | **A** is wrong: the description is reversed. In nested CV, the INNER loop handles model selection (hyperparameter tuning) and the OUTER loop handles model assessment (estimating generalization error). **B** is correct: nested CV audits the full pipeline — including the selection step — providing an unbiased estimate of how well the "select-then-train" procedure generalizes. **C** is correct: a large gap between inner-loop (optimistic, selected) error and outer-loop (honest) error signals that the model is overfitting to the hyperparameter selection noise. **D** is wrong: nested CV is especially important when hyperparameters are tuned; AIC/BIC are not a substitute because they also suffer from selection-induced bias if chosen from a set of models. |
| 7 | A, C, D | **A** is correct: this is the core statement of the curse of dimensionality — the volume of the space grows exponentially, so fixed N becomes sparse. **B** is wrong: in the elastic net, $\alpha = 0$ gives pure Ridge and $\alpha = 1$ gives pure Lasso (not the other way around). **C** is correct: the elastic net penalty $\lambda[(1/2)(1-\alpha)\|\beta\|_2^2 + \alpha\|\beta\|_1]$ combines $L_2$ (grouping correlated predictors) with $L_1$ (sparsity/variable selection). **D** is correct: Donoho (2000) listed the manifold hypothesis — that real data often lies on a low-dimensional manifold embedded in high-dimensional space — as one of the three "blessings" of dimensionality. |
| 8 | A, C, D | **A** is correct: Gini index $G = \sum_k p_{mk}(1-p_{mk}) = 0$ when all observations in the node belong to a single class ($p_{mk} = 1$ for one class). **B** is wrong: misclassification rate is NOT differentiable and is NOT the preferred criterion for growing trees — it is insensitive to probability shifts within the majority class. Gini index and cross-entropy are preferred for growing. **C** is correct: both Gini and cross-entropy respond to any shift in class probabilities, whereas misclassification rate does not change as long as the majority class is unchanged. **D** is correct: in regression trees, the prediction in region $R_j$ is $\hat{c}_j = \text{mean}(y_i : x_i \in R_j)$. |
| 9 | A, C, D | **A** is correct: $C_\alpha(T) = R(T) + \alpha|T|$ where $R(T)$ = total node impurity and $|T|$ = number of terminal nodes. **B** is wrong: when $\alpha = 0$, the full (unpruned) tree $T_0$ is selected because there is no penalty for complexity — not the root node. The root is selected when $\alpha$ is very large. **C** is correct: increasing $\alpha$ imposes a larger per-leaf penalty, causing the algorithm to prefer fewer leaves (smaller trees). **D** is correct: the standard CART procedure grows $T_0$, finds the sequence of subtrees via weakest-link pruning, then uses K-fold CV to select $\alpha^*$. |
| 10 | A, C, D | **A** is correct: the limiting variance as $B \to \infty$ is $\rho\sigma^2$, since the term $(1-\rho)\sigma^2/B \to 0$. This is the irreducible floor set by inter-tree correlation. **B** is wrong: bagging does NOT reduce bias. The bias of the bagged predictor equals the bias of any single tree, since $E[(1/B)\sum\hat{y}_b - y] = E[\hat{y}_b - y]$. **C** is correct: $P(\text{observation } i \text{ not in a bootstrap sample}) = (1 - 1/N)^N \to 1/e \approx 0.368$, so approximately 63.2% are included. **D** is correct: for each training observation, predictions are made only by trees for which that observation was OOB; these OOB predictions give a free CV-like error estimate. |
| 11 | A, B, C, D | **A** is correct: the default for classification is $m = \lfloor\sqrt{p}\rfloor$, as stated in the lecture. **B** is correct: when all $p$ features are considered at every split, no random subsampling occurs, which reduces to standard bagging. **C** is correct: by excluding strong predictors from some splits, random feature subsampling prevents trees from always splitting on the same variable, lowering pairwise correlation $\rho$ and thus reducing the variance floor $\rho\sigma^2$. **D** is correct: RF uses deep trees (low bias, high variance — bagging then reduces variance) while gradient boosting uses shallow trees/stumps (high bias, low variance — boosting then reduces bias sequentially). |
| 12 | A, B | **A** is correct: when $\text{err}_m = 0.5$, $\alpha_m = \log[(1-0.5)/0.5] = \log(1) = 0$, so the weak learner is ignored. **B** is correct: boosting reduces bias by sequentially targeting errors (hard cases); using high-bias weak learners (stumps) ensures each step corrects a specific weakness without already being complex. **C** is wrong: the exponential loss grows FASTER than binomial deviance for misclassified observations (negative margin), making AdaBoost MORE sensitive to label noise, not less robust. **D** is wrong: forward stagewise additive modelling fixes previously fitted trees — once added, their weights are never adjusted; only new $(\beta_m, b_m)$ pairs are added. |
| 13 | A, C, D | **A** is correct: margin $C = 1/\|\beta\|$, so minimizing $(1/2)\|\beta\|^2$ maximizes $C$ — this is exactly the canonical SVM formulation. **B** is wrong: the relationship is reversed. KKT complementary slackness states $\alpha_i[y_i(x_i^T\beta + \beta_0) - 1] = 0$. Points with a large margin (far from boundary) have bracket $> 0$, so $\alpha_i = 0$ (non-support vectors). Support vectors are ON the margin (bracket $= 0$) and have $\alpha_i > 0$. **C** is correct: the RBF kernel corresponds to the dot product in an infinite-dimensional RKHS — this is the fundamental mathematical fact underlying the kernel trick. **D** is correct: in the dual problem, $\max_\alpha \sum\alpha_i - (1/2)\sum\sum \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$, data appear only as inner products $\langle x_i, x_j \rangle$, enabling substitution with any kernel $K(x_i, x_j)$. |
| 14 | A, B, C | **A** is correct: eigenvalues of the covariance matrix are $\lambda_k = d_k^2/(n-1)$. So the squared singular values are $d_1^2=64, d_2^2=36, d_3^2=16, d_4^2=4, d_5^2=1$; total $= 121$; first two PCs explain $(64+36)/121 = 100/121 \approx 82.6\%$. **B** is correct: the right singular vectors $V$ from $X = UDV^T$ are identical to the eigenvectors of $X^TX/(n-1)$, which is the sample covariance matrix — this is the standard relationship between SVD and EVD of the covariance. **C** is correct: PCA on unscaled data is dominated by features with large variance (e.g., features in kg vs. mm); standardizing to unit variance ensures all features contribute equally. **D** is wrong: the description is reversed. PLS uses the response $y$ to guide dimension reduction (it maximizes covariance between $Xv$ and $y$), while PCA maximizes variance in $X$ alone (unsupervised). |
| 15 | A, C, D | **A** is correct: the K-means objective function is exactly this within-cluster sum of squares. **B** is wrong: K-means is NOT guaranteed to find the global optimum — it converges to a local minimum that depends on initialization. Multiple random restarts are recommended. **C** is correct: $s(i) \in [-1, 1]$ by definition (since $a(i)$ and $b(i)$ are non-negative distances); $s(i)$ near $+1$ means $a(i) \ll b(i)$, so the point is much closer to its own cluster than to the next nearest. **D** is correct: the gap statistic $G(K) = \log(U_k) - \log(W_k)$ compares actual within-cluster dispersion $W_k$ to $U_k$ from simulated uniform reference data; a large gap signals real cluster structure. |
| 16 | A, B, C | **A** is correct: this is the E-step formula — Bayes' rule applied to compute the posterior cluster membership probability, combining prior $\pi_j$ with Gaussian likelihood $\mathcal{N}(x_i; \mu_j, \Sigma_j)$. **B** is correct: the M-step mean update is a weighted average of data points, weighted by soft assignments $\gamma_{ij}$ — identical to standard MLE but with fractional counts. **C** is correct: K-means can be derived as a limiting case of GMM where all $\Sigma_j = \varepsilon^2 I \to 0$ (identical spherical covariances) and assignments are hard ($\gamma_{ij} \to 0$ or $1$). **D** is wrong: GMM likelihood is not concave — the EM algorithm finds a local maximum, not the guaranteed global maximum. Multiple restarts are needed. |
| 17 | A, C, D | **A** is correct: $(5\times3+3) + (3\times3+3) + (3\times2+2) = 18 + 12 + 8 = 38$ parameters total. **B** is correct (but check — this is binary CE, applicable for 2-class problems): binary cross-entropy $-\sum[y \log \hat{y} + (1-y)\log(1-\hat{y})]$ is derived from the Bernoulli negative log-likelihood. For the 2-output softmax here this corresponds to categorical CE, but the formula is derived the same way. **C** is correct: vectorized backpropagation computes $\delta^{(\ell)}$ by transposing the weight matrix and multiplying with the upstream error, then element-wise multiplying with the local activation derivative. **D** is correct: RNNs propagate error back through time, and gradients of distant time steps involve products of many Jacobians, causing vanishing (or exploding) gradients — LSTM and GRU use gating mechanisms to address this. |
| 18 | A, B, D | **A** is correct: NMF forces $W \geq 0$ and $H \geq 0$, so all components are additive — no negative entries means no cancellation, producing a parts-based representation (e.g., face parts, topic words). **B** is correct: ICA requires non-Gaussian sources because the Central Limit Theorem states that mixtures become more Gaussian; ICA reverses this by maximizing non-Gaussianity. For Gaussian sources, the mixing matrix $A$ is unidentifiable. **C** is wrong: NMF solutions are NOT unique. For any invertible $Q$ with $WQ^{-1} \geq 0$ and $QH \geq 0$, $(WQ^{-1})(QH)$ is an equally valid factorization. **D** is correct: whitening (sphering) transforms the data so its covariance is the identity matrix, reducing the ICA problem from finding arbitrary $W$ to finding an orthogonal matrix (rotation), which is far simpler. |
| 19 | A, B, C | **A** is correct: archetypes lie on or near the convex hull — they are extreme points, not averages. This is the defining feature that distinguishes AA from K-means (centroids = interior points) and PCA (directions of maximum variance, not extreme profiles). **B** is correct: sparse coding uses an overcomplete dictionary $W$ with $K > I$ atoms, and represents each data point as $Wh$ where $h$ is sparse (most entries zero). This is exactly the Lasso problem in the coding step. **C** is correct: the $S$ matrix has columns summing to 1 with $s_{ij} \geq 0$, ensuring each archetype $Z = XS$ is a convex combination (weighted average) of real data points, so archetypes cannot lie outside the data cloud. **D** is wrong: AA and K-means find fundamentally different solutions. K-means places centroids at interior cluster means (inside the data cloud), while AA places archetypes on the convex hull (extreme boundary). Their solutions generally differ significantly. |
| 20 | A, C, D | **A** is correct: PARAFAC = Tucker3 with $\mathcal{G} = \mathcal{I}^{R\times R\times R}$ (super-diagonal identity tensor), where $g_{rrr} = 1$ and all other entries are 0. This eliminates all cross-component interactions. **B** is wrong: the uniqueness relationship is the opposite. PARAFAC IS essentially unique (up to sign and permutation of components), under Kruskal's conditions — this is a major practical advantage. Tucker3 is NOT unique due to rotational freedom: $\mathcal{G} \times_1 A$ can be rewritten as $(\mathcal{G} \times_1 Q) \times_1 (AQ^{-1})$ for any invertible $Q$. **C** is correct: $\text{CORCONDIA} = 100 \cdot (1 - \|\mathcal{I} - \mathcal{G}\|_F^2 / \|\mathcal{I}\|_F^2)$. Close to 100 means the fitted core $\mathcal{G}$ is nearly super-diagonal, confirming the PARAFAC model structure is appropriate for the chosen $R$. **D** is correct: Tucker3 unfolded form $X_{(1)} \approx A G_{(1)} (C \otimes B)^T$ uses the Kronecker product $\otimes$ (all outer products between columns), while PARAFAC $X_{(1)} \approx A (C \odot B)^T$ uses the Khatri-Rao product $\odot$ (column-wise Kronecker — only matching columns). |

---

## Open Question Solutions

---

### Q21 Solution: Ridge Regression — Derivation, Geometry, and Model Selection

**Part (a) — Closed-form Ridge derivation [3 pts]**

Start from the penalized objective:

$$J(\beta) = (Y - X\beta)^T(Y - X\beta) + \lambda\beta^T\beta$$

Expand:

$$J(\beta) = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta + \lambda\beta^T\beta$$

Take the derivative with respect to $\beta$:

$$\frac{\partial J}{\partial \beta} = -2X^TY + 2X^TX\beta + 2\lambda I\beta = 0$$

Key derivative rules used:
- $\frac{\partial}{\partial \beta}(\beta^TA\beta) = (A + A^T)\beta = 2A\beta$ when $A$ is symmetric ($X^TX$ is symmetric)
- $\frac{\partial}{\partial \beta}(b^T\beta) = b$

Set derivative to zero and rearrange:

$$2X^TX\beta + 2\lambda I\beta = 2X^TY$$
$$(X^TX + \lambda I)\beta = X^TY$$

**Ridge estimator:**

$$\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1} X^TY$$

This always exists because $X^TX$ is positive semi-definite, and adding $\lambda I$ (with $\lambda > 0$) makes $X^TX + \lambda I$ strictly positive definite, hence invertible.

---

**Part (b) — Geometric explanation of sparsity [3 pts]**

The penalized form of Ridge can be rewritten as a constrained problem:

- **Ridge ($L_2$):** $\min_\beta \text{RSS}$ subject to $\|\beta\|_2^2 \leq s$
- **Lasso ($L_1$):** $\min_\beta \text{RSS}$ subject to $\|\beta\|_1 \leq s$

The RSS contours form ellipses centered at the OLS solution. The solution is found where the RSS ellipse first contacts the constraint region.

- **Lasso constraint region ($L_1$):** In 2D, this is a **diamond** with corners on the coordinate axes. The RSS ellipse will typically first contact the diamond at a **corner**, where one coordinate equals exactly zero → sparse solution.
- **Ridge constraint region ($L_2$):** In 2D, this is a **circle** (sphere in higher dimensions). The circle has no corners. The RSS ellipse contacts the sphere at a smooth point that is almost never exactly on an axis → coefficients are never exactly zero.

This geometric difference is why Lasso performs variable selection (sets coefficients to zero) while Ridge only shrinks them.

---

**Part (c) — Effective degrees of freedom [2 pts]**

$$df(\lambda) = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$$

**As $\lambda \to 0$:** The ridge smoother matrix $S_\lambda = X(X^TX + \lambda I)^{-1}X^T$ approaches the OLS hat matrix $S = X(X^TX)^{-1}X^T$. The trace of the OLS hat matrix equals $p$ (the number of predictors). Therefore $df(\lambda) \to p$.

**Interpretation:** With no regularization, the model uses all $p$ degrees of freedom — equivalent to fitting $p$ free parameters.

**As $\lambda \to \infty$:** The penalty dominates; all coefficients are shrunk toward zero. $S_\lambda \to 0$ (the zero matrix), so $\text{trace}(S_\lambda) \to 0$. Therefore $df(\lambda) \to 0$.

**Interpretation:** With extreme regularization, the model has effectively zero degrees of freedom — it makes the same constant prediction regardless of $x$.

The effective $df(\lambda)$ provides a continuous, interpretable measure of model complexity that interpolates between 0 and $p$ as $\lambda$ decreases from $\infty$ to 0.

---

**Part (d) — The 1-SE rule [2 pts]**

The analyst is applying the **one-standard-error rule** (1-SE rule), introduced by Breiman et al. (1984) in the CART monograph.

**Rule:** After cross-validation, do not choose the $\lambda$ that minimizes CV error. Instead, choose the **largest $\lambda$** (most regularized, simplest model) whose CV error is within one standard error of the minimum CV error.

**Why this is preferable:**
1. The minimum CV error estimate is itself subject to estimation noise — the true optimal $\lambda$ may be somewhat larger.
2. Models with CV errors within 1 SE of the minimum are statistically indistinguishable from the optimal model.
3. The 1-SE rule selects a simpler, more regularized model that is more stable across repeated analyses and generalizes more reliably.
4. In the example: $\lambda^* = 0.1$ minimizes CV error, but $\lambda = 0.5$ is within 1 SE and produces a simpler model. The analyst prefers parsimony without sacrificing predictive accuracy.

---

### Q22 Solution: PARAFAC for Fluorescence Spectroscopy and Comparison with K-means

**Part (a) — Why PARAFAC is natural here [3 pts]**

Fluorescence spectroscopic data is generated by a physically additive process: the measured fluorescence intensity at excitation wavelength $j$ and emission wavelength $k$ for sample $i$ is the sum of contributions from each fluorescent compound:

$$x_{ijk} = \sum_r a_{ir} \cdot b_{jr} \cdot c_{kr} + \text{noise}$$

This is exactly the PARAFAC model: $\mathcal{X} \approx \sum_r a_r \circ b_r \circ c_r$

**Physical interpretation with $R = 2$ components:**

- **$A \in \mathbb{R}^{80 \times 2}$ (sample mode):** Each row $i$ gives the scores $[a_{i1}, a_{i2}]$ for sample $i$. $a_{ir}$ is proportional to the concentration of compound $r$ in sample $i$. If the model works correctly, $a_{i1}$ correlates linearly with the known concentration of compound A, and $a_{i2}$ with compound B.

- **$B \in \mathbb{R}^{30 \times 2}$ (excitation mode):** Column $b_r$ (length 30) is the excitation spectrum (profile) of compound $r$ — how fluorescence intensity varies with the excitation wavelength.

- **$C \in \mathbb{R}^{50 \times 2}$ (emission mode):** Column $c_r$ (length 50) is the emission spectrum of compound $r$ — how fluorescence intensity varies with the emission wavelength.

The trilinear structure of PARAFAC directly matches the physics of fluorescence: each compound contributes independently with its own spectral fingerprint, and the total signal is their additive mixture.

---

**Part (b) — Selecting R [2 pts]**

**Method 1: CORCONDIA (Core Consistency Diagnostic)**

Fit PARAFAC for several values of $R$. For each $R$, compute the Tucker core tensor $\mathcal{G}$ from the PARAFAC loading matrices ($\mathcal{G} = \mathcal{X} \times_1 A^{-1} \times_2 B^{-1} \times_3 C^{-1}$) and calculate:

$$\text{CORCONDIA} = 100 \cdot \left(1 - \frac{\|\mathcal{I} - \mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$

- CORCONDIA $\approx 100$: $\mathcal{G}$ is nearly super-diagonal → PARAFAC structure is appropriate → this $R$ is suitable.
- CORCONDIA drops sharply below $\approx 50$: model is strained → $R$ is too large.
- Select the largest $R$ before CORCONDIA drops.

**What it assesses:** Whether the trilinear PARAFAC structure truly fits the data at the chosen $R$ — a structural diagnostic.

**Method 2: Split-Half Analysis (FMS)**

1. Randomly split the 80 samples into two halves of 40.
2. Fit PARAFAC with $R$ components to each half independently.
3. Compute the Factor Match Score (FMS) $= \sum_r (\text{cosine similarity of } a_r) \times (\text{cosine similarity of } b_r) \times (\text{cosine similarity of } c_r)$.
4. FMS close to $R$: both halves find the same components → stable, reliable solution.
5. FMS $\ll R$: the two halves disagree → $R$ too large, solutions are not reproducible.

**What it assesses:** Reproducibility and stability of the solution across data subsets.

Use both methods together: choose $R$ where CORCONDIA is high AND FMS is close to $R$.

---

**Part (c) — PARAFAC vs K-means [3 pts]**

**(i) Type of structure recovered:**

- **PARAFAC** decomposes the full 3-way tensor into $R$ additive rank-one components. It recovers the underlying spectral profiles and concentration profiles simultaneously. The result is a physically meaningful decomposition reflecting the two fluorescent compounds.
- **K-means** groups the 80 samples into $K$ clusters based on the Euclidean distance between vectorized spectra. It finds which samples have similar spectra but does not decompose or interpret the spectral variation — it only assigns group membership.

**(ii) Physical interpretability:**

- **PARAFAC:** The loading vectors $B$ and $C$ directly recover the excitation and emission spectra of each compound — these are physically interpretable as the pure-component spectra. The score vector $A$ gives concentration estimates. This is directly usable by a chemist.
- **K-means:** The cluster centroids are average spectra of samples in each cluster. They do not correspond to pure-component spectra and have no direct physical interpretation in terms of the underlying compounds.

**(iii) Trilinear constraint:**

PARAFAC imposes the trilinear structure $x_{ijk} = \sum_r a_{ir} b_{jr} c_{kr}$, which directly encodes the physics of how fluorescence signals combine. This constraint:
- Acts as strong regularization, preventing overfitting by restricting the model to physically plausible solutions.
- Ensures that PARAFAC is essentially unique (up to sign and permutation of components), meaning the recovered spectra are the true spectra, not an arbitrary rotation.
- K-means imposes no such structure — it vectorizes the spectra and treats each of the 1500 entries as an independent feature, destroying the 2D excitation-emission structure and the trilinear relationship.

---

**Part (d) — Interpreting CORCONDIA = 87 [2 pts]**

**CORCONDIA = 87** (close to 100, but not perfect):

This indicates that the fitted core tensor $\mathcal{G}$ is close to super-diagonal but not exactly so. The PARAFAC model with $R = 2$ is a good fit — the trilinear structure is approximately satisfied, meaning the two-component model captures the main variation in the data. The small deviation from 100 may reflect minor model misfit, measurement noise, or small violations of the pure trilinear assumption (e.g., slight instrumental noise or overlapping spectra). Overall, $R = 2$ is likely appropriate and the solution is physically interpretable.

**CORCONDIA $\approx 0$ (or negative):**

This would indicate that $\mathcal{G}$ deviates severely from the super-diagonal identity tensor — the fitted core is dense (many non-zero off-diagonal elements). This means the PARAFAC model is strained: the trilinear structure does not adequately describe the data at the chosen $R$. The components are interacting (cross-talk), which contradicts PARAFAC's independence assumption. This signals that $R$ is too large — the model is fitting noise and the extra components are not physically meaningful. In this case, reduce $R$ or switch to Tucker3 (which explicitly models cross-talk via a full core tensor $\mathcal{G}$).
