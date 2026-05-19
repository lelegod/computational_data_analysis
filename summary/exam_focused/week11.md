# Week 11 — Decomposition Methods for Unsupervised Learning (Exam Focus)

## Must-Know Facts

### General Matrix Decomposition
- All four methods approximate $X \approx WH$ but with different structural constraints
- NMF: $W \geq 0$, $H \geq 0$ (non-negative)
- ICA: rows of $H$ are statistically independent and non-Gaussian
- AA: archetypes on convex hull, reconstructions are convex mixtures
- Sparse Coding: $H$ is sparse (mostly zeros), $W$ is overcomplete (more columns than rows)

### NMF
- NMF: $X \approx WH$ with $W \geq 0$ AND $H \geq 0$ — BOTH must be non-negative
- Non-negativity enforces parts-based additive representation (no cancellation between atoms)
- NMF is NOT jointly convex in $(W,H)$ — only convex in $W$ given $H$ fixed (and vice versa)
- This justifies alternating minimization (ALS or multiplicative updates)
- Multiplicative update for $H$: $H_{kj} \leftarrow H_{kj} \cdot \frac{(W^T X)_{kj}}{(W^T WH)_{kj}}$
- Multiplicative update for $W$: $W_{ik} \leftarrow W_{ik} \cdot \frac{(XH^T)_{ik}}{(WHH^T)_{ik}}$
- Multiplicative updates = gradient descent with spatially-varying learning rate $\eta_H = H/(W^T WH)$
- Multiplicative updates preserve non-negativity IF initialized with positive values (no projection needed)
- NMF solutions are NOT unique: $WH = (WQ^{-1})(QH)$ for any invertible $Q$ — valid as long as both sides non-negative
- Disambiguation: geometric constraints (minimize volume of cone) or sparsity ($L_1$ penalties)
- Fast ALS: solve unconstrained LS, project negatives to 0 — does not require strict NNLS solver

### ICA
- ICA assumes: (1) sources are statistically independent AND (2) sources are non-Gaussian
- Mixing model: $x = As$; goal is to find $W \approx A^{-1}$ such that $\hat{s} = Wx$ separates sources
- ICA CANNOT separate Gaussian-distributed sources — CLT argument breaks down for Gaussians
- Strategy: find $W$ that maximizes non-Gaussianity of estimated signals
- **Why maximizing non-Gaussianity works (the "paradox" resolved):**
  - CLT and ICA run in **opposite directions**:
    - Mixing direction: non-Gaussian sources $\xrightarrow{x = As}$ more Gaussian mixed signals (CLT)
    - Unmixing direction: ICA finds $W$ that reverses this — recovering non-Gaussianity
  - A **wrong** $W$: output is still a blend of multiple sources → still partially Gaussian
  - The **correct** $W$: output = original sources → spiky/non-Gaussian (speech, images)
  - Non-Gaussianity is a **compass needle** — it points toward the unmixed sources; maximizing it finds $A^{-1}$
  - Gaussian sources are unidentifiable: all rotations of a multivariate Gaussian are equally Gaussian → no compass
- Excess kurtosis $= \mu_4/\sigma^4 - 3$; Gaussian $= 0$; Laplace $= 3$; Uniform $= -1.2$
- Whitening (pre-processing): transform data so $\mathbb{E}[\tilde{x}\tilde{x}^T] = I$ — makes $A$ orthogonal, reduces search to rotations only
- FastICA iteration: $w_\text{new} \leftarrow \mathbb{E}[\tilde{x}\, g(w^T \tilde{x})] - \mathbb{E}[g'(w^T \tilde{x})]\, w$; then normalize $w \leftarrow w/\|w\|$
- Normalization step is critical: keeps search on the whitened sphere (rotations only)
- FastICA converges cubically/quadratically — much faster than ordinary gradient descent
- Deflationary approach: after finding $w_1$, project out: $w_2 \leftarrow w_2 - (w_2^T w_1)w_1$
- PCA finds uncorrelated components; ICA finds statistically INDEPENDENT components — these are different

### Archetypal Analysis (AA)
- AA finds prototypes at the EXTREMES (convex hull boundary), NOT at interior centroids
- Objective: $\min_{S,H} \|X - XSH\|_F^2$
- $S$ matrix: $s_{ij} \geq 0$, $\sum_i s_{ij} = 1$ — archetypes ARE convex combinations of real data points
- $H$ matrix: $h_{ij} \geq 0$, $\sum_i h_{ij} = 1$ — data IS convex combination of archetypes
- Archetypes: $Z = XS$ (must be built from real data, not arbitrary points in space)
- Full reconstruction: $\hat{X} = ZH = XSH$
- AA vs k-means: AA puts prototypes at extremes; k-means puts centroids in the interior
- AA vs NMF: NMF uses $X \approx WH$ ($W$ arbitrary); AA uses $X \approx XSH$ (archetypes must be data)
- AA vs PCA: PCA finds average profile; AA finds extreme profiles

### Sparse Coding
- Sparse coding uses overcomplete dictionary: $K > I$ (more basis vectors than dimensions)
- Model: $x \approx Wh$, where $h$ is sparse (most entries $= 0$)
- Objective: $L(W,H) = \frac{1}{2}\|X - WH\|_F^2 + \lambda \sum_j \|h_j\|_1$
- $L_1$ penalty = convex proxy for $L_0$ (counting non-zeros)
- $L_1$ (not $L_2$) causes exact zeros via shrinkage; $L_2$ only shrinks toward zero
- Step 1 (fix $W$, update $h$): reduces to Lasso — solve with Coordinate Descent or LARS
- Step 2 (fix $H$, update $W$): standard LS subject to unit norm constraint $\|w_k\|_2 \leq 1$
- Unit norm constraint on $W$ is REQUIRED: without it, scale $W \to \infty$, $H \to 0$ drives $L_1$ to zero trivially
- Speckled CV: randomly mask individual entries of $X$; train ignoring masked; evaluate on masked only
- Row-holdout CV FAILS for matrix methods: cannot learn $H$ (mixture weights) for held-out sample

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $\min_{W,H \geq 0} \frac{1}{2}\|X-WH\|_F^2$ | NMF objective (Frobenius) | Any NMF question |
| $H_{kj} \leftarrow H_{kj} \cdot \frac{(W^T X)_{kj}}{(W^T WH)_{kj}}$ | NMF multiplicative update for $H$ | Step-by-step NMF |
| $W_{ik} \leftarrow W_{ik} \cdot \frac{(XH^T)_{ik}}{(WHH^T)_{ik}}$ | NMF multiplicative update for $W$ | Step-by-step NMF |
| $X \approx WH = (WQ^{-1})(QH)$ | NMF non-uniqueness ($Q$-ambiguity) | Explaining ambiguity |
| Excess kurtosis $= \mu_4/\sigma^4 - 3$ | Non-Gaussianity measure | ICA / distributions |
| $\mathbb{E}[\tilde{x}\tilde{x}^T] = I$ | Whitening condition | ICA preprocessing |
| $w_\text{new} \leftarrow \mathbb{E}[\tilde{x}\,g(w^T \tilde{x})] - \mathbb{E}[g'(w^T \tilde{x})]\,w$ | FastICA update | ICA algorithm |
| $w_2 \leftarrow w_2 - (w_2^T w_1)w_1$ | Deflationary orthogonalization | Multiple ICA components |
| $\min_{S,H} \|X - XSH\|_F^2$ | AA objective | Any AA question |
| $s_{ij} \geq 0,\ \sum_i s_{ij} = 1$ | AA constraint on $S$ | AA vs NMF comparison |
| $h_{ij} \geq 0,\ \sum_i h_{ij} = 1$ | AA constraint on $H$ | AA vs NMF comparison |
| $Z = XS;\ \hat{X} = ZH = XSH$ | AA two-stage archetype formula | AA derivation |
| $L(W,H) = \frac{1}{2}\|X-WH\|_F^2 + \lambda\sum_j\|h_j\|_1$ | Sparse coding objective | Sparse coding questions |
| $\min_{h_j} \frac{1}{2}\|x_j-Wh_j\|_2^2 + \lambda\|h_j\|_1$ | Sparse coding Step 1 = Lasso | Sparse coding algorithm |

---

## Common Traps (Wrong Answers in Exams)

- NMF is convex → NOT jointly convex in $(W,H)$; only convex in one given the other
- NMF produces a unique solution → NOT unique; $Q$-ambiguity exists for any invertible $Q$
- Multiplicative updates use standard gradient descent → they are GD with spatially-varying learning rate $\eta = H/(W^T WH)$
- Fast ALS uses a strict NNLS solver → Fast ALS solves unconstrained LS then projects negatives to 0 (or $\varepsilon$)
- ICA requires Gaussian sources → ICA REQUIRES NON-Gaussian sources; it completely fails for Gaussians
- PCA and ICA find the same components → PCA finds uncorrelated components; ICA finds statistically independent components (independence is stricter than uncorrelatedness)
- Whitening in ICA is optional → Whitening is a necessary pre-processing step that converts the problem to finding rotations only
- AA archetypes can be any point in the feature space → Archetypes MUST be convex combinations of real data points ($Z = XS$ constraint)
- AA and k-means prototypes are similar → AA puts prototypes on the BOUNDARY (convex hull); k-means puts them in the INTERIOR
- In sparse coding, you can skip the unit norm constraint on $W$ → Without it, model trivially drives $L_1$ to zero by scaling $W$ up and $H$ down
- Lasso is only for supervised learning → In sparse coding Step 1, fixing $W$ and solving for $h$ IS a Lasso problem, even in unsupervised context
- Row-holdout CV works for NMF/sparse coding → WRONG — must use Speckled CV (mask individual entries) because you cannot learn $H$ for held-out samples
- $L_2$ regularization causes sparsity → $L_1$ causes EXACT zeros (sparsity); $L_2$ only shrinks toward zero but never reaches it

---

## Quick Decision Rules

- If asked which method produces parts-based additive representation → NMF (non-negativity forces additive)
- If asked which method finds boundary/extreme prototypes → AA (Archetypal Analysis)
- If asked which method separates statistically independent signals → ICA
- If the question says "cocktail party problem" → ICA
- If the question mentions "kurtosis = 0" → Gaussian distribution → ICA CANNOT separate these
- If asked about NMF uniqueness → mention $Q$-ambiguity and the two disambiguation strategies
- If asked which NMF update preserves non-negativity without projection → multiplicative updates
- To derive multiplicative update for $H$: gradient is $W^T WH - W^T X$; set $\eta_H = H/(W^T WH)$
- If asked why $L_1$ and not $L_0$ for sparsity → $L_0$ is non-convex and NP-hard to optimize; $L_1$ is the convex relaxation
- If asked about CV for matrix methods → Speckled CV (mask entries), NOT row holdout
- If AA question asks why $X$ appears in objective as $XSH$ not $WH$ → archetypes must be compositions of actual data points (geometric grounding)
- If asked what constraint makes AA unique vs NMF → doubly convex constraint ($S$ and $H$ both sum to 1 with non-negativity), AND archetypes $= XS$ (anchored to data)
- If sparse coding Step 1 asked → it IS a Lasso problem; solve with Coordinate Descent or LARS (not standard GD, $L_1$ is non-differentiable at 0)
