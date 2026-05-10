# Week 11 — Decomposition Methods for Unsupervised Learning (Exam Focus)

## Must-Know Facts

### General Matrix Decomposition
- All four methods approximate X ≈ WH but with different structural constraints
- NMF: W ≥ 0, H ≥ 0 (non-negative)
- ICA: rows of H are statistically independent and non-Gaussian
- AA: archetypes on convex hull, reconstructions are convex mixtures
- Sparse Coding: H is sparse (mostly zeros), W is overcomplete (more columns than rows)

### NMF
- NMF: X ≈ WH with W ≥ 0 AND H ≥ 0 — BOTH must be non-negative
- Non-negativity enforces parts-based additive representation (no cancellation between atoms)
- NMF is NOT jointly convex in (W,H) — only convex in W given H fixed (and vice versa)
- This justifies alternating minimization (ALS or multiplicative updates)
- Multiplicative update for H: H_{kj} ← H_{kj} · (W^T X)_{kj} / (W^T WH)_{kj}
- Multiplicative update for W: W_{ik} ← W_{ik} · (XH^T)_{ik} / (WHH^T)_{ik}
- Multiplicative updates = gradient descent with spatially-varying learning rate η_H = H/(W^T WH)
- Multiplicative updates preserve non-negativity IF initialized with positive values (no projection needed)
- NMF solutions are NOT unique: WH = (WQ⁻¹)(QH) for any invertible Q — valid as long as both sides non-negative
- Disambiguation: geometric constraints (minimize volume of cone) or sparsity (L₁ penalties)
- Fast ALS: solve unconstrained LS, project negatives to 0 — does not require strict NNLS solver

### ICA
- ICA assumes: (1) sources are statistically independent AND (2) sources are non-Gaussian
- Mixing model: x = As; goal is to find W ≈ A⁻¹ such that ŝ = Wx separates sources
- ICA CANNOT separate Gaussian-distributed sources — CLT argument breaks down for Gaussians
- Strategy: find W that maximizes non-Gaussianity of estimated signals
- Excess kurtosis = μ₄/σ⁴ − 3; Gaussian = 0; Laplace = 3; Uniform = −1.2
- Whitening (pre-processing): transform data so E[x̃x̃^T] = I — makes A orthogonal, reduces search to rotations only
- FastICA iteration: w_new ← E[x̃ g(w^T x̃)] − E[g'(w^T x̃)] w; then normalize w ← w/||w||
- Normalization step is critical: keeps search on the whitened sphere (rotations only)
- FastICA converges cubically/quadratically — much faster than ordinary gradient descent
- Deflationary approach: after finding w₁, project out: w₂ ← w₂ − (w₂^T w₁)w₁
- PCA finds uncorrelated components; ICA finds statistically INDEPENDENT components — these are different

### Archetypal Analysis (AA)
- AA finds prototypes at the EXTREMES (convex hull boundary), NOT at interior centroids
- Objective: min_{S,H} ||X − XSH||²_F
- S matrix: s_{ij} ≥ 0, Σ_i s_{ij} = 1 — archetypes ARE convex combinations of real data points
- H matrix: h_{ij} ≥ 0, Σ_i h_{ij} = 1 — data IS convex combination of archetypes
- Archetypes: Z = XS (must be built from real data, not arbitrary points in space)
- Full reconstruction: X̂ = ZH = XSH
- AA vs k-means: AA puts prototypes at extremes; k-means puts centroids in the interior
- AA vs NMF: NMF uses X ≈ WH (W arbitrary); AA uses X ≈ XSH (archetypes must be data)
- AA vs PCA: PCA finds average profile; AA finds extreme profiles

### Sparse Coding
- Sparse coding uses overcomplete dictionary: K > I (more basis vectors than dimensions)
- Model: x ≈ Wh, where h is sparse (most entries = 0)
- Objective: L(W,H) = (1/2)||X − WH||²_F + λ Σ_j ||h_j||₁
- L₁ penalty = convex proxy for L₀ (counting non-zeros)
- L₁ (not L₂) causes exact zeros via shrinkage; L₂ only shrinks toward zero
- Step 1 (fix W, update h): reduces to Lasso — solve with Coordinate Descent or LARS
- Step 2 (fix H, update W): standard LS subject to unit norm constraint ||w_k||₂ ≤ 1
- Unit norm constraint on W is REQUIRED: without it, scale W→∞, H→0 drives L₁ to zero trivially
- Speckled CV: randomly mask individual entries of X; train ignoring masked; evaluate on masked only
- Row-holdout CV FAILS for matrix methods: cannot learn H (mixture weights) for held-out sample

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| `min_{W,H≥0} (1/2)||X−WH||²_F` | NMF objective (Frobenius) | Any NMF question |
| `H_{kj} ← H_{kj}·(W^T X)_{kj}/(W^T WH)_{kj}` | NMF multiplicative update for H | Step-by-step NMF |
| `W_{ik} ← W_{ik}·(XH^T)_{ik}/(WHH^T)_{ik}` | NMF multiplicative update for W | Step-by-step NMF |
| `X ≈ WH = (WQ⁻¹)(QH)` | NMF non-uniqueness (Q-ambiguity) | Explaining ambiguity |
| Excess kurtosis = μ₄/σ⁴ − 3 | Non-Gaussianity measure | ICA / distributions |
| `E[x̃x̃^T] = I` | Whitening condition | ICA preprocessing |
| `w_new ← E[x̃g(w^T x̃)] − E[g'(w^T x̃)]w` | FastICA update | ICA algorithm |
| `w₂ ← w₂ − (w₂^T w₁)w₁` | Deflationary orthogonalization | Multiple ICA components |
| `min_{S,H} ||X − XSH||²_F` | AA objective | Any AA question |
| `s_{ij}≥0, Σ_i s_{ij}=1` | AA constraint on S | AA vs NMF comparison |
| `h_{ij}≥0, Σ_i h_{ij}=1` | AA constraint on H | AA vs NMF comparison |
| `Z = XS; X̂ = ZH = XSH` | AA two-stage archetype formula | AA derivation |
| `L(W,H) = (1/2)||X−WH||²_F + λΣ_j||h_j||₁` | Sparse coding objective | Sparse coding questions |
| `min_{h_j} (1/2)||x_j−Wh_j||²₂ + λ||h_j||₁` | Sparse coding Step 1 = Lasso | Sparse coding algorithm |

---

## Common Traps (Wrong Answers in Exams)

- NMF is convex → NOT jointly convex in (W,H); only convex in one given the other
- NMF produces a unique solution → NOT unique; Q-ambiguity exists for any invertible Q
- Multiplicative updates use standard gradient descent → they are GD with spatially-varying learning rate η = H/(W^T WH)
- Fast ALS uses a strict NNLS solver → Fast ALS solves unconstrained LS then projects negatives to 0 (or ε)
- ICA requires Gaussian sources → ICA REQUIRES NON-Gaussian sources; it completely fails for Gaussians
- PCA and ICA find the same components → PCA finds uncorrelated components; ICA finds statistically independent components (independence is stricter than uncorrelatedness)
- Whitening in ICA is optional → Whitening is a necessary pre-processing step that converts the problem to finding rotations only
- AA archetypes can be any point in the feature space → Archetypes MUST be convex combinations of real data points (Z = XS constraint)
- AA and k-means prototypes are similar → AA puts prototypes on the BOUNDARY (convex hull); k-means puts them in the INTERIOR
- In sparse coding, you can skip the unit norm constraint on W → Without it, model trivially drives L₁ to zero by scaling W up and H down
- Lasso is only for supervised learning → In sparse coding Step 1, fixing W and solving for h IS a Lasso problem, even in unsupervised context
- Row-holdout CV works for NMF/sparse coding → WRONG — must use Speckled CV (mask individual entries) because you cannot learn H for held-out samples
- L₂ regularization causes sparsity → L₁ causes EXACT zeros (sparsity); L₂ only shrinks toward zero but never reaches it

---

## Quick Decision Rules

- If asked which method produces parts-based additive representation → NMF (non-negativity forces additive)
- If asked which method finds boundary/extreme prototypes → AA (Archetypal Analysis)
- If asked which method separates statistically independent signals → ICA
- If the question says "cocktail party problem" → ICA
- If the question mentions "kurtosis = 0" → Gaussian distribution → ICA CANNOT separate these
- If asked about NMF uniqueness → mention Q-ambiguity and the two disambiguation strategies
- If asked which NMF update preserves non-negativity without projection → multiplicative updates
- To derive multiplicative update for H: gradient is W^T WH − W^T X; set η_H = H/(W^T WH)
- If asked why L₁ and not L₀ for sparsity → L₀ is non-convex and NP-hard to optimize; L₁ is the convex relaxation
- If asked about CV for matrix methods → Speckled CV (mask entries), NOT row holdout
- If AA question asks why X appears in objective as XSH not WH → archetypes must be compositions of actual data points (geometric grounding)
- If asked what constraint makes AA unique vs NMF → doubly convex constraint (S and H both sum to 1 with non-negativity), AND archetypes = XS (anchored to data)
- If sparse coding Step 1 asked → it IS a Lasso problem; solve with Coordinate Descent or LARS (not standard GD, L₁ is non-differentiable at 0)
