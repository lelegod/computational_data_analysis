# Week 12 — Multiway Models: Tucker3 and PARAFAC (Exam Focus)

## Must-Know Facts

### Tensor Basics
- Tensors use calligraphic notation: X ∈ R^{I₁×I₂×...×I_N}
- N = order of tensor (number of modes/dimensions)
- I_N = dimensionality of the N-th mode (number of elements in that mode)
- 3-way tensor slices: Horizontal X(i,:,:), Lateral X(:,j,:), Frontal X(:,:,k)
- 3-way tensor fibers: column X(:,j,k), row X(i,:,k), tube X(i,j,:)
- Frobenius norm: ||A||_F = √(Σᵢ Σⱼ Σₖ a²_{ijk})
- Matricization (unfolding) is equivalent to numpy.reshape
- Mode-1 unfolding of X^{I×J×K}: X_{(1)} ∈ R^{I × JK}

### N-mode Multiplication
- [X ×_n M]_{(n)} = M X_{(n)} — the compact definition
- Three steps: (1) unfold along mode n, (2) multiply matrix M, (3) fold back
- mod_dot() = one n-mode multiplication; multi_mode_dot() = mod_dot() in sequence for all modes

### Tucker3
- Decomposes X^{I×J×K} as: X ≈ G ×₁ A ×₂ B ×₃ C
- G = core tensor (P×Q×R) — defines cross-talk between components
- A ∈ R^{I×P}, B ∈ R^{J×Q}, C ∈ R^{K×R} — loading matrices for each mode
- P, Q, R are the ranks for modes 1, 2, 3 — can be DIFFERENT
- Scalar form: x_{ijk} ≈ Σ_p Σ_q Σ_r g_{pqr} a_{ip} b_{jq} c_{kr}
- Matrix form: X_{(1)} ≈ A G_{(1)} (C ⊗ B)^T (uses Kronecker product ⊗)
- Optimization: ALS — update A, B, C in alternating least squares steps
- ALS update for A: A ← X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}, where Z_A = G_{(1)}(C ⊗ B)^T
- † = Moore-Penrose pseudoinverse, used to handle singularities
- After convergence: G ← X ×₁ A^{-1} ×₂ B^{-1} ×₃ C^{-1}
- Initialize with random guesses or HOSVD (Higher-Order SVD)
- Core tensor G defines all cross-talk options between A, B, C loadings
- Tucker is NOT unique: G ×₁ Q and A ×₁ Q^{-1} give same reconstruction for any rotation Q
- Tucker is good for DATA COMPRESSION tasks

### PARAFAC
- Also called CP (Canonical Decomposition); proposed by Harshman and Carroll & Chang (1970)
- Decomposes into a sum of R rank-one tensors: X ≈ Σ_{r=1}^R a_r ∘ b_r ∘ c_r
- R = tensor rank (number of components)
- Scalar form: x_{ijk} ≈ Σ_{r=1}^R a_{ir} b_{jr} c_{kr}
- Matrix form: X_{(1)} ≈ A (C ⊙ B)^T (uses Khatri-Rao product ⊙, NOT Kronecker)
- ALS update for A: A ← X_{(1)} (C ⊙ B)(C^T C * B^T B)^{-1}
- Note: (C^T C * A^T A) = (C ⊙ A)^T(C ⊙ A) where * = Hadamard (element-wise) product
- PARAFAC IS essentially unique — a major advantage over Tucker
- PARAFAC is good for RESOLVING ADDITIVE PHYSICAL PROFILES (spectra, kinetics)
- PARAFAC components are NOT nested — changing R changes all components

### PARAFAC vs Tucker: The Core Relationship
- PARAFAC is a SPECIAL CASE of Tucker3 where the core tensor G is super-diagonal
- Super-diagonal G: ones on main diagonal, zeros everywhere else (like identity tensor I^{R×R×R})
- Tucker can do everything PARAFAC can, but with more flexibility (and less uniqueness)
- Tucker: full P×Q×R core → cross-talk allowed between all components
- PARAFAC: diagonal R×R×R core → NO cross-talk; each component is independent

### Uniqueness (Critical Distinction)
- Tucker3 is NOT unique: can rotate core by any Q and compensate in loading matrices
- PARAFAC IS essentially unique: rotating G away from super-diagonal → no longer valid PARAFAC
- Uniqueness of PARAFAC makes it superior for extracting physically interpretable components

### CORCONDIA
- Used to select R (number of components) for PARAFAC
- Formula: CORCONDIA = 100 · (1 − ||I − G||²_F / ||I||²_F)
- I = perfect super-diagonal tensor R×R×R (what PARAFAC assumes the core to be)
- G = actual core fitted from data
- CORCONDIA ≈ 100: model is appropriate (G is nearly super-diagonal)
- CORCONDIA ≈ 0 or negative: model is strained (G deviates from super-diagonal) → R too large
- Choose the LARGEST R before CORCONDIA drops sharply

### Split-Half Analysis (FMS)
- Randomly split data into two halves along the sample mode
- Fit PARAFAC to each half independently
- Compute Factor Match Score (FMS):
  `FMS = Σ_{r=1}^R cos(a_r, â_r) · cos(b_r, b̂_r) · cos(c_r, ĉ_r)`
- Each cosine: `(a_r^T â_r)/(||a_r|| ||â_r||)` — cosine similarity between two half-model loadings
- FMS close to R: stable solution → good R choice
- FMS << R: unstable → R too large
- Best practice: compare CORCONDIA and FMS — they should agree

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| `||A||_F = √(Σᵢ Σⱼ Σₖ a²_{ijk})` | Tensor Frobenius norm | Any norm question |
| `[X ×_n M]_{(n)} = M X_{(n)}` | N-mode multiplication definition | Mode product questions |
| `x_{ijk} ≈ Σ_p Σ_q Σ_r g_{pqr} a_{ip} b_{jq} c_{kr}` | Tucker3 scalar form | Tucker element computation |
| `X_{(1)} ≈ A G_{(1)} (C ⊗ B)^T` | Tucker3 matrix form (mode 1) | Tucker optimization/ALS |
| `A ← X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}` where `Z_A = G_{(1)}(C⊗B)^T` | Tucker ALS update for A | Tucker ALS derivation |
| `x_{ijk} ≈ Σ_{r=1}^R a_{ir} b_{jr} c_{kr}` | PARAFAC scalar form | PARAFAC element computation |
| `X_{(1)} ≈ A (C ⊙ B)^T` | PARAFAC matrix form (mode 1) | PARAFAC optimization/ALS |
| `A ← X_{(1)} (C⊙B)(C^T C * B^T B)^{-1}` | PARAFAC ALS update for A | PARAFAC ALS derivation |
| CORCONDIA = `100·(1 − ||I−G||²_F/||I||²_F)` | Core consistency diagnostic | Choosing R for PARAFAC |
| `FMS = Σ_r (a_r^T â_r·b_r^T b̂_r·c_r^T ĉ_r)/(||a_r||·||â_r||·||b_r||·||b̂_r||·||c_r||·||ĉ_r||)` | Factor Match Score | Split-half stability |

---

## Common Traps (Wrong Answers in Exams)

- Tucker3 and PARAFAC are completely different models → PARAFAC is a SPECIAL CASE of Tucker3 where G is super-diagonal
- Tucker3 is unique → Tucker3 is NOT unique; any rotation Q can be applied to G with compensating change in loading matrices
- PARAFAC is not unique → PARAFAC IS essentially unique — its super-diagonal core constraint prevents arbitrary rotations
- Tucker uses Khatri-Rao product → Tucker uses KRONECKER product (⊗); PARAFAC uses KHATRI-RAO product (⊙)
- PARAFAC components are nested (like PCA) → PARAFAC components are NOT nested; changing R changes ALL components
- CORCONDIA close to 0 means good fit → CORCONDIA close to 100 means good fit; close to 0 or negative means R is too large
- CORCONDIA can be used for Tucker3 model selection → CORCONDIA is specifically for PARAFAC (it measures deviation from super-diagonal)
- A high R always gives a better PARAFAC model → High R causes CORCONDIA to drop (G becomes non-diagonal) → model becomes invalid even if reconstruction improves
- The Frobenius norm of a tensor is defined differently than for matrices → Same concept: square root of sum of all squared elements
- Tucker ranks P, Q, R must be equal → They can be DIFFERENT for each mode — this is a key advantage of Tucker over PARAFAC
- Unfolding (matricization) changes the data → Unfolding is just a reshape; no information is lost; folding back gives the original tensor
- Tucker is better for resolving physical profiles → PARAFAC is better for physical profiles (unique, additive); Tucker is better for compression
- In split-half analysis, you fit one model and split it → You split the DATA first, then fit SEPARATE models to each half independently

---

## Quick Decision Rules

- If asked which model is a special case of which → PARAFAC is a special case of Tucker3 (with super-diagonal core G)
- If asked which has a unique solution → PARAFAC (unique); Tucker (NOT unique — rotation ambiguity)
- If the question asks which product: Tucker or PARAFAC → Tucker uses Kronecker ⊗; PARAFAC uses Khatri-Rao ⊙
- If asked what CORCONDIA = 100 means → The core G is perfectly super-diagonal → R is appropriate for PARAFAC
- If asked what CORCONDIA = 0 or negative means → G deviates strongly from super-diagonal → R is too large
- If the question is about image compression → Tucker3 (flexible ranks per mode, compress each independently)
- If the question is about separating spectral/kinetic profiles → PARAFAC (additive, interpretable, unique)
- If asked why PARAFAC is unique but Tucker is not → Tucker can rotate G by any Q; PARAFAC's super-diagonal G cannot be rotated and remain super-diagonal
- If asked how to compute G after fitting PARAFAC loading matrices → G ← X ×₁ A^{-1} ×₂ B^{-1} ×₃ C^{-1}
- If asked for the Tucker ALS update for A → Define Z_A = G_{(1)}(C⊗B)^T, then A = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}
- If asked for PARAFAC ALS update for A → A = X_{(1)}(C⊙B)(C^T C * B^T B)^{-1}
- If asked which method for choosing R in PARAFAC → CORCONDIA and/or Split-half FMS (both together for confirmation)
- If asked why PARAFAC components are not nested → Fitting R=3 gives three components that are JOINTLY optimal; none of them equals any component from R=2
- If mode-n multiplication question → unfold along mode n, multiply by M, fold back; compact form: [X ×_n M]_{(n)} = M X_{(n)}
