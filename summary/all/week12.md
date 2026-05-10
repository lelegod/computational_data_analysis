# Week 12 — Multiway Models (Tucker3 and PARAFAC)

## Overview
Week 12 extends matrix factorization to three-way (and higher-order) tensors. The core tools are Tucker3 (a generalization of PCA to 3-way data with a core tensor and three loading matrices) and PARAFAC/CP (Parallel Factor Analysis, a special constrained Tucker where the core tensor is super-diagonal). Key topics include tensor notation, n-mode multiplication, the Tucker and PARAFAC objective functions and ALS optimization, comparison of the two models, and model selection methods (CORCONDIA and split-half analysis). Applications include image compression, spectroscopic reaction monitoring, and simultaneous enzyme quantification.

---

## Tensor Notation and Basics

### Notation
- Tensors denoted using **calligraphic letters**: X, G, etc.
- `X ∈ R^{I₁ × I₂ × ... × I_N}` — an N-way tensor
- `X^{I₁ × I₂ × ... × I_N}` — alternative dimension notation
- **N** = the **order** of the tensor (number of modes/dimensions)
- **X** is the measured response across the tensor
- **I_N** = dimensionality of the N-th mode = number of elements along that mode

### 3-Way Tensors
For a tensor X^{I × J × K}:
- **Slices** (fix one index): 2D sub-matrices
  - Horizontal: X(i,:,:) — fix first index
  - Lateral: X(:,j,:) — fix second index
  - Frontal: X(:,:,k) — fix third index
- **Fibers** (fix two indices): 1D vectors
  - Column fibers: X(:,j,k)
  - Row fibers: X(i,:,k)
  - Tube fibers: X(i,j,:)

### Frobenius Norm of a Tensor
For a 3-way tensor A:

`||A||_F = √(Σᵢ Σⱼ Σₖ a²_{ijk})`

Direct generalization of the matrix Frobenius norm.

---

## Matricization (Unfolding) and Folding

### Concept
Matricization (unfolding) = rearranging a tensor into a 2D matrix.

For tensor X^{I × J × K}:
- **Mode-1 unfolding**: X_{(1)} ∈ R^{I × JK} — rows = mode-1 fibers
- **Mode-2 unfolding**: X_{(2)} ∈ R^{J × IK} — rows = mode-2 fibers
- **Mode-3 unfolding**: X_{(3)} ∈ R^{K × IJ} — rows = mode-3 fibers

**Mode-1 example:**
- Tensor: X^{I × J × K}
- Matricize (unfold): X_{(1)} ∈ R^{I × J·K}
- Fold back: X^{I × J × K}

**Python equivalent:** `numpy.reshape` — unfolding is a reshape operation

---

## N-mode Multiplication

### Definition
The n-mode multiplication of an order-N tensor X^{I₁×I₂×...×I_N} with a matrix M^{J×I_n}:

**Step 1** — Unfold tensor along mode n:
`X_n ∈ R^{I_n × (I₁×...×I_{n−1}×I_{n+1}×...×I_N)}`

**Step 2** — Multiply along mode n:
`Z = M X_n ∈ R^{J × (I₁×...×I_{n−1}×I_{n+1}×...×I_N)}`

**Step 3** — Fold back to tensor:
`X ×_n M = Z^{I₁×...×I_{n−1}×J×I_{n+1}×...×I_N}`

**Compact notation:**
`[X ×_n M]_{(n)} = M X_{(n)}`

**Python functions:**
- `mod_dot(X, M, n)`: performs one n-mode multiplication (unfold, multiply, fold)
- `multi_mode_dot(X, [A, B, C])`: applies mod_dot in sequence for all modes

---

## Tucker 3 Model

### What is Tucker3?
- Proposed by Tucker in 1970
- Decomposes a 3rd-order tensor X^{I × J × K} as:
  `X ≈ G ×₁ A ×₂ B ×₃ C + E`
- **G**: the core tensor (shape P × Q × R) — defines "cross-talk" between components
- **A** ∈ R^{I × P}: loading matrix for mode 1 (P components)
- **B** ∈ R^{J × Q}: loading matrix for mode 2 (Q components)
- **C** ∈ R^{K × R}: loading matrix for mode 3 (R components)
- Ranks (P, Q, R) can be different for each mode

**Tucker(3,3,3)**: all three modes have the same rank 3
**Tucker(2,4,3)**: ranks 2, 4, 3 for modes 1, 2, 3

### Three Equivalent Representations

**1. Outer Product (Sum of Scaled Outer Products):**

`X ≈ Σ_{p=1}^P Σ_{q=1}^Q Σ_{r=1}^R g_{pqr} a_p ∘ b_q ∘ c_r = G ×₁ A ×₂ B ×₃ C`

- Each outer product a_p ∘ b_q ∘ c_r is a rank-1 tensor
- Scaled by core element g_{pqr}
- Sum over all P×Q×R combinations

**2. Scalar (Element-wise):**

`x_{ijk} ≈ Σ_{p=1}^P Σ_{q=1}^Q Σ_{r=1}^R g_{pqr} a_{ip} b_{jq} c_{kr}`

Clearest for seeing what happens to each element.

**3. Matrix (Unfolded) Representation:**

`X_{(1)} ≈ A G_{(1)} (C ⊗ B)^T`

`X_{(2)} ≈ B G_{(2)} (C ⊗ A)^T`

`X_{(3)} ≈ C G_{(3)} (B ⊗ A)^T`

where ⊗ is the **Kronecker product** (yields all possible outer vector products between columns of two matrices).

- This form is used for optimization (ALS)
- Define Z_A = G_{(1)}(C ⊗ B)^T, then estimate A by least squares: `A = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}`

### Tucker3 Optimization (ALS)
Loss function:

`min_X ||X − X̂||²_F` where `X̂ ≈ Σ_{pqr} g_{pqr} a_p ∘ b_q ∘ c_r`

**Alternating Least Squares updates:**

`A ← X_{(1)} (G_{(1)}(C ⊗ B)^T)^† = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}` ... (eq 10)

`B ← X_{(2)} (G_{(2)}(C ⊗ A)^T)^† = X_{(2)} Z_B^T (Z_B Z_B^T)^{-1}` ... (eq 11)

`C ← X_{(3)} (G_{(3)}(B ⊗ A)^T)^† = X_{(3)} Z_C^T (Z_C Z_C^T)^{-1}` ... (eq 12)

Upon convergence, update core tensor:
`G ← X ×₁ A^{-1} ×₂ B^{-1} ×₃ C^{-1}`

**Initialization**: random guesses or Higher-Order SVD (HOSVD)
**Extension**: impose non-negativity constraints on A, B, C

† denotes the Moore-Penrose pseudoinverse (used to handle singularities)

### The Core Tensor G
- Dimensionality of G defines the ranks (P, Q, R) in the respective modes
- G defines ALL possible cross-talk options between the A, B, and C loadings
- At Tucker(P,Q,R), G has P×Q×R elements — each says "how much does component p in mode 1 interact with component q in mode 2 and component r in mode 3?"
- When R=1 (channel rank=1): model can isolate individual color channels (e.g., red, green, blue in image compression)
- Tip: set the rank of the "low-information" mode (e.g., color channels in an image) to its natural maximum and tune other ranks

### Tucker3 Image Compression Example
Data: color image X^{200 × 200 × 3} (pixels × pixels × RGB channels)

Progressive decomposition results (P=Q increasing, R fixed):
- **R=1**: Tucker isolates individual colors — residuals show color artifacts
- **R=3**: all three color channels captured — residuals much smaller
- **P=Q=1, R=3**: very coarse reconstruction (only 1 spatial component per mode)
- **P=Q=5, R=3**: improved but still coarse
- **P=Q=20, R=3**: reasonable reconstruction
- **P=Q=100, R=3**: good reconstruction, small residuals
- **P=Q=200, R=3**: near-perfect (full rank in spatial modes)

Key insight: "What you see is the product between B_p C_q^T scaled with core tensor G" — the reconstruction is driven by outer products of loading vectors scaled by core elements.

---

## PARAFAC Model (Parallel Factor Analysis / CP Decomposition)

### What is PARAFAC?
- Also called CP (Canonical Decomposition) or Canonical Polyadic decomposition
- Independently proposed by **Harshman** and **Carroll & Chang** in 1970
- Decomposes tensor into a sum of **R rank-one tensors**:
  `X ≈ Σ_{r=1}^R a_r ∘ b_r ∘ c_r`
- R = "tensor rank" (number of components)
- **Unlike Tucker**: no cross-talk between components — each rank-one term is independent

### Three Equivalent Representations

**1. Outer Product:**

`X ≈ Σ_{r=1}^R a_r ∘ b_r ∘ c_r`

Each component r is a single rank-1 tensor (outer product of three vectors).

**2. Scalar (Element-wise):**

`x_{ijk} ≈ Σ_{r=1}^R a_{ir} b_{jr} c_{kr}`

**3. Matrix (Unfolded) Representation:**

`X_{(1)} ≈ A (C ⊙ B)^T`

where ⊙ is the **Khatri-Rao product** (column-wise Kronecker product).

Define Z_A = (C ⊙ B)^T, then:

`A = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}` ... (eq 14)

### PARAFAC vs Tucker Matrix Form
| Aspect | PARAFAC | Tucker |
|--------|---------|--------|
| Product used | Khatri-Rao (⊙) | Kronecker (⊗) |
| Core tensor | Super-diagonal (G is I tensor) | Full (P×Q×R) |
| Cross-talk | None | Full |

### PARAFAC Optimization (ALS)
Loss function:

`min_X ||X − X̂||²_F` where `X̂ = Σ_{r=1}^R a_r ∘ b_r ∘ c_r`

**Alternating Least Squares updates:**

`A ← X_{(1)} (C ⊙ B) (C^T C * B^T B)^{-1}` ... (eq 15)

`B ← X_{(2)} (C ⊙ A) (C^T C * A^T A)^{-1}` ... (eq 16)

`C ← X_{(3)} (B ⊙ A) (B^T B * A^T A)^{-1}` ... (eq 17)

Note: `(C^T C * A^T A) = (C ⊙ A)^T (C ⊙ A)` where * = Hadamard (element-wise) product

**Initialization**: Random guess or HOSVD (Higher-Order SVD)
**Extension**: impose non-negativity on loading matrices

### PARAFAC Application 1: Enzyme Activity (GOx)
- Data: 3-way tensor X^{I × J × K} = samples × time-points × wavenumbers
- Reaction: β-D-Glucose + O₂ → δ-D-Gluconolactone + H₂O₂ (catalyzed by Glucose Oxidase)
- MAJOR ASSUMPTION: substrate(s) and product(s) have **distinguishable spectral fingerprints**
- PARAFAC decomposes X into:
  - a_r scores: sample mode (linear with enzyme concentration)
  - b_r loadings: kinetic profile (how reaction progresses over time)
  - c_r loadings: spectral profile (the spectrum of each chemical species)
- With R=1 component: scores (a₁) correlate linearly with U (enzyme units) for each enzyme (Pectin Lyase, Celluclast, Glucose Oxidase)

### PARAFAC Application 2: Two Enzymes Simultaneously (PL + PME)
- Data: 56 samples × 40 time-points × 131 wavenumbers
- Reactions: Pectin Lyase (PL) and Pectin Methyl Esterase (PME) acting on different substrates
- PARAFAC with R=2: Component 1 captures PME; Component 2 captures PL
  - a₁ scores: linear with PME concentration; a₂ scores: linear with PL concentration
  - b₁, b₂: different kinetic profiles (reaction rates)
  - c₁, c₂: different spectral fingerprints
- TUCKER3 with P=2, Q=1, R=2 recovers the two enzyme patterns **better** than PARAFAC with R=2
  - Tucker adds cross-talk in sample mode (P=2) while sharing kinetic mode (Q=1)

---

## PARAFAC vs Tucker 3

### Key Differences

| Aspect | PARAFAC | Tucker3 |
|--------|---------|---------|
| Core tensor | Super-diagonal (I = identity-like R×R×R) | Full (P×Q×R) |
| Cross-talk | No (each component independent) | Yes (G defines cross-talk) |
| Ranks | One rank R for all modes | Separate P, Q, R per mode |
| Parameters | R(I+J+K) | PQR + IP + JQ + KR |
| Uniqueness | **Yes** (essentially unique) | **No** (rotation ambiguity) |
| Best for | Resolving additive physical profiles | Data compression |

### PARAFAC is a Special Case of Tucker3
PARAFAC = Tucker3 where the core tensor G is **super-diagonal**:
- Super-diagonal: G has ones along the main diagonal and zeros everywhere else (like the identity tensor I^{R×R×R})
- Formally: `X ≈ G ×₁ A ×₂ B ×₃ C` where G = I (super-diagonal)

### Uniqueness
**Tucker3 is NOT unique:**
`X ≈ (G ×₁ Q) ×₁ (AQ^{-1}) ×₂ B ×₃ C = G̃ ×₁ Ã ×₂ B ×₃ C`

You can rotate the core tensor by any rotation matrix Q and still get the same tensor representation.

**PARAFAC IS essentially unique:**
- If G is super-diagonal, applying rotation Q gives a new G̃ which is no longer super-diagonal
- Uniqueness holds under mild conditions (Kruskal's condition on ranks of loading matrices)
- The unique solution is a major advantage of PARAFAC for physically interpretable components

### Practical Guidance
- **Tucker**: good for data compression (flexible ranks, can compress each mode independently)
- **PARAFAC**: good when you want to deconvolute/resolve additive physical profiles (spectra, kinetics, etc.)
- For the image example: Tucker(200,200,3) ≈ full reconstruction; PARAFAC with large R also works
- For enzyme spectroscopy: PARAFAC naturally decomposes into interpretable spectral + kinetic + sample profiles

---

## Model Selection

### Problem
For PARAFAC: must choose R (number of components). Components are **not nested** (adding one more component doesn't preserve the previous components — unlike PCA/Tucker where you can do sequential extraction).

### Method 1: CORCONDIA (Core Consistency Diagnostic)
A heuristic specifically for PARAFAC model selection.

For a PARAFAC model with R components:

`CORCONDIA = 100 · (1 − ||I − G||²_F / ||I||²_F)`

where:
- **I** ∈ R^{R×R×R}: the perfect super-diagonal identity core tensor (what PARAFAC assumes)
- **G**: the actual core tensor obtained from the fitted model (computed as `G ← X ×₁ A^{-1} ×₂ B^{-1} ×₃ C^{-1}`)

**Interpretation:**
- **CORCONDIA ≈ 100**: G is nearly super-diagonal → the PARAFAC structure is appropriate for this R → use this R
- **CORCONDIA ≈ 0 or negative**: G deviates significantly from super-diagonal → the PARAFAC model is strained → R is too large
- Plot CORCONDIA vs R: choose the largest R before CORCONDIA drops sharply

### Method 2: Split-Half Analysis
Tests **stability** of the solution.

**Procedure:**
1. Randomly split data into two halves along one mode (usually the sample mode)
2. Fit PARAFAC with R components to each half independently
3. Compute the **Factor Match Score (FMS)** between the two halves:

`FMS = Σ_{r=1}^R (a_r^T â_r)/(||a_r|| ||â_r||) · (b_r^T b̂_r)/(||b_r|| ||b̂_r||) · (c_r^T ĉ_r)/(||c_r|| ||ĉ_r||)`

- Each term is the cosine similarity of loading vectors between the two half-models
- FMS = sum of products of cosine similarities across all R components
- **FMS close to R**: high stability (both halves find the same components)
- **FMS << R**: low stability → R is likely too large

**Best practice:** Compare CORCONDIA and split-half FMS and see if they agree on the optimal R.

### Method 3: Residual Analysis and Domain Knowledge
- Plot residuals as a function of R → look for elbow (diminishing returns)
- Use domain knowledge: if you know there are 2 chemical species, R=2 is a natural starting point
- Examine loading vectors for physical interpretability

### Key Point: PARAFAC Components are NOT Nested
Unlike PCA (where PC₁ at R=3 is the same as PC₁ at R=5), PARAFAC components change when R changes. You cannot extract R=2 as a sub-model of R=3. Each R must be fitted independently and evaluated.
