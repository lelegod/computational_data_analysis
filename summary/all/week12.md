# Week 12 — Multiway Models (Tucker3 and PARAFAC)

## Overview
Week 12 extends matrix factorization to three-way (and higher-order) tensors. The core tools are Tucker3 (a generalization of PCA to 3-way data with a core tensor and three loading matrices) and PARAFAC/CP (Parallel Factor Analysis, a special constrained Tucker where the core tensor is super-diagonal). Key topics include tensor notation, n-mode multiplication, the Tucker and PARAFAC objective functions and ALS optimization, comparison of the two models, and model selection methods (CORCONDIA and split-half analysis). Applications include image compression, spectroscopic reaction monitoring, and simultaneous enzyme quantification.

---

## Tensor Notation and Basics

### Notation
- Tensors denoted using **calligraphic letters**: $\mathcal{X}$, $\mathcal{G}$, etc.
- $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ — an $N$-way tensor
- $\mathcal{X}^{I_1 \times I_2 \times \cdots \times I_N}$ — alternative dimension notation
- **$N$** = the **order** of the tensor (number of modes/dimensions)
- **$\mathcal{X}$** is the measured response across the tensor
- **$I_N$** = dimensionality of the $N$-th mode = number of elements along that mode

### 3-Way Tensors
For a tensor $\mathcal{X}^{I \times J \times K}$:
- **Slices** (fix one index): 2D sub-matrices
  - Horizontal: $X(i,:,:)$ — fix first index
  - Lateral: $X(:,j,:)$ — fix second index
  - Frontal: $X(:,:,k)$ — fix third index
- **Fibers** (fix two indices): 1D vectors
  - Column fibers: $X(:,j,k)$
  - Row fibers: $X(i,:,k)$
  - Tube fibers: $X(i,j,:)$

### Frobenius Norm of a Tensor
For a 3-way tensor $\mathcal{A}$:

$$\|\mathcal{A}\|_F = \sqrt{\sum_i \sum_j \sum_k a_{ijk}^2}$$

Direct generalization of the matrix Frobenius norm.

---

## Matricization (Unfolding) and Folding

### Concept
Matricization (unfolding) = rearranging a tensor into a 2D matrix.

For tensor $\mathcal{X}^{I \times J \times K}$:
- **Mode-1 unfolding**: $X_{(1)} \in \mathbb{R}^{I \times JK}$ — rows = mode-1 fibers
- **Mode-2 unfolding**: $X_{(2)} \in \mathbb{R}^{J \times IK}$ — rows = mode-2 fibers
- **Mode-3 unfolding**: $X_{(3)} \in \mathbb{R}^{K \times IJ}$ — rows = mode-3 fibers

**Mode-1 example:**
- Tensor: $\mathcal{X}^{I \times J \times K}$
- Matricize (unfold): $X_{(1)} \in \mathbb{R}^{I \times J \cdot K}$
- Fold back: $\mathcal{X}^{I \times J \times K}$

**Python equivalent:** `numpy.reshape` — unfolding is a reshape operation

---

## N-mode Multiplication

### Definition
The $n$-mode multiplication of an order-$N$ tensor $\mathcal{X}^{I_1 \times I_2 \times \cdots \times I_N}$ with a matrix $M^{J \times I_n}$:

**Step 1** — Unfold tensor along mode $n$:

$$X_n \in \mathbb{R}^{I_n \times (I_1 \times \cdots \times I_{n-1} \times I_{n+1} \times \cdots \times I_N)}$$

**Step 2** — Multiply along mode $n$:

$$Z = M X_n \in \mathbb{R}^{J \times (I_1 \times \cdots \times I_{n-1} \times I_{n+1} \times \cdots \times I_N)}$$

**Step 3** — Fold back to tensor:

$$\mathcal{X} \times_n M = \mathcal{Z}^{I_1 \times \cdots \times I_{n-1} \times J \times I_{n+1} \times \cdots \times I_N}$$

**Compact notation:**

$$[\mathcal{X} \times_n M]_{(n)} = M X_{(n)}$$

**Python functions:**
- `mod_dot(X, M, n)`: performs one $n$-mode multiplication (unfold, multiply, fold)
- `multi_mode_dot(X, [A, B, C])`: applies mod_dot in sequence for all modes

---

## Tucker 3 Model

### What is Tucker3?
- Proposed by Tucker in 1970
- Decomposes a 3rd-order tensor $\mathcal{X}^{I \times J \times K}$ as:

$$\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C + \mathcal{E}$$

- $\mathcal{G}$: the core tensor (shape $P \times Q \times R$) — defines "cross-talk" between components
- $A \in \mathbb{R}^{I \times P}$: loading matrix for mode 1 ($P$ components)
- $B \in \mathbb{R}^{J \times Q}$: loading matrix for mode 2 ($Q$ components)
- $C \in \mathbb{R}^{K \times R}$: loading matrix for mode 3 ($R$ components)
- Ranks $(P, Q, R)$ can be different for each mode

**Tucker(3,3,3)**: all three modes have the same rank 3
**Tucker(2,4,3)**: ranks 2, 4, 3 for modes 1, 2, 3

### Three Equivalent Representations

**1. Outer Product (Sum of Scaled Outer Products):**

$$\mathcal{X} \approx \sum_{p=1}^P \sum_{q=1}^Q \sum_{r=1}^R g_{pqr}\, a_p \circ b_q \circ c_r = \mathcal{G} \times_1 A \times_2 B \times_3 C$$

- Each outer product $a_p \circ b_q \circ c_r$ is a rank-1 tensor
- Scaled by core element $g_{pqr}$
- Sum over all $P \times Q \times R$ combinations

**2. Scalar (Element-wise):**

$$x_{ijk} \approx \sum_{p=1}^P \sum_{q=1}^Q \sum_{r=1}^R g_{pqr}\, a_{ip}\, b_{jq}\, c_{kr}$$

Clearest for seeing what happens to each element.

**3. Matrix (Unfolded) Representation:**

$$X_{(1)} \approx A\, G_{(1)}\, (C \otimes B)^T$$

$$X_{(2)} \approx B\, G_{(2)}\, (C \otimes A)^T$$

$$X_{(3)} \approx C\, G_{(3)}\, (B \otimes A)^T$$

where $\otimes$ is the **Kronecker product** (yields all possible outer vector products between columns of two matrices).

- This form is used for optimization (ALS)
- Define $Z_A = G_{(1)}(C \otimes B)^T$, then estimate $A$ by least squares: $A = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}$

### Tucker3 Optimization (ALS)
Loss function:

$$\min_{\mathcal{X}} \|\mathcal{X} - \hat{\mathcal{X}}\|_F^2 \quad \text{where} \quad \hat{\mathcal{X}} \approx \sum_{pqr} g_{pqr}\, a_p \circ b_q \circ c_r$$

**Alternating Least Squares updates:**

$$A \leftarrow X_{(1)}\,(G_{(1)}(C \otimes B)^T)^\dagger = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1} \tag{eq 10}$$

$$B \leftarrow X_{(2)}\,(G_{(2)}(C \otimes A)^T)^\dagger = X_{(2)} Z_B^T (Z_B Z_B^T)^{-1} \tag{eq 11}$$

$$C \leftarrow X_{(3)}\,(G_{(3)}(B \otimes A)^T)^\dagger = X_{(3)} Z_C^T (Z_C Z_C^T)^{-1} \tag{eq 12}$$

Upon convergence, update core tensor:

$$\mathcal{G} \leftarrow \mathcal{X} \times_1 A^{-1} \times_2 B^{-1} \times_3 C^{-1}$$

**Initialization**: random guesses or Higher-Order SVD (HOSVD)
**Extension**: impose non-negativity constraints on $A$, $B$, $C$

$\dagger$ denotes the Moore-Penrose pseudoinverse (used to handle singularities)

### The Core Tensor $\mathcal{G}$
- Dimensionality of $\mathcal{G}$ defines the ranks $(P, Q, R)$ in the respective modes
- $\mathcal{G}$ defines ALL possible cross-talk options between the $A$, $B$, and $C$ loadings
- At Tucker$(P,Q,R)$, $\mathcal{G}$ has $P \times Q \times R$ elements — each says "how much does component $p$ in mode 1 interact with component $q$ in mode 2 and component $r$ in mode 3?"
- When $R=1$ (channel rank=1): model can isolate individual color channels (e.g., red, green, blue in image compression)
- Tip: set the rank of the "low-information" mode (e.g., color channels in an image) to its natural maximum and tune other ranks

### Tucker3 Image Compression Example
Data: color image $\mathcal{X}^{200 \times 200 \times 3}$ (pixels × pixels × RGB channels)

Progressive decomposition results ($P=Q$ increasing, $R$ fixed):
- **$R=1$**: Tucker isolates individual colors — residuals show color artifacts
- **$R=3$**: all three color channels captured — residuals much smaller
- **$P=Q=1$, $R=3$**: very coarse reconstruction (only 1 spatial component per mode)
- **$P=Q=5$, $R=3$**: improved but still coarse
- **$P=Q=20$, $R=3$**: reasonable reconstruction
- **$P=Q=100$, $R=3$**: good reconstruction, small residuals
- **$P=Q=200$, $R=3$**: near-perfect (full rank in spatial modes)

Key insight: "What you see is the product between $B_p C_q^T$ scaled with core tensor $\mathcal{G}$" — the reconstruction is driven by outer products of loading vectors scaled by core elements.

---

## PARAFAC Model (Parallel Factor Analysis / CP Decomposition)

### What is PARAFAC?
- Also called CP (Canonical Decomposition) or Canonical Polyadic decomposition
- Independently proposed by **Harshman** and **Carroll & Chang** in 1970
- Decomposes tensor into a sum of $R$ rank-one tensors:

$$\mathcal{X} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r$$

- $R$ = "tensor rank" (number of components)
- **Unlike Tucker**: no cross-talk between components — each rank-one term is independent

### Three Equivalent Representations

**1. Outer Product:**

$$\mathcal{X} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r$$

Each component $r$ is a single rank-1 tensor (outer product of three vectors).

**2. Scalar (Element-wise):**

$$x_{ijk} \approx \sum_{r=1}^R a_{ir}\, b_{jr}\, c_{kr}$$

**3. Matrix (Unfolded) Representation:**

$$X_{(1)} \approx A\,(C \odot B)^T$$

where $\odot$ is the **Khatri-Rao product** (column-wise Kronecker product).

Define $Z_A = (C \odot B)^T$, then:

$$A = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1} \tag{eq 14}$$

### PARAFAC vs Tucker Matrix Form
| Aspect | PARAFAC | Tucker |
|--------|---------|--------|
| Product used | Khatri-Rao ($\odot$) | Kronecker ($\otimes$) |
| Core tensor | Super-diagonal ($\mathcal{G}$ is $\mathcal{I}$ tensor) | Full ($P \times Q \times R$) |
| Cross-talk | None | Full |

### PARAFAC Optimization (ALS)
Loss function:

$$\min_{\mathcal{X}} \|\mathcal{X} - \hat{\mathcal{X}}\|_F^2 \quad \text{where} \quad \hat{\mathcal{X}} = \sum_{r=1}^R a_r \circ b_r \circ c_r$$

**Alternating Least Squares updates:**

$$A \leftarrow X_{(1)}\,(C \odot B)\,(C^T C * B^T B)^{-1} \tag{eq 15}$$

$$B \leftarrow X_{(2)}\,(C \odot A)\,(C^T C * A^T A)^{-1} \tag{eq 16}$$

$$C \leftarrow X_{(3)}\,(B \odot A)\,(B^T B * A^T A)^{-1} \tag{eq 17}$$

Note: $(C^T C * A^T A) = (C \odot A)^T (C \odot A)$ where $*$ = Hadamard (element-wise) product

**Initialization**: Random guess or HOSVD (Higher-Order SVD)
**Extension**: impose non-negativity on loading matrices

### PARAFAC Application 1: Enzyme Activity (GOx)
- Data: 3-way tensor $\mathcal{X}^{I \times J \times K}$ = samples × time-points × wavenumbers
- Reaction: $\beta$-D-Glucose $+ O_2 \to \delta$-D-Gluconolactone $+ H_2O_2$ (catalyzed by Glucose Oxidase)
- MAJOR ASSUMPTION: substrate(s) and product(s) have **distinguishable spectral fingerprints**
- PARAFAC decomposes $\mathcal{X}$ into:
  - $a_r$ scores: sample mode (linear with enzyme concentration)
  - $b_r$ loadings: kinetic profile (how reaction progresses over time)
  - $c_r$ loadings: spectral profile (the spectrum of each chemical species)
- With $R=1$ component: scores ($a_1$) correlate linearly with $U$ (enzyme units) for each enzyme (Pectin Lyase, Celluclast, Glucose Oxidase)

### PARAFAC Application 2: Two Enzymes Simultaneously (PL + PME)
- Data: 56 samples × 40 time-points × 131 wavenumbers
- Reactions: Pectin Lyase (PL) and Pectin Methyl Esterase (PME) acting on different substrates
- PARAFAC with $R=2$: Component 1 captures PME; Component 2 captures PL
  - $a_1$ scores: linear with PME concentration; $a_2$ scores: linear with PL concentration
  - $b_1$, $b_2$: different kinetic profiles (reaction rates)
  - $c_1$, $c_2$: different spectral fingerprints
- TUCKER3 with $P=2$, $Q=1$, $R=2$ recovers the two enzyme patterns **better** than PARAFAC with $R=2$
  - Tucker adds cross-talk in sample mode ($P=2$) while sharing kinetic mode ($Q=1$)

---

## PARAFAC vs Tucker 3

### Key Differences

| Aspect | PARAFAC | Tucker3 |
|--------|---------|---------|
| Core tensor | Super-diagonal ($\mathcal{I} =$ identity-like $R \times R \times R$) | Full ($P \times Q \times R$) |
| Cross-talk | No (each component independent) | Yes ($\mathcal{G}$ defines cross-talk) |
| Ranks | One rank $R$ for all modes | Separate $P, Q, R$ per mode |
| Parameters | $R(I+J+K)$ | $PQR + IP + JQ + KR$ |
| Uniqueness | **Yes** (essentially unique) | **No** (rotation ambiguity) |
| Best for | Resolving additive physical profiles | Data compression |

### PARAFAC is a Special Case of Tucker3
PARAFAC = Tucker3 where the core tensor $\mathcal{G}$ is **super-diagonal**:
- Super-diagonal: $\mathcal{G}$ has ones along the main diagonal and zeros everywhere else (like the identity tensor $\mathcal{I}^{R \times R \times R}$)
- Formally: $\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$ where $\mathcal{G} = \mathcal{I}$ (super-diagonal)

### Uniqueness
**Tucker3 is NOT unique:**

$$\mathcal{X} \approx (\mathcal{G} \times_1 Q) \times_1 (AQ^{-1}) \times_2 B \times_3 C = \tilde{\mathcal{G}} \times_1 \tilde{A} \times_2 B \times_3 C$$

You can rotate the core tensor by any rotation matrix $Q$ and still get the same tensor representation.

**PARAFAC IS essentially unique:**
- If $\mathcal{G}$ is super-diagonal, applying rotation $Q$ gives a new $\tilde{\mathcal{G}}$ which is no longer super-diagonal
- Uniqueness holds under mild conditions (Kruskal's condition on ranks of loading matrices)
- The unique solution is a major advantage of PARAFAC for physically interpretable components

### Practical Guidance
- **Tucker**: good for data compression (flexible ranks, can compress each mode independently)
- **PARAFAC**: good when you want to deconvolute/resolve additive physical profiles (spectra, kinetics, etc.)
- For the image example: Tucker$(200,200,3)$ ≈ full reconstruction; PARAFAC with large $R$ also works
- For enzyme spectroscopy: PARAFAC naturally decomposes into interpretable spectral + kinetic + sample profiles

---

## Model Selection

### Problem
For PARAFAC: must choose $R$ (number of components). Components are **not nested** (adding one more component doesn't preserve the previous components — unlike PCA/Tucker where you can do sequential extraction).

### Method 1: CORCONDIA (Core Consistency Diagnostic)
A heuristic specifically for PARAFAC model selection.

For a PARAFAC model with $R$ components:

$$\text{CORCONDIA} = 100 \cdot \left(1 - \frac{\|\mathcal{I} - \mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$

where:
- $\mathcal{I} \in \mathbb{R}^{R \times R \times R}$: the perfect super-diagonal identity core tensor (what PARAFAC assumes)
- $\mathcal{G}$: the actual core tensor obtained from the fitted model (computed as $\mathcal{G} \leftarrow \mathcal{X} \times_1 A^{-1} \times_2 B^{-1} \times_3 C^{-1}$)

**Interpretation:**
- **CORCONDIA $\approx 100$**: $\mathcal{G}$ is nearly super-diagonal → the PARAFAC structure is appropriate for this $R$ → use this $R$
- **CORCONDIA $\approx 0$ or negative**: $\mathcal{G}$ deviates significantly from super-diagonal → the PARAFAC model is strained → $R$ is too large
- Plot CORCONDIA vs $R$: choose the largest $R$ before CORCONDIA drops sharply

### Method 2: Split-Half Analysis
Tests **stability** of the solution.

**Procedure:**
1. Randomly split data into two halves along one mode (usually the sample mode)
2. Fit PARAFAC with $R$ components to each half independently
3. Compute the **Factor Match Score (FMS)** between the two halves:

$$\text{FMS} = \sum_{r=1}^R \frac{a_r^T \hat{a}_r}{\|a_r\|\|\hat{a}_r\|} \cdot \frac{b_r^T \hat{b}_r}{\|b_r\|\|\hat{b}_r\|} \cdot \frac{c_r^T \hat{c}_r}{\|c_r\|\|\hat{c}_r\|}$$

- Each term is the cosine similarity of loading vectors between the two half-models
- FMS = sum of products of cosine similarities across all $R$ components
- **FMS close to $R$**: high stability (both halves find the same components)
- **FMS $\ll R$**: low stability → $R$ is likely too large

**Best practice:** Compare CORCONDIA and split-half FMS and see if they agree on the optimal $R$.

### Method 3: Residual Analysis and Domain Knowledge
- Plot residuals as a function of $R$ → look for elbow (diminishing returns)
- Use domain knowledge: if you know there are 2 chemical species, $R=2$ is a natural starting point
- Examine loading vectors for physical interpretability

### Key Point: PARAFAC Components are NOT Nested
Unlike PCA (where $\text{PC}_1$ at $R=3$ is the same as $\text{PC}_1$ at $R=5$), PARAFAC components change when $R$ changes. You cannot extract $R=2$ as a sub-model of $R=3$. Each $R$ must be fitted independently and evaluated.
