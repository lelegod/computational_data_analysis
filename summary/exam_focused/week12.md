# Week 12 — Multiway Models: Tucker3 and PARAFAC (Exam Focus)

## Must-Know Facts

### Tensor Basics
- Tensors use calligraphic notation: $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$
- $N$ = order of tensor (number of modes/dimensions)
- $I_N$ = dimensionality of the $N$-th mode (number of elements in that mode)
- 3-way tensor slices: Horizontal $X(i,:,:)$, Lateral $X(:,j,:)$, Frontal $X(:,:,k)$
- 3-way tensor fibers: column $X(:,j,k)$, row $X(i,:,k)$, tube $X(i,j,:)$
- Frobenius norm: $\|\mathcal{A}\|_F = \sqrt{\sum_i \sum_j \sum_k a_{ijk}^2}$
- Matricization (unfolding) is equivalent to numpy.reshape
- Mode-1 unfolding of $\mathcal{X}^{I \times J \times K}$: $X_{(1)} \in \mathbb{R}^{I \times JK}$

### N-mode Multiplication
- $[\mathcal{X} \times_n M]_{(n)} = M X_{(n)}$ — the compact definition
- Three steps: (1) unfold along mode $n$, (2) multiply matrix $M$, (3) fold back
- `mod_dot()` = one $n$-mode multiplication; `multi_mode_dot()` = `mod_dot()` in sequence for all modes

### Tucker3
- Decomposes $\mathcal{X}^{I \times J \times K}$ as: $\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$
- $\mathcal{G}$ = core tensor ($P \times Q \times R$) — defines cross-talk between components
- $A \in \mathbb{R}^{I \times P}$, $B \in \mathbb{R}^{J \times Q}$, $C \in \mathbb{R}^{K \times R}$ — loading matrices for each mode
- $P$, $Q$, $R$ are the ranks for modes 1, 2, 3 — can be DIFFERENT
- Scalar form: $x_{ijk} \approx \sum_p \sum_q \sum_r g_{pqr}\, a_{ip}\, b_{jq}\, c_{kr}$
- Matrix form: $X_{(1)} \approx A\, G_{(1)}\, (C \otimes B)^T$ (uses Kronecker product $\otimes$)
- Optimization: ALS — update $A$, $B$, $C$ in alternating least squares steps
- ALS update for $A$: $A \leftarrow X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}$, where $Z_A = G_{(1)}(C \otimes B)^T$
- $\dagger$ = Moore-Penrose pseudoinverse, used to handle singularities
- After convergence: $\mathcal{G} \leftarrow \mathcal{X} \times_1 A^{-1} \times_2 B^{-1} \times_3 C^{-1}$
- Initialize with random guesses or HOSVD (Higher-Order SVD)
- Core tensor $\mathcal{G}$ defines all cross-talk options between $A$, $B$, $C$ loadings
- Tucker is NOT unique: $\mathcal{G} \times_1 Q$ and $A \times_1 Q^{-1}$ give same reconstruction for any rotation $Q$
- Tucker is good for DATA COMPRESSION tasks

### PARAFAC
- Also called CP (Canonical Decomposition); proposed by Harshman and Carroll & Chang (1970)
- Decomposes into a sum of $R$ rank-one tensors: $\mathcal{X} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r$
- $R$ = tensor rank (number of components)
- Scalar form: $x_{ijk} \approx \sum_{r=1}^R a_{ir}\, b_{jr}\, c_{kr}$
- Matrix form: $X_{(1)} \approx A\,(C \odot B)^T$ (uses Khatri-Rao product $\odot$, NOT Kronecker)
- ALS update for $A$: $A \leftarrow X_{(1)}\,(C \odot B)\,(C^T C * B^T B)^{-1}$
- General identity: $(P^T P * Q^T Q) = (P \odot Q)^T(P \odot Q)$ where $*$ = Hadamard product — applies to B-update denominator $(C^TC * A^TA)$ and C-update denominator $(B^TB * A^TA)$, not the A-update denominator $(C^TC * B^TB)$
- PARAFAC IS essentially unique — a major advantage over Tucker
- PARAFAC is good for RESOLVING ADDITIVE PHYSICAL PROFILES (spectra, kinetics)
- PARAFAC components are NOT nested — changing $R$ changes all components

### PARAFAC vs Tucker: The Core Relationship
- PARAFAC is a SPECIAL CASE of Tucker3 where the core tensor $\mathcal{G}$ is super-diagonal
- Super-diagonal $\mathcal{G}$: ones on main diagonal, zeros everywhere else (like identity tensor $\mathcal{I}^{R \times R \times R}$)
- Tucker can do everything PARAFAC can, but with more flexibility (and less uniqueness)
- Tucker: full $P \times Q \times R$ core → cross-talk allowed between all components
- PARAFAC: diagonal $R \times R \times R$ core → NO cross-talk; each component is independent

### Uniqueness (Critical Distinction)
- Tucker3 is NOT unique: can rotate core by any $Q$ and compensate in loading matrices
- PARAFAC IS essentially unique: rotating $\mathcal{G}$ away from super-diagonal → no longer valid PARAFAC
- Uniqueness of PARAFAC makes it superior for extracting physically interpretable components

### CORCONDIA
- Used to select $R$ (number of components) for PARAFAC
- Formula: $\text{CORCONDIA} = 100 \cdot \left(1 - \dfrac{\|\mathcal{I} - \mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$
- $\mathcal{I}$ = perfect super-diagonal tensor $R \times R \times R$ (what PARAFAC assumes the core to be)
- $\mathcal{G}$ = actual core fitted from data
- CORCONDIA $\approx 100$: model is appropriate ($\mathcal{G}$ is nearly super-diagonal)
- CORCONDIA $\approx 0$ or negative: model is strained ($\mathcal{G}$ deviates from super-diagonal) → $R$ too large
- Choose the LARGEST $R$ before CORCONDIA drops sharply

### Split-Half Analysis (FMS)
- Randomly split data into two halves along the sample mode
- Fit PARAFAC to each half independently
- Compute Factor Match Score (FMS):

$$\text{FMS} = \sum_{r=1}^R \cos(a_r, \hat{a}_r) \cdot \cos(b_r, \hat{b}_r) \cdot \cos(c_r, \hat{c}_r)$$

- Each cosine: $\dfrac{a_r^T \hat{a}_r}{\|a_r\|\|\hat{a}_r\|}$ — cosine similarity between two half-model loadings
- FMS close to $R$: stable solution → good $R$ choice
- FMS $\ll R$: unstable → $R$ too large
- Best practice: compare CORCONDIA and FMS — they should agree

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $\|\mathcal{A}\|_F = \sqrt{\sum_i \sum_j \sum_k a_{ijk}^2}$ | Tensor Frobenius norm | Any norm question |
| $[\mathcal{X} \times_n M]_{(n)} = M X_{(n)}$ | $N$-mode multiplication definition | Mode product questions |
| $x_{ijk} \approx \sum_p \sum_q \sum_r g_{pqr}\, a_{ip}\, b_{jq}\, c_{kr}$ | Tucker3 scalar form | Tucker element computation |
| $X_{(1)} \approx A\, G_{(1)}\, (C \otimes B)^T$ | Tucker3 matrix form (mode 1) | Tucker optimization/ALS |
| $A \leftarrow X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}$ where $Z_A = G_{(1)}(C \otimes B)^T$ | Tucker ALS update for $A$ | Tucker ALS derivation |
| $x_{ijk} \approx \sum_{r=1}^R a_{ir}\, b_{jr}\, c_{kr}$ | PARAFAC scalar form | PARAFAC element computation |
| $X_{(1)} \approx A\,(C \odot B)^T$ | PARAFAC matrix form (mode 1) | PARAFAC optimization/ALS |
| $A \leftarrow X_{(1)}\,(C \odot B)\,(C^T C * B^T B)^{-1}$ | PARAFAC ALS update for $A$ | PARAFAC ALS derivation |
| $\text{CORCONDIA} = 100 \cdot \left(1 - \dfrac{\|\mathcal{I}-\mathcal{G}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$ | Core consistency diagnostic | Choosing $R$ for PARAFAC |
| $\text{FMS} = \sum_r \dfrac{a_r^T \hat{a}_r \cdot b_r^T \hat{b}_r \cdot c_r^T \hat{c}_r}{\|a_r\|\|\hat{a}_r\|\|b_r\|\|\hat{b}_r\|\|c_r\|\|\hat{c}_r\|}$ | Factor Match Score | Split-half stability |

---

## Common Traps (Wrong Answers in Exams)

- Tucker3 and PARAFAC are completely different models → PARAFAC is a SPECIAL CASE of Tucker3 where $\mathcal{G}$ is super-diagonal
- Tucker3 is unique → Tucker3 is NOT unique; any rotation $Q$ can be applied to $\mathcal{G}$ with compensating change in loading matrices
- PARAFAC is not unique → PARAFAC IS essentially unique — its super-diagonal core constraint prevents arbitrary rotations
- Tucker uses Khatri-Rao product → Tucker uses KRONECKER product ($\otimes$); PARAFAC uses KHATRI-RAO product ($\odot$)
- PARAFAC components are nested (like PCA) → PARAFAC components are NOT nested; changing $R$ changes ALL components
- CORCONDIA close to 0 means good fit → CORCONDIA close to 100 means good fit; close to 0 or negative means $R$ is too large
- CORCONDIA can be used for Tucker3 model selection → CORCONDIA is specifically for PARAFAC (it measures deviation from super-diagonal)
- A high $R$ always gives a better PARAFAC model → High $R$ causes CORCONDIA to drop ($\mathcal{G}$ becomes non-diagonal) → model becomes invalid even if reconstruction improves
- The Frobenius norm of a tensor is defined differently than for matrices → Same concept: square root of sum of all squared elements
- Tucker ranks $P$, $Q$, $R$ must be equal → They can be DIFFERENT for each mode — this is a key advantage of Tucker over PARAFAC
- Unfolding (matricization) changes the data → Unfolding is just a reshape; no information is lost; folding back gives the original tensor
- Tucker is better for resolving physical profiles → PARAFAC is better for physical profiles (unique, additive); Tucker is better for compression
- In split-half analysis, you fit one model and split it → You split the DATA first, then fit SEPARATE models to each half independently

---

## Quick Decision Rules

- If asked which model is a special case of which → PARAFAC is a special case of Tucker3 (with super-diagonal core $\mathcal{G}$)
- If asked which has a unique solution → PARAFAC (unique); Tucker (NOT unique — rotation ambiguity)
- If the question asks which product: Tucker or PARAFAC → Tucker uses Kronecker $\otimes$; PARAFAC uses Khatri-Rao $\odot$
- If asked what CORCONDIA $= 100$ means → The core $\mathcal{G}$ is perfectly super-diagonal → $R$ is appropriate for PARAFAC
- If asked what CORCONDIA $= 0$ or negative means → $\mathcal{G}$ deviates strongly from super-diagonal → $R$ is too large
- If the question is about image compression → Tucker3 (flexible ranks per mode, compress each independently)
- If the question is about separating spectral/kinetic profiles → PARAFAC (additive, interpretable, unique)
- If asked why PARAFAC is unique but Tucker is not → Tucker can rotate $\mathcal{G}$ by any $Q$; PARAFAC's super-diagonal $\mathcal{G}$ cannot be rotated and remain super-diagonal
- If asked how to compute $\mathcal{G}$ after fitting PARAFAC loading matrices → $\mathcal{G} \leftarrow \mathcal{X} \times_1 A^{-1} \times_2 B^{-1} \times_3 C^{-1}$
- If asked for the Tucker ALS update for $A$ → Define $Z_A = G_{(1)}(C \otimes B)^T$, then $A = X_{(1)} Z_A^T (Z_A Z_A^T)^{-1}$
- If asked for PARAFAC ALS update for $A$ → $A = X_{(1)}(C \odot B)(C^T C * B^T B)^{-1}$
- If asked which method for choosing $R$ in PARAFAC → CORCONDIA and/or Split-half FMS (both together for confirmation)
- If asked why PARAFAC components are not nested → Fitting $R=3$ gives three components that are JOINTLY optimal; none of them equals any component from $R=2$
- If mode-$n$ multiplication question → unfold along mode $n$, multiply by $M$, fold back; compact form: $[\mathcal{X} \times_n M]_{(n)} = M X_{(n)}$
