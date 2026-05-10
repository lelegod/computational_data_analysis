# Week 11 — Decomposition Methods for Unsupervised Learning

## Overview
Week 11 covers four matrix decomposition methods that learn structure from unlabelled data by approximating a data matrix $X \approx WH$ with different constraints. The four methods are: Non-negative Matrix Factorization (NMF, parts-based additive representation), Independent Component Analysis (ICA, separates statistically independent non-Gaussian sources), Archetypal Analysis (AA, finds extreme prototypes on the convex hull), and Sparse Coding (overcomplete dictionary with sparsity constraint). Each imposes a unique structural constraint that makes it suited for specific problems.

---

## Factor Analysis and Matrix Decomposition — Core Concept

The unifying idea: approximate a complex data matrix **X** as the product of lower-dimensional factors:

$$X \approx WH$$

This reveals hidden structures, latent variables, and meaningful representations.

**Four Methods, Four Unique Constraints:**
| Method | Constraint | What it finds |
|--------|-----------|--------------|
| NMF | $W \geq 0$, $H \geq 0$ | Parts-based, additive representation |
| ICA | Statistical independence + non-Gaussianity of rows of $H$ | Independent signal sources |
| AA | Convex hull constraint on $W$, convex mixture constraint on $H$ | Extreme prototypes (boundary) |
| Sparse Coding | $L_1$ penalty on $H$ (sparsity), overcomplete $W$ | Sparse dictionary codes |

---

## Non-negative Matrix Factorization (NMF)

### Key Concepts
- Approximate a non-negative matrix $\mathbf{X} \geq 0$ as $\mathbf{X} \approx WH$ where $\mathbf{W} \geq 0$ and $\mathbf{H} \geq 0$
- $\mathbf{W} \in \mathbb{R}^{I \times K}$: basis/parts matrix (columns = "parts" or dictionary atoms), $I$ features $\times$ $K$ components
- $\mathbf{H} \in \mathbb{R}^{K \times J}$: coefficient/activation matrix (how each sample is a mixture of parts), $K \times J$ samples
- Unlike PCA: non-negativity forces **additive, parts-based** representations — no cancellation between basis vectors
- Classic example: Grolier encyclopedia (30,991 articles) — NMF discovers semantic clusters (legal terms, botany, medical) as separate parts; each document = weighted sum of parts

**Applications:** text mining (topic models), audio source separation (NMF of spectrograms), image decomposition, bioinformatics (gene expression patterns)

### The Ambiguity (Non-uniqueness) Problem
NMF solutions are NOT unique. For any invertible matrix $\mathbf{Q}$:

$$X \approx WH = (WQ^{-1})(QH) = \tilde{W}\tilde{H}$$

As long as $\tilde{W} \geq 0$ and $\tilde{H} \geq 0$, it is an equally valid NMF solution.

**Disambiguation strategies:**
- **Geometric constraints**: minimize the geometric volume of the positive cone spanned by columns of $W$ → forces a unique tight-fitting decomposition
- **Sparsity constraints**: enforce $L_1$ penalties → structural zeros → drastically shrinks the space of valid $Q$

### Objective Function
**Squared Frobenius Norm** (most common):

$$\min_{W,H \geq 0} \frac{1}{2}\|X - WH\|_F^2 = \frac{1}{2} \sum_{i,j} (x_{ij} - (WH)_{ij})^2$$

**Convexity properties** (critical to understand):
- NOT jointly convex in $(W, H)$
- IS convex in $W$ given $H$ fixed
- IS convex in $H$ given $W$ fixed
- This "biconvex" structure motivates **alternating minimization**

### Optimization: Alternating Least Squares (ALS)
Alternate between fixing one matrix and optimizing the other:

1. **Fix $H$, update $W$**: $\min_{W \geq 0} \|X - WH\|_F^2$
2. **Fix $W$, update $H$**: $\min_{H \geq 0} \|X^T - H^T W^T\|_F^2$

- Strictly requires Non-Negative Least Squares (NNLS) solver
- **Fast ALS**: solve unconstrained least squares and project negative values to 0 (or small $\varepsilon$)
- Highly parallelizable across columns of $X$

### Optimization: Multiplicative Updates (Lee & Seung, 1999)
Guarantees non-negativity automatically if initialized with positive values.

**Update for $H$:**

$$H_{kj} \leftarrow H_{kj} \cdot \frac{(W^T X)_{kj}}{(W^T WH)_{kj}}$$

**Update for $W$:**

$$W_{ik} \leftarrow W_{ik} \cdot \frac{(XH^T)_{ik}}{(WHH^T)_{ik}}$$

**Derivation (for $H$):**
- Standard gradient for $H$: $\nabla_H J = W^T WH - W^T X$
- Standard update: $H \leftarrow H - \eta_H \circ \nabla_H J$
- Choose spatially-varying learning rate: $\eta_H = H / (W^T WH)$ (element-wise)
- Substitute: $H \leftarrow H - \left[\frac{H}{W^T WH}\right] \circ (W^T WH - W^T X)$
- Simplify: $H \leftarrow H \circ \frac{W^T X}{W^T WH}$
- Key insight: the subtraction term cancels → only multiplication remains → if $H$ starts positive, it stays positive

### Worked Numerical Example
Given: $X = [4;\ 6]$, $W_\text{old} = [1;\ 1]$, $H_\text{old} = [1]$

**Update $H$ (fix $W$):**
- Numerator $(W^T X)$: $[1,1][4;6] = 10$
- Denominator $(W^T W H_\text{old})$: $([1,1][1;1]) \cdot 1 = 2 \cdot 1 = 2$
- $H_\text{new} = 1 \cdot (10/2) = \mathbf{5}$

**Update $W$ (fix $H = 5$):**
- Numerator $(XH^T)$: $[4;6] \cdot 5 = [20;30]$
- Denominator $(W_\text{old} H_\text{new} H_\text{new}^T)$: $[1;1] \cdot (5 \cdot 5) = [25;25]$
- $W_\text{new} = [1 \cdot 20/25;\ 1 \cdot 30/25] = \mathbf{[0.8;\ 1.2]}$

Verification: $W_\text{new} \cdot H_\text{new} = [0.8;1.2] \cdot 5 = [4;6] = X$ ✓

---

## Independent Component Analysis (ICA)

### Key Concepts
- **Blind Source Separation**: observed signals $x(t)$ are linear mixtures of unknown independent sources $s(t)$
- Mixing model: $x = As$, where $A$ is the unknown mixing matrix
- Goal: find un-mixing matrix $W \approx A^{-1}$ such that $\hat{s} = Wx$ recovers the original sources
- Classic example: **Cocktail Party Problem** — two microphones record two speakers:
  - $x_1 = a_{11}s_1 + a_{12}s_2$
  - $x_2 = a_{21}s_1 + a_{22}s_2$

**Why not PCA?** PCA finds orthogonal directions of variance. It can decorrelate signals but cannot fully separate statistically independent sources with non-Gaussian distributions.

### Two Key Assumptions of ICA
1. The source components $s_i$ are **statistically independent**
2. The source components $s_i$ have **non-Gaussian distributions**

### The Central Limit Theorem Foundation
- By CLT: the mixture $x = \sum a_i s_i$ is **more Gaussian** than the original sources
- Strategy: find $W$ that produces signals that are **as non-Gaussian as possible**
- The most non-Gaussian projection corresponds to the original source signals
- This is the fundamental insight that makes ICA work

### Measuring Non-Gaussianity: Excess Kurtosis
**Excess Kurtosis** $= \mu_4/\sigma^4 - 3$

| Distribution | Excess Kurtosis |
|-------------|-----------------|
| Gaussian (Normal) | $0$ |
| Laplace | $3$ (heavy-tailed, super-Gaussian) |
| Uniform | $-1.2$ (sub-Gaussian) |
| Logistic | $1.2$ |
| Raised cosine | $\approx -0.594$ |

- Kurtosis $> 0$: super-Gaussian (sharper peak, heavier tails than Gaussian)
- Kurtosis $< 0$: sub-Gaussian (flatter peak, lighter tails)
- Kurtosis $= 0$: Gaussian (ICA CANNOT separate Gaussian sources — CLT argument fails)

### Pre-processing: Centering and Whitening
Before ICA, always:
1. **Centering**: subtract the mean → zero-mean variable
2. **Whitening (Sphering)**: transform data so components are uncorrelated and unit-variance:
   $$\mathbb{E}[\tilde{x}\tilde{x}^T] = I$$
   - Achieved via PCA: $\tilde{x} = \Lambda^{-1/2} U^T x$, where $U$ and $\Lambda$ are eigenvectors and eigenvalues of the covariance matrix

**Why whiten?** Reduces the ICA problem from finding arbitrary $W$ to finding an **orthogonal matrix** (rotation) only. After whitening, $A$ becomes an orthogonal matrix, vastly simplifying the optimization.

### FastICA Algorithm (Hyvärinen, 1999)
Newton-like fixed-point algorithm. Converges cubically/quadratically (much faster than gradient descent).

**Iteration rule for one component:**

$$w_\text{new} \leftarrow \mathbb{E}[\tilde{x} \cdot g(w^T \tilde{x})] - \mathbb{E}[g'(w^T \tilde{x})] \cdot w$$

Then normalize: $w \leftarrow w / \|w\|$

where:
- $g(\cdot)$ is a non-linearity that measures non-Gaussianity, e.g., $g(u) = \tanh(a_1 u)$ (robust to outliers)
- First term: gradient step toward maximum non-Gaussianity
- Second term: Newton correction (curvature adjustment)
- Normalization: projects back onto the whitened sphere (constrains to rotations)

**Why normalize?** Normalization ensures we search only over rotations of the whitened sphere, not arbitrary scaling.

### Extracting Multiple Independent Components
Two strategies to avoid finding the same component twice:

**1. Deflationary (Sequential):**
- Find component $w_1$
- When searching for $w_2$: project out $w_1$ component:
  $$w_2 \leftarrow w_2 - (w_2^T w_1)w_1, \text{ then normalize}$$
- Repeat for each subsequent component

**2. Symmetric (Parallel):**
- Optimize all $w_i$ simultaneously
- After each iteration, symmetrically orthogonalize all vectors:
  $$W \leftarrow (WW^T)^{-1/2} W$$

---

## Archetypal Analysis (AA)

### Key Concepts
- PCA finds the **average** profile and directions of variance
- K-means finds **centroids** (interior average points)
- AA finds **archetypes** = extreme points on the **convex hull** (boundary) of the data
- Every observation described as a **convex mixture** of these extreme archetypes
- Archetypes must be composed of actual data points (not arbitrary vectors)
- Useful when "pure types" or extremes are more meaningful than averages

**Example:** In acoustic feature space (Pitch vs. Spectral Flux):
- PCA: finds average vocal profile
- AA: finds $A_1$ = purest bass voice, $A_2$ = purest high-pitched voice, $A_3$ = purest whisper/noise
- Every recording = mixture of these three pure corners

### Objective Function
Minimize reconstruction error:

$$\min_{S,H} \|X - XSH\|_F^2$$

**Two matrices with constraints:**

**$S$ matrix (Archetype Formation), shape $K \times J_\text{data}$:**
- $s_{ij} \geq 0$ and $\sum_i s_{ij} = 1$ (columns sum to 1)
- Forces each archetype to be a convex combination of real data points

**$H$ matrix (Data Reconstruction), shape $K \times J$:**
- $h_{ij} \geq 0$ and $\sum_i h_{ij} = 1$ (columns sum to 1)
- Forces each reconstructed data point to be a convex combination of archetypes

### Why $XSH$ (not $WH$)?
**Two-stage geometric constraint:**

- **Step 1** — Define archetypes: $Z = XS$
  - Each archetype is a weighted average of real data points
  - Cannot invent points outside the data cloud
- **Step 2** — Reconstruct data: $\hat{X} = ZH = (XS)H = XSH$
  - Each data point is a fractional mixture of the archetypes

Prototype formula for $d$-th archetype: $w_d = Xs_d$

### Comparison: NMF vs AA vs K-means
| Aspect | NMF | AA | K-means |
|--------|-----|----|---------|
| Prototype location | Arbitrary ($W \geq 0$) | On data convex hull | Interior centroids |
| Reconstruction | Additive | Convex mixture | Nearest prototype |
| Constraint | Non-negativity | Doubly convex | None |
| Prototypes are data? | No | Yes ($Z=XS$) | No |

---

## Sparse Coding

### Key Concepts
- Learn an **overcomplete dictionary** $W$ to represent each data point using **as few active components as possible**
- **Overcomplete**: $K > I$ (more basis vectors than data dimensions)
- Model: $x \approx Wh$
  - $\mathbf{W} \in \mathbb{R}^{I \times K}$: The Dictionary (basis functions)
  - $\mathbf{h} \in \mathbb{R}^K$: The Sparse Code (most entries = exactly 0)
- Biological motivation (Olshausen & Field, 1996): model of how V1 (primary visual cortex) processes information — only a tiny fraction of neurons fire for any given image; basis functions look like Gabor wavelets

### Objective Function
Minimize reconstruction error + sparsity penalty:

$$L(W, H) = \frac{1}{2}\|X - WH\|_F^2 + \lambda \sum_j \|h_j\|_1$$

- **$L_1$ norm** ($\sum |h_i|$): convex proxy for $L_0$ (counting non-zeros)
- **Shrinkage effect**: $L_1$ penalty pushes small coefficients to **exactly zero** (unlike $L_2$ which shrinks toward zero but never reaches it)
- $\lambda$ controls sparsity/reconstruction tradeoff: larger $\lambda$ → sparser $h$, worse reconstruction

### Optimization: Alternating Minimization
Not jointly convex in $(W, H)$, but convex in each when the other is fixed.

**Step 1 — Sparse Coding (fix $W$, update each $h_j$):**

$$\min_{h_j} \frac{1}{2}\|x_j - Wh_j\|_2^2 + \lambda\|h_j\|_1$$

This is exactly the **Lasso problem!** Algorithms:
- **Coordinate Descent**: update one coordinate of $h$ at a time (modern standard)
  - For each $k$: $h_k \leftarrow \text{soft\_threshold}\!\left(h_k + \frac{w_k^T(x - Wh)}{\|w_k\|^2},\ \frac{\lambda}{\|w_k\|^2}\right)$
- **LARS (Least Angle Regression)**: stage-wise algorithm that adds active variables one at a time

**Step 2 — Dictionary Update (fix $H$, update $W$):**

$$\min_W \frac{1}{2}\|X - WH\|_F^2 \quad \text{subject to } \|w_k\|_2^2 \leq 1 \text{ for all } k$$

**Why the unit norm constraint?** Without it, the model can set $W$ arbitrarily large and scale $H$ proportionally small, driving the $L_1$ penalty to zero while the reconstruction stays unchanged — the constraint prevents this trivial solution.

Solution: standard least squares applied to each column $w_k$, then project to unit sphere.

### Choosing Number of Components: Speckled CV (Matrix Masking)
Standard row-holdout cross-validation **fails** for matrix methods:
- Problem: if you hold out a row, you cannot learn the $H$ coefficients (mixture weights) for that sample during training, so you cannot reconstruct it for evaluation

**Speckled CV (Matrix Masking):**
1. **Mask**: randomly hide a percentage of individual entries in $X$ (mark as missing)
2. **Fit**: train the model for $k$ components, ignoring masked entries during loss:
   $$\text{train loss} = \frac{1}{2}\sum_{i,j:\ \text{observed}} (x_{ij} - (WH)_{ij})^2$$
3. **Reconstruct**: use trained $W$, $H$ to predict ALL entries including masked ones
4. **Evaluate**: compute MSE ONLY on originally masked entries:
   $$\text{CV Error} = \sum_{\text{masked}\ i,j} (X_{ij} - \hat{X}_{ij})^2$$
5. Pick $k$ that minimizes CV Error (use elbow criterion if no clear minimum)

**With mask matrix $M$** ($m_{ij}=0$ if masked, $1$ if observed):
- Train loss: $\frac{1}{2}\|(1-M) \odot (X - WH)\|_F^2$
- Test loss: $\frac{1}{2}\|M \odot (X - WH)\|_F^2$

Works because model must "fill in" masked entries by learning the underlying low-rank structure.

### Modern Application: Mechanistic Interpretability of LLMs
- **Problem (Polysemanticity)**: individual neurons in LLMs fire for multiple unrelated concepts (e.g., "Canada" AND "Syntax Errors") → uninterpretable black boxes
- **Hypothesis**: neural networks use superposition — representing more features than dimensions via near-orthogonal sparse codes
- **Solution (Sparse Autoencoders)**: train a sparse coder on internal activations of an LLM
  - Dictionary $W \in \mathbb{R}^{d \times K}$ with $K \gg d$ (overcomplete)
  - Sparse code $h$ for each activation
  - Result: each dictionary atom $w_k$ becomes a **monosemantic** feature (one concept)
- Applications: understanding circuits in GPT-4, Claude, etc.

---

## Comparison of All Four Methods

| Method | Unique constraint | Prototype location | Interpretable? | Unique solution? |
|--------|-----------------|-------------------|---------------|-----------------|
| NMF | Non-negativity ($W,H \geq 0$) | Arbitrary positive | Parts-based | No ($Q$-ambiguity) |
| ICA | Statistical independence + non-Gaussian | N/A (mixing matrix) | Independent sources | Yes (up to sign/order) |
| AA | Doubly convex ($Z=XS$, $H$ convex) | On convex hull | Extreme archetypes | Approximately yes |
| Sparse Coding | Sparsity ($L_1$) + overcomplete | Arbitrary | Sparse codes | No |
