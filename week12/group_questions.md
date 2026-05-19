# Week 12 — Group Discussion Questions

## Q1: Think for Yourself — PARAFAC/Tucker with Only Two Modes

**Question (slide 83):** What happens if we do not have a 3D multiway structure? Can we have a PARAFAC or Tucker model with just A and B?

**Answer:**

**Short answer:** Yes — a two-mode (matrix) case is perfectly valid, and PARAFAC and Tucker3 reduce to well-known matrix decompositions.

---

**The 3-mode standard setup (review):**

A three-way tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$ decomposes as:

- **PARAFAC** (CP): $\mathcal{X} \approx \sum_{r=1}^{R} \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r$ — three loading vectors per component, super-diagonal core tensor $\mathcal{G} = \mathcal{I}$.
- **Tucker3**: $\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{A} \times_2 \mathbf{B} \times_3 \mathbf{C}$ — full core tensor $\mathcal{G} \in \mathbb{R}^{P \times Q \times R}$ and three factor matrices.

---

**What happens with only two modes (a matrix $\mathbf{X} \in \mathbb{R}^{I \times J}$)?**

**PARAFAC with 2 modes** reduces to:

$$\mathbf{X} \approx \sum_{r=1}^{R} \mathbf{a}_r \circ \mathbf{b}_r = \mathbf{A}\mathbf{B}^T$$

where $\mathbf{A} \in \mathbb{R}^{I \times R}$ and $\mathbf{B} \in \mathbb{R}^{J \times R}$. This is simply **rank-$R$ matrix factorization** (i.e., SVD / PCA / NMF depending on constraints). The "super-diagonal core" in 2D is just the identity matrix, so there is no separate core — only two factor matrices.

**Tucker with 2 modes** reduces to:

$$\mathbf{X} \approx \mathbf{G} \times_1 \mathbf{A} \times_2 \mathbf{B} = \mathbf{A} \mathbf{G} \mathbf{B}^T$$

where $\mathbf{G} \in \mathbb{R}^{P \times Q}$ is the (now 2D) core matrix. This is precisely the **truncated SVD** when $\mathbf{A}$ and $\mathbf{B}$ are orthonormal: $\mathbf{X} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ with $\mathbf{G} = \mathbf{\Sigma}$, $\mathbf{A} = \mathbf{U}$, $\mathbf{B} = \mathbf{V}$.

---

**Why multiway methods add value for 3+ modes:**

| Property | 2-mode (matrix) | 3-mode (tensor) |
|---|---|---|
| PARAFAC uniqueness | Not unique (rotation freedom, like PCA) | **Generically unique** under mild conditions (Kruskal's theorem) |
| Tucker uniqueness | Not unique | Not unique (rotation freedom in core) |
| Core interpretation | $\mathbf{G}$ is singular values (diagonal for SVD) | $\mathcal{G}$ captures interactions between all three modes |

The key insight is that **PARAFAC gains its uniqueness property specifically because of the third (or higher) mode**. In the 2-mode matrix case, you can always multiply $\mathbf{A}$ by an invertible matrix $\mathbf{Q}$ and $\mathbf{B}$ by $(\mathbf{Q}^{-1})^T$ and get the same $\mathbf{X}$ — this is the standard rotation indeterminacy of PCA/SVD. But for a 3-mode PARAFAC, the diagonal structure of the core tensor constrains the solution and removes this rotation freedom, giving a unique decomposition.

---

**Practical implication:**

If your data is genuinely 2D (a single matrix), use SVD/PCA/NMF — these are well-optimized and theoretically grounded. The multiway framework becomes essential when data has a natural three-way structure (e.g., subjects × time-points × frequencies, or samples × time × wavelength) where the interaction structure across all three modes carries meaning that would be lost by unfolding ("matricizing") the tensor into a 2D matrix.
