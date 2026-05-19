# Q21-F — PARAFAC vs Tucker Tensor Decompositions
> Related: 2022 Q16, 2024 Q17

---

## Tensors: Setup

A 3-way tensor $\mathcal{X} \in \mathbb{R}^{I\times J\times K}$ has three modes: e.g., samples × variables × time.

Matrix unfolding (matricization) of mode $n$: rearrange the tensor into a matrix by fixing one mode. $X_{(1)} \in \mathbb{R}^{I\times JK}$ is the mode-1 unfolding.

---

## Tucker3 Model

$$\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$

- $A \in \mathbb{R}^{I\times P}$, $B \in \mathbb{R}^{J\times Q}$, $C \in \mathbb{R}^{K\times R}$: factor matrices (one per mode)
- $\mathcal{G} \in \mathbb{R}^{P\times Q\times R}$: **core tensor** — encodes all interactions between components
- Ranks $P, Q, R$ can be different for each mode
- Mode-$n$ product $\times_n A$: multiply the $n$-th mode by matrix $A$

**Matrix form** (mode-1 unfolding):
$$X_{(1)} \approx A\, G_{(1)}\,(C \otimes B)^T$$

where $\otimes$ is the Kronecker product.

**Core tensor interpretation**: Entry $g_{pqr}$ measures the interaction between component $p$ in mode 1, component $q$ in mode 2, and component $r$ in mode 3. If $g_{pqr} = 0$, those components are independent.

**Tucker3 is NOT unique**: for any invertible $P\times P$ matrix $Q$, the substitution $A \to AQ^{-1}$, $\mathcal{G} \to \mathcal{G}\times_1 Q$ gives the same reconstruction. This rotation ambiguity means components cannot be uniquely interpreted without additional constraints (e.g., orthogonality, which Tucker imposes by convention).

**Best for**: data compression — independently choosing small $P,Q,R$ compresses each mode.

---

## PARAFAC (CP) Model

$$\mathcal{X} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r$$

- $\circ$ denotes outer product: each term $a_r \circ b_r \circ c_r$ is a rank-1 tensor
- $A = [a_1,\ldots,a_R]$, $B = [b_1,\ldots,b_R]$, $C = [c_1,\ldots,c_R]$
- Single rank $R$ for all modes

**Matrix form** (mode-1 unfolding):
$$X_{(1)} \approx A(C \odot B)^T$$

where $\odot$ is the **Khatri-Rao product** (column-wise Kronecker): $(C\odot B)_{:,r} = c_r \otimes b_r$.

**Relationship to Tucker3**: PARAFAC = Tucker3 with super-diagonal core $\mathcal{G}$:
$$g_{pqr} = \begin{cases}1 & p=q=r \\ 0 & \text{otherwise}\end{cases}$$

This means component $r$ in mode 1 interacts ONLY with component $r$ in modes 2 and 3. No cross-talk between different components.

**PARAFAC IS essentially unique** (under mild conditions): the super-diagonal constraint prevents arbitrary rotations. Uniqueness holds when $R$ satisfies the Kruskal condition. This uniqueness is the main advantage — components have physical meaning.

**Components are NOT nested**: changing $R$ from 3 to 4 changes ALL components (unlike PCA where the first 3 components don't change when you add a 4th).

**Best for**: resolving physically interpretable profiles (spectroscopy, EEG, chromatography).

---

## CORCONDIA (Core Consistency Diagnostic)

Measures how close the fitted PARAFAC model's normalized core is to the ideal super-diagonal:

$$\text{CORCONDIA} = 100\left(1 - \frac{\|\mathcal{I}-\tilde{\mathcal{G}}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$

where $\mathcal{I}$ is the identity (super-diagonal) core and $\tilde{\mathcal{G}}$ is the normalized estimated core.

| CORCONDIA | Interpretation |
|-----------|---------------|
| $\approx 100$ | $R$ is appropriate — core is nearly super-diagonal |
| $50–90$ | Marginal — consider trying $R-1$ |
| $< 50$ or negative | $R$ too large — forcing extra components destroys super-diagonal structure |

**How to use**: fit PARAFAC for $R=1,2,3,\ldots$; plot CORCONDIA vs $R$. Choose the largest $R$ before CORCONDIA drops sharply.

**Split-half FMS**: additional validation. Split data into two halves, fit PARAFAC to each, compare factors via Factor Match Score (FMS). High FMS = stable, reproducible components.

---

## Comparison Table

| Property | Tucker3 | PARAFAC |
|----------|---------|---------|
| Ranks | $(P,Q,R)$ — one per mode | Single $R$ |
| Core tensor | Full $P\times Q\times R$ | Super-diagonal (identity) |
| Special case of other? | More general | Special case of Tucker3 |
| Unique? | No (rotation ambiguity) | Yes (essentially unique) |
| Number of parameters | $IP+JQ+KR+PQR$ | $(I+J+K)R$ |
| Best for | Compression | Physical interpretation |
| Mode-1 matrix form | $AG_{(1)}(C\otimes B)^T$ | $A(C\odot B)^T$ |
| Product used | Kronecker $\otimes$ | Khatri-Rao $\odot$ |

---

## Additional Possible Exam Questions

**Q: Why does PARAFAC need non-trivial conditions for uniqueness, while Tucker does not have uniqueness at all?**
Tucker's core can be right-multiplied by any invertible matrix without changing the model. PARAFAC's super-diagonal core cannot be freely rotated because the constraint $g_{pqr}=0$ for $p\neq q$ or $p\neq r$ is not preserved under general rotations. Kruskal (1977) proved uniqueness holds when $k_A+k_B+k_C \geq 2R+2$, where $k_A$ is the k-rank of $A$.

**Q: What is a rank-1 tensor?**
A rank-1 tensor can be written as the outer product of vectors: $\mathcal{X} = a\circ b\circ c$. Element: $x_{ijk} = a_i b_j c_k$. PARAFAC decomposes the tensor as a sum of $R$ rank-1 tensors.

**Q: What is the role of $\mathcal{G}$ in Tucker3?**
The core tensor describes the structure of interactions. If $\mathcal{G}$ were super-diagonal, the model reduces to PARAFAC. Off-diagonal elements in $\mathcal{G}$ capture "mixing" between different components across modes. A Tucker3 with orthogonal factors $A,B,C$ is the tensor analogue of SVD (Higher-Order SVD, HOSVD).

**Q: In a fluorescence spectroscopy experiment (samples × excitation wavelengths × emission wavelengths), why is PARAFAC preferred?**
Each fluorescent compound contributes an independent rank-1 component: $a_r \circ b_r \circ c_r$ where $a_r$ = concentration profile across samples, $b_r$ = excitation spectrum, $c_r$ = emission spectrum. These are physically real profiles. PARAFAC's uniqueness guarantees that the recovered spectra are the true pure-component spectra, not linear combinations. Tucker3 would mix components arbitrarily due to rotation ambiguity.

**Q: Why are Tucker ranks chosen as $(P,Q,R)$ rather than a single $R$?**
Different modes may have different intrinsic complexity. For example, in a (samples × genes × time) tensor: there may be 5 distinct sample groups ($P=5$), 100 gene patterns ($Q=100$), but only 3 temporal phases ($R=3$). Forcing a single $R$ would be wasteful in some modes and underpowered in others. Tucker independently compresses each mode to its appropriate intrinsic dimension.
