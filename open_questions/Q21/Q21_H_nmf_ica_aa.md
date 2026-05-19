# Q21-H — NMF / ICA / AA / PCA Comparison
> Comparing matrix factorization and decomposition methods with different constraints

---

## General Framework

All methods approximate $X \approx WH$ (or equivalent), but with different constraints on $W$ and $H$ that encode different assumptions about the structure of the data.

$X \in \mathbb{R}^{N\times p}$ (observations × variables), $W \in \mathbb{R}^{N\times K}$ (scores/loadings), $H \in \mathbb{R}^{K\times p}$ (components/dictionary).

---

## PCA — Orthogonal Components, Maximum Variance

**Constraint**: columns of $W$ are orthonormal; rows of $H$ are orthonormal.
$$X \approx WH, \quad W^TW = I_K, \quad HH^T = \Lambda \text{ (diagonal)}$$

**Objective**: Minimize $\|X - WH\|_F^2$ subject to orthogonality.

**Solution**: $H$ = top $K$ right singular vectors of $X$; $W$ = scores $XV^T$.

**Properties**:
- Unique (up to sign flip per component)
- Nested: top-$K$ solution uses same components as top-$(K-1)$
- Components may contain positive and negative values (allowing cancellation)
- Prototypes (loadings) are interior — each component averages the data
- No interpretability guarantee beyond variance maximization

---

## NMF — Non-Negative Matrix Factorization

**Constraint**: $W \geq 0$, $H \geq 0$ (all entries non-negative).
$$X \approx WH, \quad W_{ik} \geq 0, \; H_{kj} \geq 0$$

**Objective**: Minimize $\|X-WH\|_F^2$ (or KL divergence for count data).

**Algorithm**: Multiplicative updates (Lee & Seung):
$$H \leftarrow H \cdot \frac{W^TX}{W^TWH}, \quad W \leftarrow W \cdot \frac{XH^T}{WH H^T}$$

**Why non-negativity matters**:
- No cancellation: reconstruction is a **sum of parts**, not a sum-and-subtract
- Each component represents an additive "part" of the data
- Natural for: image pixels, word counts, spectra (all non-negative)

**Properties**:
- NOT unique ($Q$-ambiguity: $W \to WQ$, $H \to Q^{-1}H$ for non-negative $Q$)
- NOT nested (changing $K$ changes all components)
- Local optima (gradient descent, multiplicative updates)
- Sparse solutions tend to emerge naturally

---

## ICA — Independent Components

**Constraint**: rows of $H$ (sources $s_i$) are statistically independent AND non-Gaussian.
$$x = As = H^{-1}s, \quad \hat{s} = Wx$$

**Objective**: maximize non-Gaussianity of recovered sources (via kurtosis or negentropy).

**Properties**:
- Unique up to permutation and sign of components (NOT rotation)
- Requires non-Gaussian sources (Gaussian = completely unidentifiable)
- Pre-processing: whiten data ($E[\tilde{x}\tilde{x}^T]=I$), then optimize rotation
- Best for: cocktail party problem, EEG source separation
- Components can be positive and negative

---

## Archetypal Analysis (AA)

**Constraint**: archetypes $Z = XS$ (convex combinations of data); data $X \approx ZH = XSH$ (convex combinations of archetypes). Both $S$ and $H$ are convex weight matrices (rows sum to 1, all entries $\geq 0$).

**Objective**: 
$$\min_{S,H} \|X - XSH\|_F^2 \quad \text{s.t.} \quad S_{kj}\geq0, \sum_j S_{kj}=1, H_{ik}\geq0, \sum_k H_{ik}=1$$

**Key property**: Archetypes lie on (or near) the **convex hull** of the data — they are extreme observations, not averages.

**Why this is useful**: If the data is mixture of "pure types" (patient phenotypes, spectral end-members), archetypes represent those pure types. Every other observation is a mix of the archetypes.

**Properties**:
- Archetypes are interpretable as "extreme cases"
- Convex hull constraint makes archetypes data-driven (no parametric assumption)
- More interpretable than PCA components when extreme phenotypes exist
- Computationally more expensive (non-convex optimization)

---

## Sparse Coding

**Constraint**: $H$ is sparse (each observation $x_i$ is explained by few dictionary atoms).
$$\min_{W,H} \|X-WH\|_F^2 + \lambda\|H\|_1$$

where $W$ is the dictionary (learned) and $H$ has sparse coefficients.

**Connection to ICA**: Sparse coding with a sparsity prior on $H$ is equivalent to ICA with Laplacian (super-Gaussian) source distributions. The Laplace prior $\propto \exp(-|h|/\sigma)$ favors sparse solutions — this is the probability model underlying L1-penalized ICA.

---

## Full Comparison Table

| Method | Constraint on components | Unique? | Prototypes at | Best for |
|--------|--------------------------|---------|---------------|----------|
| PCA | Orthogonal | Yes (up to sign) | Interior (weighted mean) | Variance, compression |
| NMF | Non-negative | No ($Q$-ambiguity) | Interior (additive parts) | Non-negative data, parts |
| ICA | Independent, non-Gaussian | Yes (up to ±perm) | Directions (extremes) | Source separation |
| AA | Convex hull + convex mix | Partially | Boundary (extreme points) | Extreme phenotypes |
| K-means | Hard cluster assignments | No (local opt) | Interior (centroids) | Clustering |
| Sparse coding | Sparse $H$ | Approximately | Depends on dictionary | Feature learning |

---

## Additional Possible Exam Questions

**Q: Why does NMF produce "parts-based" representations?**
Because $W \geq 0$ and $H \geq 0$, each reconstruction $x_i \approx \sum_k w_{ik} h_k$ is a pure sum with no subtractions. Each component $h_k$ represents a part (e.g., eyes, nose, mouth in face images), and $w_{ik}$ indicates how much of each part is present. PCA would represent faces as global patterns that can subtract from each other (eigen-faces), which is less interpretable as "parts."

**Q: When would you choose AA over PCA or NMF?**
When you believe the data comes from a mixture of extreme "pure" types and you want to find those extremes. Clinical example: patients who are clearly diabetic, clearly healthy, clearly obese — these archetypes are on the boundary of the data cloud. Intermediate patients are mixtures. PCA finds the cloud's principal axes; NMF finds additive parts; AA finds the corners.

**Q: Why is K-means a special case of both GMM and NMF?**
K-means as GMM: spherical equal-variance GMM + hard assignments. K-means as NMF: if $H$ is the cluster indicator matrix ($H_{ij}=1$ iff point $i$ belongs to cluster $j$) and $W$ = centroid matrix, then minimizing $\|X-WH\|_F^2$ over non-negative matrices $W,H$ with one-hot $H$ is K-means.

**Q: What is the practical difference between ICA and sparse PCA?**
Both find non-Gaussian directions. ICA enforces statistical independence (all higher-order cumulants); sparse PCA adds $L_1$ penalty to enforce sparsity of loadings (few non-zero weights). ICA components are dense but independent; sparse PCA components are sparse but only uncorrelated. Sparse PCA is often preferred in genomics (each component = few active genes).

**Q: Can NMF be applied to data with negative values?**
No — the non-negativity constraint is violated. Solutions: (1) shift data to make it non-negative (add minimum value); (2) use semi-NMF (allow $W$ to have negative values but keep $H\geq0$); (3) use a different method (PCA, ICA).
