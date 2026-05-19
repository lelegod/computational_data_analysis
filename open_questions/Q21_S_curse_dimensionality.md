# Q21-S — Curse of Dimensionality
> Week 3. Conceptual but mathematically grounded; often asked in context of KNN or density estimation.

---

## The Core Problem

In high dimensions ($p$ large), the geometry of space behaves counterintuitively. Distances lose meaning, volumes concentrate in shells, and nearest neighbors are no longer "near."

---

## Volume Concentrates in the Shell

Consider a $p$-dimensional unit hypersphere. The fraction of volume within distance $\epsilon$ of the surface:
$$\frac{\text{Vol}(B_p(1)) - \text{Vol}(B_p(1-\epsilon))}{\text{Vol}(B_p(1))} = 1-(1-\epsilon)^p \to 1 \text{ as } p\to\infty$$

For $\epsilon=0.05$, $p=100$: fraction $= 1-0.95^{100} \approx 99.4\%$. Nearly all volume is in a thin shell near the surface.

**Implication**: a uniformly distributed dataset in $p$ dimensions has almost all points near the boundary — interior density estimates are unreliable.

---

## Nearest Neighbors Become Distant

To capture a fixed fraction $r$ of the training data in a $p$-dimensional hypercube using a local neighborhood, the neighborhood must have edge length:
$$l = r^{1/p}$$

For $r=0.01$ (1% of data), $p=10$: $l = 0.01^{0.1} \approx 0.63$ — the neighborhood spans 63% of the range in each dimension. This is no longer "local."

**Implication**: KNN and local kernel methods require exponentially more data to maintain the same density of neighbors as $p$ grows.

---

## All Distances Become Similar

For $N$ IID points in $\mathbb{R}^p$, the ratio of maximum to minimum pairwise Euclidean distances:
$$\frac{d_\text{max} - d_\text{min}}{d_\text{min}} \to 0 \text{ as } p\to\infty$$

In high dimensions, all pairs of points have approximately the same distance. Nearest-neighbor queries lose discriminative power.

---

## Sparsity of High-Dimensional Grids

To cover $p$-dimensional space with a grid of $m$ points per dimension requires $m^p$ total grid points.

For $m=10$ points per dimension: $p=1$: 10 points. $p=2$: 100. $p=10$: $10^{10}$. $p=100$: $10^{100}$ — far more than atoms in the observable universe.

The training data is vanishingly sparse relative to the volume of the space.

---

## Implications for Statistical Models

| Method | How it suffers | Fix |
|--------|---------------|-----|
| KNN | Neighbors are far away — not local | Use small $K$; reduce $p$ first |
| Kernel density estimation | Bandwidth must be huge to capture any data | Dimensionality reduction |
| OLS (no regularization) | $p>N$: singular $X^TX$, no unique solution | Ridge, Lasso, PCR |
| Parametric models | Number of parameters $\sim p^2$ or $p!$ for interactions | Sparse models, structured assumptions |
| All distance-based methods | All distances converge | Transform features, use inner products |

---

## The Blessing of Dimensionality (Donoho 2000)

High dimensions are not always bad. When the data has **low intrinsic dimensionality** (lies near a low-dimensional manifold) or **sparse structure** (few relevant features), high-dimensional methods can exploit this:

- **Manifold hypothesis**: real data (images, text, speech) lives near a low-dimensional manifold embedded in high-dimensional space. Methods like PCA, autoencoders, t-SNE discover this structure.
- **Sparsity**: if the true signal involves only $s \ll p$ features, Lasso can recover it with $N \sim s\log(p)$ observations — much less than $p$.
- **Random projections**: by the Johnson-Lindenstrauss lemma, $N$ points in $\mathbb{R}^p$ can be projected to $\mathbb{R}^k$ (with $k \sim \log N$) while approximately preserving all pairwise distances.
- **Linear separability**: in very high dimensions, random classes are often linearly separable (SVM benefits from this).

---

## KNN as a Concrete Example

KNN with $K=1$ in $p=1$: uses the 1 nearest neighbor → very local, low bias.

The same KNN in $p=100$: the "nearest" neighbor may be far away (all distances converge). The neighborhood is no longer local → the prediction is based on a non-local average → KNN becomes equivalent to the global mean → high bias.

**EPE of 1-NN**: $\text{EPE}(x_0) = \sigma^2 + \frac{2}{N}\sum_i \text{Cov}(\hat{y}(x_i),y_i)$. In low $p$: neighbors are close → low bias. In high $p$: neighbors are far → bias term grows.

---

## Additional Possible Exam Questions

**Q: Why does regularization help in high dimensions?**
With $p > N$: OLS has infinitely many solutions (underdetermined system). Ridge selects the minimum-norm solution, which corresponds to assuming small coefficients. Lasso selects a sparse solution, assuming few features matter. Both impose structure that compensates for the lack of data relative to the number of parameters. Without structure, any estimate has extremely high variance.

**Q: What is the connection between the curse of dimensionality and overfitting?**
In high dimensions, the training data is sparse — the model can always find a complex function that passes through the training points (memorization) because there is so much "empty space" between them. The memorized function will perform poorly on new test points. This is the geometric root of overfitting: high-dimensional space has room to overfit.

**Q: What is the intrinsic dimensionality of a dataset?**
The number of dimensions needed to represent the data structure, even if the ambient dimension $p$ is much larger. Example: a 2D sheet crumpled in 3D has intrinsic dimension 2 even though it lives in $\mathbb{R}^3$. PCA estimates intrinsic dimensionality by finding how many principal components explain most variance (elbow in scree plot). Methods like ISOMAP, LLE, and UMAP discover nonlinear intrinsic structure.

**Q: Why are inner products more stable than Euclidean distances in high dimensions?**
Euclidean distance $\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2\langle x,y\rangle$. In high dimensions, $\|x\|^2$ and $\|y\|^2$ concentrate (law of large numbers) → all norms are similar → the distance is dominated by noise. The inner product $\langle x,y\rangle$ directly measures alignment and can still discriminate. This is why kernel methods (which use inner products) and cosine similarity (normalized inner product) remain effective in high dimensions, while raw Euclidean distance does not.
