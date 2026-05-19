# Week 7 — Lecture Notes
## Computational Data Analysis (02582)

---

## Support Vector Machines (SVM)

### Core Idea

Find the hyperplane that separates two classes with the **largest possible margin**. The points on the edge of the margin are the **support vectors** — they alone define the boundary.

### Hard Margin SVM (Perfectly Separable Data)

Maximise the margin by minimising $\|\beta\|^2$:

$$\min_{\beta,\beta_0} \|\beta\|^2 \quad \text{subject to } y_i(\beta^T x_i + \beta_0) \geq 1 \quad \forall i$$

Margin width $= \dfrac{2}{\|\beta\|}$. Larger margin → smaller $\|\beta\|$.

Only works when classes are linearly separable.

### Soft Margin SVM (Slack Variables)

Allow points to violate the margin, but penalise them with slack $\xi_i \geq 0$:

$$\min_{\beta,\beta_0} \|\beta\|^2 + C\sum_{i=1}^{N} \xi_i \quad \text{subject to } y_i(\beta^T x_i + \beta_0) \geq 1 - \xi_i$$

- $\xi_i = 0$: point correctly outside margin
- $0 < \xi_i \leq 1$: point inside margin but correct side
- $\xi_i > 1$: point misclassified

**Tuning parameter $C$:**
- Large $C$: strict — penalise violations heavily, narrow margin
- Small $C$: tolerant — allow violations, wide margin

### The Kernel Trick (Non-linear Boundaries)

Replace the dot product $x_i^T x_j$ with a kernel function $K(x_i, x_j)$ — implicitly projects data into a higher-dimensional space without computing it explicitly.

| Kernel | Formula |
|---|---|
| Linear | $K(x_i, x_j) = x_i^T x_j$ |
| Polynomial | $K(x_i, x_j) = (1 + x_i^T x_j)^d$ |
| RBF / Gaussian | $K(x_i, x_j) = e^{-\gamma\|x_i - x_j\|^2}$ |

### SVM vs Logistic Regression

| | SVM | Logistic Regression |
|---|---|---|
| Loss function | Hinge loss | Log-loss |
| Which points matter | Only support vectors (near boundary) | All points |
| Solution sparsity | Sparse (few support vectors) | Dense |
| Robustness to outliers | High (far-away points ignored) | Lower |

Both produce a linear decision boundary; the difference is in what they optimise.

---

## Lagrangian Duality in SVM

### Why Duality?

The SVM primal is a constrained optimization problem. Lagrangian duality converts it into an unconstrained form that is easier to solve and reveals the kernel trick.

### The Lagrangian

Introduce multipliers $\alpha_i \geq 0$, one per constraint:

$$L(\beta, \beta_0, \alpha) = \frac{1}{2}\|\beta\|^2 - \sum_{i=1}^{N} \alpha_i\bigl[y_i(\beta^T x_i + \beta_0) - 1\bigr]$$

Violated constraints inflate $L$ → the minimiser naturally avoids them.

### Primal vs Dual

$$p^* = \min_{\beta}\max_{\alpha \geq 0} L(\beta,\alpha) \qquad d^* = \max_{\alpha \geq 0}\min_{\beta} L(\beta,\alpha)$$

Taking $\partial L / \partial \beta = 0$ and $\partial L / \partial \beta_0 = 0$, the dual becomes:

$$\max_{\alpha} \sum_{i=1}^{N}\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j\, x_i^T x_j \quad \text{s.t. } \alpha_i \geq 0,\; \sum_i \alpha_i y_i = 0$$

### Weak vs Strong Duality

- **Weak duality:** $d^* \leq p^*$ always. The dual gives a lower bound.
- **Strong duality:** $d^* = p^*$. Holds for SVM because it is a convex problem. Solving the dual gives the exact same answer as the primal.

### Why the Dual is Better

**1. Kernel trick:** Data appears only as dot products $x_i^T x_j$ — replace with $K(x_i, x_j)$ for non-linear SVM.

**2. Sparsity (KKT condition):**

$$\alpha_i\bigl[y_i(\beta^T x_i + \beta_0) - 1\bigr] = 0$$

Either $\alpha_i = 0$ (point irrelevant) or constraint is tight (point is a support vector). Most $\alpha_i = 0$ — only support vectors matter.

**3. Dimensionality:** Dual has $N$ variables vs $p$ in the primal — better when $p \gg N$.

| | Primal | Dual |
|---|---|---|
| Variables | $\beta$ ($p$-dim) | $\alpha$ ($N$-dim) |
| Operation | Minimise | Maximise |
| Data appears as | Raw $x_i$ | Dot products $x_i^T x_j$ |
| Kernel trick | Not obvious | Falls out naturally |
| Support vectors | Not obvious | $\alpha_i > 0$ exactly at SVs |

---

## Canonical Correlation Analysis (CCA)

### Goal

Find the relationship between **two sets of variables** $X$ ($p$-dim) and $Y$ ($q$-dim) by finding linear combinations of each that are maximally correlated.

### The Canonical Variates

Find vectors $a$ and $b$ such that:

$$u = a^T X, \qquad v = b^T Y$$

$$\max_{a,\,b} \;\text{Cor}(u,v) = \max_{a,\,b} \frac{a^T \Sigma_{XY}\,b}{\sqrt{a^T \Sigma_{XX}\,a \cdot b^T \Sigma_{YY}\,b}}$$

The pair $(u_1, v_1)$ is the **first canonical pair**. Subsequent pairs are orthogonal to previous ones. There are $\min(p,q)$ canonical pairs in total.

### The Canonical Correlations

The resulting correlation values $\rho_1 \geq \rho_2 \geq \cdots \geq \rho_{\min(p,q)}$:
- $\rho_k = 1$: perfect linear association on dimension $k$
- $\rho_k = 0$: no association on dimension $k$
- The number of large $\rho_k$ tells you how many meaningful "dimensions of association" exist between $X$ and $Y$

### CCA Generalises Other Methods

| Method | CCA special case |
|---|---|
| Multiple regression | $q = 1$ (single response) |
| PCA | $X = Y$ (association within one dataset) |
| LDA | $Y$ is a class indicator matrix |

### Intuition

Like PCA but for **two datasets simultaneously** — instead of maximising variance, you maximise cross-correlation between the two groups.
