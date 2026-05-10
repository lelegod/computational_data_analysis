# Week 7 — Support Vector Machines (SVM)

## Overview
SVMs find the optimal hyperplane that maximally separates two classes by maximizing the margin — the distance from the boundary to the nearest points of each class. There is no probabilistic model (unlike LDA or logistic regression). SVMs have two powerful properties: informational sparsity (only support vectors matter) and the kernel trick (implicit infinite-dimensional feature mapping).

---

## 1. The Decision Function

### Key Concepts
- Classes are labeled **+1** and **-1** (not 0/1 — this is important for the math).
- Linear decision function: $\hat{y}_{\text{new}} = \text{sign}(\beta_0 + x_{\text{new}}^T \beta)$
- Many hyperplanes can separate two linearly separable classes; SVM finds the **optimal** one.
- SVM = Convex Optimization + Implicit Feature Mapping.

### The Two Pillars
1. **Optimal Separating Hyperplane (OSH):** Find the unique boundary maximizing distance to nearest points. Solved via Quadratic Programming. Result: Sparsity.
2. **Feature Space Transformation:** Move data from $\mathbb{R}^d$ to a higher-dimensional space $\mathcal{H}$ where linear separation is possible. Calculated implicitly via the Kernel Trick. Result: Non-linear boundaries.

---

## 2. The Margin and Distance to the Hyperplane

### Key Concepts
- The hyperplane is defined by: $x^T \beta + \beta_0 = 0$
- **$\beta$ is orthogonal (perpendicular) to the hyperplane.**
- Unit normal vector: $\hat{n} = \beta / \|\beta\|$
- The signed distance from point $x_i$ to the hyperplane:

### Formulas
- **Point-to-Plane Distance**: $d = (x_i^T \beta + \beta_0) / \|\beta\|$
  - If $d > 0$: point is on the positive side.
  - If $d < 0$: point is on the negative side.
- **SVM maximizes signed distance**: $y_i \cdot (x_i^T \beta + \beta_0) / \|\beta\|$, where $y_i \in \{-1, 1\}$

### Derivation Steps
1. Unit normal: $\hat{n} = \beta/\|\beta\|$
2. Vector from plane point $x$ to data point $x_i$: $v = x_i - x$
3. Distance = projection of $v$ onto $\hat{n}$: $d = \langle (x_i - x), \hat{n} \rangle = (x_i - x)^T \beta / \|\beta\|$
4. Since $x$ is on the hyperplane: $x^T \beta = -\beta_0$, so $d = (x_i^T \beta + \beta_0) / \|\beta\|$

---

## 3. The OSH Optimization Problem

### Key Concepts
- **Canonical scaling / Canonical Hyperplane:** We fix the scale so that for Support Vectors: $|x_i^T \beta + \beta_0| = 1$
- With canonical scaling, the margin width becomes: $C = 1/\|\beta\|$
- To maximize $C = 1/\|\beta\|$, we minimize $\|\beta\|$, equivalently minimize $\|\beta\|^2$.

### Primal Formulation
- **Geometric Goal**: $\arg\max_{\beta,\beta_0} C$ subject to $y_i(x_i^T \beta + \beta_0)/\|\beta\| \geq C \; \forall i$
- **Computational Goal (Primal):**

$$
\min_{\beta,\beta_0} \; \frac{1}{2}\|\beta\|^2 \quad \text{subject to} \quad y_i(x_i^T \beta + \beta_0) \geq 1 \; \forall i
$$

- The $\frac{1}{2}$ is for mathematical convenience (cancels the 2 from derivative).
- This is a **Quadratic Program** (convex objective, linear constraints). Solved in Python via CVXOPT.

---

## 4. Primal vs. Dual: Lagrangian Duality

### Key Concepts
- **Primal Problem ($p^*$):** Minimize the objective directly ("View from the Hill").
- **Dual Problem ($d^*$):** Maximize the lower bound ("View from the Valley").
- **Weak Duality:** $d^* \leq p^*$ always holds. The gap $p^* - d^*$ is the **Duality Gap**.
- **Strong Duality:** $d^* = p^*$ (no gap). Holds for convex problems satisfying Slater's condition.
- For SVM, strong duality holds — the dual solution equals the primal solution.

### The Lagrangian

$$
L_P = \frac{1}{2}\|\beta\|^2 - \sum_i \alpha_i [y_i(x_i^T \beta + \beta_0) - 1]
$$

- $\alpha_i \geq 0$ are the **Lagrange multipliers** (one per training point).

### Dual Problem

$$
\max_{\alpha} \; \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle \quad \text{subject to} \quad \alpha_i \geq 0, \; \sum_i \alpha_i y_i = 0
$$

- The data $x_i$ only appears as a **dot product** $\langle x_i, x_j \rangle$ — this is the key to the kernel trick.

---

## 5. Property 1 — Informational Sparsity

### Key Concepts
- An SVM model is **completely defined by a tiny fraction of the training data**.
- **Support Vectors:** The "difficult" points on the margin edges. They dictate the boundary.
- **Safe Points:** The "easy" cases far from the boundary. Once trained, they carry **zero** information.
- You could delete ~90% of points (non-support vectors) and the decision boundary would not move.

### The Mathematical Reason: KKT Complementary Slackness
For every data point $i$, the optimization guarantees:

$$
\alpha_i \cdot [y_i(x_i^T \beta + \beta_0) - 1] = 0
$$

- The term in brackets is the "distance beyond the margin."
- If a point is **safe** (distance from boundary > margin, so bracket > 0), then **$\alpha_i$ must be exactly zero**.
- The optimization automatically zeroes out the weights of safe points — they are erased from the equation.

### Result
- Safe points: $\alpha_i = 0$ (contribute nothing).
- Support vectors: $\alpha_i > 0$ (on the margin, bracket = 0).
- Model built from $N=200$ points is identical to model built from $N=3$ support vectors.

---

## 6. Property 2 — The Kernel Trick (Infinite Magic)

### Key Concepts
- If data cannot be separated by a straight line in 2D, map it to 3D (or 100D, or $\infty$D).
- The **Kernel Trick:** We can calculate geometric relationships in **infinite dimensions** without actually mapping the data there.

### Why It Works
- In the Dual Problem, data coordinates $x$ only appear as dot products $\langle x_i, x_j \rangle$.
- We can replace the literal dot product $\langle x_i, x_j \rangle$ with a **Kernel similarity function** $K(x_i, x_j)$.

### Key Kernels
- **Linear Kernel:** $K(x, x') = \langle x, x' \rangle$ (standard dot product, no mapping)
- **Polynomial Kernel:** $K(x, x') = (\langle x, x' \rangle + c)^d$
- **RBF / Gaussian Kernel:** $K(x, x') = \exp(-\gamma \|x - x'\|^2)$
  - The RBF kernel **mathematically represents a dot product in an infinite-dimensional space**.
  - We gain infinite representational power for the computational cost of a simple exponent.

### Dual with Kernel

$$
\max_{\alpha} \; \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i, x_j)
$$

Just replace $\langle x_i, x_j \rangle$ with $K(x_i, x_j)$.

---

## 7. Appendix: Lagrangian Duality — Worked Example

### The Primal Problem

$$
\min_x \; f(x) = x^2 \quad \text{s.t.} \quad g(x) = x - 2 \geq 0
$$

- Solution: $x^* = 2$, $p^* = f(2) = 4$

### The Lagrangian

$$
L(x, \lambda) = x^2 - \lambda(x - 2) = x^2 - \lambda x + 2\lambda
$$

- $\lambda \geq 0$ is the Lagrange multiplier.
- Dual function: $g(\lambda) = \inf_x L(x, \lambda)$
- Setting $\partial L/\partial x = 0$: $2x - \lambda = 0 \Rightarrow x = \lambda/2$
- Substituting: $g(\lambda) = (\lambda/2)^2 - \lambda(\lambda/2) + 2\lambda = 2\lambda - \lambda^2/4$

### The Dual Problem

$$
\max_{\lambda \geq 0} \; g(\lambda) = 2\lambda - \frac{\lambda^2}{4}
$$

- Setting $g'(\lambda) = 0$: $2 - \lambda/2 = 0 \Rightarrow \lambda^* = 4$
- $d^* = g(4) = 2(4) - 16/4 = 4$

### Strong Duality
- $p^* = 4 = d^*$ → **Duality Gap = 0**, strong duality holds (consistent with Slater's condition for convex problems).

---

## 8. Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| Decision function | $\hat{y} = \text{sign}(\beta_0 + x^T \beta)$ |
| Point-to-plane distance | $d = (x_i^T \beta + \beta_0) / \|\beta\|$ |
| Margin width (canonical) | $C = 1/\|\beta\|$ |
| Primal SVM | $\min \frac{1}{2}\|\beta\|^2$ s.t. $y_i(x_i^T \beta + \beta_0) \geq 1$ |
| Lagrangian | $L_P = \frac{1}{2}\|\beta\|^2 - \sum_i \alpha_i[y_i(x_i^T \beta + \beta_0) - 1]$ |
| Dual SVM | $\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i, x_j \rangle$ |
| KKT complementary slackness | $\alpha_i[y_i(x_i^T \beta + \beta_0) - 1] = 0$ |
| RBF Kernel | $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ |
