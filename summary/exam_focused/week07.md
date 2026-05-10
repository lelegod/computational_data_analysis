# Week 7 — Support Vector Machines (Exam Focus)

## Must-Know Facts

- SVM finds the hyperplane that **maximizes the margin** (distance to nearest points of each class).
- Classes are labeled **+1 and −1** (not 0 and 1).
- $\beta$ is **orthogonal (perpendicular)** to the hyperplane.
- The margin width is $C = 1/\|\beta\|$ — maximizing margin = minimizing $\|\beta\|$.
- The **primal SVM** minimizes $\frac{1}{2}\|\beta\|^2$ subject to $y_i(x_i^T\beta + \beta_0) \geq 1$.
- The constraint $y_i(x_i^T\beta + \beta_0) \geq 1$ ensures each point is at least 1 canonical unit from the boundary.
- SVM is solved by **Quadratic Programming** (convex objective, linear constraints).
- In the **dual problem**, data only appears as dot products $\langle x_i, x_j \rangle$.
- **Support Vectors** are the points ON the margin ($|x_i^T\beta + \beta_0| = 1$).
- **Safe points** (far from margin) have $\alpha_i = 0$ and contribute nothing to the model.
- You can delete ~90% of non-support vectors and the boundary does not move.
- **KKT complementary slackness:** $\alpha_i[y_i(x_i^T\beta + \beta_0) - 1] = 0$ for every point $i$.
- If a point is safe (beyond the margin), its $\alpha$ **must** be exactly zero — guaranteed by KKT.
- The **RBF kernel** $K(x,x') = \exp(-\gamma\|x-x'\|^2)$ represents a dot product in **infinite-dimensional** space.
- The kernel trick lets us work in infinite dimensions at the cost of computing a simple exponential.
- **Weak Duality:** $d^* \leq p^*$ always. **Strong Duality:** $d^* = p^*$ (holds for SVM via Slater's condition).
- There is **no probabilistic model** in SVM (unlike LDA or logistic regression).
- SVM = Convex Optimization + Implicit Feature Mapping.
- The $\frac{1}{2}$ in $\frac{1}{2}\|\beta\|^2$ is for convenience: it cancels the factor of 2 in the derivative.
- The canonical hyperplane is chosen so that for support vectors: $|x_i^T\beta + \beta_0| = 1$.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|------------|-------------|
| $\hat{y} = \text{sign}(\beta_0 + x^T \beta)$ | Decision function | Classifying new points |
| $d = (x_i^T \beta + \beta_0) / \|\beta\|$ | Signed distance to hyperplane | Deriving margin |
| $C = 1/\|\beta\|$ | Margin width (canonical scaling) | Understanding what SVM maximizes |
| $\min \frac{1}{2}\|\beta\|^2$ s.t. $y_i(x_i^T\beta+\beta_0)\geq 1$ | Primal SVM (OSH) | Setting up the optimization |
| $L_P = \frac{1}{2}\|\beta\|^2 - \sum_i \alpha_i[y_i(x_i^T\beta+\beta_0)-1]$ | Lagrangian (primal) | Deriving dual |
| $\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle x_i,x_j \rangle$ | Dual SVM | Kernel trick entry point |
| $\alpha_i[y_i(x_i^T\beta+\beta_0)-1] = 0$ | KKT complementary slackness | Explaining sparsity |
| $K(x,x') = \exp(-\gamma\|x-x'\|^2)$ | RBF/Gaussian kernel | Non-linear SVM |
| $\hat{n} = \beta/\|\beta\|$ | Unit normal to hyperplane | Distance derivation |

---

## Common Traps (Wrong Answers in Exams)

- **❌ SVM uses a probabilistic model** → ✓ SVM has NO probabilistic model; it is purely geometric.
- **❌ All training points define the SVM boundary** → ✓ Only the support vectors (a tiny fraction) define the boundary.
- **❌ Deleting non-support vectors changes the boundary** → ✓ The boundary is identical with or without non-support vectors.
- **❌ $\alpha_i > 0$ for all training points** → ✓ $\alpha_i = 0$ for safe points (non-support vectors); only $\alpha_i > 0$ for support vectors.
- **❌ The kernel trick maps data explicitly to high dimensions** → ✓ The mapping is implicit; we only compute the kernel function $K(x_i,x_j)$, never the mapped coordinates.
- **❌ The RBF kernel maps to a finite-dimensional space** → ✓ The RBF kernel corresponds to an infinite-dimensional feature space.
- **❌ Weak duality means $d^* = p^*$** → ✓ Weak duality means $d^* \leq p^*$; strong duality means $d^* = p^*$.
- **❌ $\beta$ is parallel to the hyperplane** → ✓ $\beta$ is orthogonal (perpendicular) to the hyperplane.
- **❌ Maximizing the margin means maximizing $\|\beta\|$** → ✓ $C = 1/\|\beta\|$, so maximizing margin means MINIMIZING $\|\beta\|$.
- **❌ The constraint is $y_i(x_i^T\beta+\beta_0) \geq 0$** → ✓ The canonical constraint is $\geq 1$ (not zero).
- **❌ Labels are 0 and 1 in SVM** → ✓ Labels are **−1 and +1** in SVM math.

---

## Quick Decision Rules

- If a point is a **support vector** → it lies exactly on the margin, $\alpha_i > 0$, bracket = 0.
- If a point is a **safe point** (far from boundary) → $\alpha_i = 0$, bracket > 0.
- If data is **not linearly separable** → use the kernel trick (RBF is most common).
- If you see dot products $\langle x_i,x_j \rangle$ in the dual → replace with $K(x_i,x_j)$ to kernelize.
- If primal has $p$ features → $p+1$ parameters ($\beta_0, \beta_1,\ldots,\beta_p$); dual has $N$ parameters (one $\alpha_i$ per observation).
- If data set has many more features than observations ($p \gg n$) → dual is more efficient.
- If problem is **convex + Slater's condition holds** → strong duality holds → solve dual instead of primal.
