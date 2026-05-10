# Week 7 — Support Vector Machines (Exam Focus)

## Must-Know Facts

- SVM finds the hyperplane that **maximizes the margin** (distance to nearest points of each class).
- Classes are labeled **+1 and −1** (not 0 and 1).
- β is **orthogonal (perpendicular)** to the hyperplane.
- The margin width is `C = 1/‖β‖` — maximizing margin = minimizing ‖β‖.
- The **primal SVM** minimizes `(1/2)‖β‖²` subject to `yᵢ(xᵢ^Tβ + β₀) ≥ 1`.
- The constraint `yᵢ(xᵢ^Tβ + β₀) ≥ 1` ensures each point is at least 1 canonical unit from the boundary.
- SVM is solved by **Quadratic Programming** (convex objective, linear constraints).
- In the **dual problem**, data only appears as dot products `⟨xᵢ, xⱼ⟩`.
- **Support Vectors** are the points ON the margin (|xᵢ^Tβ + β₀| = 1).
- **Safe points** (far from margin) have αᵢ = 0 and contribute nothing to the model.
- You can delete ~90% of non-support vectors and the boundary does not move.
- **KKT complementary slackness:** `αᵢ[yᵢ(xᵢ^Tβ + β₀) − 1] = 0` for every point i.
- If a point is safe (beyond the margin), its α **must** be exactly zero — guaranteed by KKT.
- The **RBF kernel** `K(x,x') = exp(−γ‖x−x'‖²)` represents a dot product in **infinite-dimensional** space.
- The kernel trick lets us work in infinite dimensions at the cost of computing a simple exponential.
- **Weak Duality:** d* ≤ p* always. **Strong Duality:** d* = p* (holds for SVM via Slater's condition).
- There is **no probabilistic model** in SVM (unlike LDA or logistic regression).
- SVM = Convex Optimization + Implicit Feature Mapping.
- The `1/2` in `(1/2)‖β‖²` is for convenience: it cancels the factor of 2 in the derivative.
- The canonical hyperplane is chosen so that for support vectors: `|xᵢ^Tβ + β₀| = 1`.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|------------|-------------|
| `ŷ = sign(β₀ + x^T β)` | Decision function | Classifying new points |
| `d = (xᵢ^T β + β₀) / ‖β‖` | Signed distance to hyperplane | Deriving margin |
| `C = 1/‖β‖` | Margin width (canonical scaling) | Understanding what SVM maximizes |
| `min (1/2)‖β‖²` s.t. `yᵢ(xᵢ^Tβ+β₀)≥1` | Primal SVM (OSH) | Setting up the optimization |
| `L_P = (1/2)‖β‖² − Σαᵢ[yᵢ(xᵢ^Tβ+β₀)−1]` | Lagrangian (primal) | Deriving dual |
| `max_α Σαᵢ − (1/2)ΣΣ αᵢαⱼyᵢyⱼ⟨xᵢ,xⱼ⟩` | Dual SVM | Kernel trick entry point |
| `αᵢ[yᵢ(xᵢ^Tβ+β₀)−1] = 0` | KKT complementary slackness | Explaining sparsity |
| `K(x,x') = exp(−γ‖x−x'‖²)` | RBF/Gaussian kernel | Non-linear SVM |
| `n̂ = β/‖β‖` | Unit normal to hyperplane | Distance derivation |

---

## Common Traps (Wrong Answers in Exams)

- **❌ SVM uses a probabilistic model** → ✓ SVM has NO probabilistic model; it is purely geometric.
- **❌ All training points define the SVM boundary** → ✓ Only the support vectors (a tiny fraction) define the boundary.
- **❌ Deleting non-support vectors changes the boundary** → ✓ The boundary is identical with or without non-support vectors.
- **❌ αᵢ > 0 for all training points** → ✓ αᵢ = 0 for safe points (non-support vectors); only αᵢ > 0 for support vectors.
- **❌ The kernel trick maps data explicitly to high dimensions** → ✓ The mapping is implicit; we only compute the kernel function K(xᵢ,xⱼ), never the mapped coordinates.
- **❌ The RBF kernel maps to a finite-dimensional space** → ✓ The RBF kernel corresponds to an infinite-dimensional feature space.
- **❌ Weak duality means d* = p*** → ✓ Weak duality means d* ≤ p*; strong duality means d* = p*.
- **❌ β is parallel to the hyperplane** → ✓ β is orthogonal (perpendicular) to the hyperplane.
- **❌ Maximizing the margin means maximizing ‖β‖** → ✓ C = 1/‖β‖, so maximizing margin means MINIMIZING ‖β‖.
- **❌ The constraint is yᵢ(xᵢ^Tβ+β₀) ≥ 0** → ✓ The canonical constraint is `≥ 1` (not zero).
- **❌ Labels are 0 and 1 in SVM** → ✓ Labels are **−1 and +1** in SVM math.

---

## Quick Decision Rules

- If a point is a **support vector** → it lies exactly on the margin, αᵢ > 0, bracket = 0.
- If a point is a **safe point** (far from boundary) → αᵢ = 0, bracket > 0.
- If data is **not linearly separable** → use the kernel trick (RBF is most common).
- If you see dot products ⟨xᵢ,xⱼ⟩ in the dual → replace with K(xᵢ,xⱼ) to kernelize.
- If primal has p features → p+1 parameters (β₀, β₁,...,βₚ); dual has N parameters (one αᵢ per observation).
- If data set has many more features than observations (p >> n) → dual is more efficient.
- If problem is **convex + Slater's condition holds** → strong duality holds → solve dual instead of primal.
