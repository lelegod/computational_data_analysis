# Week 7 — Support Vector Machines (SVM)

## Overview
SVMs find the optimal hyperplane that maximally separates two classes by maximizing the margin — the distance from the boundary to the nearest points of each class. There is no probabilistic model (unlike LDA or logistic regression). SVMs have two powerful properties: informational sparsity (only support vectors matter) and the kernel trick (implicit infinite-dimensional feature mapping).

---

## 1. The Decision Function

### Key Concepts
- Classes are labeled **+1** and **-1** (not 0/1 — this is important for the math).
- Linear decision function: `y_new = sign(β₀ + x_new^T β)`
- Many hyperplanes can separate two linearly separable classes; SVM finds the **optimal** one.
- SVM = Convex Optimization + Implicit Feature Mapping.

### The Two Pillars
1. **Optimal Separating Hyperplane (OSH):** Find the unique boundary maximizing distance to nearest points. Solved via Quadratic Programming. Result: Sparsity.
2. **Feature Space Transformation:** Move data from ℝ^d to a higher-dimensional space ℋ where linear separation is possible. Calculated implicitly via the Kernel Trick. Result: Non-linear boundaries.

---

## 2. The Margin and Distance to the Hyperplane

### Key Concepts
- The hyperplane is defined by: `x^T β + β₀ = 0`
- **β is orthogonal (perpendicular) to the hyperplane.**
- Unit normal vector: `n̂ = β / ‖β‖`
- The signed distance from point xᵢ to the hyperplane:

### Formulas
- **Point-to-Plane Distance**: `d = (xᵢ^T β + β₀) / ‖β‖`
  - If d > 0: point is on the positive side.
  - If d < 0: point is on the negative side.
- **SVM maximizes signed distance**: `yᵢ · (xᵢ^T β + β₀) / ‖β‖`, where yᵢ ∈ {−1, 1}

### Derivation Steps
1. Unit normal: `n̂ = β/‖β‖`
2. Vector from plane point x to data point xᵢ: `v = xᵢ − x`
3. Distance = projection of v onto n̂: `d = ⟨(xᵢ − x), n̂⟩ = (xᵢ − x)^T β / ‖β‖`
4. Since x is on the hyperplane: `x^T β = −β₀`, so `d = (xᵢ^T β + β₀) / ‖β‖`

---

## 3. The OSH Optimization Problem

### Key Concepts
- **Canonical scaling / Canonical Hyperplane:** We fix the scale so that for Support Vectors: `|xᵢ^T β + β₀| = 1`
- With canonical scaling, the margin width becomes: `C = 1/‖β‖`
- To maximize C = 1/‖β‖, we minimize ‖β‖, equivalently minimize ‖β‖².

### Primal Formulation
- **Geometric Goal**: `arg max_{β,β₀} C` subject to `yᵢ(xᵢ^T β + β₀)/‖β‖ ≥ C ∀i`
- **Computational Goal (Primal):**

```
min_{β,β₀}  (1/2)‖β‖²
subject to  yᵢ(xᵢ^T β + β₀) ≥ 1   ∀i
```

- The `1/2` is for mathematical convenience (cancels the 2 from derivative).
- This is a **Quadratic Program** (convex objective, linear constraints). Solved in Python via CVXOPT.

---

## 4. Primal vs. Dual: Lagrangian Duality

### Key Concepts
- **Primal Problem (p*):** Minimize the objective directly ("View from the Hill").
- **Dual Problem (d*):** Maximize the lower bound ("View from the Valley").
- **Weak Duality:** d* ≤ p* always holds. The gap p* − d* is the **Duality Gap**.
- **Strong Duality:** d* = p* (no gap). Holds for convex problems satisfying Slater's condition.
- For SVM, strong duality holds — the dual solution equals the primal solution.

### The Lagrangian
```
L_P = (1/2)‖β‖² − Σᵢ αᵢ[yᵢ(xᵢ^T β + β₀) − 1]
```
- αᵢ ≥ 0 are the **Lagrange multipliers** (one per training point).

### Dual Problem
```
max_α  Σᵢ αᵢ − (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ⟨xᵢ, xⱼ⟩
subject to  αᵢ ≥ 0,  Σᵢ αᵢyᵢ = 0
```
- The data xᵢ only appears as a **dot product** ⟨xᵢ, xⱼ⟩ — this is the key to the kernel trick.

---

## 5. Property 1 — Informational Sparsity

### Key Concepts
- An SVM model is **completely defined by a tiny fraction of the training data**.
- **Support Vectors:** The "difficult" points on the margin edges. They dictate the boundary.
- **Safe Points:** The "easy" cases far from the boundary. Once trained, they carry **zero** information.
- You could delete ~90% of points (non-support vectors) and the decision boundary would not move.

### The Mathematical Reason: KKT Complementary Slackness
For every data point i, the optimization guarantees:
```
αᵢ · [yᵢ(xᵢ^T β + β₀) − 1] = 0
```
- The term in brackets is the "distance beyond the margin."
- If a point is **safe** (distance from boundary > margin, so bracket > 0), then **αᵢ must be exactly zero**.
- The optimization automatically zeroes out the weights of safe points — they are erased from the equation.

### Result
- Safe points: αᵢ = 0 (contribute nothing).
- Support vectors: αᵢ > 0 (on the margin, bracket = 0).
- Model built from N=200 points is identical to model built from N=3 support vectors.

---

## 6. Property 2 — The Kernel Trick (Infinite Magic)

### Key Concepts
- If data cannot be separated by a straight line in 2D, map it to 3D (or 100D, or ∞D).
- The **Kernel Trick:** We can calculate geometric relationships in **infinite dimensions** without actually mapping the data there.

### Why It Works
- In the Dual Problem, data coordinates x only appear as dot products ⟨xᵢ, xⱼ⟩.
- We can replace the literal dot product ⟨xᵢ, xⱼ⟩ with a **Kernel similarity function K(xᵢ, xⱼ)**.

### Key Kernels
- **Linear Kernel:** `K(x, x') = ⟨x, x'⟩` (standard dot product, no mapping)
- **Polynomial Kernel:** `K(x, x') = (⟨x, x'⟩ + c)^d`
- **RBF / Gaussian Kernel:** `K(x, x') = exp(−γ‖x − x'‖²)`
  - The RBF kernel **mathematically represents a dot product in an infinite-dimensional space**.
  - We gain infinite representational power for the computational cost of a simple exponent.

### Dual with Kernel
```
max_α  Σᵢ αᵢ − (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ K(xᵢ, xⱼ)
```
Just replace ⟨xᵢ, xⱼ⟩ with K(xᵢ, xⱼ).

---

## 7. Appendix: Lagrangian Duality — Worked Example

### The Primal Problem
```
min_x   f(x) = x²
s.t.    g(x) = x − 2 ≥ 0
```
- Solution: x* = 2, p* = f(2) = 4

### The Lagrangian
```
L(x, λ) = x² − λ(x − 2) = x² − λx + 2λ
```
- λ ≥ 0 is the Lagrange multiplier.
- Dual function: `g(λ) = inf_x L(x, λ)`
- Setting ∂L/∂x = 0: `2x − λ = 0 ⟹ x = λ/2`
- Substituting: `g(λ) = (λ/2)² − λ(λ/2) + 2λ = 2λ − λ²/4`

### The Dual Problem
```
max_{λ≥0}  g(λ) = 2λ − λ²/4
```
- Setting g'(λ) = 0: `2 − λ/2 = 0 ⟹ λ* = 4`
- d* = g(4) = 2(4) − 16/4 = 4

### Strong Duality
- p* = 4 = d* → **Duality Gap = 0**, strong duality holds (consistent with Slater's condition for convex problems).

---

## 8. Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| Decision function | `ŷ = sign(β₀ + x^T β)` |
| Point-to-plane distance | `d = (xᵢ^T β + β₀) / ‖β‖` |
| Margin width (canonical) | `C = 1/‖β‖` |
| Primal SVM | `min (1/2)‖β‖²` s.t. `yᵢ(xᵢ^T β + β₀) ≥ 1` |
| Lagrangian | `L_P = (1/2)‖β‖² − Σᵢ αᵢ[yᵢ(xᵢ^T β + β₀) − 1]` |
| Dual SVM | `max_α Σαᵢ − (1/2)ΣΣ αᵢαⱼyᵢyⱼ⟨xᵢ,xⱼ⟩` |
| KKT complementary slackness | `αᵢ[yᵢ(xᵢ^T β + β₀) − 1] = 0` |
| RBF Kernel | `K(x, x') = exp(−γ‖x − x'‖²)` |
