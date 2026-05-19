# Q21-AK — N-mode Multiplication and Tensor Unfolding
> Week 12. Could ask: explain tensor unfolding, derive mode-$n$ multiplication, and show why these operations are the bridge to Tucker and PARAFAC updates.

---

## Why This Topic Matters

Tucker3 and PARAFAC look complicated because they are tensor models, but their algorithms are built from a small number of tensor operations:

- unfolding (matricization)
- mode-$n$ multiplication
- Kronecker and Khatri-Rao products

If you understand these operations, the tensor decompositions become matrix least-squares problems in disguise.

---

## Tensor Basics

A 3-way tensor is written:
$$
\mathcal{X} \in \mathbb{R}^{I \times J \times K}
$$

You can think of it as a data cube with three modes.

- mode 1 might be samples
- mode 2 variables
- mode 3 time or condition

The Frobenius norm is:
$$
\|\mathcal{X}\|_F = \sqrt{\sum_{i=1}^I \sum_{j=1}^J \sum_{k=1}^K x_{ijk}^2}
$$

This is just the tensor analogue of the matrix Frobenius norm.

---

## Unfolding (Matricization)

Unfolding means reshaping the tensor into a matrix by choosing one mode to stay as rows and flattening the remaining modes into columns.

For a 3-way tensor:

- mode-1 unfolding:
  $$
  X_{(1)} \in \mathbb{R}^{I \times JK}
  $$
- mode-2 unfolding:
  $$
  X_{(2)} \in \mathbb{R}^{J \times IK}
  $$
- mode-3 unfolding:
  $$
  X_{(3)} \in \mathbb{R}^{K \times IJ}
  $$

**Key idea**: unfolding does not lose information. It is just a reshape.

---

## Mode-$n$ Multiplication

To multiply a tensor by a matrix along mode $n$, write:
$$
\mathcal{Y} = \mathcal{X} \times_n M
$$

If $M \in \mathbb{R}^{L \times I_n}$, then mode $n$ changes from size $I_n$ to size $L$.

The compact matrix identity is:
$$
[\mathcal{X} \times_n M]_{(n)} = M X_{(n)}
$$

This is the most important operational formula in the tensor course material.

---

## Why Mode-$n$ Multiplication Is Useful

It lets us express tensor decompositions in matrix form.

Example:
$$
\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C
$$

If we unfold along mode 1:
$$
X_{(1)} \approx A G_{(1)} (C \otimes B)^T
$$

So a tensor model becomes a matrix factorization after unfolding.

That is exactly why ALS updates are possible.

---

## Tucker3 in This Language

Tucker3 writes:
$$
\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C
$$

where:
- $A \in \mathbb{R}^{I \times P}$
- $B \in \mathbb{R}^{J \times Q}$
- $C \in \mathbb{R}^{K \times R}$
- $\mathcal{G} \in \mathbb{R}^{P \times Q \times R}$

Mode-1 unfolding gives:
$$
X_{(1)} \approx A G_{(1)} (C \otimes B)^T
$$

This is a least-squares problem in $A$ if $B$, $C$, and $\mathcal{G}$ are fixed.

So the tensor update becomes an ordinary matrix regression step.

---

## PARAFAC in This Language

PARAFAC writes:
$$
\mathcal{X} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r
$$

Its mode-1 unfolding is:
$$
X_{(1)} \approx A(C \odot B)^T
$$

where $\odot$ is the Khatri-Rao product.

Again, this becomes a matrix least-squares problem in $A$ when $B$ and $C$ are fixed.

So unfolding is what makes ALS practical.

---

## Kronecker vs Khatri-Rao

These two products are easy to confuse.

### Kronecker product
$$
C \otimes B
$$
- uses all pairwise combinations of entries
- appears in Tucker

### Khatri-Rao product
$$
C \odot B
$$
- columnwise Kronecker product
- combines matching columns only
- appears in PARAFAC

**Memory shortcut**:
- Tucker = full core, full cross-talk, full Kronecker
- PARAFAC = matched components, matched columns, Khatri-Rao

---

## ALS Interpretation

Alternating least squares works because after fixing all but one factor, the tensor problem reduces to linear least squares in the remaining factor.

For example, in Tucker:
$$
X_{(1)} \approx A Z^T
$$
with
$$
Z = G_{(1)}(C \otimes B)^T
$$

Then:
$$
A = X_{(1)} Z^T (ZZ^T)^{-1}
$$

So the tensor algorithm is really just repeated matrix regression after the correct reshape.

---

## Why This Topic Is Exam-Relevant

This is the bridge question between:

- understanding what tensors are
- understanding how Tucker/PARAFAC are fitted
- understanding why matrix identities appear in the derivations

If the exam asks you to "show how Tucker can be solved by ALS" or "explain the role of mode-$n$ multiplication", this is the core material.

---

## Limitations / Common Confusions

1. Unfolding is only a reshape, not dimension reduction.
2. Mode-$n$ multiplication changes one mode while leaving the others intact.
3. Tucker uses Kronecker, not Khatri-Rao.
4. PARAFAC uses Khatri-Rao because components are matched across modes.
5. The notation looks abstract, but every ALS step is just least squares after unfolding.

---

## Additional Possible Exam Questions

**Q: What is the purpose of tensor unfolding?**
To convert a tensor problem into a matrix problem so that ordinary linear algebra tools can be used. Unfolding is the bridge from multiway structure to matrix least squares.

**Q: Why is mode-$n$ multiplication important?**
Because it provides the compact algebraic language for tensor decompositions. Tucker and PARAFAC are written as repeated mode products, and their fitting algorithms depend on this notation.

**Q: Why does Tucker use the Kronecker product while PARAFAC uses the Khatri-Rao product?**
Tucker has a full core tensor, so every component in one mode can interact with every component in the other modes. PARAFAC has matched rank-1 components, so only corresponding columns combine, which gives the Khatri-Rao product.

**Q: What is the main computational idea behind ALS for tensors?**
Fix all factors except one, unfold the tensor in the relevant mode, and solve an ordinary least-squares problem for that factor. Repeat cyclically until convergence.
