# Q21-AU — PCA vs Sparse PCA vs NMF
> Weeks 8/11. Could ask: compare three low-rank representation methods in terms of variance, sparsity, and interpretability.

---

## The Shared Goal

All three methods seek a lower-dimensional representation of the data, but they optimize different structural goals:

- **PCA**: variance maximization
- **Sparse PCA**: variance plus sparse loadings
- **NMF**: additive parts under non-negativity

So this is really a question about what kind of interpretability or structure you want from latent factors.

---

## PCA

PCA finds orthogonal directions of maximum variance:
$$
\max_v \operatorname{Var}(Xv)
\quad \text{s.t. } \|v\|=1
$$

### Main characteristics

- orthogonal components
- dense loadings
- unique up to sign
- best for compression and variance explanation

The downside is that each component may involve many variables, which can hurt interpretability.

---

## Sparse PCA

Sparse PCA modifies PCA by encouraging many loading entries to be zero.

### Main characteristics

- still tries to explain variance
- adds sparsity to the loading vectors
- components become easier to interpret

So Sparse PCA is a compromise between variance explanation and variable-level interpretability.

---

## NMF

NMF factorizes:
$$
X \approx WH, \quad W,H \ge 0
$$

### Main characteristics

- additive parts
- no cancellation
- often highly interpretable when data are non-negative

Unlike PCA and Sparse PCA, NMF is not based on orthogonal variance directions.

---

## Core Comparison

### PCA

- best pure variance explanation
- dense global components
- lower interpretability at variable level

### Sparse PCA

- retains PCA’s variance logic
- adds sparse loadings
- easier to name components

### NMF

- gives parts-based decomposition
- requires non-negative structure
- interpretability comes from additivity rather than orthogonality

---

## Comparison Table

| Property | PCA | Sparse PCA | NMF |
|----------|-----|------------|-----|
| Main target | Variance | Variance + sparse loadings | Additive parts |
| Orthogonal? | Yes | Usually relaxed / not strict in same sense | No |
| Sparse loadings? | No | Yes | Often indirectly, but via NN structure |
| Signs allowed? | Yes | Yes | No |
| Best for | Compression | Interpretable variance factors | Parts-based decomposition |

---

## Interpretability Differences

This is the heart of the compare question.

- **PCA** components are often hard to interpret because many variables contribute with both positive and negative signs
- **Sparse PCA** makes interpretation easier by selecting a smaller subset of active variables
- **NMF** makes interpretation easier through additive non-negative parts

So Sparse PCA and NMF both improve interpretability, but in fundamentally different ways.

---

## When to Use Which

**Use PCA when**:
- compression is the main goal
- orthogonal latent directions are desirable

**Use Sparse PCA when**:
- you still want variance-based components
- you need variable-level interpretability

**Use NMF when**:
- data are non-negative
- additive parts are scientifically meaningful

---

## Limitations

1. PCA can be hard to interpret.
2. Sparse PCA needs tuning of sparsity level.
3. NMF requires non-negative data.
4. Sparse PCA and NMF are more computationally involved than ordinary PCA.

---

## Additional Possible Exam Questions

**Q: Why is Sparse PCA often preferred over PCA in genomics?**
Because ordinary PCA gives dense components involving almost all variables, while Sparse PCA can identify components driven by a smaller subset of genes, which is easier to interpret biologically.

**Q: Why is NMF not simply a sparse version of PCA?**
Because NMF changes the geometry completely through non-negativity and additive reconstruction, whereas PCA is based on orthogonal variance-maximizing directions.

**Q: Which method is best if the data matrix contains negative values after centering?**
PCA or Sparse PCA. Standard NMF is not natural in that setting because it requires non-negative input.
