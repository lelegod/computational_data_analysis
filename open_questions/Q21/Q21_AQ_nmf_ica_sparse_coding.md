# Q21-AQ — NMF vs ICA vs Sparse Coding
> Week 11. Could ask: compare three unsupervised representation methods in terms of constraints, interpretability, and identifiability.

---

## The Shared Goal

All three methods try to represent data using latent components:
$$
X \approx WH
$$
or an equivalent source-separation model.

But they encode very different assumptions about what a “good” representation should look like.

---

## NMF

NMF imposes:
$$
W \ge 0, \quad H \ge 0
$$

### Main idea

- representation is additive
- no subtraction / cancellation
- components often look like parts

So NMF is ideal when the data are naturally non-negative and interpretability of additive components matters.

---

## ICA

ICA assumes:
- latent sources are statistically independent
- sources are non-Gaussian

### Main idea

Recover the latent signals whose mixtures produced the observed data.

So ICA is about **source separation**, not primarily about parts or sparsity.

---

## Sparse Coding

Sparse coding imposes an $L_1$ penalty on the coefficients:
$$
\min_{W,H} \frac{1}{2}\|X-WH\|_F^2 + \lambda \sum_i \|h_i\|_1
$$

### Main idea

- each sample uses only a few atoms
- dictionary can be overcomplete
- representation is flexible but locally sparse

So sparse coding is about sparse representation rather than strict non-negativity or independence.

---

## Core Comparison

### NMF

- additive parts
- non-negative only
- often highly interpretable

### ICA

- independent components
- signed sources allowed
- aims for identifiability of latent causes

### Sparse Coding

- sparse coefficients
- signed atoms allowed
- flexible feature-learning representation

---

## Comparison Table

| Property | NMF | ICA | Sparse Coding |
|----------|-----|-----|---------------|
| Main constraint | Non-negativity | Independence + non-Gaussianity | Sparsity |
| Signs allowed? | No | Yes | Yes |
| Main interpretation | Parts | Sources | Few active atoms |
| Overcomplete dictionary? | Usually not central | Usually constrained | Yes, often |
| Uniqueness | No | Essentially yes | No |
| Typical use | Counts, spectra, images | EEG, audio, mixtures | Image patches, feature learning |

---

## Identifiability and Uniqueness

This is one of the highest-value distinctions.

- **ICA** is essentially unique up to permutation, sign, and scale
- **NMF** is not unique because of factorization ambiguity
- **Sparse coding** is also generally not unique

So if the exam asks which method has the strongest claim to recovering true latent sources, the answer is ICA.

---

## Interpretability

- NMF is often easiest to explain visually because it produces additive parts
- ICA is interpretable when real independent latent sources exist
- Sparse coding is interpretable as a sparse dictionary, especially for localized patterns

So interpretability means different things in each method.

---

## When to Use Which

**Use NMF when**:
- data are non-negative
- parts-based interpretation matters

**Use ICA when**:
- source separation is the scientific goal
- independence is plausible
- non-Gaussianity is present

**Use sparse coding when**:
- you want a flexible overcomplete dictionary
- each sample is expected to activate only a few features

---

## Limitations

1. NMF requires non-negative data.
2. ICA fails for Gaussian sources.
3. Sparse coding needs tuning of sparsity strength $\lambda$.
4. NMF and sparse coding are non-unique and nonconvex.

---

## Additional Possible Exam Questions

**Q: Which of the three is most naturally linked to the cocktail-party problem?**
ICA, because it is designed to recover independent latent sources from observed mixtures.

**Q: Which of the three is most naturally linked to image “parts”?**
NMF, because non-negativity prevents cancellation and encourages additive component interpretation.

**Q: Why is sparse coding often connected to ICA?**
Because ICA with a super-Gaussian prior leads to sparse latent coefficients, so sparse coding can be viewed as a related representation-learning formulation.
