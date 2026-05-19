# Q21-AY — CORCONDIA vs Split-Half FMS
> Week 12. Could ask: compare two PARAFAC validation tools, explain what each measures, and why both are useful when choosing rank.

---

## The Shared Problem

In PARAFAC, the difficult practical question is:

**How many components $R$ should we use?**

A larger $R$ always improves raw reconstruction, so reconstruction error alone is not enough.

Two important validation tools from the course are:

- **CORCONDIA**
- **Split-half FMS**

---

## CORCONDIA

Core consistency diagnostic:
$$
\text{CORCONDIA}
=
100\left(1-\frac{\|\mathcal I-\tilde{\mathcal G}\|_F^2}{\|\mathcal I\|_F^2}\right)
$$

### What it measures

How close the fitted PARAFAC structure is to the ideal super-diagonal core.

### Interpretation

- near 100: PARAFAC structure is appropriate
- low or negative: rank too large, off-diagonal interaction appears

So CORCONDIA is a **model-form diagnostic**.

---

## Split-Half FMS

Procedure:

1. split the data into two halves
2. fit PARAFAC with the same rank $R$ to each half
3. compare the resulting factors

The factor match score measures agreement between corresponding components.

### What it measures

Reproducibility / stability of the estimated factors across independent subsamples.

### Interpretation

- high FMS: components are stable
- low FMS: solution is unstable or rank is too large

So FMS is a **reproducibility diagnostic**.

---

## The Core Difference

### CORCONDIA

- checks whether the fitted structure is consistent with PARAFAC assumptions
- focuses on the core geometry

### FMS

- checks whether the components reappear across data splits
- focuses on stability

So the two tools answer different questions.

---

## Why Both Are Useful

This is the most important conceptual point.

A model can:

- fit the PARAFAC form well, but be unstable across subsets
- or be somewhat stable, but not actually have the right super-diagonal structure

So strong evidence for rank $R$ usually means:

- good CORCONDIA
- and high split-half FMS

Using both gives much better justification than using only one.

---

## Comparison Table

| Property | CORCONDIA | Split-half FMS |
|----------|-----------|----------------|
| Main purpose | Check PARAFAC form | Check reproducibility |
| Uses core tensor? | Yes | No |
| Uses data splitting? | No | Yes |
| Sensitive to rank too large? | Yes | Yes |
| Main interpretation | Structural validity | Stability |

---

## How to Use in Practice

For $R=1,2,3,\dots$:

1. fit PARAFAC
2. compute CORCONDIA
3. run split-half FMS
4. choose the largest $R$ before:
   - CORCONDIA drops sharply
   - or FMS becomes unstable / low

This is the exam-ready workflow.

---

## Relation to Tucker

CORCONDIA is specifically meaningful because PARAFAC assumes a super-diagonal core.

Tucker allows a full core tensor, so the same diagnostic does not play the same role there.

This is why CORCONDIA is a PARAFAC-specific validation concept.

---

## Limitations

1. CORCONDIA can be sensitive to noisy estimation.
2. Split-half FMS depends on the randomness of the split.
3. Neither measure alone is perfect.
4. Both are diagnostics, not absolute proofs.

---

## Additional Possible Exam Questions

**Q: What does low CORCONDIA mean?**
That the chosen rank is too large or the PARAFAC structure is not appropriate.

**Q: What does low split-half FMS mean?**
That the recovered components are not reproducible across data subsets.

**Q: Why is CORCONDIA not just a stability measure?**
Because it checks whether the estimated core behaves like the super-diagonal structure PARAFAC assumes, which is about model form rather than reproducibility.
