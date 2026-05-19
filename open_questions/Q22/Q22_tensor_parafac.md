# Q22 — Tensor / PARAFAC Component Selection
> Possible unseen Q22 variant. This is the main unsupervised multiway-data version of Q22.

---

## Typical Dataset

A plausible exam dataset:

- excitation-emission fluorescence tensor
- samples × emission wavelengths × excitation wavelengths

Question:
- how many underlying chemical components are present?

This is not a CV-design question in the usual grouped sense. It is a **tensor model-selection** question.

---

## Correct Methodology

Use **PARAFAC** rather than flattening the tensor into a matrix.

Why:

- the data are naturally multiway
- PARAFAC preserves the sample × mode-2 × mode-3 structure
- components are interpretable and essentially unique

Model:
$$
\mathcal X \approx \sum_{r=1}^R a_r \circ b_r \circ c_r
$$

---

## Choosing the Number of Components

### 1. CORCONDIA

Compute the core consistency diagnostic:
$$
\text{CORCONDIA}
=
100\left(1-\frac{\|\mathcal I-\tilde{\mathcal G}\|_F^2}{\|\mathcal I\|_F^2}\right)
$$

Interpretation:

- near 100: good PARAFAC structure
- low or negative: too many components / bad PARAFAC fit

Choose the largest $R$ before CORCONDIA drops sharply.

### 2. Split-Half FMS

Split samples into two halves:

- fit PARAFAC to each half independently
- compare corresponding factors

High factor match score means the components are reproducible.

---

## Why This Is the Right Answer

The question is not:
- “predict a label”

It is:
- “discover how many latent components explain the tensor”

So the correct methodology is unsupervised tensor decomposition with model validation.

Flattening to PCA or ordinary clustering would lose multiway structure.

---

## Full Exam-Style Answer

*"Because the data are naturally multiway, I would model them with PARAFAC rather than flattening them into a matrix. PARAFAC decomposes the tensor into a sum of rank-1 components and preserves the structure across samples, excitation wavelengths, and emission wavelengths. To choose the number of components, I would fit PARAFAC models with increasing rank $R$ and evaluate CORCONDIA, selecting the largest $R$ before the core consistency drops sharply. I would then validate that choice using split-half analysis with Factor Match Score to ensure the recovered components are reproducible across independent sample splits."*

---

## Additional Possible Exam Questions

**Q: Why not use standard PCA?**
Because PCA ignores the multiway structure by flattening the tensor.

**Q: Why both CORCONDIA and split-half FMS?**
CORCONDIA checks model form; FMS checks reproducibility.

**Q: What does a low CORCONDIA mean?**
That the chosen rank is too large or PARAFAC is not appropriate.
