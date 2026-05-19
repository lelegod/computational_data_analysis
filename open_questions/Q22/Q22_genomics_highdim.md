# Q22 — High-Dimensional Genomics
> Possible unseen Q22 variant. High-yield because it combines patient-level CV with feature-selection leakage in a $p \gg n$ setting.

---

## Typical Dataset

A plausible exam dataset:

- 200 patients
- 20,000 gene-expression variables
- target: disease subtype, treatment response, survival class

This is the classic:
$$
p \gg n
$$
setting.

---

## Main Statistical Challenge

The biggest issue is not only grouped CV, but also:

- severe overfitting risk
- unstable feature selection
- leakage if genes are filtered outside CV

---

## Correct CV Design

If the goal is prediction for new patients:

- use **Leave-One-Patient-Out**
- or **stratified K-fold** if class balance matters and no repeated measures are present

If there are repeated biopsies per patient:
- then the split unit must be the patient

---

## Feature Selection Trap

This is the most likely genomics exam trap.

Wrong:

1. run a t-test on all 200 patients
2. keep top 100 genes
3. then do CV

This leaks the test patients into gene selection.

Correct:

1. split training/test fold
2. within the training fold only, select genes
3. fit model on selected genes
4. evaluate on held-out patients

So feature selection must be nested inside CV.

---

## Model Choice

Reasonable answers:

- Lasso
- Elastic Net
- Ridge

Strongest answer:
- Elastic Net, because genes are often highly correlated

---

## Hyperparameter Tuning

Use nested CV:

- outer loop: honest patient-level generalization estimate
- inner loop: tune $\lambda$ and other hyperparameters

This is essential in high-dimensional settings.

---

## Performance Metrics

If the target is imbalanced:

- balanced accuracy
- AUC

For multiclass subtype prediction:
- accuracy plus class-wise metrics

---

## Full Exam-Style Answer

*"This is a high-dimensional prediction problem with many more variables than patients, so overfitting is a major risk. If the goal is prediction for new patients, I would validate at the patient level using leave-one-patient-out or stratified K-fold cross-validation, depending on sample size and class balance. The critical rule is that gene selection must happen inside each training fold only. If genes are selected using the full dataset before CV, the test patients leak into the feature-selection step and performance becomes optimistically biased. A regularized method such as Elastic Net is especially appropriate because gene-expression predictors are high-dimensional and strongly correlated."*

---

## Additional Possible Exam Questions

**Q: Why is Elastic Net often better than Lasso here?**
Because correlated genes tend to enter in groups rather than one arbitrary representative.

**Q: What is the main leakage risk?**
Selecting genes before cross-validation.

**Q: Why is ordinary OLS inappropriate?**
Because $p \gg n$ makes the problem unstable or non-identifiable.
