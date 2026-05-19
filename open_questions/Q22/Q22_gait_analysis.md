# Q22 — Gait Analysis / Motion Dataset
> Possible unseen Q22 variant. Very close to wearables, so it is one of the most realistic alternatives.

---

## Typical Dataset

A plausible exam dataset:

- 24 subjects
- multiple walking conditions
- several sessions per condition
- features: joint angles, force plate summaries, stride variability
- target: classify gait condition or predict fall risk

This is a repeated-measures subject-level dataset.

---

## Task Framing, Features, and Model Choice

**Task:** Supervised classification (walking condition, fall risk) or regression from biomechanical features. State this explicitly.

**Feature extraction:** Summarize each trial into a feature vector:
- **Joint angles:** mean, range, peak flexion/extension per joint
- **Ground reaction forces:** peak force, impulse, loading rate
- **Stride variability:** stride time CV, step length symmetry

**Classification model:**

| Method | Why suitable |
|--------|-------------|
| **LDA** | Works well with small $n$ per fold (few sessions per subject); interpretable; fast |
| **Regularized Logistic Regression** | Handles correlated biomechanical features; regularization essential with small training sets |
| **Random Forest** | Non-linear gait patterns; automatic feature importance (useful for clinical interpretation) |

For fall-risk prediction (rare event): use balanced accuracy or AUC, not raw accuracy. Tune regularization via nested CV inside the subject-level outer loop.

---

## Why Random CV Fails

Trials from the same subject share:

- gait style
- body mechanics
- walking speed baseline
- device / calibration effects

So random splitting leaks subject-specific movement patterns into test.

---

## Generalized Model — New Subject

Use **Leave-One-Subject-Out CV**:

```text
Fold i:
Train on all subjects except i
Test  on subject i
```

This estimates performance on a new patient or participant.

---

## Personalized Model — Known Subject

If the goal is subject-specific monitoring:

- use **Leave-One-Session-Out**
- or hold out one visit/session at a time

This measures future performance for a subject already seen before.

---

## Why This Mirrors Wearables

The logic is almost identical:

- same subject contributes multiple correlated observations
- generalized model must hold out whole subjects
- personalized model uses within-subject validation

So if you forget details under exam pressure, you can map this directly to the wearables template.

---

## Full Exam-Style Answer

*"This is a repeated-measures biomechanical dataset, so observations are not IID at the subject level. Trials from the same subject share gait style and body-specific movement patterns. Therefore random cross-validation would leak subject-specific structure from training into test and would overestimate performance for deployment on new individuals."*

*"If the goal is generalized prediction for a new subject, I would use leave-one-subject-out cross-validation. If the goal is personalized monitoring for a known subject, I would instead use leave-one-session-out validation within that subject. The generalized error will be larger because it includes between-subject variation that the personalized model does not need to handle."*

---

## Additional Possible Exam Questions

**Q: What is the grouping variable?**
The subject.

**Q: What is the personalized split unit?**
The session.

**Q: Why is this so similar to wearables?**
Because both are repeated-measures human-sensor datasets with subject-specific baselines.
