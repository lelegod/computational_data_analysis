# Q22 — EEG / Brain Imaging Subject-Level CV
> Possible unseen Q22 variant. High-yield because it is structurally identical to wearables, but with an extra trial/time-series leakage trap.

---

## Typical Dataset

A plausible exam dataset:

- 30 subjects
- 40 trials per subject
- EEG features from 64 electrodes
- target: mental state, cognitive load, stress, or disease status

This is a repeated-measures dataset:
- many trials per subject
- strong subject-specific baseline structure
- possible temporal dependence within trials

---

## Why Random CV Fails

Trials from the same subject share:

- resting spectral baseline
- electrode-specific noise profile
- subject-specific anatomy / signal amplitude

If some trials from subject $i$ are in training and others from the same subject are in test, the model partly learns that subject’s neural fingerprint.

So random CV gives an optimistically biased estimate for new-subject deployment.

---

## Generalized Model — Predict for a New Subject

### Correct design

Use **Leave-One-Subject-Out CV**:

```text
Fold i:
Train on subjects {1,...,N} \ {i}
Test  on subject i
```

### What EPE measures

$$
\text{EPE}_\text{gen}
=
E_{i_\text{new}}\left[E_{x,y\mid i_\text{new}}[\mathcal{L}(y,\hat f(x))]\right]
$$

This is prediction error for a truly unseen subject.

---

## Personalized Model — Predict for the Same Subject

If the question asks about a subject-specific model:

- use **Leave-One-Trial-Out** or **Leave-One-Block-Out**
- if trials are ordered in time, prefer **forward holdout** by trial block

This estimates within-subject future prediction error.

---

## EEG-Specific Leakage Trap

If each trial is broken into many short windows, there is an additional risk:

- adjacent windows from the same trial are strongly autocorrelated

So the split unit should usually be the **trial**, not individual windows.

Wrong:
- random split over all EEG windows

Correct:
- hold out complete trials
- and for generalized deployment, hold out complete subjects

---

## Feature Extraction

A very natural EEG answer is:

- PCA for dimension reduction
- or ICA if the question emphasizes source separation / artifact removal

ICA is especially well-motivated in EEG because EEG signals are mixtures of latent neural and artifact sources.

But:
- feature extraction must be fit inside each training fold only
- otherwise subject information leaks through the projection

---

## Hyperparameter Tuning

Use nested CV:

- outer loop: Leave-One-Subject-Out
- inner loop: subject-grouped CV on training subjects only

This is required if:
- classifier tuning is needed
- regularization is tuned
- component count is selected

---

## Performance Metrics

If the target is classification:
- balanced accuracy
- AUC
- sensitivity / specificity

If the target is continuous:
- RMSE
- MAE

---

## Full Exam-Style Answer

*"This is a grouped repeated-measures dataset, because each EEG subject contributes many correlated trials. Therefore the IID assumption fails: observations from the same subject share neural baseline structure and noise characteristics. Random cross-validation would place some trials from the same subject in both training and test, causing data leakage and overly optimistic performance estimates."*

*"If the goal is prediction for a new subject, I would use leave-one-subject-out cross-validation, holding out one complete subject in each fold and training on the remaining subjects. If each trial is split into shorter windows, I would also ensure that complete trials remain together, since nearby windows are temporally correlated. If the goal is prediction for a known subject, I would instead use leave-one-trial-out or blockwise temporal validation within that subject."*

*"If dimensionality reduction such as PCA or ICA is used, it must be fit inside each training fold only. Hyperparameter tuning must be nested inside the outer subject-level CV loop."*

---

## Additional Possible Exam Questions

**Q: Why is EEG especially vulnerable to leakage?**
Because there can be both subject-level correlation and within-trial temporal correlation.

**Q: Why might ICA be preferable to PCA here?**
Because EEG is naturally modeled as a mixture of latent independent sources, making ICA scientifically meaningful.

**Q: What is the correct split unit if trials are windowed?**
The trial, and for new-subject deployment the subject.
