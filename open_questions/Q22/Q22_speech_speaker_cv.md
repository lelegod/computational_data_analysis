# Q22 — Speech / Speaker Dataset CV Design
> Possible unseen Q22 variant. Same grouped-CV logic as wearables, but the grouping unit is the speaker.

---

## Typical Dataset

A plausible exam dataset:

- 50 speakers
- 20 utterances per speaker
- acoustic features such as MFCCs
- target: word, emotion, phoneme, or disease state from speech

Repeated measurements come from the same speaker, so observations are not IID.

---

## Why Random CV Fails

Utterances from the same speaker share:

- accent
- pitch range
- timbre
- microphone conditions

If the same speaker appears in both train and test, the model can exploit speaker identity rather than the intended target.

That makes performance look much better than it will be on a new speaker.

---

## Generalized Model — Predict for a New Speaker

### Correct design

Use **Leave-One-Speaker-Out CV**:

```text
Fold i:
Train on all speakers except i
Test  on speaker i
```

### What EPE measures

Prediction error for a completely unseen speaker.

---

## Personalized / Speaker-Dependent Model

If the exam asks about a model for a known speaker:

- use **Leave-One-Utterance-Out**
- or hold out future utterances / sessions if there is temporal structure

This measures within-speaker generalization.

---

## Additional Leakage Trap

If each utterance is split into multiple frames or short windows:

- adjacent windows from the same utterance are correlated

So the split unit should be the **utterance**, not individual frames.

For new-speaker deployment:
- keep complete utterances together
- and hold out complete speakers

---

## Task Framing, Features, and Model Choice

**Task:** Supervised classification of word, emotion, or disease state from acoustic features. If asked “how many unique speakers?” — unsupervised variant (GMM + BIC). State the task explicitly.

**Feature extraction:**
- Extract MFCCs (Mel-Frequency Cepstral Coefficients) — standard acoustic features
- Optionally reduce with PCA inside each training fold if dimension is high
- Each utterance → fixed-length feature vector (summary statistics of MFCC trajectories: mean, variance, delta)

**Classification model:**

| Method | Why suitable |
|--------|-------------|
| **Regularized Logistic Regression** | Handles correlated MFCC features; L1 selects discriminative coefficients |
| **LDA** | Fast, interpretable; works well when MFCC features approximately Gaussian per class |
| **SVM (RBF kernel)** | Strong acoustic classification baseline; handles non-linear boundaries |
| **Random Forest** | Non-linear; robust to irrelevant features |

**Unsupervised variant (“how many unique speakers?”):** Extract MFCC mean vectors per utterance → PCA → GMM + BIC to estimate $K$ speakers. Same pipeline as face clustering.

**Critical:** Feature standardization and any PCA projection must be computed from training speakers only — applying training-set statistics to the held-out speaker is correct; fitting them on the combined data is leakage.

---

## Hyperparameter Tuning

Use nested CV:

- outer loop: Leave-One-Speaker-Out
- inner loop: grouped CV on the training speakers only

Any feature selection or normalization should also happen inside the training fold only.

---

## Full Exam-Style Answer

*"This dataset contains repeated utterances from the same speakers, so the IID assumption fails. Utterances from one speaker share voice characteristics such as pitch, accent, and timbre. Therefore random cross-validation would leak speaker identity from training into test and would overestimate performance for deployment on new speakers."*

*"If the goal is speaker-independent prediction, I would use leave-one-speaker-out cross-validation, where all utterances from one complete speaker are held out in each fold. If each utterance is split into many frames, I would also keep all frames from the same utterance together, since adjacent frames are strongly correlated. If the goal is prediction for a known speaker, I would instead validate within that speaker using leave-one-utterance-out or session-based holdout."*

---

## Additional Possible Exam Questions

**Q: What is the grouping variable?**
The speaker.

**Q: Why is frame-level random splitting invalid?**
Because frames from the same utterance are highly correlated.

**Q: What is the right design for unseen-speaker deployment?**
Leave-One-Speaker-Out CV.
