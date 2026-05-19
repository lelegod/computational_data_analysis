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

## Feature Extraction

A strong answer can mention:

- MFCC extraction first
- then PCA for dimension reduction if needed

If the question asks “how many unique speakers?” then it becomes an unsupervised variant:
- PCA or MFCC embeddings
- then GMM + BIC

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
