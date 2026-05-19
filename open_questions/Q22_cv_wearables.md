# Q22 — Cross-Validation Design for Wearables
> Appeared: 2022 Q22, 2024 Q22, 2025 Q22 (same dataset every year — memorize this cold)

## The Dataset

- 16 subjects × 3 activity conditions × 4 seasons = **192 total observations**
- Each subject contributes exactly 12 observations (3 × 4)
- Task: predict physical activity from wearable biosignals
- Structure: repeated-measures (multiple observations per person)

---

## Part a) Personalized Model — Test on Known Individual

**Goal**: Estimate EPE for a model that will predict future sessions for the **same person** it was trained on.

### CV Design: Leave-One-Season-Out (within subject)

For a given subject $i$:
- Use only their 12 observations
- 4-fold CV: each fold = 1 season as test, 3 seasons (9 obs) as train
- Report average error across 4 folds

### What EPE measures here
$\text{EPE}_\text{personal} = E[\text{loss}(y_\text{new season}, \hat{f}_i(x_\text{new season}))]$

The expectation is over **new seasons for person $i$**, given a model trained on their past seasons. This is intra-individual prediction.

### Why this design is valid
- The test set (held-out season) is from the same individual but a genuinely unseen time period
- Respects temporal structure: model trained on past, tested on future
- No leakage: held-out season's data never appears in training

### Limitations
- Only 9 training observations per fold → high variance estimates
- Cannot be used for new, unseen patients

---

## Part b) Generalized Model — Test on Unseen Individual

**Goal**: Estimate EPE for a model deployed on a **brand-new patient** never seen during training.

### CV Design: Leave-One-Individual-Out (LOIO-CV)

- 16 folds total
- Fold $i$: train on subjects $\{1,\ldots,16\}\setminus\{i\}$ (15 subjects × 12 = 180 obs), test on subject $i$ (12 obs)
- Report average error across 16 folds

### What EPE measures here
$\text{EPE}_\text{general} = E[\text{loss}(y_\text{new person}, \hat{f}(x_\text{new person}))]$

The expectation is over both new observations AND new individuals. This is inter-individual generalization.

### Why NOT standard random CV?
If you randomly split 192 observations without respecting subject identity:
- Some observations from person $i$ land in training, others in test
- The model learns person $i$'s individual physiological fingerprint during training
- Test performance on person $i$'s held-out obs is **artificially inflated**
- This is **data leakage** — the IID assumption fails because observations from the same person share latent individual-specific structure

Formally: $x_{i,1}, \ldots, x_{i,12}$ are NOT identically distributed to observations from a new subject drawn from the population.

---

## Part c) Trade-off Table

| Property | Personalized (LOSO) | Generalized (LOIO) |
|----------|--------------------|--------------------|
| Training set size | 9 obs | 180 obs |
| Test set size | 3 obs (1 season) | 12 obs |
| Number of folds | 4 | 16 |
| Captures | Intra-individual variation | Inter-individual variation |
| EPE estimates | Performance for known person | Performance for new person |
| Typical EPE | Lower (knows the person) | Higher (unknown person) |
| Clinical use case | Monitoring existing patient | Screening new patient |

---

## Part d) Hyperparameter Tuning (Advanced)

If a hyperparameter (e.g., regularization $\lambda$) must be tuned:

**Correct approach** (nested CV):
- Outer loop: LOIO for unbiased generalization estimate
- Inner loop (within each outer training fold): e.g., LOIO on the 15 training subjects to select $\lambda$
- Never use test-fold data to select $\lambda$ — this leaks the test label distribution

**Wrong approach**: tune $\lambda$ on full dataset first, then do LOIO. The selected $\lambda$ has "seen" all individuals → optimistic bias.

---

## Full Written Answer (Exam-Ready)

*"For a personalized model, we restrict training and evaluation to a single individual's data. Using leave-one-season-out cross-validation within that person's 12 observations, we train on 9 observations (3 seasons) and test on the held-out 4th season, repeating for all 4 seasons. This design respects the temporal structure of the data and provides an unbiased estimate of how well the model predicts future sessions for a known individual.*

*For a generalized model, we apply leave-one-individual-out cross-validation across all 16 subjects. In each of the 16 folds, one complete subject (all 12 observations) is held out as the test set while the model trains on the remaining 15 subjects (180 observations). This ensures the test individual is entirely unseen during training, directly simulating deployment on a new patient.*

*The critical distinction is the source of variation being estimated: personalized CV measures intra-individual prediction error, while generalized CV measures inter-individual generalization. Standard random splitting would constitute data leakage because observations from the same individual share physiological structure — they violate the IID assumption. Including any of a test individual's observations in training would allow the model to learn their personal physiological baseline, producing over-optimistic generalization estimates.*

*For clinical deployment where the model must serve new patients, the generalized CV estimate is the appropriate performance metric."*

---

## Additional Possible Exam Questions

**Q: Why is the IID assumption violated in this dataset?**
Each individual has a unique physiological baseline (resting heart rate, respiration patterns, activity signature). Observations from the same person are correlated across seasons and conditions — they share individual-level random effects. Treating them as IID would ignore this grouping structure.

**Q: What happens if you do 4-fold random CV instead of LOIO?**
With random 4-fold CV on 192 observations: each fold's test set contains observations from all 16 subjects, some of whose other observations are in the training set. The model "sees" every individual during training → generalizes to known individuals, not new ones. EPE will be optimistically biased compared to true new-patient performance.

**Q: Can you use both personalized and generalized approaches together?**
Yes — train a generalized base model (LOIO), then fine-tune with a few calibration sessions from the new individual (transfer learning / personalized adaptation). The generalized model gives good starting parameters; the fine-tuning adapts to the individual. This is the clinical gold standard: it requires fewer calibration sessions than training from scratch.

**Q: What if some subjects have missing seasons?**
LOIO remains valid — simply exclude the missing observations. Personalized LOSO requires at least 2 seasons per subject. With missing data, report the number of complete subjects used.

**Q: How does sample size affect the CV estimate reliability?**
- Personalized (4 folds, 9 training obs): high variance due to tiny training set. Estimate of personalized EPE has wide confidence intervals.
- Generalized (16 folds, 180 training obs): more stable. Each test fold has 12 observations, giving reasonable error estimates. However, only 16 independent test results → CI on mean EPE still moderate width.
