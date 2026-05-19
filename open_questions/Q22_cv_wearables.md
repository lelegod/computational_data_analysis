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

---

## Extended Question Bank

**Q: Could you use Leave-One-Activity-Out (LOAO) for the generalized model? Why or why not?**
No. LOAO holds out one activity type (e.g., all "running" observations) across all subjects. But data from the same person still appears in both training and test — the model learns each person's physiological baseline from other activities. The individual-level correlation is not broken. The correct grouping unit is the **individual**, not the activity. LOAO would still produce an optimistically biased EPE for new-patient generalization.

---

**Q: A student applies 5-fold CV by randomly assigning all 192 observations to folds. The test error is 12%. Your LOIO estimate is 21%. Why do the estimates differ, and which is correct?**

The 5-fold random CV estimate (12%) is **optimistically biased** because:
- Each fold contains observations from all 16 subjects
- The model has "seen" partial data from every test subject during training
- It learns individual physiological baselines → evaluates nearly-memorised individuals, not new ones

The LOIO estimate (21%) is **correct** for the stated goal (predicting new patients). The gap of 9 percentage points represents the cost of individual-level generalization — how much harder it is to predict a completely unseen person vs. a person the model has partially seen.

For clinical deployment, use the LOIO estimate of 21% as the honest performance expectation.

---

**Q: Derive the EPE decomposition for the generalized model. What does each term represent?**

$$\text{EPE}_\text{gen} = E_{i_\text{new}}\left[E_{x,y \mid i_\text{new}}\left[\mathcal{L}(y, \hat{f}(x))\right]\right]$$

Breaking down the squared-error loss:
$$= \underbrace{\left(f(x) - E[\hat{f}(x)]\right)^2}_{\text{Bias}^2} + \underbrace{E\left[(\hat{f}(x) - E[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Irreducible noise}} + \underbrace{E_{i_\text{new}}\left[\text{Var}(f(x) \mid i_\text{new})\right]}_{\text{Between-individual variance}}$$

The last term is specific to the generalized model: it captures how much the true function $f(x)$ varies between different individuals. The personalized EPE does not have this term because $i_\text{new}$ is fixed. This between-individual variance is why $\text{EPE}_\text{gen} > \text{EPE}_\text{pers}$.

---

**Q: The examiner asks: "Is LOIO-CV biased? If so, in which direction?"**

LOIO is **nearly unbiased** (slightly pessimistically biased). Reasoning:
- In each fold, the model trains on $15/16 \approx 94\%$ of the full data
- The true EPE corresponds to a model trained on 100% of the data
- Training on slightly less data → slightly worse model → EPE is slightly overestimated (pessimistic bias)

This pessimistic bias is small and decreases as $N$ (number of subjects) increases. It is much preferable to the large optimistic bias of random CV. For the exam: state "nearly unbiased, slight pessimistic bias from reduced training set size per fold."

---

**Q: What if the 2025 exam adds a Part c) asking you to compare EPE estimates using a boxplot across folds. What would you look for?**

In the LOIO boxplot (16 values, one per held-out individual):
- **Width of the box** (IQR): reflects heterogeneity across individuals — a wide box means some individuals are much harder to predict than others (large between-individual variance)
- **Median**: the robust estimate of EPE, less affected by outlier folds than the mean
- **Outlier folds**: specific individuals who are systematically harder to predict — investigate these individuals (unusual physiology? missing seasons? high activity variability?)

For the personalized LOSO boxplot (4 values per individual):
- Very few folds → hard to draw reliable inference
- Report range rather than IQR; acknowledge high uncertainty

---

**Q: What does "IID" mean and where exactly does this dataset violate it?**

IID = independently and identically distributed. For a sample $\{(x_i, y_i)\}$ to be IID:
- **Independent:** $P(x_i, y_i \mid x_j, y_j) = P(x_i, y_i)$ for all $i \neq j$
- **Identically distributed:** All observations are drawn from the same marginal distribution $P(x, y)$

This dataset violates independence in two ways:
1. **Within-individual:** $(x_{i,1}, y_{i,1}), \ldots, (x_{i,12}, y_{i,12})$ share individual $i$'s physiological parameters — they are correlated, not independent
2. **Within-season:** Observations from the same season across individuals are more similar than across seasons — seasonal confounding

It may also violate identical distribution:
- Individual-specific means: $E[x \mid \text{individual } i] \neq E[x \mid \text{individual } j]$ — the marginal distribution of $x$ differs by person

Standard CV assumes all observations are exchangeable (can be freely swapped between train and test). This is a direct consequence of IID. When IID fails, exchangeability fails, and standard CV is invalid.

---

**Q: Nested CV — what goes wrong if you skip the inner loop?**

If you tune $\lambda$ on the full 192 observations (without CV) and then run LOIO as the outer loop:

The selected $\lambda^*$ was chosen by looking at the stress labels and biosignals of all 16 subjects, including the future test subjects. This means:
- $\lambda^*$ is implicitly optimised for the test subjects
- The outer CV error is measuring the performance of a model with a hyperparameter selected using test data
- EPE estimate is **optimistically biased** — the model is unrealistically well-tuned

Formally: the test subject's data has leaked into the hyperparameter selection step, even though it was never directly in the training set for the outer fold. This is a subtle but real form of data leakage.

Correct approach: run LOIO on the outer loop, and for each outer fold, rerun an inner LOIO on the 15 training subjects to select a potentially different $\lambda^*_i$ for that fold.

---

**Q: 1-SE rule — when and how to apply it to this dataset?**

The 1-SE rule: choose the simplest model (largest $\lambda$, fewest features, smallest tree) whose CV error is within 1 standard error of the minimum CV error.

For the LOIO outer loop with 16 folds, the SE of the mean EPE is:
$$\text{SE} = \frac{\hat{\sigma}_{\text{fold}}}{\sqrt{16}}$$

where $\hat{\sigma}_{\text{fold}}$ is the SD of the 16 per-fold errors. Accept any $\lambda$ whose mean error $\leq \text{EPE}_{\min} + \text{SE}$.

**Why use it here:** With only 16 outer folds, the variance of the EPE estimate is moderate. The "optimal" $\lambda$ found by LOIO may just be one of many $\lambda$ values with nearly equal performance. The 1-SE rule selects a simpler, more interpretable model that is statistically indistinguishable from the minimum-error model.
