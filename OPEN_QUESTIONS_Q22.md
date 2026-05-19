# CDA 02582 — Q22 CHEAT SHEET
> Q22 = 20 points. Same wearables dataset used in 2022, 2024, 2025 — memorize cold.
> Full model answer in `open_questions/Q22_cv_wearables.md`.

---

## The Dataset (Always the Same)

- **16 subjects** × **3 activity conditions** × **4 seasons** = **192 observations**
- Each subject has exactly **12 observations** (3 × 4)
- Task: predict physical activity from wearable biosignals
- Structure: **repeated-measures** (multiple observations per person)

---

## Part a) Personalized Model — Predict for SAME Individual

**Goal**: estimate EPE for a model that predicts future sessions for a person it was already trained on.

**CV design**: **Leave-One-Season-Out** within a single subject
- Use only that subject's 12 observations
- 4 folds: hold out 1 season (3 obs) as test, train on remaining 3 seasons (9 obs)
- Repeat for all 4 seasons → average error

**What EPE measures**:
$$\text{EPE}_\text{personal} = E[\text{loss}(y_\text{new season}, \hat{f}_i(x_\text{new season}))]$$
Expectation over future seasons for the **same person $i$** → intra-individual variation.

**Why valid**: held-out season is from the same person but a genuinely unseen time period. Respects temporal structure.

**Limitation**: only 9 training observations per fold → high-variance estimate.

---

## Part b) Generalized Model — Predict for NEW Individual

**Goal**: estimate EPE for deployment on a brand-new patient never seen during training.

**CV design**: **Leave-One-Individual-Out** (LOIO-CV)
- 16 folds total
- Fold $i$: train on subjects $\{1,\ldots,16\}\setminus\{i\}$ (180 obs), test on subject $i$ (12 obs)
- Report average across all 16 folds

**What EPE measures**:
$$\text{EPE}_\text{general} = E[\text{loss}(y_\text{new person}, \hat{f}(x_\text{new person}))]$$
Expectation over new observations AND new individuals → inter-individual generalization.

**Why NOT standard random CV**:
- Random split puts some observations from person $i$ in training, others in test
- Model learns person $i$'s individual physiological baseline during training
- Performance on held-out obs from same person = artificially inflated
- This is **data leakage** — IID assumption is violated: observations from the same individual share latent individual-level structure (correlated within subject)

---

## Part c) Comparison

| Property | Personalized (LOSO) | Generalized (LOIO) |
|----------|--------------------|--------------------|
| Fold count | 4 | 16 |
| Training size | 9 obs | 180 obs |
| Test size | 3 obs | 12 obs |
| Captures | Intra-individual variation | Inter-individual variation |
| EPE estimate | For known person | For new person |
| Typical EPE | Lower | Higher |
| Clinical use | Monitor existing patient | Screen new patient |

**For clinical deployment** (new patients): use **generalized** model. Personalized model cannot be used — no training data for the new patient.

**Can you combine both?** Yes: train generalized base model, then fine-tune with calibration sessions from the new individual (transfer learning). Gold standard in clinical practice.

---

## Part d) Hyperparameter Tuning — Nested CV

If a hyperparameter (e.g., $\lambda$ in regularized model) must be selected:

**Correct (nested CV)**:
- Outer loop: LOIO to estimate generalization EPE (16 folds)
- Inner loop: within each outer training fold, LOIO on the 15 training subjects to select $\lambda$
- The outer test subject never influences $\lambda$ selection

**Wrong**: tune $\lambda$ on full dataset first, then do LOIO. The selected $\lambda$ has "seen" all 16 subjects → the outer CV estimate is optimistically biased.

---

## Full Exam Answer (Write This Cold)

*"For a personalized model, we restrict training and evaluation to a single individual's 12 observations. Using leave-one-season-out cross-validation, we train on 9 observations (3 seasons) and test on the held-out 4th season, repeating for all 4 seasons. This estimates how well the model predicts future sessions for a known individual.*

*For a generalized model, we apply leave-one-individual-out cross-validation across all 16 subjects. In each fold, one complete subject is held out as the test set while the model trains on the remaining 15 subjects. This ensures the test individual is entirely unseen during training, simulating prediction for a new patient.*

*The key distinction is the source of variation: personalized CV measures intra-individual prediction error; generalized CV measures inter-individual generalization. Standard random splitting would violate the IID assumption — observations from the same person share physiological structure, creating data leakage that produces over-optimistic generalization estimates. For clinical deployment on new patients, the generalized CV estimate is the appropriate performance metric."*

---

## Additional Exam Questions

**Q: Why does random CV produce optimistic estimates here?**
Observations from the same person share individual physiology (their resting heart rate, typical signal amplitudes). Random splits allow the model to "see" some of a test subject's data during training → learns their personal baseline → inflated performance. LOIO prevents this by holding out all observations from one subject at once.

**Q: What if some subjects have missing data?**
LOIO still valid with unequal numbers of observations per subject — simply exclude missing observations. Report how many complete subjects were used.

**Q: Which EPE is larger and why?**
Generalized EPE > Personalized EPE. The personalized model knows the individual's physiology; the generalized model must predict for an entirely new person with unknown characteristics → harder task → higher error.

**Q: Can the personalized model be used clinically?**
Only for monitoring known patients with existing calibration data. Cannot be used for triage, screening, or first-contact assessment of new patients.
