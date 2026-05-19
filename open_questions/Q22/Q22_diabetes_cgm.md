# Q22 — Diabetes / Continuous Glucose Monitoring (CGM)
> Possible unseen Q22 variant. High-yield because it combines repeated measures, patient-level grouping, temporal leakage, and a very natural personalized-vs-generalized deployment question.

---

## Typical Dataset

A plausible exam dataset:

- 40 diabetes patients
- repeated glucose measurements over many days
- wearable or sensor features: CGM curve summaries, insulin dose, meal timing, physical activity, sleep
- target:
  - **classification**: hypoglycemia event yes/no, poor glycemic control yes/no
  - **regression**: next-hour glucose level, HbA1c proxy, time-in-range percentage

The key structural fact is the same as wearables:
- many observations per patient
- strong within-patient correlation
- possible temporal ordering

So random CV is dangerous.

---

## Why Standard Random CV Fails

Observations from the same patient share:

- baseline glucose level
- insulin sensitivity
- typical meal pattern
- medication regime
- sensor calibration / device effects

If some windows from patient $i$ are in training and other windows from patient $i$ are in test, the model partly learns that patient's physiological fingerprint.

That means random CV estimates performance on a **known patient**, not on a truly unseen patient.

So the IID assumption fails at the patient level.

---

## Variant A — Personalized Diabetes Model

### Question type

"How well does the model predict future glucose excursions for a patient already being monitored?"

### Correct design

Use a **within-patient temporal holdout** or **leave-one-day-out** design.

Example for one patient with 14 monitored days:

```text
Fold 1: Train on days 1-13  -> Test day 14
Fold 2: Train on days 1-12  -> Test day 13
...
```

or rolling-origin evaluation:

```text
Train days 1-7   -> Test day 8
Train days 1-8   -> Test day 9
Train days 1-9   -> Test day 10
...
```

### Why

This respects both:

- patient identity
- temporal direction: train on past, test on future

### What EPE measures

$$
\text{EPE}_\text{pers} = E_{t_\text{future} \mid i_\text{fixed}}[\mathcal{L}(y_t,\hat f_i(x_t))]
$$

This is future prediction for a **known diabetic patient**.

---

## Variant B — Generalized Diabetes Model

### Question type

"How well does the model work for a new diabetes patient not seen during training?"

### Correct design

Use **Leave-One-Patient-Out CV** or **grouped K-fold by patient**.

For $N=40$ patients:

```text
Fold i:
Train on patients {1,...,40} \ {i}
Test  on patient i
```

If temporal structure matters strongly, each patient's test data should also be chronologically later than their train data in any personalized sub-analysis.

### What EPE measures

$$
\text{EPE}_\text{gen} =
E_{i_\text{new}}\left[
E_{x,y \mid i_\text{new}}[\mathcal{L}(y,\hat f(x))]
\right]
$$

This is the correct estimate for deployment on a **new diabetes patient**.

---

## Personalized vs Generalized in Diabetes

| Property | Personalized | Generalized |
|----------|--------------|-------------|
| Test subject | Known patient | New patient |
| Split unit | Future day/window within one patient | Entire patient held out |
| Main variation captured | Within-patient temporal variation | Between-patient heterogeneity |
| Expected EPE | Lower | Higher |
| Clinical use | Ongoing patient monitoring | First-use decision support |

Exactly as in wearables:
$$
\text{EPE}_\text{gen} > \text{EPE}_\text{pers}
$$
because generalized prediction must handle between-patient variation.

---

## If the Dataset Is Time Series

This is the biggest diabetes-specific exam trap.

If the predictors are sliding windows from CGM time series, then there are **two leakage risks**:

1. **Patient leakage**: same patient in train and test
2. **Temporal leakage**: future glucose information used to predict the past

So the rule is:

- never split overlapping windows across train and test
- never train on future windows to predict earlier windows
- for deployment on new patients, hold out entire patients

If asked to choose between random K-fold and forward chaining, the correct answer is forward-chaining within patient or grouped forward-chaining across patients, depending on the deployment goal.

---

## Hyperparameter Tuning

If tuning is required, use nested CV.

### Correct

- outer loop: Leave-One-Patient-Out
- inner loop: grouped CV on the remaining training patients

### Wrong

- tune hyperparameters on the full dataset first
- then report LOPO performance

That leaks test-patient information into model selection.

---

## Feature Selection Trap

Diabetes datasets often have many engineered features:

- glucose variability metrics
- time-since-meal
- insulin-on-board
- activity summaries
- sleep summaries

If the examiner asks about selecting the "best features":

**Feature selection must happen inside each training fold only.**

Otherwise the selected features have already "seen" the test patients, and EPE becomes optimistically biased.

---

## Performance Metrics

Use a metric matching the target:

### Classification

- balanced accuracy
- sensitivity / specificity
- AUC-ROC

This is important if hypoglycemia events are rare.

### Regression

- RMSE
- MAE

For rare-event clinical alarms, raw accuracy is usually a poor choice because class imbalance can make it misleading.

---

## Full Exam-Style Answer

*"This dataset contains repeated glucose-related measurements from the same diabetes patients, so observations are not IID. Measurements from one patient share baseline glucose regulation, insulin sensitivity, medication effects, and behavioral patterns. Therefore random cross-validation would leak patient-specific structure from training into test and would produce an optimistically biased error estimate."*

*"If the goal is personalized prediction for a known patient, I would use a within-patient temporal validation design such as leave-one-day-out or rolling-origin evaluation, always training on past data and testing on future data. This estimates future prediction error for that same patient."*

*"If the goal is deployment on a new patient, I would use leave-one-patient-out cross-validation, where all observations from one complete patient are held out in each fold and the model is trained on the remaining patients. This gives an honest estimate of inter-patient generalization. If hyperparameters or feature selection are needed, they must be performed inside an inner grouped CV loop on the training patients only."*

*"The generalized error will be larger than the personalized error because it includes between-patient variation in glucose regulation, which the personalized model does not need to handle."*

---

## Additional Possible Exam Questions

**Q: Why is this diabetes dataset not IID?**
Because repeated measurements from the same patient are correlated through shared physiology, treatment regime, and lifestyle patterns. Time ordering can create additional autocorrelation.

**Q: Why is leave-one-observation-out invalid here?**
Because the held-out observation still comes from a patient whose many other observations remain in training. The model effectively sees that patient already, so the estimate is optimistic for new-patient deployment.

**Q: What is the correct split unit?**
Usually the patient. If the question is personalized forecasting, the split unit is a future time block within one patient.

**Q: What if the question asks about predicting next-hour glucose?**
Then temporal ordering is essential. Use forward-chaining or rolling-origin validation, not random splits.

**Q: What if some patients wear different sensors?**
Then sensor/device effects act like an additional grouping or batch variable. If deployment is to new devices or clinics, the design may need leave-one-device-out or leave-one-site-out logic as well.
