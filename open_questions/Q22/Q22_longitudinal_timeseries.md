# Q22 — Longitudinal / Time-Series Prediction
> Possible unseen Q22 variant. High-yield because it introduces temporal leakage explicitly, which is a favorite exam trap.

---

## Typical Dataset

A plausible exam dataset:

- daily or weekly measurements from many patients
- target: relapse next week, symptom score tomorrow, biomarker in 7 days

There are usually two kinds of dependence:

1. repeated measurements within patient
2. temporal autocorrelation within each patient

So random CV is especially inappropriate.

---

## Task Framing and Model Choice

**Task:** Regression (predict symptom score, biomarker) or binary classification (relapse yes/no). State this explicitly.

**Feature engineering:** From the time series up to time $t$, extract lagged features, rolling means, trend slopes, and variability metrics as inputs. The temporal ordering dictates that only past data can be used as features for predicting time $t+k$.

**Model choice:**

| Method | Why suitable |
|--------|-------------|
| **Ridge / Lasso** | Handles many lagged features with regularization; Lasso selects relevant lags |
| **Gradient Boosting** | Captures non-linear temporal patterns; robust |
| **Regularized Logistic Regression** | For binary outcome (relapse yes/no); probability output useful clinically |
| **LDA** | If feature count is modest and Gaussian assumption holds |

**Critical:** In forward-chaining CV, fit the model using only past observations in each fold. Never train on future time points. All feature normalization must use statistics from the training window only.

---

## Why Random CV Fails

Random splits can put:

- future observations in training
- earlier observations in test

This leaks information backward in time.

If the same patient also appears in both train and test, there is an additional patient-level leakage problem.

---

## Correct Design — Forward-Chaining

If the goal is forecasting:

```text
Train on times 1,...,T
Test  on time T+1
```

or:

```text
Train on days 1-30 -> Test days 31-37
Train on days 1-37 -> Test days 38-44
...
```

The rule is always:
- train on past
- test on future

---

## If There Are Multiple Patients

Then the design depends on deployment:

### Known patients, future time

- forward-chaining within patient

### New patients

- hold out entire patients
- and within training/test, preserve time order

So a grouped temporal design may be needed.

---

## What EPE Measures

This depends on the deployment question:

- future prediction for the same patient
- or future prediction for a new patient

You should state this explicitly in the answer, because that is where many Q22 marks come from.

---

## Sliding-Window Trap

If time series are split into overlapping windows:

- nearby windows share many of the same observations

So random assignment of windows is invalid.

Split whole future blocks, not overlapping fragments.

---

## Hyperparameter Tuning

Nested validation is still required:

- outer loop: forward-chaining evaluation
- inner loop: choose hyperparameters using earlier training blocks only

Never tune on future blocks.

---

## Full Exam-Style Answer

*"This is a longitudinal time-series dataset, so random cross-validation is invalid because it breaks temporal order. The model would be trained using future observations to predict the past, which causes temporal leakage. If the same patient also appears in both training and test, there is an additional grouped-data leakage problem."*

*"If the goal is forecasting, I would use forward-chaining validation: in each fold the model is trained on earlier time points and evaluated on later time points. If the deployment goal is prediction for new patients, I would also hold out complete patients, while still preserving temporal order within each fold."*

---

## Additional Possible Exam Questions

**Q: What is the main leakage risk here?**
Future information leaking into training.

**Q: Why is random K-fold invalid?**
Because time points are not exchangeable.

**Q: What if windows overlap?**
Keep overlapping windows on the same side of the split.
