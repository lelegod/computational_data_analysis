# Q22 — Multi-Site Medical Dataset
> Possible unseen Q22 variant. High-yield because the grouping unit changes from patient to hospital/site, but the validation logic is identical.

---

## Typical Dataset

A plausible exam dataset:

- 5 hospitals
- 100 patients per hospital
- same biomarker or imaging feature set across sites
- target: disease status, severity, treatment response

Even if patient records are independent within site, the full dataset is not IID across sites.

---

## Task Framing and Model Choice

**Task:** Supervised classification or regression (disease severity, diagnosis). State the task type explicitly.

**Feature extraction:** Features are typically already tabular (blood markers, imaging summaries). Apply standardization inside each training fold — never use the held-out site's data to compute mean/std.

**Model choice:**

| Method | Why suitable |
|--------|-------------|
| **Regularized Logistic Regression** (L1/L2) | Robust to site-level batch effects when combined with within-fold standardization; interpretable coefficients for clinical reporting |
| **LDA** | Efficient when features are approximately Gaussian per site; good with moderate $p$ |
| **Random Forest** | Handles non-linear biomarker interactions; robust to outliers from different scanner protocols |
| **Ridge Regression** | For continuous outcomes; handles collinear imaging features well |

**Site harmonization:** If scanner or protocol batch effects are large, apply a site-correction step (e.g., residualizing site effects) inside each training fold before fitting the classifier. Never harmonize using the held-out site's statistics — that is data leakage at the site level.

---

## Why Random CV Fails

Patients from the same hospital share:

- scanner calibration
- local protocol
- demographic mix
- batch / laboratory effects

Random splitting across all patients allows site-specific information to leak from training into test.

So the reported performance will overestimate generalization to a truly new hospital.

---

## Correct Design — Leave-One-Site-Out

```text
Fold 1: Train on sites 2,3,4,5 -> Test on site 1
Fold 2: Train on sites 1,3,4,5 -> Test on site 2
...
```

This directly measures performance on a site not seen during training.

---

## What EPE Measures

The correct target is:

$$
\text{EPE}_\text{site}
=
E_{s_\text{new}}\left[E_{x,y\mid s_\text{new}}[\mathcal{L}(y,\hat f(x))]\right]
$$

This is expected error on a new hospital/site.

---

## If the Goal Is Only New Patients at Known Sites

Then grouped splitting by site is not necessary if deployment truly stays within the same hospitals.

But:
- if the exam mentions generalization to other hospitals
- or deployment beyond the original study centers

then **Leave-One-Site-Out** is the correct design.

This deployment distinction is exactly what Q22 usually tests.

---

## Preprocessing Trap

Site harmonization or normalization may be discussed.

Important rule:
- site correction must be estimated on training sites only
- then applied to the test site

If harmonization uses the held-out site while estimating parameters, that is leakage.

---

## Hyperparameter Tuning

Use nested CV:

- outer loop: Leave-One-Site-Out
- inner loop: site-grouped CV within the training sites

This is especially important if:
- regularization is tuned
- thresholding is tuned
- feature selection is applied

---

## Full Exam-Style Answer

*"This dataset is not IID at the site level, because patients from the same hospital share scanner settings, local protocols, and batch effects. Therefore random cross-validation across all patients would leak site-specific information from training into test and would overestimate generalization to a new hospital."*

*"If the deployment goal is use at a hospital not included in the original dataset, I would use leave-one-site-out cross-validation, holding out one complete hospital in each fold and training on the remaining sites. This gives an honest estimate of site-level generalization. Any preprocessing, feature selection, or hyperparameter tuning must be performed inside the training folds only."*

---

## Additional Possible Exam Questions

**Q: What is the grouping variable here?**
Hospital or site.

**Q: Why is leave-one-patient-out not enough?**
Because patients from the same site would still appear in both train and test, so site effects would leak.

**Q: When is Leave-One-Site-Out the right answer?**
When the intended deployment includes hospitals not represented in training.
