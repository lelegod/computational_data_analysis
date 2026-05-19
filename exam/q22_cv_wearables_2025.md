# Q22 — CV Design for Wearable Biosignals (2022, 2024, 2025 Exams)

> **10 points** | Appeared in **all 3 exams** — know this cold
> **Dataset:** 16 individuals × 3 activities × 4 seasons = **192 observations**
> **Features:** BVP (blood volume pressure), skin temperature, heart rate (HR)
> **Target:** Stress level prediction
>
> **Question:** Design training/validation/test splits to estimate EPE for:
> a) A **personalized** model — predict stress for a specific individual
> b) A **generalized** model — predict stress for a new individual
> Also: discuss trade-offs and which is more appropriate for clinical deployment.

---

## Why Standard Random CV Fails Here

Standard $K$-fold CV randomly assigns observations to folds, which is only valid when data are **IID (independently and identically distributed)**.

This dataset violates IID in two ways:
1. **Within-individual dependency:** Multiple observations from the same person share physiology — they are NOT independent
2. **Temporal dependency:** Observations from the same season are correlated

If you randomly split 192 observations into folds, individual 5's winter data ends up in training AND test folds. The model has already "seen" that person → EPE estimate is **too optimistic**. This is **data leakage**.

The CV design must reflect the **prediction scenario** you actually care about.

---

## a) Personalized Model (~4 pts)

**Goal:** Predict stress for a **specific, known individual** — like a smartwatch that has been calibrated to your body.

**Data available:** Only the target individual's observations → $3 \times 4 = 12$ observations

**CV Design — Leave-One-Season-Out (4-fold):**

```
Fold 1: Train on seasons {Spring, Summer, Autumn} (9 obs) → Test on Winter  (3 obs)
Fold 2: Train on seasons {Winter, Summer, Autumn} (9 obs) → Test on Spring  (3 obs)
Fold 3: Train on seasons {Winter, Spring, Autumn} (9 obs) → Test on Summer  (3 obs)
Fold 4: Train on seasons {Winter, Spring, Summer} (9 obs) → Test on Autumn  (3 obs)
```

**Why leave-one-season-out and not random?**

Even within one individual, the 12 observations are **not independent**. Randomly splitting them would mix seasonal data across train/test, leaking temporal patterns. Holding out a full season tests whether the model generalises to **new time points** — which is the realistic deployment scenario (your watch trains on past months and predicts future stress).

**EPE estimated:** Expected error when predicting a new season for THIS individual.

**If hyperparameter tuning needed — Nested CV:**
- Outer loop: leave-one-season-out (4-fold) → estimates EPE
- Inner loop: leave-one-condition-out within training folds → selects hyperparameters

---

## b) Generalized Model (~4 pts)

**Goal:** Predict stress for a **new, unseen individual** — for clinical deployment where new patients have no prior data.

**Data:** All 16 individuals × 12 observations = 192 observations

**CV Design — Leave-One-Individual-Out (LOIO-CV, 16-fold):**

```
Fold 1:  Train on individuals {2, 3, ..., 16} (180 obs) → Test on individual 1  (12 obs)
Fold 2:  Train on individuals {1, 3, ..., 16} (180 obs) → Test on individual 2  (12 obs)
Fold 3:  Train on individuals {1, 2, 4, ..., 16} (180 obs) → Test on individual 3  (12 obs)
...
Fold 16: Train on individuals {1, 2, ..., 15}  (180 obs) → Test on individual 16 (12 obs)
```

**Critical rule:** All 12 observations from one individual must stay in the **same fold**. Never split one person's data across train and test.

**Why?** Observations from the same person share physiology. If person 7's data appears in both training and test folds, the model has implicitly "learned" that person's patterns → EPE estimate is inflated/optimistic. We want to know: *"How well does this model generalise to a person it has never seen?"* — LOIO-CV answers exactly that.

**EPE estimated:** Expected error when predicting a completely new individual.

**If hyperparameter tuning needed — Nested CV:**
- Outer loop: leave-one-individual-out (16-fold) → estimates EPE
- Inner loop: leave-one-season-out within training folds → selects hyperparameters

---

## Trade-offs and Clinical Recommendation (~2 pts)

| | Personalized | Generalized |
|--|-------------|------------|
| **Accuracy** | Higher — calibrated to one individual's physiology | Lower — must generalise across different physiologies |
| **Training data required** | Prior data from that specific person | No prior data from the target person |
| **Applicable to** | Known individuals with history | Anyone, including new patients |
| **Clinical scenario** | Long-term monitoring of existing patients | First-visit assessment of new patients |
| **EPE measures** | Error on new time points for THIS person | Error on a new person entirely |
| **Deployment bottleneck** | Cannot deploy to new patient — no data to train on | Ready to deploy immediately |

**Clinical recommendation: Generalized model is more appropriate for deployment.**

In a clinical setting, patients arrive with **no prior wearable data**. A personalized model cannot be trained until data is collected — it is not useful at the point of need. The generalized model learns population-level patterns that transfer to new individuals, which is what a clinician needs when seeing a patient for the first time.

**Bonus insight (extra marks):** The two approaches are not mutually exclusive. A sensible clinical pipeline would:
1. Deploy the generalized model immediately for new patients
2. As patient-specific data accumulates over visits, fine-tune toward a personalized model
This hybrid approach starts general and becomes increasingly personalized over time.

---

## Quick Reference — What Changed Across Exams

| Exam | Q22 Topic | Key difference |
|------|-----------|----------------|
| 2022 | Face image clustering | Unsupervised — find number of unique people |
| 2024 | Wearable biosignals CV | Personalized vs generalized, same dataset |
| 2025 | Wearable biosignals CV | Same dataset, adds clinical deployment trade-off question |

The 2024 and 2025 Q22 are almost identical — master this answer and you cover both.
