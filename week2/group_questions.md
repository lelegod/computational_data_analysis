# Week 2 — Group Discussion Questions

---

## Q1: The One-SE Rule (5 mins)

**Question (slide 25):**
Your 10-fold CV results for Ridge Regression are:
- Minimum Error: 1.2 ($\lambda = 0.1$), Standard Error (SE): 0.2
- Model B Error: 1.35 ($\lambda = 1.0$)
- Model C Error: 1.50 ($\lambda = 10.0$)

Calculate the selection threshold. Which $\lambda$ setting is the most "Scientific" choice?

*Vevox Poll: Pick the $\lambda$ value that an auditor would justify.*

**Answer:**

The One-SE Rule says: choose the **simplest model** whose CV error is no more than one standard error above the minimum.

**Threshold** = Minimum Error + 1 SE = $1.2 + 0.2 = 1.4$

Now check each candidate:
- $\lambda = 0.1$: Error = 1.2 — passes (below 1.4), but most complex.
- $\lambda = 1.0$: Error = 1.35 — passes ($1.35 \leq 1.4$). Simpler than $\lambda = 0.1$.
- $\lambda = 10.0$: Error = 1.50 — fails ($1.50 > 1.4$).

**Most scientific choice: $\lambda = 1.0$.**

It is the largest (most regularised, simplest) $\lambda$ whose error still falls within one SE of the minimum. An auditor would prefer this because it is more robust and less likely to have overfit to the CV noise. The key insight is that CV already tends to choose too-complex models, so the 1-SE rule compensates for this systematic bias.

---

## Q2: When Would You Use Leave-k-Groups-Out?

**Question (slide 26):**
Looking at the Leave-one-group-out CV diagram, when would you use Leave-k-groups-out cross-validation?

**Answer:**

You use Leave-k-groups-out (also called grouped or stratified CV) whenever observations are **not independently and identically distributed (IID)** but instead come in natural clusters or groups. Specific scenarios:

1. **Repeated measurements on the same subject** — e.g., multiple time points per patient. If the same patient appears in both train and test, the model has seen correlated data and the CV estimate is optimistically biased (data leakage).
2. **Spatial or temporal autocorrelation** — e.g., sensor readings from the same location or adjacent time windows. Nearby observations share information; splitting randomly leaks it.
3. **Batch effects** — experiments run in different labs, on different days, or with different equipment. Each batch is a group.
4. **Hierarchical data** — students nested within schools; pixels within images.

The rule: **if two observations share an unobserved grouping variable that affects the response, they should not be split across train and test**. Leaving one (or $k$) groups out ensures the test set represents truly unseen conditions.

---

## Q3: Spot the Leak! (5 mins)

**Question (slide 29):**
A model for clinical diagnosis follows this workflow:

- Step 1 Normalization: Subtract mean/std of the *entire* dataset.
- Step 2 Split: 80% Training, 20% Test.
- Step 3 Tuning: Use 10-fold CV on Training set to find best $\lambda$.
- Step 4 Predict: Evaluate on the Test set. Result: 99% Accuracy.

Find the "Data Leakage." How did the test set bleed into the training?

*Vevox Word Cloud: Which step or action is the source of the leakage?*

**Answer:**

**The leak is in Step 1.**

Normalisation was performed on the **entire dataset** (training + test combined) before the train/test split. This means the mean and standard deviation used to normalise the training data were computed using information from the test set.

**Why this causes leakage:**
- The test set is supposed to represent unseen future data.
- By including test observations in the mean/std calculation, the model has implicitly "seen" the test set before training.
- The test set's statistical properties have leaked into the training pipeline, making the 99% accuracy artificially inflated.

**The correct procedure:**
1. Split data first: 80% train, 20% test.
2. Compute mean and std **on the training set only**.
3. Apply those training-set statistics to normalise both the training and test sets.
4. Perform CV tuning **within the training set** (applying the same normalisation inside each fold).

**General rule:** ALL pre-processing steps (normalisation, imputation, feature selection, PCA, etc.) must be computed on training data only and then applied to test/validation data. This is the single most common source of overly optimistic results in applied ML.

---

## Q4: The Nested Detective (10 mins)

**Question (slide 45):**
An AI researcher for a surgical robot uses Nested CV:
- Outer Loop (K=5): General Error = 12%.
- In each outer fold, the "Best $\lambda$" chosen by the inner loop was different.

Group Discussion (Vevox):
1. Does the fact that $\lambda$ changed mean the audit failed?
2. Which $\lambda$ do you use for the final robot deployed in the hospital?
3. If the Outer Error (12%) is much higher than the Inner Error (5%), what does this tell the Auditor?

*The Auditor's Insight: Nested CV audits the methodology, not a specific single model.*

**Answer:**

**Sub-question 1: Does $\lambda$ changing across folds mean the audit failed?**

No. This is expected and desirable. The inner loop's job is to find the best $\lambda$ *for each specific training partition*. Since each outer fold produces a different training set (different 80% of data), it is completely normal — even informative — that the optimal $\lambda$ shifts. It means the model is sensitive to the training data, which is worth knowing. The audit has not failed; it has revealed that $\lambda$ selection is not perfectly stable across data partitions.

**Sub-question 2: Which $\lambda$ for the final deployed model?**

This is the key insight of Nested CV: **the outer loop produces an error estimate, not a final model**. For deployment, you train a new final model on **all available data** using $\lambda$ selected by a fresh inner CV loop run on the full dataset. The 12% outer error is your honest estimate of how well *that procedure* will generalise, even though the specific $\lambda$ used in the final model may differ from any of the per-fold $\lambda^*$ values.

**Sub-question 3: Outer Error (12%) much higher than Inner Error (5%)**

This gap is a warning sign of **selection-induced bias** (also called optimism bias or hyperparameter overfitting):

- The inner CV error (5%) is optimistically biased because the inner loop picked the $\lambda$ that happened to minimise error on those specific validation folds.
- The outer error (12%) is the honest estimate: it measures how well the *entire pipeline* (including the inner $\lambda$ selection step) generalises to truly unseen data.
- A large gap ($12\% \gg 5\%$) means the inner loop is overfitting to CV noise — the model is learning not just the signal but also quirks of the particular train/validation splits.

The auditor should report 12% as the expected generalisation error and investigate whether the inner CV folds are too few, the $\lambda$ grid is too fine, or the dataset is too small.

---

## Q5: Bootstrap (10 mins)

**Question (slide 51):**
You ran a Bootstrap ($B = 2000$) to assess the reliability of a feature (Age) in your model.
- Mean coefficient: 0.5
- 95% Confidence Interval: $[-0.1, 1.1]$

As a clinical auditor, do you include this feature in the final medical report? Why or why not?

*Vevox Poll: Trust, Reject, or Collect More Data?*

**Answer:**

**Verdict: Collect More Data (or provisionally reject the feature for the report).**

**Reasoning:**

The bootstrap 95% CI for the Age coefficient is $[-0.1, 1.1]$. Because this interval **contains zero**, the coefficient is not statistically distinguishable from zero at the 5% level. This means:

- We cannot confidently conclude that Age has a positive (or any) effect on the outcome.
- The mean of 0.5 is not negligible in magnitude, but the high uncertainty (CI width = 1.2) swamps it.
- Inclusion in a clinical report requires reliable evidence of a non-zero effect.

**Why not simply "Reject"?**

The mean of 0.5 is plausible and not negligible clinically. The uncertainty may stem from a small sample size. With more data ($B$ is already large at 2000, so the issue is sample size $N$, not bootstrap replicates), the CI may tighten and become entirely positive.

**Clinical auditor's framing:**
- If pressed to decide now: **do not include** Age in the final report as a reliable predictor. Reporting an unstable coefficient could mislead clinical decisions.
- The correct action is to flag this feature, collect more data, and re-run the analysis.
- Note: $B = 2000$ bootstrap replicates is appropriate for CI estimation (the lecture recommends 1000–2000 for CIs), so the uncertainty is real, not an artefact of too few replicates.
