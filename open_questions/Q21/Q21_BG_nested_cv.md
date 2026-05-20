# Q21-BG — Nested Cross-Validation and Data Leakage
> Weeks 2 and 3. Could ask: why single-loop CV is biased when hyperparameters are tuned, describe the nested CV structure, enumerate and explain the three types of data leakage, or apply nested CV to the wearables dataset.

---

## The Problem

Suppose we want to:
1. Select the best hyperparameter (e.g., $\lambda$ for Ridge, $K$ for K-NN, tree depth for RF).
2. Report an honest estimate of the test error for the resulting model.

**Why naive CV fails**: if we use the same data to both select $\lambda$ and estimate test error, the reported error is optimistically biased. The $\lambda$ was chosen to minimise error on those folds — we have effectively trained on the test set (through the hyperparameter).

**Example**: select $\lambda^*$ on a 10-fold CV loop over all 192 observations. Then report the CV error at $\lambda^*$. This error is **not** an estimate of how well a new model trained with the same procedure would perform on held-out data — it has already been used to select $\lambda^*$.

---

## Nested CV Structure

Nested CV uses two loops:

```
Outer loop (K folds) — estimates test error
  For each outer fold i (i = 1, ..., K):
    Outer test set:    fold i
    Outer training set: all folds except i

    Inner loop (K' folds, on the outer training set only) — selects hyperparameters
      For each candidate λ:
        Fit model on K'-1 inner folds
        Evaluate on the remaining 1 inner fold
      Choose λ*(i) = argmin inner CV error

    Fit final model with λ*(i) on the full outer training set
    Evaluate on outer test fold i → record error_i

Outer test error estimate = mean(error_1, ..., error_K)
```

**Key**: $\lambda^*(i)$ is selected using only the outer training set for fold $i$ — it never sees fold $i$. The outer test error is therefore an honest estimate of the error for the whole pipeline (including hyperparameter selection).

**Note**: $\lambda^*$ may differ across outer folds. This is expected and correct — each outer fold gives a slightly different training set.

---

## The Wearables Application

For the 16-subject wearables dataset:

- **Outer loop**: Leave-One-Individual-Out (LOIO, 16 folds). Outer fold $i$ holds out all 12 observations from subject $i$.
- **Inner loop**: K-fold CV (e.g., $K' = 5$ or LOIO on the 15 remaining subjects) to select $\lambda$ for regularised logistic regression.

```
Outer loop: LOIO (16 folds)
  Fold i: training = 15 subjects × 12 obs = 180 obs; test = 12 obs

    Inner loop: LOIO on 15 training subjects (15 folds)
      For each λ: fit on 14 subjects, eval on 1
      Select λ*(i) = argmin inner CV error

    Refit with λ*(i) on all 180 training obs
    Evaluate on the 12 test obs of subject i

Final estimate: mean over 16 outer fold errors
```

**Critical**: selecting $\lambda$ once on all 192 observations before the outer loop is leakage — the test subject's physiology influenced the choice of $\lambda$.

---

## The Three Types of Data Leakage

### 1. Test contamination (naive leakage)
Test observations appear in the training set. Trivially wrong but sometimes occurs through indexing errors. Example: `train_test_split` with wrong indices; shuffling before grouping.

### 2. Selection leakage (feature leakage)
Feature selection (e.g., choose top 50 most correlated features, backward elimination) is performed on the full dataset, including future test folds. The selected features are optimistically predictive on the test folds because the test fold's response variable influenced which features were selected.

**Correct procedure**: feature selection must occur inside each training fold, independently.

### 3. Preprocessing leakage
Scaling, normalisation, PCA, imputation — any preprocessing that uses statistics (mean, variance, principal components) is fitted on the full data including test observations. The test set's mean and variance bleed into the training standardisation.

**Correct procedure**: fit all preprocessing transformers on the training fold; apply (without refitting) to the test fold.

---

## The 1-SE Rule

After CV, the minimum-CV-error model is not necessarily the best choice. The **1-SE rule**: choose the simplest (most regularised) model whose CV error is within one standard error of the minimum:

$$\text{SE} = \frac{\hat{\sigma}_\text{fold}}{\sqrt{K}}$$

where $\hat{\sigma}_\text{fold}$ is the standard deviation of the $K$ fold errors.

**Justification**: CV errors at nearby $\lambda$ values are statistically indistinguishable given the noise in fold-level estimates. Preferring the simpler model protects against overfitting to the CV noise.

**Example (wearables)**: two values $\lambda_1 < \lambda_2$ have mean LOIO errors that differ by less than 1 SE. The 1-SE rule picks $\lambda_2$ (stronger regularisation) even though $\lambda_1$ has slightly lower mean CV error.

---

## Why Not AIC Instead of Nested CV?

AIC is asymptotically equivalent to LOO-CV under two conditions:
1. Observations are **IID**.
2. The model class is correctly specified and likelihood-based.

Both conditions fail for the wearables dataset:
1. Observations from the same subject are correlated — the IID assumption is violated.
2. Standard LOO (leave one *observation* out) still exposes the test subject's data in the training fold. AIC inherits this leakage.

AIC/BIC can be used *inside* the inner loop (as an alternative to inner CV for model selection) but cannot replace the outer LOIO structure that enforces inter-individual generalization.

---

## Key Properties

**What nested CV actually estimates**: the outer loop estimates the test error of the **entire pipeline** — "if I apply this model selection procedure to a new dataset of the same size and structure, what test error will I see?" It audits the selection procedure, not just a single model.

**Computational cost**: $O(K \times K' \times |\Lambda| \times \text{fit cost})$ where $|\Lambda|$ is the number of hyperparameter candidates. For LOIO on the wearables data: $16 \times 15 \times |\Lambda|$ model fits.

**Gap between inner and outer CV errors**: if the outer CV error is much larger than the inner CV error for the selected $\lambda$, this indicates selection-induced overfitting — the $\lambda$ chosen by the inner loop is too optimistic.

---

## Comparison to Alternatives

| Method | Hyperparameter selection | Test error estimate | IID required? | Grouped data? |
|--------|--------------------------|---------------------|---------------|---------------|
| Single train/val/test split | On val set | Test set only | No | Works if grouped |
| Single-loop CV | On same loop | Biased (optimistic) | Yes | No |
| Nested CV | Inner loop | Outer loop (honest) | No | Yes |
| AIC/BIC | Analytic penalty | Asymptotically consistent | Yes | No |

---

## Limitations

- **Computational cost**: $K \times K'$ times more expensive than single-loop CV.
- **Small outer test folds**: with $K = 16$ (LOIO), each fold has only 12 test observations — high fold-level variance. Mitigate by reporting mean ± SE over outer folds.
- **$\lambda^*$ instability**: $\lambda^*$ may differ substantially across outer folds if the training set is small. This instability is real information — it means the optimal regularisation is sensitive to which subjects are in the training set.
- **Not a single model**: nested CV produces an error estimate for the pipeline, not a deployable model. For deployment, refit the final model on all data with $\lambda^*$ selected by a single inner CV loop on the full dataset.

---

## Additional Possible Exam Questions

**Q: A student selects $\lambda$ using 10-fold CV on all 192 observations and then reports that CV error as the test error. What is wrong with this?**
The student selected $\lambda$ by minimising the CV error — effectively finding $\lambda$ that performs best on those folds. Then reporting the same CV error at $\lambda^*$ is double-dipping: the data was used for both selection and evaluation. The reported error is optimistically biased. Nested CV solves this by separating the selection loop (inner) from the evaluation loop (outer).

**Q: What is preprocessing leakage and how does it arise in the wearables dataset?**
Preprocessing leakage occurs when transformations such as feature scaling are fitted on all data (including test observations) before CV splits. In the wearables dataset: if we standardise each feature using the mean and standard deviation of all 192 observations and then run LOIO-CV, the test subject's physiological values have already informed the scaling. The correct approach is to fit the scaler on the 180 training observations of each outer fold and apply the same transform to the 12 test observations.

**Q: Why is it important that $\lambda^*$ can differ across outer folds in nested CV?**
Each outer fold trains on a different subset of 15 subjects. The optimal regularisation for that training set may differ from the optimal for other subsets — this is genuine variability in what $\lambda$ is appropriate for a given sample. If $\lambda^*$ is very stable across folds, the hyperparameter is not very sensitive to the specific training sample. If it varies widely, the regularisation is sensitive to which subjects are in training — this is important information about model stability.

**Q: For the 1-SE rule, which direction does "simpler model" correspond to for Ridge regression?**
For Ridge regression, larger $\lambda$ = stronger regularisation = simpler model (coefficients are pulled harder toward zero, fewer effective parameters). So the 1-SE rule selects the largest $\lambda$ whose CV error is within 1 SE of the minimum-error $\lambda$. This is in the direction of more regularisation, protecting against overfitting to CV noise.

**Q: What does nested CV estimate if $K' = 1$ (a single inner validation fold)?**
A single inner validation fold is equivalent to a simple train/val split for hyperparameter selection. The outer CV loop still gives a valid estimate of the pipeline's test error, but the inner validation split is noisy — $\lambda^*$ is highly variable across inner splits because it is based on a single fold. Using $K' \geq 5$ inner folds reduces this variance at the cost of more computation.
