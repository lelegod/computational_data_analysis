# Week 2 — Model Selection & Assessment, KNN (Exam Focus)

## Must-Know Facts

### KNN (K-Nearest Neighbours)
- **Classification**: assign the majority class among the $K$ nearest neighbours (by Euclidean distance).
- **Regression**: predict the average response of the $K$ nearest neighbours.
- Standardize features to mean 0, variance 1 before computing distances.
- Small $K$: low bias, high variance (overfit — jagged boundaries).
- Large $K$: high bias, low variance (underfit — smooth/flat boundaries).
- $K$ is chosen by cross-validation.

### Model Selection vs Model Assessment
- **Model selection**: choosing a tuning parameter $\lambda$ or between model classes — uses a **validation set**.
- **Model assessment**: estimating the generalisation error of the *chosen* model — uses a **test set**.
- Test set should be used **once only** — using it for selection contaminates the assessment.
- If data is limited: use cross-validation as a substitute for separate validation/test sets.

### Training / Validation / Test Split
- **Training set**: fit model parameters.
- **Validation set** (= dev set = hold-out CV set): select $\lambda$, features, model class.
- **Test set**: final honest performance estimate — no decisions based on this.
- Dev/test sets must come from the same distribution as future data.

### K-Fold Cross-Validation
- Split data into $K$ equal parts (folds).
- For each fold $k$: fit model on the other $K-1$ folds, evaluate on fold $k$.
- **CV error**: $CV(\lambda) = \frac{1}{K}\sum_{k=1}^{K} Err_k(\lambda)$
- **SE of CV error**: $S.E.(\lambda) = \frac{1}{\sqrt{K}}\sqrt{\frac{1}{K}\sum_k (Err_k(\lambda) - CV(\lambda))^2}$
- NOTE: the SE is a **biased** (underestimated) variance — folds are correlated!
- Typical choices: $K = 5$ or $K = 10$. LOOCV ($K = N$): unbiased but folds are too similar.

### The 1-SE Rule
- After CV, do NOT always pick the $\lambda$ with minimum CV error.
- **1-SE rule**: choose the **most regularised** $\lambda$ whose CV error is within 1 SE of the minimum.
- Effect: picks a simpler/sparser model that is almost as good — more stable.

### Leave-k-Groups-Out CV
- When observations are grouped (e.g., time series, subjects), let each **group** form a fold.
- Prevents information leakage between folds from correlated observations.
- Example: 4 groups → 4-fold CV where each fold = 1 group.

### CV Considerations (Common Leakage Traps)
- **Permute data** before splitting (sorted data can bias folds).
- **Normalise training data separately** for each fold (do not use test-fold statistics).
- **Impute** missing values separately within each training fold.
- If observations are sampled in groups: keep groups together in one fold.
- All pre-processing must happen **inside** the CV loop.

### Optimism of Training Error
- Training error $\overline{err}$ is always optimistic (underestimates true error) because the model was fit to that data.
- **In-sample error** $Err_{in}$: error on new outcomes at the *same* training input locations.
- **Optimism**: $op \equiv Err_{in} - \overline{err}$
- **Expected optimism**: $\omega \equiv E_y[op] = \frac{2}{N}\sum_{i=1}^N \text{Cov}(\hat{y}_i, y_i)$
- Higher complexity $\Rightarrow$ larger $\text{Cov}(\hat{y}_i, y_i)$ $\Rightarrow$ more optimism.
- For linear models with $d$ parameters: $E[Err_{in}] = E[\overline{err}] + 2\frac{d}{N}\sigma_\varepsilon^2$

### $C_p$ Statistic
- Adds a penalty to training error to correct for optimism.
- $C_p = \overline{err} + 2\frac{d}{N}\hat{\sigma}_\varepsilon^2$
  - $d$: number of free parameters in the model.
  - $\hat{\sigma}_\varepsilon^2$: noise floor (MSE of a low-bias/full model).
- **Choose the model that minimises $C_p$.**

### AIC (Akaike Information Criterion)
- Generalises $C_p$ to models with any likelihood:
  $$AIC = -\frac{2}{N}\log L + \frac{2d}{N}$$
- For the Gaussian case: $AIC(\lambda) = \overline{err}(\lambda) + 2\frac{d(\lambda)}{N}\hat{\sigma}_\varepsilon^2$ (same as $C_p$).
- Choose model with **minimum AIC**.

### BIC (Bayes Information Criterion)
- Penalises complexity more aggressively than AIC for large $N$:
  $$BIC = -2\log L + d\log N$$
- BIC penalty $\propto d\log N$ grows with $N$; AIC penalty $\propto 2d$ does not.
- For $N > e^2 \approx 7$: BIC penalises more than AIC → selects **simpler** models.
- BIC is consistent (selects the true model as $N \to \infty$); AIC is not.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $CV(\lambda) = \frac{1}{K}\sum_k Err_k(\lambda)$ | K-fold CV error | Model selection |
| $S.E.(\lambda) = \frac{1}{\sqrt{K}}\sqrt{\frac{1}{K}\sum_k(Err_k - CV)^2}$ | SE of CV error | 1-SE rule |
| $\omega = \frac{2}{N}\sum_i \text{Cov}(\hat{y}_i, y_i)$ | Expected optimism | Motivation for Cp/AIC/BIC |
| $E[Err_{in}] = E[\overline{err}] + \frac{2d}{N}\sigma_\varepsilon^2$ | In-sample error correction | Linear case |
| $C_p = \overline{err} + 2\frac{d}{N}\hat{\sigma}_\varepsilon^2$ | Cp statistic | Model selection (linear) |
| $AIC = -\frac{2}{N}\log L + \frac{2d}{N}$ | AIC | Model selection (general) |
| $BIC = -2\log L + d\log N$ | BIC | Model selection (consistent) |

---

## Common Traps (wrong answers in exams)

- ❌ The test set can be used to select the best model → ✓ The test set is used for ASSESSMENT only — never for decisions; use a validation set or CV for selection
- ❌ LOOCV ($K=N$) is always the best CV choice → ✓ LOOCV gives very similar folds (high correlation) which underestimates the SE; $K=5$ or $K=10$ is the recommended compromise
- ❌ The 1-SE rule selects the model with minimum CV error → ✓ It selects the MOST REGULARISED model within 1-SE of the minimum
- ❌ CV SE estimate is unbiased → ✓ It is BIASED (underestimated) because fold errors are correlated
- ❌ Training error is an unbiased estimate of test error → ✓ Training error is OPTIMISTICALLY BIASED; Cp/AIC/BIC correct for this
- ❌ AIC penalises more than BIC for large $N$ → ✓ BIC penalty is $d\log N$, which is LARGER than AIC's $2d$ whenever $N > e^2 \approx 7$
- ❌ Normalize data before splitting into CV folds → ✓ Normalize WITHIN each fold using training-fold statistics only — normalizing before causes leakage
- ❌ Small $K$ in KNN means smoother boundaries → ✓ Small $K$ means JAGGED/complex boundaries (low bias); large $K$ is smoother (high bias)
- ❌ BIC is consistent and so is AIC → ✓ BIC is consistent (recovers true model asymptotically); AIC is NOT consistent (overfits as $N \to \infty$)
- ❌ $C_p$ and AIC are unrelated → ✓ For Gaussian models, $C_p$ and AIC are IDENTICAL

---

## Quick Decision Rules

- "How to choose $\lambda$?" → K-fold CV; use 1-SE rule for simpler/more stable model
- "Smallest model within 1 SE of CV minimum" → 1-SE rule → more regularised/simpler
- "Test set used too early" → data leakage → assessment is optimistically biased
- "Large $N$, want consistent model selection" → BIC (penalises $d \log N$)
- "Any likelihood, penalise degrees of freedom" → AIC
- "Groups/clusters in data: which CV?" → Leave-k-groups-out (keep groups in same fold)
- "Pre-processing before or inside CV?" → INSIDE CV loop (normalize each training fold separately)
- "Cp vs AIC for Gaussian?" → They are the same formula
- BH procedure is in Week 3 (Sparse Regression) — not this week
- KNN small $K$: low bias, high variance. Large $K$: high bias, low variance.
