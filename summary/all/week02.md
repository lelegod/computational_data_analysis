# Week 2 — Model Selection & Assessment, KNN

## Overview
Week 2 introduces the framework for rigorously choosing and evaluating models. It begins with K-Nearest Neighbours (KNN) as a motivating example of bias-variance tradeoff via the complexity parameter $K$. The lecture then covers over- and underfitting, Ridge regression recap, and the central distinction between model *selection* (tuning $\lambda$) and model *assessment* (estimating generalisation error). The main tools introduced are K-fold cross-validation (with the 1-SE rule and SE formula), leave-k-groups-out CV, the optimism framework, and information criteria ($C_p$, AIC, BIC).

---

## 1. K-Nearest Neighbours (KNN)

### Key Concepts
- Classify an observation $x$ by the majority class among its $K$ nearest neighbours (by Euclidean distance).
- Predict a continuous response by averaging the $K$ nearest neighbours' responses.
- Standardize features to mean 0, variance 1 before computing distances.

### Bias-Variance in KNN
- **Small $K$** (e.g. $K=1$): highly flexible, jagged decision boundary, low bias, high variance (overfits).
- **Large $K$** (e.g. $K=N$): smooth boundary close to majority class, high bias, low variance (underfits).
- $K$ is a hyperparameter chosen by cross-validation.

---

## 2. Overfitting, Underfitting, and Complexity

- **Underfitting**: model too simple → wrong assumptions → high training AND test error.
- **Overfitting**: model too flexible → memorises noise → low training error, high test error.
- Example: polynomial regression of degree $p$ on $n=10$ data points:
  - $p=0$ (constant): high bias.
  - $p=3$: good fit (if true function is cubic).
  - $p=9$: interpolates all points, wildly wrong between them.
- More data reduces overfitting: $p=9$, $n=100$ looks reasonable; $p=9$, $n=10$ is disastrous.
- Ridge regression adds a quadratic penalty to control complexity via $\lambda$:
  $$\hat{\beta}_\text{Ridge} = \arg\min_\beta \|Y-X\beta\|_2^2 + \lambda\|\beta\|_2^2 = (X^TX+\lambda I)^{-1}X^TY$$
  - $\lambda=0$ → OLS (no regularisation, high variance).
  - $\lambda\to\infty$ → all coefficients $\to 0$ (high bias, no variance).

---

## 3. Model Selection vs Model Assessment

### The Two Tasks
1. **Model selection**: choose a tuning parameter $\lambda$ or choose between model classes. Use a **validation set**.
2. **Model assessment**: estimate the generalisation error of the *final chosen* model. Use a **test set**.

### Training / Validation / Test Split
| Set | Purpose | Used for decision? |
|-----|---------|-------------------|
| Training | Fit parameters | Yes |
| Validation (dev) | Tune hyperparameters, select features | Yes |
| Test | Final performance estimate | **No — report only** |

- Test set must be used **only once**. Using it for decisions makes it a validation set.
- Dev and test sets must come from the same distribution as future data (Andrew Ng: "reflect data you expect to get in the future").

### Practical Protocol (Repeated Splitting)
For each repetition $m=1,\ldots,R$:
1. Randomise (permute) data.
2. Split into training, validation, test.
3. Train on training set with range of $\lambda$.
4. Select best $\lambda$ on validation set.
5. Test model on test set — report error.

Report mean and SE of test errors over $R$ repetitions.

---

## 4. K-Fold Cross-Validation

### Algorithm
1. Randomly split data into $K$ roughly equal folds $F_1, F_2, \ldots, F_K$.
2. For each fold $k = 1, \ldots, K$:
   a. Fit the model with parameter $\lambda$ on the data excluding fold $k$: obtain $\hat{\beta}^{-k}(\lambda)$.
   b. Compute error on fold $k$: $Err_k(\lambda) = \sum_{i \in \text{fold } k}(y_i - x_i\hat{\beta}^{-k}(\lambda))^2$
3. **CV error**: $CV(\lambda) = \frac{1}{K}\sum_{k=1}^K Err_k(\lambda)$
4. Choose $\lambda^* = \arg\min_\lambda CV(\lambda)$.

### SE of the CV Error
$$S.E.(\lambda) = \frac{1}{\sqrt{K}}\sqrt{\frac{1}{K}\sum_{k=1}^K (Err_k(\lambda) - CV(\lambda))^2}$$

- This SE is **biased downward** (underestimated) because the $K$ fold errors are correlated (they share training data).

### Choice of $K$
| $K$ | Bias | Variance | Notes |
|-----|------|---------|-------|
| $K=N$ (LOOCV) | Lowest | High | Folds too similar; SE unreliable |
| $K=10$ | Low | Moderate | Good default |
| $K=5$ | Slightly higher | Lower | Good compromise |

---

## 5. The 1-SE Rule

- After CV, plot $CV(\lambda) \pm S.E.(\lambda)$ vs $\lambda$.
- The minimum CV error point is the point estimate of the optimal $\lambda$.
- **1-SE rule**: choose the **most regularised** $\lambda$ whose CV error $\leq \min CV + S.E.(\lambda^*)$.
- Rationale: CV slightly underestimates error (selection bias); picking a slightly simpler model compensates.
- Result: simpler, more stable model that generalises nearly as well.

---

## 6. Leave-k-Groups-Out Cross-Validation

- When observations are **grouped** (subjects, patients, time series segments), put each entire group into a single fold.
- Prevents information leakage between training and validation folds.
- Example: 4 groups of subjects → 4-fold CV, each fold = one group.
- Generalises to: leave-one-subject-out (LOSO), leave-one-season-out (LOSO), etc.
- This is the correct design for the Q22 wearables question.

---

## 7. CV Considerations and Data Leakage

Observations must be **independent** (IID assumption). Violating this leaks information:

| Mistake | Consequence |
|---------|------------|
| Normalise entire dataset before splitting | Test mean/std leaks into training normalisation |
| Impute missing values on full dataset before CV | Imputation uses test set information |
| Sort data before splitting | Temporal/ordering structure leaks |
| Keep correlated observations across folds | Apparent performance better than reality |

**Correct procedure**: all pre-processing (normalisation, imputation, feature selection) must happen **inside the CV loop**, using only training-fold data.

Example of leakage: normalize ALL data first, then split 80/20. The normalisation uses test set statistics → the "99% accuracy" result is inflated.

---

## 8. Optimism of Training Error

### The Problem
- Training error $\overline{err} = \frac{1}{N}\sum_i(y_i - \hat{y}_i)^2$ is **optimistic**: the model was fit to these exact points.
- The in-sample error $Err_{in} = \frac{1}{N}\sum_i E_{y^0}[(y_i^0 - \hat{y}_i)^2]$ measures error on NEW outcomes at the SAME training inputs.

### Optimism
$$op \equiv Err_{in} - \overline{err}$$

### Expected Optimism
$$\omega \equiv E_y[op] = \frac{2}{N}\sum_{i=1}^N \text{Cov}(\hat{y}_i, y_i)$$

- Higher model complexity $\Rightarrow$ larger $\text{Cov}(\hat{y}_i, y_i)$ (model tracks each $y_i$ more closely) $\Rightarrow$ more optimism.

### Linear Case: $d$ free parameters, noise $\sigma_\varepsilon^2$
$$E[Err_{in}] = E[\overline{err}] + \frac{2d}{N}\sigma_\varepsilon^2$$

This means: unbiased test error estimate = training error + penalty for complexity. This is the **theoretical foundation** of $C_p$, AIC, and BIC.

---

## 9. $C_p$ Statistic

$$C_p = \overline{err} + 2\frac{d}{N}\hat{\sigma}_\varepsilon^2$$

- $\overline{err}$: actual training error.
- $d$: number of free parameters in the model.
- $\hat{\sigma}_\varepsilon^2$: noise floor — estimated from a low-bias model (e.g. OLS with all features).
- **Rule**: choose the model that minimises $C_p$.
- Intuition: adding a variable ($d \uparrow$) always decreases $\overline{err}$ but increases the penalty; $C_p$ identifies the crossover point.

---

## 10. AIC (Akaike Information Criterion)

$$AIC = -\frac{2}{N}\log L + \frac{2d}{N}$$

- $L$: maximised log-likelihood of the model.
- Generalises $C_p$ to any likelihood-based model (logistic regression, etc.).
- **Gaussian special case**: $AIC(\lambda) = \overline{err}(\lambda) + 2\frac{d(\lambda)}{N}\hat{\sigma}_\varepsilon^2$ (identical to $C_p$).
- $d(\lambda)$: effective degrees of freedom (for Ridge: $\sum_j d_j^2/(d_j^2+\lambda)$).
- **Rule**: choose model with minimum AIC.

---

## 11. BIC (Bayes Information Criterion)

$$BIC = -2\log L + d\log N$$

- Heavier penalty than AIC: $d\log N$ vs $2d$ (since $\log N > 2$ for $N > e^2 \approx 7$).
- For large $N$, BIC penalises extra parameters far more aggressively.
- **BIC is consistent**: as $N \to \infty$, BIC selects the true model (if it is in the candidate set).
- AIC is NOT consistent: tends to select too complex a model asymptotically.
- **Rule**: choose model with minimum BIC.

### AIC vs BIC

| Property | AIC | BIC |
|----------|-----|-----|
| Penalty | $2d$ | $d\log N$ |
| Selects more complex or simple? | More complex (for large $N$) | Simpler |
| Consistent? | No | Yes |
| Based on | KL divergence | Bayesian model evidence |
| Use when | Prediction focus | True model in candidate set |

---

## 12. Comparison: CV vs Information Criteria

| | Cross-Validation | $C_p$ / AIC / BIC |
|--|-----------------|------------------|
| Assumptions | None (only IID) | Distributional (linear / Gaussian) |
| Cost | $K \times$ fitting cost | Single fit |
| Consistent? | Yes (in probability) | BIC yes; AIC no |
| Good for non-linear models? | Yes | Only if likelihood available |
| Handles pipeline honestly? | Yes (if done correctly) | No (can't capture preprocessing) |
