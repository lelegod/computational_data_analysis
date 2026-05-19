# Q21-O — Cross-Validation and Model Selection
> Weeks 2/3/5. The 1-SE rule, nested CV, and bias of CV estimates are common Q21 candidates.

---

## Why Cross-Validation?

Training error underestimates EPE (the model was fitted to minimize it). We need an estimate of **test error** — performance on unseen data — without wasting data on a held-out set.

**CV gives**: approximately unbiased estimate of EPE for a model trained on $N(1-1/K)$ observations.

---

## K-Fold Cross-Validation

1. Partition data into $K$ equally-sized folds $F_1,\ldots,F_K$
2. For $k=1,\ldots,K$: train on all folds except $F_k$, evaluate on $F_k$
3. $\text{CV}(K) = \frac{1}{N}\sum_{k=1}^K\sum_{i\in F_k}L(y_i, \hat{f}^{-k}(x_i))$

where $\hat{f}^{-k}$ = model trained without fold $k$.

**Leave-One-Out CV** (LOO-CV, $K=N$):
- Unbiased estimate of EPE for model trained on $N-1$ observations
- Very high variance (each training set is almost the same → highly correlated estimates)
- Computationally expensive (except for linear models: $\text{LOO-CV} = \frac{1}{N}\sum_i\left(\frac{y_i-\hat{y}_i}{1-h_{ii}}\right)^2$ where $h_{ii}$ is the leverage)

**5-fold vs 10-fold vs LOO**:
| | Bias | Variance | Cost |
|--|------|---------|------|
| 5-fold | Higher (trains on 80%) | Lower | Cheap |
| 10-fold | Lower (trains on 90%) | Moderate | Moderate |
| LOO | Lowest | Highest | Expensive |

Standard choice: **10-fold** (good bias-variance tradeoff for CV estimate itself).

---

## The 1-Standard-Error Rule

**Problem**: CV selects the $\lambda$ that minimizes mean CV error, but this is noisy. The true minimum may differ from the estimated minimum.

**The rule**:
1. Compute $\text{CV}(\lambda)$ and its standard error $\text{SE}(\lambda)$ for all $\lambda$ in a grid
2. Find $\lambda^* = \arg\min_\lambda \text{CV}(\lambda)$
3. Set threshold $= \text{CV}(\lambda^*) + 1\cdot\text{SE}(\lambda^*)$
4. Select the **simplest model** (largest $\lambda$, fewest parameters) whose CV error is below this threshold

**Why**: prefers parsimony. Two models with CV errors within 1 SE are statistically indistinguishable → choose the simpler one. Reduces overfitting of the model selection process itself.

**Graphically**: find the minimum of the CV curve, draw a horizontal line 1 SE above it, pick the simplest model below that line.

---

## Nested Cross-Validation

When both model training AND hyperparameter selection must be evaluated:

**Wrong** (single CV loop):
- Use CV to select $\lambda^*$
- Report CV error at $\lambda^*$
- **Problem**: the same data was used to select $\lambda^*$ and evaluate it → optimistic bias

**Correct** (nested CV):
- **Outer loop**: $K_\text{out}$ folds for unbiased generalization error estimate
- **Inner loop**: within each outer training fold, $K_\text{in}$-fold CV to select $\lambda^*$
- Outer test fold evaluates the model trained with the inner-loop-selected $\lambda^*$

$$\text{CV}_\text{nested} = \frac{1}{K_\text{out}}\sum_{k=1}^{K_\text{out}}L\left(y_{F_k}, \hat{f}^{-k}_{\lambda^*_{-k}}(x_{F_k})\right)$$

where $\lambda^*_{-k}$ is the hyperparameter selected by inner CV on fold $-k$.

**Cost**: $K_\text{out}\times K_\text{in}$ model fits. Common: $5\times5 = 25$ or $10\times5 = 50$ fits.

---

## What CV Actually Estimates

CV estimates EPE for a model trained on the given data. It does NOT estimate:
- Performance of a specific fitted model (it averages over $K$ re-trained models)
- Population-level performance if the data has non-IID structure

**IID assumption**: CV requires that observations are exchangeable. Violated by:
- Repeated measures (same subject multiple times → use subject-level splits)
- Time series (future depends on past → use time-based splits, never random)
- Spatial data (nearby points correlated → use spatial blocks)

---

## AIC and BIC as Alternatives to CV

For linear models: analytical approximations to test error that don't require re-fitting.

$$\text{AIC} = -2\ell(\hat{\theta}) + 2p$$
$$\text{BIC} = -2\ell(\hat{\theta}) + p\log N$$

where $\ell$ = log-likelihood, $p$ = number of parameters.

- AIC penalty ($2p$) is constant → consistent with asymptotic LOO-CV
- BIC penalty ($p\log N$) grows with $N$ → selects sparser models for large $N$ → consistent (selects true model as $N\to\infty$ if true model is in the candidate set)
- For $N > 7$: $\log N > 2$ → BIC penalizes complexity more than AIC

---

## Additional Possible Exam Questions

**Q: Why does CV have higher variance than a single train/test split?**
Each CV fold uses a different training set → $K$ different models. Their errors are correlated (training sets overlap) but not identical. LOO-CV in particular: all $N$ models differ by only one observation → training sets almost identical → estimates highly correlated → variance of the CV mean is high. A single large test split gives one estimate but with less correlation between folds.

**Q: When is AIC preferred over CV?**
AIC: computationally cheap (fit once, compute penalty). CV: computationally expensive but model-agnostic. AIC assumes Gaussian errors and correct model specification — may be inaccurate for complex models or non-Gaussian responses. CV makes no distributional assumptions and works for any loss function. For small $N$ or unusual losses: prefer CV. For fast model comparison with standard models: AIC is fine.

**Q: What is the optimism of training error and how does it lead to AIC?**
$\text{EPE} \approx \overline{\text{err}} + \text{Optimism}$ where $\overline{\text{err}}$ is training error. For linear models: $\text{Optimism} = 2p\sigma^2/N$ (Stein's lemma). Substituting: $\text{EPE} \approx -2\ell/N + 2p/N$ → multiply by $N$ → $\text{AIC} = -2\ell + 2p$. AIC is a closed-form estimate of the CV error.

**Q: What is data leakage in the context of model selection?**
Data leakage occurs when information from the test set influences the training or selection process. Examples: (1) fitting a scaler/normalizer on the full dataset before CV (test fold statistics leak into training); (2) selecting features on the full dataset before CV (test fold labels influence feature selection); (3) tuning hyperparameters on the CV error and reporting that same CV error as generalization estimate. The fix: all preprocessing steps must be inside the CV loop.
