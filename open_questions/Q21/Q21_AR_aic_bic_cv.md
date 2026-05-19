# Q21-AR — AIC vs BIC vs Cross-Validation
> Weeks 1/2. Could ask: compare information criteria to cross-validation, explain their penalties, and discuss prediction versus model identification.

---

## The Shared Goal

All three methods are used for model selection, but they answer slightly different questions:

- How well does the model predict?
- How much complexity should be penalized?
- Are we trying to identify the “true” model or just minimize expected prediction error?

---

## AIC

AIC is:
$$
\text{AIC} = -2\ell + 2p
$$

where:
- $\ell$ is the maximized log-likelihood
- $p$ is the number of free parameters

### Interpretation

AIC estimates out-of-sample predictive quality through an approximate KL-risk correction.

### Consequence

- favors predictive accuracy
- uses a fixed penalty $2p$
- tends to select larger models than BIC

---

## BIC

BIC is:
$$
\text{BIC} = -2\ell + p\log N
$$

### Interpretation

BIC comes from a large-sample approximation to Bayesian model evidence.

### Consequence

- penalizes complexity more strongly as $N$ grows
- tends to prefer smaller models
- more oriented toward consistent model identification

---

## Cross-Validation

Cross-validation estimates prediction error directly by repeated train/test splitting.

For $K$-fold CV:

1. split data into $K$ folds
2. train on $K-1$ folds
3. evaluate on the held-out fold
4. average the held-out errors

### Interpretation

CV is model-agnostic and loss-agnostic:
- no likelihood needed
- no Gaussian assumption needed
- directly measures predictive performance

---

## The Main Comparison

### AIC

- fast
- likelihood-based
- prediction-oriented

### BIC

- fast
- likelihood-based
- stronger complexity penalty

### CV

- computationally heavier
- assumption-light
- directly estimates test error

---

## Comparison Table

| Property | AIC | BIC | Cross-Validation |
|----------|-----|-----|------------------|
| Formula-based? | Yes | Yes | No |
| Uses likelihood? | Yes | Yes | Not required |
| Penalty strength | $2p$ | $p\log N$ | Implicit via held-out error |
| Main focus | Prediction | Parsimony / identification | Prediction |
| Model-agnostic | No | No | Yes |
| Computational cost | Low | Low | Higher |

---

## Why AIC and BIC Differ

For moderate or large $N$:
$$
\log N > 2
$$

So BIC penalizes additional parameters more heavily than AIC.

That is why:
- AIC typically keeps more complexity
- BIC typically prefers smaller models

---

## When CV Is Preferable

Cross-validation is usually preferable when:

- the model is not likelihood-based
- preprocessing is complicated
- the loss of interest is not Gaussian log-likelihood
- you care about realistic predictive performance more than analytic convenience

This is a very common exam discussion point.

---

## When AIC/BIC Are Preferable

They are attractive when:

- fitting is expensive and repeated CV is costly
- models are standard likelihood-based models
- you want a quick analytic criterion

BIC is especially natural if the question is framed as identifying the simplest plausible true model.

---

## Limitations

1. AIC/BIC depend on likelihood assumptions.
2. CV can be noisy depending on the split.
3. CV is more computationally expensive.
4. None of the three fully solves model uncertainty.

---

## Additional Possible Exam Questions

**Q: Which of AIC, BIC, and CV is most model-agnostic?**
Cross-validation, because it does not require a likelihood and can be used with essentially any fitted predictor and any loss.

**Q: Why does BIC usually select smaller models than AIC?**
Because its penalty grows like $p\log N$, which exceeds $2p$ once $N$ is moderately large.

**Q: If the goal is prediction rather than finding the true model, which criterion is usually favored?**
AIC or cross-validation, because both are more prediction-oriented than BIC.
