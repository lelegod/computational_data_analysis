# Q21-AS — Bootstrap vs Cross-Validation vs OOB Error
> Weeks 2/5. Could ask: compare three resampling-based ideas, explain what each estimates, and discuss why they are not interchangeable.

---

## The Shared Theme

All three use data reuse, but they answer different questions.

- **Bootstrap**: what is the sampling variability of an estimator?
- **Cross-validation**: what is the out-of-sample prediction error?
- **OOB error**: what is the bagging/random-forest prediction error without extra resampling?

So even though they all involve repeated data splits or resamples, they are not solving the same problem.

---

## Bootstrap

Bootstrap resamples with replacement from the observed dataset.

For each bootstrap sample:
1. sample $N$ observations with replacement
2. recompute the statistic or refit the model
3. examine the distribution across resamples

### Main use

- standard errors
- confidence intervals
- estimator stability

Bootstrap is mainly about **uncertainty quantification**.

---

## Cross-Validation

Cross-validation partitions the data into train/test splits without replacement.

For $K$-fold CV:
- train on $K-1$ folds
- test on the held-out fold
- average the test errors

### Main use

- estimate test error
- choose tuning parameters
- compare predictive models

CV is mainly about **prediction assessment**.

---

## OOB Error

In bagging or Random Forest, each bootstrap sample leaves out about 36.8% of observations.

For a given observation:
- only trees that did not include it in training are used to predict it
- these are its out-of-bag trees

Then OOB error is computed by aggregating these predictions across observations.

### Main use

- internal test-error estimate for bagging / RF
- no extra CV loop required

So OOB is a built-in validation mechanism for bootstrap ensembles.

---

## The Core Comparison

### Bootstrap

- resample with replacement
- estimate variability, not mainly test error

### Cross-Validation

- split without replacement
- estimate predictive error directly

### OOB

- byproduct of bootstrap ensembles
- estimate predictive error for bagging/RF specifically

---

## Comparison Table

| Property | Bootstrap | Cross-Validation | OOB Error |
|----------|-----------|------------------|-----------|
| Sampling style | With replacement | Without replacement | Implicit from bootstrap trees |
| Main purpose | Uncertainty | Prediction error | Prediction error |
| Model-agnostic | Mostly yes | Yes | Only for bootstrap ensembles |
| Typical output | SE / CI / distribution | CV error | OOB error |
| Extra fitting cost | Yes | Yes | No extra beyond the ensemble |

---

## Why They Are Not Interchangeable

This is often misunderstood.

- Bootstrap is not the default tool for estimating test error
- CV is not the default tool for confidence intervals
- OOB is not a general-purpose alternative to CV for all models

Each method is tailored to a different inferential target.

---

## When to Use Which

**Use bootstrap when**:
- you want standard errors or confidence intervals
- the sampling distribution is analytically difficult

**Use cross-validation when**:
- you want to tune a model
- you want to compare predictive methods

**Use OOB error when**:
- the model is bagging or Random Forest
- you want a free internal generalization estimate

---

## Limitations

1. Bootstrap may be biased for some predictive-error tasks.
2. CV can be noisy and computationally expensive.
3. OOB applies naturally only to bootstrap-based ensembles.
4. All three depend on data representativeness.

---

## Additional Possible Exam Questions

**Q: Why is OOB error often described as “free”?**
Because the necessary held-out predictions are already created by bootstrap sampling inside bagging or Random Forest. No extra resampling procedure is needed.

**Q: Why is bootstrap better suited to confidence intervals than CV?**
Because bootstrap approximates the sampling distribution of an estimator by repeated resampling, which is exactly what confidence-interval construction needs.

**Q: Why is CV preferred over bootstrap for model tuning?**
Because CV directly evaluates out-of-sample predictive performance on held-out data, which is the target of tuning decisions.
