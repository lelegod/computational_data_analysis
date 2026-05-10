# Week 5 — CART (Continued) and Bagging (Exam Focus)

## Must-Know Facts

### Impurity Measures (Classification Trees)
- Gini and cross-entropy are used to GROW trees (sensitive to probability changes).
- Misclassification rate is used to PRUNE / EVALUATE trees (not sensitive to probability changes).
- All three measures equal 0 for a pure node and are maximised for equally distributed classes.
- For binary classification ($p$ = proportion of class 1):
  - Misclassification $= \min(p,\, 1-p)$
  - Gini $= 2p(1-p)$
  - Entropy $= -p\log(p) - (1-p)\log(1-p)$

### Bootstrapping
- A bootstrap sample of $N$ observations is drawn WITH REPLACEMENT from $N$ training observations.
- About 63.2% of unique observations appear in each bootstrap sample (on average).
- About 36.8% of observations are left out = out-of-bag (OOB) sample.
- $P(\text{observation NOT in bootstrap}) \to \frac{1}{e} \approx 0.368$ as $N \to \infty$.

### Bagging Core Facts
- Bagging = Bootstrap Aggregating (Breiman, 1996).
- Works best for HIGH-VARIANCE, LOW-BIAS methods (i.e., deep trees).
- Bagging REDUCES VARIANCE but does NOT change bias.
- Bias of bagged ensemble = bias of a single tree (trees are identically distributed).
- For regression: average predictions of $B$ trees.
- For classification: majority vote of $B$ trees.
- Increasing $B$ (number of trees) never causes overfitting.
- The limiting factor for variance reduction is $\rho$ (inter-tree correlation).
- As $B \to \infty$, variance $\to \rho \sigma^2$ (a floor, not zero).

### Bagging Variance Formula

$$\text{Var}(\hat{y}_{\text{bag}}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

  - $\rho$ = pairwise correlation between any two trees
  - $\sigma^2$ = variance of a single tree prediction
  - Second term $\to 0$ as $B \to \infty$
  - First term ($\rho \sigma^2$) is the irreducible floor due to tree correlation

### OOB Error
- OOB error $\approx$ leave-one-out cross-validation error.
- Free by-product of bagging (no extra CV runs needed).
- For each observation $i$: only trees trained WITHOUT observation $i$ are used to predict $i$.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $\hat{p}_{mk} = \frac{1}{N_m} \sum \mathbf{I}(y_i = k)$ | Class proportion in leaf $m$ | Compute impurity measures |
| $G = \sum_k \hat{p}_{mk}(1 - \hat{p}_{mk})$ | Gini index | Growing classification tree |
| $D = -\sum_k \hat{p}_{mk} \log(\hat{p}_{mk})$ | Cross-entropy | Growing classification tree |
| $E = 1 - \max_k(\hat{p}_{mk})$ | Misclassification rate | Pruning/evaluation |
| $C_\alpha(T) = R(T) + \alpha \lvert T \rvert$ | Cost-complexity (pruning) | Select tree size |
| $\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$ | Bagging regression prediction | Predict with bagged tree |
| $E(\hat{y} - y) = E(\hat{y}_b - y)$ | Bagging bias = individual tree bias | Understand that bagging doesn't reduce bias |
| $\text{Var} = \rho\sigma^2 + \frac{1-\rho}{B} \sigma^2$ | Bagging variance | Understand variance reduction limits |
| $P(\text{OOB}) = \left(1 - \frac{1}{N}\right)^N \to \frac{1}{e} \approx 0.368$ | OOB probability | Understand OOB sample size |

---

## Common Traps (wrong answers in exams)

- Bagging reduces bias → Bagging does NOT reduce bias; only variance is reduced.
- More trees (larger $B$) causes overfitting → Bagging does NOT overfit with more trees. Error stabilises or decreases.
- OOB error is optimistic (like training error) → OOB error is an UNBIASED estimate (like CV error), because OOB observations were not used to train those trees.
- Bagging gives full variance reduction (to $\sigma^2 / B$) → Only if trees are uncorrelated ($\rho = 0$). In practice, trees are correlated and $\rho > 0$, so the floor is $\rho \sigma^2 > 0$.
- Bagging trees should be pruned → Bagging uses UNPRUNED trees (maximum depth, minimum node size). Pruning reduces variance on individual trees, making bagging less effective.
- Misclassification rate is used to grow trees → It is NOT used for growing (insensitive to probability shifts). Gini and entropy are used.
- A bootstrap sample of size $N$ contains $N$ unique observations → It contains ~63.2% unique observations; the rest are duplicates.
- Bagging improves interpretability → Bagging REDUCES interpretability (no single readable tree; it is an ensemble average).
- The second term in the bagging variance formula is the limiting factor → The FIRST term ($\rho \sigma^2$) is the floor that limits variance reduction; the second term goes to 0 with $B$.
- Gini and misclassification rate give the same tree → They do NOT; Gini is more sensitive and will often split differently from misclassification rate.

---

## Quick Decision Rules

- If asked what bagging changes: VARIANCE decreases, bias stays the same.
- If asked what limits bagging's variance reduction: inter-tree correlation $\rho$.
- If $B \to \infty$: variance $\to \rho \sigma^2$ (not zero).
- If $\rho = 0$ (impossible in pure bagging but theoretical): variance $\to \sigma^2 / B$.
- If the method has high variance and low bias: bagging is suitable.
- If the method has high bias: bagging is NOT the right fix (boosting is).
- If asked about OOB error: it approximates cross-validation error, unbiased.
- If asked what proportion of data is in a bootstrap sample: ~63.2% unique, ~36.8% OOB.
- If asked about interpretability after bagging: interpretability is lost.
- If growing a tree: use Gini or cross-entropy.
- If pruning a tree: use misclassification rate or cross-validated RSS.
- If $\alpha = 0$ in pruning: full tree. If $\alpha$ is large: tree approaches stump/root.
