# Q21-A — Random Forest Algorithm
> Appeared: 2022 Q21

---

## Core Idea

Random Forest = Bagging + Random Feature Subsampling on decision trees. Both modifications reduce variance without increasing bias.

---

## The Algorithm (Step by Step)

### Step 1: Bootstrap Sampling
For each tree $b = 1, \ldots, B$:
- Draw bootstrap sample $Z^{*b}$ of size $N$ with replacement from training data
- Each sample contains ~63.2% unique observations; ~36.8% are out-of-bag (OOB)

**Why**: Creates $B$ different training sets → each tree sees a different data distribution → trees make different errors → averaging cancels errors.

### Step 2: Random Feature Subsampling
At each node split, select $m$ features at random (NOT all $p$ features):
- Classification default: $m = \lfloor\sqrt{p}\rfloor$
- Regression default: $m = \lfloor p/3 \rfloor$
- Only the best split among these $m$ features is used

**Why**: If one feature is very strong, pure bagging produces correlated trees (all use that feature at the root). By forcing random subsets, dominant features are excluded from some nodes → trees become **decorrelated**.

### Step 3: Grow Full Unpruned Trees
- Grow each tree until minimum node size (e.g., 1 sample per leaf)
- No pruning → each tree has high variance, low bias
- **Acceptable** because averaging will remove the variance

### Step 4: Aggregate Predictions
$$\hat{f}_{RF}(x) = \frac{1}{B}\sum_{b=1}^B T_b(x) \quad \text{(regression)}$$
$$\hat{G}_{RF}(x) = \text{majority vote}\{T_b(x)\}_{b=1}^B \quad \text{(classification)}$$

**The variance formula**: For $B$ correlated trees with pairwise correlation $\rho$ and individual variance $\sigma^2$:
$$\text{Var}(\text{average}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B\to\infty$: $\rho\sigma^2$ remains. Random feature subsampling reduces $\rho$ → reduces this irreducible floor.

### Step 5: OOB Error Estimate (Free Validation)
- For observation $i$: predict using only the ~$B/3$ trees where $i$ was NOT in the bootstrap sample
- Aggregate these OOB predictions → OOB error
- OOB error $\approx$ LOO-CV error — no extra computation needed

### Step 6: Variable Importance
Two methods:
1. **Gini importance**: sum of impurity decrease for all splits on feature $j$ across all trees
2. **OOB permutation importance**: permute feature $j$ in OOB data, measure accuracy drop → more reliable, less biased toward high-cardinality features

---

## Key Properties

| Property | Random Forest |
|----------|--------------|
| Bias | Same as a single deep tree (RF does not increase bias) |
| Variance | $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ → reduced by large $B$ and small $\rho$ |
| Overfitting | Does NOT overfit as $B\to\infty$ (variance floor is $\rho\sigma^2$) |
| Interpretability | Low (black-box ensemble) |
| Parallelism | Fully parallelizable (trees are independent) |
| High-dim ($p\gg n$) | Works well — random subsampling handles irrelevant features |

---

## Comparison: RF vs Bagging vs Boosting

| | Plain Bagging | Random Forest | Boosting |
|--|--------------|--------------|---------|
| Base learner | Any (usually trees) | Deep trees | Shallow trees/stumps |
| Tree depth | Deep | Deep | Shallow |
| Feature selection | All features | Random $m$ features | All features |
| Sequential? | No | No | Yes |
| Reduces | Variance | Variance (more than bagging) | Bias |
| Correlation $\rho$ | Higher | Lower | N/A |
| Can overfit? | No | No | Yes (noisy data) |

---

## Additional Possible Exam Questions

**Q: Why does increasing $B$ beyond some point give diminishing returns?**
The variance formula shows the floor is $\rho\sigma^2$. Once $B$ is large enough that $(1-\rho)\sigma^2/B \approx 0$, adding more trees doesn't help. Typical values: $B=500$ is usually sufficient.

**Q: What is the effect of reducing $m$ (number of features per split)?**
Smaller $m$ → more randomization → lower $\rho$ → lower variance floor. But $m$ too small means each split ignores relevant features → increases bias. There is an optimal $m$ that balances this; the defaults ($\sqrt{p}$, $p/3$) are empirically validated.

**Q: Why is OOB error approximately equal to LOO-CV error?**
Each observation appears in roughly 63.2% of bootstrap samples → is OOB in ~36.8% of trees ≈ $B/e$ trees. Averaging $\approx B/e$ predictions is similar to averaging $N-1$ predictions in LOO-CV. The approximation is excellent for large $B$.

**Q: Can Random Forest handle missing values?**
Yes, via surrogate splits: if the primary split feature is missing, use a correlated backup feature. Alternatively, use median imputation before training.

**Q: What does a proximity matrix measure in RF?**
Two observations have high proximity if they end up in the same leaf across many trees. Proximity $p_{ij} = $ (# trees where $i$ and $j$ co-occur in leaf) / $B$. Used for outlier detection and data visualization — it measures similarity between **observations**, not variables.
