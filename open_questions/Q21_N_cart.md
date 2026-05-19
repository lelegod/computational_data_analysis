# Q21-N — CART: Classification and Regression Trees
> Week 4. Building block for RF and boosting; could be asked for algorithm, pruning, or impurity measures.

---

## What CART Does

Recursively partition the feature space into axis-aligned rectangles. Each leaf predicts:
- **Regression**: mean of training responses in that region
- **Classification**: majority class (or class proportions)

The model is a piecewise constant function over the partition.

---

## Growing a Tree (Top-Down Greedy)

At each node, find the best split: feature $j$ and threshold $s$ that minimizes impurity.

**Regression** — minimize RSS:
$$\min_{j,s}\left[\sum_{x_i\in R_1(j,s)}(y_i-\bar{y}_{R_1})^2 + \sum_{x_i\in R_2(j,s)}(y_i-\bar{y}_{R_2})^2\right]$$

For fixed $j$: sort data on feature $j$, sweep through $N-1$ candidate splits → $O(Np)$ per node.

**Classification** — minimize impurity of child nodes. Three measures for node $m$ with class proportions $\hat{p}_{mk}$:

| Measure | Formula | Properties |
|---------|---------|-----------|
| Misclassification | $1 - \max_k\hat{p}_{mk}$ | Not differentiable, insensitive to probability changes |
| Gini index | $\sum_k\hat{p}_{mk}(1-\hat{p}_{mk})$ | Measures total variance across classes; preferred for splitting |
| Cross-entropy | $-\sum_k\hat{p}_{mk}\log\hat{p}_{mk}$ | More sensitive to impure nodes |

**In practice**: use Gini or entropy for splitting; use misclassification for pruning (matches final objective).

---

## When to Stop Growing

1. **Minimum node size**: stop if fewer than $n_\text{min}$ observations in a node
2. **Pure node**: all observations in a node have the same label → no impurity to reduce
3. **No improvement**: best split gives zero impurity reduction

In practice: grow the full tree (low bias, high variance), then prune.

---

## Cost-Complexity Pruning (Weakest Link Pruning)

Penalize tree complexity:
$$C_\alpha(T) = \sum_{m=1}^{|T|} N_m Q_m(T) + \alpha|T|$$

where $|T|$ = number of leaves, $N_m Q_m$ = impurity × size at leaf $m$, $\alpha \geq 0$ = complexity penalty.

**Algorithm**:
1. Grow full tree $T_\text{max}$
2. For each internal node $t$: compute "weakest link" = gain per leaf from the subtree rooted at $t$
3. Collapse the weakest link (node giving smallest gain per leaf added)
4. Repeat, producing a nested sequence $T_\text{max} \supset T_1 \supset T_2 \supset \cdots \supset \{\text{root}\}$
5. Choose $\alpha$ (and hence the best subtree) by cross-validation

**Why nested?** Each pruned subtree is optimal for some range of $\alpha$ values. The sequence is finite and can be enumerated efficiently.

---

## Categorical Variables

CART handles categorical variables with $q$ levels by considering $2^{q-1}-1$ possible splits (all binary partitions of the levels). For ordered categoricals: treat as numeric. For unordered: exhaustive search is exponential — use greedy heuristics for large $q$.

---

## Key Properties

| Property | CART |
|----------|------|
| Interpretable? | Yes (single tree easy to visualize) |
| Handles mixed types? | Yes (numeric + categorical naturally) |
| Missing values? | Via surrogate splits |
| Bias | Low (unpruned) / High (heavily pruned) |
| Variance | High (unstable — small data changes → different tree) |
| Overfits? | Yes — must prune or use ensemble |
| Scale invariant? | Yes (splits use $>$/$\leq$, not distances) |

**Instability**: this is CART's main weakness. A small change in training data can produce a completely different tree topology. This is why bagging and random forests (averaging many trees) dramatically reduce variance.

---

## CART vs Linear Models

| | Linear Model | CART |
|--|-------------|------|
| Decision boundary | Linear (hyperplane) | Axis-aligned steps |
| Interactions | Must engineer manually | Captured automatically |
| Interpretability | Coefficient table | Tree diagram |
| Extrapolation | Can extrapolate | Predicts training leaf mean (no extrapolation) |
| Feature selection | Implicit (zero coeff) | Implicit (unused features never split on) |

---

## Additional Possible Exam Questions

**Q: Why is Gini preferred over misclassification for splitting?**
Gini is differentiable and more sensitive to changes in class proportions. Consider a node with 100 points split equally (50/50): misclassification rate = 0.5 for both a pure split (100/0) and an impure split (49/51). Gini distinguishes these. Gini favors pure nodes more aggressively → better tree structure.

**Q: What is the relationship between CART and random forests?**
Random Forest = bagging on deep (unpruned) CART trees + random feature subsampling. The instability of CART (high variance) is exactly what makes bagging effective: averaging many uncorrelated unstable trees gives a stable low-variance predictor. You want base learners with high variance for bagging.

**Q: How does CART handle regression with a continuous response?**
At each node, predict the mean of training responses in that region. The prediction for a new point $x$ is $\hat{f}(x) = \bar{y}_{R(x)}$ where $R(x)$ is the leaf region containing $x$. Split criterion minimizes total RSS across the two child nodes. Pruning uses RSS + $\alpha|T|$.

**Q: Can CART model linear relationships efficiently?**
No — a linear relationship $y = \beta x$ requires many axis-aligned splits to approximate. Linear models are far more efficient for linear signal. CART excels when: (1) interactions exist between variables, (2) the relationship is piecewise constant, (3) variable types are mixed.

**Q: What is a surrogate split in CART?**
When the primary split variable is missing for a test point, use a "surrogate" — another variable that produces a similar partition to the primary split. Surrogates are identified during training by finding splits on other variables that agree most with the primary split's partition. This allows CART to handle missing values without imputation.
