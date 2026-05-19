# Week 5 — Group Discussion Questions

Topics: CART (Classification and Regression Trees), Bootstrap, Bagging

---

## Q1: How Many Splits for a Categorical Variable with k Categories?

**Question (slide 24):** The slides count splits for 2-category and 3-category inputs. How many distinct splits are there for a categorical variable with $k$ categories?

**Answer:**

For a categorical variable with $k$ categories, a split partitions the $k$ categories into two non-empty groups (left child and right child).

**Counting:**

The total number of ways to partition $k$ items into two non-empty subsets, where the two groups are unordered (left/right is symmetric), is:

$$\text{number of splits} = \frac{2^k - 2}{2} = 2^{k-1} - 1$$

**Why:** There are $2^k$ ways to assign each category to one of two groups (each category gets a binary label). We subtract 2 for the two all-one-side cases (empty groups not allowed), then divide by 2 because swapping left and right gives the same split.

**Verification with the examples from the slides:**

| $k$ | Formula $2^{k-1}-1$ | Examples given in slide |
|-----|---------------------|-------------------------|
| 2 | $2^1 - 1 = 1$ | (male) vs (female) — 1 split |
| 3 | $2^2 - 1 = 3$ | (apple) vs (orange, banana); (orange) vs (apple, banana); (banana) vs (apple, orange) — 3 splits |
| 4 | $2^3 - 1 = 7$ | — |
| 10 | $2^9 - 1 = 511$ | — |

**Practical implication:** For variables with many categories, the number of candidate splits grows exponentially ($O(2^k)$), which can make tree growing computationally expensive. In practice:
- For continuous variables: $n_l - 1$ splits for $n_l$ observations in a node (linear in data size)
- For categorical variables: up to $2^{k-1} - 1$ splits (exponential in number of categories)

This is why high-cardinality categorical variables can be problematic in CART, and why one-hot encoding or grouping is sometimes preferred.

---

## Q2: What Do You Think of This Fit?

**Question (slide 33):** The tree is grown with the stopping rule "do not split nodes with 10 or fewer observations." Looking at the resulting fit on the training data, what do you think of it?

**Answer:**

The fit shown is a severely **overfit** tree.

**Signs of overfitting:**

1. **Too many splits:** The tree has a large number of internal nodes and terminal nodes — it has memorised fine-grained patterns in the training data.
2. **Jagged/blocky predictions:** The piecewise-constant fit has many narrow steps, following the noise in the training data rather than the underlying smooth signal.
3. **High variance:** Small changes in the training data would produce a very different tree structure. The tree is highly unstable.

**Why this happens:**

A minimum node size of 10 is too permissive for this dataset — it allows the tree to keep splitting as long as there are more than 10 observations, growing a nearly full tree that overfits the training data.

**Bias-variance perspective:**

- A deep tree has **low bias** (can represent complex functions) but **high variance** (sensitive to training data noise)
- A shallow tree has **high bias** (underfits) but **low variance** (stable across datasets)
- The tree shown is at the high-variance end of this spectrum

**Solution — pruning:**

Rather than stopping early, the correct approach is to:
1. Grow the tree very large (aggressive splitting)
2. **Prune** it back using cost-complexity (weakest-link) pruning
3. Select the pruned tree size using cross-validation

This avoids the problem of early stopping missing important splits that only become useful after a "seemingly worthless" intermediate split.

---

## Q3: Where Do We Prune?

**Question (bee slide, slide ~44):** Given a grown tree with RSS values at each node, which node(s) should be pruned first using the weakest-link pruning rule?

**Answer:**

**Weakest-link pruning rule:** At each step, prune the non-terminal node whose subtree gives the **smallest per-node reduction in RSS**:

$$\text{per-node reduction} = \frac{RSS_{\text{node}} - RSS_{\text{subtree}}}{\text{(number of terminal nodes in subtree)} - 1}$$

**Working through the example from the slides** (tree with RSS values):

The tree shown has:
- Root: RSS = 100
- Left child: RSS = 30; Right child: RSS = 50
- Left-left leaf: RSS = 15
- Left-right internal node: RSS = 12 (with children RSS = 6, RSS = 2)
- Right-left leaf: RSS = 25; Right-right leaf: RSS = 20

**Candidate prunings:**

1. **Prune left-right subtree** (node with RSS = 12, children RSS = 6 and RSS = 2):
   - Reduction = $12 - (6 + 2) = 4$
   - Number of terminal nodes gained back = 1 (the 2 leaves become 1 leaf)
   - Per-node reduction = $4 / (2-1) = 4$

2. **Prune right subtree** (node with RSS = 50, children RSS = 25 and RSS = 20):
   - Reduction = $50 - (25 + 20) = 5$
   - Per-node reduction = $5 / (2-1) = 5$

3. **Prune left subtree** (node with RSS = 30, children RSS = 15 and subtree RSS = 6+2=8):
   - This subtree has 3 terminal nodes (RSS=15, RSS=6, RSS=2)
   - Reduction = $30 - (15 + 6 + 2) = 7$
   - Per-node reduction = $7 / (3-1) = 3.5$

**Order of pruning (weakest link first):**
1. First prune the left subtree (per-node reduction = 3.5) — smallest value
2. Then prune the left-right node (per-node reduction = 4)
3. Then prune the right subtree (per-node reduction = 5)

**Key insight:** We always prune the subtree that yields the least RSS improvement per node removed — this gives us a sequence of nested trees, and we use cross-validation to pick the best one.

---

## Q4: Bias and Variance of Regression Trees

**Question (slide ~57):** What do we conclude about regression trees in terms of bias and variance? How does tree depth relate to the bias-variance trade-off?

**Answer:**

**Regression trees exhibit a clear bias-variance trade-off controlled by tree depth (or equivalently, the pruning parameter $\alpha$):**

| Tree size | Bias | Variance | Test error |
|-----------|------|----------|------------|
| Very deep (large) | Low | High | High (overfitting) |
| Very shallow (small, e.g. stump) | High | Low | High (underfitting) |
| Optimally pruned | Medium | Medium | Lowest |

**Why deep trees have high variance:**

A deep tree memorises the training data — each terminal node may contain only a handful of observations. The node prediction (the mean of those observations) is a noisy estimate of the true conditional mean. Small perturbations in the training set lead to completely different tree structures.

Formally, the variance of the prediction at a terminal node with $n_l$ observations is $\sigma^2 / n_l$ — as nodes become smaller, variance grows.

**Why deep trees have low bias:**

A deep tree can represent arbitrarily complex piecewise-constant functions. With enough splits it can fit any training set exactly (zero training error), meaning near-zero bias.

**Why shallow trees have high bias:**

A single split (stump) can only represent two constant values — a very restricted function class. It cannot capture nonlinear or interactive relationships.

**Conclusion from the lecture:** Unpruned regression trees are **high-variance, low-bias** models. This is why:
- **Bagging** works well on trees: averaging many trees reduces variance without increasing bias
- **Boosting** uses shallow trees (stumps): each tree is high-bias, but sequential fitting reduces bias
- **Pruning** is needed for a single tree to achieve acceptable generalisation

**The empirical observation from slide ~57:** When we plot test error vs tree size, we see a U-shape — error decreases as we add splits (bias reducing), then increases as the tree overfits (variance dominating). The optimal tree size is at the bottom of this curve, often selected by the 1-SE rule in cross-validation.
