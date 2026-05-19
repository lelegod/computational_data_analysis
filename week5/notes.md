# Week 5 — Lecture Notes
## Computational Data Analysis (02582)

---

## Decision Trees / Regression Trees

### The Core Idea

The feature space is **recursively partitioned** into rectangular regions (called **intervals** $I$, or nodes/leaves). Within each region, the model predicts a single constant.

### What is Interval $I$?

An interval $I$ is a **region of the feature space** containing a subset of training points. In 1D, a single split at threshold $s$ produces:

$$I_1 = \{i \mid x_i < s\}, \qquad I_2 = \{i \mid x_i \geq s\}$$

In higher dimensions, each interval is a **hyperrectangle** defined by the sequence of splits made along each feature.

### Prediction Within an Interval

Within each interval, the prediction $\hat{y}$ is a **constant** — the mean of all response values in that region:

$$\hat{y}_I = \frac{1}{|I|} \sum_{i \in I} y_i$$

This is optimal because it minimises the Residual Sum of Squares (RSS) within the interval.

### RSS Within an Interval

$$\text{RSS}_I = \sum_{i \in I} (y_i - \hat{y}_I)^2$$

### What Makes a Good Split?

For each candidate split $(j, s)$ — feature $j$ at threshold $s$ — compute the total RSS across both resulting intervals:

$$\text{RSS}_{\text{total}} = \text{RSS}_{I_1} + \text{RSS}_{I_2}$$

The **best split** minimises $\text{RSS}_{\text{total}}$. The algorithm tries every feature and every threshold, choosing the globally best $(j, s)$.

This is then applied **recursively** inside each interval — that is how the tree grows.

### How the Tree Grows: The Full Algorithm

At each node, search over **all features $j = 1,\dots,p$** and **all thresholds $s$** (one per unique data value):

$$\text{Total candidates at each step} = p \times (n - 1)$$

Pick the $(j, s)$ that minimises $\text{RSS}_{I_1} + \text{RSS}_{I_2}$, then recurse independently into each child region:

```
All n points (1 region)
  → best split → 2 regions
      → best split in region 1 → 2 sub-regions
      → best split in region 2 → 2 sub-regions
          → ... until stopping criterion
```

**Stopping criteria** (any of):
- Maximum tree depth reached
- Minimum number of points per leaf
- RSS improvement below a threshold

---

## Pruning (Weakest-Link Pruning)

### Why Prune?

Growing a large tree overfits. But stopping early is also bad — a seemingly useless split may enable a great split below it. The solution: **grow large, then prune back**.

### The Pruning Rule

For each internal node, compute the **per-node reduction** in RSS from keeping its subtree vs. collapsing it to a leaf:

$$\text{Per-node reduction} = \frac{\text{RSS}_{\text{node}} - \sum_{\text{leaves}} \text{RSS}_{\text{leaf}}}{|\text{leaves}| - 1}$$

**Prune the node with the smallest per-node reduction** — it is the "weakest link."

### Example

```
Root (RSS=100)
├── Left (RSS=30)
│   ├── Leaf (RSS=15)
│   └── Internal (RSS=12)
│       ├── Leaf (RSS=6)
│       └── Leaf (RSS=2)
└── Right (RSS=50)
    ├── Leaf (RSS=25)
    └── Leaf (RSS=20)
```

Candidates (nodes whose direct children are both leaves):

| Node | RSS | Children sum | Reduction | Leaves−1 | Per-node reduction |
|---|---|---|---|---|---|
| RSS=12 | 12 | 6+2=8 | 4 | 1 | $4/1 = 4$ |
| RSS=50 | 50 | 25+20=45 | 5 | 1 | $5/1 = 5$ |

**Prune RSS=12** — smallest per-node reduction. It becomes a leaf. Then repeat on the updated tree.

### Intuition

> "Is this split worth the extra leaves it creates?" A split with low per-node reduction is wasteful — remove it.

---

## Classification Trees: Split vs. Prune Criteria

### Standard Practice

| Stage | Criterion | Why |
|---|---|---|
| **Growing** | Gini index | Sensitive to small impurity changes — gives a rich signal |
| **Pruning** | Misclassification rate | Directly measures prediction accuracy |

### Gini Index

$$G = \sum_{k=1}^{K} \hat{p}_k(1 - \hat{p}_k)$$

Measures **node impurity**. Low when one class dominates; high when classes are balanced. Used during tree growing because it differentiates finely between candidate splits.

**Why not misclassification rate for splitting?**

Misclassification rate only cares about the majority class label. If a split changes probabilities from 40/60 to 45/55, the predicted class stays the same and the misclassification rate does not change — yet the node is genuinely purer. Gini detects this; misclassification rate does not.

Example: two candidate splits on a 40A/60B node:

| Split | Left | Right | Miss. rate signal |
|---|---|---|---|
| A | 30A/70B | 50A/50B | barely changes |
| B | 45A/55B | 35A/65B | barely changes |

Both look identical to misclassification rate. Gini ranks them differently — providing the signal needed for the greedy search.

### Misclassification Rate

$$E = 1 - \max_k(\hat{p}_k)$$

Used during **pruning** because it directly answers: *"does this subtree reduce classification errors?"* A split that improves Gini but does not change any predictions does not deserve to stay in the final tree.
