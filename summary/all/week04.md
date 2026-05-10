# Week 4 — Classification and Regression Trees (CART)

## Overview
This week introduces Classification and Regression Trees (CART), a non-parametric method that partitions the feature space into rectangular regions and fits a constant model in each region. The lecture covers the mathematical foundations of splitting criteria, tree-growing algorithms, pruning strategies, and handling of missing data.

---

## 1. What is a Decision Tree?

### Key Concepts
- A tree model partitions the input space into a set of rectangles, then fits a simple constant model (the mean for regression, the majority class for classification) in each rectangle.
- Each internal node applies a binary test on a single feature: $X_j \leq s$ (a split).
- Each leaf node contains a constant prediction.
- Trees are non-parametric: no assumption about the shape of the decision boundary.
- Trees are highly interpretable: the model is a set of readable if-then-else rules.
- Trees handle both continuous and categorical predictors naturally.
- Trees handle missing data naturally (surrogate splits).

### Advantages
- Interpretability: rules are easy to follow.
- Handles missing data.
- Can handle both continuous and categorical variables.
- Automatically performs variable selection.
- No need to scale/normalise features.

### Disadvantages
- High variance: small changes in data can completely change the tree structure.
- Deep trees overfit (high variance, low bias).
- Small trees underfit (low variance, high bias).
- Piecewise-constant predictions are not smooth.

---

## 2. Regression Trees

### Key Concepts
- Goal: predict a continuous response $Y$ from features $X$.
- We partition the $p$-dimensional feature space into $J$ disjoint regions $R_1, R_2, \ldots, R_J$.
- In each region $R_j$, the prediction is the mean of the training responses in that region.

### Prediction Formula
- **Regression tree prediction**: $f(x) = \sum_{j=1}^{J} c_j \cdot \mathbf{I}(x \in R_j)$
  - where $c_j = \text{mean}(y_i : x_i \in R_j)$ (the average response in region $j$)
  - $\mathbf{I}(x \in R_j) = 1$ if $x$ falls in region $j$, $0$ otherwise

### Splitting Criterion: RSS (Residual Sum of Squares)
- At each split, we choose the feature $j$ and threshold $s$ to minimise total RSS:
- **RSS split criterion**:

$$\text{RSS} = \sum_{i:\, x_i \in R_1(j,s)} (y_i - c_1)^2 + \sum_{i:\, x_i \in R_2(j,s)} (y_i - c_2)^2$$

  - where $c_1 = \text{mean}(y_i : x_i \in R_1)$, $c_2 = \text{mean}(y_i : x_i \in R_2)$
  - $R_1(j,s) = \{X \mid X_j \leq s\}$, $R_2(j,s) = \{X \mid X_j > s\}$
- We scan all features $j$ and all split points $s$, pick the $(j, s)$ pair that gives the lowest RSS.
- This is a greedy algorithm: we do not look ahead to future splits.

### Algorithm: Tree Growing (CART)
1. Start at the root with all training data.
2. For each feature $j$ and each possible split value $s$:
   - Compute the RSS reduction from splitting on $(j, s)$.
3. Choose the $(j, s)$ that minimises RSS (or maximises impurity reduction).
4. Split data into two child nodes.
5. Repeat recursively for each child node.
6. Stop when a stopping criterion is met (e.g., minimum node size $n_{\min}$, maximum depth).

---

## 3. Classification Trees

### Key Concepts
- Goal: predict a categorical class label $Y$ from features $X$.
- In each leaf region, the prediction is the majority class (the most common class in that leaf).
- We need a different splitting criterion since RSS is not appropriate for classification.

### Impurity Measures
Three common measures of node impurity (all should be minimised):

#### Misclassification Rate
- **Misclassification rate**: $E = 1 - \max_k(\hat{p}_{mk})$
  - where $\hat{p}_{mk} = \frac{1}{N_m} \sum_{x_i \in R_m} \mathbf{I}(y_i = k)$ = proportion of class $k$ in region $m$
  - $N_m$ = number of observations in region $m$
  - Not differentiable, not used much for growing trees.

#### Gini Index (preferred for tree growing)
- **Gini index**: $G = \sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk})$
  - Measures total variance across classes.
  - Equals 0 when a node is pure (all one class).
  - Maximum when classes are equally distributed.
  - Can be written as: $G = 1 - \sum_k \hat{p}_{mk}^2$

#### Cross-Entropy / Deviance
- **Cross-entropy**: $D = -\sum_{k=1}^{K} \hat{p}_{mk} \log(\hat{p}_{mk})$
  - Also called information gain when used as splitting criterion.
  - Has similar properties to Gini; both are differentiable and more sensitive to class proportions than misclassification rate.

### Splitting Rule for Classification
- Find the split $(j, s)$ that maximises the reduction in impurity:
- **Impurity reduction**:

$$\Delta I = I(\text{parent}) - \left[\frac{N_L}{N} \cdot I(\text{left}) + \frac{N_R}{N} \cdot I(\text{right})\right]$$

  - where $N_L$, $N_R$ = number of observations in left/right child
  - $I(\cdot)$ = chosen impurity measure (Gini or cross-entropy)

### Prediction in leaves
- In each leaf, predict the majority class: $k^* = \arg\max_k \hat{p}_{mk}$
- Probabilities can also be read off: $P(\text{class}=k \mid x \in R_m) = \hat{p}_{mk}$

---

## 4. Tree Size and Pruning

### The Bias-Variance Trade-off in Trees
- Large/deep trees: low bias, high variance (overfit).
- Small/shallow trees: high bias, low variance (underfit).
- Optimal tree size is somewhere in between.

### Stopping Criteria (Pre-pruning)
- Stop splitting when node size $< n_{\min}$ (minimum node size).
- Stop when impurity reduction $<$ threshold.
- Stop at maximum depth.
- Problem: a split that looks poor now might enable a very good split later (greedy limitation).

### Cost-Complexity Pruning (Post-pruning, preferred)
- Grow a large tree $T_0$ (until minimum node size is reached).
- Then prune back using cost-complexity criterion.
- **Cost-complexity criterion**:

$$C_\alpha(T) = \sum_{m=1}^{|T|} N_m \cdot Q_m(T) + \alpha \cdot |T|$$

  - $|T|$ = number of terminal nodes (leaves)
  - $Q_m(T)$ = node impurity (e.g., RSS$/N_m$ for regression, Gini for classification)
  - $\alpha \geq 0$ = complexity penalty (tuning parameter)
  - When $\alpha = 0$: full tree is selected (no penalty).
  - As $\alpha$ increases: smaller trees are preferred.
- For each $\alpha$, there is a unique smallest subtree $T_\alpha$ that minimises $C_\alpha(T)$.
- A sequence of nested trees is produced as $\alpha$ increases: $T_0 \supset T_1 \supset T_2 \supset \cdots \supset T_{\text{root}}$.

### Selecting alpha via Cross-Validation
1. Grow full tree $T_0$ on training data.
2. For each value of $\alpha$ in the sequence:
   - Obtain the corresponding pruned subtree $T_\alpha$.
   - Estimate prediction error via $K$-fold cross-validation.
3. Choose $\alpha^*$ that gives the lowest CV error.
4. Return $T_{\alpha^*}$ grown on all training data.

---

## 5. Missing Data Handling: Surrogate Splits

### Key Concepts
- When a feature used for splitting is missing in a test observation, CART uses surrogate splits.
- A surrogate split is an alternative split variable that best mimics the primary split.
- At training time, for each primary split $(j, s)$, a list of surrogate splits is learned.
- At prediction time, if feature $j$ is missing, use the best available surrogate.
- This is a major advantage over many other methods.

### How surrogates are ranked
- Rank surrogate splits by how well they agree with the primary split on training data.
- Agreement = proportion of observations sent to the same child by both splits.

---

## 6. Categorical Predictors in CART

### Key Concepts
- For a categorical predictor with $K$ levels, CART can consider all $2^{K-1} - 1$ possible splits (all subsets of levels assigned to left/right child).
- This is computationally expensive for large $K$.
- For regression and binary classification: an ordering trick reduces this to $K-1$ comparisons.
  - For regression: order levels by their mean response value, then split on this ordering.
  - For binary classification: order levels by proportion of class 1, then split.

---

## 7. The CART Algorithm Summary

### Complete Algorithm (Regression)
1. Start with all training data at the root.
2. **Grow**: Recursively apply binary splits minimising RSS until $n_{\min}$ is reached. (No pruning yet.)
3. **Prune**: Apply cost-complexity pruning to generate sequence of subtrees $T_\alpha$.
4. **Select**: Use $K$-fold CV to choose the best $\alpha$, and hence the best subtree.
5. **Predict**: Drop new observations down the selected tree; return leaf mean (regression) or majority class (classification).

### Algorithm Complexity
- At each node, scanning all features and split points: $O(N \cdot p)$ per node.
- Total tree growth: $O(N \cdot p \cdot \log N)$ approximately.

---

## 8. Interpreting Trees

### Variable Importance
- For each feature $j$, accumulate the total RSS (or impurity) reduction across all splits that use feature $j$, weighted by the number of observations at that node.
- **Variable importance for feature $j$**:

$$VI_j = \sum_{\text{splits using } j} N_t \cdot \Delta I_t$$

  - where $N_t$ = number of observations at node $t$, $\Delta I_t$ = impurity reduction at node $t$
- Higher $VI$ means the variable contributes more to prediction.

### Reading a Tree
- Each internal node shows the split rule (e.g., $X_1 \leq 0.5$).
- Left branch = TRUE ($X_1 \leq 0.5$), Right branch = FALSE.
- Leaf nodes show: prediction value (regression mean or class label), number of observations, class proportions.

---

## 9. Comparison to Other Methods

| Property | CART | Linear Regression | KNN |
|----------|------|-------------------|-----|
| Interpretable | Yes | Yes | No |
| Handles missing data | Yes | No | No |
| Handles categorical | Yes | Needs dummies | No |
| Linear boundaries | No | Yes | No |
| Variance | High | Low | Medium |
| Bias | Low (deep) / High (shallow) | High (if misspecified) | Medium |

---

## 10. Key Numbers and Defaults
- Typical minimum node size: $n_{\min} = 5$ (regression), $n_{\min} = 1$ (classification)
- Gini and cross-entropy give similar results in practice
- Misclassification rate is NOT used for growing trees (insensitive to probability changes)
- Misclassification rate IS used for pruning (since we care about final predictions)
