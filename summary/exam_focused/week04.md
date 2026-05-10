# Week 4 — CART: Classification and Regression Trees (Exam Focus)

## Must-Know Facts

### General CART
- CART partitions feature space into axis-aligned rectangles; predicts a constant in each leaf.
- Prediction in a regression leaf = mean of training responses in that leaf.
- Prediction in a classification leaf = majority class in that leaf.
- Trees handle missing data via surrogate splits (a major advantage).
- Trees handle both continuous and categorical predictors natively.
- Deep trees = low bias, HIGH variance. Shallow trees = high bias, LOW variance.
- A small change in the training data can completely change the tree structure (instability = high variance).
- The growing algorithm is GREEDY — it does not look ahead beyond the current split.

### Splitting Criteria
- Regression trees use RSS to select splits.
- Classification trees use Gini index or cross-entropy (NOT misclassification rate) for growing.
- Misclassification rate IS used for pruning (where final class prediction matters).
- Gini and cross-entropy give similar trees in practice.
- Gini = 0 means a node is perfectly pure (all one class).
- Gini is maximised when classes are equally distributed.

### Pruning
- Cost-complexity pruning grows a full tree then prunes back.
- Larger $\alpha$ = smaller (more pruned) tree.
- $\alpha = 0$ = full unpruned tree.
- Cross-validation is used to choose the best $\alpha$.
- The result is a sequence of nested subtrees $T_0 \supset T_1 \supset \cdots \supset T_{\text{root}}$.
- Pre-pruning (stopping early) is worse than post-pruning because a bad-looking split now may enable a great split later.

### Categorical Variables
- For regression or binary classification with a categorical variable with $K$ levels: only $K-1$ ordered comparisons are needed (not all $2^{K-1} - 1$ subsets).
- Order by mean response (regression) or class 1 proportion (binary classification).

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $c_j = \text{mean}(y_i : x_i \in R_j)$ | Leaf prediction (regression) | Predict in regression leaf |
| $\text{RSS} = \sum (y_i - c_1)^2 + \sum (y_i - c_2)^2$ | Split criterion for regression | Choosing best split in regression tree |
| $\hat{p}_{mk} = \frac{1}{N_m} \sum \mathbf{I}(y_i = k)$ | Class proportion in leaf $m$ | Computing Gini, entropy, error |
| $G = \sum_k \hat{p}_{mk}(1 - \hat{p}_{mk})$ | Gini index | Split criterion for classification |
| $D = -\sum_k \hat{p}_{mk} \log(\hat{p}_{mk})$ | Cross-entropy / deviance | Alternative split criterion |
| $E = 1 - \max_k(\hat{p}_{mk})$ | Misclassification rate | Pruning evaluation |
| $C_\alpha(T) = \sum_m N_m Q_m(T) + \alpha \lvert T \rvert$ | Cost-complexity criterion | Pruning: balance fit vs. tree size |
| $\Delta I = I(\text{parent}) - \left[\frac{N_L}{N} I(\text{left}) + \frac{N_R}{N} I(\text{right})\right]$ | Impurity reduction at a split | Choosing best split (classification) |
| $VI_j = \sum_{\text{splits on } j} N_t \cdot \Delta I_t$ | Variable importance for feature $j$ | Interpreting which features matter |

---

## Common Traps (wrong answers in exams)

- Misclassification rate is used to GROW trees → Misclassification rate is NOT used to grow trees; Gini/entropy are used. Misclassification rate is insensitive to probability changes within a class.
- Bagging improves bias → Bagging does NOT improve bias; it only reduces variance. (Week 5 concept, but often confused here.)
- A deeper tree always has lower error → A deeper tree has lower TRAINING error but can have much higher TEST error (overfitting).
- CART cannot handle missing data → CART handles missing data via surrogate splits.
- CART assumes features are on the same scale → CART does NOT require scaling; splits are based on ordering, not distances.
- Gini and entropy give very different trees → They give very similar trees in practice; the choice rarely matters.
- For categorical variables with $K$ levels, CART must try all $2^{K-1} - 1$ subsets → For regression and binary classification, only $K-1$ ordered splits are needed.
- Cross-entropy and misclassification rate are equivalent for tree growing → Cross-entropy is more sensitive to probability changes and preferred for growing; misclassification rate is flat in large regions.
- $\alpha = 0$ gives the smallest tree → $\alpha = 0$ gives the LARGEST (full) tree. Larger $\alpha$ = smaller tree.
- Pre-pruning is better than post-pruning → Post-pruning (cost-complexity) is generally preferred because greedy early stopping can miss good splits.

---

## Quick Decision Rules

- If regression tree: split on RSS reduction, predict leaf mean.
- If classification tree: split on Gini or entropy reduction, predict majority class.
- If $\alpha$ increases: tree gets smaller (fewer leaves).
- If node is pure (all same class): Gini $= 0$, entropy $= 0$, misclassification $= 0$.
- If classes are equal (50/50 binary): Gini $= 0.5$, misclassification $= 0.5$, entropy $= \log(2)$.
- If a feature is missing at prediction: use surrogate split (next best split that agrees with primary).
- If asked which impurity measure to use for GROWING: Gini or cross-entropy (not misclassification).
- If asked which impurity measure to use for PRUNING/EVALUATION: misclassification rate.
- If asked about tree variance: trees have HIGH variance — small data changes cause large structural changes.
- If $n_{\min}$ is large: shallower tree, more bias, less variance.
