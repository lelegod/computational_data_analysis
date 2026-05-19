# Week 5 — CART: Classification and Regression Trees + Bagging

## Overview
This week introduces Classification and Regression Trees (CART) and Bagging (Bootstrap Aggregating). CART partitions the feature space into axis-aligned rectangles and fits a constant prediction in each leaf — using RSS for regression trees and Gini/cross-entropy for classification trees. Trees are grown greedily and pruned via cost-complexity pruning with CV. Bagging then uses bootstrapping to aggregate many trees, reducing variance without affecting bias.

---

## 1. CART: Classification Trees

### Key Concepts
- Classification trees partition the feature space into $J$ regions and predict the majority class in each region.
- The probability estimate for class $k$ in region $m$: $\hat{p}_{mk}$ = proportion of class-$k$ observations in region $m$.
- Splitting is done by maximising impurity reduction using Gini index or cross-entropy.

### Impurity Measures (Full Comparison)

| Measure | Formula | Properties |
|---------|---------|-----------|
| Misclassification rate | $E = 1 - \max_k(\hat{p}_{mk})$ | Not sensitive to probability shifts; used for pruning |
| Gini index | $G = \sum_k \hat{p}_{mk}(1 - \hat{p}_{mk})$ | Sensitive to probability shifts; preferred for growing |
| Cross-entropy | $D = -\sum_k \hat{p}_{mk} \log(\hat{p}_{mk})$ | Sensitive to probability shifts; preferred for growing |

- Gini and cross-entropy are differentiable and more sensitive to the class probability distribution.
- Misclassification rate does not change if a split moves probability mass within the majority class.
- For binary classification ($K=2$), with $p = \hat{p}_{m1}$:
  - Misclassification: $\min(p,\, 1-p)$
  - Gini: $2p(1-p)$
  - Entropy: $-p\log(p) - (1-p)\log(1-p)$

### CART for Multi-class Problems
- For $K$ classes, Gini and cross-entropy generalise naturally.
- The prediction in each leaf is the class with the highest proportion ($\arg\max_k \hat{p}_{mk}$).

---

## 2. Cost-Complexity Pruning (Detail)

### Key Concepts
- After growing a full tree $T_0$ (until $n_{\min}$ is reached), we prune using cost-complexity.
- **Cost-complexity criterion**: $C_\alpha(T) = R(T) + \alpha \cdot |T|$
  - $R(T)$ = total misclassification cost (or RSS for regression) $= \sum_{m=1}^{|T|} N_m \cdot Q_m$
  - $|T|$ = number of terminal nodes
  - $\alpha$ = regularisation parameter (complexity penalty per leaf)
- For each $\alpha$, there is a unique smallest subtree $T_\alpha$ that minimises $C_\alpha(T)$.
- As $\alpha$ increases from 0 to infinity, we get a sequence of nested subtrees: $T_0 \supset T_1 \supset \cdots \supset \text{root}$.
- The subtrees can be found efficiently using the weakest link pruning algorithm.

### Weakest Link Pruning
- For each internal subtree $t$, compute the effective $\alpha$ at which $t$ would be pruned:

$$\alpha_t = \frac{R(t) - R(T_t)}{|T_t| - 1}$$

  - where $R(t)$ = impurity if $t$ is a leaf, $R(T_t)$ = impurity of subtree rooted at $t$, $|T_t|$ = leaves in subtree
- Prune the subtree with the smallest effective $\alpha$ first.
- Repeat until only the root remains.

### Cross-Validation for alpha
1. Grow full tree $T_0$.
2. Find the sequence of subtrees $T_0 \supset T_1 \supset \cdots$ via weakest link pruning.
3. For each $\alpha$ (or equivalently each subtree), compute $K$-fold CV error.
4. Choose $\alpha^*$ minimising CV error.
5. Final model: grow $T_0$ on ALL training data, then prune to $T_{\alpha^*}$.

---

## 3. Bootstrapping

### Key Concepts
- Bootstrapping is a general statistical technique for estimating properties of an estimator by resampling.
- A bootstrap sample of size $N$ is drawn WITH REPLACEMENT from the original $N$ training observations.
- On average, each bootstrap sample contains approximately 63.2% of the unique original observations.
  - $P(\text{observation } i \text{ NOT in bootstrap sample}) = \left(1 - \frac{1}{N}\right)^N \to \frac{1}{e} \approx 0.368$ as $N \to \infty$
  - So about 36.8% of observations are left out (out-of-bag).
- The remaining ~36.8% of observations form the out-of-bag (OOB) sample for that bootstrap.

### Why Bootstrap Works
- The empirical distribution of the sample mimics the true population distribution $X$ is drawn from.
- Averaging predictions over many bootstrap models reduces variance (law of large numbers applied to predictions).

---

## 4. Bagging (Bootstrap Aggregating)

### Key Concepts
- Bagging = Bootstrap Aggregating; proposed by Leo Breiman (1996).
- Particularly good for high-variance, low-bias methods such as deep trees.
- Algorithm:
  1. Draw $B$ bootstrap samples from the training data.
  2. Fit a separate model (e.g., a CART tree) to each bootstrap sample.
  3. For regression: average the $B$ predictions.
  4. For classification: take majority vote across $B$ classifiers.

### Bagging Algorithm (Formal)
1. For $b = 1$ to $B$:
   a. Draw bootstrap sample $Z^*_b$ of size $N$ from training data (with replacement).
   b. Fit tree $T_b$ to $Z^*_b$ (grow to minimum node size, no pruning).
2. **Regression prediction**: $\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$
3. **Classification prediction**: majority vote of $\{T_1(x), T_2(x), \ldots, T_B(x)\}$

### Bagging Bias
- The bias of the bagged estimator equals the bias of any individual tree (since all trees are identically distributed):

$$E(\hat{y} - y) = E\!\left[\frac{1}{B} \sum_{b=1}^{B} (\hat{y}_b - y)\right] = \frac{1}{B} \sum_b E(\hat{y}_b - y) = E(\hat{y}_b - y)$$

- Bagging does NOT reduce bias.

### Bagging Variance
- The variance of the average of $B$ identically distributed (but correlated) random variables with variance $\sigma^2$ and pairwise correlation $\rho$:
- **Bagging variance**:

$$\text{Var}(\hat{y}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

  - where $\rho$ = pairwise correlation between any two trees
  - $\sigma^2$ = variance of a single tree's predictions
  - First term $\rho \sigma^2$ does NOT go to zero as $B$ increases.
  - Second term $\frac{1-\rho}{B} \sigma^2$ goes to zero as $B \to \infty$.
- As $B \to \infty$: $\text{Var} \to \rho \sigma^2$.
- The limiting factor for variance reduction is $\rho$ (the correlation between trees).
- If $\rho = 0$ (trees independent): $\text{Var} = \sigma^2 / B$ (full variance reduction).
- If $\rho = 1$ (all trees identical): $\text{Var} = \sigma^2$ (no variance reduction).

### Why Trees are Correlated in Bagging
- All trees are trained on slightly different versions of the same data.
- If one feature is very strong (dominant), all trees will use it near the root → high $\rho$.
- This limits the variance reduction achievable by bagging.

### Effect of B (Number of Trees)
- Increasing $B$ always helps (reduces the second term in variance formula).
- $B$ does not cause overfitting in bagging — using more trees is always safe.
- In practice, $B \sim 100$–500 is usually sufficient; error stabilises.
- The $\sin(x)$ example: 50 bootstraps of 9th-degree polynomial fits show the spread of individual models; their average is smoother and closer to the truth.

### Bagging for Regression vs. Classification
- **Regression**: average the predictions of $B$ trees.
- **Classification**: each tree votes for one class; predict the class with the most votes.
  - Can also average class probabilities from each tree instead of hard votes.

### Bagging Summary
- Good for high-variance, low-bias methods (deep unpruned trees).
- Reduces variance by averaging (law of large numbers).
- Does NOT reduce bias.
- Variance reduction is limited by inter-tree correlation $\rho$.
- $B$ increasing reduces variance (second term), but $\rho \sigma^2$ is a floor.

---

## 5. Out-of-Bag (OOB) Error Estimation

### Key Concepts
- Each bootstrap sample leaves out ~36.8% of the training observations (OOB observations).
- For each training observation $i$, use only the trees for which $i$ was OOB to predict $y_i$.
- Average these predictions to get the OOB prediction for observation $i$.
- **OOB error** = average prediction error computed using OOB predictions.
- OOB error approximates leave-one-out cross-validation error.
- OOB error is a free by-product of the bagging procedure (no extra CV needed).

---

## 6. Comparison: Individual Tree vs. Bagged Trees

| Property | Single CART | Bagged Trees |
|----------|------------|-------------|
| Bias | Low (deep tree) | Same as single tree |
| Variance | High | Lower (by factor up to $1/B$ if $\rho=0$) |
| Interpretability | High (readable rules) | Low (black box: average of many trees) |
| Overfitting risk | High | Low |
| Computational cost | Low | $B$ times higher |

---

## 7. Key Insight: Why Bagging Works on Trees

- Trees have HIGH variance and LOW bias.
- Averaging reduces variance without changing bias.
- Bagging exploits this by averaging many high-variance, low-bias trees.
- The result: a model with the same low bias but much lower variance.
- The bottleneck is $\rho$: correlated trees limit how much variance can be reduced.
- This motivates Random Forests (Week 6): decorrelate trees by random feature selection.
