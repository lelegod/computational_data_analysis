# Week 6 — Ensemble Methods: Random Forests and Boosting

## Overview
This week covers the two main tree-based ensemble methods beyond bagging: Random Forests (which decorrelate bagged trees by random feature subsampling) and Boosting (which sequentially builds weak learners with adaptive reweighting to reduce bias). The lecture also covers additive models, forward stagewise fitting, AdaBoost.M1, gradient boosting, and loss function choices.

---

## 1. Tree-Based Ensemble Methods: Overview

The tree-based ensemble hierarchy:
- **CART** (single tree)
  - **Bagging** (bootstrap + average, uses bootstrapping)
    - **Random Forests** (bagging + random feature subsets, uses bootstrapping)
  - **Boosting** (sequential reweighting)

---

## 2. Recap: Bagging Bias and Variance

### Bagging Bias
- The bias of the bagged estimator = bias of a single tree (trees are identically distributed):

$$E(\hat{y} - y) = E\!\left[\frac{1}{B} \sum_{b=1}^{B} (\hat{y}_b - y)\right] = \frac{1}{B} \sum_b E(\hat{y}_b - y) = E(\hat{y}_b - y)$$

- Bagging does NOT reduce bias.

### Bagging Variance

$$\text{Var} = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

  - $\rho$ = pairwise correlation between trees, $\sigma^2$ = variance of one tree
- As $B \to \infty$: $\text{Var} \to \rho \sigma^2$.
- The inter-tree correlation $\rho$ is what limits variance reduction.
- Drawback of bagging: variance is limited by $\rho$; bagging does nothing to the bias.

---

## 3. Random Forests

### Key Concepts
- Random Forests (RF) are a refinement of bagging that decorrelates the trees.
- By reducing $\rho$, RF reduces the limiting floor ($\rho \sigma^2$) in the variance formula.
- RF is simple to tune and train; many implementations exist.
- Trees in RF are independent of each other → can be parallelised easily.
- RF is popular and often the best out-of-the-box algorithm.

### The Key Ingredient: Random Feature Subsampling
- At each split in each tree, instead of considering all $p$ features, only a random subset of $m < p$ features is considered as candidates for splitting.
- This decorrelates the trees: different trees will use different features at each split.
- If a strong predictor exists, in bagging it dominates the top of all trees → high $\rho$.
- In RF, strong predictors are excluded from some splits → trees differ more → lower $\rho$ → lower variance.

### Random Forest Algorithm
1. Define number of trees $B$ (typically a few hundred; overfitting is not a problem).
2. For $b = 1$ to $B$:
   a. Draw a bootstrap sample of size $N$ from training data (with replacement).
   b. Grow a tree by repeating until minimum node size $n_{\min}$ is reached (DO NOT prune):
      - Take a random sample WITHOUT REPLACEMENT of $m$ features ($m < p$).
      - Find the best split among those $m$ candidate features.
      - Split the node.
3. Output: $B$ trees.

### Prediction with Random Forest
- **Classification**: drop $x$ down each of the $B$ trees; take majority vote.
- **Regression**: $\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$
  - where $T_b(x)$ = prediction of $b$-th tree for input $x$

### Hyperparameter m (Number of Features per Split)
- $m$ is the main tuning parameter of RF.
- Default heuristics (rules of thumb):
  - **Classification**: $m = \lfloor\sqrt{p}\rfloor$
  - **Regression**: $m = \lfloor p/3 \rfloor$
- These should be tuned using OOB error; they depend on the problem.
- When $m = p$: Random Forest reduces to Bagging (no decorrelation).
- Smaller $m$ → more decorrelation → lower $\rho$ → lower variance, but potentially higher bias (fewer good features considered at each split).

### Effect of m on Performance
- In the sand data set example: RF with small $m$ outperforms bagging ($m=p$) in terms of OOB MSE.
- As $B$ (number of trees) increases, all RF variants converge; more trees are always better.
- Small $m$: lower variance due to decorrelation, converges to lower error floor.
- Large $m$ (bagging): higher variance due to higher $\rho$, converges to higher error floor.

### Connection to Ridge Regression
- Both bagging/RF and ridge regression handle $p \gg n$ (more variables than observations) problems.
- Ensemble averaging in RF reduces the contribution of any one variable, similar to shrinkage in ridge.
- This effect is especially strong when $m$ is small.
- RF works with $p > n$ (more variables than observations), though can have problems in very high-dimensional settings with many noise variables.

### OOB Error Estimation in Random Forest
- Samples not in each bootstrap sample = out-of-bag (OOB) samples.
- For each observation $i$, use only trees for which $i$ was OOB to predict $i$.
- OOB error = average prediction error using OOB predictions.
- Results are similar to cross-validation.
- Can monitor OOB error as the forest grows: stop when it plateaus.
- OOB error vs. test error: very similar curves (OOB is slightly pessimistic).

### Model Selection Using OOB
- As forest grows ($B$ increases), assess OOB error.
- Stop when OOB error no longer decreases.
- This provides a free estimate of generalisation error during training.
- Pitfall: Cannot directly compare OOB error of RF with CV error of another method (different estimation procedures).

---

## 4. Variable Importance in Random Forests

### Two Measures of Variable Importance

#### 1. Gini Importance
- For each tree, at each split using feature $j$, record the impurity reduction (Gini decrease) weighted by the number of observations.
- Sum these over all trees and all splits on feature $j$.
- **Gini importance for $j$**:

$$VI_j^{\text{Gini}} = \sum_{\text{trees}} \sum_{\text{splits on } j} N_t \cdot \Delta\text{Gini}_t$$

- Tends to give very large importance to a few top variables (concentrated distribution).

#### 2. OOB Permutation Importance
- For each tree $b$, using the OOB sample for that tree:
  1. Compute OOB prediction accuracy with original feature values.
  2. Permute (randomly shuffle) the values of feature $j$ among OOB observations.
  3. Compute OOB prediction accuracy again.
  4. $\Delta$ error = accuracy loss due to permuting feature $j$.
- Average $\Delta$ error over all trees = variable importance for $j$.
- Interpretation: how much does prediction accuracy drop when we randomise feature $j$?
- OOB permutation importance tends to spread importance more uniformly than Gini.
- Rankings from both methods are usually similar.

### Using Variable Importance for Feature Selection
- After computing OOB permutation importance, select features with importance above a threshold.
- Refit RF with only selected features → often improves performance and reduces computation.
- Example: sand data set — after selecting important features, fewer trees are needed for good OOB MSE.

---

## 5. Proximity Matrix in Random Forests

### Key Concepts
- An $n \times n$ proximity matrix $P$ is built from the random forest.
- $P(i, j)$ is incremented by 1 every time observations $i$ and $j$ both appear in the OOB sample of a tree AND end up in the same terminal node.
- Large $P(i, j)$ means observations $i$ and $j$ are similar according to the RF classifiers.
- The proximity matrix can be visualised using Multidimensional Scaling (MDS): find a 2D embedding that preserves pairwise distances.
- Proximity plots look like stars: observations far from the decision boundary are in the extremities; observations near the boundary are near the center.

---

## 6. Boosting

### Core Idea
- Boosting builds an ensemble of weak classifiers/learners sequentially.
- Each new learner focuses on observations that previous learners got wrong.
- Unlike bagging (parallel, averaging), boosting is sequential and adaptive.
- Boosting reduces BIAS (and also variance), not just variance.
- Weak learners = classifiers only slightly better than random (e.g., stumps: trees with one split).
- Combine weak learners via a weighted majority vote into a strong classifier.
- Boosting dominates bagging on most problems.
- Caveat: Boosting is not known for overfitting but CAN overfit (especially with noisy data).

### Bagging vs. Boosting: Key Difference
| Property | Bagging | Boosting |
|----------|---------|---------|
| What it reduces | Variance | Bias (and variance) |
| Tree type used | Deep trees (low bias) | Shallow trees/stumps (high bias) |
| Combination | Average / majority vote | Weighted vote |
| Sequential? | No (parallel) | Yes (sequential) |
| Observations reweighted? | Bootstrap resampling | Adaptive weights on observations |
| Trees independent? | Yes | No (each tree depends on previous) |
| Overfitting risk | Low | Can overfit (especially with noise) |

---

## 7. AdaBoost.M1

### Key Concepts
- Most popular boosting algorithm for binary classification.
- Proposed by Freund and Schapire (1997).
- For two-class problems with labels $y_i \in \{-1, +1\}$.
- Empirically motivated; optimal properties are difficult to show theoretically.
- The final classifier is: $G(x) = \text{sign}\!\left[\sum_{m=1}^{M} \alpha_m G_m(x)\right]$
  - $G_m(x) \in \{-1, +1\}$ = $m$-th weak classifier
  - $\alpha_m$ = weight for the $m$-th classifier (larger for more accurate classifiers)

### AdaBoost.M1 Algorithm (Full Steps)
1. Initialise observation weights: $w_i = \frac{1}{N}$ for $i = 1, 2, \ldots, N$.
2. For $m = 1$ to $M$:
   a. Fit classifier $G_m(x)$ to training data using weights $w_i$.
   b. Compute weighted error of $G_m$:

$$\text{err}_m = \frac{\sum_{i=1}^{N} w_i \cdot \mathbf{I}(y_i \neq G_m(x_i))}{\sum_{i=1}^{N} w_i}$$

   c. Compute classifier weight:

$$\alpha_m = \log\!\left[\frac{1 - \text{err}_m}{\text{err}_m}\right]$$

   d. Update observation weights:

$$w_i \leftarrow w_i \cdot \exp\!\left[\alpha_m \cdot \mathbf{I}(y_i \neq G_m(x_i))\right]$$

      Renormalise $w_i$ to sum to 1.
3. Output: $G(x) = \text{sign}\!\left[\sum_{m=1}^{M} \alpha_m G_m(x)\right]$

### Key Properties of AdaBoost
- Misclassified observations get higher weights → future classifiers focus on hard cases.
- If $\text{err}_m = 0.5$ (random): $\alpha_m = 0$ (no contribution).
- If $\text{err}_m = 0$ (perfect): $\alpha_m \to \infty$.
- If $\text{err}_m > 0.5$ (worse than random): $\alpha_m < 0$ (classifier is negated).
- The weights $\alpha_m$ weight better classifiers more heavily in the final vote.
- Shrinkage (learning rate): contribution of each tree can be scaled by factor $0 < \nu < 1$ to slow down learning and improve generalisation.

### Shrinkage in Boosting (Learning Rate)

$$F_m(x) = F_{m-1}(x) + \nu \cdot f_m(x), \quad 0 < \nu < 1$$

- Smaller $\nu$: each tree contributes less; need more trees $M$ to converge; better generalisation.
- Works analogously to ridge regularisation.

---

## 8. Boosting and Additive Models

### Additive Model Framework
- AdaBoost can be viewed as fitting an additive model:

$$F(x) = \sum_{m=1}^{M} \beta_m \cdot b(x;\, \gamma_m)$$

  - $b(x;\, \gamma_m)$ = base learner (tree), parameterised by splits $\gamma_m$
  - $\beta_m$ = weight (contribution) of the $m$-th tree
- Traditionally, all parameters would be fit jointly. Boosting uses forward stagewise fitting instead.

### Forward Stagewise Additive Modelling
- Fit one basis function at a time; previously fit functions are fixed.
- Works as regularisation (slows overfitting because past contributions are not adjusted).
- **Forward Stagewise Algorithm (Regression)**:
  1. Start with $F_0(x) = 0$ and residual $r = y$, $m = 0$.
  2. Repeat $M$ times:
     a. $m = m + 1$
     b. Fit a CART regression tree $g(x)$ to the residual $r$.
     c. Set $f_m(x) = \varepsilon \cdot g(x)$ (shrink with $\varepsilon$).
     d. Update: $F_m(x) = F_{m-1}(x) + f_m(x)$, $r = r - f_m(x)$.

### AdaBoost as Stagewise Modelling with Exponential Loss
- ESL Theorem 10.4: AdaBoost is equivalent to forward stagewise additive modelling with the exponential loss function:
- **Exponential loss**: $L(y,\, F(x)) = \exp(-y \cdot F(x))$
- AdaBoost builds an additive logistic regression model: $F(x) = \log\!\left[\frac{P(y=1\mid x)}{P(y=-1\mid x)}\right]$
- The exponential loss leads to reweighting original data (instead of fitting residuals).

### Probabilities from AdaBoost
- From the connection to logistic regression:

$$P(y=1\mid x) = \frac{e^{2F(x)}}{1 + e^{2F(x)}}$$

- Factor of 2 difference from standard logistic regression (due to labels being in $\{-1, +1\}$).

---

## 9. Loss Functions in Boosting

### Comparison of Loss Functions (for classification, $y \cdot f$ is the margin)
| Loss | Formula | Properties |
|------|---------|-----------|
| Misclassification | $\mathbf{I}(y \neq \text{sign}(f))$ | Not differentiable; not usable in gradient methods |
| Exponential | $\exp(-y \cdot f)$ | Fast computation; sensitive to mislabelled data |
| Binomial Deviance | $\log(1 + \exp(-2yf))$ | Robust; similar to exponential for positive margin |
| Squared Error | $(y - f)^2$ | Used in regression; not suitable for classification |
| Support Vector | $\max(0,\, 1 - y \cdot f)$ | Sparse; used in SVMs |

- Exponential and Binomial Deviance are similar for observations with positive margins (correctly classified).
- For negative margins (misclassified), exponential loss penalises MUCH more heavily than binomial deviance (exponential vs. linear growth).
- Binomial deviance is more robust to noise and label errors.
- Exponential loss is computationally convenient (leads to AdaBoost's simple reweighting scheme).

### Boosting and Noise
- Boosting is NOT robust to noisy data (high Bayes error rate settings).
- Sensitive to wrongly labelled observations.
- The exponential emphasis on misclassifications causes the algorithm to focus excessively on noise points.
- In noisy settings: use binomial deviance loss instead of exponential loss.

---

## 10. Gradient Boosting (Friedman, 2001)

### Key Concepts
- Generalisation of boosting to arbitrary differentiable loss functions.
- Can handle: regression, $K$-class classification, logistic regression, Poisson model, Cox model, risk modelling.
- At each iteration, fit a tree to the NEGATIVE GRADIENT of the loss function (pseudo-residuals).
- Inherits pros of trees (variable selection, mixed variable types, missing data) and improves prediction performance.

### Gradient Tree Boosting Algorithm
1. Initialise: $F_0(x) = \arg\min_\gamma \sum_{i=1}^{N} L(y_i,\, \gamma)$ and $m = 0$.
2. For $m = 1$ to $M$:
   a. For $i = 1$ to $N$, compute negative gradient (pseudo-residuals):

$$r_{im} = -\left[\frac{\partial L(y_i,\, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

   b. Fit a regression tree to $r_{im}$, giving terminal regions $R_{jm}$, $j=1,\ldots,J_m$.
   c. For $j = 1$ to $J_m$, compute optimal leaf value:

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L\!\left(y_i,\, F_{m-1}(x_i) + \gamma\right)$$

   d. Update:

$$F_m(x) = F_{m-1}(x) + \sum_{j=1}^{J_m} \gamma_{jm} \cdot \mathbf{I}(x \in R_{jm})$$

3. Output: $G(x) = F_M(x)$

### Special Cases of Gradient Boosting
- If loss = squared error: negative gradient = residual ($r_i = y_i - F_{m-1}(x_i)$). This is just least-squares boosting.
- If loss = absolute error: negative gradient = $\text{sign}(y_i - F_{m-1}(x_i))$. More robust to outliers.
- If loss = exponential: recover AdaBoost.

### Tree Size in Gradient Boosting (ANOVA Decomposition)
- The depth of the tree determines the order of interaction in the model.
- ANOVA expansion:

$$\eta(X) = \sum_j \eta_j(X_j) + \sum_{jk} \eta_{jk}(X_j, X_k) + \sum_{jkl} \eta_{jkl}(X_j, X_k, X_l) + \cdots$$

  - Stumps (1 split): only main effects (additive model, no interactions).
  - 2-split trees: main effects + 2-way interactions.
  - $J$-leaf trees: up to $(J-1)$-way interactions.
- If the true model is additive (no interactions), stumps work best.
- Typical default: $J = 4$ to $8$ terminal nodes.
- Tree size $J$ is an important hyperparameter.

### Multiclass Gradient Boosting
- Loss: $-\sum_{k=1}^{K} y_k \cdot \log P(y_k = 1 \mid x)$
- Score function per class $k$: $F_k$, $k=1,\ldots,K$ (initialised as zeros).
- Class probabilities via softmax:

$$P_k(y=1\mid x) = \frac{e^{F_k(x)}}{\sum_{k'=1}^{K} e^{F_{k'}(x)}}$$

- Negative gradient (pseudo-residuals) for class $k$: $g_k(x_i) = y_i^k - P_k(x_i)$
  - $y_i^k = 1$ if observation $i$ has class $k$, else $0$.
- Fit a tree to $g_k$ for each class at each iteration.
- Minimise KL-divergence through fitting trees to negative gradient (pseudo-residuals).

---

## 11. Ensemble Methods (General)

- Ensemble methods can combine any number and kind of regression methods as a weighted average.
- For classification: combine via majority voting.
- Random Forests and Bagging: same bias as individual tree, lower variance.
- Boosting: lower bias AND lower variance (but needs more careful tuning).

---

## 12. Summary Comparison of Ensemble Methods

| Method | Bias | Variance | Trees | Sequential? | Overfitting risk |
|--------|------|----------|-------|------------|-----------------|
| Bagging | Same as single tree | Lower | Many deep trees | No | Low |
| Random Forest | Same as single tree | Much lower (low $\rho$) | Many deep trees | No | Low |
| AdaBoost | Lower | Lower | Many stumps | Yes | Can overfit (noise) |
| Gradient Boosting | Lower | Lower | Shallow trees | Yes | Can overfit |

### Key Takeaways
- Bagging and RF: gain from VARIANCE reduction. Bias unchanged.
- Boosting: gain from BIAS reduction (and some variance reduction). Uses small/shallow trees.
- RF vs. Bagging: RF decorrelates trees by random feature selection → lower $\rho$ → lower variance floor.
- Boosting vs. Bagging: Boosting generally outperforms bagging on most problems, making it the preferred ensemble method.
- RF: trees are independent (parallelisable). Boosting: trees are dependent (sequential).
