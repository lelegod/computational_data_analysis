# Week 6 — Random Forests and Boosting (Exam Focus)

## Must-Know Facts

### Random Forests
- RF = Bagging + random feature subsampling at each split.
- RF reduces variance by DECORRELATING trees (lowering rho in the variance formula).
- At each split: only m < p randomly chosen features are considered (NOT all p features).
- Trees in RF are grown deep (no pruning), to minimum node size.
- RF does NOT reduce bias — same bias as a single tree / bagging.
- Default m: classification = floor(sqrt(p)), regression = floor(p/3).
- When m = p: RF reduces to plain Bagging.
- Increasing B (number of trees) never causes overfitting in RF.
- RF can handle p > n (more features than observations).
- RF trees are independent → training is parallelisable.
- OOB error in RF approximates cross-validation error (free, no extra runs needed).
- Pitfall: Cannot directly compare OOB error with CV error of a different model (different estimation procedures).

### Variable Importance in RF
- Two measures: Gini importance and OOB permutation importance.
- Gini: accumulates impurity reduction across all splits on feature j across all trees.
- OOB permutation: permute feature j values for OOB samples; measure accuracy drop.
- Gini concentrates importance in top few features; OOB spreads more uniformly.
- Rankings from both are usually similar.
- Use variable importance for feature selection: drop low-importance features, refit RF.

### Proximity Matrix
- n x n matrix: P(i,j) incremented when OOB observations i and j end up in same terminal node.
- Large P(i,j) = observations i and j are similar according to the forest.
- Visualised using Multidimensional Scaling (MDS).
- Points far from decision boundary = extremities. Points near boundary = near center.

### Boosting Core Facts
- Boosting = sequential ensemble of WEAK learners (small trees, stumps).
- Reduces BIAS (not just variance) — use high-bias, low-variance base learners (stumps).
- Trees are NOT independent; each tree depends on the errors of previous trees.
- Boosting dominates bagging on most problems → preferred ensemble method.
- Boosting CAN overfit, especially with noisy data or wrongly labelled observations.
- Shrinkage (learning rate nu): scale each tree's contribution by 0 < nu < 1 → more trees needed but better generalisation.

### Bagging vs. Boosting
- Bagging: parallel, reduces variance, uses deep trees, trees are independent.
- Boosting: sequential, reduces bias AND variance, uses shallow trees/stumps, trees are dependent.
- Bagging/RF: same bias as single tree. Boosting: lower bias than single tree.

### AdaBoost.M1
- Binary classification with y in {-1, +1}.
- Initialise weights w_i = 1/N; increase weights for misclassified points.
- err_m = weighted misclassification rate; alpha_m = log[(1-err_m)/err_m].
- If err_m = 0.5: alpha_m = 0 (classifier contributes nothing).
- If err_m = 0: alpha_m = infinity (perfect classifier, maximum weight).
- Final: G(x) = sign[sum_m alpha_m G_m(x)].
- AdaBoost is equivalent to forward stagewise additive modelling with EXPONENTIAL loss.
- AdaBoost is sensitive to noise because exponential loss heavily penalises misclassifications.

### Loss Functions
- Exponential loss: simple reweighting (AdaBoost); sensitive to noise.
- Binomial deviance: more robust; similar to exponential for correct predictions; linear (not exponential) for misclassified.
- For noisy data: prefer binomial deviance over exponential.
- Gradient boosting generalises to any differentiable loss function.

### Gradient Boosting
- At each step: fit a tree to the NEGATIVE GRADIENT of the loss (pseudo-residuals).
- For squared error loss: pseudo-residual = ordinary residual y_i - F_{m-1}(x_i).
- Tree size J determines interaction order: stumps → additive, J-leaf tree → (J-1)-way interactions.
- If the true model is additive (no interactions): stumps are best.

### Multiclass Boosting
- Use softmax probabilities: P_k(y=1|x) = exp(F_k(x)) / sum_k exp(F_k(x)).
- Pseudo-residual for class k: g_k(x_i) = y_i^k - P_k(x_i).
- Fit a tree to g_k for each class at each iteration.

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| `Var = rho*sigma^2 + (1-rho)/B * sigma^2` | Bagging/RF variance | Understand variance reduction; RF lowers rho |
| `y_hat = (1/B) sum T_b(x)` | RF regression prediction | Predict with RF |
| `m = floor(sqrt(p))` | RF default m (classification) | Setting RF hyperparameter |
| `m = floor(p/3)` | RF default m (regression) | Setting RF hyperparameter |
| `err_m = sum w_i I(y_i != G_m(x_i)) / sum w_i` | AdaBoost weighted error | AdaBoost step b |
| `alpha_m = log[(1-err_m)/err_m]` | AdaBoost classifier weight | AdaBoost step c |
| `w_i <- w_i * exp[alpha_m * I(y_i != G_m(x_i))]` | AdaBoost weight update | AdaBoost step d |
| `G(x) = sign[sum_m alpha_m G_m(x)]` | AdaBoost final classifier | AdaBoost output |
| `L(y, F(x)) = exp(-yF(x))` | Exponential loss (AdaBoost) | Connection to additive models |
| `r_{im} = -[dL/dF]_{F=F_{m-1}}` | Negative gradient / pseudo-residual | Gradient boosting step a |
| `F_m(x) = F_{m-1}(x) + sum_j gamma_{jm} I(x in R_{jm})` | Gradient boosting update | Gradient boosting step d |
| `P_k(y=1|x) = e^{F_k} / sum_k e^{F_k}` | Softmax (multiclass boosting) | Multiclass gradient boosting |
| `P(y=1|x) = e^{2F(x)} / (1 + e^{2F(x)})` | Probability from AdaBoost | AdaBoost probability calibration |
| `F(x) = (1/2) log[P(y=1|x)/P(y=-1|x)]` | AdaBoost log-odds (factor of 2 vs LR) | Connection to logistic regression |

---

## Common Traps (wrong answers in exams)

- RF reduces bias → RF does NOT reduce bias; same bias as a single tree / bagging.
- When m = p in RF, we get a stronger model → When m = p, RF is identical to Bagging (no decorrelation benefit).
- Boosting reduces variance only → Boosting reduces BIAS (primarily); both bias and variance decrease.
- Boosting uses deep trees → Boosting uses SHALLOW trees (stumps); bagging/RF use deep trees.
- Boosting trees are independent → Boosting trees are SEQUENTIAL and DEPENDENT (each depends on previous errors).
- Boosting never overfits → Boosting CAN overfit, especially with noisy data / wrongly labelled observations.
- AdaBoost uses equal weights for all classifiers → Weights alpha_m vary; better classifiers (lower err_m) get higher weight.
- OOB error in RF is equivalent to CV error and can be directly compared → Results are SIMILAR but not identical; direct comparison with CV from another model is a pitfall.
- RF cannot handle p > n → RF CAN handle p > n (more variables than observations).
- Gradient boosting fits trees to residuals in all cases → Only for squared error loss. For general loss, trees are fit to NEGATIVE GRADIENT (pseudo-residuals).
- Tree depth doesn't matter in boosting → Tree depth determines interaction order; stumps = no interactions; deeper trees = higher-order interactions.
- Exponential and binomial deviance losses are equivalent → They are similar for correctly classified points; exponential penalises misclassifications MUCH MORE heavily (exponential vs. linear growth).
- Larger learning rate (nu) in boosting is always better → Smaller nu (shrinkage) generally gives better generalisation but requires more trees M.
- RF proximity matrix is based on training error → It is based on OOB samples ending in the same terminal node.

---

## Quick Decision Rules

- If choosing between Bagging and RF: RF almost always better (lower variance due to lower rho).
- If asked what RF changes vs. Bagging: RF uses m < p features per split → lower rho → lower variance.
- If m = p in RF → equivalent to Bagging.
- If m = 1 in RF → maximum decorrelation, but may miss strong predictors.
- If asked what Boosting changes vs. Bagging: Boosting reduces BIAS; Bagging reduces VARIANCE.
- If the task has noisy data: prefer binomial deviance (robust) over exponential loss (AdaBoost).
- If the true model is additive (no interactions): use stumps in gradient boosting.
- If model has k-way interactions: use trees with at least k+1 leaves.
- If err_m > 0.5 in AdaBoost: alpha_m < 0 (the classifier votes against its output).
- If err_m = 0.5: alpha_m = 0 (no contribution to ensemble).
- If B → infinity: RF variance → rho * sigma^2 (floor set by inter-tree correlation).
- If increasing learning rate nu in boosting: convergence faster but risk of overfitting.
- If OOB error plateaus as B increases: enough trees; adding more gives no benefit.
- If asked about parallelism: RF trees are independent → parallelisable. Boosting trees are sequential → NOT parallelisable.
