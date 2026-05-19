# Week 6 — Lecture Notes
## Computational Data Analysis (02582)

---

## Bagging (Bootstrap Aggregating)

### Bias — Bagging Does Not Help

Each bootstrap tree is identically distributed (same expected value). Averaging is linear:

$$E\left[\frac{1}{B}\sum_{b=1}^{B}(\hat{y}_b - y)\right] = \frac{1}{B}\sum_{b=1}^{B} E(\hat{y}_b - y) = E(\hat{y}_b - y)$$

The bias of the bagged model equals the bias of a single tree. If individual trees are biased, bagging cannot fix it.

### Variance — Bagging Helps, But Has a Ceiling

For $B$ trees each with variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}(\bar{\hat{y}}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

- **Second term** $\to 0$ as $B \to \infty$ — averaging reduces this
- **First term** $\rho\sigma^2$ **remains** — this is the variance floor

The bottleneck is $\rho$: **how correlated are the trees?**

Bagged trees tend to be correlated because they all see the same $p$ features and dominant predictors appear at the top of nearly every tree.

---

## Random Forests

### Key Idea: Decorrelate the Trees

At every split, randomly sample $m < p$ predictors **without replacement** and choose the best split **only among those $m$**.

$$m \approx \sqrt{p} \;\text{(classification)}, \qquad m \approx p/3 \;\text{(regression)}$$

This forces trees to use different features → trees look different from each other → $\rho$ decreases:

$$\underbrace{\rho\sigma^2}_{\text{lower in RF}} + \underbrace{\frac{1-\rho}{B}\sigma^2}_{\to 0}$$

Small increase in bias (restricted search), but the variance reduction far outweighs it.

### Algorithm

1. Choose $B$ (typically a few hundred; overfitting from more trees is not a problem)
2. For $b = 1$ to $B$:
   - (a) **Bootstrap**: draw $N$ samples **with** replacement
   - (b) Grow a full CART tree (**no pruning**); at each split:
     - Draw $m$ predictors **without** replacement
     - Find the best split among those $m$ predictors only
     - Repeat until minimum node size $n_{\min}$ is reached
3. Output: $B$ trees

**Prediction:**
- Regression: $\hat{y} = \frac{1}{B}\sum_{b=1}^{B} \hat{y}_b$
- Classification: majority vote across $B$ trees

### Why No Pruning?

The randomness in bootstrap sampling + random feature selection already controls variance. Pruning would only add bias without benefit.

### Bagging vs Random Forest

| | Bagging | Random Forest |
|---|---|---|
| Bootstrap | Yes | Yes |
| Features per split | All $p$ | Random $m < p$ |
| Tree correlation $\rho$ | High | Lower |
| Variance reduction | Limited by $\rho$ | Better — $\rho$ actively reduced |
| Pruning | No | No |

---

## OOB Error vs Test Error

### Out-of-Bag (OOB) Error

Each bootstrap sample of size $N$ (with replacement) leaves out each observation with probability:

$$P(\text{not selected}) = \left(1 - \frac{1}{N}\right)^N \xrightarrow{N\to\infty} e^{-1} \approx 0.368$$

So ~36.8% of observations are **out-of-bag** for any given tree. For each observation $i$, predict using only the trees that never trained on it, then measure error across all such predictions.

### Comparison

| | OOB Error | Test Error |
|---|---|---|
| Requires separate test set | No — free byproduct | Yes |
| Uses full training data | Yes | No (some held out) |
| Each prediction uses | $\approx B/3$ trees | All $B$ trees |
| Bias | Slightly pessimistic | Unbiased (if test set is clean) |

### Key Point

OOB is slightly **pessimistic** — predictions use only ~$B/3$ trees, but the final model uses all $B$. Fewer trees = slightly worse performance → OOB slightly overestimates true generalisation error.

As $B \to \infty$, OOB error $\approx$ leave-one-out cross-validation (LOOCV).

> Use OOB when data is scarce. Use a held-out test set when you have enough data and need an unbiased estimate.

---

## Variable Importance

### Gini Importance

At every split in every tree, record the reduction in Gini impurity for the variable used. Sum over all trees:

$$\text{Importance}_j = \sum_{\text{trees}} \sum_{\text{splits on } j} \Delta G$$

Tends to give inflated importance to variables used near the root (used more often).

### OOB Permutation Importance

1. Drop OOB samples down the tree → record accuracy $A$
2. **Permute** (shuffle) variable $j$ in the OOB samples → record accuracy $A_j$
3. Importance$_j = A - A_j$ (averaged over all trees)

More direct: "if I scramble this variable, how much does accuracy drop?" Variables that are truly useful show large drops; noise variables show near zero.

**OOB spreads importance more uniformly than Gini** — Gini concentrates importance on a few top variables because root-level splits are counted more.

---

## $p > n$ and Feature Selection

Random forests can handle $p > n$ (more variables than observations), but struggle when **most of the $p$ variables are noise** — at each split you randomly sample from a pool mostly containing garbage.

**Fix:** Use variable importance (OOB permutation) to identify truly useful variables, then refit the forest using only those. This typically lowers OOB error further.

### Reading the OOB Importance Plot

Each spike in the stem plot = one variable. Height = how much accuracy drops when that variable is permuted.
- Tall spike → important
- Near zero → noise

Threshold (e.g. $\Delta \text{Error} > 0.025$): keep only variables above it, refit.

---

## Graph Summaries

| Graph | What it shows |
|---|---|
| OOB MSE vs trees (varying $m$) | Too small $m$ = high bias; too large $m$ = high correlation. Sweet spot exists. |
| Gini vs OOB importance bar charts | OOB spreads importance; Gini concentrates it on frequently-split variables. |
| Sand dataset stem plot | Most variables are noise; a few spikes = truly useful predictors. |
| Feature selection (after thresholding) | Removing noise variables lowers OOB error floor. |
| Zip digit OOB error vs trees | Error plateaus ~100 trees. Pitfall: do not compare OOB error directly to CV error. |
| Pixel importance heatmap | Variable importance arranged spatially — confirms centre pixels matter most for digit classification. |

---

## Boosting as Forward Stagewise Additive Modeling

### The Additive Model View

Boosting fits an **additive model** where each basis function is a tree:

$$F(x) = \sum_{m=1}^{M} \beta_m\, b(x;\, \gamma_m)$$

- $b(x;\gamma_m)$: a tree parameterised by its splits $\gamma_m$
- $\beta_m$: weight of tree $m$

AdaBoost is identical in structure:

$$G(x) = \text{sign}\!\left[\sum_{m=1}^{M} \alpha_m G_m(x)\right] \quad \longleftrightarrow \quad \alpha_m = \beta_m,\quad G_m = b(x;\gamma_m)$$

### Forward Stagewise Fitting

Fitting all $M$ trees simultaneously is intractable. Instead, fit **one tree at a time, never changing previous trees**:

At step $m$, fix $F_{m-1}(x) = \sum_{k=1}^{m-1}\beta_k b(x;\gamma_k)$ and solve:

$$(\beta_m, \gamma_m) = \arg\min_{\beta,\gamma} \sum_{i=1}^{N} L\!\left(y_i,\; F_{m-1}(x_i) + \beta\, b(x_i;\gamma)\right)$$

Then update: $F_m(x) = F_{m-1}(x) + \beta_m\, b(x;\gamma_m)$

### Loss Function Determines the Algorithm

| Loss $L(y, F)$ | Algorithm |
|---|---|
| $e^{-yF}$ (exponential) | AdaBoost.M1 |
| $(y - F)^2$ (squared error) | $L_2$ Boosting |
| $\log(1 + e^{-2yF})$ (log-likelihood) | LogitBoost |

### Why AdaBoost Reweights — The Exponential Loss Derivation

Expanding the exponential loss at step $m$:

$$\sum_{i=1}^{N} \underbrace{e^{-y_i F_{m-1}(x_i)}}_{w_i^{(m)}} \cdot e^{-y_i \beta\, b(x_i;\gamma)}$$

The weight $w_i^{(m)} = e^{-y_i F_{m-1}(x_i)}$ is large when observation $i$ was previously misclassified ($y_i F_{m-1}(x_i) < 0$). This is **exactly** the AdaBoost observation weighting — not arbitrary, but the direct result of minimising exponential loss forward-stagewise.

### Summary

> Boosting = greedily fitting an additive model of trees, one at a time, where each new tree focuses on what the current model got wrong.
