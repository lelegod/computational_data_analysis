# Q21-AM — Bagging vs Random Forest vs Boosting
> Weeks 5/6. Could ask: compare the three main tree-ensemble ideas, explain which one reduces variance vs bias, and when each should be preferred.

---

## The Shared Motivation

All three methods improve on a single decision tree, which is usually:

- low bias if grown deep
- high variance
- unstable to small changes in training data

The difference is *how* they improve it.

---

## Bagging

Bagging trains many models on bootstrap samples and averages them:
$$
\hat{f}_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}^{*b}(x)
$$

### Main effect

- reduces variance
- leaves bias roughly unchanged
- works best with unstable base learners such as deep trees

Its key variance formula is:
$$
\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^B X_b\right)
=
\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

So the remaining problem is correlation $\rho$ between trees.

---

## Random Forest

Random Forest is bagging plus random feature subsampling at each split.

At each node, only a random subset of features is considered:
$$
m < p
$$

### Main effect

- also reduces variance
- reduces it more than bagging by lowering tree correlation $\rho$
- keeps deep low-bias trees

So RF improves bagging by lowering the variance floor.

---

## Boosting

Boosting builds learners sequentially, each one focusing on what the current ensemble gets wrong.

For AdaBoost:
- misclassified observations get upweighted
- later learners focus on hard cases

For gradient boosting:
- each learner fits the negative gradient / pseudo-residuals of the current loss

### Main effect

- primarily reduces bias
- can also control variance with shrinkage and shallow trees
- is more prone to overfitting if not regularized

---

## Parallel vs Sequential

- **Bagging**: parallel
- **Random Forest**: parallel
- **Boosting**: sequential

This is one of the cleanest exam distinctions.

Parallel methods average independent-ish learners.
Sequential methods correct previous errors.

---

## Comparison Table

| Property | Bagging | Random Forest | Boosting |
|----------|---------|---------------|----------|
| Training style | Parallel | Parallel | Sequential |
| Base learner | Deep unstable trees | Deep unstable trees | Usually weak/shallow trees |
| Main gain | Variance reduction | More variance reduction | Bias reduction |
| Bootstrap samples | Yes | Yes | Not essential |
| Random feature subset | No | Yes | Usually no |
| Sensitive to tree correlation? | Yes | Less | Not the key issue |
| Overfitting risk | Low | Low | Higher |
| OOB error | Yes | Yes | No |

---

## Why Random Forest Usually Beats Bagging

Bagging averages many trees, but if the trees are highly correlated, averaging has limited benefit.

Random Forest addresses exactly that by forcing trees to consider different feature subsets at splits. This makes trees less similar and lowers:
$$
\rho\sigma^2
$$
in the variance formula.

That is the conceptual reason RF is often better than plain bagging.

---

## Why Boosting Is Different

Boosting is not mainly about decorrelating trees.

Instead, it builds an additive model:
$$
F_M(x) = \sum_{m=1}^M \alpha_m h_m(x)
$$

Each new learner moves the ensemble toward lower loss.

So while bagging and RF improve unstable trees by averaging, boosting improves weak learners by iterative correction.

---

## When to Use Which

**Use bagging when**:
- the base learner is unstable
- you want a simple, robust variance reduction method

**Use Random Forest when**:
- trees dominate your workflow
- predictors are many and correlated
- you want strong default predictive performance

**Use boosting when**:
- bias matters more than variance
- a shallow learner is too weak alone
- you are willing to tune learning rate, depth, and number of iterations

---

## Limitations

1. Bagging and RF sacrifice interpretability relative to one tree.
2. Boosting needs more tuning and can overfit if pushed too far.
3. RF variable importance can be misleading under correlated predictors.
4. None of the three gives the simple transparency of a single CART model.

---

## Additional Possible Exam Questions

**Q: Why does bagging help trees much more than linear regression?**
Because trees are unstable and high-variance, while linear regression is already relatively stable. Averaging helps only when there is substantial variance to reduce.

**Q: Why do boosting methods often use stumps or shallow trees?**
Because boosting mainly reduces bias by building an additive ensemble. Weak learners are enough, and using overly complex trees can make the boosting path too aggressive and prone to overfitting.

**Q: Which method should be framed as variance reduction and which as bias reduction?**
Bagging and Random Forest are variance-reduction methods. Boosting is primarily a bias-reduction method.
