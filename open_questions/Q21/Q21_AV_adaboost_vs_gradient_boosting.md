# Q21-AV — AdaBoost vs Gradient Boosting
> Week 6. Could ask: compare AdaBoost and gradient boosting, explain their loss functions, and show why AdaBoost is a special case of gradient boosting.

---

## The Shared Idea

Both methods build an additive ensemble:
$$
F_M(x) = \sum_{m=1}^M \alpha_m h_m(x)
$$

and improve the model sequentially.

So both are boosting methods, but they differ in how they decide what the next learner should do.

---

## AdaBoost

AdaBoost focuses on **misclassified observations** by reweighting them.

For binary classification with $y_i \in \{-1,+1\}$:

1. fit weak learner $h_m$
2. compute weighted error
   $$
   \mathrm{err}_m = \frac{\sum_i w_i \mathbf{1}(y_i \neq h_m(x_i))}{\sum_i w_i}
   $$
3. assign learner weight
   $$
   \alpha_m = \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}
   $$
4. upweight the misclassified points

### Main interpretation

AdaBoost is a reweighting algorithm that puts more attention on hard cases.

---

## Gradient Boosting

Gradient boosting generalizes the idea:

At step $m$, fit the next learner to the **negative gradient** of the loss:
$$
r_{im} = -\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}
$$

Then update:
$$
F_m(x) = F_{m-1}(x) + \nu h_m(x)
$$

where $\nu$ is the learning rate (shrinkage).

### Main interpretation

Gradient boosting is stagewise functional gradient descent in function space.

---

## The Core Difference

### AdaBoost

- fixed to classification
- fixed to exponential loss
- reweights observations directly

### Gradient Boosting

- much more general
- works for regression and classification
- handles many differentiable losses
- fits pseudo-residuals, not just misclassification indicators

So AdaBoost is a special, more specific boosting algorithm.

---

## Loss Functions

### AdaBoost

Equivalent to minimizing exponential loss:
$$
L(y,F(x)) = \exp(-yF(x))
$$

This heavily penalizes misclassified points and is sensitive to noisy labels.

### Gradient Boosting

Can minimize:

- squared error
- binomial deviance / logistic loss
- exponential loss
- many other differentiable losses

This flexibility is one of its biggest advantages.

---

## Why AdaBoost Is a Special Case

If gradient boosting is run with:

- exponential loss
- stagewise additive fitting for classification

then its updates match AdaBoost.

So AdaBoost can be viewed as one specific gradient-boosting procedure under exponential loss.

This is exactly the kind of conceptual bridge that examiners like.

---

## Sensitivity to Noise

This is one of the most important practical distinctions.

### AdaBoost

- exponential loss grows very rapidly for badly misclassified points
- noisy labels or outliers can dominate the fit

### Gradient Boosting

- can use binomial deviance instead
- this is more robust than exponential loss

So for noisy data, generic gradient boosting is often safer than AdaBoost.

---

## Comparison Table

| Property | AdaBoost | Gradient Boosting |
|----------|----------|-------------------|
| Main idea | Reweight misclassified points | Fit negative gradients |
| Loss | Exponential | General differentiable loss |
| Task | Primarily classification | Regression and classification |
| Noise sensitivity | Higher | Lower if robust loss used |
| Tuning | Number of learners | Learners + depth + learning rate + loss |

---

## Base Learners

Both methods often use shallow trees or stumps.

Why:

- boosting mainly reduces bias
- weak learners are enough
- deep learners can overfit more aggressively

In gradient boosting, tree depth also controls interaction order.

---

## When to Use Which

**Use AdaBoost when**:
- the task is clean binary classification
- you want the classic boosting formulation
- label noise is limited

**Use Gradient Boosting when**:
- you want flexibility in the loss
- you have regression or multiclass tasks
- you want shrinkage and more robust control

---

## Limitations

1. Both are sequential and less parallelizable than bagging/RF.
2. Both can overfit if pushed too far.
3. AdaBoost is especially sensitive to noise.
4. Gradient boosting requires more tuning.

---

## Additional Possible Exam Questions

**Q: Why is AdaBoost more sensitive to mislabeled data?**
Because exponential loss heavily upweights badly misclassified observations, so noisy labels can dominate later rounds.

**Q: What is the main mathematical idea behind gradient boosting?**
At each stage it moves in the direction of the negative functional gradient of the loss.

**Q: Why is AdaBoost considered a special case of gradient boosting?**
Because if gradient boosting is applied to exponential loss in binary classification, the resulting stagewise updates match AdaBoost.
