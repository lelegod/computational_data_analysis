# Q21-E — Boosting Algorithms
> Related: 2022 Q20, 2024 Q10, regularly tested in MC

---

## Core Idea

Build an ensemble **sequentially**: each new learner focuses on the errors made by the current ensemble. Unlike bagging (which reduces variance via averaging), boosting reduces **bias** by fitting residuals.

---

## AdaBoost.M1 Algorithm

**Setup**: Binary classification, $y_i \in \{-1, +1\}$, weak classifiers $G_m: \mathbb{R}^p \to \{-1,+1\}$.

**Initialize**: $w_i = 1/N$ for all $i$.

**For $m = 1, \ldots, M$**:

1. Fit classifier $G_m(x)$ to training data using weights $w_i$

2. Compute weighted misclassification rate:
$$\text{err}_m = \frac{\sum_{i=1}^N w_i \cdot \mathbf{I}(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i}$$

3. Compute classifier weight:
$$\alpha_m = \log\frac{1-\text{err}_m}{\text{err}_m}$$
   - $\text{err}_m \to 0$: $\alpha_m \to \infty$ (near-perfect classifier, high weight)
   - $\text{err}_m = 0.5$: $\alpha_m = 0$ (random, contributes nothing)
   - $\text{err}_m > 0.5$: $\alpha_m < 0$ (worse than random, flip its predictions)

4. Update weights: increase for misclassified, keep for correct:
$$w_i \leftarrow w_i \cdot \exp\!\big[\alpha_m \cdot \mathbf{I}(y_i \neq G_m(x_i))\big]$$

5. Normalize: $w_i \leftarrow w_i / \sum_j w_j$

**Final classifier**:
$$G(x) = \text{sign}\!\left[\sum_{m=1}^M \alpha_m G_m(x)\right]$$

---

## The Exponential Loss Connection

AdaBoost = **forward stagewise additive modelling** minimizing exponential loss:
$$L(y, F(x)) = \exp(-y\cdot F(x))$$

At step $m$, we add the best $(\alpha_m, G_m)$ pair:
$$(\alpha_m, G_m) = \arg\min_{\alpha,G} \sum_i \exp\!\left[-y_i(F_{m-1}(x_i) + \alpha G(x_i))\right]$$

Setting $\partial/\partial\alpha = 0$ recovers exactly the AdaBoost $\alpha_m$ formula. The weight update $w_i \leftarrow w_i \cdot \exp(\alpha_m\mathbf{I}(y_i\neq G_m(x_i)))$ is the gradient with respect to the model — misclassified points get higher weight because they have higher exponential loss.

**Why weights grow faster for badly misclassified points**:
- Misclassification loss: flat increment of 1 regardless of how wrong
- Exponential loss: $\exp(\alpha_m)$ — grows exponentially with $\alpha_m$ (which grows as $\text{err}_m \to 0$)

For a 95%-accurate classifier: $\alpha_m \approx 2.94$, weight multiplier $= e^{2.94} \approx 19$. Badly misclassified points are 19× more important in the next round.

---

## Gradient Boosting (General Framework)

Generalize: replace exponential loss with any differentiable loss $L(y, F)$.

**Step $m$**:
1. Compute pseudo-residuals (negative gradient of loss):
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

2. Fit a regression tree $h_m(x)$ to $r_{im}$

3. Update with shrinkage:
$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x), \quad \nu \in (0,1]$$

**Loss-specific pseudo-residuals**:
| Loss $L$ | $r_{im}$ | Task |
|---------|---------|------|
| Squared error: $(y-F)^2/2$ | $y_i - F_{m-1}(x_i)$ | Regression |
| Exponential: $\exp(-yF)$ | $y_i\exp(-y_iF_{m-1}(x_i))$ | AdaBoost |
| Binomial deviance | $y_i - P(y_i=1|x_i)$ | Classification |

**Shrinkage** $\nu < 1$: each step contributes less → slower but better generalization. Tradeoff with $M$: smaller $\nu$ requires larger $M$.

---

## Why Stumps? (Bias vs Variance)

**Boosting framework reduces bias** — it sequentially fits residuals, driving training error toward zero.

The base learner must:
- Be a **weak learner** (slightly better than random) — ensures $\text{err}_m < 0.5$ at each step
- Have **high bias** (simple) — stumps (depth-1 trees) are prototypical

**Contrast with bagging**: uses deep trees (low bias, high variance) and averages to reduce variance.

**Why KNN cannot be a base learner for boosting**:
- Boosting requires fitting weighted residuals at each step
- KNN with fixed $k$ cannot weight observations differently
- More importantly: small-$k$ KNN (low bias) has high variance but low bias — bagging reduces variance well, but gradient boosting needs to fit the residuals via a model that can be guided by a loss function
- Standard KNN does not natively support instance-weighted training

---

## Comparison Table

| | Bagging/RF | AdaBoost | Gradient Boosting |
|--|-----------|----------|-------------------|
| Base learner | Deep trees | Stumps | Trees (any depth) |
| Sequential? | No | Yes | Yes |
| Reduces | Variance | Bias | Bias |
| Loss function | N/A | Exponential | Any differentiable |
| Can overfit? | No (RF) | Yes (with noise) | Yes (without shrinkage) |
| Works with noise? | Yes | Sensitive | Sensitive |

---

## Additional Possible Exam Questions

**Q: Does boosting overfit?**
Yes, in the presence of noisy labels. If some $y_i$ are mislabeled, AdaBoost eventually concentrates all weight on those noisy points (exponential loss is unbounded) and memorizes them. Gradient boosting with shrinkage $\nu < 1$ and early stopping (via CV) avoids this.

**Q: What is the AdaBoost training error bound?**
$$\text{Training error} \leq \prod_m 2\sqrt{\text{err}_m(1-\text{err}_m)} = \prod_m \exp(-\gamma_m^2/2)$$
where $\gamma_m = 0.5 - \text{err}_m > 0$ is the edge above random. If each weak learner has edge $\gamma > 0$, training error decreases exponentially to zero. This is the **boosting theorem**.

**Q: Compare AdaBoost to logistic regression.**
Both minimize loss functions that are monotone decreasing in the margin $y_if(x_i)$. Logistic regression uses log loss (bounded gradient); AdaBoost uses exponential loss (unbounded gradient → sensitive to outliers). Both learn linear combinations of features, but AdaBoost's "features" are weak classifiers.

**Q: What is $\nu$ in gradient boosting and how do you tune it?**
$\nu$ is the learning rate (shrinkage parameter). Smaller $\nu$ → smaller steps → need more trees $M$ → better generalization but slower. Tune via cross-validation on $(M, \nu)$ jointly. Common range: $\nu \in [0.01, 0.1]$ with $M \in [100, 1000]$.

**Q: Why is boosting in the "function space" sense of gradient descent?**
Each step adds $\nu h_m$ to $F$ (the model = a function). This is gradient descent in the space of functions: $F_m = F_{m-1} - \nu \nabla_F L$, where $-\nabla_F L = r_{im}$ (pseudo-residuals). The tree $h_m$ approximates the negative gradient. This interpretation shows boosting is a general optimization framework, not specific to classification.
