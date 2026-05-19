# Q21-AD — Gradient Boosting
> Week 6. Could ask: derive pseudo-residuals, connect AdaBoost to exponential loss, explain shrinkage.

---

## The General Framework: Forward Stagewise Additive Modelling

Gradient Boosting is a special case of **Forward Stagewise Additive Modelling (FSAM)**. The model is an additive expansion of basis functions (weak learners):
$$f_M(x) = \sum_{m=1}^M \beta_m b(x;\,\gamma_m)$$

**Forward stagewise fitting**: at each step $m$, hold previous learners fixed and find the new learner that most reduces the loss:
$$(\beta_m, \gamma_m) = \arg\min_{\beta,\gamma} \sum_{i=1}^N L\bigl(y_i,\, f_{m-1}(x_i) + \beta\, b(x_i;\gamma)\bigr)$$

This greedy approach avoids jointly optimizing over all $M$ learners, which would be intractable. Each step makes the best single-step improvement given the current ensemble.

---

## Key Insight: Gradient Descent in Function Space

For general loss $L$, fitting a weak learner to the FSAM step is hard. Friedman's (2001) insight: the **pseudo-residuals** are the negative gradient of the loss with respect to the current predictions:
$$r_{im} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = f_{m-1}}$$

**Interpretation**: $r_{im}$ points in the direction of steepest descent of the loss. By fitting a tree to the $r_{im}$, we are approximating the gradient step — this is **gradient descent in function space** (rather than in parameter space).

### Concrete Pseudo-Residuals by Loss

| Loss | $L(y,f)$ | Pseudo-residual $r_{im}$ | Equivalent algorithm |
|------|----------|--------------------------|----------------------|
| Squared error | $\frac{1}{2}(y-f)^2$ | $y_i - f_{m-1}(x_i)$ | L2Boost |
| Absolute error | $|y-f|$ | $\text{sign}(y_i - f_{m-1}(x_i))$ | L1Boost (robust) |
| Deviance (log-loss) | $-[yf - \log(1+e^f)]$ | $y_i - p_{m-1}(x_i)$ | Logistic Boosting |
| Exponential | $e^{-yf}$ | $y_i \exp(-y_i f_{m-1}(x_i))$ | AdaBoost |

**Algorithm (gradient boosting with trees)**:
1. Initialize $f_0(x) = \bar{y}$ (or intercept term for other losses)
2. For $m = 1, \ldots, M$:
   a. Compute pseudo-residuals $r_{im}$
   b. Fit a regression tree $T_m$ to $\{(x_i, r_{im})\}$
   c. Find optimal step size: $\gamma_m = \arg\min_\gamma \sum_i L(y_i, f_{m-1}(x_i) + \gamma T_m(x_i))$
   d. Update: $f_m(x) = f_{m-1}(x) + \nu \cdot \gamma_m T_m(x)$

---

## Connection to AdaBoost

AdaBoost (Freund & Schapire 1997) was proposed as a classification algorithm. Friedman, Hastie & Tibshirani (2000) showed it is **exactly FSAM with exponential loss** $L(y,f) = \exp(-yf)$ for $y \in \{-1, +1\}$.

**Exponential loss pseudo-residual**: $r_{im} = y_i \exp(-y_i f_{m-1}(x_i)) = y_i w_i^{(m)}$, where $w_i^{(m)} = \exp(-y_i f_{m-1}(x_i))$ are the AdaBoost **sample weights**.

**Why exponential loss leads to AdaBoost weights**: when a point is correctly classified by the current ensemble ($y_i f_{m-1}(x_i) > 0$), its weight $w_i = e^{-y_i f} < 1$. When misclassified ($y_i f_{m-1}(x_i) < 0$), $w_i = e^{|y_i f|} > 1$. The update $w_i \leftarrow w_i e^{\alpha_m \cdot \mathbf{I}(y_i \neq G_m(x_i))}$ is exactly the exponential loss FSAM step.

**Caution**: exponential loss penalizes large negative margins ($y_i f(x_i) \ll 0$) extremely aggressively. This makes AdaBoost sensitive to outliers. Deviance loss (logistic boosting) is more robust.

---

## Shrinkage Parameter

The **shrinkage** (learning rate) $\nu \in (0,1]$ scales each update:
$$f_m(x) = f_{m-1}(x) + \nu \cdot \beta_m T_m(x)$$

- Small $\nu$ → small steps → need more trees $M$ for the same training fit → better generalisation (more regularization).
- Large $\nu$ → big steps → fewer trees needed but higher variance → can overfit.
- $\nu$ and $M$ are **traded off**: smaller $\nu$ + larger $M$ typically outperforms larger $\nu$ + smaller $M$ given sufficient compute budget.
- Typical values: $\nu \in \{0.01, 0.05, 0.1\}$; $M$ chosen by early stopping (monitor validation error).

---

## Stochastic Gradient Boosting

At each step, subsample a fraction $\eta \in (0.5, 0.8]$ of the training data (without replacement) before fitting the next tree.

**Benefits**:
1. Reduces computation ($\eta N$ instead of $N$ points per tree).
2. Adds randomisation → trees are less correlated → reduces variance (similar mechanism to RF).
3. Often improves generalisation, especially with deep trees.
4. Provides an OOB-like error estimate from the unused fraction.

---

## Tree Depth and Interactions

Tree **depth** $d$ controls the order of feature interactions captured:
- Depth 1 (stumps): additive model — no interactions between features.
- Depth 2: up to pairwise interactions.
- Depth $d$: up to $d$-way interactions.

Typical depth for GBM: $d \in \{3, 4, 5\}$. Unlike bagging (which uses deep unpruned trees to minimize bias), GBM trees can be shallow because the **sequential correction** handles bias — shallow trees add stability.

---

## Bias vs Variance: Boosting Reduces Bias

The fundamental distinction:
- **Bagging** (and Random Forest): average many low-bias, high-variance trees → **reduces variance** (the correlation term $\rho\sigma^2$ in $\text{Var}(\text{avg}) = \rho\sigma^2 + (1-\rho)\sigma^2/B$).
- **Boosting**: sequentially correct the errors of a simple (high-bias) model → **reduces bias**. Each new tree fits the residuals of the current ensemble, gradually reducing the systematic error.

Because boosting reduces bias, it benefits from **high-bias base learners** (stumps or shallow trees). Using deep trees in boosting adds unnecessary variance and can cause overfitting.

---

## Comparison: Gradient Boosting vs Random Forest

| Property | Gradient Boosting | Random Forest |
|----------|------------------|---------------|
| Ensemble type | Sequential | Parallel (independent) |
| Primary effect | Reduces **bias** | Reduces **variance** |
| Base learners | Shallow trees (stumps to depth 5) | Deep unpruned trees |
| Sensitivity to $M$ | Can overfit as $M\to\infty$ | Does not overfit as $B\to\infty$ |
| Key hyperparameters | $\nu$, $M$, depth, $\eta$ (4 params) | $B$, $m$ (2 params, robust) |
| Robustness to tuning | Requires careful tuning | More robust to default settings |
| Outlier robustness | Low (especially exponential loss) | Moderate |
| Typical performance | Often best on tabular data | Excellent default |

---

## Limitations

1. **Computationally expensive**: $M$ sequential tree fits; cannot be parallelised across $M$.
2. **Many hyperparameters**: $\nu$, $M$, depth, $\eta$, loss function — interaction effects make tuning non-trivial.
3. **Sensitive to outliers**: squared error and exponential loss down-weight outliers poorly; use absolute error or Huber loss for robustness.
4. **Can overfit**: if $\nu$ is too large or $M$ too high without early stopping.
5. **No OOB error by default** (unless using stochastic GBM with $\eta < 1$).

---

## Additional Possible Exam Questions

**Q: What is the pseudo-residual for log-loss (deviance) and what does it represent?**
For logistic loss $L(y,f) = \log(1 + e^{-yf})$ (equivalent to cross-entropy with $y\in\{0,1\}$ and $f=\log p/(1-p)$), the pseudo-residual is $r_{im} = y_i - p_{m-1}(x_i)$ where $p_{m-1}(x_i) = \sigma(f_{m-1}(x_i))$ is the current predicted probability. This is exactly the residual from logistic regression — the difference between the observed label and the model's current probability estimate. Fitting a tree to these residuals corrects the current model's probability estimates.

**Q: Why does small $\nu$ + large $M$ generalise better than large $\nu$ + small $M$?**
Small $\nu$ makes each update a small perturbation — the model explores the loss landscape gradually, averaging over many small correlated steps rather than a few large jumps. This implicit averaging reduces variance. It also allows early stopping to select the optimal $M$ with high resolution (the validation error curve is smoother). Concretely: small $\nu = 0.01$ with $M = 1000$ trees typically outperforms $\nu = 0.1$ with $M = 100$ trees on held-out data.

**Q: Prove that exponential loss leads to the AdaBoost weight update.**
Exponential loss: $L = \sum_i e^{-y_i f_{m-1}(x_i)} e^{-y_i \beta G_m(x_i)}$. Let $w_i^{(m)} = e^{-y_i f_{m-1}(x_i)}$. Split the sum: correctly classified ($y_i = G_m(x_i)$): weight $\times e^{-\beta}$; misclassified: weight $\times e^{+\beta}$. Taking $\partial/\partial\beta = 0$ and solving: $\beta_m = \frac{1}{2}\log\frac{1-\text{err}_m}{\text{err}_m}$ (this is $\alpha_m$ in AdaBoost). Weight update after step $m$: $w_i^{(m+1)} \propto w_i^{(m)} e^{-y_i \beta_m G_m(x_i)} = w_i^{(m)} e^{\alpha_m \mathbf{I}(y_i \neq G_m(x_i))} e^{-\alpha_m/2}$ (the constant $e^{-\alpha_m/2}$ cancels after normalisation). This is exactly the AdaBoost weight update.
