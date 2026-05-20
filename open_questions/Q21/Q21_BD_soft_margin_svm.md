# Q21-BD — Soft-Margin SVM
> Week 7. Could ask: derive the primal and dual, explain the role of $C$, characterize support vectors, connect to hinge loss.

---

## The Model

Binary classification: $y_i \in \{-1,+1\}$, data $x_i \in \mathbb{R}^p$.

**Motivation**: the hard-margin SVM requires the data to be linearly separable — if no hyperplane correctly classifies all points, the primal is infeasible. Soft-margin SVM relaxes this by allowing violations, penalised in proportion to their magnitude.

**Slack variables**: $\xi_i \geq 0$ measures how far observation $i$ falls on the wrong side of its margin. $\xi_i = 0$ means $i$ is correctly classified outside or on the margin; $\xi_i > 0$ means $i$ is inside the margin or on the wrong side.

---

## Primal Formulation

$$\min_{\beta,\beta_0,\xi} \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^N \xi_i$$

subject to:
$$y_i(x_i^T\beta + \beta_0) \geq 1 - \xi_i \quad \forall i, \qquad \xi_i \geq 0 \quad \forall i$$

**Interpretation**:
- $\frac{1}{2}\|\beta\|^2$: margin-maximization term (minimize $\Rightarrow$ maximize margin $2/\|\beta\|$).
- $C\sum_i \xi_i$: total penalty for all violations. Slack $\xi_i$ is the distance by which $i$ violates the margin.
- $C > 0$: the regularization hyperparameter, trading off margin width against training error.

---

## Mechanism

**Forming the Lagrangian** (multipliers $\alpha_i \geq 0$ for the margin constraints, $\mu_i \geq 0$ for $\xi_i \geq 0$):

$$L_P = \frac{1}{2}\|\beta\|^2 + C\sum_i\xi_i - \sum_i\alpha_i[y_i(x_i^T\beta+\beta_0) - 1 + \xi_i] - \sum_i\mu_i\xi_i$$

**Stationarity conditions**:

$$\frac{\partial L}{\partial \beta} = 0 \;\Rightarrow\; \beta = \sum_i\alpha_i y_i x_i$$

$$\frac{\partial L}{\partial \beta_0} = 0 \;\Rightarrow\; \sum_i \alpha_i y_i = 0$$

$$\frac{\partial L}{\partial \xi_i} = 0 \;\Rightarrow\; C - \alpha_i - \mu_i = 0 \;\Rightarrow\; \alpha_i = C - \mu_i \leq C$$

Because $\mu_i \geq 0$, we get the **box constraint** $0 \leq \alpha_i \leq C$.

---

## Dual Problem

Substituting the stationarity conditions back yields the dual:

$$\max_\alpha \sum_i\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \langle x_i, x_j\rangle$$

subject to:
$$0 \leq \alpha_i \leq C, \qquad \sum_i \alpha_i y_i = 0$$

**The only difference from hard-margin SVM**: the constraint $\alpha_i \geq 0$ becomes $0 \leq \alpha_i \leq C$ (an upper bound — the box constraint). The kernel trick applies identically: replace $\langle x_i, x_j\rangle$ with $K(x_i, x_j)$.

---

## KKT Conditions and Point Classification

Complementary slackness gives two sets of conditions:
- $\alpha_i[y_i(x_i^T\beta+\beta_0) - 1 + \xi_i] = 0$
- $\mu_i \xi_i = 0$, i.e., $(C - \alpha_i)\xi_i = 0$

This leads to three mutually exclusive categories for each observation:

| Case | $\alpha_i$ | $\xi_i$ | Location |
|------|-----------|---------|----------|
| Correctly outside margin | $\alpha_i = 0$ | $\xi_i = 0$ | $y_i f(x_i) > 1$ |
| On the margin (support vector) | $0 < \alpha_i < C$ | $\xi_i = 0$ | $y_i f(x_i) = 1$ |
| Margin violator | $\alpha_i = C$ | $\xi_i > 0$ | $y_i f(x_i) < 1$ |

**Recovering $\beta_0$**: use any support vector with $0 < \alpha_i < C$ (these satisfy $y_i f(x_i) = 1$ exactly):
$$\beta_0 = y_s - x_s^T\beta$$

---

## Key Properties

**Effect of $C$ on bias-variance**:
- Large $C$: violations are expensive → nearly all constraints enforced → narrow margin → model fits training data closely → **low bias, high variance** (approaches hard-margin SVM as $C \to \infty$).
- Small $C$: violations are cheap → wide margin allowed → many points can violate the margin → **high bias, low variance**.
- $C$ is selected by cross-validation.

**Sparsity**: the solution $\beta = \sum_i \alpha_i y_i x_i$ depends only on observations with $\alpha_i > 0$ (support vectors plus margin violators). Most $\alpha_i = 0$ → solution is sparse in observations.

**Non-probabilistic**: the decision function $f(x) = x^T\beta + \beta_0$ gives a margin score, not a probability. Probability calibration requires Platt scaling.

---

## Hinge Loss Interpretation

The soft-margin objective is equivalent to ridge-penalised hinge loss minimisation:

$$\min_{\beta,\beta_0} \frac{1}{N}\sum_{i=1}^N \max(0, 1 - y_i f(x_i)) + \frac{\lambda}{2}\|\beta\|^2$$

where $\lambda = 1/(CN)$. The **hinge loss** $\max(0, 1 - y_i f(x_i))$:
- Is zero when $y_i f(x_i) \geq 1$ (correctly classified and beyond the margin).
- Increases linearly when the point is inside the margin or misclassified.
- Has a kink at $y_i f(x_i) = 1$ — it is not differentiable there (subdifferentiable).

This interpretation makes the connection to logistic regression (log loss instead of hinge loss) and ridge regression ($L_2$ penalty in both) explicit.

---

## Comparison to Alternatives

| Property | Hard-Margin SVM | Soft-Margin SVM | Logistic Regression |
|----------|-----------------|-----------------|---------------------|
| Separable data required? | Yes | No | No |
| Loss function | N/A (feasibility) | Hinge loss | Log loss |
| Probabilistic output? | No | No | Yes |
| Handles noisy labels? | No | Yes (via $C$) | Yes |
| Support vectors? | All margin points | Margin + violators | All points (non-zero gradient) |
| Hyperparameters | None | $C$ (+ kernel params) | $\lambda$ |
| $C \to \infty$ | Hard-margin | Hard-margin | N/A |

**Key distinction from logistic regression**: logistic regression uses log loss, which has non-zero gradient for all correctly classified points. SVM hinge loss is zero for points outside the margin → sparse support-vector solution → SVM ignores easy examples during training.

---

## Limitations

- **$C$ selection**: must be tuned by cross-validation; the optimal value depends on the scale of features (features should be standardised).
- **No probability output**: Platt scaling is a post-hoc workaround; SVM is not designed for probabilistic prediction.
- **Multi-class**: not natively multi-class. Requires One-vs-One ($K(K-1)/2$ classifiers) or One-vs-Rest ($K$ classifiers).
- **Interpretability**: coefficients $\beta$ are a linear combination of support vectors — not directly interpretable as feature importances.
- **Kernel choice**: with non-linear kernels (RBF), a second hyperparameter $\gamma$ must be tuned jointly with $C$.
- **Computational cost**: the dual QP is $O(N^2)$–$O(N^3)$ in training observations.

---

## Additional Possible Exam Questions

**Q: What is the key difference between the hard-margin and soft-margin dual problems?**
The dual constraints change from $\alpha_i \geq 0$ (hard-margin) to $0 \leq \alpha_i \leq C$ (soft-margin). The upper bound $C$ comes from the Lagrange multiplier for the $\xi_i \geq 0$ constraint: stationarity gives $\alpha_i + \mu_i = C$ with $\mu_i \geq 0$, so $\alpha_i \leq C$. The dual objective is identical in both cases.

**Q: An observation has $\alpha_i = C$. What does this tell you about that observation?**
That observation is a margin violator: $\xi_i > 0$, so $y_i f(x_i) < 1$. It lies inside the margin or is misclassified. The complementary slackness condition $(C - \alpha_i)\xi_i = 0$ is satisfied with $C - \alpha_i = 0$, so $\xi_i$ can be positive. The observation is still a support vector (it contributes to $\beta$), but it is on the "wrong" side of the margin.

**Q: Show that the soft-margin SVM minimises hinge loss plus an $L_2$ penalty.**
The slack variable satisfies $\xi_i = \max(0, 1 - y_i f(x_i))$ at the optimum (any larger $\xi_i$ wastes budget; any smaller violates the constraint). Substituting into the primal: $\frac{1}{2}\|\beta\|^2 + C\sum_i \max(0, 1 - y_i f(x_i))$. Dividing by $CN$ and setting $\lambda = 1/(CN)$ gives the hinge loss form.

**Q: Why should features be standardised before fitting SVM?**
The margin $2/\|\beta\|$ is measured in the original feature space. Features with large scales dominate $\|\beta\|^2$ — the SVM will try to be robust to the high-scale features and ignore the low-scale ones. After standardisation, all features contribute equally to the margin, and the $C$ parameter has a consistent meaning across problems.

**Q: How does the kernel trick extend the soft-margin SVM?**
The dual depends on observations only through inner products $\langle x_i, x_j \rangle$. Replacing these with a positive-definite kernel $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ implicitly maps data to a higher-dimensional feature space. The soft-margin primal is solved in that space, yielding a non-linear decision boundary in the original space — without ever computing $\phi(x)$ explicitly. The box constraint $0 \leq \alpha_i \leq C$ is unchanged.
