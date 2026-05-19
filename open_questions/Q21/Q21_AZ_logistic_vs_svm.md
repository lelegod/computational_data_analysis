# Q21-AZ — Logistic Regression vs SVM
> Weeks 4/7. Could ask: compare logistic regression and SVM as linear classifiers, focusing on loss function, probabilistic output, and regularization.

---

## The Shared Setting

Both methods often produce linear decision boundaries:

- logistic regression by modeling posterior probabilities
- SVM by maximizing the separating margin

So they can look similar in practice, but the optimization principle is different.

---

## Logistic Regression

Logistic regression models:
$$
\log\frac{P(C_1\mid x)}{P(C_0\mid x)} = \beta_0 + x^T\beta
$$

and fits parameters by maximum likelihood.

### Main idea

- directly model posterior probabilities
- minimize log loss
- probabilistic classifier

### Key consequence

Every training point influences the fit because log loss is never exactly zero.

---

## SVM

SVM solves a margin-based optimization:
$$
\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 + C\sum_i \xi_i
$$

subject to classification constraints.

### Main idea

- maximize margin
- minimize hinge loss
- geometric classifier

### Key consequence

Only points near or inside the margin matter strongly. These are the support vectors.

---

## Loss Functions

### Logistic regression

Uses log loss:
$$
L(y,F)=\log(1+\exp(-yF))
$$

### SVM

Uses hinge loss:
$$
L(y,F)=\max(0,1-yF)
$$

### Main difference

- log loss is smooth and probabilistic
- hinge loss is margin-based and becomes zero once the point is correctly classified with enough margin

This is one of the most important distinctions.

---

## Probabilities vs Margins

### Logistic regression

- outputs calibrated probabilities
- natural if risk prediction matters

### SVM

- outputs a score / margin
- not naturally probabilistic
- probabilities require extra calibration such as Platt scaling

So if the exam asks “which method should you use if probabilities matter?”, the answer is logistic regression.

---

## Regularization

Both methods are commonly regularized with $L_2$ penalties.

### Logistic regression

- penalized log-likelihood
- still probabilistic

### SVM

- $C$ controls margin softness
- large $C$ = less regularization, tighter fit
- small $C$ = more regularization, wider margin

So both have bias-variance tuning, but with different interpretations.

---

## Comparison Table

| Property | Logistic Regression | SVM |
|----------|---------------------|-----|
| Main objective | Max likelihood / min log loss | Max margin / min hinge loss |
| Output | Probability | Margin / score |
| Uses all points? | Yes | Mainly support vectors |
| Probabilistic? | Yes | Not natively |
| Kernel extension | Not standard in this course sense | Natural and central |

---

## High-Dimensional Setting

SVM is often especially strong when:
- $p \gg n$
- classification is the only goal
- margin-based separation is attractive

Logistic regression is often especially strong when:
- probability estimation matters
- interpretability of coefficients matters
- the log-odds model is a reasonable approximation

---

## When to Use Which

**Use logistic regression when**:
- calibrated probabilities matter
- you want interpretable coefficients
- you are comfortable with a linear log-odds model

**Use SVM when**:
- classification accuracy is the main goal
- high-dimensional separation matters
- kernels may be useful

---

## Limitations

1. Logistic regression is limited to linear log-odds unless features are expanded.
2. SVM does not naturally output probabilities.
3. Both can struggle when the boundary is highly nonlinear unless feature engineering or kernels are used.

---

## Additional Possible Exam Questions

**Q: Why do logistic regression and SVM often give similar boundaries?**
Because both are regularized linear classifiers, even though they optimize different loss functions.

**Q: Which one is more naturally robust to far-away correctly classified points?**
SVM, because hinge loss becomes zero outside the margin.

**Q: Which one is better if the output must be a risk probability?**
Logistic regression.
