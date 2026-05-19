# Week 10 — Lecture Notes
## Computational Data Analysis (02582)

---

## Gradient Descent

### The Update Rule

For a function $f(w)$, gradient descent updates parameters by stepping opposite to the gradient:

$$w_{t+1} = w_t - \eta \cdot \nabla f(w_t)$$

$\eta$ = **learning rate** — controls step size.

### Example: $f(w) = w^2$

$$\nabla f(w) = 2w$$

$$w_{t+1} = w_t - \eta \cdot 2w_t = w_t(1 - 2\eta)$$

Starting at $w_0 = 10$:

| Scenario | $\eta$ | $w_1$ | Factor $|1-2\eta|$ | Behaviour |
|---|---|---|---|---|
| A | $0.1$ | $8$ | $0.8 < 1$ | Converges to 0 |
| B | $1.1$ | $-12$ | $1.2 > 1$ | Diverges (oscillates and grows) |

**Scenario B step-by-step:**
$$w_1 = 10 - 1.1 \cdot 20 = -12 \quad \to \quad w_2 = 14.4 \quad \to \quad w_3 = -17.28 \quad \to \cdots$$

Overshoots the minimum, flips sign, overshoots further — explodes.

### The Learning Rate Trade-off

$$\eta \text{ too small} \Rightarrow \text{slow convergence} \qquad \eta \text{ too large} \Rightarrow \text{overshooting / divergence}$$

Convergence condition for $f(w) = w^2$: $|1 - 2\eta| < 1 \Rightarrow \eta < 1$.

---

## Convex vs Non-Convex Loss Surfaces

### Convex (Shape A — single bowl)

- One global minimum, no local minima
- Gradient descent **always** finds the global minimum (for small enough $\eta$)
- Initialisation does not matter

### Non-Convex (Shape B — multiple valleys)

- Multiple local minima; one global minimum
- Gradient descent converges to the **nearest local minimum** from the initialisation point
- Initialisation **matters enormously**

**Strategies to escape local minima:**

| Strategy | Idea |
|---|---|
| Random restarts | Run from multiple initialisations, keep best result |
| Larger $\eta$ | May jump over shallow local minima (risky) |
| Momentum | Accumulates velocity to roll over small bumps |
| SGD | Noisy gradient estimates help escape local minima |

---

## Binary Cross-Entropy (BCE) from MLE

### Step 1: Bernoulli Probability Model

For binary $y \in \{0,1\}$, the model predicts $\hat{y} = P(y=1\mid x) = \sigma(\beta^T x)$.

Both cases in one expression:

$$P(y \mid x) = \hat{y}^{\,y}(1-\hat{y})^{1-y}$$

### Step 2: Joint Likelihood (N independent observations)

$$L(\beta) = \prod_{i=1}^{N} \hat{y}_i^{\,y_i}(1-\hat{y}_i)^{1-y_i}$$

### Step 3: Log-Likelihood

$$\ell(\beta) = \sum_{i=1}^{N} \left[ y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]$$

### Step 4: Minimise Negative Log-Likelihood = BCE

MLE maximises $\ell$; gradient descent minimises — so flip sign:

$$\boxed{\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]}$$

### Penalty Behaviour

| Case | Loss | |
|---|---|---|
| $y=1,\; \hat{y}\to 1$ | $0$ | Correct, no penalty |
| $y=1,\; \hat{y}\to 0$ | $\to\infty$ | Confident wrong → huge penalty |
| $y=0,\; \hat{y}\to 0$ | $0$ | Correct, no penalty |
| $y=0,\; \hat{y}\to 1$ | $\to\infty$ | Confident wrong → huge penalty |

### Key Insight

$$\underbrace{\text{MLE under Bernoulli}}_{\text{statistical}} \;=\; \underbrace{-\log\text{-likelihood}}_{\text{optimisation}} \;=\; \underbrace{\text{BCE loss}}_{\text{deep learning}}$$

BCE is convex in $\beta$ for a single-layer network (logistic regression) → gradient descent finds the global minimum.

---

## Automatic Differentiation (AutoDiff)

### Why AutoDiff?

Three ways to obtain gradients:

| Approach | Example | Method |
|---|---|---|
| Closed-form solution | Linear Regression | Solve analytically — no gradient needed |
| Hand-derived gradient | Logistic Regression | Derive $\nabla L$ once, hardcode it |
| **AutoDiff** | Deep Learning (everything else) | Computer applies chain rule automatically |

Neural networks have millions of parameters — deriving gradients by hand is impossible. AutoDiff does it automatically by tracking every operation in the forward pass and applying the chain rule in reverse.

### The Chain Rule Foundation

To update weight $w_{ij}^{(\ell)}$, decompose the gradient:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{(\ell)}} = \underbrace{\frac{\partial \mathcal{L}}{\partial a_i^{(\ell)}}}_{\text{upstream error}} \cdot \underbrace{\frac{\partial a_i^{(\ell)}}{\partial z_i^{(\ell)}}}_{\text{local sensitivity } \sigma'} \cdot \underbrace{\frac{\partial z_i^{(\ell)}}{\partial w_{ij}^{(\ell)}}}_{\text{local input } a_j^{(\ell-1)}}$$

### The Error Signal $\delta$

Combine upstream error and local sensitivity into one reusable quantity:

$$\delta_i^{(\ell)} = \frac{\partial \mathcal{L}}{\partial a_i^{(\ell)}} \cdot \sigma'(z_i^{(\ell)})$$

**Vectorised backprop** (pass $\delta$ backwards):

$$\delta^{(\ell)} = \left(W^{(\ell+1)T}\,\delta^{(\ell+1)}\right) \odot \sigma'(z^{(\ell)})$$

**Weight gradient** (the final product):

$$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)} \times \left(a^{(\ell-1)}\right)^T$$

### Why $W^T\delta$? (Multivariate Chain Rule)

Each neuron $a_i^{(\ell)}$ connects to every neuron in the next layer, so:

$$\frac{\partial \mathcal{L}}{\partial a_i^{(\ell)}} = \sum_k \delta_k^{(\ell+1)} \cdot W_{ki}^{(\ell+1)}$$

This summation is exactly the $i$-th element of $\left(W^{(\ell+1)}\right)^T\delta^{(\ell+1)}$.

### Sigmoid is AutoDiff-Friendly

$$\sigma'(x) = \sigma(x)(1-\sigma(x))$$

Once $\sigma(z)$ is cached from the forward pass, the gradient costs only one subtraction and one multiplication — no extra exponentials.

### Full Picture

$$\underbrace{x \to z^{(1)} \to a^{(1)} \to \cdots \to \hat{y} \to \mathcal{L}}_{\text{Forward pass: compute and cache all activations}}$$

$$\underbrace{\delta \text{ flows backwards layer by layer using cached values}}_{\text{Backward pass: chain rule applied mechanically}}$$

$$\underbrace{w \leftarrow w - \eta \cdot \tfrac{\partial \mathcal{L}}{\partial w}}_{\text{Parameter update}}$$
