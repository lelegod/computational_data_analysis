# Week 10 — Group Discussion Questions

## Q1: More on Gradient Descent — Local vs Global Minima

**Question (slide 7):** Two functions are shown:
- (A) A smooth periodic function with multiple equal-height minima. The red dot is placed at $x \approx +0.7$ (near a local minimum).
- (B) A complex non-convex function with a clear global minimum far to the left. The red dot is at $x \approx -0.5$ (near a local, non-global minimum).

Discuss in groups and answer the following questions:

- **A.1** Where does function (A) attain its minimum?
- **A.2** If we initialize gradient descent at the red point in (A), where would the model converge?
- **B.1** Where is function (B) at its minimum?
- **B.2** If we initialize gradient descent at the red point in (B), where would the model converge?
- **B.3** How to get the model to converge to another/better minimum?

**Answer:**

**A.1** Function (A) is a smooth periodic function (resembling a cosine/sine wave). It attains its global minimum at multiple symmetric locations — at all the troughs of the wave. Since the function appears symmetric, all minima have equal function values, so every trough is a global minimum.

**A.2** Gradient descent initialized at the red point (near a local minimum) will follow the negative gradient downhill. Since the red point is already near a local trough in (A), the model converges to the nearest local minimum — which in this case is also a global minimum (all troughs are equally deep). The model does converge to *a* global minimum, but not necessarily the one the user might want if there is a preferred location.

**B.1** Function (B) is non-convex with a prominent deep minimum far to the left (around $x \approx -2$). The global minimum is that deep leftmost trough.

**B.2** Gradient descent initialized at the red point in (B) (around $x \approx -0.5$) follows the local gradient. There is a local minimum nearby on the right side of the curve. The gradient at the red point points toward this nearby local minimum, so the model converges to the **local** minimum close to the initialization point — NOT the global minimum. This is the classic problem of gradient descent being trapped by local minima.

**B.3** Strategies to escape a local minimum and find a better one:
- **Random restarts**: Run gradient descent multiple times from different random starting points and keep the solution with the lowest loss.
- **Stochastic Gradient Descent (SGD)**: Using mini-batches introduces noise into the gradient estimates, which can help the optimizer "bounce out" of shallow local minima.
- **Momentum-based optimizers** (e.g., SGD with momentum, Adam): These carry velocity from past gradients and can roll through shallow local minima.
- **Simulated annealing / large learning rate early on**: A larger learning rate early in training allows larger jumps that skip over local minima. Schedules that decay $\eta$ over time (warm restarts, cosine annealing) exploit this.
- **Increase model capacity / change architecture**: Sometimes a better-conditioned loss landscape has fewer problematic local minima.

---

## Q2: Activity — The Learning Rate

**Question (slide 8):** Minimize $f(w) = w^2$, where $\nabla f(w) = 2w$. Start at $w_0 = 10$.

The gradient descent update rule is $w \leftarrow w - \eta \cdot \nabla f(w)$.

- **Scenario A**: $\eta = 0.1$ — compute $w_1 =$ ?
- **Scenario B**: $\eta = 1.1$ — compute $w_1 =$ ?

Discussion:
- Which scenario converged?
- What happened in Scenario B? Did we go "downhill"?

**Answer:**

**Scenario A** ($\eta = 0.1$):
$$w_1 = w_0 - \eta \cdot \nabla f(w_0) = 10 - 0.1 \times (2 \times 10) = 10 - 2 = \mathbf{8}$$

The step moves from $w=10$ toward $w=0$ (the minimum). $f(w_1) = 64 < f(w_0) = 100$. We went downhill. With repeated steps, $w$ will converge to 0.

**Scenario B** ($\eta = 1.1$):
$$w_1 = w_0 - \eta \cdot \nabla f(w_0) = 10 - 1.1 \times (2 \times 10) = 10 - 22 = \mathbf{-12}$$

$f(w_1) = (-12)^2 = 144 > f(w_0) = 100$. We went **uphill** — the loss increased! The step overshot past the minimum and ended up on the other side, further away.

**Which scenario converged?** Scenario A converges. For $f(w) = w^2$, gradient descent converges when $\eta < \frac{1}{L}$ where $L$ is the Lipschitz constant of the gradient. Here $\nabla^2 f = 2$, so convergence requires $\eta < \frac{1}{2} = 0.5$. With $\eta = 0.1 < 0.5$, Scenario A converges geometrically to zero.

**What happened in Scenario B?** With $\eta = 1.1 > 0.5$, each step overshoots past the minimum and the iterates diverge: $10 \to -12 \to 26.4 \to -58.1 \to \ldots$ The algorithm **diverges**. Even though we moved in the direction of the negative gradient, the step was so large that we ended up further from the minimum than before. This illustrates that a learning rate that is too large can cause divergence, not just slow convergence.

---

## Q3: Activity — Think Pair Share: Backpropagation

**Question (slide 32):** A single-layer network with sigmoid activation. Given:
- Input activation: $a^{(0)} = 2.0$
- Target: $y = 1.0$ (Positive class)
- Current weight: $W^{(1)} = -1.0$
- Learning rate: $\eta = 0.1$
- Loss: $\mathcal{L} = \frac{1}{2}(a^{(1)} - y)^2$

The model is currently 12% confident. Use the backprop equations to find the new weight. Poll: should $W$ move in direction **A) More Positive**, **B) More Negative**, or **C) No Change**?

**Answer:**

**Step 1 — Forward Pass:**

Pre-activation:
$$z^{(1)} = W^{(1)} a^{(0)} = (-1.0)(2.0) = -2.0$$

Activation (sigmoid):
$$a^{(1)} = \sigma(z^{(1)}) = \frac{1}{1 + e^{2.0}} = \frac{1}{1 + 7.389} \approx 0.119 \approx \mathbf{0.12}$$

This confirms the "12% confident" statement.

**Step 2 — Backward Pass (error signal $\delta$):**

The loss gradient with respect to $a^{(1)}$:
$$\frac{\partial \mathcal{L}}{\partial a^{(1)}} = a^{(1)} - y = 0.12 - 1.0 = -0.88$$

Local sensitivity (sigmoid derivative, using $\sigma'(z) = \sigma(z)(1-\sigma(z))$):
$$\sigma'(z^{(1)}) = 0.12 \times (1 - 0.12) = 0.12 \times 0.88 = 0.1056$$

Error signal:
$$\delta^{(1)} = \frac{\partial \mathcal{L}}{\partial a^{(1)}} \cdot \sigma'(z^{(1)}) = (-0.88)(0.1056) \approx -0.093$$

**Step 3 — Weight Gradient:**

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \delta^{(1)} \cdot (a^{(0)})^T = (-0.093)(2.0) \approx -0.186$$

**Step 4 — Parameter Update:**

$$W^{(1)} \leftarrow W^{(1)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial W^{(1)}} = -1.0 - (0.1)(-0.186) = -1.0 + 0.0186 \approx \mathbf{-0.981}$$

**Answer to Vevox Poll: A — More Positive.** The weight moved from $-1.0$ to $-0.981$, i.e., it became less negative / more positive. This makes intuitive sense: the model predicted $\approx 0.12$ but the target is $1.0$. A negative weight suppresses the output when the input is positive. To increase the output toward 1.0, the weight must become less negative (more positive), allowing the input signal $a^{(0)} = 2.0$ to produce a larger pre-activation $z$ and thus a larger sigmoid output.
