# Q21-BE — Backpropagation
> Week 10. Could ask: derive the delta rule and gradient formulas, explain the vanishing gradient problem, compare activation functions, or show why ReLU solves the vanishing gradient.

---

## The Model

A feedforward neural network (MLP) with $L$ layers:

$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = g(z^{(l)})$$

- $a^{(0)} = x$ (input), $\hat{y} = a^{(L)}$ (output)
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$: weight matrix for layer $l$
- $b^{(l)} \in \mathbb{R}^{d_l}$: bias vector
- $g$: activation function (applied elementwise)

**Challenge**: to train by gradient descent we need $\partial L / \partial W^{(l)}$ for every layer $l$. Naive finite differences would cost $O(p \times \text{forward pass})$ where $p$ is the number of parameters. Backpropagation computes all gradients in $O(\text{forward pass})$ by applying the chain rule once, storing intermediate activations.

---

## The Forward Pass

Compute and store activations layer by layer:

$$a^{(l)} = g\!\left(W^{(l)}a^{(l-1)} + b^{(l)}\right), \quad l = 1, \ldots, L$$

Define a scalar loss function over the training set:

$$L = \frac{1}{N}\sum_{i=1}^N \ell\!\left(y_i, a^{(L)}_i\right)$$

Common choices: cross-entropy for classification, MSE for regression.

**Why store activations?** The backward pass requires $a^{(l-1)}$ to compute $\partial L / \partial W^{(l)} = \delta^{(l)} (a^{(l-1)})^T$. Storing them during the forward pass avoids recomputing them — a dynamic programming trick.

---

## The Backward Pass (Chain Rule)

Define the **error signal** at layer $l$:

$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}} \in \mathbb{R}^{d_l}$$

**Output layer** (layer $L$):

$$\delta^{(L)} = \nabla_{a^{(L)}} L \;\odot\; g'(z^{(L)})$$

The first term is the gradient of the loss with respect to the network's output; the second applies the chain rule through the activation function. $\odot$ denotes elementwise multiplication.

**Hidden layers** (back-propagate the error signal):

$$\delta^{(l)} = \left(W^{(l+1)T} \delta^{(l+1)}\right) \odot g'(z^{(l)})$$

Interpretation: the error at layer $l$ is the weighted sum of errors from layer $l+1$ (through the transposed weight matrix), modulated by the local gradient of the activation.

**Parameter gradients**:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T, \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

---

## Weight Update (SGD)

Gradient descent update:

$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}, \qquad b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}$$

- $\eta > 0$: learning rate.
- **Mini-batch SGD**: compute gradients on a batch of $B$ observations; update parameters; repeat. Noisier than batch gradient descent but much faster per epoch.
- **Adam**: adaptive per-parameter learning rates; first and second moment estimates; widely used in practice.

---

## Key Properties

**Computational efficiency**: backpropagation runs in $O(\text{forward pass cost})$ by storing activations. All gradients $\partial L / \partial W^{(l)}$ for $l = 1, \ldots, L$ are computed in a single backward sweep.

**Non-convex objective**: the loss surface has many local minima and saddle points. SGD with good initialisation and learning rate schedule converges to a useful local minimum in practice.

**Initialisation**: weights are typically initialised from a small random distribution (Xavier, He). Symmetry breaking: if all weights are identical, all neurons in a layer learn the same function — random initialisation prevents this.

---

## The Vanishing Gradient Problem

For a sigmoid activation $g(z) = 1/(1+e^{-z})$:

$$g'(z) = g(z)(1-g(z)) \in (0, 0.25]$$

The error signal at layer $l$ involves:

$$\delta^{(l)} = \left(W^{(l+1)T}\cdots W^{(L)T}\right) \delta^{(L)} \odot \prod_{k=l}^{L-1} g'(z^{(k)})$$

Each $g'$ factor is at most 0.25. For a network with $L$ layers, the gradient at the first layer involves a product of $L-1$ such factors:

$$\left|\delta^{(1)}\right| \propto 0.25^{L-1}$$

For $L = 20$, this is $\approx 10^{-12}$. **Early layers receive near-zero gradients → they barely train → the network cannot learn hierarchical representations**.

Tanh has the same problem: $g'(z) = 1 - \tanh^2(z) \in (0, 1]$, and saturates symmetrically.

---

## The ReLU Fix

Rectified Linear Unit (ReLU):

$$g(z) = \max(0, z), \qquad g'(z) = \mathbf{1}(z > 0)$$

When $z > 0$, the gradient is exactly 1 — it passes through unchanged. No squashing, no vanishing. The product $\prod_{k=l}^{L-1} g'(z^{(k)})$ is 1 whenever all intermediate pre-activations are positive.

**Dead ReLU problem**: if a unit's pre-activation $z^{(l)}_j < 0$ for all inputs, $g'(z^{(l)}_j) = 0$ always → that unit receives zero gradient forever → it never activates again. Once "dead," it contributes nothing.

**Leaky ReLU** mitigates this:

$$g(z) = \max(0.01z, z), \qquad g'(z) = \begin{cases} 1 & z > 0 \\ 0.01 & z \leq 0 \end{cases}$$

A small non-zero slope for $z < 0$ keeps the gradient alive.

---

## Activation Function Comparison

| Function | Formula | Derivative | Output range | Gradient saturation | Zero-centred? |
|----------|---------|------------|-------------|---------------------|---------------|
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)$ | $(0,1)$ | Yes, for large $|z|$ | No |
| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | $1-\tanh^2(z)$ | $(-1,1)$ | Yes, for large $|z|$ | Yes |
| ReLU | $\max(0,z)$ | $\mathbf{1}(z>0)$ | $[0,\infty)$ | No (for $z>0$) | No |
| Leaky ReLU | $\max(0.01z,z)$ | $1$ or $0.01$ | $(-\infty,\infty)$ | No | Approx. |

**Preferred in deep networks**: ReLU or Leaky ReLU for hidden layers; sigmoid for binary output; softmax for multi-class output.

---

## Comparison to Alternatives

| Property | Backprop + SGD | Finite Differences | BFGS (quasi-Newton) |
|----------|---------------|--------------------|---------------------|
| Gradient cost | $O(\text{forward})$ | $O(p \times \text{forward})$ | $O(\text{forward})$ per step |
| Memory | Store activations | None | Store Hessian approx. ($O(p^2)$) |
| Scales to large $p$? | Yes | No | No |
| Exact gradients? | Yes | No (numerical error) | Via backprop |

---

## Limitations

- **Non-convex optimisation**: convergence to global minimum not guaranteed; result depends on initialisation and learning rate schedule.
- **Vanishing gradient**: mitigated by ReLU but not eliminated for very deep networks; residual connections (ResNets) are the architectural fix.
- **Exploding gradient**: the product of weight matrices can grow exponentially in deep networks. Gradient clipping (cap $\|\nabla L\|$ at a threshold) is the standard mitigation.
- **Computational cost**: one epoch through $N$ examples costs $O(p \cdot N)$. With many parameters and large datasets, hardware acceleration (GPU) is essential.
- **Hyperparameter sensitivity**: learning rate, architecture, and regularisation must all be tuned; no single default setting works universally.

---

## Additional Possible Exam Questions

**Q: Why must activations be stored during the forward pass?**
The gradient $\partial L / \partial W^{(l)} = \delta^{(l)} (a^{(l-1)})^T$ requires $a^{(l-1)}$, the activation of the previous layer computed during the forward pass. Without storing it, you would need to recompute it during the backward pass — an extra $O(L \times \text{forward})$ cost. Backpropagation is efficient precisely because it avoids this redundancy using dynamic programming.

**Q: What does the transpose $W^{(l+1)T}$ in the backward pass represent?**
The forward pass propagates signals forward: $W^{(l+1)}$ transforms the post-activation $a^{(l)}$ into the pre-activation $z^{(l+1)}$. The backward pass propagates error signals backward: the transposed weight matrix $W^{(l+1)T}$ routes the error at layer $l+1$ back to layer $l$. This is the linear algebra expression of the chain rule: $\partial z^{(l+1)} / \partial a^{(l)} = W^{(l+1)}$, so $\partial L / \partial a^{(l)} = W^{(l+1)T} \delta^{(l+1)}$.

**Q: Why does depth (more layers) cause the vanishing gradient problem for sigmoid networks?**
Each sigmoid layer contributes a multiplicative factor $g'(z^{(l)}) \leq 0.25$ to the error signal. For $L$ layers, the gradient at the first layer is suppressed by approximately $0.25^{L-1}$, which approaches zero exponentially. Depth compounds the suppression. For $L = 10$: suppression is $\approx 4 \times 10^{-6}$; for $L = 20$: $\approx 10^{-12}$.

**Q: What is the exploding gradient problem and how is it mitigated?**
If weights are large, the product of weight matrices in the backward pass can grow exponentially with depth, causing gradients to diverge — weights oscillate wildly and training fails. Mitigation: (1) gradient clipping — rescale the gradient if $\|\nabla L\| > $ threshold; (2) careful weight initialisation (He, Xavier); (3) batch normalisation — keeps activations in a controlled range; (4) residual connections — add identity shortcuts that provide constant-magnitude gradient paths.

**Q: How does mini-batch SGD differ from full batch gradient descent, and why is it preferred?**
Full batch gradient descent computes the exact gradient over all $N$ examples per update — computationally expensive for large $N$, requires loading the full dataset into memory. Mini-batch SGD computes noisy gradient estimates over batches of size $B$ (typically $B = 32$–$256$). Benefits: (1) faster iterations; (2) noise helps escape local minima and saddle points; (3) compatible with GPU parallelism. The noise is controlled by batch size — larger batches give smoother but less stochastic updates.
