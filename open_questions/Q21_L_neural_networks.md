# Q21-L — Neural Networks and Backpropagation
> Week 10. Could be asked to derive backprop, explain the architecture, or compare to other methods.

---

## The Four Ingredients of a Neural Network

1. **Architecture**: number of layers, units per layer, connectivity
2. **Activation functions**: introduce nonlinearity
3. **Loss function**: defines what "good predictions" means
4. **Optimization algorithm**: gradient descent + backpropagation

---

## MLP Architecture (Feedforward)

$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}, \quad a^{(l)} = g(z^{(l)})$$

- Layer $l$: pre-activation $z^{(l)}$, post-activation $a^{(l)}$
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$: weight matrix
- $b^{(l)} \in \mathbb{R}^{d_l}$: bias vector
- $g$: activation function (applied elementwise)
- $a^{(0)} = x$ (input), $\hat{y} = a^{(L)}$ (output)

**Universal approximation theorem**: an MLP with one hidden layer and enough units can approximate any continuous function on a compact domain. More layers = more efficient representation for complex functions.

---

## Activation Functions

| Function | Formula | Derivative | Properties |
|----------|---------|-----------|------------|
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z))$ | Output $\in(0,1)$; vanishing gradient for large $|z|$ |
| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | $1-\tanh^2(z)$ | Output $\in(-1,1)$; zero-centered |
| ReLU | $\max(0,z)$ | $\mathbf{I}(z>0)$ | No vanishing gradient for $z>0$; dead neurons for $z<0$ |
| Softmax | $e^{z_k}/\sum_j e^{z_j}$ | — | Output layer, multi-class; outputs sum to 1 |

**Why ReLU is preferred in deep networks**: sigmoid/tanh gradients vanish for large $|z|$ (saturate) → gradients in early layers become $\approx 0$ → slow/no learning. ReLU gradient is constant 1 for $z>0$ → gradient flows through.

---

## Loss Functions

| Task | Loss | Formula |
|------|------|---------|
| Regression | MSE | $\frac{1}{N}\sum_i(y_i-\hat{y}_i)^2$ |
| Binary classification | Binary cross-entropy | $-\frac{1}{N}\sum_i[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)]$ |
| Multi-class | Cross-entropy | $-\frac{1}{N}\sum_i\sum_k y_{ik}\log\hat{p}_{ik}$ |

---

## Backpropagation Algorithm

Backpropagation = **chain rule applied efficiently** to compute $\partial L/\partial W^{(l)}$ for all layers.

**Forward pass** (store activations):
$$a^{(l)} = g(W^{(l)}a^{(l-1)}+b^{(l)}) \quad \text{for } l=1,\ldots,L$$

**Backward pass** (compute gradients layer by layer):

Define error signal at layer $l$:
$$\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \odot g'(z^{(L)})$$

Backpropagate:
$$\delta^{(l)} = \left[(W^{(l+1)})^T\delta^{(l+1)}\right] \odot g'(z^{(l)})$$

Gradients:
$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^T, \quad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

**Key insight**: by storing activations from the forward pass, backprop computes all gradients in $O(\text{forward pass cost})$ — as cheap as one forward pass. Contrast with finite differences: $O(p \times \text{forward cost})$ where $p$ = number of parameters.

---

## Gradient Descent and Variants

**Batch gradient descent**:
$$W \leftarrow W - \eta \frac{1}{N}\sum_i \nabla_{W}L_i$$
Uses full dataset per update. Slow for large $N$.

**Stochastic gradient descent (SGD)**: update with one sample at a time. Noisy but fast.

**Mini-batch SGD**: update with a batch of $B$ samples. Standard in deep learning. Balance of speed and stability.

**Learning rate $\eta$**:
- Too large: diverges (overshoots minimum)
- Too small: slow convergence
- Common: learning rate schedules (decay), Adam optimizer (adaptive per-parameter $\eta$)

---

## Regularization in Neural Networks

| Method | Mechanism |
|--------|-----------|
| $L_2$ weight decay | Add $\lambda\|W\|_F^2$ to loss → penalizes large weights |
| Dropout | Randomly zero out units during training with probability $p$ → forces redundancy |
| Early stopping | Stop training when validation loss starts increasing → implicit regularization |
| Batch normalization | Normalize layer inputs → stabilizes training, acts as regularizer |

---

## Neural Networks vs Other Methods

| Property | Neural Network | SVM | Random Forest |
|----------|---------------|-----|---------------|
| Interpretability | Low | Low | Moderate (feature importance) |
| Feature engineering | Learns features automatically | Needs kernel design | None needed |
| $p \gg n$ | Needs regularization | Handles well (dual) | Handles well |
| Training cost | High ($O(pN)$ per epoch) | $O(N^2)$–$O(N^3)$ | $O(BNm\log N)$ |
| Probabilistic output | With softmax + calibration | Via Platt scaling | With proportion |
| Hyperparameters | Many (architecture, $\eta$, regularization) | Few ($C$, kernel) | Few ($B$, $m$) |

---

## Additional Possible Exam Questions

**Q: Why does backpropagation require the forward pass first?**
The backward pass uses $a^{(l-1)}$ (the pre-activation from the forward pass) to compute $\partial L/\partial W^{(l)} = \delta^{(l)}(a^{(l-1)})^T$. Without storing the activations during the forward pass, you would need to recompute them for every layer during backprop — much more expensive. Backprop is a dynamic programming algorithm: it stores intermediate results (activations) to avoid redundant computation.

**Q: What is the vanishing gradient problem?**
In deep networks with sigmoid/tanh activations, $|g'(z)| < 1$ everywhere. After backpropagating through $L$ layers: $\delta^{(1)} \propto \prod_{l=1}^L g'(z^{(l)})$. Each factor $< 1$ → product shrinks exponentially with depth → gradients in early layers are nearly zero → early layers learn extremely slowly or not at all. Solution: ReLU activations (gradient = 1 for positive inputs), batch normalization, residual connections.

**Q: What does dropout do at test time?**
During training: randomly zero out each unit with probability $p$ (typically $p=0.5$). During test: use all units but scale weights by $(1-p)$ (or equivalently: scale activations by $(1-p)$ at test time). This ensures the expected activation at test time equals the expected activation during training. Dropout approximates training an ensemble of $2^d$ sub-networks (where $d$ = number of units), making predictions more robust.

**Q: How does a neural network with sigmoid output differ from logistic regression?**
Logistic regression: $\hat{p} = \sigma(x^T\beta)$ — a single linear layer followed by sigmoid. Neural network with hidden layers: $\hat{p} = \sigma(W^{(L)}\cdots g(W^{(1)}x))$ — applies nonlinear transformations before the sigmoid. With zero hidden layers, a neural network reduces exactly to logistic regression. Hidden layers allow the network to learn a nonlinear feature representation of $x$.

**Q: What is the universal approximation theorem and why is it not sufficient for practice?**
The theorem states that an MLP with one hidden layer and enough units can approximate any continuous function. But it says nothing about: (1) how many units are needed (may be exponential), (2) whether gradient descent finds the approximation (optimization is non-convex), (3) whether the network generalizes from finite training data. In practice, depth (many layers with few units) is more efficient than width (one layer with many units) for many real-world functions.
