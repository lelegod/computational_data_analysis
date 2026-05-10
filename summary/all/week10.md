# Week 10 — Artificial Neural Networks

## Overview
Week 10 covers the full pipeline for building and training Artificial Neural Networks (ANNs). Topics include the four core ingredients of deep learning (Data, Objective, Engine, Architecture), gradient descent optimization, loss functions derived from maximum likelihood (MSE for regression, Binary Cross-Entropy for classification), the backpropagation algorithm and its derivation from the chain rule via AutoDiff, and a survey of major deep learning architectures (MLP, CNN, RNN, Autoencoder, Transformer). The lecture emphasizes that deep learning is maximum likelihood estimation at its core.

---

## The Ingredients of Artificial Neural Networks

### Four Primary Ingredients
To build and train a modern deep learning model, four components are needed:

1. **Data (D)**: Observed features **X** and targets **Y**
2. **The Objective (L)**: A loss function to minimize, derived from the Negative Log-Likelihood
3. **The Engine ($\nabla$)**: An optimizer (Gradient Descent) and an algorithm to find gradients (Backpropagation)
4. **Architecture ($f_\theta$)**: The choice of non-linear function approximator (MLPs, CNNs, Transformers)

### Framework Notation
| Concept | Notation |
|---------|----------|
| Observed data | $\mathbf{X} = \{x_1,\ldots,x_N\}$; $x_i \in \mathbb{R}^{C \cdot H \cdot W \cdot D}$ |
| Labels/Targets | $\mathbf{Y} = \{y_1,\ldots,y_N\}$; $y_i \in \mathbb{R}^{M \cdot H \cdot W \cdot D}$ |
| Decision function | $f_\theta(\cdot): X \to Y$ |
| Data distribution | $p(X)$ |
| Joint distribution | $p(X,Y)$ |
| Modeled distribution | $p_\theta(X)$, parameterized by $\theta$ |

- $f_\theta(\cdot)$ can be any deep learning model: MLP, CNN, RNN, Transformer, etc.
- The model is an over-parameterized, non-linear function approximator
- $\theta$ are trainable parameters, typically $|\theta| \gg N$ (more parameters than data points)
- Usually requires large labelled datasets

---

## Optimization Algorithm — Gradient Descent

### Core Principle
All deep learning optimization reduces to maximum likelihood estimation:

$$w^* = \arg\min_w -\log \ell(D; w)$$

| Scenario | Method |
|----------|--------|
| Linear Regression | Analytical closed-form solution |
| Logistic Regression | Analytical gradients → iterative gradient descent |
| Deep Learning / everything else | **Automatic Differentiation (AutoDiff)** |

### Gradient Descent Update Rule

$$w \leftarrow w - \eta \cdot \nabla_w L$$

where:
- $\eta$ is the learning rate (step size)
- $\nabla_w L$ is the gradient of the loss with respect to all parameters
- Repeated until convergence

### Key Property: Non-convexity
- Increasing the number of parameters gives more flexibility (model expressiveness)
- But creates **complex, non-convex loss landscapes** with many local minima
- Gradient descent can get stuck in local minima
- In practice, deep networks often find good enough minima due to the geometry of high-dimensional spaces

### Calculus Review — Key Derivatives
| Function | Derivative |
|----------|------------|
| $f(x) = c$ (constant) | $f'(x) = 0$ |
| $f(x) = cx$ | $f'(x) = c$ |
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ |
| $f(x) = e^x$ | $f'(x) = e^x$ |
| $f(x) = \ln(x)$ | $f'(x) = 1/x$ |
| $f(x) = c/x$ | $f'(x) = -c/x^2$ |
| $f(x) = \sigma(x)$ | $f'(x) = \sigma(x)(1-\sigma(x))$ |

Rules: Sum Rule, Product Rule, Quotient Rule, Chain Rule

---

## Optimization Objective Function

### The Sigmoid Function
- **Definition**: $\sigma(x) = \frac{1}{1+e^{-x}}$
- **Range**: $(0, 1)$ — outputs a probability
- **Derivative derivation**:
  - $\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2}$
  - Factor: $= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}}$
  - Note: $\frac{e^{-x}}{1+e^{-x}} = 1 - \frac{1}{1+e^{-x}} = 1 - \sigma(x)$
  - **Final form**: $\sigma'(x) = \sigma(x)(1-\sigma(x))$
- **Computational advantage**: once $\sigma(x)$ is computed (forward pass), the derivative needs only subtraction and multiplication — no exponential recomputation

### Loss for Regression: Mean Squared Error
Derived from the Gaussian likelihood assumption:

$$L(w) = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2$$

Minimizing MSE is equivalent to maximizing the Gaussian log-likelihood.

### Loss for Binary Classification: Binary Cross-Entropy
Derived from the Bernoulli likelihood:
- Model output: $\hat{y}_i = h_w(x_i)$ represents $P(y_i=1|x_i)$
- Bernoulli likelihood for one point: $p(y_i|x_i, w) = \hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i}$
- Log-likelihood for the whole dataset:

$$\ln L(w) = \sum_i \left[y_i \ln(\hat{y}_i) + (1-y_i) \ln(1-\hat{y}_i)\right]$$

- Negative log-likelihood (to minimize):

$$-\ln L(w) = -\sum_i \left[y_i \ln(\hat{y}_i) + (1-y_i) \ln(1-\hat{y}_i)\right]$$

This is exactly **Binary Cross-Entropy**. Minimizing it = maximizing probability of the correct class.

---

## Deep Learning — Backpropagation and AutoDiff

### MLP Architecture
- **Input layer**: $x_0, x_1, \ldots, x_d$
- **Hidden layers**: each node computes weighted sum then applies non-linear activation
- **Output layer**: computes final prediction $\hat{y}$
- Weights $W^{(\ell)}$ connect layer $\ell-1$ to layer $\ell$
- Loss $L(Y, \hat{Y})$ computed at output

**Layer computations (forward pass):**
- Pre-activation: $z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}$
- Activation: $a^{(\ell)} = \sigma(z^{(\ell)})$

### The Core Insight
> "The secret sauce of Deep Learning is just recursive calculus applied to massive matrices."

Deep learning = Automatic Differentiation through a computational graph.

### Three Phases of Training (Repeated Until Convergence)

**Phase 1 — Forward Pass (Signal Flow):**
- Input $x$ flows layer by layer through the network
- At each layer: compute $z^{(\ell)} = W^{(\ell)}a^{(\ell-1)} + b^{(\ell)}$ and $a^{(\ell)} = \sigma(z^{(\ell)})$
- Final output $\hat{y}$ and loss $L(y, \hat{y})$ computed
- All intermediate values $z^{(\ell)}$ and $a^{(\ell)}$ stored (needed for backward pass)

**Phase 2 — Backward Pass (Error Credit Assignment):**
- Use the Chain Rule in reverse through the computational graph
- Compute error signal $\delta^{(\ell)}$ for each layer
- Answers: "How much did each neuron contribute to the final error?"

**Phase 3 — Parameter Update (The Nudge):**
- Compute gradient $\nabla_w L$ from error signals
- Nudge every weight in the opposite direction: $w \leftarrow w - \eta \cdot \nabla_w L$

**Key principle:** Signal (activations) go forward; blame (gradients) flows backward.

### AutoDiff: Chain Rule Decomposition
Gradient of loss $L$ w.r.t. weight $W^{(\ell)}_{ij}$:

$$\frac{\partial L}{\partial W^{(\ell)}_{ij}} = \frac{\partial L}{\partial a^{(\ell)}_i} \cdot \frac{\partial a^{(\ell)}_i}{\partial z^{(\ell)}_i} \cdot \frac{\partial z^{(\ell)}_i}{\partial W^{(\ell)}_{ij}}$$

= **(Upstream Error) · (Local Sensitivity) · (Local Input)**

Each term:
- $\frac{\partial L}{\partial a^{(\ell)}_i}$: How much loss changes when this activation changes (comes from layer above)
- $\frac{\partial a^{(\ell)}_i}{\partial z^{(\ell)}_i}$: Sensitivity of activation to pre-activation $= \sigma'(z^{(\ell)}_i) = \sigma(z)(1-\sigma(z))$
- $\frac{\partial z^{(\ell)}_i}{\partial W^{(\ell)}_{ij}}$: $= a^{(\ell-1)}_j$ (just the input activation from previous layer)

### The Error Signal $\delta$
Combine the first two chain rule terms into a single vector $\delta^{(\ell)}$:

$$\delta^{(\ell)}_i = \frac{\partial L}{\partial a^{(\ell)}_i} \cdot \frac{\partial a^{(\ell)}_i}{\partial z^{(\ell)}_i} = \text{(Upstream error)} \times \text{(Local sensitivity)}$$

**Vectorized Backprop (recursive across layers):**

$$\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$$

where $\odot$ = element-wise multiplication.

**Weight Gradient:**

$$\frac{\partial L}{\partial W^{(\ell)}} = \delta^{(\ell)} \times (a^{(\ell-1)})^T$$

= (Error signal at this layer) × (Input activation from previous layer)

### Why $(W^T \delta)$? The Multivariate Chain Rule
In a dense layer, activation $a^{(\ell)}_i$ flows into **every** pre-activation $z^{(\ell+1)}_k$ in the next layer. By the multivariate chain rule:

$$\frac{\partial L}{\partial a^{(\ell)}_i} = \sum_k \frac{\partial L}{\partial z^{(\ell+1)}_k} \cdot \frac{\partial z^{(\ell+1)}_k}{\partial a^{(\ell)}_i} = \sum_k \delta^{(\ell+1)}_k \cdot W^{(\ell+1)}_{ki}$$

In vector form, this sum $\sum_k \delta^{(\ell+1)}_k W^{(\ell+1)}_{ki}$ is exactly the $i$-th element of $(W^{(\ell+1)})^T \delta^{(\ell+1)}$.

---

## Architectures

### Common Deep Learning Architectures

**Multi-Layer Perceptron (MLP):**
- Fully connected layers
- Each neuron connects to every neuron in the next layer
- Used for tabular data, fixed-size inputs
- Parameters scale with: (input_dim × hidden_dim) per layer

**Convolutional Neural Networks (CNN):**
- Designed for grid-structured data (images, time series)
- Convolutional layers share weights spatially → translation equivariance
- Pooling layers reduce spatial dimensions
- Much fewer parameters than equivalent fully-connected network
- Classic architectures: LeNet (LeCun 1998), AlexNet, VGG, ResNet

**Recurrent Neural Networks (RNN):**
- Designed for sequential data (text, time series)
- Hidden state carries information across time steps
- Suffers from vanishing gradient for long sequences
- Variants: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit)

**Autoencoders:**
- Unsupervised architecture: encoder compresses input → bottleneck → decoder reconstructs
- Encoder: $X \to z$ (latent code, low-dimensional representation)
- Decoder: $z \to \hat{X}$ (reconstruction)
- Loss: reconstruction error $\|X - \hat{X}\|^2$
- Applications: dimensionality reduction, anomaly detection, denoising

**Transformers:**
- Attention-based architecture — "Attention is All You Need" (Vaswani et al. 2017)
- Self-attention mechanism: each token attends to all other tokens
- Basis of all modern LLMs (BERT, GPT, etc.)
- Parallelizable (unlike RNNs) → scales well with data and compute
- Key operation: $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

### Historical Context
- Rumelhart et al. (1985): Backpropagation popularized
- LeCun et al. (1998): CNNs applied to digit recognition
- 2012: AlexNet — deep learning "ImageNet moment"
- Modern era: Big Data + Big Compute + better optimization = transformative results
