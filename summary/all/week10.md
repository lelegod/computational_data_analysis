# Week 10 — Artificial Neural Networks

## Overview
Week 10 covers the full pipeline for building and training Artificial Neural Networks (ANNs). Topics include the four core ingredients of deep learning (Data, Objective, Engine, Architecture), gradient descent optimization, loss functions derived from maximum likelihood (MSE for regression, Binary Cross-Entropy for classification), the backpropagation algorithm and its derivation from the chain rule via AutoDiff, and a survey of major deep learning architectures (MLP, CNN, RNN, Autoencoder, Transformer). The lecture emphasizes that deep learning is maximum likelihood estimation at its core.

---

## The Ingredients of Artificial Neural Networks

### Four Primary Ingredients
To build and train a modern deep learning model, four components are needed:

1. **Data (D)**: Observed features **X** and targets **Y**
2. **The Objective (L)**: A loss function to minimize, derived from the Negative Log-Likelihood
3. **The Engine (∇)**: An optimizer (Gradient Descent) and an algorithm to find gradients (Backpropagation)
4. **Architecture (f_θ)**: The choice of non-linear function approximator (MLPs, CNNs, Transformers)

### Framework Notation
| Concept | Notation |
|---------|----------|
| Observed data | **X** = {x₁,...,x_N}; xᵢ ∈ R^(C·H·W·D) |
| Labels/Targets | **Y** = {y₁,...,y_N}; yᵢ ∈ R^(M·H·W·D) |
| Decision function | f_θ(·): X → Y |
| Data distribution | p(X) |
| Joint distribution | p(X,Y) |
| Modeled distribution | p_θ(X), parameterized by θ |

- f_θ(·) can be any deep learning model: MLP, CNN, RNN, Transformer, etc.
- The model is an over-parameterized, non-linear function approximator
- θ are trainable parameters, typically |θ| >> N (more parameters than data points)
- Usually requires large labelled datasets

---

## Optimization Algorithm — Gradient Descent

### Core Principle
All deep learning optimization reduces to maximum likelihood estimation:

`w* = argmin_w −log ℓ(D; w)`

| Scenario | Method |
|----------|--------|
| Linear Regression | Analytical closed-form solution |
| Logistic Regression | Analytical gradients → iterative gradient descent |
| Deep Learning / everything else | **Automatic Differentiation (AutoDiff)** |

### Gradient Descent Update Rule
`w ← w − η · ∇_w L`

where:
- η is the learning rate (step size)
- ∇_w L is the gradient of the loss with respect to all parameters
- Repeated until convergence

### Key Property: Non-convexity
- Increasing the number of parameters gives more flexibility (model expressiveness)
- But creates **complex, non-convex loss landscapes** with many local minima
- Gradient descent can get stuck in local minima
- In practice, deep networks often find good enough minima due to the geometry of high-dimensional spaces

### Calculus Review — Key Derivatives
| Function | Derivative |
|----------|------------|
| f(x) = c (constant) | f'(x) = 0 |
| f(x) = cx | f'(x) = c |
| f(x) = xⁿ | f'(x) = nxⁿ⁻¹ |
| f(x) = eˣ | f'(x) = eˣ |
| f(x) = ln(x) | f'(x) = 1/x |
| f(x) = c/x | f'(x) = −c/x² |
| f(x) = σ(x) | f'(x) = σ(x)(1−σ(x)) |

Rules: Sum Rule, Product Rule, Quotient Rule, Chain Rule

---

## Optimization Objective Function

### The Sigmoid Function
- **Definition**: `σ(x) = 1/(1+e^{−x})`
- **Range**: (0, 1) — outputs a probability
- **Derivative derivation**:
  - `σ'(x) = e^{−x}/(1+e^{−x})²`
  - Factor: `= [1/(1+e^{−x})] · [e^{−x}/(1+e^{−x})]`
  - Note: `e^{−x}/(1+e^{−x}) = 1 − 1/(1+e^{−x}) = 1 − σ(x)`
  - **Final form**: `σ'(x) = σ(x)(1−σ(x))`
- **Computational advantage**: once σ(x) is computed (forward pass), the derivative needs only subtraction and multiplication — no exponential recomputation

### Loss for Regression: Mean Squared Error
Derived from the Gaussian likelihood assumption:

`L(w) = (1/N) Σᵢ (yᵢ − ŷᵢ)²`

Minimizing MSE is equivalent to maximizing the Gaussian log-likelihood.

### Loss for Binary Classification: Binary Cross-Entropy
Derived from the Bernoulli likelihood:
- Model output: `ŷᵢ = h_w(xᵢ)` represents P(yᵢ=1|xᵢ)
- Bernoulli likelihood for one point: `p(yᵢ|xᵢ, w) = ŷᵢ^{yᵢ}(1−ŷᵢ)^{1−yᵢ}`
- Log-likelihood for the whole dataset:
  `ln L(w) = Σᵢ [yᵢ ln(ŷᵢ) + (1−yᵢ) ln(1−ŷᵢ)]`
- Negative log-likelihood (to minimize):
  `−ln L(w) = −Σᵢ [yᵢ ln(ŷᵢ) + (1−yᵢ) ln(1−ŷᵢ)]`

This is exactly **Binary Cross-Entropy**. Minimizing it = maximizing probability of the correct class.

---

## Deep Learning — Backpropagation and AutoDiff

### MLP Architecture
- **Input layer**: x₀, x₁, ..., x_d
- **Hidden layers**: each node computes weighted sum then applies non-linear activation
- **Output layer**: computes final prediction ŷ
- Weights W^(ℓ) connect layer ℓ−1 to layer ℓ
- Loss L(Y, Ŷ) computed at output

**Layer computations (forward pass):**
- Pre-activation: `z^(ℓ) = W^(ℓ) a^(ℓ−1) + b^(ℓ)`
- Activation: `a^(ℓ) = σ(z^(ℓ))`

### The Core Insight
> "The secret sauce of Deep Learning is just recursive calculus applied to massive matrices."

Deep learning = Automatic Differentiation through a computational graph.

### Three Phases of Training (Repeated Until Convergence)

**Phase 1 — Forward Pass (Signal Flow):**
- Input x flows layer by layer through the network
- At each layer: compute `z^(ℓ) = W^(ℓ)a^(ℓ−1) + b^(ℓ)` and `a^(ℓ) = σ(z^(ℓ))`
- Final output ŷ and loss L(y, ŷ) computed
- All intermediate values z^(ℓ) and a^(ℓ) stored (needed for backward pass)

**Phase 2 — Backward Pass (Error Credit Assignment):**
- Use the Chain Rule in reverse through the computational graph
- Compute error signal δ^(ℓ) for each layer
- Answers: "How much did each neuron contribute to the final error?"

**Phase 3 — Parameter Update (The Nudge):**
- Compute gradient ∇_w L from error signals
- Nudge every weight in the opposite direction: `w ← w − η · ∇_w L`

**Key principle:** Signal (activations) go forward; blame (gradients) flows backward.

### AutoDiff: Chain Rule Decomposition
Gradient of loss L w.r.t. weight W^(ℓ)_{ij}:

`∂L/∂W^(ℓ)_{ij} = (∂L/∂a^(ℓ)_i) · (∂a^(ℓ)_i/∂z^(ℓ)_i) · (∂z^(ℓ)_i/∂W^(ℓ)_{ij})`

= **(Upstream Error) · (Local Sensitivity) · (Local Input)**

Each term:
- **∂L/∂a^(ℓ)_i**: How much loss changes when this activation changes (comes from layer above)
- **∂a^(ℓ)_i/∂z^(ℓ)_i**: Sensitivity of activation to pre-activation = σ'(z^(ℓ)_i) = σ(z)(1−σ(z))
- **∂z^(ℓ)_i/∂W^(ℓ)_{ij}**: = a^(ℓ−1)_j (just the input activation from previous layer)

### The Error Signal δ
Combine the first two chain rule terms into a single vector δ^(ℓ):

`δ^(ℓ)_i = (∂L/∂a^(ℓ)_i) · (∂a^(ℓ)_i/∂z^(ℓ)_i)` = (Upstream error) × (Local sensitivity)

**Vectorized Backprop (recursive across layers):**

`δ^(ℓ) = (W^(ℓ+1))^T δ^(ℓ+1) ⊙ σ'(z^(ℓ))`

where ⊙ = element-wise multiplication.

**Weight Gradient:**

`∂L/∂W^(ℓ) = δ^(ℓ) × (a^(ℓ−1))^T`

= (Error signal at this layer) × (Input activation from previous layer)

### Why (W^T δ)? The Multivariate Chain Rule
In a dense layer, activation a^(ℓ)_i flows into **every** pre-activation z^(ℓ+1)_k in the next layer. By the multivariate chain rule:

`∂L/∂a^(ℓ)_i = Σ_k (∂L/∂z^(ℓ+1)_k) · (∂z^(ℓ+1)_k/∂a^(ℓ)_i) = Σ_k δ^(ℓ+1)_k · W^(ℓ+1)_{ki}`

In vector form, this sum `Σ_k δ^(ℓ+1)_k W^(ℓ+1)_{ki}` is exactly the i-th element of `(W^(ℓ+1))^T δ^(ℓ+1)`.

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
- Encoder: X → z (latent code, low-dimensional representation)
- Decoder: z → X̂ (reconstruction)
- Loss: reconstruction error ||X − X̂||²
- Applications: dimensionality reduction, anomaly detection, denoising

**Transformers:**
- Attention-based architecture — "Attention is All You Need" (Vaswani et al. 2017)
- Self-attention mechanism: each token attends to all other tokens
- Basis of all modern LLMs (BERT, GPT, etc.)
- Parallelizable (unlike RNNs) → scales well with data and compute
- Key operation: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

### Historical Context
- Rumelhart et al. (1985): Backpropagation popularized
- LeCun et al. (1998): CNNs applied to digit recognition
- 2012: AlexNet — deep learning "ImageNet moment"
- Modern era: Big Data + Big Compute + better optimization = transformative results
