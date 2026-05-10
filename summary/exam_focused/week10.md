# Week 10 — Artificial Neural Networks (Exam Focus)

## Must-Know Facts

### The Four Ingredients
- The four ingredients of ANN are: (1) Data, (2) Objective/Loss, (3) Engine/Optimizer+Backprop, (4) Architecture
- The objective/loss is ALWAYS derived from the Negative Log-Likelihood — not an arbitrary design choice
- w* = argmin_w −log ℓ(D; w) is the universal formulation for ALL supervised deep learning

### Gradient Descent
- Gradient descent update: w ← w − η · ∇_w L (move opposite to gradient)
- η is the learning rate (step size); too large → diverge; too small → slow convergence
- Increasing parameters increases flexibility BUT creates non-convex loss landscapes with local minima
- Deep networks operate in non-convex settings — no guarantee of finding global minimum

### Sigmoid and its Derivative
- σ(x) = 1/(1+e^{−x}); maps any real number to (0,1)
- σ'(x) = σ(x)(1−σ(x)) — derivable from just the sigmoid output, no e^{−x} recomputation
- This "self-referential" derivative makes backprop through sigmoid computationally cheap
- Sigmoid output is used directly as P(y=1|x) in binary classification

### Loss Functions
- MSE for regression ↔ Gaussian likelihood assumption
- Binary Cross-Entropy for classification ↔ Bernoulli likelihood assumption
- BCE formula: −Σᵢ [yᵢ ln(ŷᵢ) + (1−yᵢ) ln(1−ŷᵢ)]
- Minimizing BCE = maximizing the probability assigned to the correct class
- These are not separate inventions — they both come from the same MLE framework

### Backpropagation
- Backprop is the Chain Rule applied recursively through a computational graph
- Three phases: (1) Forward pass — compute activations and loss; (2) Backward pass — propagate error signals; (3) Update — nudge weights
- Activations (signal) flow forward; gradients (blame) flow backward
- All intermediate values z^(ℓ) and a^(ℓ) must be STORED during forward pass (needed for backward)
- Error signal at layer ℓ: δ^(ℓ) = (W^(ℓ+1))^T δ^(ℓ+1) ⊙ σ'(z^(ℓ))
- Weight gradient at layer ℓ: ∂L/∂W^(ℓ) = δ^(ℓ) × (a^(ℓ−1))^T
- The (W^T δ) term: because each activation in layer ℓ connects to ALL neurons in layer ℓ+1, the error must sum over all those paths (multivariate chain rule)

### MLP Architecture
- Pre-activation: z^(ℓ) = W^(ℓ) a^(ℓ−1) + b^(ℓ)
- Activation: a^(ℓ) = σ(z^(ℓ))
- Output layer has no activation (regression) or sigmoid/softmax (classification)

### Architecture Comparisons
- MLP: fully connected, for tabular/fixed-size data
- CNN: weight sharing, translation equivariance, for images/grids
- RNN: sequential, hidden state — suffers vanishing gradient; LSTM/GRU fix this
- Autoencoder: encoder → bottleneck → decoder; unsupervised; learns latent representation
- Transformer: attention-based, parallelizable, scales with data — basis of all modern LLMs

---

## Must-Know Formulas

| Formula | What it is | When to use |
|---------|-----------|-------------|
| `w* = argmin_w −log ℓ(D;w)` | Universal MLE objective | Any "what is the objective?" question |
| `w ← w − η · ∇_w L` | Gradient descent update | Optimization questions |
| `σ(x) = 1/(1+e^{−x})` | Sigmoid function | Classification, computing probabilities |
| `σ'(x) = σ(x)(1−σ(x))` | Sigmoid derivative | Backprop through sigmoid layers |
| `−Σᵢ[yᵢln(ŷᵢ)+(1−yᵢ)ln(1−ŷᵢ)]` | Binary Cross-Entropy Loss | Binary classification loss |
| `z^(ℓ) = W^(ℓ)a^(ℓ−1) + b^(ℓ)` | Pre-activation (forward pass) | Describing MLP forward pass |
| `a^(ℓ) = σ(z^(ℓ))` | Activation (forward pass) | Describing MLP forward pass |
| `δ^(ℓ) = (W^(ℓ+1))^T δ^(ℓ+1) ⊙ σ'(z^(ℓ))` | Backprop error signal | Backpropagation algorithm |
| `∂L/∂W^(ℓ) = δ^(ℓ) × (a^(ℓ−1))^T` | Weight gradient | Backpropagation algorithm |
| `∂L/∂a^(ℓ)_i = Σ_k δ^(ℓ+1)_k · W^(ℓ+1)_{ki}` | Multivariate chain rule | Why (W^T δ) appears |

---

## Common Traps (Wrong Answers in Exams)

- Binary cross-entropy is an arbitrary design choice → BCE is the EXACT negative log-likelihood under Bernoulli; it's mathematically derived, not chosen
- MSE is arbitrary → MSE is the negative log-likelihood under Gaussian assumptions
- σ'(x) requires recomputing e^{−x} → σ'(x) = σ(x)(1−σ(x)) — only needs the output of the sigmoid, not e^{−x}
- Backprop stores nothing during the forward pass → WRONG — intermediate values z^(ℓ) and a^(ℓ) MUST be stored; they are required for the backward pass
- The gradient flows forward → WRONG — activations flow forward, gradients flow backward
- Vanishing gradient is a problem for CNNs → Vanishing gradient is the specific problem of RNNs on long sequences; CNNs use local connectivity which doesn't suffer from this
- Autoencoders are supervised → Autoencoders are UNSUPERVISED; the reconstruction loss uses the input as its own target
- Transformers use RNN-style sequential processing → Transformers use self-attention, which is fully parallelizable (no sequential dependency)
- More parameters always means better generalization → More parameters means more expressive but also more prone to overfitting; creates a more complex non-convex loss landscape
- The (W^T) in δ^(ℓ) is just a transpose trick → It arises from the multivariate chain rule: each a^(ℓ)_i affects ALL z^(ℓ+1)_k, so gradients sum over all those paths

---

## Quick Decision Rules

- If asked what loss to use for binary classification → Binary Cross-Entropy (from Bernoulli likelihood)
- If asked what loss to use for regression → MSE (from Gaussian likelihood)
- If asked for the sigmoid derivative → σ(x)(1−σ(x)), never write out the full e^{−x} derivation unless asked
- If asked why gradients can vanish in deep networks → sigmoid derivative σ(1−σ) ≤ 0.25; multiplied through many layers → exponential decay
- If asked which architecture for images → CNN (translation equivariance, weight sharing)
- If asked which architecture for sequences → RNN/LSTM (or Transformer for long-range)
- If asked which architecture for unsupervised representation learning → Autoencoder
- If asked to derive BCE: start from P(y|x) = ŷ^y(1−ŷ)^{1−y}, take log, negate, sum over i
- If asked why (W^T δ) appears in backprop → each activation connects to ALL neurons in next layer → multivariate chain rule sums over all paths → becomes a matrix-vector product with W^T
- If asked what makes Transformers special vs RNNs → parallelism through self-attention (no sequential hidden state)
