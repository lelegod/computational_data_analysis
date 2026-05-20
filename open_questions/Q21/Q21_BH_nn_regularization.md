# Q21-BH — Neural Network Regularization
> Week 10. Could ask: compare regularization methods and explain their mechanisms, derive the weight decay gradient update, explain dropout at training vs test time, or discuss when each method is appropriate.

---

## The Model

A feedforward neural network (MLP) with $L$ layers, parameters $\{W^{(l)}, b^{(l)}\}_{l=1}^L$, trained by minimising:

$$L_\text{total}(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(y_i, \hat{y}_i) + \text{Regularisation}(\theta)$$

**Why NNs overfit**: with parameter count $p \gg N$, a deep network has sufficient capacity to memorise the training data exactly, achieving near-zero training loss while generalising poorly. Regularisation imposes inductive biases that limit effective capacity.

---

## Method 1: Weight Decay (L2 Regularisation)

**Objective**: add an $L_2$ penalty on all weights:

$$L_\text{total} = \frac{1}{N}\sum_i \ell(y_i, \hat{y}_i) + \frac{\lambda}{2}\sum_{l,j,k} (W^{(l)}_{jk})^2$$

**Gradient update** (for weight $W^{(l)}_{jk}$):

$$\frac{\partial L_\text{total}}{\partial W^{(l)}_{jk}} = \frac{\partial L}{\partial W^{(l)}_{jk}} + \lambda W^{(l)}_{jk}$$

**SGD update**: $W^{(l)} \leftarrow W^{(l)} - \eta\left(\frac{\partial L}{\partial W^{(l)}} + \lambda W^{(l)}\right) = (1-\eta\lambda)W^{(l)} - \eta\frac{\partial L}{\partial W^{(l)}}$

The factor $(1-\eta\lambda)$ shrinks weights toward zero at each step — this is the "decay." Biases are typically not regularised.

**Connection to Ridge regression**: for a single linear layer with no activation, weight decay reduces exactly to Ridge regression. For deep networks it plays an analogous role — shrinks weights, reduces effective complexity.

**Effect**: prevents any individual weight from growing very large; forces the network to distribute the representational load across many weights rather than relying on a few dominant connections.

---

## Method 2: Dropout

**Training time**: independently zero out each unit in a layer with probability $p$ (dropout rate). The surviving units are scaled by $1/(1-p)$ so the expected activation is preserved. Each forward pass trains a different "thinned" subnetwork.

**Test time**: all units are active (no dropout). Weights are scaled by $(1-p)$ (equivalently, activations are scaled by $(1-p)$). This produces the expectation over all thinned subnetworks.

**Critical exam point**: at test time weights are scaled by $(1-p)$, NOT randomly dropped. Dropping at test time would give stochastic predictions — expectations would be correct but individual outputs would be random.

**Why it works**:
1. **Ensemble interpretation**: training with dropout is approximately equivalent to training $2^d$ different subnetworks (one for each binary dropout mask over $d$ units), sharing parameters. At test time, we use the geometric mean of all subnetworks — an ensemble prediction without the cost of training many models.
2. **Prevents co-adaptation**: each unit cannot rely on the presence of specific other units. This forces each unit to learn independently useful features — prevents neurons from "memorising" training examples cooperatively.
3. **Noise injection**: dropout acts as a form of noise that regularises the learned representation.

**Typical values**: $p = 0.5$ for fully-connected layers; $p = 0.2$–$0.3$ for convolutional layers (less redundancy to exploit).

---

## Method 3: Early Stopping

**Procedure**: split data into training and validation sets. Monitor validation loss during training. Stop when validation loss starts consistently increasing (overfitting signal). Use the model weights at the stopping epoch.

**Why it regularises**: during gradient descent, the model moves from the origin (zero initialisation) toward the minimum of the training loss. Early stopping restricts the model to a ball around the origin — it cannot fully fit the training noise. This is analogous to $L_2$ regularisation: early stopping in gradient descent is approximately equivalent to weight decay with $\lambda \propto 1/t$ where $t$ is the number of steps.

**Advantages**: no change to the loss function; free once the validation set is chosen; acts on the effective function space explored by gradient descent.

**Limitation**: sensitive to validation set size and the definition of "consistently increasing" (need a patience parameter to avoid stopping on temporary fluctuations).

---

## Method 4: Batch Normalisation

**Procedure**: for each mini-batch $\mathcal{B}$, normalise the pre-activations within each layer before the activation function:

$$\hat{z}^{(l)}_j = \frac{z^{(l)}_j - \mu_{\mathcal{B},j}}{\sqrt{\sigma^2_{\mathcal{B},j} + \epsilon}}, \qquad \tilde{z}^{(l)}_j = \gamma_j \hat{z}^{(l)}_j + \beta_j$$

where $\mu_{\mathcal{B},j}$ and $\sigma^2_{\mathcal{B},j}$ are the mean and variance of unit $j$ within the batch; $\gamma_j, \beta_j$ are learned scale and shift parameters (restored expressiveness).

**At test time**: use running mean and variance (exponential moving average over training batches) instead of batch statistics.

**Why it helps**:
1. **Internal covariate shift**: as parameters in earlier layers change, the distribution of inputs to later layers shifts continuously — later layers must constantly adapt. Batch norm stabilises these distributions, allowing higher learning rates.
2. **Regularisation**: the batch-level mean and variance inject noise into each activation (different batches have different statistics) — a mild regularisation effect, though smaller than dropout.
3. **Gradient flow**: normalised activations stay in a stable range → activation functions (sigmoid, tanh) remain in their non-saturating region → reduced vanishing gradient.

---

## Method 5: Data Augmentation

**Procedure**: artificially expand the training set by applying label-preserving transformations:
- Images: horizontal/vertical flips, random crops, rotations, colour jitter, cutout.
- Audio: time stretching, pitch shifting, additive noise.
- Tabular: mixup (linear interpolation of two observations and their labels).

**Why it works**: forces the network to be invariant to transformations that should not affect the label. Effectively increases the training set size without collecting new data. Reduces overfitting indirectly by exposing the model to more variation in the input.

---

## Comparison Table

| Method | Mechanism | Computational cost | Where applied | Analogous to |
|--------|-----------|-------------------|---------------|--------------|
| Weight decay (L2) | $+\lambda W$ shrinkage in gradient | None | All weights, every step | Ridge regression |
| Dropout | Zero units with prob $p$; scale at test | Minor (masking) | Hidden units; not usually output | Ensemble averaging |
| Early stopping | Stop before convergence | None (validation monitoring) | Training loop | L2 penalty (approximately) |
| Batch norm | Normalise per mini-batch, learned rescale | Small (extra stats) | Between layers; before activation | Conditioning / whitening |
| Data augmentation | Transform training inputs | Data generation cost | Training data pipeline | Increasing $N$ |

---

## Comparison to OLS / Classical Methods

| Property | Weight Decay | Ridge Regression |
|----------|-------------|-----------------|
| Loss | Non-convex (deep network) | Convex (linear) |
| Closed form? | No | Yes |
| Effect of $\lambda$ | Shrinks all weights | Shrinks all coefficients |
| Bias increase? | Yes | Yes |
| Variance reduction? | Yes | Yes |

Weight decay is the neural network analogue of Ridge regression, generalised to non-linear, non-convex models trained by gradient descent rather than analytically.

---

## Limitations

- **Weight decay**: treats all weights equally; may not be appropriate for layers with very different scales. Does not promote sparsity (unlike L1).
- **Dropout**: increases training time (need more epochs to converge, since effective training sample is smaller each step). Can hurt performance if $p$ is too large. Less effective for convolutional layers than fully-connected.
- **Early stopping**: requires a held-out validation set, reducing training data. Sensitive to the patience parameter. Can stop too early on noisy validation loss curves.
- **Batch normalisation**: introduces a dependency between examples in the same batch; performance can degrade with very small batches. Behaviour differs between training and test time (running vs batch statistics).

---

## Additional Possible Exam Questions

**Q: At test time, dropout uses weights scaled by $(1-p)$, not random dropping. Why?**
If we randomly dropped units at test time, each prediction would be stochastic — different runs would give different outputs. The correct approach is to scale weights by $(1-p)$, which gives a deterministic prediction equal to the expected output over all possible dropout masks during training. This approximation works because the expected activation of a unit trained with dropout rate $p$ is $(1-p)$ times the activation of the fully-connected unit.

**Q: Show that weight decay adds a term $\lambda W^{(l)}$ to the gradient.**
$L_\text{total} = L + \frac{\lambda}{2}\|W^{(l)}\|_F^2$. $\frac{\partial}{\partial W^{(l)}}\frac{\lambda}{2}\|W^{(l)}\|_F^2 = \lambda W^{(l)}$. So $\frac{\partial L_\text{total}}{\partial W^{(l)}} = \frac{\partial L}{\partial W^{(l)}} + \lambda W^{(l)}$. The SGD update is $W^{(l)} \leftarrow W^{(l)} - \eta(\frac{\partial L}{\partial W^{(l)}} + \lambda W^{(l)}) = (1-\eta\lambda)W^{(l)} - \eta\frac{\partial L}{\partial W^{(l)}}$.

**Q: How does early stopping implicitly regularise the network?**
Gradient descent from a small random initialisation moves the parameters progressively further from the origin (zero weights) toward the training-loss minimum. Parameters that are far from zero are needed to fit fine-grained training patterns — including noise. Stopping early restricts the parameters to a ball of radius proportional to $\sqrt{t}$ (number of steps), which is equivalent to bounding the model complexity. This is analogous to adding an $L_2$ penalty, though the relationship is exact only for quadratic loss and linear models.

**Q: Why does batch normalisation sometimes act as a regulariser?**
During training, batch normalisation computes statistics ($\mu_\mathcal{B}$, $\sigma_\mathcal{B}$) from the current mini-batch. Because different mini-batches have slightly different statistics, the normalised activations carry mini-batch-level noise. This noise prevents the network from memorising exact activation patterns of individual training examples — a mild regularisation effect similar to dropout noise injection, though generally weaker.

**Q: When would you prefer dropout over weight decay?**
Dropout is preferred when: (1) the network is very large with dense fully-connected layers, where co-adaptation is a major overfitting mechanism; (2) data is scarce relative to model capacity; (3) the task benefits from ensemble-like robustness. Weight decay is preferred when: (1) the network is relatively shallow; (2) an analogy to $L_2$ regularisation is desired; (3) interpretability of magnitude control is important. In practice both are often combined.
