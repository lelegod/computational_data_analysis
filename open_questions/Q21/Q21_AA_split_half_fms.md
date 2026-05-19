# Q21-AA — Split-Half Analysis and FMS for PARAFAC Validation
> Week 12. Validating PARAFAC solutions — reproducibility and rank selection.

---

## Why Validation Is Needed for PARAFAC

PARAFAC finds components by minimizing $\|X - \sum_r a_r\circ b_r\circ c_r\|_F^2$. The algorithm is iterative (alternating least squares, ALS) and:
- May converge to local minima
- May overfit if $R$ is too large
- The correct rank $R$ is not known in advance

Two complementary tools: **CORCONDIA** (measures how well the model structure matches) and **Split-half / FMS** (measures reproducibility across independent data splits).

---

## The Split-Half Procedure

**Idea**: if the PARAFAC components are real (physically meaningful, not noise), they should be reproducible — fitting the same model on two independent halves of the data should give the same components.

**For a 3-way tensor** $\mathcal{X} \in \mathbb{R}^{I\times J\times K}$ (e.g., samples × variables × time):

**Split along the sample mode** (mode 1):
1. Randomly split the $I$ samples into two halves: $\mathcal{X}^{(1)} \in \mathbb{R}^{I/2\times J\times K}$ and $\mathcal{X}^{(2)} \in \mathbb{R}^{I/2\times J\times K}$
2. Fit PARAFAC with rank $R$ to each half independently → get loadings $(A^{(1)},B^{(1)},C^{(1)})$ and $(A^{(2)},B^{(2)},C^{(2)})$
3. Compare the shared modes (modes 2 and 3) across halves

**Why compare shared modes**: mode 1 is the split mode (samples) — components from each half use different samples so $A^{(1)} \neq A^{(2)}$ by construction. Modes 2 and 3 (e.g., variables and time) are shared across halves — if the structure is real, $B^{(1)} \approx B^{(2)}$ and $C^{(1)} \approx C^{(2)}$.

---

## Factor Match Score (FMS)

The FMS quantifies how well components from two PARAFAC solutions match, accounting for the fact that PARAFAC is unique only up to permutation and scaling.

**For a single component pair** $(b_r^{(1)}, b_r^{(2)})$ (normalized to unit norm):
$$\text{fms}_r = |b_r^{(1)\,T} b_r^{(2)}| \in [0,1]$$

- $\text{fms}_r = 1$: perfect match (same direction)
- $\text{fms}_r = 0$: orthogonal (no match)

**For $R$ components combined** (after optimal permutation matching):
$$\text{FMS} = \frac{1}{R}\sum_{r=1}^R \text{fms}_r^{(B)} \cdot \text{fms}_r^{(C)}$$

where $\text{fms}_r^{(B)}$ and $\text{fms}_r^{(C)}$ are the match scores for modes 2 and 3 respectively.

**Threshold**: FMS $> 0.95$ (or sometimes $> 0.9$) indicates reproducible components. Below this: components are not stable across halves → $R$ may be too large (overfitting noise) or the data does not have PARAFAC structure.

---

## Interpreting Split-Half Results

| FMS | Interpretation |
|-----|---------------|
| $\approx 1.0$ | Components are highly reproducible — $R$ is appropriate |
| $0.9$–$1.0$ | Good reproducibility |
| $0.7$–$0.9$ | Marginal — consider reducing $R$ |
| $< 0.7$ | Poor reproducibility — $R$ too large or no real structure |

**Repeating the split**: because the split is random, repeat 10–50 times with different random splits. Report the median (or minimum) FMS. A single split may be fortunate or unlucky.

---

## CORCONDIA vs Split-Half: Complementary Tools

| | CORCONDIA | Split-Half FMS |
|--|-----------|---------------|
| What it measures | How close core is to super-diagonal | Reproducibility across independent splits |
| Based on | Full dataset | Two random halves |
| Detects | Model misfit (wrong structure) | Overfitting (unstable components) |
| Fast? | Yes (one fit) | No ($2\times$ num_splits fits) |
| Interpretation | $\approx100$: good, $<50$: too many components | FMS $>0.95$: reproducible |
| Use together? | Yes — both should agree on $R$ |

**Decision rule**: choose the largest $R$ where CORCONDIA $\geq 95$ AND FMS $\geq 0.95$. If they disagree, be conservative and use the smaller $R$.

---

## Practical Workflow for Choosing R

1. Fit PARAFAC for $R = 1, 2, 3, \ldots, R_\text{max}$
2. For each $R$: compute CORCONDIA
3. For each $R$: run split-half analysis (e.g., 20 random splits), compute median FMS
4. Plot both vs $R$
5. Choose the largest $R$ before either CORCONDIA or FMS drops sharply below threshold

---

## Additional Possible Exam Questions

**Q: Why do we compare modes 2 and 3 but not mode 1 in split-half analysis?**
Mode 1 (samples) is the split dimension — by construction, $A^{(1)}$ and $A^{(2)}$ are estimated from different samples, so they represent the loadings of different observations. There is no reason for them to be the same. Modes 2 and 3 (e.g., spectral or temporal profiles) describe the shared physical structure — if the PARAFAC components represent real phenomena, the spectral profiles should be identical regardless of which samples were used to estimate them.

**Q: What does it mean for FMS to be low even when CORCONDIA is high?**
CORCONDIA being high means the fitted model has a near-super-diagonal core — the model structure is consistent with PARAFAC. But low FMS means the specific components change across different data splits — the solution is not reproducible. This can happen when: (1) $R$ is at the boundary (one too many components, some are noisy), (2) there are near-degenerate solutions (multiple essentially equivalent PARAFAC decompositions), or (3) the ALS algorithm found different local optima on the two halves.

**Q: Can you apply split-half validation to Tucker3?**
Tucker3 is not unique (rotation ambiguity), so comparing loadings from two halves is ambiguous — any rotation of the components is equally valid. FMS as defined does not apply directly. Instead, you compare the Tucker3 core tensors $\mathcal{G}^{(1)}$ and $\mathcal{G}^{(2)}$ after applying a Procrustes rotation to align them. This is less standard and not commonly used; CORCONDIA-equivalent diagnostics are more common for Tucker3.

**Q: What is the ALS algorithm for PARAFAC and why might it converge slowly?**
ALS (Alternating Least Squares) fixes all modes except one and solves for that mode's loadings via ordinary least squares, then cycles through modes. Convergence is guaranteed (objective decreases) but can be extremely slow when: (1) components are nearly collinear (ill-conditioned normal equations), (2) the model has "swamping" (two components nearly identical), (3) $R$ is too large. Slow convergence and sensitivity to initialization are practical reasons to run ALS many times with different starts.
