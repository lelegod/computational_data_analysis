# Week 3 — Group Discussion Questions

---

## Q1: Think-Pair-Share — The Curse of Dimensionality (3 mins)

**Question (slide 6):**
Context: In high-dimensional settings ($p \gg N$), our intuition about space and distance often fails.

1. **Think:** What specific problems arise for learning algorithms when the number of features $p$ becomes very large?
2. **Pair:** Discuss your identified problems with your neighbor.
3. **Share:** Enter your keywords into the Vevox word cloud.

**Answer:**

Five core problems the slides identify (slide 7):

**1. Sparsity**
Data becomes incredibly sparse. In $D$ dimensions with a fixed $N$ observations, the volume of the space grows as $\sim r^D$. A neighbourhood of radius $r$ that contained 10% of data in 1D now contains a vanishingly small fraction. Local methods (KNN, kernel regression) break down because "local" neighbourhoods must expand to encompass enough points, and at that scale they are no longer local.

**2. Distances lose meaning**
Euclidean distances concentrate. As $p \to \infty$, the ratio $(\text{max dist} - \text{min dist}) / \text{min dist} \to 0$. All points become roughly equidistant from each other and from a query point. Any algorithm that relies on distance-based similarity (KNN, RBF kernels, clustering) becomes unreliable.

**3. Overfitting ($p > N$)**
With more features than observations, OLS has infinitely many solutions (the system $X\beta = y$ is underdetermined). A model can perfectly fit the training data by memorising noise — degrees of freedom equal or exceed sample size. Regularisation (Ridge, Lasso) is required.

**4. Edge Effect**
Most data points reside near the boundaries (corners) of the sample space rather than the interior. In high dimensions, the fraction of volume near the surface of a hypercube approaches 1. Models trained in the interior will see test points near edges they have never encountered.

**5. Computational Cost**
Search algorithms (e.g., KD-trees, exhaustive nearest-neighbour search) scale exponentially with dimension. Operations that are $O(p)$ become infeasible for $p \sim 10^4$ or higher.

**Bonus — Blessings of dimensionality (slide 8, Donoho 2000):**
Not everything is bad. High dimensions also bring:
1. Correlated features — averaging over them improves signal-to-noise.
2. Data lies on a low-dimensional manifold — intrinsic dimensionality is often much lower than $p$.
3. Continuous processes (images, spectra) have approximate finite dimensionality.

---

## Q2: Fill-in — Norms of $\beta$ (slide 11)

**Question:**
Fill in the definitions:

$$\|\beta\|_2^2 = $$

$$\|\beta\|_1 = $$

**Answer:**

$$\|\beta\|_2^2 = \sum_{j=1}^{p} \beta_j^2$$

This is the squared $L_2$-norm (sum of squared coefficients). It is used as the penalty in **Ridge regression**:
$$\min_\beta \frac{1}{2n}\|Y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

The $L_2$ penalty shrinks all coefficients smoothly toward zero but never exactly to zero. It corresponds geometrically to a **hypersphere** constraint region.

$$\|\beta\|_1 = \sum_{j=1}^{p} |\beta_j|$$

This is the $L_1$-norm (sum of absolute values of coefficients). It is used as the penalty in **LASSO**:
$$\min_\beta \frac{1}{2n}\|Y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

The $L_1$ penalty induces **sparsity** — it forces some coefficients exactly to zero, performing automatic variable selection. Geometrically, it corresponds to a **diamond/hypercube** constraint region whose corners lie on the coordinate axes.

**Key difference:** Ridge shrinks but keeps all variables; Lasso shrinks and selects (zeros out irrelevant variables).

---

## Q3: FWER for the Jelly Bean Example (slide 53)

**Question:**
The xkcd jelly bean comic scenario:
- 20 experiments conducted at a 5% significance level (one per jelly bean colour).
- Assume that the effect of different colours are independent.

Calculate: $FWER = ???$

**Answer:**

Using the formula from slide 49:
$$FWER = 1 - (1 - \alpha)^M$$

where $M = 20$ is the number of independent tests and $\alpha = 0.05$.

$$FWER = 1 - (1 - 0.05)^{20} = 1 - (0.95)^{20} \approx 1 - 0.358 \approx \mathbf{0.64}$$

**Interpretation:** There is a **64% probability of at least one false rejection** across the 20 experiments, even if no jelly bean colour actually causes acne. This is why the comic concludes "Green Jelly Beans Linked to Acne! 95% Confidence!" — it was simply the one test (out of 20) that happened to hit $p < 0.05$ by chance. This is the multiple testing problem in its purest form.

**Why $\alpha = 0.05$ is no longer safe with $M$ tests:**
Each individual test has a 5% false positive rate. With 20 independent tests, the probability of getting *at least one* spurious rejection explodes to 64%. With $M = 100$ tests: $FWER = 1 - 0.95^{100} \approx 0.994$ — a virtual certainty of at least one false discovery.

**Bonferroni correction** (slide 57): To control $FWER \leq \alpha$ across $M$ tests, reject hypothesis $i$ only if $p_i \leq \alpha / M = 0.05 / 20 = 0.0025$.

---

## Q4: Think-Share-Pair — Manual FDR Calculation (2 mins)

**Question (slide 63, Exercise 3: Multiple Testing):**
You perform $m = 5$ hypothesis tests. You set your False Discovery Rate (FDR) level to $q = 0.20$.

Observed p-values (sorted):
$$0.01, \quad 0.03, \quad 0.15, \quad 0.40, \quad 0.50$$

Task (2 min): Calculate the Benjamini-Hochberg thresholds $\left(\frac{i}{m} \cdot q\right)$ for $i = 1, 2, 3, \ldots$ and decide which hypotheses to reject.

**Answer:**

**Step 1: Compute BH thresholds** $\frac{i}{m} \cdot q$ for each rank $i$:

| $i$ | Sorted $p_{(i)}$ | BH threshold $\frac{i}{5} \times 0.20$ | $p_{(i)} \leq$ threshold? |
|-----|-----------------|----------------------------------------|--------------------------|
| 1   | 0.01            | 0.04                                   | Yes ($0.01 \leq 0.04$)   |
| 2   | 0.03            | 0.08                                   | Yes ($0.03 \leq 0.08$)   |
| 3   | 0.15            | 0.12                                   | No ($0.15 > 0.12$)       |
| 4   | 0.40            | 0.16                                   | No                       |
| 5   | 0.50            | 0.20                                   | No                       |

**Step 2: Find $k$**

$k = \max\{i : p_{(i)} \leq \frac{i}{m} q\}$

The last $i$ where the condition holds is $i = 2$.

Therefore $k = 2$.

**Step 3: Reject**

Reject $H_{(1)}$ and $H_{(2)}$ — the hypotheses with $p$-values 0.01 and 0.03.

---

## Q5: Vevox Poll 3 — Rejection Decisions (slide 64)

**Question (Exercise 3: Multiple Testing):**
According to the Benjamini-Hochberg procedure (same setup as Q4 above: $m=5$, $q=0.20$, p-values $0.01, 0.03, 0.15, 0.40, 0.50$), which hypotheses are rejected?

A. Only the first ($p = 0.01$)
B. First and second ($p = 0.01, 0.03$)
C. First, second, and third ($p = 0.01, 0.03, 0.15$)
D. None

**Answer: B**

**Working shown on slide 64:**
- $i = 1$: Threshold $= 1/5 \times 0.20 = 0.04$ → $0.01 \leq 0.04$ — Pass
- $i = 2$: Threshold $= 2/5 \times 0.20 = 0.08$ → $0.03 \leq 0.08$ — Pass
- $i = 3$: Threshold $= 3/5 \times 0.20 = 0.12$ → $0.15 > 0.12$ — Fail

The largest $k$ satisfying the condition is $k = 2$. Therefore reject all null hypotheses up to rank 2: $H_{(1)}$ and $H_{(2)}$.

**Why not C?** Although $H_{(3)}$ has $p = 0.15$ which seems small, the BH threshold at rank 3 is only 0.12. The p-value exceeds its threshold, so the walk stops. The BH procedure is a sequential step-up test — once a threshold is missed, no higher-ranked hypotheses (with larger p-values) can be rejected.

**Conceptual note:** BH controls the FDR at level $q = 0.20$, meaning among our 2 rejections we expect at most $0.20 \times 2 = 0.4$ false discoveries in expectation. This is a weaker guarantee than Bonferroni's FWER control, but it gives more power (we reject 2 hypotheses instead of possibly just 1 under Bonferroni).
