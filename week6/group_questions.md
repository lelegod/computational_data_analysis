# Week 6 — Group Discussion Questions

Topics: Random Forests, AdaBoost, Gradient Boosting, Ensemble Methods

---

## Q1: Why Pick a Random Subset of Variables at Each Split?

**Question (Key ingredient bee slide, slide 11):** Why does Random Forest pick a random subset of $m < p$ variables at each split? Use the bagging variance formula as a hint.

**Answer:**

**The bagging variance formula is the key:**

For $B$ identically distributed trees, each with variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the bagged average is:

$$\text{Var}(\hat{y}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

As $B \to \infty$, the second term vanishes, leaving:

$$\text{Var}(\hat{y}) \xrightarrow{B \to \infty} \rho \sigma^2$$

**The bottleneck is $\rho$, the correlation between trees.**

**Why are bagged trees correlated?**

In ordinary bagging, at every split each tree considers all $p$ predictors. If there is one very strong predictor (e.g., a dominant feature in the data), **every tree will use it at the root split**. The trees end up with similar structures — they are highly correlated, and the variance reduction from averaging is limited.

**How Random Forest fixes this:**

At each split, only $m$ randomly chosen predictors are considered (typically $m = \lfloor\sqrt{p}\rfloor$ for classification, $m = \lfloor p/3 \rfloor$ for regression). This means:

- The dominant variable is excluded from $\approx (p-m)/p$ fraction of splits
- Different trees are forced to use different variables at key splits
- Trees become structurally diverse — **lower pairwise correlation $\rho$**

**Result:** $\rho \sigma^2$ is reduced, which reduces the limiting variance of the ensemble. Each individual tree has slightly higher variance than a bagged tree (because it uses fewer variables), but the ensemble variance is lower due to decorrelation.

**The bias** of Random Forest is the same as that of a single deep tree (and of bagging) — the random subsampling does not change the expected prediction, only the variance.

**In summary:** Random variable subsampling is a deliberate mechanism to **decorrelate the trees** by preventing any single strong predictor from dominating every tree's structure, thereby pushing $\rho$ down and reducing ensemble variance beyond what plain bagging achieves.

---

## Q2: When Does Random Forest Struggle with $p > n$?

**Question (slide 19):** Random forests work when $p > n$ (more variables than observations) — but the slide says "sometimes has problems." When?

**Answer:**

Random Forest generally handles $p > n$ well because:
- Each tree only sees $m \ll p$ variables per split
- Bootstrapping provides diversity
- The ensemble averages out noise

However, RF **struggles** in the following $p > n$ scenarios:

**1. Many irrelevant noise variables (pure noise predictors):**

When $p \gg n$ and most variables are uninformative, the random subset of $m$ variables at each split may contain no relevant predictors at all with high probability. The tree is forced to split on noise, producing useless splits. The signal-to-noise ratio at each split becomes very low.

For example, if only 5 out of $p = 10{,}000$ variables are relevant and $m = \lfloor\sqrt{10000}\rfloor = 100$, each split considers 100 random variables — a good chance of including a relevant one. But if $m$ is set too small relative to the number of signal variables, splits will consistently miss the signal.

**2. Correlated blocks of variables:**

When many variables are highly correlated (e.g., gene expression data where whole pathways move together), RF can implicitly "waste" its random draws on redundant variables. This inflates $\rho$ and reduces the benefit of decorrelation.

**3. Small $n$, large $p$ with few samples per class:**

With very few observations, each bootstrap sample is very similar to the others (only $\approx 63\%$ unique observations), reducing diversity. The OOB samples are very small, making OOB error estimates unreliable.

**4. When $m$ is not tuned:**

The default $m = \lfloor\sqrt{p}\rfloor$ may not be appropriate for all $p > n$ problems. An overly small $m$ relative to the number of true signal variables can degrade performance. Proper tuning of $m$ using OOB error is essential.

**Practical guidance:** In high-dimensional settings, consider pre-filtering variables, tuning $m$ carefully, or using sparse methods (Lasso, Sparse PCA) for variable selection before applying RF.

---

## Q3: Proximity Plots — Why Is the Middle Observation Hard to Classify?

**Question (slide 28):** In the proximity plot (MDS of the RF proximity matrix), identify one of the observations in the middle of the plot. Look at the actual digit it represents. Why is this observation harder to classify?

**Answer:**

**What the proximity plot shows:**

The proximity matrix $P_{ij}$ counts how often observations $i$ and $j$ end up in the same terminal node across all trees. It is a data-driven similarity measure:

$$P_{ij} = \frac{\text{number of trees where } i \text{ and } j \text{ land in same leaf}}{B}$$

MDS (multidimensional scaling) embeds this in 2D, preserving pairwise distances. Points far from the center = easily classified (consistently land with their own class). Points near the center = hard to classify (land with observations from multiple classes).

**Why central observations are hard to classify:**

Observations in the middle of the proximity plot are similar (via the RF's learned representation) to observations from **multiple different classes**. The RF is uncertain about them — across different trees, they end up in terminal nodes alongside observations from different digit classes.

**For the digit classification example (zip code / handwritten digits):**

The digits that appear in the centre of proximity plots are typically those that are visually ambiguous or have high within-class variation:

- **"1" vs "7":** A handwritten 7 with a crossbar can look like a 1; a 1 with a serif can look like a 7
- **"3" vs "8":** Incomplete loops on an 8 can look like a 3
- **"4" vs "9":** A closed-top 4 can look like a 9
- **"2" vs "7":** Script-style writing creates confusion
- **"0" vs "6":** Large, open 6s can resemble 0s

An observation in the centre is one that the RF's tree structure cannot consistently separate from other classes — it is a **boundary case** in the learned feature space. The digit itself likely has unusual writing style, stroke thickness, or proportion that makes it ambiguous under the axis-aligned splits that trees use.

**Key insight:** The proximity plot reveals the **geometry of the classification problem as seen by the RF**, not by any single linear projection. Central points are genuinely hard cases, not just noisy labels — they represent real ambiguity in the feature space that the ensemble cannot resolve.

**Connection to OOB error:** Observations that sit near the boundary in the proximity plot tend to have higher individual OOB misclassification rates — they are the "swing votes" of the ensemble.
