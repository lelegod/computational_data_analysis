# Q22 — Face Image Clustering (2022 Exam)
> **10 points** | Appeared as Q22 in 2022 — the only unsupervised Q22 so far
> This variant is the odd one out: no CV design needed, the task is purely unsupervised discovery.

---

## The Exact Question (2022)

> "You are given a dataset consisting of images of faces that have been taken in the passport control in Copenhagen Airport. Security would like to know how many unique people have entered Denmark and compare it to the unique passport numbers that entered in the same time-period, as they fear a systematic fraud is happening. Which methodology, and why, would you use to analyse the data and find the number of unique people from the available face images to help security?"

---

## Why This Is Unsupervised

This problem cannot be solved with supervised learning because:
- **No labels exist** — we do not know which images belong to the same person
- **Classes are unknown** — the whole point is to discover them
- **No training targets** — there is no $y$ to learn from

The task is *discovery* of natural groupings in unlabelled data — the textbook definition of unsupervised learning. Specifically it is a **clustering + model selection** problem where the number of clusters $K$ equals the number of unique identities.

---

## Full Model Answer (Exam-Ready)

### Step 1: Feature Extraction

Raw pixel images are high-dimensional ($p \gg n$ for small airports), and raw pixels capture irrelevant variation (lighting, background, expression). We need a compact, discriminative representation.

**PCA / Eigenfaces (course-standard approach):**

1. Flatten each image to a vector $x_i \in \mathbb{R}^p$ (e.g., $100 \times 100 = 10{,}000$ pixels)
2. Centre the data: $\tilde{X} = X - \bar{x}\mathbf{1}^T$
3. Apply SVD: $\tilde{X} = U S V^T$
4. Project onto first $k$ principal components (eigenfaces): $z_i = V_k^T x_i \in \mathbb{R}^k$

The eigenfaces $V_k$ capture the dominant modes of facial variation across all images. Choosing $k \ll p$ eliminates noise and irrelevant dimensions while preserving identity-discriminative structure.

**Why PCA is justified here:**
- Face images lie on a low-dimensional manifold (blessing of dimensionality — manifold hypothesis)
- Most variance is in identity-relevant structure (face shape, proportions), not noise
- Reduces computational cost of subsequent clustering

---

### Step 2: Clustering in Feature Space

After projection, we have $N$ feature vectors $z_1, \ldots, z_N \in \mathbb{R}^k$. We want to group them by identity.

#### Primary: Gaussian Mixture Models (GMM)

$$p(z) = \sum_{j=1}^{K} \pi_j \, \mathcal{N}(z \mid \mu_j, \Sigma_j)$$

- Each component $j$ represents one unique individual
- Fit via EM algorithm (E-step: compute soft assignments $\gamma_{ij}$; M-step: update $\mu_j, \Sigma_j, \pi_j$)
- **Select $K$** using BIC: $\text{BIC}(K) = -2\log\hat{L} + d_K \log N$, where $d_K$ is the number of free parameters. Choose the $K$ that minimises BIC.

**Why GMM over K-means:**
- Images of the same person form elliptical clusters (varying pose/lighting) — GMM with full $\Sigma_j$ handles this; K-means assumes spherical clusters
- Soft assignment handles ambiguous images naturally
- BIC provides a principled criterion for selecting $K$

#### Alternative: K-means

$$\hat{K} = \arg\min_K \left\{ \text{WSS}(K) \right\} \quad \text{via elbow or silhouette}$$

Silhouette score for choosing $K$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean distance to nearest other cluster. Maximise average $s(i)$ over $K$.

#### Alternative: Hierarchical Clustering

Build a dendrogram using Ward linkage. Cut at the height with the largest gap in merge distances → gives $K$ natural clusters. Useful for visualising the identity structure.

---

### Step 3: Model Selection — Choosing $K$

| Method | Criterion | How to use |
|--------|-----------|------------|
| GMM + BIC | $-2\log\hat{L} + d_K\log N$ | Minimise over $K$ |
| K-means + silhouette | Mean $s(i)$ | Maximise over $K$ |
| K-means + elbow | WSS vs. $K$ | Find kink in curve |
| Hierarchical | Dendrogram gap | Largest merge-distance gap |
| Gap statistic | $\text{Gap}(K) = E^*[\log W_K] - \log W_K$ | Maximise over $K$ |

The Gap statistic compares within-cluster dispersion to a reference distribution (uniform random data) — it is the most principled approach but computationally expensive.

---

### Step 4: Fraud Detection Logic

Once $\hat{K}_{\text{faces}}$ is estimated (number of unique faces) and $K_{\text{passports}}$ is known (number of unique passport numbers):

| Result | Interpretation |
|--------|---------------|
| $\hat{K}_{\text{faces}} \approx K_{\text{passports}}$ | No evidence of fraud — people match passports |
| $\hat{K}_{\text{faces}} < K_{\text{passports}}$ | Fewer unique people than passports → same person used multiple passports → **fraud signal** |
| $\hat{K}_{\text{faces}} > K_{\text{passports}}$ | More unique people than passports → possible passport sharing or detection errors |

The magnitude of the discrepancy $K_{\text{passports}} - \hat{K}_{\text{faces}}$ estimates the number of fraudulent passport entries.

---

### Recommended Pipeline and Justification (Write This Cold)

*"This is an unsupervised learning problem: we have no identity labels and must discover groupings from unlabelled image data. I would apply the following pipeline:*

*First, dimensionality reduction via PCA (eigenfaces): flatten each image to a pixel vector, centre the data, and project onto the first $k$ principal components. This removes noise and exploits the fact that face images lie on a low-dimensional manifold, making subsequent clustering tractable.*

*Second, cluster the projected feature vectors using a Gaussian Mixture Model (GMM). Each Gaussian component represents one unique individual. GMM is preferred over K-means because images of the same person form elliptical clusters (due to pose and lighting variation) — GMM with full covariance matrices handles this geometry, while K-means assumes spherical clusters. EM is used to fit the model.*

*Third, select the number of components $K$ (= number of unique people) by minimising BIC over a range of $K$ values. BIC balances model fit against complexity and provides a principled, quantitative criterion for choosing $K$.*

*Finally, compare $\hat{K}_{\text{faces}}$ to the number of unique passport numbers. If fewer unique faces are found than passports, this suggests multiple passports are associated with one physical person — strong evidence of systematic fraud."*

---

## Limitations to Acknowledge

1. **Image quality:** Blurry, occluded, or poorly lit images may not cluster correctly — creates noise in the identity assignment
2. **Non-Gaussian clusters:** Extreme pose variation or mixed lighting can produce non-Gaussian distributions in feature space, weakening GMM assumptions
3. **Underrepresented individuals:** A person who passed through only once has a single image — cannot form a reliable cluster and may be merged with another person or treated as noise
4. **PCA loses discriminative structure:** PCA maximises variance, not inter-class separability. The first few eigenfaces may capture lighting/expression variation rather than identity variation. Using LDA (Fisherfaces) in a semi-supervised setting (if some labels exist) would be better
5. **Uncertainty in $K$:** All $K$-selection methods give heuristic estimates. BIC/silhouette may disagree. Report a range or confidence interval rather than a single number

---

## Extended Question Bank

**Q: Why not use supervised face recognition here?**
Supervised face recognition (e.g., train a classifier with known identities) requires labelled training data — images paired with known person IDs. We have none. The identities are exactly what we want to discover, so supervision is circular. Only unsupervised methods can solve this.

---

**Q: Why is PCA appropriate here? What assumption does it rely on?**
PCA assumes the data lies near a low-dimensional linear subspace (approximately a linear manifold). For face images, this is reasonable: the dominant variation across faces is smooth (shape, proportions, lighting direction) and can be captured by a small number of linear components. The manifold hypothesis (Donoho's blessing of dimensionality) supports this — face images do not fill their high-dimensional pixel space uniformly.

---

**Q: BIC formula — what does each term penalise?**
$$\text{BIC}(K) = -2\log\hat{L} + d_K \log N$$

- $-2\log\hat{L}$: reward for fit — larger $K$ always improves fit (lower value = better)
- $d_K \log N$: penalty for complexity — $d_K$ grows with $K$ (each extra Gaussian adds $k + k(k+1)/2 + 1$ parameters for mean, covariance, mixing weight)
- The optimal $K$ trades off fit against complexity → prevents overfitting (too many small clusters)

---

**Q: What does CORCONDIA have to do with this question?**
Nothing directly — CORCONDIA is a model selection criterion for PARAFAC/Tucker tensor decompositions, not for GMM or K-means. However, the conceptual parallel is identical: both select the number of components by measuring whether adding more components improves the model meaningfully. CORCONDIA drops below ~90 when $R$ is too large (spurious components); BIC increases when $K$ is too large (over-parameterised). The underlying logic — "find the simplest model that fits the data well" — is the same.

---

**Q: Could you use hierarchical clustering instead of GMM? When would it be preferable?**
Yes. Hierarchical clustering with Ward linkage would:
1. Compute pairwise distances between feature vectors
2. Merge the pair with minimum linkage distance at each step
3. Produce a dendrogram — a tree of all merges

Preferable when: you want to visually inspect the cluster structure, when $K$ is truly unknown and you want to explore multiple levels, or when clusters are not well-described by Gaussian distributions.

Drawback: computationally $O(N^2)$ in memory for large $N$ — infeasible for thousands of images. GMM with EM scales better. For airport security with potentially millions of images, GMM or K-means would be preferred.

---

**Q: The examiner asks: "Why not simply use K-means with $K$ chosen by the elbow method?"**
K-means is a valid approach, but has two limitations relative to GMM for this task:

1. **Assumes spherical clusters:** K-means minimises $\sum_k \sum_{i \in C_k} \|z_i - \mu_k\|^2$, which penalises all directions equally. Face clusters in PCA space are typically elliptical (lighting varies one way, expression another) — K-means will split or merge clusters incorrectly when they are non-spherical.

2. **Hard assignment:** K-means forces each image to exactly one cluster. An ambiguous image (poor lighting, partial occlusion) gets hard-assigned to one identity, potentially creating errors. GMM's soft assignment reflects genuine uncertainty.

The elbow method for $K$ is also less principled than BIC — the "elbow" is often ambiguous in practice.

---

**Q: What if we had some labelled images (e.g., a few known individuals) — how would you use them?**
This becomes a **semi-supervised learning** problem. Options:

1. **Constrained clustering:** Force images from known individuals to be in the same cluster (must-link constraints in constrained K-means). This guides the algorithm without full supervision.

2. **LDA for feature extraction:** Use the labelled images to compute Fisher's Linear Discriminant directions (Fisherfaces) — these maximise between-class separation relative to within-class scatter, which is more discriminative than PCA. Then cluster the unlabelled images in this discriminative space.

3. **Transfer learning:** Use a pre-trained face recognition network (FaceNet) to extract embeddings, even if the network was trained on different identities. The embedding space is already structured for identity discrimination.

---

**Q: How do you handle the fact that the same person may appear hundreds of times vs. another person appearing only once?**
Highly imbalanced cluster sizes are expected (frequent travellers vs. one-time visitors). This causes problems:

- K-means and GMM with equal mixing weights assume roughly balanced clusters — rare individuals may be absorbed into nearby clusters
- Solution: use GMM with **unconstrained mixing weights** $\pi_k$ — the EM algorithm learns different component sizes naturally
- Hierarchical clustering handles imbalanced sizes well — small clusters remain distinct until explicitly merged

Report the estimated cluster sizes as part of the output: the largest clusters are frequent travellers; singleton clusters are one-time visitors (hardest to validate).
