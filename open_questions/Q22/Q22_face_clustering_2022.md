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

## Feature Extraction — Method Comparison (Key Exam Upgrade)

Slide 9 of Lecture 11 (Lee & Seung, Nature 1999) shows a direct three-way comparison of NMF vs VQ vs PCA on face images. This is the central demonstration of **why NMF matters for faces** and is exam-relevant.

| Method | What the basis vectors look like | Why |
|--------|----------------------------------|-----|
| **PCA** | Holistic "ghostly" eigenfaces — blended whole faces, contain negative values | Maximises variance with no sign constraint |
| **VQ** (Vector Quantization) | Whole prototype faces — one per cluster | Hard-assignment centroids |
| **NMF** | Localised facial parts — eyes, nose patches, mouth region, cheekbones | Non-negativity forces additive, parts-based decomposition |

**NMF wins for faces** because it learns the actual visual parts of a face, not a holistic blend. Each face image = a weighted sum of parts. This is both more interpretable and more consistent with how faces actually vary across identities (same nose structure, different eye shape, etc.).

---

## Full Model Answer (Exam-Ready)

### Step 1: Feature Extraction

Raw pixel images are high-dimensional ($p \gg n$ for small airports), and raw pixels capture irrelevant variation (lighting, background, expression). We need a compact representation. **Two valid approaches from the course — NMF is the better-justified one for faces.**

---

**Recommended: NMF (Non-negative Matrix Factorization)**

Represent the face image matrix $X \in \mathbb{R}_+^{p \times N}$ (pixels × images, all values ≥ 0) as:
$$X \approx WH, \quad W \geq 0,\ H \geq 0$$

- $W \in \mathbb{R}_+^{p \times K}$: **basis matrix** — columns are $K$ facial part patches (eye region, nose, mouth, etc.)
- $H \in \mathbb{R}_+^{K \times N}$: **coefficient matrix** — column $j$ gives how much of each part is present in image $j$

Fit by minimising:
$$\min_{W,H \geq 0} \frac{1}{2}\|X - WH\|_F^2$$

using multiplicative updates (Lee & Seung) or alternating least squares.

**Why NMF over PCA for faces (from Lecture 11, slide 8–9):**
1. **Physical realism:** pixel intensities are non-negative — NMF's $W, H \geq 0$ constraint is physically meaningful; PCA allows negative basis vectors (negative pixel intensities are meaningless)
2. **Parts-based representation:** non-negativity forces additive combinations only (no cancellation) → basis vectors become localised facial parts (a nose, an eye region) rather than holistic ghostly blends
3. **Interpretability:** "a nose on a face" vs. "a ghostly average face" — parts-based features are much more useful for identity clustering
4. **Lee & Seung (1999) demonstrated this directly** on a face image dataset — this is the seminal NMF paper shown in the lecture

**How to use NMF for clustering:** use the coefficient vectors $h_j \in \mathbb{R}^K$ (column $j$ of $H$) as the feature representation of image $j$. These $K$-dimensional vectors capture which parts are active in each image → cluster these vectors.

---

**Alternative: PCA / Eigenfaces**

1. Flatten each image to a vector $x_i \in \mathbb{R}^p$
2. Centre: $\tilde{X} = X - \bar{x}\mathbf{1}^T$
3. SVD: $\tilde{X} = USV^T$
4. Project: $z_i = V_k^T x_i \in \mathbb{R}^k$

PCA is valid and computationally simpler, but the basis vectors (eigenfaces) are holistic blends with negative components — less interpretable. Use PCA when computational simplicity matters or as a preprocessing step before NMF to initialise $W$.

**When the exam says "better method" or "what does Lecture 11 suggest?" — answer NMF, not PCA.**

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

*First, feature extraction using Non-negative Matrix Factorization (NMF). Pixel intensities are non-negative, so the NMF constraint $W, H \geq 0$ is physically meaningful. More importantly, NMF forces parts-based representations: because no cancellation is allowed, the basis vectors $W$ learn localised facial features — eyes, nose patches, mouth regions — rather than the holistic 'ghostly' eigenfaces produced by PCA. This was demonstrated by Lee & Seung (1999) directly on face images. The coefficient vector $h_j \in \mathbb{R}^K$ for each image becomes its feature representation: how strongly each facial part is present.*

*Second, cluster the coefficient vectors using a Gaussian Mixture Model (GMM). Each Gaussian component represents one unique individual. GMM is preferred over K-means because images of the same person form elliptical clusters (varying pose, lighting, expression) — full covariance GMM handles this; K-means assumes spherical clusters. Fit with EM.*

*Third, select $K$ (the number of unique people) by minimising BIC: $\text{BIC}(K) = -2\log\hat{L} + d_K\log N$. BIC penalises extra components, preventing overfitting to noise as spurious extra identities.*

*Finally, compare $\hat{K}_{\text{faces}}$ to the number of unique passport numbers. If fewer unique faces are found than passports, multiple passports are associated with the same physical person — strong evidence of systematic fraud."*

---

**PCA-based answer (acceptable but weaker):** Replace NMF with PCA/eigenfaces in the above. Acknowledge the limitation: PCA eigenfaces are holistic blends with negative values, which are less interpretable and less physically grounded for pixel data.

---

## Limitations to Acknowledge

1. **Image quality:** Blurry, occluded, or poorly lit images may not cluster correctly — creates noise in the identity assignment
2. **Non-Gaussian clusters:** Extreme pose variation or mixed lighting can produce non-Gaussian distributions in feature space, weakening GMM assumptions
3. **Underrepresented individuals:** A person who passed through only once has a single image — cannot form a reliable cluster and may be merged with another person or treated as noise
4. **NMF non-uniqueness:** NMF solutions are not unique — for any invertible $Q$ with $WQ^{-1} \geq 0$ and $QH \geq 0$, the product $WH$ is unchanged. Different random initialisations may converge to different local minima. Run multiple restarts and pick the solution with lowest reconstruction error.
5. **PCA eigenfaces lack discriminative structure:** PCA maximises variance, not inter-class separability. The first eigenfaces may capture lighting/expression variation rather than identity. NMF parts-based features are more robust to this, but still not fully discriminative.
6. **Uncertainty in $K$:** All $K$-selection methods give heuristic estimates. BIC/silhouette may disagree. Report a range or confidence interval rather than a single number.

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

---

## NMF vs PCA — Exam-Focused Comparison (Lecture 11 Core)

**Q: The exam asks "which feature extraction method is better for face images and why?" — what do you write?**

NMF is better for face images. Three reasons from Lecture 11:

1. **Physical realism:** Pixel intensities are non-negative ($x_{ij} \geq 0$). NMF's constraint $W, H \geq 0$ respects this. PCA eigenfaces contain negative values — a "negative nose" has no physical meaning.

2. **Parts-based representation:** NMF learns localised facial components (eye patches, nose region, mouth area) because no cancellation is allowed. Each face image is an additive sum of parts. PCA produces holistic "ghostly" eigenfaces — blends of the entire face that are hard to interpret. Lee & Seung (1999) demonstrated this directly on face images (the seminal NMF paper from Lecture 11 slide 9).

3. **More robust to identity-irrelevant variation:** Lighting and expression change the *intensity* of parts (coefficient in $H$) without completely reshaping the basis vectors $W$ — the same eye patch is active at different intensities under different lighting. PCA's holistic eigenfaces mix lighting, expression, and identity into the same components.

**When would you still use PCA?** When computational simplicity is required, or as an initialisation step before NMF. PCA has a closed-form solution (SVD); NMF requires iterative optimisation. For a quick first pass, PCA + GMM is standard. For the most principled answer on the exam: NMF.

---

**Q: How does NMF's non-uniqueness problem affect the clustering result?**

NMF is not unique: for any $Q$ such that $WQ^{-1} \geq 0$ and $QH \geq 0$, we get $(WQ^{-1})(QH) = WH$ — an equally valid decomposition. In practice this means:
- Different random initialisations may give different $W$ and $H$ with equal reconstruction error
- The specific part patches learned may differ across runs

Mitigation for the clustering application:
- Run NMF with $r$ random restarts → pick solution with minimum $\|X - WH\|_F^2$
- The coefficient matrix $H$ (used for clustering) is less sensitive to this ambiguity than $W$ in practice — the relative activation pattern across parts is stable even if the absolute parts differ slightly
- Cross-validate the number of components $K$ using speckled CV (matrix masking) — the course method for NMF model selection

---

**Q: What is speckled CV and when is it needed here?**

Standard row-holdout CV fails for NMF: if you hold out an entire image (row), you cannot estimate the coefficients $h_j$ for that image during training — so you cannot evaluate it.

**Speckled CV (Matrix Masking):**
1. Randomly mask a fraction of pixel entries across all images (mark as missing)
2. Fit NMF for $K$ components using only the observed entries in the loss
3. Predict the masked entries using $\hat{X} = WH$
4. Evaluate MSE on masked entries only
5. Choose $K^*$ that minimises masked-entry MSE

This is the principled way to select $K$ (number of parts) in NMF. It avoids the row-holdout problem because partial information from every image remains in the training data to estimate $H$.

---

**Q: NMF vs AA for faces — when would Archetypal Analysis be the better choice?**

| | NMF | AA |
|---|-----|-----|
| Prototypes are | Learned basis parts ($W$ free) | Convex combinations of actual data images |
| Represents | Additive parts of faces | Extreme prototype faces at the boundary |
| Good for | "What parts make up a face?" | "What are the most extreme face types?" |
| For clustering | Use $H$ (activation coefficients) | Use $H$ (mixture weights toward archetypes) |

AA would be better if the question is "find the most extreme types of faces in the dataset" — e.g., the "youngest-looking", "oldest-looking", "most distinctive features". Each real face = mixture of these boundary archetypes. For identity clustering (who is this person?), NMF coefficient vectors are more useful because they describe the intensity of specific facial parts, not how extreme the face is.
