# CDA 02582 — Q22 MASTER CHEAT SHEET
> **20 points total.** Q22 has appeared in every exam. Know all three variants cold.
> Detailed files: `open_questions/Q22_cv_wearables.md` · `Q22_face_clustering_2022.md` · `Q22_other_datasets.md`

---

## Step 0 — Recognise Which Variant You Have (30 seconds)

| Signal in the question | Variant | Core answer |
|------------------------|---------|-------------|
| "predict for **new** patient / individual" | Supervised, generalized CV | LOIO-CV |
| "predict for **same** individual" | Supervised, personalized CV | LOSO-CV |
| "how many **unique** X" | Unsupervised, model selection | PCA + GMM + BIC |
| "multiple measurements per person" | IID violation → grouped CV | Group K-fold by person |
| "multiple sites / hospitals / batches" | Site-level grouping | LOSO by site |
| "time series, predict next week" | Temporal leakage risk | Forward-chaining / temporal holdout |
| "tensor / multi-way data, how many components" | PARAFAC model selection | CORCONDIA + split-half FMS |

---

## Variant 1 — Wearables CV (2024 & 2025)

### Dataset
**16 subjects × 3 activities × 4 seasons = 192 observations** (12 obs/subject)
Features: BVP, skin temperature, HR. Target: stress/activity level.

### Why standard CV fails
Observations from the same person share physiology (resting HR, signal amplitudes). Random splits let the model "see" a test subject's data during training → learns their personal baseline → **data leakage** → EPE is optimistically biased. The IID assumption fails: observations within an individual are correlated, not independent.

---

### a) Personalized Model — predict new season for KNOWN individual

**CV: Leave-One-Season-Out** (within one subject, 4 folds)

```
Fold 1: Train {Spring, Summer, Autumn} (9 obs) → Test Winter  (3 obs)
Fold 2: Train {Winter, Summer, Autumn} (9 obs) → Test Spring  (3 obs)
Fold 3: Train {Winter, Spring, Autumn} (9 obs) → Test Summer  (3 obs)
Fold 4: Train {Winter, Spring, Summer} (9 obs) → Test Autumn  (3 obs)
```

$$\text{EPE}_\text{pers} = E_{x,y \mid i_\text{fixed}}\left[\mathcal{L}(y, \hat{f}_i(x))\right]$$

Intra-individual variation only. EPE is lower — model knows this person. **Limitation:** only 9 training obs/fold → high-variance estimate.

---

### b) Generalized Model — predict for NEW individual

**CV: Leave-One-Individual-Out** (LOIO-CV, 16 folds)

```
Fold i:  Train on subjects {1..16} \ {i}  (180 obs)
         Test  on subject i               (12 obs)
EPE = mean(error_1, ..., error_16)
```

$$\text{EPE}_\text{gen} = E_{i_\text{new}}\left[E_{x,y \mid i_\text{new}}\left[\mathcal{L}(y, \hat{f}(x))\right]\right]$$

Inter-individual generalization. EPE is higher — model must handle between-individual variance (unknown physiology). **Critical rule:** all 12 obs from one subject stay in the same fold — never split a person across train/test.

---

### c) Comparison Table

| Property | Personalized (LOSO) | Generalized (LOIO) |
|----------|--------------------|--------------------|
| Folds | 4 | 16 |
| Training size | 9 obs | 180 obs |
| Test size | 3 obs | 12 obs |
| Captures | Intra-individual variation | Inter-individual variation |
| Typical EPE | Lower | Higher |
| Clinical use | Monitor known patient | Screen new patient |

**$\text{EPE}_\text{gen} > \text{EPE}_\text{pers}$ always** — generalized integrates over between-individual variance; personalized does not.

**Clinical recommendation:** Generalized for deployment. New patients arrive with no prior data — personalized model cannot be trained. Combine both: deploy generalized first, fine-tune as patient data accumulates (transfer learning).

---

### d) Nested CV — Hyperparameter Tuning

```
Outer loop  (16-fold LOIO)  → estimates EPE
  Outer fold i: hold out subject i
  │
  └── Inner loop  (15-fold LOIO on training subjects)  → selects λ*
        For each candidate λ: train on 14 subjects, eval on 1
        Pick λ* = argmin inner CV error
  │
  Refit on all 15 training subjects with λ*
  Eval on held-out subject i → record error_i
```

**Wrong:** tune λ on full 192 obs first, then do outer LOIO. The chosen λ has "seen" all 16 subjects → outer CV error is optimistically biased (subtle leakage via hyperparameter).

**What nested CV actually estimates (Week 2):** The outer loop audits the **entire methodology** — selection + training pipeline. The λ* from the inner loop will likely differ across outer folds; this is expected, not a problem. A large gap between inner CV error and outer CV error indicates selection-induced overfitting.

**Performance metric:** Specify $\mathcal{L}$ matching the task:
- Stress level continuous → RMSE / MAE
- Stress level categorical → balanced accuracy or AUC-ROC (not raw accuracy — likely class-imbalanced)

---

### e) Full Written Answer (Write This Cold)

*"For a personalized model, we restrict training and evaluation to a single individual's 12 observations. Leave-one-season-out CV trains on 9 observations (3 seasons) and tests on the held-out 4th season, repeating for all 4 seasons. This estimates how well the model predicts future sessions for a known individual.*

*For a generalized model, we apply leave-one-individual-out CV across all 16 subjects. In each fold, one complete subject (12 observations) is held out while the model trains on the remaining 15 subjects (180 observations). This ensures the test individual is entirely unseen, simulating deployment on a new patient.*

*The critical distinction is the source of variation: personalized CV measures intra-individual prediction error; generalized CV measures inter-individual generalization. Standard random splitting would constitute data leakage — observations from the same person share physiological structure, violating the IID assumption. For clinical deployment on new patients, the generalized CV estimate is the appropriate performance metric."*

---

## Variant 2 — Face Image Clustering (2022)

### The Question (exact)
*"Given face images from passport control at Copenhagen Airport, determine the number of unique people. Compare to unique passport numbers to detect fraud."*

### Why Unsupervised
No labels exist — identities are exactly what we are trying to discover. Supervised classification requires known classes. This is unsupervised discovery: clustering with unknown $K$.

---

### Pipeline

**Step 1 — Feature Extraction (NMF preferred; PCA is acceptable fallback)**

**Why NMF, not PCA, for faces (Lecture 11, Lee & Seung 1999):**

| | PCA (eigenfaces) | NMF |
|---|---|---|
| Basis vectors | Holistic "ghostly" blended faces | Localised parts — eyes, nose, mouth patches |
| Sign | Allows negative values (meaningless for pixels) | $W, H \geq 0$ — physically realistic |
| Representation | Each image = $\sum$ positive + negative eigenfaces | Each image = additive sum of parts |
| Interpretability | Low | High |

**NMF setup:**
$$X \approx WH, \quad W \geq 0,\ H \geq 0$$
- $W \in \mathbb{R}_+^{p \times K}$: basis parts (eye patch, nose region, etc.)
- $H \in \mathbb{R}_+^{K \times N}$: coefficients — how strongly each part appears in each image
- Fit with multiplicative updates; use coefficient columns $h_j$ as features for clustering

**PCA fallback:** flatten → centre → SVD → project onto $k$ components. Simpler to compute but produces holistic eigenfaces. If the exam says "from Lecture 11, what is better?" → **NMF**.

**Step 2 — Clustering (GMM preferred)**

$$p(z) = \sum_{j=1}^{K} \pi_j \, \mathcal{N}(z \mid \mu_j, \Sigma_j)$$

Each component = one unique person. Fit with EM. Full $\Sigma_j$ handles elliptical clusters from pose/lighting variation (K-means assumes spherical → biased splits).

**Step 3 — Model Selection (BIC)**

$$\text{BIC}(K) = -2\log\hat{L} + d_K \log N$$

Fit GMM for $K = 1, 2, \ldots, K_\text{max}$. Choose $K^* = \arg\min_K \text{BIC}(K)$. BIC penalises extra Gaussians — prevents overfitting (too many small clusters).

Alternatives: silhouette score for K-means, gap statistic, dendrogram gap for hierarchical.

**Step 4 — Fraud Detection**

| Result | Interpretation |
|--------|---------------|
| $\hat{K}_\text{faces} \approx K_\text{passports}$ | No fraud signal |
| $\hat{K}_\text{faces} < K_\text{passports}$ | Same person used multiple passports → **fraud** |
| $\hat{K}_\text{faces} > K_\text{passports}$ | Detection errors or passport sharing |

---

### Written Answer (Write This Cold)

*"This is an unsupervised learning problem — no identity labels exist and the task is to discover natural groupings. I would: (1) extract features using NMF ($X \approx WH$, $W,H \geq 0$): pixel intensities are non-negative so the constraint is physically meaningful, and NMF learns localised facial parts (eyes, nose, mouth patches) rather than PCA's holistic ghostly eigenfaces — this was demonstrated by Lee & Seung (1999) on face images; the coefficient vectors $h_j$ become each image's feature representation; (2) cluster using a Gaussian Mixture Model, where each component represents one unique individual — GMM handles elliptical clusters from pose/lighting variation; (3) select $K$ by minimising BIC; (4) compare $\hat{K}_\text{faces}$ to unique passport numbers — fewer unique faces than passports indicates multiple passports per person, strong evidence of fraud."*

**If NMF feels risky:** PCA + GMM + BIC is acceptable and will earn most marks. Mention the NMF advantage as an upgrade ("a better-justified approach from Lecture 11 would be NMF because...").

---

### Key Exam Traps — Face Variant

**"Why not supervised face recognition?"** → No labels. Supervision requires known identities as training targets — circular, since finding identities is the goal.

**"Why NMF over PCA for faces?"** → Three reasons from Lecture 11: (1) pixel intensities are non-negative — NMF's $W,H \geq 0$ is physically meaningful, PCA allows negative "pixel" values; (2) NMF learns localised parts (eye, nose, mouth patches) vs PCA's holistic ghostly eigenfaces; (3) Lee & Seung (1999) demonstrated this directly on a face image dataset.

**"Why GMM over K-means?"** → K-means assumes spherical clusters. Face images of the same person form elliptical clusters (lighting varies one way, expression another). GMM with full covariance handles this correctly.

**"BIC formula — what does $d_K$ penalise?"** → $d_K$ = number of free parameters per extra Gaussian ($= k + k(k+1)/2 + 1$ for mean + covariance + weight). Larger $K$ always improves fit ($-2\log\hat{L}$ decreases) but BIC penalises the added complexity — prevents spurious clusters.

**"How to select $K$ in NMF itself?"** → Speckled CV (matrix masking): randomly mask pixel entries, fit NMF ignoring masked entries, evaluate reconstruction MSE on masked entries only. Choose $K^*$ minimising masked MSE. Cannot do row-holdout CV because you need partial data from every image to estimate $H$.

**"Person with only 1 image?"** → Cannot form a cluster. May be absorbed into nearest cluster or treated as noise. Use GMM with unconstrained mixing weights $\pi_k$ to allow very small components.

---

## Variant 3 — Other Possible Datasets

The CV logic is always the same — only the domain changes. Know the pattern, not each dataset.

| Dataset type | Grouping unit | Correct CV | Typical Q22 ask |
|-------------|--------------|-----------|----------------|
| EEG / brain imaging | Subject | LOSO by subject | Classify mental state for new subject |
| Speech / audio | Speaker | LOSO by speaker | Predict word for unseen speaker |
| Multi-site medical | Hospital / site | LOSO by site | Generalise to new clinical site |
| Longitudinal time-series | Patient + time | Forward-chaining (train past → test future) | Predict next week |
| Gait analysis | Subject | LOSO by subject | Same as wearables, different domain |
| Genomics ($p \gg n$) | Patient | Stratified K-fold + nested CV | Select genes + predict response, no leakage |
| Tensor / EEM fluorescence | — (unsupervised) | CORCONDIA + split-half FMS | How many chemical components? |

**For any of these, the answer follows the same 5-step template:**

1. **Identify the grouping variable** — what makes observations non-IID?
2. **Name the IID violation** — explain why random CV leaks
3. **Design the CV scheme** — draw fold structure, state sizes
4. **State what EPE measures** — new individual? new time point? new site?
5. **Clinical / deployment recommendation** — which model fits the use case?

---

## Universal Q&A Bank

**Why does random CV give optimistic estimates on grouped data?**
The model sees partial data from every group during training → it learns group-specific patterns (individual physiology, speaker accent, hospital equipment). Evaluation on held-out observations from the same group is evaluating partial memorisation, not generalization. Grouped CV holds out entire groups → forces genuine generalization.

**Feature selection outside the CV loop — what goes wrong?**
Feature selection uses all observations including the future test fold to identify predictive features. The selected features are optimistically good on that fold because they were implicitly chosen using it. Feature selection must happen inside each training fold: `Training fold → feature select → fit model → evaluate on held-out fold`.

**Is LOIO biased?**
Slightly pessimistically biased — each fold trains on $\frac{N-1}{N}$ of the data, slightly less than the full dataset. Bias is small and decreases with $N$. Much preferable to the large optimistic bias of random CV. State: "nearly unbiased, slight pessimistic bias from reduced training set per fold."

**$\text{EPE}_\text{gen} > \text{EPE}_\text{pers}$ — statistical explanation:**
$$\text{EPE}_\text{gen} = E_{i_\text{new}}\left[E_{x,y|i_\text{new}}[\mathcal{L}]\right] \quad \text{vs} \quad \text{EPE}_\text{pers} = E_{x,y|i_\text{fixed}}[\mathcal{L}]$$
Generalized integrates over between-individual variance (outer expectation over $i_\text{new}$). Personalized fixes $i$, so no between-individual variance. The gap = between-individual heterogeneity in the population.

**Nested CV — why skip the inner loop causes leakage?**
Selecting λ on the full dataset lets the test subject's data inform λ — the hyperparameter is implicitly tuned to perform well on subjects including the test one. Even though the test subject is not directly in the training set for the outer fold, their data influenced the model configuration. Subtle leakage, real optimistic bias.

**1-SE rule applied to LOIO:**
$$\text{SE} = \frac{\hat{\sigma}_\text{fold}}{\sqrt{16}}$$
Accept the simplest λ whose mean error $\leq \text{EPE}_\text{min} + \text{SE}$. Useful because with 16 outer folds, EPE estimates for nearby λ values are statistically indistinguishable — prefer the simpler, more interpretable model.

**"AIC ≈ LOO-CV asymptotically — can it replace LOIO here?" (exam trap from Week 1):**
No. AIC is equivalent to LOO-CV under the **IID assumption** (Stone 1977). Standard LOO-CV on 192 observations leaves out one obs at a time but trains on data from the same person — still data leakage. AIC shares the same IID assumption and the same leakage problem. Neither replaces grouped CV. AIC/BIC can be used *inside* an outer fold (inner loop model selection on 15 training subjects), but cannot replace the LOIO outer structure.

**Bootstrap for the personalized model's small-sample problem (Week 2):**
The personalized model has only 4 folds (3 test obs each) — very high variance estimate. Bootstrap alternative: draw $B = 200$ resamples from the 12 observations, evaluate on OOB (~37%), average → more stable estimate. For extra rigor, use the .632 estimator:
$$\text{EPE}_{.632} = 0.368 \cdot \text{err}_\text{train} + 0.632 \cdot \text{err}_\text{OOB}$$
This corrects for the slight optimistic bias of the basic bootstrap.
