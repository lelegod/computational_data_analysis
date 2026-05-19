# Open Questions Guide — CDA 02582 (2025 Exam Set)

> Model 10-point answers for **Q21 and Q22 from the May 2025 exam** (exam_vFinal.pdf).
> Q22 (wearable biosignals) has appeared in **all 3 exams (2022, 2024, 2025)** — know this cold.

---

## Q21 — LDA vs GMM (10 pts)

**Question:** Discuss LDA and GMM in terms of: a) probabilistic assumptions, b) how model fitting is performed, c) goals, supervision, and use of labels, d) how each handles class overlap and latent structure.

---

### a) Probabilistic Assumptions (~2.5 pts)

**LDA:**

$$p(x \mid C_k) = \mathcal{N}(\mu_k, \Sigma)$$

- Each class is Gaussian with its **own mean** $\mu_k$ but **shared covariance** $\Sigma$ across all classes
- Uses Bayes' theorem: $p(C_k \mid x) \propto p(x \mid C_k)\,\pi_k$
- The shared $\Sigma$ assumption causes the quadratic terms in the log-posterior ratio to cancel → **linear boundary**

**GMM:**

$$p(x) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x \mid \mu_k, \Sigma_k)$$

- Each component has its **own** $\mu_k$ and $\Sigma_k$ — covariances are not shared
- Models the **marginal density** of $x$, not class-conditional densities

**Key contrast:** LDA forces equal covariance → linear boundary. GMM allows different covariances per component → elliptical clusters of different shapes and sizes.

---

### b) Model Fitting (~2.5 pts)

**LDA — supervised, closed-form:**

| Parameter | Estimate |
|-----------|----------|
| Class means | $\hat{\mu}_k = \frac{1}{n_k}\sum_{i: y_i=k} x_i$ |
| Pooled covariance | $\hat{\Sigma} = \frac{1}{N-K}\sum_k \sum_{i: y_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$ |
| Class priors | $\hat{\pi}_k = n_k / N$ |

Direct MLE from labeled data — no iteration needed.

**GMM — unsupervised, EM algorithm:**

- **E-step:** Compute soft responsibilities (posterior membership probabilities):

$$r_{nk} = \frac{\pi_k\,\mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j}\pi_j\,\mathcal{N}(x_n \mid \mu_j, \Sigma_j)}$$

- **M-step:** Update parameters using weighted averages:

$$\hat{\mu}_k = \frac{\sum_n r_{nk}\,x_n}{\sum_n r_{nk}}, \quad \hat{\Sigma}_k = \frac{\sum_n r_{nk}(x_n - \hat{\mu}_k)(x_n-\hat{\mu}_k)^T}{\sum_n r_{nk}}, \quad \hat{\pi}_k = \frac{\sum_n r_{nk}}{N}$$

Iterates E → M → E → M until convergence (local maximum of log-likelihood).

---

### c) Goals, Supervision, and Use of Labels (~2.5 pts)

| | LDA | GMM |
|--|-----|-----|
| **Supervision** | Supervised | Unsupervised |
| **Labels needed?** | Yes — required for fitting | No |
| **Goal** | Classification into known classes | Density estimation / clustering |
| **Output** | Class label (or posterior probability) | Soft cluster membership $r_{nk}$ |
| **Secondary use** | Dimensionality reduction (Fisher's discriminant) | Density estimation, anomaly detection |

LDA cannot run without labels. GMM discovers structure from data alone — labels can be assigned post-hoc by the highest-responsibility component.

---

### d) Class Overlap and Latent Structure (~2.5 pts)

**LDA:**
- Handles overlap via posterior probabilities — assigns to most probable class: $\hat{C} = \arg\max_k\, p(C_k \mid x)$
- **Hard boundary** in feature space (linear)
- Does **not** model latent structure — classes are pre-defined by labels
- Sensitive to outliers (mean and covariance estimated directly)

**GMM:**
- Handles overlap naturally via **soft assignments** — a point partially belongs to multiple components via $r_{nk}$
- Can reveal **latent structure**: discovers hidden subgroups with no supervision
- Different $\Sigma_k$ per component → can model clusters of different shapes and orientations
- Richer model but no supervision signal — cluster labels are not guaranteed to align with true classes

---

## Q22 — CV Design for Wearable Biosignals (10 pts)

**Question:** Dataset: 16 individuals × 3 activities × 4 seasons = 192 observations (BVP, skin temperature, HR). Design training/validation/test splits to estimate EPE for: a) a personalized model (specific individual), b) a generalized model (new individual). Discuss trade-offs and which is better for clinical deployment.

---

### a) Personalized Model (~4 pts)

**Goal:** Predict stress for a **specific known individual** — like a smart watch that learns your patterns.

**Data available for this individual:** $3 \times 4 = 12$ observations

**CV Design — Leave-One-Season-Out (4-fold):**

```
Fold 1: Train on seasons {2,3,4} (9 obs) → Test on season 1 (3 obs)
Fold 2: Train on seasons {1,3,4} (9 obs) → Test on season 2 (3 obs)
Fold 3: Train on seasons {1,2,4} (9 obs) → Test on season 3 (3 obs)
Fold 4: Train on seasons {1,2,3} (9 obs) → Test on season 4 (3 obs)
```

**Why not random splits?**
Observations within one individual are NOT independent — they share the same physiology. Random splits would leak information and produce optimistically biased EPE estimates. Holding out a full season tests generalization to **new time points**.

**If hyperparameter tuning needed:** Use nested CV — inner loop = leave-one-condition-out within the training seasons.

---

### b) Generalized Model (~4 pts)

**Goal:** Predict stress for a **new, unseen individual** — for clinical deployment with new patients.

**Data:** All 16 individuals × 12 observations = 192 observations

**CV Design — Leave-One-Individual-Out (16-fold):**

```
Fold 1:  Train on individuals {2,...,16} (180 obs) → Test on individual 1 (12 obs)
Fold 2:  Train on individuals {1,3,...,16} (180 obs) → Test on individual 2 (12 obs)
...
Fold 16: Train on individuals {1,...,15} (180 obs) → Test on individual 16 (12 obs)
```

**Critical rule:** All 12 observations from one individual must stay in the **same fold**. Random assignment would mix individuals across folds → data leakage → the model sees the test individual's data during training → EPE estimate is too optimistic.

This CV directly estimates: *"How well does the model predict a person it has never seen?"*

---

### Trade-offs and Clinical Recommendation (~2 pts)

| | Personalized | Generalized |
|--|-------------|------------|
| **Accuracy** | Higher (calibrated to one individual) | Lower (must generalize across physiologies) |
| **Data needed** | Requires prior data from that person | Works immediately for new patients |
| **Applicability** | Only for known individuals with history | Works for anyone |
| **Clinical use** | Long-term monitoring of existing patients | New patients at first visit |

**Clinical recommendation: Generalized model is more appropriate.**

New patients arrive with **no prior data** — a personalized model cannot be trained. The generalized model learns population-level patterns that transfer to new individuals, which is exactly what a clinician needs when seeing a patient for the first time.

> Possible extension (bonus insight): start with the generalized model, then fine-tune it as patient-specific data accumulates over time.
