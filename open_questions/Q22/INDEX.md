# Q22 — Index
> **20 points.** Q22 has appeared in every exam since 2022. The wearable dataset appeared twice in a row (2024, 2025). Know the main variants cold.
> Master cheat sheet (one page, all variants): `OPEN_QUESTIONS_Q22.md` in the repo root.

---

## Past Exam Pattern

| Year | Q22 Topic | Core skill tested |
|------|-----------|------------------|
| 2022 | Face image clustering at Copenhagen Airport | Unsupervised learning, model selection |
| 2024 | CV design for wearable biosignals | Grouped CV, IID violation, EPE |
| 2025 | CV design for wearable biosignals | Same dataset + clinical deployment question |

**Pattern:** Q22 always gives a real-world dataset and asks "which methodology, and why?" It tests whether you can match the CV design (or unsupervised pipeline) to the actual deployment goal. The same dataset appeared in 2024 and 2025 — know the wearable answer cold first.

---

## File Index

| File | Variant | Appeared | Read order |
|------|---------|---------|------------|
| [Q22\_cv\_wearables.md](Q22_cv_wearables.md) | Supervised, grouped CV design | 2024, 2025 | **First** |
| [Q22\_face\_clustering\_2022.md](Q22_face_clustering_2022.md) | Unsupervised, NMF + GMM + BIC | 2022 | Second |
| [Q22\_diabetes\_cgm.md](Q22_diabetes_cgm.md) | Supervised, patient-grouped CV + temporal leakage | Not yet | Third |
| [Q22\_eeg\_subject\_cv.md](Q22_eeg_subject_cv.md) | Supervised, subject-grouped CV + trial/window leakage | Not yet | Fourth |
| [Q22\_speech\_speaker\_cv.md](Q22_speech_speaker_cv.md) | Supervised, speaker-grouped CV | Not yet | Fifth |
| [Q22\_multisite\_medical.md](Q22_multisite_medical.md) | Supervised, site-level grouped CV | Not yet | Sixth |
| [Q22\_longitudinal\_timeseries.md](Q22_longitudinal_timeseries.md) | Supervised, forward-chaining / temporal holdout | Not yet | Seventh |
| [Q22\_tensor\_parafac.md](Q22_tensor_parafac.md) | Unsupervised, PARAFAC + CORCONDIA + FMS | Not yet | Eighth |
| [Q22\_genomics\_highdim.md](Q22_genomics_highdim.md) | Supervised, high-dimensional nested CV | Not yet | Ninth |
| [Q22\_gait\_analysis.md](Q22_gait_analysis.md) | Supervised, subject-level grouped CV | Not yet | Tenth |
| [Q22\_other\_datasets.md](Q22_other_datasets.md) | Summary bank across all unseen variants | Not yet | Last |

---

## Recognise the Variant in 30 Seconds

| Phrase in the question | Variant | File |
|------------------------|---------|------|
| "predict for a **new** individual / patient" | Supervised generalized | wearables |
| "predict for the **same** individual" | Supervised personalized | wearables |
| "how many **unique** X" | Unsupervised discovery | face clustering |
| "repeated measures / multiple obs per person" | IID violation → grouped CV | wearables |
| "multiple sites / hospitals / batches" | Site-level grouping | other datasets |
| "time series / predict next week" | Temporal leakage | other datasets |
| "glucose / insulin / diabetes / CGM" | Patient-grouped + temporal CV | diabetes |
| "tensor / multi-way / PARAFAC" | Component selection | other datasets |

---

## What Is in Each File

### [Q22\_cv\_wearables.md](Q22_cv_wearables.md)
The core Q22. Covers:
- Dataset: 16 subjects × 3 activities × 4 seasons = 192 observations
- Part a) Personalized model → Leave-One-Season-Out (4-fold within one subject)
- Part b) Generalized model → Leave-One-Individual-Out (16-fold across all subjects)
- Part c) Comparison table, clinical recommendation
- Part d) Nested CV structure + performance metric selection
- Extended Q&A: LOAO trap, 5-fold vs LOIO gap, EPE decomposition, LOIO bias, bootstrap for small samples, AIC trap, 1-SE rule, boxplot interpretation

### [Q22\_face\_clustering\_2022.md](Q22_face_clustering_2022.md)
The 2022 variant. Covers:
- Why this is unsupervised (no labels, discovery problem)
- Feature extraction: **NMF preferred over PCA** (Lecture 11 — parts-based, non-negative pixels, Lee & Seung 1999)
- NMF vs PCA vs VQ comparison table
- Clustering: GMM with full covariance (handles elliptical clusters)
- Model selection: BIC to choose $K$ = number of unique people
- Fraud detection logic: compare $\hat{K}_\text{faces}$ vs $K_\text{passports}$
- Extended Q&A: NMF non-uniqueness, speckled CV for component selection, AA vs NMF, semi-supervised variant, imbalanced cluster sizes

### [Q22\_diabetes\_cgm.md](Q22_diabetes_cgm.md)
Plausible supervised healthcare variant. Covers:
- repeated CGM / insulin / lifestyle measurements per patient
- why random CV leaks patient-specific physiology
- personalized model: future prediction within known patient
- generalized model: Leave-One-Patient-Out CV
- temporal leakage from overlapping windows / future data
- nested CV and feature-selection-inside-loop rule
- classification vs regression metric choice
- exam-ready full written answer

### Additional standalone unseen variants

- [Q22\_eeg\_subject\_cv.md](Q22_eeg_subject_cv.md): subject-level CV, trial/window leakage, PCA/ICA note
- [Q22\_speech\_speaker\_cv.md](Q22_speech_speaker_cv.md): Leave-One-Speaker-Out, utterance/frame leakage
- [Q22\_multisite\_medical.md](Q22_multisite_medical.md): Leave-One-Site-Out, hospital/batch leakage
- [Q22\_longitudinal\_timeseries.md](Q22_longitudinal_timeseries.md): forward-chaining, future-data leakage
- [Q22\_tensor\_parafac.md](Q22_tensor_parafac.md): unsupervised tensor component selection
- [Q22\_genomics\_highdim.md](Q22_genomics_highdim.md): $p \gg n$, nested CV, feature-selection leakage
- [Q22\_gait\_analysis.md](Q22_gait_analysis.md): wearables-style subject/session CV

### [Q22\_other\_datasets.md](Q22_other_datasets.md)
Preparation for unseen variants. Covers 8 scenarios:
- EEG / brain imaging (LOSO by subject)
- Speech / audio (LOSO by speaker)
- Multi-site medical (LOSO by hospital)
- Longitudinal time-series (forward-chaining CV)
- Tensor / EEM fluorescence (PARAFAC + CORCONDIA + split-half FMS)
- High-dimensional genomics (stratified K-fold, feature selection inside loop)
- Gait analysis (same structure as wearables)
- Universal 5-step answer template for any Q22 dataset

---

## Writing Strategy for Q22

Q22 is not about describing an algorithm — it is about **matching methodology to deployment goal**. Marks come from:

1. **Identifying the prediction scenario** — personalized vs. generalized vs. unsupervised. One sentence.
2. **Naming the IID violation** — why are observations not independent? Which grouping variable?
3. **Designing the CV scheme** — draw the fold structure, state training and test sizes explicitly.
4. **Stating what EPE measures** — "expected error when predicting a new individual from the population" not just "test error".
5. **Justifying the method** — why this grouping, why not random CV, why not AIC.
6. **Clinical/deployment recommendation** — which model is appropriate, and why the other one cannot be used.

### Leave-One-Group-Out vs Grouped K-Fold

- **Grouped CV** is the rule; **leave-one-group-out** is one special case.
- If the deployment target is “new patient / new subject / new site,” both are acceptable only if the whole group stays in one fold.
- **Leave-one-group-out** is usually the best exam answer when the number of groups is small or moderate, because it maps most directly to the deployment question.
- **Grouped K-fold** is often preferred when there are many groups and you want larger test folds and a more stable estimate.
- The core tradeoff is: leave-one-group-out is often cleaner but noisier; grouped K-fold is often more stable but slightly less direct.

### Common Mistakes to Avoid
- Proposing random K-fold without acknowledging the IID violation
- Saying "LOIO gives unbiased estimates" without explaining *what* it is unbiased for (new-patient performance)
- Missing the data leakage argument: model learns personal physiological baseline from partial within-subject data
- For the face question: defaulting to PCA without mentioning NMF (Lecture 11 makes NMF the better-justified answer)
- Tuning hyperparameters outside the CV loop (feature selection, λ selection must be inside each outer fold)
- Using AIC as a substitute for grouped CV (AIC ≈ LOO-CV only under IID — same leakage problem)

---

## Key Formulas to Memorise

| Formula | What it is | When to use |
|---------|-----------|-------------|
| $\text{EPE}_\text{pers} = E_{x,y\mid i_\text{fixed}}[\mathcal{L}(y,\hat{f}_i(x))]$ | Personalized EPE | Part a) answer |
| $\text{EPE}_\text{gen} = E_{i_\text{new}}[E_{x,y\mid i_\text{new}}[\mathcal{L}(y,\hat{f}(x))]]$ | Generalized EPE | Part b) answer |
| $\text{SE} = \hat{\sigma}_\text{fold}/\sqrt{K}$ | SE of mean EPE across $K$ folds | Reliability of CV estimate |
| $\text{EPE}_{.632} = 0.368\cdot\text{err}_\text{train} + 0.632\cdot\text{err}_\text{OOB}$ | Bootstrap .632 estimator | Small-sample personalized model |
| $X \approx WH,\ W,H\geq 0$ | NMF decomposition | Face feature extraction |
| $H_{kj}\leftarrow H_{kj}\cdot\frac{(W^TX)_{kj}}{(W^TWH)_{kj}}$ | NMF multiplicative update for $H$ | NMF fitting |
| $p(z)=\sum_j\pi_j\mathcal{N}(z\mid\mu_j,\Sigma_j)$ | GMM | Clustering in feature space |
| $\text{BIC}(K)=-2\log\hat{L}+d_K\log N$ | BIC for GMM | Selecting number of clusters $K$ |
| $s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}$ | Silhouette score | Alternative $K$ selection |
