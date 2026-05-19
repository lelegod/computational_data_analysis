# Q22 — Other Dataset Variants (Exam Scenarios)
> Q22 always presents a real-world dataset and asks: "which methodology, and why?"
> Pattern: either (a) CV design for a non-IID dataset, or (b) unsupervised discovery with model selection.
> This file covers every realistic variant based on course content.

---

## How to Recognise Which Q22 Variant You Have

| Signal in the question | What it is | Core answer |
|------------------------|------------|-------------|
| "predict for new patients / individuals" | CV design, generalized | LOIO-CV |
| "predict for same individuals" | CV design, personalized | LOSO-CV |
| "how many unique X" | Unsupervised, model selection | PCA + GMM/K-means + BIC |
| "multiple measurements per person" | IID violation → grouped CV | Group K-fold by person |
| "time series / temporal data" | Temporal leakage risk | Temporal holdout / forward CV |
| "multiple sites / batches" | Batch effect, site-level grouping | LOSO by site |
| "tensor / multi-way data" | PARAFAC model selection | CORCONDIA + split-half FMS |

---

## Scenario A — EEG / Brain Imaging Data

**Typical setup:** 30 subjects × 40 trials × 256 time points × 64 electrodes. Task: classify mental state (e.g., stress vs. rest) from EEG features.

**Why standard CV fails:** Multiple trials per subject share neural baseline (resting alpha power, noise floor). A random split puts some trials from subject 5 in training and some in test — the model learns subject 5's background EEG → inflated accuracy.

**CV Design:**
- **Generalized model** (predict new subject): Leave-One-Subject-Out (LOSO) — identical logic to LOIO in wearables
- **Within-subject model** (predict new trial): Leave-One-Trial-Out within one subject

**Additional wrinkle — temporal structure within trials:** EEG trials have temporal autocorrelation. If you split randomly within a trial, adjacent time points in test are correlated with training time points. Must hold out complete trials, not individual time samples.

**Feature extraction note:** Apply PCA or ICA to reduce 64 electrodes to $k \ll 64$ components before fitting the classifier. ICA is especially appropriate for EEG because it unmixes independent neural sources.

---

## Scenario B — Speech Recognition / Speaker Dataset

**Typical setup:** 50 speakers × 20 utterances × $p$ acoustic features (MFCCs). Task: predict the spoken word or emotion from audio.

**Why standard CV fails:** Same speaker appearing in both train and test → model learns speaker-specific voice timbre, pitch, accent → tests speaker recognition, not word recognition.

**CV Design:**
- **Speaker-independent model**: Leave-One-Speaker-Out (LOSO by speaker, identical to LOIO)
- **Speaker-dependent model**: Leave-One-Utterance-Out within one speaker

**Key distinction from wearables:** The grouping variable is **speaker**, not **individual** — same concept, different domain. The examiner may change the domain but the CV logic is identical.

**Unsupervised variant:** "How many unique speakers are on this recording?" → PCA on MFCC vectors + GMM + BIC to estimate $K$ speakers. Same pipeline as face clustering.

---

## Scenario C — Multi-Site Medical Study

**Typical setup:** Data collected at 5 hospitals, 100 patients per site, same features (blood markers, imaging). Task: predict disease severity.

**IID violation:** Patients from the same hospital share equipment calibration, local patient population demographics, and clinical protocols. A model trained on 4 hospitals evaluated on the 5th tests genuine generalization. Random splitting mixes hospitals → inflated performance.

**CV Design — Leave-One-Site-Out:**
```
Fold 1: Train on hospitals {2, 3, 4, 5} → Test on hospital 1
Fold 2: Train on hospitals {1, 3, 4, 5} → Test on hospital 2
...
Fold 5: Train on hospitals {1, 2, 3, 4} → Test on hospital 5
```

**What EPE measures:** Generalization to a new clinical site — the relevant metric for a model intended for deployment at hospitals not in the study.

**Why this matters clinically:** Scanner differences (different MRI machines), population differences (different demographics), and protocol differences (different measurement times) all create site-level batch effects. A model that only works at the sites it was trained on has no clinical value.

**Extra: batch effect correction** Before CV, consider harmonising features across sites (e.g., ComBat method). But: harmonisation must happen inside the CV loop (applied to training folds only) to avoid leakage of site statistics from test folds.

---

## Scenario D — Longitudinal / Time-Series Prediction

**Typical setup:** Daily measurements from 200 patients over 2 years. Task: predict patient outcome at time $t+7$ given history up to $t$.

**IID violations — two kinds:**
1. **Within-patient temporal autocorrelation:** Observation at day $t$ is strongly correlated with day $t+1$ for the same patient
2. **Between-patient correlation:** As in wearables

**CV Design — depends on deployment goal:**

*Goal: predict next week for known patients (personalized, future time):*
```
Train on data up to date T → Test on data from T+1 to T+7
Repeat with rolling window
```
This is **forward chaining / time-series CV** — always train on past, test on future. Never use future data to train.

*Goal: predict for new patients (generalized):*
```
Leave-One-Patient-Out, but within each fold also respect temporal ordering
(train on early time points of training patients, test on later time points)
```

**Critical rule:** Never use future time points in the training set, even for the training patients. This creates temporal leakage.

---

## Scenario E — Tensor / Multi-Way Data (PARAFAC)

**Typical setup:** Fluorescence EEM (excitation-emission-sample) tensor, or amino acid fluorescence data: $I$ samples × $J$ emission wavelengths × $K$ excitation wavelengths. Task: determine the number of chemical components in the samples.

**This is model selection for PARAFAC, not CV design.**

**Methodology:**
1. Fit PARAFAC models with $R = 1, 2, 3, \ldots$ components
2. For each $R$, compute CORCONDIA:
   $$\text{CORCONDIA}(R) = 100\left(1 - \frac{\|\mathcal{I} - \tilde{\mathcal{G}}\|_F^2}{\|\mathcal{I}\|_F^2}\right)$$
3. Choose the largest $R$ for which CORCONDIA $\geq 90$
4. Validate with **split-half FMS (Factor Match Score):**
   - Split samples randomly into two halves
   - Fit PARAFAC($R$) to each half independently
   - Compute FMS between the two solutions: $\text{FMS} = \frac{1}{R}\sum_r \cos\theta_r$ (average congruence)
   - Repeat many times; if mean FMS $\geq 0.95$ across splits, $R$ is stable

**Why CORCONDIA:** Measures how close the PARAFAC core is to the super-identity tensor. If $R$ is too large, the core has off-diagonal structure → CORCONDIA drops. This is the uniqueness diagnostic.

**Why split-half FMS:** Even if CORCONDIA is high, the components must be reproducible across subsets of the data. FMS $\approx 1$ means the same chemical components are recovered from independent data splits — strong evidence the components are real.

---

## Scenario F — High-Dimensional Genomics

**Typical setup:** 200 cancer patients × 20,000 gene expression features. Task: predict treatment response (binary). $p \gg n$ — classic high-dimensional setting.

**Why standard CV fails here (a different issue):**
The $p \gg n$ problem means a model trained without regularization will perfectly fit the training data (zero training error) but generalize poorly. The CV design must account for:
1. Patient-level grouping (as usual)
2. Feature selection / regularization must be inside the CV loop

**CV Design:**
- Outer loop: Leave-One-Patient-Out or stratified K-fold (stratify by treatment response label to ensure class balance in each fold)
- Inner loop: select regularization parameter $\lambda$ (Lasso or Ridge) using inner CV
- Feature selection (if used): INSIDE each outer training fold only

**Trap the exam may set:** "A researcher selects the top 100 genes by univariate t-test on the full dataset, then does LOOCV." This is **data leakage** — the t-test used the test patient's data to pick genes. The 100-gene set is optimistically biased. Gene selection must happen inside each training fold.

**EPE decomposition insight:**
$$\text{EPE} = \text{Bias}^2 + \text{Variance} + \sigma^2$$
In $p \gg n$ settings without regularization: Variance is large (model is unstable across training folds). Lasso reduces variance by zeroing out uninformative genes, at the cost of some bias. Ridge shrinks all coefficients smoothly. Elastic net combines both.

---

## Scenario G — Gait Analysis (Close Wearables Variant)

**Typical setup:** 24 subjects × 5 walking conditions × 3 sessions per condition = 360 trials. Features: joint angles, ground reaction forces. Task: classify walking condition or predict fall risk.

This is structurally identical to the wearables Q22, just with walking instead of stress biosignals:
- **Personalized model:** Leave-One-Session-Out within one subject
- **Generalized model:** Leave-One-Subject-Out across 24 subjects

**Extra Q22 angle:** "The researcher wants to use the model to detect fall risk in elderly patients in a care home. Should they use the personalized or generalized model?" Answer: Generalized — new patients have no prior data. Same clinical reasoning as wearables.

---

## Scenario H — Diabetes / Glucose Monitoring

**Typical setup:** 40 diabetes patients, each with repeated glucose windows across many days. Features may include CGM summaries, insulin dose, meal timing, sleep, and activity. Task: predict hypoglycemia, time-in-range, or next-hour glucose.

**Why this is a strong Q22 candidate:** It combines the same grouped-CV logic as wearables with explicit temporal leakage risk.

### Why random CV fails

Observations from the same patient share:
- baseline physiology
- medication regime
- insulin sensitivity
- sensor behavior

So random splitting leaks patient-specific information.

### Correct design

**New-patient deployment:** Leave-One-Patient-Out CV

**Known-patient forecasting:** forward-chaining or leave-one-day-out within one patient

**If using sliding windows:** do not place overlapping windows in both training and test.

### Additional exam angle

If the target is rare, such as hypoglycemia:
- use sensitivity, specificity, balanced accuracy, or AUC
- do not rely on raw accuracy

### Full one-line template

*"This is a repeated-measures longitudinal dataset, so validation must respect both patient identity and temporal order. For new-patient deployment I would use leave-one-patient-out CV; for within-patient forecasting I would use forward-chaining on future windows only."*

---

## Summary Table — All 8 Scenarios

| Dataset | Grouping variable | Personalized CV | Generalized CV | Unsupervised alternative |
|---------|-----------------|----------------|---------------|--------------------------|
| Wearables (stress) | Individual | Leave-one-season-out | Leave-one-individual-out | — |
| Face images | Identity | — | — | PCA + GMM + BIC |
| EEG | Subject | Leave-one-trial-out | Leave-one-subject-out | ICA for source separation |
| Speech | Speaker | Leave-one-utterance-out | Leave-one-speaker-out | PCA + GMM for #speakers |
| Multi-site medical | Hospital/site | — | Leave-one-site-out | — |
| Longitudinal | Patient + time | Rolling forward CV | LOPO + temporal order | — |
| Tensor (EEM) | Sample | — | — | PARAFAC + CORCONDIA + FMS |
| Genomics | Patient | — | Stratified K-fold | PCA for exploration |
| Gait analysis | Subject | Leave-one-session-out | Leave-one-subject-out | — |
| Diabetes / CGM | Patient + time | Leave-one-day-out / forward CV | Leave-one-patient-out | — |

---

## Universal Answer Template for Q22

No matter the dataset, structure your answer as:

1. **Identify the prediction scenario** — personalized (known individual) vs. generalized (new individual) vs. unsupervised (unknown groups)

2. **Identify the IID violation** — state which grouping variable causes correlation (individual, site, time, etc.) and why random CV would leak

3. **Design the CV scheme** — name the method (LOIO, LOSO, forward-chaining, etc.), draw the fold structure, state training and test sizes

4. **State what the EPE measures** — be specific: "expected error when predicting a new individual drawn from the same population"

5. **Handle hyperparameters** — if a model has tunable parameters, nested CV is required; the test fold must never inform parameter selection

6. **Clinical/deployment recommendation** — which model is appropriate given the deployment context

Every exam variant maps onto this template. The domain changes; the logic does not.
