# AGENTS.md — CDA 02582 Computational Data Analysis
> Read this file first. It gives you full context on the course, exam format, folder structure, and how to help effectively.

---

## Who You Are Helping

DTU student preparing for the **CDA 02582 — Computational Data Analysis** written exam. The student has already generated comprehensive study material across this folder. Your job is to help with exam prep: explaining concepts, testing knowledge, answering questions, and pointing to the right files.

---

## Exam Format (Critical — Memorize This)

### Multiple Choice — 18 Questions (58 points total)
- **18 MCQ questions**, each worth up to 2 points
- Each question has **one correct answer** (single select)
- Grading per question:
  - Correct answer: **+1 point**
  - Incorrect answer: **-0.25 points**
  - No answer: **0 points**
- Strategy: only answer if confidence > ~20% (expected value of guessing = 0)

### Open Questions — 2 Questions (40 points total)
- **Q21**: 20 points — explain/compare/derive a key algorithm or concept in depth
- **Q22**: 20 points — almost always the wearables CV design question (same dataset 2022, 2024, 2025)
- Both require structured written answers with mathematical justification

### Total: 58 + 40 = 98 points (roughly)

### Past Exam Pattern
| Year | Q21 Topic | Q22 Topic |
|------|-----------|-----------|
| 2022 | Random Forest algorithm | Clustering for face images |
| 2024 | ICA uniqueness and distributions | CV design for wearables |
| 2025 | LDA vs GMM comparison | CV design for wearables |

**Q22 prediction**: CV design for wearables dataset will appear again. Prepare it cold.

---

## Folder Structure — Where Everything Is

```
computational_data_analysis/
│
├── AGENTS.md                    ← You are here
├── OPEN_QUESTIONS_Q21.md        ← Root cheat sheet: all 28 Q21 candidates (A–AB), clickable TOC
├── OPEN_QUESTIONS_Q22.md        ← Root cheat sheet: Q22 CV wearables, full exam-ready answer
│
├── open_questions/              ← DETAILED answers, split by question number
│   │
│   ├── Q21/                     ← All Q21 candidate deep-dives (28 files, A–AB)
│   │   ├── INDEX.md             ← Overview table of all candidates + writing strategy
│   │   ├── Q21_A_random_forest.md
│   │   ├── Q21_B_ica.md
│   │   ├── Q21_C_lda_vs_gmm.md
│   │   ├── Q21_D_svm.md
│   │   ├── Q21_E_boosting.md
│   │   ├── Q21_F_parafac_tucker.md
│   │   ├── Q21_G_pca_pls_cca.md
│   │   ├── Q21_H_nmf_ica_aa.md
│   │   ├── Q21_I_ridge_lasso.md
│   │   ├── Q21_J_clustering.md
│   │   ├── Q21_K_multiple_testing.md
│   │   ├── Q21_L_neural_networks.md
│   │   ├── Q21_M_epe_bias_variance.md
│   │   ├── Q21_N_cart.md
│   │   ├── Q21_O_cross_validation.md
│   │   ├── Q21_P_logistic_regression.md
│   │   ├── Q21_Q_ols_gauss_markov.md
│   │   ├── Q21_R_bootstrap.md
│   │   ├── Q21_S_curse_dimensionality.md
│   │   ├── Q21_T_aic_bic.md
│   │   ├── Q21_U_bagging.md
│   │   ├── Q21_V_cluster_validation.md
│   │   ├── Q21_W_sparse_pca.md
│   │   ├── Q21_X_qda.md
│   │   ├── Q21_Y_kmedoids.md
│   │   ├── Q21_Z_gmm.md
│   │   ├── Q21_AA_split_half_fms.md
│   │   └── Q21_AB_pcr.md
│   │
│   └── Q22/                     ← All Q22 deep-dives (move Q22_*.md here)
│       ├── Q22_cv_wearables.md  ← Full Q22 model answer (LOSO vs LOIO, nested CV, EPE)
│       ├── Q22_face_clustering_2022.md
│       └── Q22_other_datasets.md
│
├── summary/
│   ├── exam_review.md           ← Topic frequency analysis + 3 flagged errors in official solutions
│   ├── all/                     ← Full lecture-faithful summaries (week01–week12.md)
│   └── exam_focused/            ← Exam-optimized summaries (week01–week12.md)
│
├── exam/
│   ├── solutions_2022.md        ← Full detailed solutions, all 22 questions
│   ├── solutions_2024.md        ← Full detailed solutions, all 22 questions
│   ├── solutions_2025.md        ← Full detailed solutions, all 22 questions
│   ├── q21_lda_vs_gmm_2025.md   ← Extended Q21 answer for 2025
│   └── q22_cv_wearables_2025.md ← Extended Q22 answer for 2025
│
├── gen_set/
│   ├── set1/                    ← Practice exam set 1 (questions.md + solutions.md)
│   └── set2/                    ← Practice exam set 2, harder, skewed weeks 7–12
│
├── week1/ … week12/             ← Raw lecture PDFs (week<N>/lecture.pdf or similar)
│
└── extras/                      ← Older/archived open question files
```

---

## Course Content — Week by Week

| Week | Topic | Key concepts |
|------|-------|-------------|
| 1 | Regression & Bias-Variance | OLS, Ridge, EPE decomposition |
| 2 | Model Selection & Assessment | KNN, train/val/test split, K-fold CV, 1-SE rule, data leakage, Cp, AIC, BIC |
| 3 | Sparse Regression | Curse of dimensionality, Ridge, Lasso, Elastic Net, LARS, Multiple Testing (Bonferroni, BH) |
| 4 | Linear Classification | LDA, QDA, RDA, RRDA, Logistic Regression, generative vs discriminative |
| 5 | CART and Bagging | Regression trees, classification trees, Gini, cost-complexity pruning, Bootstrap Aggregating |
| 6 | Ensemble Methods | Random Forest, AdaBoost, Gradient Boosting, additive models |
| 7 | SVM | Max-margin, Lagrangian dual, kernel trick, RBF, soft margin |
| 8 | Subspace Methods | PCA, Sparse PCA (PMD), PLS, CCA |
| 9 | Unsupervised Clustering | K-means, K-medoids, Hierarchical, GMM, LDA, QDA, Silhouette, Gap |
| 10 | Neural Networks | Backpropagation, activations (ReLU/sigmoid), dropout, vanishing gradient |
| 11 | Matrix Decompositions | NMF, ICA, Archetypal Analysis, Sparse Coding |
| 12 | Tensor Decompositions | Tucker3, PARAFAC, CORCONDIA, Split-half FMS |

---

## Known Errors in Official Exam Solutions

Three errors have been identified in the official answer keys:

| Exam | Question | Issue |
|------|---------|-------|
| 2022 | Q20 | Options A and E are contradictory — correct answer is C (any tree) |
| 2024 | Q11 | Option C is wrong — proximity plots measure observations, not variables |
| 2022 | Q9 | Ambiguous hedging — Option A is most defensible |

Full details in `summary/exam_review.md`.

---

## How to Use This Material Effectively

### For a specific Q21 topic
1. Quick review: `OPEN_QUESTIONS_Q21.md` (condensed, with clickable TOC)
2. Full model answer: `open_questions/Q21/Q21_<letter>_<topic>.md`
3. Index of all candidates: `open_questions/Q21/INDEX.md`

### For Q22
1. `OPEN_QUESTIONS_Q22.md` — memorize the full written answer cold
2. Extended answers: `open_questions/Q22/Q22_cv_wearables.md` (and other dataset variants)

### For MCQ prep
1. `summary/exam_focused/week<N>.md` — exam-optimized, includes common traps
2. `exam/solutions_20XX.md` — full solutions with detailed reasoning per option

### For practice exams
- `gen_set/set1/` and `gen_set/set2/` — 18 MCQ + 2 open questions each

---

## Q21 Writing Template (20 points)

Every Q21 answer should follow this structure:
1. **State the model** — formula, one sentence
2. **Explain the mechanism** — why does each step work?
3. **Key properties** — bias/variance, uniqueness, complexity
4. **Compare to alternatives** — name the distinguishing property
5. **Limitations** — when does it fail?

**Marks come from**: correct objective function · explaining WHY · comparison · formula · edge case behavior

**Common mistakes**: saying "it minimizes error" without specifying which loss · listing steps without explaining what each achieves · forgetting model assumptions · confusing variance reduction (bagging) with bias reduction (boosting)

---

## Q22 Dataset (Always the Same)

- **16 subjects × 3 conditions × 4 seasons = 192 observations**
- Task: predict activity from wearable biosignals
- Part a) Personalized: Leave-One-Season-Out CV within one subject (4 folds, 9 train obs)
- Part b) Generalized: Leave-One-Individual-Out CV (16 folds, 180 train obs)
- Key concept: IID violated — random CV = data leakage; must split by individual

---

## Formatting Conventions

- All math is LaTeX in markdown: `$...$` for inline, `$$...$$` for display
- Files are designed to render in VS Code, Obsidian, or any markdown previewer
- cSpell warnings on technical terms (PARAFAC, negentropy, etc.) are false positives — ignore them
