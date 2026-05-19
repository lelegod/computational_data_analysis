# Exam Solutions — CDA 02582 (2024)

**Date:** May 16, 2024
**Format:** 20 MC questions + 2 open questions
**Scoring:** +1 per correct option selected, −1 per wrong option selected, 0 if unanswered
**Open questions:** 0–10 points each

---

## Answer Key Summary

| Q | Correct Options |
|---|----------------|
| 1 | E |
| 2 | A, B, D |
| 3 | D |
| 4 | B, D |
| 5 | B, C |
| 6 | D |
| 7 | B, C, D |
| 8 | A |
| 9 | C |
| 10 | B, C |
| 11 | A, D ⚠️ (C is marked correct officially but contains an error) |
| 12 | A, C |
| 13 | E |
| 14 | B |
| 15 | B |
| 16 | C |
| 17 | A, D |
| 18 | B |
| 19 | C |
| 20 | A, C |

---

## Multiple Choice Questions

---

### Question 1 — Supervised vs. Unsupervised Methods

**Question:** Which of the following methods are supervised methods?

**Official Answer:** E (None of the above)

**Option A — Gaussian Mixture Modeling:** ❌ Wrong
GMM fits a mixture of Gaussian distributions to data without using any class labels. It is an unsupervised generative model that finds clusters by maximizing the marginal likelihood $p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ using the EM algorithm. No output variable $y$ is involved.

**Option B — Autoencoder:** ❌ Wrong
An autoencoder is a neural network trained to reconstruct its own input: $\hat{x} = f_\theta(x)$, minimizing $\|x - \hat{x}\|^2$. It is an unsupervised method — the "label" is the input itself. No external response variable is used during training.

**Option C — K-means Clustering:** ❌ Wrong
K-means partitions data into $K$ clusters by minimizing within-cluster sum of squared distances: $\arg\min_{C_1,\ldots,C_K} \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$. No labels are used. It is the canonical unsupervised clustering algorithm.

**Option D — Tucker:** ❌ Wrong
Tucker decomposition is a multiway/tensor factorization method: $\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$, where $\mathcal{G}$ is a core tensor and $A, B, C$ are factor matrices. It is an unsupervised dimensionality reduction technique for multi-dimensional arrays — no response variable is present.

**Option E — None of the above:** ✓ Correct
All four methods (A–D) are unsupervised: they discover structure in data without relying on a labeled response variable $y$. Supervised methods (e.g., Random Forest, Logistic Regression, SVM, Linear Regression) require labeled training data.

> **Key takeaway:** Supervised methods require a labeled response variable $y$; GMM, Autoencoder, K-means, and Tucker are all unsupervised — they find structure without labels.

---

### Question 2 — Methods for High-Dimensional Data (p >> n)

**Question:** Which of the following methods handle problems with more variables than observations (p >> n) well?

**Official Answer:** A, B, D

**Option A — The Elastic Net Regression:** ✓ Correct
Elastic Net combines L1 and L2 penalties: $\arg\min_\beta \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$. The regularization shrinks and selects variables, making it well-suited for $p \gg n$ settings where OLS is undefined. The L2 component handles correlated groups of variables, and the L1 component induces sparsity.

**Option B — Principal Component Analysis:** ✓ Correct
PCA reduces a $p$-dimensional dataset to $K \ll p$ principal components by computing the SVD: $X = U D V^T$. Even when $p \gg n$, only $\min(n, p)$ non-zero eigenvalues exist, so PCA can operate in the $n$-dimensional space. PCA does not require $X^TX$ to be invertible; it works with the rank-$n$ decomposition.

**Option C — Ordinary Least Squares Regression:** ❌ Wrong
OLS requires computing $\hat{\beta} = (X^TX)^{-1}X^Ty$. When $p > n$, the matrix $X^TX$ is rank-deficient (at most rank $n < p$) and therefore singular — the inverse does not exist. OLS is completely undefined in the $p \gg n$ regime, and even when a pseudoinverse is used, the solution is not unique and overfits completely.

**Option D — Random Forest Classification:** ✓ Correct
At each split, Random Forest randomly selects $m$ out of $p$ features (commonly $m = \sqrt{p}$), so it never needs to process all $p$ variables simultaneously. This random feature subsampling makes RF scalable and effective even when $p \gg n$, without needing matrix inversions of size $p \times p$.

**Option E — None of the above:** ❌ Wrong
Options A, B, and D all handle $p \gg n$ well, so "none of the above" is incorrect.

> **Key takeaway:** OLS fails for $p \geq n$ because $X^TX$ becomes singular; Elastic Net, PCA, and Random Forest each have mechanisms (regularization, dimensionality reduction, random subsampling) that bypass this problem.

---

### Question 3 — Random Forest: True Statements

**Question:** Which of the following statements are true for Random Forest?

**Official Answer:** D

**Option A — Choosing a small number of randomly selected variables to try in each split leads to a smaller reduction in variance than choosing a larger number of randomly selected variables:** ❌ Wrong
This is the opposite of what happens. Using fewer variables per split ($m$ small) creates more diverse, less correlated trees. The variance of an average of $B$ correlated trees is $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$, where $\rho$ is the pairwise correlation. Smaller $m$ reduces $\rho$, which reduces the ensemble variance more. So fewer variables per split leads to a *larger* reduction in variance (through lower tree correlation), not smaller.

**Option B — The Random Forest algorithm cannot be parallelized because a tree in the forest depends on the previous trees:** ❌ Wrong
This describes Boosting, not Random Forest. In RF, each tree is grown independently on its own bootstrap sample with no reference to other trees. All $B$ trees can be trained simultaneously in parallel. The sequential dependency is the defining feature of boosting (forward stagewise additive modelling), not bagging/RF.

**Option C — Instead of trees, it is possible to use e.g., a KNN model as the individual models in the Random Forest:** ❌ Wrong
Random Forest is specifically a forest of decision trees. The key mechanism — random feature selection at each split — is intrinsic to tree-based splitting and cannot be applied to KNN in the same way. KNN does not have a "split" operation, so the RF framework (bootstrap + random feature selection at splits) does not translate to KNN base learners.

**Option D — Fully grown trees are more suitable than smaller trees or stumps as the individual models in a Random Forest:** ✓ Correct
The bias of a Random Forest equals the bias of a single individual tree. Deep (fully grown) trees have low bias; shallow trees or stumps have high bias. Because RF reduces variance through averaging and random feature selection, the remaining performance bottleneck is bias — hence deep trees are preferred. Stumps are instead the preferred weak learner for Boosting.

**Option E — None of the above:** ❌ Wrong
Option D is correct, so "none of the above" is incorrect.

> **Key takeaway:** RF bias = individual tree bias, so deep trees are preferred; RF trees are grown independently (parallelizable), which is the opposite of boosting's sequential structure.

---

### Question 4 — Lasso: Effect of Too-Small Lambda

**Question:** You happen to select a too small $\lambda$ in your lasso regularization, $\arg\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_1$. How will that affect the estimated model?

**Official Answer:** B, D

**Option A — It will have high bias:** ❌ Wrong
A small $\lambda$ imposes very little regularization. The model is close to the OLS solution, which fits the training data tightly. This means low bias (the model captures the true signal well), not high bias. High bias is caused by *large* $\lambda$, which over-shrinks coefficients toward zero.

**Option B — It will have high variance:** ✓ Correct
With a very small $\lambda$, the L1 penalty has minimal effect and the model approximates OLS. OLS is known to overfit when the signal-to-noise ratio is modest — the estimated coefficients $\hat{\beta}$ vary considerably across different training samples. This is the definition of high variance.

**Option C — Not possible to say:** ❌ Wrong
The bias-variance tradeoff is entirely predictable from the value of $\lambda$: small $\lambda$ = low bias + high variance. The relationship is deterministic given the regularization framework.

**Option D — It will have low bias:** ✓ Correct
Small $\lambda$ means the penalty term $\lambda\|\beta\|_1$ barely constrains the optimization. The solution closely tracks the training data, so the expected value of the estimator is close to the true $\beta$ — this is low bias. Both B and D are simultaneously correct: small $\lambda$ produces low bias AND high variance.

**Option E — None of the above:** ❌ Wrong
Options B and D are both correct, so E is wrong.

> **Key takeaway:** Small $\lambda$ in Lasso approximates OLS: low bias (model fits data) but high variance (sensitive to training data fluctuations). Large $\lambda$ gives the reverse — high bias, low variance.

---

### Question 5 — Regularized Regression Algorithms

**Question:** Which of the following statements are true for regularized regression algorithms?

**Official Answer:** B, C

**Option A — Lasso has a closed form solution:** ❌ Wrong
The Lasso objective $\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$ is not differentiable at $\beta_j = 0$ due to the L1 norm $\|\beta\|_1 = \sum_j |\beta_j|$. Because the subgradient at zero creates an interval rather than a unique derivative, no closed-form solution exists. Numerical algorithms (coordinate descent, LARS) are required. By contrast, Ridge has the closed-form solution $\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1}X^Ty$ because the L2 penalty is smooth everywhere.

**Option B — Lasso is a path algorithm:** ✓ Correct
The Lasso solution path (LARS algorithm) traces $\hat{\beta}(\lambda)$ as $\lambda$ decreases from $\infty$ (all coefficients zero) to 0 (OLS). The path is piecewise linear, meaning it can be computed exactly and efficiently for all values of $\lambda$ simultaneously.

**Option C — A path algorithm has the advantage of providing solutions for all relevant regularization values:** ✓ Correct
Path algorithms compute the entire regularization path $\{\hat{\beta}(\lambda) : \lambda \geq 0\}$ in one pass, rather than solving a separate optimization for each $\lambda$ value. This is computationally advantageous for cross-validation over a grid of $\lambda$ values, since the number of steps equals the number of variables.

**Option D — The closed form solutions are derived using asymptotic theory ($N \to \infty$):** ❌ Wrong
The Ridge closed-form solution $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$ is an exact algebraic result valid for any finite $n$ — it is derived by setting the gradient of the objective to zero and solving the resulting linear system. No asymptotic theory ($N \to \infty$) is required or used. Asymptotic theory is used for inference (standard errors, hypothesis tests), not for deriving the point estimate.

**Option E — None of the above:** ❌ Wrong
Options B and C are both correct.

> **Key takeaway:** Lasso has no closed form (non-differentiable at zero) but is a path algorithm that efficiently provides solutions for the entire $\lambda$ range in one computation.

---

### Question 6 — Information Criteria

**Question:** Which of the following statements are true for information criteria? ($n$ is the number of observations and $p$ is the number of variables)

**Official Answer:** D

**Option A — The in-sample error is defined as the error when we sample a new training set at random:** ❌ Wrong
The in-sample error (also called training error) is defined as the error computed on the same data that was used to fit the model: $\frac{1}{n}\sum_{i=1}^n L(y_i, \hat{f}(x_i))$. It is NOT the error when sampling a new training set — that would be the expected training error or generalization error. Information criteria correct for the optimistic bias of in-sample error by adding a penalty term.

**Option B — AIC works equally well for n >> p and n << p situations:** ❌ Wrong
AIC $= -2\log\hat{L} + 2p$ is derived under the assumption that the number of parameters $p$ is small relative to $n$. Its correction to the in-sample error is $2p/n$, which becomes unreliable when $p$ is comparable to or greater than $n$. For high-dimensional settings ($p \gg n$), AIC is not valid; regularization-based methods or corrected AIC variants (AICc) are needed.

**Option C — BIC works for all supervised models in the course, as long as n is high:** ❌ Wrong
BIC $= -2\log\hat{L} + p\log(n)$ requires computing a log-likelihood $\log\hat{L}$. This presupposes a correctly specified parametric model with a well-defined likelihood. Models like Random Forest, K-Nearest Neighbors, or SVMs do not have a simple closed-form likelihood, so BIC cannot be directly applied to them regardless of sample size.

**Option D — BIC is comparing models according to their posterior odds:** ✓ Correct
BIC approximates $-2 \log p(\text{data} \mid M_k)$, the log marginal likelihood of model $M_k$. Comparing BIC values across two models approximates the log Bayes factor $\log \frac{p(\text{data} \mid M_1)}{p(\text{data} \mid M_2)}$, which is equivalent to comparing posterior odds when priors over models are equal. This is the fundamental Bayesian interpretation of BIC.

**Option E — None of the above:** ❌ Wrong
Option D is correct.

> **Key takeaway:** BIC has a Bayesian interpretation (approximates log marginal likelihood / Bayes factors); AIC is only valid when $p \ll n$; both require a likelihood and cannot be applied to arbitrary models like Random Forest.

---

### Question 7 — Cross-Validation

**Question:** Which of the following statements are true for running cross validation?

**Official Answer:** B, C, D

**Option A — Normalization of variables should be performed before running cross validation:** ❌ Wrong
This is a classic data leakage pitfall. If you compute the mean and standard deviation from the entire dataset (including future test folds) and normalize before splitting, test fold statistics influence the training normalization. Correct procedure: fit normalization parameters (mean, std) on the training fold only, then apply to the test fold. Normalization must happen *inside* each fold.

**Option B — A double-loop cross validation can assist in both model selection and model assessment:** ✓ Correct
Nested (double-loop) CV has an inner loop for hyperparameter selection (model selection) and an outer loop for estimating generalization performance (model assessment). Using the same loop for both gives an optimistic bias in the performance estimate. The outer loop provides an unbiased assessment after the inner loop has tuned hyperparameters.

**Option C — If a subset of observations is dependent, it can be a good idea to keep this subset in the same cross validation fold:** ✓ Correct
Standard CV assumes exchangeable (IID) observations. When observations are dependent (e.g., time series measurements, repeated measures on the same individual, spatial autocorrelation), mixing dependent observations across folds leaks information from training to test. Keeping dependent groups together in the same fold prevents this leakage and gives a valid estimate of generalization error.

**Option D — Cross validation provides reasonable estimates of the expected prediction error:** ✓ Correct
CV estimates the expected prediction error $\text{EPE} = E[L(y, \hat{f}(x))]$ by averaging prediction errors across held-out folds, where each observation acts as a test point exactly once. With enough folds and sufficient data, this provides a low-variance, nearly unbiased estimate of EPE.

**Option E — None of the above:** ❌ Wrong
Options B, C, and D are all correct.

> **Key takeaway:** Never normalize before CV (data leakage); use nested CV for combined selection and assessment; keep dependent observations in the same fold; CV is the standard tool for EPE estimation.

---

### Question 8 — Multiple Testing

**Question:** Which of the following statements are true when considering the multiple testing problem?

**Official Answer:** A

**Option A — Benjamini-Hochberg's algorithm controls the upper bound of the False Discovery Rate:** ✓ Correct
The BH procedure guarantees that $\text{FDR} = E[V/R] \leq \alpha$, where $V$ is the number of false discoveries and $R$ is the total number of discoveries (rejected null hypotheses). It ranks p-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(M)}$ and rejects all $H_{(i)}$ where $i \leq \max\{i : p_{(i)} \leq i\alpha/M\}$. This controls the upper bound of the FDR, making it less conservative than Bonferroni.

**Option B — The family-wise error rate is the probability of at most $\alpha/M$ false rejections, where M is the number of tests, and $\alpha$ the significance level of the individual tests:** ❌ Wrong
The family-wise error rate (FWER) is defined as $P(V \geq 1)$ — the probability of making **at least one** false rejection (Type I error) across all $M$ tests. It is not "at most $\alpha/M$ false rejections." Bonferroni controls FWER by setting individual test thresholds to $\alpha/M$, so that by the union bound $P(V \geq 1) \leq M \cdot (\alpha/M) = \alpha$.

**Option C — Bonferroni correction is useful when the number of tests is large but can be too conservative for small number of tests:** ❌ Wrong
This statement has the relationship backwards. Bonferroni divides the significance threshold by $M$: $\alpha_\text{adj} = \alpha/M$. For **small** $M$, the threshold $\alpha/M$ is only slightly below $\alpha$, making it barely conservative. For **large** $M$ (e.g., genome-wide association studies with millions of tests), $\alpha/M$ becomes extremely small, making Bonferroni very conservative and causing it to miss many true signals (low power). The statement is the reverse of reality.

**Option D — The plug-in estimate of the False Discovery Rate (as defined in Elements of Statistical Learning) is in general NOT a consistent estimate of the False Discovery Rate:** ❌ Wrong
The plug-in FDR estimate $\widehat{\text{FDR}} = M \cdot p_\text{threshold} / \#\{\text{rejections}\}$ is actually a conservative (overestimating) estimate of the true FDR under standard assumptions. It is used as the basis of the BH procedure precisely because its conservative nature provides valid FDR control. Saying it is "not a consistent estimate" would mean it systematically fails to converge to the true FDR — this is not what ESL states.

**Option E — None of the above:** ❌ Wrong
Option A is correct.

> **Key takeaway:** BH controls the FDR upper bound (less conservative than Bonferroni); FWER is the probability of at least one false rejection; Bonferroni becomes overly conservative with large $M$, not small $M$.

---

### Question 9 — Classification

**Question:** Which of the following statements are true when considering Classification?

**Official Answer:** C

**Option A — Gaussian Mixture Models uses the k class labels to fit k Gaussian distributions to the data, one to each class:** ❌ Wrong
GMM in its standard form is an *unsupervised* clustering method that fits Gaussian components to unlabeled data. While you can use GMM in a supervised manner (one Gaussian per class, which is essentially equivalent to LDA or QDA), GMM itself does not require or use class labels — it discovers structure without supervision. The statement misleadingly makes GMM sound like a supervised method.

**Option B — Classification and regression trees (CART) explicitly handle categorical variables by one-hot encoding them:** ❌ Wrong
CART handles categorical variables natively through the tree's split mechanism. For a categorical variable with $K$ levels, CART searches over all possible subsets $S \subset \{1, \ldots, K\}$ and splits on whether the value falls in $S$ or its complement. No one-hot encoding is performed. One-hot encoding is a preprocessing step needed by linear models, neural networks, and distance-based methods — not trees.

**Option C — Random Forest, Sparse Discriminant Analysis, and Regularized Logistic Regression are all suitable models for problems with more variables than observations:** ✓ Correct
All three handle $p \gg n$: Random Forest uses random feature subsampling ($m \ll p$ features per split); Sparse Discriminant Analysis regularizes the discriminant directions with sparsity penalties; Regularized Logistic Regression (with L1 or L2 penalty) performs variable selection/shrinkage enabling estimation when $p > n$. Unregularized logistic regression would fail for $p \geq n$.

**Option D — The Support Vector Machine gives nonlinear boundaries whenever the dual formulation is used for optimization:** ❌ Wrong
The dual formulation of SVM is a mathematical reformulation of the primal — it does not itself determine whether the boundary is linear or nonlinear. The boundary's linearity depends entirely on the kernel function $K(x, x')$. With a linear kernel $K(x, x') = x \cdot x'$, the dual SVM still produces a linear decision boundary. Nonlinear boundaries emerge only with nonlinear kernels (RBF, polynomial, etc.).

**Option E — None of the above:** ❌ Wrong
Option C is correct.

> **Key takeaway:** CART handles categorical variables natively via split subsets, not one-hot encoding; SVM linearity depends on kernel choice, not on using the dual formulation; RF, Sparse DA, and regularized logistic regression all handle $p \gg n$.

---

### Question 10 — Boosting

**Question:** Which of the following statements are true when considering Boosting?

**Official Answer:** B, C

**Option A — The forward stagewise additive modelling approach works by adaptively updating the weights for all trees in the boosting, as each new tree is added:** ❌ Wrong
In forward stagewise additive modelling, the model is built as $f_M(x) = \sum_{m=1}^{M} \beta_m b(x; \gamma_m)$, where at each step $m$, we find the new $(\beta_m, \gamma_m)$ that minimizes the loss while holding all previous $(\beta_1, \gamma_1), \ldots, (\beta_{m-1}, \gamma_{m-1})$ **fixed**. The weights of previously added trees are never updated. This "greedy" approach is what makes it "stagewise."

**Option B — The forward stagewise additive modelling approach works by consecutively adding trees and corresponding weights to the existing trees in the boosting:** ✓ Correct
This accurately describes the algorithm: at each step, a new tree $b(x; \gamma_m)$ and its weight $\beta_m$ are computed and appended to the current ensemble. The existing trees remain unchanged. The final model is the sum of all tree-weight pairs: $f(x) = \sum_{m=1}^{M} \beta_m b(x; \gamma_m)$.

**Option C — Using AdaBoost.M1 approximates using the exponential loss in a forward stagewise additive modelling:** ✓ Correct
This is a fundamental theoretical result (Friedman, Hastie, Tibshirani, 2000). AdaBoost.M1 re-weights observations by $w_i \propto e^{-y_i f(x_i)}$, and it can be shown that this is equivalent to minimizing the exponential loss $L(y, f) = e^{-yf(x)}$ under the forward stagewise additive model framework. The equivalence is not an approximation in a loose sense — it is an exact correspondence.

**Option D — The misclassification loss gives higher weight to mis-specified labels than the exponential loss:** ❌ Wrong
The exponential loss $L(y, f(x)) = e^{-y f(x)}$ grows exponentially as $yf(x)$ becomes increasingly negative (i.e., as the margin becomes more negative for misclassified points). The misclassification loss $\mathbf{1}[y \neq \hat{y}]$ assigns equal weight (1) to all misclassified points regardless of margin. So the exponential loss gives *higher* weight to confidently misclassified examples, not lower — the statement has the comparison reversed.

**Option E — None of the above:** ❌ Wrong
Options B and C are both correct.

> **Key takeaway:** Forward stagewise additive modelling adds trees sequentially without modifying previous ones; AdaBoost.M1 is equivalent to minimizing exponential loss (not misclassification loss) under this framework.

---

### Question 11 — Random Forest Variable Importance

**Question:** Which of the following statements are true for Random Forests and Variable importance?

**Official Answer:** A, C, D ⚠️ *Note: C contains a factual error — see below.*

**Option A — The Gini Variable Importance for variable j is an aggregation of the gini index at every split in the forest containing variable j:** ✓ Correct
For each tree in the forest, every split that uses variable $j$ contributes its Gini decrease $\Delta G_j = G_\text{parent} - w_L G_L - w_R G_R$ (where $w_L, w_R$ are proportional node sizes). The Gini Variable Importance for variable $j$ is the sum (or average) of all these contributions across all trees and all splits involving variable $j$. Higher total Gini decrease → more important variable.

**Option B — The OOB Variable Importance aggregates the gini index only of trees that do not contain the OOB samples:** ❌ Wrong
The OOB (Out-Of-Bag) Variable Importance works differently: for a given tree, the OOB samples are those observations NOT included in that tree's bootstrap sample. To compute importance of variable $j$, the OOB observations are fed through the tree twice — once normally, and once with variable $j$ randomly permuted. The increase in OOB error from permuting variable $j$ is the importance measure. The approach uses *permutation* on OOB samples, not Gini index aggregation.

**Option C — Proximity plots measure the closeness of variables and thereby gives an idea of the grouping of variables, which occur in a Random Forest, similar to what it does in Ridge regression:** ❌ Wrong (⚠️ **Error in official solution**)
Proximity plots in Random Forest measure the closeness of **observations** (data points), not variables. Two observations have high proximity if they frequently end up in the same terminal node across many trees in the forest — i.e., if the forest consistently classifies them together. This is used for visualizing structure among samples (e.g., identifying clusters in the data or outliers). The comparison to Ridge regression is also non-standard. Despite being marked correct in the official answer sheet, this statement is factually incorrect.

**Option D — It is a good idea to use deep (large) trees in the Random Forest since the bias of the Random Forest is the same as that of the individual trees:** ✓ Correct
The bias of a Random Forest equals the bias of a single tree in the ensemble (averaging unbiased estimators preserves unbiasedness; averaging biased estimators preserves the bias). Therefore, to achieve low bias in the forest, individual trees must have low bias — which requires deep (fully grown) trees. Variance is reduced by the averaging mechanism, so there is no cost to using deep trees in RF.

**Option E — None of the above:** ❌ Wrong
Options A and D are correct (and C is marked correct in the official solution despite containing an error).

> **Key takeaway:** Gini VI sums Gini decreases at each split involving variable $j$; OOB VI uses permutation of OOB samples (not Gini); proximity measures closeness between *observations* (not variables); RF bias = individual tree bias, so use deep trees.

---

### Question 12 — Neural Networks

**Question:** Which of the following statements about Neural Networks are true?

**Official Answer:** A, C

**Option A — The loss of a neural network autoencoder measures the difference between the input and the output of the network:** ✓ Correct
An autoencoder compresses the input through a bottleneck encoder to a low-dimensional representation $z = f_\text{enc}(x)$, then reconstructs the input via $\hat{x} = f_\text{dec}(z)$. The training objective is the reconstruction loss, typically $L = \|x - \hat{x}\|_2^2$ (mean squared error) or cross-entropy (for binary inputs). The "label" is the input itself — making it unsupervised.

**Option B — Neural networks rarely overfit to training data:** ❌ Wrong
Neural networks are among the most powerful and flexible function approximators and are notorious for overfitting, especially with many parameters and limited data. Without regularization (dropout, weight decay, early stopping, batch normalization), neural networks will memorize training data. This is why regularization techniques are central to deep learning practice.

**Option C — A feed forward neural network with 10 inputs and 2 hidden layers each with 2 units (nodes) and one output unit has $(10 \times 2 + 2) + (2 \times 2 + 2) + (2 \times 1 + 1) = 31$ parameters/weights that should be estimated:** ✓ Correct
Counting parameters layer by layer (weights + biases):
- Input (10) → Hidden layer 1 (2 units): $10 \times 2$ weights $+ 2$ biases $= 22$
- Hidden layer 1 (2) → Hidden layer 2 (2 units): $2 \times 2$ weights $+ 2$ biases $= 6$
- Hidden layer 2 (2) → Output (1 unit): $2 \times 1$ weight $+ 1$ bias $= 3$
- **Total: $22 + 6 + 3 = 31$** ✓

The formula in the option is $= (10 \times 2 + 2) + (2 \times 2 + 2) + (2 \times 1 + 1) = 22 + 6 + 3 = 31$.

**Option D — A feed forward neural network with 10 inputs and 2 hidden layers each with 2 units (nodes) and one output unit has $10 \times 2 \times 2 \times 1 = 40$ parameters/weights that should be estimated:** ❌ Wrong
This formula simply multiplies the layer sizes together, ignoring biases and the actual layer-to-layer connectivity structure. The correct approach is to count per-layer: (inputs_to_layer × units_in_layer) + units_in_layer (biases). The result is 31, not 40.

**Option E — None of the above:** ❌ Wrong
Options A and C are both correct.

> **Key takeaway:** Autoencoder loss = reconstruction error $\|x - \hat{x}\|^2$; neural networks heavily overfit without regularization; parameter count per layer = (in × out) + out (biases), giving 31 total for the 10→2→2→1 architecture.

---

### Question 13 — Least Angle Regression vs. Coordinate Descent for Lasso

**Question:** When comparing Least Angle Regression (LARS) with Coordinate Descent for Lasso, which of the following statements are true?

**Official Answer:** E (None of the above)

**Option A — Both algorithms can be considered path algorithms:** ❌ Wrong (as a strict claim for both)
LARS is explicitly a path algorithm — it traces the full Lasso regularization path from $\lambda = \infty$ to $\lambda = 0$ by following equiangular directions. Coordinate Descent, however, solves the Lasso at a *specific* $\lambda$ value by cycling through variables and applying soft-thresholding. It can be run over a grid of $\lambda$ values (warm-starting), but it is not inherently a path algorithm in the way LARS is. The claim that *both* are path algorithms in the same sense is not strictly true.

**Option B — Both algorithms update the parameter estimate for one variable at a time:** ❌ Wrong (as a strict claim for both)
Coordinate Descent updates exactly one coefficient $\beta_j$ at a time while holding all others fixed, cycling through variables. LARS, however, moves in an equiangular direction in the space of active variables — it can simultaneously move several coefficients along this direction. LARS does not strictly update one variable at a time; it adds one variable to the active set per step but then adjusts all active-set coefficients jointly.

**Option C — Both algorithms provide Lasso solutions, without adjusting the algorithms:** ❌ Wrong
LARS in its basic form provides solutions to the *Least Angle Regression* problem, which is closely related but not identical to the Lasso. The LARS-Lasso modification (a specific step that stops when a coefficient hits zero) is needed to make LARS trace the exact Lasso path. Without this modification, LARS does not provide Lasso solutions.

**Option D — Coordinate Descent makes more assumptions about data than Least Angle Regression:** ❌ Wrong
Both algorithms assume the same standard linear regression setup $y = X\beta + \epsilon$ with an L1 penalty. Coordinate Descent makes no additional assumptions about the data beyond what is required for the Lasso objective. Neither algorithm requires distributional assumptions on $\epsilon$ or special structure in $X$ beyond what Lasso itself requires.

**Option E — None of the above:** ✓ Correct
None of the statements A–D is accurately and fully correct as written, so E is the correct choice.

> **Key takeaway:** LARS is a true path algorithm tracing the full regularization path; Coordinate Descent solves Lasso at a fixed $\lambda$; LARS needs the Lasso modification to produce exact Lasso solutions; neither algorithm makes stronger assumptions than the other.

---

### Question 14 — Clustering

**Question:** Which of the following statements about clustering are true?

**Official Answer:** B

**Option A — The Manhattan distance is used to measure the absolute distance between two points for categorical variables:** ❌ Wrong
Manhattan distance $d(x, y) = \sum_{j=1}^p |x_j - y_j|$ measures absolute distances between **continuous** (numerical) variables. For categorical variables, Hamming distance (counts mismatches) or Gower distance (handles mixed types) is appropriate. Manhattan distance has no natural meaning for unordered categorical levels (e.g., "red," "blue," "green").

**Option B — We can use different distance metrics to perform hierarchical clustering:** ✓ Correct
Hierarchical clustering builds a dendrogram by iteratively merging the closest clusters. The inter-point distance can be Euclidean, Manhattan, correlation-based, cosine similarity, or any other valid distance metric. The linkage function (single, complete, average, Ward) then determines how cluster distances are aggregated. This flexibility in distance metrics is a strength of hierarchical clustering over K-means.

**Option C — We can use AIC or BIC to find the optimal number of clusters in k-means clustering:** ❌ Wrong
AIC and BIC are defined as functions of the log-likelihood $\log\hat{L}$. K-means minimizes the within-cluster sum of squares and has no natural probability model or likelihood. Therefore AIC/BIC cannot be directly applied to K-means. Suitable alternatives for determining the optimal $K$ in K-means include the elbow method (plot inertia vs $K$), silhouette scores, or the gap statistic. (BIC can be used for GMM, which has an explicit likelihood.)

**Option D — The linkage function decides if we perform agglomerative or divisive hierarchical clustering:** ❌ Wrong
The linkage function (single linkage, complete linkage, average linkage, Ward's method) determines how the distance between two clusters is computed when merging them. The choice between agglomerative (bottom-up: start with each point as a cluster, merge) and divisive (top-down: start with all points in one cluster, split) is a separate algorithmic choice that is independent of the linkage function.

**Option E — None of the above:** ❌ Wrong
Option B is correct.

> **Key takeaway:** Manhattan distance applies to continuous data, not categorical; hierarchical clustering supports arbitrary distance metrics; AIC/BIC require a likelihood and cannot be applied to K-means; linkage vs. agglomerative/divisive are independent choices.

---

### Question 15 — K-means Clustering Plots

**Question:** Figure 1 illustrates 6 two-dimensional simulated datasets and their K-means clustering solution with two clusters. The variance explained (VE) for each dataset is: a=0.804, b=0.581, c=0.752, d=0.559, e=0.649, f=0.935. Which of the following statements are true?

**Official Answer:** B

**Option A — K-means clustering with two clusters describes least of the variance in dataset e than in any of the other datasets:** ❌ Wrong
Dataset b has VE = 0.581 and dataset d has VE = 0.559. Dataset e has VE = 0.649. Therefore dataset d has the lowest variance explained, not e. This statement is factually contradicted by the VE values shown in the figure labels.

**Option B — K-means clustering with two components is an appropriate choice to describe datasets a and c:** ✓ Correct
Dataset a (VE = 0.804) and dataset c (VE = 0.752) both show relatively high variance explained by a 2-cluster K-means solution. Visually, these datasets appear to have two separable globular clusters that K-means can capture well. The high VE values support the conclusion that a 2-cluster solution is appropriate for these datasets.

**Option C — Archetypical Analysis with three components is an appropriate choice to describe datasets d, e, and f:** ❌ Wrong
Archetypical Analysis represents data as convex combinations of extreme points (archetypes on the convex hull). Three archetypes would be appropriate if the data forms a triangular shape (three extremes). Dataset d appears as a wedge or right-angle shape (two extremes); dataset e appears elongated (two extremes); dataset f appears as a clear two-cluster structure. Three archetypes would be more than necessary for these geometries.

**Option D — Non-Negative Matrix Factorization (NMF) is an appropriate choice to describe the datasets d and f, without any pre-processing:** ❌ Wrong
NMF requires all entries of the data matrix $X$ to be non-negative (since $W, H \geq 0$ implies $X \approx WH \geq 0$). The datasets shown have data points spread across all four quadrants (with negative values on both axes). Without pre-processing to shift the data into the non-negative orthant, NMF cannot be applied directly.

**Option E — None of the above:** ❌ Wrong
Option B is correct.

> **Key takeaway:** The variance explained (VE) values from the figure directly answer option A — dataset d has the lowest VE, not e; K-means is appropriate where VE is high and clusters are globular; NMF requires non-negative data.

---

### Question 16 — Expected Prediction Error, Bias, and Variance

**Question:** For the expected prediction error (EPE), bias and variance of a given model, which of the following statements are true?

**Official Answer:** C

**Option A — The variance can be estimated as the MSE (Mean squared error) of an independent test set:** ❌ Wrong
The MSE of an independent test set estimates the EPE (expected prediction error), not the model variance. EPE decomposes as $\text{EPE} = \text{Bias}^2 + \text{Variance} + \sigma^2$. The test MSE captures all three components together. To isolate variance alone, you would need to estimate $\hat{f}(x)$ over many independent training sets and measure the spread.

**Option B — The bias is the difference between the estimate on the training data set and the true value (for the same data):** ❌ Wrong
Bias is defined as the expected difference between the estimator and the true function, averaged over all possible training datasets: $\text{Bias}(\hat{f}(x)) = E_\mathcal{T}[\hat{f}(x)] - f(x)$. The expectation is over the randomness in the training set $\mathcal{T}$. Looking at one specific training set's residuals is not bias — that is the training error, which includes both bias and variance terms.

**Option C — The expected prediction error is an expectation over data samples, whereas the generalization error is conditioned on a fixed training set:** ✓ Correct
EPE averages over both new test points AND all possible training datasets: $\text{EPE} = E_{x_0, \mathcal{T}}[L(y_0, \hat{f}(x_0))]$. The generalization error (also called conditional test error or $\text{Err}_\mathcal{T}$) is conditioned on the observed training set $\mathcal{T}$: $\text{Err}_\mathcal{T} = E_{x_0}[L(y_0, \hat{f}(x_0)) \mid \mathcal{T}]$. The EPE is $E_\mathcal{T}[\text{Err}_\mathcal{T}]$. This formal distinction is from ESL Chapter 7.

**Option D — The bias is the difference between the estimate on the test data set and the true value (for the same data):** ❌ Wrong
Same issue as Option B, but using test data instead. Evaluating $\hat{f}(x) - f(x)$ on a single test point gives the prediction error for that point, not the bias. Bias requires averaging the estimator over all possible training datasets: $E_\mathcal{T}[\hat{f}(x)] - f(x)$.

**Option E — None of the above:** ❌ Wrong
Option C is correct.

> **Key takeaway:** EPE averages over both training data variability and new test points; generalization error conditions on a fixed training set; bias = $E_\mathcal{T}[\hat{f}(x)] - f(x)$, not a single residual; test MSE estimates EPE, not variance alone.

---

### Question 17 — Multiway Models (CONCORDIA / PARAFAC / Tucker)

**Question:** Concerning multiway models, which of the following statements are true?

**Official Answer:** A, D

**Option A — When the Core Consistency Diagnostic (CONCORDIA) is close to 100, it means that the PARAFAC model has a suitable number of components, because the core tensor is close to diagonal:** ✓ Correct
CORCONDIA measures how close the core tensor $\mathcal{G}$ (extracted by fitting a Tucker model with the same number of components as the PARAFAC model) is to the super-identity tensor $\mathcal{T}$ (which has ones on the super-diagonal and zeros elsewhere). $\text{CORCONDIA} = 100\left(1 - \frac{\sum_{r,s,t}(g_{rst} - t_{rst})^2}{\sum_{r,s,t} t_{rst}^2}\right)$. Close to 100 means the core is nearly super-diagonal, which is exactly the constraint in PARAFAC — indicating the PARAFAC model fits with the chosen number of components.

**Option B — When the Core Consistency Diagnostic (CONCORDIA) is close to 100, it means that the Tucker model has a suitable number of components, because the core tensor is close to diagonal:** ❌ Wrong
CORCONDIA is defined specifically for **PARAFAC** model assessment, not Tucker. The Tucker core tensor is not constrained to be super-diagonal — Tucker's core is a full $R_1 \times R_2 \times R_3$ tensor. CORCONDIA compares the extracted core (from a Tucker re-decomposition of PARAFAC components) to the super-identity structure, which is a PARAFAC-specific criterion.

**Option C — The core tensor in Tucker is a super-diagonal:** ❌ Wrong
The Tucker core tensor $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$ is a **full** tensor — it can have non-zero entries at any position. It is the PARAFAC model that has the special constraint of a super-diagonal core (all non-super-diagonal entries are zero, i.e., component interactions only occur among same-numbered components across modes). Tucker's core being full is what makes Tucker more flexible and more general than PARAFAC.

**Option D — The dimensionality of the core tensor in Tucker defines the ranks in the modes of the decomposition:** ✓ Correct
In a 3-mode Tucker decomposition $\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$, the core tensor $\mathcal{G}$ has shape $R_1 \times R_2 \times R_3$, where $R_k$ is the number of components (rank) in mode $k$. Different modes can have different ranks, and the core tensor's dimensions directly encode these ranks. This contrasts with PARAFAC, which has a single rank $R$ across all modes.

**Option E — None of the above:** ❌ Wrong
Options A and D are both correct.

> **Key takeaway:** CORCONDIA tests PARAFAC (not Tucker) by checking if the core is super-diagonal; the Tucker core is full (not super-diagonal); Tucker core dimensions $R_1 \times R_2 \times R_3$ define the per-mode ranks.

---

### Question 18 — DNA Microarray Cancer Classification

**Question:** We have data from 1000 features from DNA Expression Microarrays for 820 individuals, all with cancer. We know the cancer type present in each of the 820 individuals and want to investigate if similar DNA Expressions share the same cancer type. Given our aim, which of the following method(s) can help us build a model and answer our question?

**Official Answer:** B

**Option A — A random forest classification:** ❌ Wrong (per official answer)
RF classification could technically be used to predict cancer type from expression features and assess accuracy, but the question asks to "investigate if similar DNA expressions share the same cancer type" — which is more of an exploratory/visualization question than a predictive one. The official answer emphasizes the investigative/exploratory framing that hierarchical clustering better addresses.

**Option B — A hierarchical clustering:** ✓ Correct
Hierarchical clustering of the 820 individuals based on their 1000-feature expression profiles produces a dendrogram. By overlaying cancer type labels (using colour-coding), you can visually assess whether individuals with similar expression profiles (close in the dendrogram) tend to share the same cancer type. This is the standard bioinformatics approach for "investigating" whether molecular profiles co-cluster with clinical labels without making explicit predictions.

**Option C — A random forest regression:** ❌ Wrong
Cancer type is a categorical variable, not a continuous one. Random forest regression predicts a continuous outcome — it is the wrong model type for a classification problem. Random forest classification (Option A) would be appropriate if a predictive approach were taken.

**Option D — A logistic regression without regularization:** ❌ Wrong
With 1000 features ($p = 1000$) and 820 observations ($n = 820$), we have $p > n$. Logistic regression without regularization requires inverting $X^TX$ or equivalent, which is singular when $p > n$. The model is unidentifiable and the MLE does not exist uniquely. Regularized logistic regression (L1/L2) would work, but not the unregularized version.

**Option E — None of the above:** ❌ Wrong
Option B is correct.

> **Key takeaway:** Hierarchical clustering + label overlay is the standard exploratory approach to "investigating whether similar features share the same class"; unregularized logistic regression fails for $p > n$; RF classification could work but is not the primary answer for this exploratory framing.

---

### Question 19 — Bagging: Methods That Benefit Most

**Question:** Which of the following methods are suitable in terms of achieving a significant reduction in EPE when using the method as the individual models in a bagging procedure?

**Official Answer:** C

**Option A — Classification and Regression Tree stump:** ❌ Wrong
Tree stumps (depth-1 trees) are shallow, high-bias, low-variance models. Bagging reduces variance by averaging, but it does not reduce bias. If the individual model has high bias (stumps), the bagged ensemble will also have high bias. The EPE reduction from bagging stumps is minimal — the bias component dominates and averaging doesn't help it. Stumps are the preferred individual model for **boosting** (which reduces bias sequentially), not bagging.

**Option B — K-Nearest Neighbors with a large number of neighbors (K):** ❌ Wrong
Large $K$ in KNN produces a smooth, low-variance, low-bias (but stable) estimator — essentially a local average over many neighbors. With low variance already, bagging provides little additional variance reduction. The model is already well-regularized, so there is little gain from averaging.

**Option C — K-Nearest Neighbors with a small number of neighbors (K):** ✓ Correct
Small $K$ in KNN (e.g., $K=1$) produces a highly variable, low-bias estimator — it perfectly interpolates training data but is very sensitive to which training points are included. Bagging, which reduces variance by averaging many bootstrap-trained estimators, works best on exactly this type of model. Averaging many high-variance, low-bias models produces an ensemble with greatly reduced variance and preserved low bias, significantly reducing EPE.

**Option D — The mean:** ❌ Wrong
The mean (of the response variable) is a completely non-adaptive model with zero variance across different bootstrap samples (all bootstrap samples have approximately the same mean) and high bias (it ignores all feature information). Bagging the mean simply averages identical models — no variance reduction, no EPE improvement.

**Option E — None of the above:** ❌ Wrong
Option C is correct.

> **Key takeaway:** Bagging reduces variance without changing bias. Models must be high-variance and low-bias to benefit: KNN with small $K$ fits perfectly; stumps are high-bias (bagging won't help); KNN with large $K$ is already low-variance (bagging won't help much).

---

### Question 20 — Kernel Trick

**Question:** Which of the following methods can be formulated such that the kernel trick can be applied?

**Official Answer:** A, C

**Option A — Support Vector Machines:** ✓ Correct
The SVM dual formulation involves the optimization $\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$, where all computations involve only inner products $x_i^T x_j$. By replacing inner products with a kernel function $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$, the SVM implicitly operates in a high-dimensional feature space $\phi(\cdot)$ without ever computing $\phi$ explicitly. This is the canonical application of the kernel trick.

**Option B — Boosting:** ❌ Wrong
Boosting (AdaBoost, gradient boosting) is a stagewise additive model that builds an ensemble by fitting weak learners to residuals or re-weighted data. It does not express computations purely in terms of inner products, so there is no natural formulation that admits the kernel trick.

**Option C — Principal Component Analysis:** ✓ Correct
Kernel PCA replaces the covariance matrix $X^TX / n$ (standard PCA) with the kernel matrix $K_{ij} = K(x_i, x_j)$. Eigendecomposing the kernel matrix gives principal components in the implicitly defined high-dimensional feature space $\phi(\cdot)$. This is a well-established method (Schölkopf et al., 1998) and a standard example of the kernel trick applied to an unsupervised method.

**Option D — Random Forest:** ❌ Wrong
Random Forest operates by building trees using recursive binary splits on individual feature values. The algorithm is inherently feature-space-based and does not express its computations in terms of inner products $x_i \cdot x_j$. There is no natural reformulation of RF that would allow replacing inner products with a kernel function.

**Option E — None of the above:** ❌ Wrong
Options A and C are both correct.

> **Key takeaway:** The kernel trick requires the algorithm to be expressible purely in terms of inner products $\langle x_i, x_j \rangle$; SVM (dual formulation) and PCA (Gram matrix eigendecomposition) both admit this; Boosting and Random Forest do not.

---

## Open Questions

---

### Question 21 — ICA: Uniqueness, Independence, and Favored Distributions

**Question:** Demonstrate whether Independent Component Analysis (ICA) is unique or not and describe what it means to have independent components. What kind of distributions are favored for the components of ICA?

---

#### Part 1: Is ICA Unique?

ICA is **not fully unique** — it has three fundamental indeterminacies:

**1. Permutation indeterminacy.**
The ICA model is $X = AS$, where $A$ is the mixing matrix and $S = (s_1, \ldots, s_K)^T$ is the vector of independent source signals. If we permute the rows of $S$ and permute the columns of $A$ correspondingly (via a permutation matrix $P$), we get an equivalent model:

$$X = AS = (AP^{-1})(PS) = A'S'$$

The new $A'$ and $S'$ satisfy the same model equation, so the order of the components cannot be determined from data alone.

**2. Sign/scaling indeterminacy.**
If we multiply source $s_k$ by a scalar $c_k \neq 0$ and divide the corresponding column of $A$ by $c_k$:

$$X = AS = A \cdot \text{diag}(c_1,\ldots,c_K) \cdot \text{diag}(c_1,\ldots,c_K)^{-1} S$$

the model is unchanged. Therefore, neither the amplitudes nor the signs of the sources are identifiable.

**3. Gaussian sources are unidentifiable.**
If any source $s_k$ is Gaussian, ICA cannot recover it uniquely. This is the most critical constraint. For Gaussian sources, the joint distribution $p(s) = \prod_k p_k(s_k)$ is rotationally symmetric — any rotation in the Gaussian subspace produces an equally valid solution. This is why ICA explicitly requires **non-Gaussian** sources.

**Summary:** ICA is unique **up to permutation and scaling of components**, provided at most one source is Gaussian. It is **not** identifiable when two or more sources are Gaussian.

---

#### Part 2: What Does It Mean to Have Independent Components?

Statistical independence is strictly stronger than uncorrelatedness. Two random variables $s_1, s_2$ are **independent** if and only if their joint density factors into the product of marginal densities:

$$p(s_1, s_2) = p_1(s_1) \cdot p_2(s_2)$$

More generally, components $s_1, \ldots, s_K$ are mutually independent if:

$$p(s_1, \ldots, s_K) = \prod_{k=1}^{K} p_k(s_k)$$

**Uncorrelatedness** only requires zero covariance: $\text{Cov}(s_j, s_k) = 0$ for $j \neq k$. PCA finds uncorrelated components (second-order independence) but does not enforce higher-order independence. ICA enforces the full factorization of the joint distribution — zero covariance plus zero higher-order cross-moments (skewness cross-terms, kurtosis cross-terms, etc.).

In practice, ICA measures independence through **contrast functions** that quantify departure from Gaussianity, such as:
- **Kurtosis:** $\text{kurt}(s) = E[s^4] - 3(E[s^2])^2$; zero for Gaussians, non-zero for non-Gaussian sources.
- **Negentropy:** $J(s) = H(s_\text{Gauss}) - H(s) \geq 0$; the Gaussian is the maximum entropy distribution, so non-Gaussian sources have lower entropy.
- **Mutual information:** $I(s_1, \ldots, s_K) = \sum_k H(s_k) - H(s_1, \ldots, s_K) \geq 0$; ICA minimizes mutual information to achieve independence.

---

#### Part 3: What Distributions Are Favored for ICA?

ICA requires **non-Gaussian** source distributions. The two preferred families are:

**1. Super-Gaussian (leptokurtic, heavy-tailed) distributions.**
These have **positive kurtosis** ($\text{kurt}(s) > 0$), meaning more probability mass near zero and in the tails than a Gaussian. Examples: Laplace distribution, student's $t$, sparse signals, speech and audio signals. The Laplace distribution $p(s) \propto e^{-|s|}$ is the canonical example.

**2. Sub-Gaussian (platykurtic, flat-tailed) distributions.**
These have **negative kurtosis** ($\text{kurt}(s) < 0$), meaning probability mass is more evenly spread than a Gaussian. Examples: uniform distribution, bimodal distributions.

The key contrast is that a Gaussian distribution has **zero kurtosis** and is the maximum-entropy distribution for a given variance. The Central Limit Theorem (CLT) states that mixtures of independent random variables converge to Gaussian. ICA exploits this: if we observe $X = AS$ (a mixture), the observed signals $X$ are more Gaussian than the original sources $S$. ICA finds the unmixing matrix $W = A^{-1}$ by searching for directions that are maximally **non-Gaussian** — recovering the original sources.

FastICA (the standard ICA algorithm) uses negentropy approximations or kurtosis as the contrast function and maximizes non-Gaussianity using a fixed-point iteration.

> **Summary:** ICA is unique only up to permutation and scaling, and requires non-Gaussian sources (Gaussian sources are unidentifiable). Independence means the joint density factors into marginals — strictly stronger than uncorrelatedness. ICA favors super-Gaussian or sub-Gaussian sources; it cannot identify Gaussian sources.

---

### Question 22 — Cross-Validation Design for Wearable Biosignal Activity Prediction

**Question:** You are given a dataset consisting of time series of blood volume pressure (BVP), temperature, and heart rate (HR) measured by a wearable device. These biosignals have been measured for 16 persons under three conditions: during 10 minutes rest, during 10 minutes running exercise, and during 10 minutes social media usage, and at 4 different time points (one for each season of the year). This results in a total of 192 ($4 \times 16 \times 3$) observations. We are interested in seeing if we can predict the activity of an individual based on the observed biosignals. Consider using a set of extracted biosignal features. Explain which methods you would use to answer whether we can predict the activity or not and to what extent of (a) one of the individuals in the experiment, and (b) a new individual not previously included in the experiment.

---

#### Dataset Structure

The data has a clear hierarchical/repeated-measures structure:
- 16 individuals (persons)
- 3 activity conditions (rest, running, social media)
- 4 time points (seasons: spring, summer, autumn, winter)
- For each observation: extracted features from BVP, temperature, HR time series (e.g., mean, variance, spectral features, HRV metrics)
- **Response variable:** Activity type (3-class classification: rest / running / social media)

This is a **supervised classification problem** with structured/dependent data.

---

#### Feature Extraction

Before applying classification models, extract meaningful features from the raw biosignal time series for each of the 192 observations. Relevant features include:
- **BVP:** mean amplitude, pulse rate, HRV (RMSSD, SDNN), frequency-domain features (LF/HF ratio)
- **Temperature:** mean, range, rate of change
- **HR:** mean HR, HR variability metrics, exercise-induced elevation
- Optionally: PCA/dimensionality reduction on extracted features if many are correlated

---

#### Classification Methods

This is a **3-class supervised classification problem** (rest / running / social media). Choose methods appropriate for the small-$n$, 3-class setting:

| Method | Why suitable | Caveat |
|--------|-------------|--------|
| **Regularized Logistic Regression** (L1 or L2) | Handles correlated features; probability outputs; L1 performs implicit feature selection | $\lambda$ must be tuned via nested CV |
| **LDA** | Works well with small $n$ (LOSO folds have only 9 training obs); fast; interpretable | Assumes equal covariance across classes |
| **Random Forest** | Non-linear, no distributional assumption; feature importance; robust | More data-hungry; less interpretable |

**Why regularization is essential:** In the personalized setting (LOSO), training folds contain only 9 observations. Unregularized estimators are high-variance and may not converge. Regularization is not optional here.

**Hyperparameter tuning:** The regularization strength $\lambda$ must be selected inside the CV loop via a nested inner loop — never on the full 192 observations. Tuning on the full dataset lets the test subject's data influence $\lambda$, producing an optimistically biased EPE estimate (subtle data leakage through the hyperparameter).

---

#### Part (a): Predicting Activity for One Individual Already in the Experiment

**Goal:** Estimate how well we can predict the 3 activity classes for a specific individual using their own data.

**Available data:** For one individual: $3 \text{ conditions} \times 4 \text{ seasons} = 12$ observations.

**Method — Leave-one-season-out cross-validation (within-subject):**
- Partition the 12 observations into 4 folds, one per season (spring, summer, autumn, winter).
- Train on 3 seasons (9 observations), test on the held-out season (3 observations — one per activity).
- Repeat for all 4 seasons; report mean accuracy across folds.

This respects the temporal structure of the data (seasons represent time progression) and provides a personalized model accuracy estimate. It answers: "Given this individual's past data from 3 seasons, can we predict their activity in a new season?"

**Why not standard random CV?** Randomly assigning the 12 observations to folds would mix temporal information — training on future-season data to predict past-season data. The leave-season-out design mimics real deployment (always predicting for new time points).

**Expected result:** Likely high accuracy for running vs. rest/social media (HR and BVP change dramatically during exercise). Rest vs. social media may be harder to distinguish.

---

#### Part (b): Predicting Activity for a New Individual Not in the Experiment

**Goal:** Estimate how well the model generalizes to **completely new individuals** (not seen during training).

**Method — Leave-one-individual-out cross-validation (LOIO-CV):**
- In each fold: hold out all 12 observations from one individual ($3 \text{ conditions} \times 4 \text{ seasons}$).
- Train the classification model on the remaining $15 \times 12 = 180$ observations.
- Evaluate on the held-out individual's 12 observations.
- Repeat for all 16 individuals; report mean classification accuracy and per-class accuracy.

This directly estimates generalization to a new, unseen individual — the model is never trained on any data from the test individual. This is the correct design for the "new individual" question.

**Why not standard K-fold CV?** Standard K-fold (or random assignment) would mix observations from the same individual across training and test folds. Since individuals differ in baseline physiology (e.g., resting HR, temperature regulation), a model trained on other seasons of the same individual would have already "seen" that individual's signal characteristics. This inflates accuracy relative to true generalization.

**Comparison of (a) and (b):**

| | Part (a): Personalized | Part (b): Generalized |
|---|---|---|
| Training data | 9 observations (3 seasons, same person) | 180 observations (15 people, all seasons) |
| Test target | New season, same individual | New individual |
| CV method | Leave-one-season-out (within-person) | Leave-one-individual-out |
| Likely accuracy | Higher (personalized to physiology) | Lower (must generalize across individuals) |
| Practical use | Personalized health monitoring | Clinical deployment to new patients |

**Additional consideration:** If we want to assess whether predictions are statistically better than chance, report accuracy alongside a baseline (uniform chance = 1/3 for 3 classes) and use a permutation test or binomial test to assess statistical significance.

> **Key takeaway:** The critical design principle is to match the CV structure to the prediction target: predicting for the same individual requires within-person CV (leave-season-out); predicting for a new individual requires leave-one-individual-out CV. Mixing individuals across folds when the goal is generalization constitutes data leakage.
