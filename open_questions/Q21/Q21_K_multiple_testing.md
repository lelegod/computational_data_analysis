# Q21-K — Multiple Testing: Bonferroni vs BH (FDR)
> Week 3. Unique topic; could be asked to explain both procedures and when to use each.

---

## The Problem

When testing $m$ hypotheses simultaneously, the probability of at least one false positive (Type I error) grows with $m$ even if each individual test is at level $\alpha$.

Under independence with $m$ tests each at level $\alpha$:
$$P(\text{at least one false positive}) = 1-(1-\alpha)^m \to 1 \text{ as } m\to\infty$$

For $m=100$ tests at $\alpha=0.05$: $P(\text{false positive}) \approx 99.4\%$ — almost certain to find a spurious result.

**Two different error rates to control**:
- **FWER** (Family-Wise Error Rate): $P(\text{at least one false positive}) \leq \alpha$. Very strict.
- **FDR** (False Discovery Rate): $E[\text{FP}/\max(R,1)] \leq \alpha$ where $R$ = total rejections. Controls the expected *proportion* of false positives among all rejections. Less strict but more powerful.

---

## Bonferroni Correction (Controls FWER)

**Procedure**: reject hypothesis $H_i$ if $p_i \leq \alpha/m$.

**Why it works**: by union bound:
$$P(\text{at least one false positive}) \leq \sum_{i=1}^m P(p_i \leq \alpha/m | H_i \text{ true}) \leq m \cdot \frac{\alpha}{m} = \alpha$$

**Properties**:
- Simple: no sorting, just compare each $p_i$ to $\alpha/m$
- Conservative: controls FWER even under arbitrary dependence between tests
- **Very low power** when $m$ is large: threshold $\alpha/m = 0.05/10000 = 5\times10^{-6}$ for genome-wide study → most true signals missed

**When to use**: when a single false positive is catastrophic (e.g., drug approval where false safety claim causes harm). Small $m$.

---

## Benjamini-Hochberg Procedure (Controls FDR)

**Procedure**:
1. Sort the $m$ p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. For each rank $k = m, m-1, \ldots, 1$: find the largest $k$ such that:
$$p_{(k)} \leq \frac{k}{m}\alpha$$
3. Reject all hypotheses $H_{(1)}, \ldots, H_{(k^*)}$ (all hypotheses with rank $\leq k^*$)

**Equivalent formulation**: compute adjusted p-values:
$$\tilde{p}_{(k)} = \min_{j\geq k}\left(\frac{m}{j}p_{(j)}\right)$$
Reject if $\tilde{p}_{(k)} \leq \alpha$.

**Why it works**: BH controls $\text{FDR} \leq \frac{m_0}{m}\alpha \leq \alpha$ where $m_0$ = number of true nulls. Proved by Benjamini & Hochberg (1995) under independence; extends to positive dependence (PRDS).

**Properties**:
- **More powerful than Bonferroni** when many true signals exist
- Threshold adapts to where p-values concentrate: if many p-values are small, the threshold is higher
- Under complete null ($m_0=m$, no true signals): FDR = $\alpha$ (tightest possible)
- Under partial null: FDR $\leq \frac{m_0}{m}\alpha < \alpha$ (conservative)

**When to use**: large-scale testing where some false positives are acceptable (genomics: 5% of discovered genes being false positives is fine if the list is long enough). Large $m$.

---

## BH Procedure — Worked Example

$m=5$ tests, $\alpha=0.05$. Sorted p-values: $0.001, 0.008, 0.039, 0.041, 0.200$.

| Rank $k$ | $p_{(k)}$ | Threshold $k\alpha/m = k\cdot0.01$ | Reject? |
|----------|-----------|--------------------------------------|---------|
| 5 | 0.200 | 0.050 | No |
| 4 | 0.041 | 0.040 | No |
| 3 | 0.039 | 0.030 | No |
| 2 | 0.008 | 0.020 | **Yes** |
| 1 | 0.001 | 0.010 | Yes (all below $k^*=2$) |

$k^* = 2$ → reject $H_{(1)}$ and $H_{(2)}$ (p-values 0.001 and 0.008).

Bonferroni threshold: $0.05/5 = 0.010$ → rejects only $H_{(1)}$ (p-value 0.001). BH is more powerful here.

---

## Comparison Table

| Property | Bonferroni | Benjamini-Hochberg |
|----------|------------|-------------------|
| Controls | FWER | FDR |
| Error definition | $P(\geq 1$ false positive$)$ | $E[$FP/Rejections$]$ |
| Threshold | $\alpha/m$ (fixed) | $k\alpha/m$ (adaptive) |
| Power | Low (very conservative) | High (more discoveries) |
| Requires sorting? | No | Yes |
| Dependence | Works under any dependence | Requires independence or PRDS |
| Use case | Critical decisions, small $m$ | Exploratory, large $m$ (genomics) |

---

## Key Intuition

**Bonferroni**: "I want to be sure I haven't made ANY mistake."

**BH**: "I'm OK with 5% of my discoveries being wrong — just control that proportion."

For $m=20000$ gene tests: Bonferroni threshold $= 2.5\times10^{-6}$ (rejects almost nothing true). BH at $\alpha=0.05$: if 1000 genes are truly differentially expressed, BH might find 950 of them while keeping the false discovery rate at 5%.

---

## Additional Possible Exam Questions

**Q: What is the difference between a Type I and Type II error in the context of multiple testing?**
Type I error: false positive (reject $H_0$ when it is true — declare a gene differentially expressed when it is not). Type II error: false negative (fail to reject $H_0$ when it is false — miss a truly differentially expressed gene). FWER controls Type I. Power (1 − Type II rate) is what BH preserves better than Bonferroni.

**Q: Why is BH more powerful than Bonferroni?**
Bonferroni uses a fixed threshold $\alpha/m$ regardless of the data. BH adapts: it finds the largest group of small p-values where the fraction of expected false positives is below $\alpha$. When many true signals exist (many small p-values), BH raises its effective threshold, discovering more true positives. When no signals exist, BH is as conservative as needed.

**Q: Does BH require independent tests?**
BH is proved under independence and under positive regression dependence (PRDS). It can fail to control FDR under arbitrary negative dependence. A conservative fix: Benjamini-Yekutieli (BY) uses threshold $k\alpha/(m\cdot\sum_{j=1}^m1/j)$ — valid under any dependence but very conservative.

**Q: What is the q-value?**
The q-value is the BH-adjusted p-value: $q_{(k)} = \min_{j\geq k}(m/j)\cdot p_{(j)}$. It is the minimum FDR level at which hypothesis $k$ would be rejected. Analogous to the p-value for FWER: "the smallest FDR level at which this discovery would be included."

**Q: In genomics, you test 20,000 genes for differential expression. 1,000 are truly different. With $\alpha=0.05$: how many false positives does Bonferroni allow vs BH?**
Bonferroni: controls FWER at 0.05 → at most $0.05\times1$ false positive expected (essentially zero). Very few true positives discovered. BH: controls FDR at 0.05 → if it discovers $R$ genes, at most $0.05R$ are false. If it discovers 950 truly differential genes + ~50 false positives = 1000 total, FDR = 50/1000 = 5%. Much more power.
