# Week 11 — Group Discussion Questions

## Q1: Think-Pair-Share — The Mechanics of Factorization

**Question (slide 4):** Suppose we have a simple $2 \times 2$ data matrix $\mathbf{X}$ that we want to perfectly decompose into a rank-1 basis matrix $\mathbf{W}$ and an activation matrix $\mathbf{H}$:

$$\mathbf{X} = \mathbf{WH} \implies \begin{bmatrix} 4 & 6 \\ 6 & 9 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} \begin{bmatrix} h_1 & h_2 \end{bmatrix}$$

What are the values of $h_1$ and $h_2$ required to reconstruct $\mathbf{X}$?

**Answer:**

Performing the matrix multiplication:

$$\begin{bmatrix} 2 \\ 3 \end{bmatrix} \begin{bmatrix} h_1 & h_2 \end{bmatrix} = \begin{bmatrix} 2h_1 & 2h_2 \\ 3h_1 & 3h_2 \end{bmatrix}$$

Matching to $\mathbf{X}$:

$$\begin{bmatrix} 2h_1 & 2h_2 \\ 3h_1 & 3h_2 \end{bmatrix} = \begin{bmatrix} 4 & 6 \\ 6 & 9 \end{bmatrix}$$

From the top-left entry: $2h_1 = 4 \implies h_1 = 2$

From the top-right entry: $2h_2 = 6 \implies h_2 = 3$

**Solution:** $h_1 = 2$, $h_2 = 3$, so $\mathbf{H} = \begin{bmatrix} 2 & 3 \end{bmatrix}$.

**Verification:** $3h_1 = 6$ ✓ and $3h_2 = 9$ ✓

**Takeaway:** $\mathbf{W}$ is the basis pattern (a "part" of the data — here the column shape $[2, 3]^T$), and $\mathbf{H}$ contains the activations that scale how much of that pattern appears in each column of $\mathbf{X}$. Column 1 uses the pattern with intensity 2, column 2 with intensity 3.

---

## Q2: Think-Pair-Share — A Full NMF Iteration

**Question (slide 16):** Perform one entire alternating update cycle (update $\mathbf{H}$, then update $\mathbf{W}$). We have a 1D column vector $\mathbf{X}$ and start with terrible guesses: all 1s.

$$\mathbf{X} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}, \quad \mathbf{W}_{old} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad \mathbf{H}_{old} = \begin{bmatrix} 1 \end{bmatrix}$$

Using the multiplicative update rules:

- **Step 1:** $\mathbf{H}_{new} = \mathbf{H}_{old} \cdot \dfrac{\mathbf{W}_{old}^T \mathbf{X}}{\mathbf{W}_{old}^T \mathbf{W}_{old} \mathbf{H}_{old}}$

- **Step 2:** $\mathbf{W}_{new} = \mathbf{W}_{old} \cdot \dfrac{\mathbf{X} \mathbf{H}_{new}^T}{\mathbf{W}_{old} \mathbf{H}_{new} \mathbf{H}_{new}^T}$ *(use the $\mathbf{H}_{new}$ you just found!)*

**Answer:**

**Step 1 — Update $\mathbf{H}$:**

$$\mathbf{W}_{old}^T \mathbf{X} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 4 \\ 6 \end{bmatrix} = 4 + 6 = 10$$

$$\mathbf{W}_{old}^T \mathbf{W}_{old} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 2$$

$$\mathbf{W}_{old}^T \mathbf{W}_{old} \mathbf{H}_{old} = 2 \times 1 = 2$$

$$\mathbf{H}_{new} = 1 \cdot \frac{10}{2} = \mathbf{5}$$

**Step 2 — Update $\mathbf{W}$ (using $\mathbf{H}_{new} = 5$):**

$$\mathbf{X} \mathbf{H}_{new}^T = \begin{bmatrix} 4 \\ 6 \end{bmatrix} \times 5 = \begin{bmatrix} 20 \\ 30 \end{bmatrix}$$

$$\mathbf{W}_{old} \mathbf{H}_{new} = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \times 5 = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$$

$$\mathbf{H}_{new} \mathbf{H}_{new}^T = 5 \times 5 = 25$$

$$\mathbf{W}_{old} \mathbf{H}_{new} \mathbf{H}_{new}^T = \begin{bmatrix} 5 \\ 5 \end{bmatrix} \times 5 = \begin{bmatrix} 25 \\ 25 \end{bmatrix}$$

$$\mathbf{W}_{new} = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \odot \frac{\begin{bmatrix} 20 \\ 30 \end{bmatrix}}{\begin{bmatrix} 25 \\ 25 \end{bmatrix}} = \begin{bmatrix} 1 \times \frac{20}{25} \\ 1 \times \frac{30}{25} \end{bmatrix} = \begin{bmatrix} 0.8 \\ 1.2 \end{bmatrix}$$

**Result after one full iteration:**

$$\mathbf{W}_{new} = \begin{bmatrix} 0.8 \\ 1.2 \end{bmatrix}, \quad \mathbf{H}_{new} = \begin{bmatrix} 5 \end{bmatrix}$$

**Reconstruction check:**
$$\mathbf{W}_{new} \mathbf{H}_{new} = \begin{bmatrix} 0.8 \\ 1.2 \end{bmatrix} \times 5 = \begin{bmatrix} 4.0 \\ 6.0 \end{bmatrix} = \mathbf{X}$$

The algorithm converged to the exact solution in a single iteration. The ratio $W_1 : W_2 = 0.8 : 1.2 = 2 : 3$ correctly captures the relative structure of $\mathbf{X}$.

**Key insight:** The multiplicative update rule guarantees non-negativity at every step. If $\mathbf{W}$ and $\mathbf{H}$ start positive, the element-wise multiplication with a positive ratio keeps everything positive indefinitely — no projection step needed.
