# Paper Fidelity Checklist & Transcription Comparison

This document records the exact mathematical and experimental alignment between our codebase and the source literature:
- **Kingma & Ba (2014)**: *Adam: A Method for Stochastic Optimization* (arXiv:1412.6980 / ICLR 2015)
- **Reddi, Kale & Kumar (2018)**: *On the Convergence of Adam and Beyond* (arXiv:1904.09237 / ICLR 2018)

---

## 1. Stochastic Counterexample Parameterization (Reddi et al. Section 3)

### Literal Paper Specification:
- Domain: $\mathcal{F} = [-1, 1]$.
- Objective function:
  $$f_t(x) = \begin{cases} C x & \text{with probability } p = \frac{1+\delta}{C} \\ -x & \text{with probability } 1 - p \end{cases}$$
- Expected gradient in paper formulation:
  $$\mathbb{E}[\nabla f_t(x)] = p \cdot C + (1 - p) \cdot (-1) = (1 + \delta) - (1 - p) = \delta + p = \delta + \frac{1+\delta}{C}$$
  For $C \gg 1$, this expectation is asymptotically $\delta + \mathcal{O}(1/C) > 0$.

### Our Specification:
- Domain: $\mathcal{F} = [-1, 1]$.
- Probability choice:
  $$p = \frac{1 + \delta}{C + 1}$$
- Exact expected gradient in our codebase:
  $$\mathbb{E}[\nabla f_t(x)] = p \cdot C - (1 - p) = p(C + 1) - 1 = (1 + \delta) - 1 = \delta \quad (\text{exact for all } C)$$
- **Documentation Note:** We document that our choice provides an algebraically exact expected drift of $\delta$ for all finite $C$, while remaining asymptotically identical to Reddi et al.'s setting as $C \to \infty$.

---

## 2. Deterministic Counterexample Construction (Reddi et al. Theorem 3)

### Literal Paper Construction:
- Theorem 3 sets up an online linear optimization problem over $T$ steps structured in repeating blocks of length $C$.
- Within each block $k \in \{1, \dots, T/C\}$:
  - Step 1 of the block: $f_t(x) = C x \implies g_t = C$.
  - Steps $2, \dots, C$ of the block: $f_t(x) = -x \implies g_t = -1$.
- Total gradient over each block:
  $$\sum_{t=1}^C g_t = C + (C - 1)(-1) = +1 > 0$$
- In hindsight, the unique optimal fixed point minimizing cumulative loss is $x^* = -1$.

### Codebase Alignment:
- Implemented in [`DeterministicCounterexample`](../src/benchmarks/counterexample.py):
  $$g_t = \begin{cases} +C & \text{if } (t - 1) \pmod C = 0 \\ -1 & \text{otherwise} \end{cases}$$
- Initialization: $x_1 = 0 \in [-1, 1]$.
- **Status:** EXACT mathematical match to the Theorem 3 epoch gradient schedule.

---

## 3. Experimental Hyperparameters in Literature

| Paper & Experiment | $\alpha$ / Schedule | $\beta_1$ | $\beta_2$ | $\epsilon$ | $x_1$ | Objective / Task |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Kingma & Ba (2014)** Standard Defaults | $\alpha = 10^{-3}$ | $0.9$ | $0.999$ | $10^{-8}$ (outside sqrt) | - | MNIST, CIFAR-10, Logistic Regression |
| **Reddi et al. (2018)** Synthetic Fig. 1 | $\alpha_t = \frac{\alpha}{\sqrt{t}}$ ($\alpha = 0.5$) | $0.9$ | $0.99$ | $10^{-8}$ | $0$ | 1D Stochastic ($C=1010, p=1/1010$) |
| **Our Phase 3 Replication** (Deterministic) | $\alpha = 0.8$ (constant) | $0.9$ | $0.5$ (fail) vs $0.999$ (ctrl) | $10^{-8}$ | $0$ | 1D Deterministic ($C=10, T=500$) |
| **Our Phase 3 Replication** (Stochastic) | $\alpha = 0.8$ (constant) | $0.9$ | $0.5$ (fail) vs AMSGrad | $10^{-8}$ | $0$ | 1D Stochastic ($C=20, \delta=0.05, T=3000$) |
| **Our Phase 4 Extension** | $\alpha = 0.8$ | $0.9$ | Swept via $k=(1-\beta_2)C$ | $10^{-8}$ | $0$ | $(k, C)$ Grid Sweep ($N=100$ seeds) |

---

## 4. Key Algorithm Variations & Notation Pinning

1. **Adam Bias Correction:**
   $$\tilde{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \tilde{v}_t = \frac{v_t}{1 - \beta_2^t}$$
2. **AMSGrad Algorithm 2 (Reddi et al.):**
   $$\hat{v}_t = \max(\hat{v}_{t-1}, v_t), \quad \text{update} = \frac{\alpha_t}{\sqrt{\hat{v}_t} + \epsilon} m_t \quad (\text{raw, no debiasing})$$
3. **AMSGrad PyTorch Variant:**
   $$\hat{v}_t = \max(\hat{v}_{t-1}, v_t), \quad \tilde{v}_t = \frac{\hat{v}_t}{1 - \beta_2^t}, \quad \tilde{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \text{update} = \frac{\alpha_t}{\sqrt{\tilde{v}_t} + \epsilon} \tilde{m}_t$$
