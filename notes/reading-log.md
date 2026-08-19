# Reading Log & Paper Notes (Three-Pass Method)

This log documents structured readings of the seminal papers underpinning adaptive gradient methods and their convergence properties.

---

## Paper 1: *Adam: A Method for Stochastic Optimization*
**Authors:** Diederik P. Kingma, Jimmy Ba (ICLR 2015 / arXiv:1412.6980)

### Pass 1: Overview & High-Level Narrative (10 min)
- **Problem:** Stochastic non-convex optimization in high dimensions with noisy and sparse gradients.
- **Core Proposal:** Combine the advantages of AdaGrad (handling sparse gradients via coordinate-wise scaling) and RMSProp (handling non-stationary objectives via exponential moving averages).
- **Key Mechanism:** First-moment estimate $m_t$ (exponential moving average of gradients) and second-moment estimate $v_t$ (exponential moving average of squared gradients), with explicit **bias correction** terms $\frac{1}{1-\beta_1^t}$ and $\frac{1}{1-\beta_2^t}$ to address initialization bias towards zero.

### Pass 2: Setup, Update Rules & Architecture (45 min)
- **Algorithm (Paper Pseudocode):**
  $$g_t = \nabla_{\theta} f_t(\theta_{t-1})$$
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\tilde{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \tilde{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\tilde{v}_t} + \epsilon} \tilde{m}_t$$
- **Bias Correction Derivation:**
  Expanding $v_t = (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} g_i^2$. Taking expectation under stationary second moment $\mathbb{E}[g_i^2] = \mathbb{E}[g_t^2]$:
  $$\mathbb{E}[v_t] = \mathbb{E}\left[(1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} g_i^2\right] = \mathbb{E}[g_t^2] (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} = \mathbb{E}[g_t^2] (1 - \beta_2^t)$$
  Dividing by $(1 - \beta_2^t)$ makes the estimator strictly unbiased at every step $t$.

### Pass 3: Detailed Scrutiny & Theoretical Gaps
- Kingma & Ba provided a regret bound in the Online Convex Optimization (OCO) setting (Theorem 10.1 in arXiv v9).
- **The Gap:** The proof required the quantity $\Gamma_{t+1} = \frac{\sqrt{v_{t+1}}}{\alpha_{t+1}} - \frac{\sqrt{v_t}}{\alpha_t}$ to be positive semi-definite ($\Gamma_{t+1} \succeq 0$). However, with exponential moving averages, $v_{t+1} < v_t$ whenever a large gradient is followed by small gradients, violating this assumption!

---

## Paper 2: *On the Convergence of Adam and Beyond*
**Authors:** Sashank J. Reddi, Satyen Kale, Sanjiv Kumar (ICLR 2018 / arXiv:1904.09237)

### Pass 1: Overview & Core Result (10 min)
- **Problem:** Many popular adaptive methods (Adam, RMSProp, AdaDelta) have fundamental flaws in their theoretical convergence proofs and can diverge even on simple 1D convex optimization problems.
- **Root Cause:** "Short-term memory" in exponential moving averages causes the effective learning rate to increase in the wrong direction when large gradients occur infrequently.
- **The Fix (AMSGrad):** Enforce monotonicity in the second-moment estimate by maintaining a running maximum of past second moments: $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$.

### Pass 2: The Counterexample & Theoretical Flaw (60 min)
- **Theoretical Flaw:** Standard OCO regret bounds rely on Telescoping terms of the form $\sum_{t=1}^T \langle x_t - x^*, \Gamma_t (x_t - x^*) \rangle$. If $\Gamma_t \not\succeq 0$, regret can grow linearly ($R_T = \Omega(T)$), implying non-zero average regret and divergence.
- **Deterministic 1D Counterexample (Theorem 3):**
  A periodic sequence where $g_t = C$ if $t \equiv 1 \pmod C$, and $g_t = -1$ otherwise.
  Over each cycle of length $C$, the cumulative gradient is $+1$, so $x^* = -1$.
  However, right after the burst at $t=1$, $v_t$ shrinks exponentially fast over the remaining $C-1$ steps. Adam takes large steps with $g_t = -1$, causing net positive drift towards $+1$.

### Pass 3: Mathematical Proof of AMSGrad
- By replacing $v_t$ with $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$, we strictly guarantee $\hat{v}_t \ge \hat{v}_{t-1}$.
- When step size $\alpha_t = \frac{\alpha}{\sqrt{t}}$ is non-increasing, $\Gamma_{t+1} = \frac{\sqrt{\hat{v}_{t+1}}}{\alpha_{t+1}} - \frac{\sqrt{\hat{v}_t}}{\alpha_t} \succeq 0$ holds unconditionally, restoring the $O(\sqrt{T})$ regret bound.
