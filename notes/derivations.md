# Mathematical Derivation: The Gamma_t Convergence Flaw & AMSGrad Resolution

This document presents a self-contained mathematical derivation of the regret bound in adaptive gradient methods, the breakdown of the positive semi-definiteness condition $\Gamma_t \succeq 0$, and the exact proof that AMSGrad's monotonic projection $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$ restores convergence.

---

## 1. The Online Convex Optimization (OCO) Framework

Let $\mathcal{F} \subset \mathbb{R}^d$ be a convex compact domain with diameter $D_{\infty} = \sup_{x, y \in \mathcal{F}} \|x - y\|_{\infty}$. At each round $t \in [1, T]$, an online convex objective $f_t: \mathcal{F} \to \mathbb{R}$ is chosen by the environment.

The cumulative regret against the optimal fixed point $x^* \in \mathcal{F}$ in hindsight is:
$$R_T = \sum_{t=1}^T \left(f_t(x_t) - f_t(x^*)\right) \le \sum_{t=1}^T \langle g_t, x_t - x^* \rangle$$

---

## 2. Generic Adaptive Projection Update

Consider the general adaptive coordinate update with metric matrix $V_t = \text{diag}(\sqrt{v_t}) / \alpha_t$:
$$x_{t+1} = \Pi_{\mathcal{F}}^{V_t} \left( x_t - V_t^{-1} g_t \right) = \arg\min_{x \in \mathcal{F}} \| x - (x_t - V_t^{-1} g_t) \|_{V_t}^2$$

By the standard projection inequality for metric $V_t$:
$$\| x_{t+1} - x^* \|_{V_t}^2 \le \| x_t - V_t^{-1} g_t - x^* \|_{V_t}^2 = \| x_t - x^* \|_{V_t}^2 - 2 \langle g_t, x_t - x^* \rangle + \| g_t \|_{V_t^{-1}}^2$$

Rearranging for the instantaneous regret:
$$\langle g_t, x_t - x^* \rangle \le \frac{1}{2} \left( \| x_t - x^* \|_{V_t}^2 - \| x_{t+1} - x^* \|_{V_t}^2 \right) + \frac{1}{2} \| g_t \|_{V_t^{-1}}^2$$

Summing from $t=1$ to $T$:
$$\sum_{t=1}^T \langle g_t, x_t - x^* \rangle \le \frac{1}{2} \| x_1 - x^* \|_{V_1}^2 + \frac{1}{2} \sum_{t=2}^T \langle x_t - x^*, (V_t - V_{t-1}) (x_t - x^*) \rangle + \frac{1}{2} \sum_{t=1}^T \| g_t \|_{V_t^{-1}}^2$$

Define the differential metric matrix:
$$\Gamma_t \triangleq V_t - V_{t-1} = \frac{\text{diag}(\sqrt{v_t})}{\alpha_t} - \frac{\text{diag}(\sqrt{v_{t-1}})}{\alpha_{t-1}}$$

---

## 3. The Flaw in Adam: When $\Gamma_t \not\succeq 0$

To bound the middle summation by diameter $D_{\infty}$:
$$\sum_{t=2}^T \langle x_t - x^*, \Gamma_t (x_t - x^*) \rangle \le D_{\infty}^2 \sum_{t=2}^T \text{Tr}(\Gamma_t) = D_{\infty}^2 \text{Tr}(V_T - V_1)$$

**This telescoping bound requires $\Gamma_t \succeq 0$ (positive semi-definite).**

If $\Gamma_t$ has negative eigenvalues ($\sqrt{v_t} < \sqrt{v_{t-1}}$), the inequality:
$$\langle x_t - x^*, \Gamma_t (x_t - x^*) \rangle \le D_{\infty}^2 \text{Tr}(\Gamma_t)$$
**FAILS** because $(x_t - x^*)^2$ can be large while $\Gamma_t$ is negative, adding unbounded positive regret!

### In Adam:
Since $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$, if a gradient burst $g_1 = C$ is followed by small gradients $g_t = -1$, then:
$$v_t \approx \beta_2^{t-1} C^2 \to 1$$
$v_t$ decreases exponentially, rendering $\Gamma_t < 0$. Over $T$ steps, the cumulative negative violations cause linear regret $R_T = \Omega(T)$.

---

## 4. The AMSGrad Solution: Enforcing Monotonicity

AMSGrad modifies the second moment update by taking the running coordinate-wise maximum:
$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$

Because $\hat{v}_t \ge \hat{v}_{t-1}$ by definition, and assuming a non-increasing step-size schedule $\alpha_t \le \alpha_{t-1}$:
$$\Gamma_t = \frac{\sqrt{\hat{v}_t}}{\alpha_t} - \frac{\sqrt{\hat{v}_{t-1}}}{\alpha_{t-1}} \ge 0 \quad \forall t$$

Thus:
1. $\Gamma_t \succeq 0$ holds strictly at every step.
2. The telescoping sum telescopes legitimately:
   $$\sum_{t=2}^T \langle x_t - x^*, \Gamma_t (x_t - x^*) \rangle \le D_{\infty}^2 \sum_{i=1}^d \frac{\sqrt{\hat{v}_{T, i}}}{\alpha_T}$$
3. Combining with the second term $\sum_{t=1}^T \frac{\alpha_t g_{t, i}^2}{\sqrt{\hat{v}_{t, i}}} \le O(\sqrt{T})$, AMSGrad guarantees sublinear regret:
   $$R_T = O(\sqrt{T})$$
   $$\lim_{T \to \infty} \frac{R_T}{T} = 0$$
