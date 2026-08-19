# Mechanistic Analysis: Why Adam Outperforms Vanilla SGD on Ill-Conditioned Landscapes

This note explains the comparative dynamics of SGD, Momentum SGD, RMSProp, and Adam on non-isotropic loss surfaces.

---

## 1. The Ill-Conditioning Problem in Vanilla SGD

Consider the ill-conditioned quadratic bowl:
$$f(x, y) = \frac{1}{2} x^2 + \frac{1}{2} \kappa y^2 \quad (\kappa \gg 1)$$

The gradient is $\nabla f = [x, \kappa y]^\top$. The Hessian $H = \begin{bmatrix} 1 & 0 \\ 0 & \kappa \end{bmatrix}$ has condition number $\kappa$.

- Along the $y$-axis (steep direction), the gradient magnitude is $\kappa y$, requiring a small step size $\alpha < \frac{2}{\kappa}$ to prevent explosive oscillation.
- Along the $x$-axis (gentle direction), the gradient magnitude is $x$. With $\alpha < \frac{2}{\kappa}$, progress along $x$ is painfully slow ($x_{t+1} \approx (1 - \alpha) x_t$).
- **Result:** Vanilla SGD oscillates violently across the steep valley walls while crawling along the valley floor.

---

## 2. The Momentum Mechanism: Damping Oscillations

Momentum maintains a velocity vector $m_t = \gamma m_{t-1} + g_t$:
- In oscillating directions (where gradient sign alternates), consecutive gradient vectors cancel out: $(g_t + g_{t+1} \approx 0)$.
- In persistent directions (where gradient sign remains constant), updates accumulate constructively: $m_t \approx \frac{1}{1-\gamma} g_t$.
- **Limitation:** Momentum speeds up the slow direction and damps oscillations, but it still applies a single uniform learning rate across all coordinates.

---

## 3. Coordinate-Wise Adaptive Scaling: RMSProp & Adam

Adam introduces coordinate-wise normalization by dividing by $\sqrt{\tilde{v}_t} + \epsilon$:

$$\Delta \theta_i \propto \frac{\tilde{m}_{t, i}}{\sqrt{\tilde{v}_{t, i}} + \epsilon}$$

Let us analyze the effective update magnitude along each coordinate:
1. **Along the steep $y$-axis:** Gradients are large ($g_y \sim \kappa y$). Consequently, $v_{t, y} \sim \kappa^2 y^2$.
   The effective step size scales as:
   $$\frac{\alpha}{\sqrt{v_{t, y}}} \sim \frac{\alpha}{\kappa y}$$
   The coordinate update becomes $\Delta y \approx \alpha \cdot \text{sign}(y)$, preventing overshoot and damping oscillations.
2. **Along the shallow $x$-axis:** Gradients are small ($g_x \sim x$). Consequently, $v_{t, x} \sim x^2$.
   The effective step size scales as:
   $$\frac{\alpha}{\sqrt{v_{t, x}}} \sim \frac{\alpha}{x}$$
   The coordinate update becomes $\Delta x \approx \alpha \cdot \text{sign}(x)$, drastically accelerating progress through flat plateaus.

---

## 4. Summary of Optimizer Characteristics

| Optimizer | Normalization | Directional Memory | Effective Step Magnitude | Behavior on Ravines |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | None | None | $\alpha \|g_t\|$ | Oscillates in steep ravines; stalls on flats |
| **SGD + Momentum** | None | First moment $m_t$ | $\frac{\alpha}{1-\gamma} \|g_t\|$ | Damps oscillations; still coordinate-isotropic |
| **RMSProp** | Coordinate $\sqrt{v_t}$ | None | $\approx \alpha \cdot \text{sign}(g_t)$ | Escapes ravines quickly; noisy in stochastic regimes |
| **Adam** | Coordinate $\sqrt{\tilde{v}_t}$ | First $m_t$ + Second $v_t$ | $\approx \alpha \cdot \text{sign}(m_t)$ | Rapid isotropic traversal with smooth acceleration |
