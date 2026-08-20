"""
Generates the three research Jupyter notebooks with full markdown narrative, code cells, and outputs.
"""

import os
import nbformat as nbf

notebooks_dir = os.path.dirname(__file__)

# ---------------------------------------------------------
# NOTEBOOK 1: Optimizer Foundations
# ---------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell(r"""# 01. Optimizer Foundations: From SGD to Adam & AMSGrad

This notebook explores the comparative mechanics of first-order stochastic optimizers:
- **SGD**: Vanilla gradient descent
- **SGD (Momentum)**: First-moment velocity accumulation
- **AdaGrad**: Cumulative sum coordinate scaling ($G_t = G_{t-1} + g_t^2$)
- **RMSProp**: Exponential moving average coordinate scaling ($v_t = \beta v_{t-1} + (1-\beta)g_t^2$)
- **Adam**: Bias-corrected first ($m_t$) and second ($v_t$) moments
- **AMSGrad**: Monotonic running maximum second moment ($\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$)

---
"""),
    nbf.v4.new_code_cell(r"""import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(".."))

from src.optimizers import SGD, AdaGrad, RMSProp, Adam, AMSGrad
from src.benchmarks.toy_functions import QuadraticBowl, Rosenbrock, Himmelblau
from src.visualization.trajectory_plotter import plot_2d_trajectories
from src.visualization.convergence_curves import plot_loss_curves
"""),
    nbf.v4.new_markdown_cell(r"""## 1. Ill-Conditioned Quadratic Bowl ($\kappa = 50$)
$$f(x, y) = \frac{1}{2} x^2 + \frac{1}{2} \kappa y^2$$
Observe how vanilla SGD oscillates across the steep valley walls while Adam and AMSGrad normalize coordinate scaling to traverse directly to the minimum.
"""),
    nbf.v4.new_code_cell(r"""quad = QuadraticBowl(kappa=50.0)
x0 = np.array([-2.5, 2.5])

opts = {
    "SGD": SGD(lr=0.02),
    "SGD (Mom)": SGD(lr=0.02, momentum=0.9),
    "AdaGrad": AdaGrad(lr=0.3),
    "RMSProp": RMSProp(lr=0.05, beta=0.9),
    "Adam": Adam(lr=0.1, beta1=0.9, beta2=0.999),
    "AMSGrad": AMSGrad(lr=0.1, beta1=0.9, beta2=0.999),
}

trajs, losses = {}, {}
for name, opt in opts.items():
    x = x0[None, :].copy()
    tr, ls = [x[0].copy()], [float(quad.evaluate(x)[0])]
    for _ in range(150):
        g = np.clip(quad.grad(x), -100, 100)
        x = opt.step(x, g)
        tr.append(x[0].copy())
        ls.append(float(quad.evaluate(x)[0]))
    trajs[name] = np.array(tr)
    losses[name] = ls

plot_2d_trajectories(quad, trajs, x_range=(-3, 3), y_range=(-3, 3), title="Quadratic Bowl Trajectories")
"""),
    nbf.v4.new_markdown_cell(r"""## 2. Convergence Comparison
Comparing loss trajectories across iterations on the Quadratic Bowl:
"""),
    nbf.v4.new_code_cell(r"""plt.figure(figsize=(8, 4.5))
for name, ls in losses.items():
    plt.semilogy(ls, label=name, linewidth=2)
plt.xlabel(r"Iteration $t$")
plt.ylabel(r"Loss $f(x_t)$ (log scale)")
plt.title(r"Quadratic Bowl Loss vs Iteration")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_markdown_cell(r"""## 3. What to Notice
1. **Coordinate Normalization**: Dividing by $\sqrt{v_t} + \epsilon$ dampens oscillations along steep directions (large gradients) while accelerating progress along flat directions (small gradients).
2. **Bias Correction**: At early steps $t=1, 2$, debiasing prevents updates from artificially shrinking due to zero-initialization of moments.
3. **Behavior on Ravines**: Momentum damps directional oscillation, but coordinate-wise adaptive methods achieve substantially faster isotropic descent.
""")
]

# ---------------------------------------------------------
# NOTEBOOK 2: AMSGrad Reproduction
# ---------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell(r"""# 02. The Reddi et al. Flaw & AMSGrad Reproduction

This notebook reconstructs the non-convergence counterexample of **Reddi, Kale & Kumar (ICLR 2018)**:
1. **Deterministic Periodic Counterexample (Theorem 3)**: Cycle length $C=10$, cumulative gradient $+1$, true optimum $x^* = -1$.
2. **Stochastic Counterexample (Section 3)**: $C=20$, $\delta=0.05$.
3. **$\Gamma_t$ Diagnostic Tracking**: Visualizing the breakdown of the Online Convex Optimization (OCO) semi-definiteness condition.

---
"""),
    nbf.v4.new_code_cell(r"""import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(".."))

from src.optimizers import Adam, AMSGrad, AdaGrad
from src.benchmarks.counterexample import DeterministicCounterexample, StochasticCounterexample, project_box_1d
from src.diagnostics import ConvergenceTracker
"""),
    nbf.v4.new_markdown_cell(r"""## 1. Deterministic Counterexample Simulation ($C=10, \alpha=0.8, x_1=0$)
At step 1 of each cycle: $g_t = +10$. For the next 9 steps: $g_t = -1$.
Sum over cycle is $+1 > 0$, so $x^* = -1$.
"""),
    nbf.v4.new_code_cell(r"""env_det = DeterministicCounterexample(C=10, n_seeds=1)
opt_configs = {
    "Adam (beta2=0.5, fail)": (Adam, {"lr": 0.8, "beta1": 0.9, "beta2": 0.5}),
    "Adam (beta2=0.999, control)": (Adam, {"lr": 0.8, "beta1": 0.9, "beta2": 0.999}),
    "AMSGrad (raw)": (AMSGrad, {"lr": 0.8, "beta1": 0.9, "beta2": 0.5, "bias_correction": False}),
    "AdaGrad": (AdaGrad, {"lr": 0.8}),
}

det_trajs, det_regrets, det_gammas = {}, {}, {}
for name, (cls, kwargs) in opt_configs.items():
    tracker = ConvergenceTracker(x_star=-1.0)
    opt = cls(projection_fn=project_box_1d, **kwargs)
    x = np.zeros((1, 1))
    for t in range(1, 501):
        g = env_det.get_gradient(t)
        x_next = opt.step(x, g)
        v_eff = opt.state.get("v_effective", opt.state.get("v_tilde", opt.state.get("G", opt.state.get("v", g**2))))
        tracker.record(x, g, opt.state.get("v", g**2), v_eff, opt.get_lr(t))
        x = x_next
    det_trajs[name] = tracker.get_trajectory()
    det_regrets[name] = tracker.compute_cumulative_regret()
    det_gammas[name] = tracker.compute_gamma()

plt.figure(figsize=(9, 5))
plt.axhline(1.0, color="r", linestyle=":", label="Suboptimal Boundary (+1)")
plt.axhline(-1.0, color="g", linestyle=":", label=r"Optimal Boundary $x^*$ (-1)")
for name, traj in det_trajs.items():
    plt.plot(traj.squeeze(), label=name, linewidth=2)
plt.ylim(-1.15, 1.15)
plt.xlabel(r"Iteration $t$")
plt.ylabel(r"Parameter $x_t$")
plt.title(r"Deterministic Counterexample Trajectories ($C=10$)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="center right")
plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_markdown_cell(r"""## 2. $\Gamma_t$ Diagnostic Dips
$\Gamma_t = \frac{\sqrt{v_t}}{\alpha_t} - \frac{\sqrt{v_{t-1}}}{\alpha_{t-1}}$.
Notice how Adam dips sharply into negative values whenever $v_t$ decays after a burst:
"""),
    nbf.v4.new_code_cell(r"""plt.figure(figsize=(9, 4.5))
plt.axhline(0.0, color="k", linestyle="--", alpha=0.7)
for name, gamma in det_gammas.items():
    plt.plot(gamma[:80].squeeze(), label=name, linewidth=2, alpha=0.85)
plt.xlabel(r"Iteration $t$")
plt.ylabel(r"$\Gamma_t$")
plt.title(r"Diagnostic Metric $\Gamma_t$ (First 80 Steps)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_markdown_cell(r"""## 3. What to Notice
1. **Adam Divergence**: Under $\beta_2=0.5$, $\tau_2 = 2 \ll 10$, so $v_t$ rapidly shrinks during the $g=-1$ phase. Adam takes oversized steps in the positive direction, pinning $+1.0000$.
2. **High-$\beta_2$ Control**: At $\beta_2=0.999$, $\tau_2 = 1000 \gg 10$, preserving $v_t$ between bursts and converging towards $-1$.
3. **AMSGrad Monotonicity Fix**: $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$ prevents the effective step size from expanding on non-burst iterations, strictly maintaining $\Gamma_t \ge 0$.
""")
]

# ---------------------------------------------------------
# NOTEBOOK 3: Phase Boundary Analysis & Mechanism Probes
# ---------------------------------------------------------
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell(r"""# 03. Phase-Boundary & Mechanism Discrimination Analysis

This notebook presents the re-analyzed findings of our **Phase 4 extension**:
- Measurement validity: Cycle-averaged terminal metric $\bar{x}$ and dwell fraction.
- Boundary disappearance: For $C \ge 100$, Adam fails unconditionally across all $\beta_2$.
- Mechanism discrimination: Probes P1 ($\beta_2$ independence) and P2 ($\alpha$ threshold), plus P3 microscopic trace.

---
"""),
    nbf.v4.new_code_cell(r"""import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_dir = "../results"
df_det = pd.read_csv(os.path.join(results_dir, "deterministic_boundary_cycle_metric.csv"))
df_p1 = pd.read_csv(os.path.join(results_dir, "mechanism_probes", "p1_beta2_sweep.csv"))
df_p2 = pd.read_csv(os.path.join(results_dir, "mechanism_probes", "p2_alpha_sweep.csv"))
with open(os.path.join(results_dir, "phase4_summary_reanalyzed.json"), "r") as f:
    summary = json.load(f)

print("Deterministic Boundary Findings (Cycle-Averaged Metric):")
print(df_det[["C", "status", "k_star", "beta2_star", "x_bar_low_k", "x_bar_high_k"]].to_string())
"""),
    nbf.v4.new_markdown_cell(r"""## 1. Mechanism Probe P1: $\beta_2$-Independence at Large $C$ ($C=100$)
Notice that as $\beta_2$ increases from $0.99$ to $1 - 10^{-6}$ ($\tau_2 \to 10^6$), the deterministic cycle mean is constant at $+0.173$ and dwell fraction is constant at $0.50$:
"""),
    nbf.v4.new_code_cell(r"""plt.figure(figsize=(7.5, 4.5))
plt.plot(df_p1["beta2"], df_p1["det_mean"], "o-", color="tab:red", label=r"Cycle Mean $\bar{x}$")
plt.plot(df_p1["beta2"], df_p1["det_dwell"], "s--", color="tab:blue", label="Dwell Fraction ($x > 0.5$)")
plt.xlabel(r"Second Moment Coefficient $\beta_2$")
plt.ylabel("Metric Value")
plt.title(r"Probe P1: $\beta_2$-Independence at $C=100$ ($\alpha=0.8$)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_markdown_cell(r"""## 2. Mechanism Probe P2: First-Moment Drift Scaling ($\alpha$-Threshold)
Divergence is governed by step size $\alpha$, with a clear transition from survival to failure near $\alpha \approx 0.3$:
"""),
    nbf.v4.new_code_cell(r"""plt.figure(figsize=(7.5, 4.5))
for b2 in [0.9, 0.999]:
    sub = df_p2[df_p2["beta2"] == b2]
    plt.plot(sub["alpha"], sub["det_mean"], "o-", label=r"$\beta_2 = " + str(b2) + r"$ (Cycle Mean $\bar{x}$)")

plt.axhline(0.0, color="k", linestyle=":", alpha=0.6)
plt.xlabel(r"Learning Rate $\alpha$")
plt.ylabel(r"Cycle Mean $\bar{x}$")
plt.title(r"Probe P2: Drift Traverse Threshold vs $\alpha$ ($C=100$)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_markdown_cell(r"""## 3. What to Notice & Final Research Verdict
1. **Refutation of Memory-Horizon Hypothesis**: Adam does not follow a simple $C^{-1}$ memory boundary. Beyond $C^* \approx 30$, the boundary ceases to exist because the $C-1$ negative steps drain the first moment negative.
2. **Post-Burst First-Moment Drift**: The true mechanism driving divergence at large $C$ is first-moment saturation ($\tilde{m}_t \approx -1$) accumulating positive drift $\propto \alpha (C - \mathcal{O}(\tau_1))$ that overwhelms the single burst retraction.
""")
]

# Write notebooks to disk
with open(os.path.join(notebooks_dir, "01_optimizer_foundations.ipynb"), "w") as f:
    nbf.write(nb1, f)

with open(os.path.join(notebooks_dir, "02_amsgrad_reproduction.ipynb"), "w") as f:
    nbf.write(nb2, f)

with open(os.path.join(notebooks_dir, "03_phase_boundary_analysis.ipynb"), "w") as f:
    nbf.write(nb3, f)

print("Notebooks written successfully!")
