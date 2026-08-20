"""Generate the two-panel alpha* validation figure."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.benchmarks.counterexample import DeterministicCounterexample, project_box_1d
from src.optimizers import Adam


def cycle_mean(C, alpha, beta2, T_cycles=150):
    T = T_cycles * (C + 1)
    env = DeterministicCounterexample(C=C, n_seeds=1)
    opt = Adam(lr=alpha, beta1=0.9, beta2=beta2, projection_fn=project_box_1d)
    x = np.zeros((1, 1))
    for t in range(1, T + 1):
        x = opt.step(x, env.get_gradient(t))
    x_cycle = []
    for _ in range(C):
        t = T + _ + 1
        x = opt.step(x, env.get_gradient(t))
        x_cycle.append(float(x[0, 0]))
    return np.mean(x_cycle)


tau1, beta1 = 10.0, 0.9

C_plot = np.array([30, 100, 300, 1000])
s_arr = tau1 * np.log((1 - beta1) * C_plot + 1)
alpha_pred_arr = 2.0 * np.sqrt(C_plot) / (C_plot - s_arr)

obs_C = [100, 300]
obs_a = [0.2514, 0.1375]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel A: alpha*(C) ---
ax = axes[0]
ax.plot(C_plot, alpha_pred_arr, "o--", color="navy", ms=7,
        label=r"Predicted: $\alpha^* = 2\sqrt{C}/(C-s^*)$")
ax.plot(obs_C, obs_a, "s", color="tab:red", ms=9, zorder=5,
        label=r"Observed $\alpha^*$ (bisection)")

ax.annotate("C=30\nalways survives\n(beta2=0.999)",
            xy=(30, alpha_pred_arr[0]), xytext=(22, 0.82), fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.annotate("C=1000\n(not bracketed)",
            xy=(1000, alpha_pred_arr[3]), xytext=(550, 0.12), fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="gray"))

ax.set_xscale("log")
ax.set_xlabel("Burst Scale $C$", fontsize=12)
ax.set_ylabel(r"Traverse Threshold $\alpha^*(C)$", fontsize=12)
ax.set_title("Predicted vs Observed alpha* (beta2=0.999)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, which="both", linestyle=":", alpha=0.6)

# --- Panel B: cycle mean vs alpha at C=100 ---
ax2 = axes[1]
alphas = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
xbars = [cycle_mean(100, a, 0.999) for a in alphas]

ax2.plot(alphas, xbars, "bs-", ms=5, label="Cycle mean (C=100, beta2=0.999)")
ax2.axhline(0, color="k", linestyle="--", alpha=0.7, label="x_bar = 0")
ax2.axvline(0.2514, color="tab:red", linestyle=":", lw=1.5, label="alpha*_obs = 0.251")
ax2.axvline(0.2631, color="navy", linestyle=":", lw=1.5, label="alpha*_pred = 0.263")
ax2.fill_betweenx([-0.25, 0.05], 0.24, 0.48, alpha=0.12, color="green", label="Survival window")

ax2.set_xlabel(r"Learning Rate $\alpha$", fontsize=12)
ax2.set_ylabel(r"Cycle Mean $\bar{x}$ (final cycle)", fontsize=12)
ax2.set_title("C=100, beta2=0.999: Cycle Mean vs alpha", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.set_ylim(-0.30, 0.75)

fig.tight_layout()
for dst in [
    "results/mechanism_probes/alpha_star_experiment.pdf",
    "results/mechanism_probes/alpha_star_experiment.png",
    "report/figures/alpha_star_experiment.pdf",
    "report/figures/alpha_star_experiment.png",
]:
    fig.savefig(dst, dpi=150)
plt.close(fig)
print("Figure saved.")
