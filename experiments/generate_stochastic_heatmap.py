"""
Generate Failure-Probability Heatmap over (C, k) from the 4,500-run stochastic sweep,
and compute the terminal burst arrival timing diagnostic.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.makedirs("report/figures", exist_ok=True)
os.makedirs("results/mechanism_probes", exist_ok=True)

# 1. Generate Heatmap from phase4_grid_results.csv
df = pd.read_csv("results/phase4_grid_results.csv")
pivot = df.pivot(index="C", columns="k", values="failure_fraction")

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(pivot.values, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="auto")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Empirical Failure Fraction ($x_T > 0.5$)", fontsize=10)

ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_xticklabels([f"{k:g}" for k in pivot.columns], fontsize=9)
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)

ax.set_xlabel(r"Dimensionless Memory Parameter $k = (1-\beta_2)C$", fontsize=10)
ax.set_ylabel(r"Burst Scale $C$", fontsize=10)
ax.set_title(r"Stochastic Failure Probability Heatmap ($4{,}500$ runs, $N=100$/cell, $\delta=0.10$)", fontsize=11, fontweight="bold")

# Annotate cell values with Wilson 95% CI error bounds
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        # Wilson 95% CI half-width for N=100
        z = 1.96
        n = 100
        denom = 1.0 + z**2 / n
        center = (val + z**2 / (2 * n)) / denom
        half_w = (z * np.sqrt(val * (1 - val) / n + z**2 / (4 * n**2))) / denom
        text_color = "white" if val > 0.65 or val < 0.25 else "black"
        ax.text(j, i, f"{val:.2f}\n" + r"$\pm$" + f"{half_w:.2f}",
                ha="center", va="center", color=text_color, fontsize=7.5)

fig.tight_layout()
fig.savefig("report/figures/stochastic_phase_heatmap.pdf")
fig.savefig("report/figures/stochastic_phase_heatmap.png", dpi=300)
fig.savefig("results/mechanism_probes/stochastic_phase_heatmap.pdf")
fig.savefig("results/mechanism_probes/stochastic_phase_heatmap.png", dpi=300)
plt.close(fig)
print("Saved stochastic_phase_heatmap.pdf/png")

# 2. Burst Arrival Timing Diagnostic (Point-Biserial Correlation and Conditional Distributions)
from src.benchmarks.counterexample import StochasticCounterexample, project_box_1d
from src.optimizers import Adam
from scipy import stats

C = 10
delta = 0.1
N_seeds = 500
T = 550
alpha = 0.8
beta1 = 0.9
beta2 = 0.5

env = StochasticCounterexample(C=C, delta=delta, n_seeds=N_seeds, base_seed=12345)
opt = Adam(lr=alpha, beta1=beta1, beta2=beta2, projection_fn=project_box_1d)
x = np.zeros((N_seeds, 1))

last_burst_step = np.zeros(N_seeds, dtype=int)
for t in range(1, T + 1):
    g = env.get_gradient(t)
    x = opt.step(x, g)
    is_burst = (g[:, 0] == C)
    last_burst_step = np.where(is_burst, t, last_burst_step)

x_T = x[:, 0]
failed = (x_T > 0.5).astype(int)
time_since_last_burst = T - last_burst_step

df_timing = pd.DataFrame({
    "seed": np.arange(N_seeds),
    "x_T": x_T,
    "failed": failed,
    "time_since_last_burst": time_since_last_burst,
})
df_timing.to_csv("results/mechanism_probes/terminal_burst_timing_diagnostic.csv", index=False)

# Point-biserial correlation
pb_corr, pb_pval = stats.pointbiserialr(failed, time_since_last_burst)
mean_time_failed = df_timing[df_timing.failed == 1].time_since_last_burst.mean()
mean_time_passed = df_timing[df_timing.failed == 0].time_since_last_burst.mean()

print(f"Point-biserial correlation(failed, time_since_last_burst): {pb_corr:.4f} (p={pb_pval:.2e})")
print(f"Mean time since last burst | failed=1: {mean_time_failed:.2f} steps")
print(f"Mean time since last burst | failed=0: {mean_time_passed:.2f} steps")
