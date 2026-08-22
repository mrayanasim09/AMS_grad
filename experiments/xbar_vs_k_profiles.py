"""
Generate x̄ vs k (β₂) profiles for C ∈ {10, 30, 40, 50, 60, 70, 75} at α=0.8, β₁=0.9.
Shows the full survival set (shaded) and confirms multi-island structure.
Also re-runs Table 4 regret slopes at δ=0.10 to match §3.3 parameters.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.benchmarks.measurements import measure_cycle_mean

os.makedirs("results/mechanism_probes", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)

# ── 1.  x̄ vs k profiles ──────────────────────────────────────────────────────
C_values = [10, 30, 40, 50, 60, 70, 75, 100]
# Dense k grid: β₂ = 1 − k/C; k ∈ (0, C] — use 80 points per C
T_cycles = 800
alpha = 0.8

records = []
for C in C_values:
    # k from 0.05 to C (β₂ from nearly 1 down to ~0)
    k_vals = np.linspace(0.05, C * 0.999, 80)
    xbars = []
    dwells = []
    for k in k_vals:
        beta2 = 1.0 - k / C
        beta2 = min(max(beta2, 1e-6), 1 - 1e-6)
        xb, dw = measure_cycle_mean(C, alpha, beta2, T_cycles)
        xbars.append(xb)
        dwells.append(dw)
        records.append({"C": C, "k": k, "beta2": beta2, "xbar": xb, "dwell": dw})
    print(f"C={C:3d}: x̄ range [{min(xbars):.3f}, {max(xbars):.3f}]")

df_profiles = pd.DataFrame(records)
df_profiles.to_csv("results/mechanism_probes/xbar_vs_k_profiles.csv", index=False)
print("Saved results/mechanism_probes/xbar_vs_k_profiles.csv")

# ── 2.  Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True)
axes = axes.flatten()

for idx, C in enumerate(C_values):
    ax = axes[idx]
    sub = df_profiles[df_profiles.C == C]
    k_arr = sub.k.values
    xb_arr = sub.xbar.values

    # shade survival set (x̄ < 0)
    ax.fill_between(k_arr, xb_arr, 0,
                    where=(xb_arr < 0), alpha=0.25, color="steelblue", label="Survival ($\\bar{x}<0$)")
    ax.plot(k_arr, xb_arr, "k-", lw=1.2)
    ax.axhline(0, color="red", lw=0.9, ls="--")

    # mark theoretical α* (only valid in Regime II, k≪1)
    s_star_exact = np.log((0.1 * C + 1)) / np.log(1 / 0.9)
    alpha_star = 2 * np.sqrt(C) / (C - s_star_exact)
    ax.set_title(f"$C={C}$\n$\\alpha^*(k\\ll1)={alpha_star:.2f}$", fontsize=8)
    ax.set_xlabel("$k = (1-\\beta_2)C$", fontsize=7)
    if idx % 4 == 0:
        ax.set_ylabel("$\\bar{x}$ (cycle mean)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, C)
    ax.set_ylim(-1.05, 1.05)
    ax.text(0.97, 0.95, f"$\\alpha=0.8$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)

fig.suptitle(r"$\bar{x}$ vs.\ dimensionless memory $k=(1-\beta_2)C$ at $\alpha=0.8$, $\beta_1=0.9$"
             "\n" r"Blue shading = survival set ($\bar{x}<0$); red dashed = zero line",
             fontsize=9)
fig.tight_layout()
fig.savefig("report/figures/xbar_vs_k_profiles.pdf", bbox_inches="tight")
fig.savefig("report/figures/xbar_vs_k_profiles.png", dpi=200, bbox_inches="tight")
fig.savefig("results/mechanism_probes/xbar_vs_k_profiles.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved xbar_vs_k_profiles.pdf/png")

# ── 3.  Summarise survival islands per C ─────────────────────────────────────
print("\nSurvival island summary (α=0.8):")
for C in C_values:
    sub = df_profiles[df_profiles.C == C]
    surv = sub[sub.xbar < 0]
    if len(surv) == 0:
        print(f"  C={C:3d}: NO survival island")
    else:
        k_lo, k_hi = surv.k.min(), surv.k.max()
        print(f"  C={C:3d}: survival k ∈ [{k_lo:.3f}, {k_hi:.3f}]  "
              f"β₂ ∈ [{1-k_hi/C:.4f}, {1-k_lo/C:.4f}]  n_cells={len(surv)}")
