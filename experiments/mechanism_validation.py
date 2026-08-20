"""Alpha* traverse threshold bisection experiment + gap tail diagnostic at C=10."""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks.counterexample import DeterministicCounterexample, StochasticCounterexample, project_box_1d
from src.optimizers import Adam

os.makedirs("results/mechanism_probes", exist_ok=True)


def cycle_mean_at_alpha(C, alpha, beta2=0.999, T_cycles=200):
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


# ─── Alpha* Bisection ─────────────────────────────────────────────────────────
tau1 = 10.0
beta1 = 0.9

print("=== ALPHA* BISECTION VALIDATION EXPERIMENT (beta2=0.999) ===\n")
records = []
C_vals_exp = [30, 100, 300, 1000]

for C in C_vals_exp:
    s_star = tau1 * np.log((1.0 - beta1) * C + 1.0)
    alpha_pred = 2.0 * np.sqrt(C) / (C - s_star)

    alpha_lo, alpha_hi = 0.01, 3.0
    x_lo = cycle_mean_at_alpha(C, alpha_lo)
    x_hi = cycle_mean_at_alpha(C, alpha_hi)

    if x_lo < 0 and x_hi > 0:
        for _ in range(20):
            alpha_mid = 0.5 * (alpha_lo + alpha_hi)
            x_mid = cycle_mean_at_alpha(C, alpha_mid)
            if x_mid < 0:
                alpha_lo = alpha_mid
            else:
                alpha_hi = alpha_mid
        alpha_star_obs = float(0.5 * (alpha_lo + alpha_hi))
        bracket_status = "genuine"
    elif x_lo > 0:
        alpha_star_obs = float(alpha_lo)
        bracket_status = "always_fails_even_at_alpha_0.01"
    else:
        alpha_star_obs = None
        bracket_status = "always_survives"

    obs_str = f"{alpha_star_obs:.4f}" if alpha_star_obs is not None else "N/A"
    print(f"C = {C:4d}: s* = {s_star:.1f}, alpha_pred = {alpha_pred:.4f}, alpha_obs = {obs_str} [{bracket_status}]")

    records.append({
        "C": C,
        "s_star_predicted": round(s_star, 2),
        "alpha_star_predicted": round(alpha_pred, 4),
        "alpha_star_observed": round(alpha_star_obs, 4) if alpha_star_obs is not None else None,
        "bracket_status": bracket_status,
    })

df_alpha_star = pd.DataFrame(records)
df_alpha_star.to_csv("results/mechanism_probes/alpha_star_bisection.csv", index=False)
print("\n", df_alpha_star.to_string(index=False))

# ─── Figure ───────────────────────────────────────────────────────────────────
C_arr = np.array([r["C"] for r in records])
alpha_pred_arr = np.array([r["alpha_star_predicted"] for r in records])
alpha_obs_arr = np.array([
    r["alpha_star_observed"] if r["alpha_star_observed"] is not None else np.nan
    for r in records
])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(C_arr, alpha_pred_arr, "o--", color="navy",
        label=r"Predicted: $\alpha^*(C) = \frac{2\sqrt{C}}{C - s^*(C)}$")
valid = ~np.isnan(alpha_obs_arr)
if valid.any():
    ax.plot(C_arr[valid], alpha_obs_arr[valid], "s-", color="tab:red",
            label=r"Observed $\alpha^*$ (bisection, $\beta_2=0.999$)")
ax.set_xscale("log")
ax.set_xlabel("Burst Scale $C$", fontsize=12)
ax.set_ylabel(r"Traverse Threshold $\alpha^*(C)$", fontsize=12)
ax.set_title(r"Alpha Traverse Threshold vs Burst Scale $C$", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, which="both", linestyle=":", alpha=0.6)
plt.tight_layout()
for dst in ["results/mechanism_probes/alpha_star_experiment.pdf",
            "results/mechanism_probes/alpha_star_experiment.png",
            "report/figures/alpha_star_experiment.pdf",
            "report/figures/alpha_star_experiment.png"]:
    plt.savefig(dst, dpi=300 if dst.endswith(".png") else None)
plt.close()
print("\nAlpha* experiment figure saved.")


# ─── Gap Tail Diagnostic C=10 ─────────────────────────────────────────────────
print("\n=== GAP TAIL DIAGNOSTIC (C=10, N=200 seeds) ===\n")

C10, delta10, N10 = 10, 0.1, 200
env10 = StochasticCounterexample(C=C10, delta=delta10, n_seeds=N10, base_seed=12345)
T10 = 50 * (C10 + 1)
alpha10, beta2_10 = 0.8, 0.5

opt10 = Adam(lr=alpha10, beta1=0.9, beta2=beta2_10, projection_fn=project_box_1d)
x10 = np.zeros((N10, 1))

# Track per-seed maximum consecutive gap length (consecutive -1 streak)
gap_tracker = np.zeros(N10, dtype=int)        # current running gap
max_gap = np.zeros(N10, dtype=int)            # max gap seen

g_history = []
for t in range(1, T10 + 1):
    g = env10.get_gradient(t)
    g_history.append(g[:, 0].copy())
    x10 = opt10.step(x10, g)
    is_neg = (g[:, 0] < 0)
    gap_tracker = np.where(is_neg, gap_tracker + 1, 0)
    max_gap = np.maximum(max_gap, gap_tracker)

x10_T = x10[:, 0]
gap_df = pd.DataFrame({
    "seed": np.arange(N10),
    "x_T": x10_T,
    "max_gap_length": max_gap,
    "failed": (x10_T > 0.5).astype(int),
})
gap_df.to_csv("results/mechanism_probes/gap_tail_diagnostic.csv", index=False)

corr = gap_df[["max_gap_length", "failed"]].corr().iloc[0, 1]
print(f"Correlation(max_gap, failed): {corr:.4f}")
print(f"Mean max_gap | failed=1 : {gap_df[gap_df.failed == 1].max_gap_length.mean():.2f}")
print(f"Mean max_gap | failed=0 : {gap_df[gap_df.failed == 0].max_gap_length.mean():.2f}")
print("(k=10, C=10 cell note: beta2 = max(1 - 10/10, 0.05) = 0.05, floored; behaves like sign-SGD-momentum)")

print("\nAll diagnostics complete.")
