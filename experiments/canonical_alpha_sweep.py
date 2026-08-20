"""
Canonical Alpha-Sweep Rebuild and Alpha* Validation.
Uses src/benchmarks/measurements.py exclusively.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.benchmarks.measurements import measure_cycle_mean

os.makedirs("results/mechanism_probes", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)


def run_canonical_alpha_sweep_C100():
    print("=== 1. CANONICAL ALPHA-SWEEP AT C=100, beta2=0.999 ===")
    alphas = np.round(np.arange(0.05, 1.001, 0.01), 2)
    records = []
    
    for a in alphas:
        xb400, dw400 = measure_cycle_mean(C=100, alpha=float(a), beta2=0.999, T_cycles=400)
        xb800, dw800 = measure_cycle_mean(C=100, alpha=float(a), beta2=0.999, T_cycles=800)
        diff = abs(xb400 - xb800)
        assert diff < 0.05, f"Convergence gate failed at alpha={a}: |400-800|={diff:.4f}"
        records.append({
            "alpha": float(a),
            "x_bar_400": xb400,
            "x_bar_800": xb800,
            "diff": diff,
            "dwell_800": dw800,
        })
        
    df = pd.DataFrame(records)
    df.to_csv("results/mechanism_probes/canonical_alpha_sweep_C100.csv", index=False)
    print(f"Swept {len(df)} points over alpha in [0.05, 1.00]. All passed convergence gate |400-800| < 0.05.")
    return df


def bisect_alpha_star(C: int, beta2: float = 0.999, T_cycles: int = 800, alpha_bracket=(0.05, 0.50)):
    """Bisect to find alpha* where cycle mean x_bar crosses 0."""
    al, ah = alpha_bracket
    xl, _ = measure_cycle_mean(C, al, beta2, T_cycles)
    xh, _ = measure_cycle_mean(C, ah, beta2, T_cycles)
    
    # Check for bracket
    if (xl > 0 and xh < 0) or (xl < 0 and xh > 0):
        for _ in range(25):
            am = 0.5 * (al + ah)
            xm, _ = measure_cycle_mean(C, am, beta2, T_cycles)
            if (xl > 0 and xm > 0) or (xl < 0 and xm < 0):
                al = am
                xl = xm
            else:
                ah = am
                xh = xm
        alpha_star = 0.5 * (al + ah)
        half_interval = 0.5 * abs(ah - al)
        return alpha_star, half_interval, "genuine_bracket"
    else:
        return None, None, f"no_bracket (x_lo={xl:.4f}, x_hi={xh:.4f})"


def run_alpha_star_bisection():
    print("\n=== 2. ALPHA* BISECTION WITH CONVERGENCE GATE ===")
    tau1 = 10.0
    beta1 = 0.9
    
    bisection_results = []
    
    for C in [100, 300]:
        s_star = tau1 * np.log((1.0 - beta1) * C + 1.0)
        alpha_pred = 2.0 * np.sqrt(C) / (C - s_star)
        
        bracket = (0.20, 0.30) if C == 100 else (0.12, 0.15)
        a_star, h_int, status = bisect_alpha_star(C=C, beta2=0.999, T_cycles=800, alpha_bracket=bracket)
        
        # Verify convergence gate at bisected alpha*
        if a_star is not None:
            xb400, _ = measure_cycle_mean(C, a_star, 0.999, 400)
            xb800, _ = measure_cycle_mean(C, a_star, 0.999, 800)
            assert abs(xb400 - xb800) < 0.05, f"Convergence gate failed at bisected alpha*={a_star}"
            ratio = a_star / alpha_pred
        else:
            ratio = None
            
        a_obs_str = f"{a_star:.4f}" if a_star is not None else "N/A"
        rat_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        print(f"C={C:4d}: s*={s_star:.2f}, alpha_pred={alpha_pred:.4f}, alpha_obs={a_obs_str}, ratio={rat_str}")
        
        bisection_results.append({
            "C": C,
            "beta2_used": 0.999,
            "s_star_predicted": round(s_star, 2),
            "alpha_star_predicted": round(alpha_pred, 4),
            "alpha_star_observed": round(a_star, 4) if a_star is not None else None,
            "half_interval": round(h_int, 6) if h_int is not None else None,
            "ratio_obs_pred": round(ratio, 3) if ratio is not None else None,
            "status": status,
        })
        
    df_bisect = pd.DataFrame(bisection_results)
    df_bisect.to_csv("results/mechanism_probes/alpha_star_bisection.csv", index=False)
    print("\nUpdated alpha_star_bisection.csv:")
    print(df_bisect.to_string(index=False))
    return df_bisect


def plot_figure_3(df_sweep, df_bisect):
    print("\n=== 3. GENERATING FIGURE 3 (report/figures/alpha_star_experiment.pdf) ===")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left Panel: Predicted vs Observed Alpha*
    ax = axes[0]
    C_dense = np.logspace(np.log10(20), np.log10(1000), 100)
    tau1 = 10.0
    beta1 = 0.9
    s_dense = tau1 * np.log((1.0 - beta1) * C_dense + 1.0)
    alpha_pred_dense = 2.0 * np.sqrt(C_dense) / (C_dense - s_dense)
    
    ax.plot(C_dense, alpha_pred_dense, "k--", label=r"Predicted: $\alpha^*(C) = \frac{2\sqrt{C}}{C - s^*(C)}$", linewidth=1.5)
    
    # Plot bisected points
    for _, row in df_bisect.iterrows():
        if pd.notna(row["alpha_star_observed"]):
            c_val = int(row["C"])
            ax.errorbar(
                row["C"], row["alpha_star_observed"],
                yerr=row["half_interval"],
                fmt="s", color="tab:red", markersize=8, capsize=4,
                label=rf"Observed $\alpha^*$ ($C={c_val}$)",
            )
            
    ax.annotate(r"$C=30$: survives at $\beta_2=0.999$", xy=(30, 0.68), xytext=(22, 0.85),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate(r"Validity condition: $k \ll 1$ ($\tau_2 \gg C$)", xy=(150, 0.45),
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
    ax.set_xscale("log")
    ax.set_xlabel("Burst Scale $C$", fontsize=11)
    ax.set_ylabel(r"Traverse Threshold $\alpha^*(C)$", fontsize=11)
    ax.set_title(r"(a) Predicted vs. Observed $\alpha^*$ ($\beta_2=0.999$)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_ylim(0.0, 1.0)
    
    # Right Panel: Canonical Alpha Sweep at C=100
    ax2 = axes[1]
    alphas = df_sweep["alpha"].values
    xbars = df_sweep["x_bar_800"].values
    
    ax2.plot(alphas, xbars, "b.-", linewidth=1.5, label=r"Canonical $\bar{x}$ ($T_{\rm cycles}=800$)")
    ax2.axhline(0, color="k", linestyle="--", alpha=0.6, label=r"$\bar{x}=0$ boundary")
    
    # Find crossing
    alpha_star_100 = df_bisect[df_bisect["C"] == 100]["alpha_star_observed"].iloc[0]
    alpha_pred_100 = df_bisect[df_bisect["C"] == 100]["alpha_star_predicted"].iloc[0]
    
    ax2.axvline(alpha_star_100, color="tab:red", linestyle=":", linewidth=1.5, label=rf"Observed $\alpha^* = {alpha_star_100:.3f}$")
    ax2.axvline(alpha_pred_100, color="navy", linestyle=":", linewidth=1.5, label=rf"Predicted $\alpha^* = {alpha_pred_100:.3f}$")
    
    # Shade the actual negative survival region (where x_bar < 0)
    survival_mask = (xbars < 0)
    if np.any(survival_mask):
        a_surv_min = alphas[survival_mask].min()
        a_surv_max = alphas[survival_mask].max()
        label_str = r"Survival window ($\bar{x} < 0$, $\alpha \in [" + f"{a_surv_min:.2f}, {a_surv_max:.2f}" + r"]$)"
        ax2.axvspan(a_surv_min, a_surv_max, color="tab:green", alpha=0.15, label=label_str)
        
    ax2.set_xlabel(r"Learning Rate $\alpha$", fontsize=11)
    ax2.set_ylabel(r"Cycle Mean $\bar{x}$", fontsize=11)
    ax2.set_title(r"(b) $C=100, \beta_2=0.999$: Cycle Mean vs. $\alpha$", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    fig.tight_layout()
    for p in ["results/mechanism_probes/alpha_star_experiment.pdf",
              "results/mechanism_probes/alpha_star_experiment.png",
              "report/figures/alpha_star_experiment.pdf",
              "report/figures/alpha_star_experiment.png"]:
        fig.savefig(p, dpi=300 if p.endswith(".png") else None)
    plt.close(fig)
    print("Figure 3 regenerated and saved.")


if __name__ == "__main__":
    df_sw = run_canonical_alpha_sweep_C100()
    df_bi = run_alpha_star_bisection()
    plot_figure_3(df_sw, df_bi)
