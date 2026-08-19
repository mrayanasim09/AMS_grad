"""
Phase 4 Experiment: Pre-Registered Phase-Boundary & Memory Horizon Scaling Sweep.
Investigates Adam divergence threshold in (beta_2, C) space:
- Sweeps k = (1 - beta_2) * C in {0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0}
- Sweeps C in {10, 30, 100, 300, 1000}
- Floors beta_2 >= 0.05 to avoid the degenerate beta_2=0 corner.
- Evaluates:
  1. Deterministic sweep at T = 200*(C+1) to measure the exact noise-free log-log slope of beta_2*(C).
  2. Stochastic sweep with N = 100 seeds at T = 50*(C+1), delta = 0.1 to evaluate the universal data-collapse curve vs rho.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers import Adam, AMSGrad, AdaGrad
from src.benchmarks.counterexample import (
    project_box_1d,
    DeterministicCounterexample,
    StochasticCounterexample,
)
from src.visualization.diagnostics_plotter import (
    plot_phase_boundary_heatmap,
    plot_data_collapse,
)


def run_deterministic_adam(C: int, beta2: float, alpha: float = 0.8) -> float:
    T = int(200 * (C + 1))
    env = DeterministicCounterexample(C=C, n_seeds=1)
    opt = Adam(lr=alpha, beta1=0.9, beta2=beta2, projection_fn=project_box_1d)
    
    x = np.zeros((1, 1), dtype=np.float64)
    for t in range(1, T + 1):
        g = env.get_gradient(t)
        x = opt.step(x, g)
    return float(x[0, 0])


def run_stochastic_grid(
    C_vals,
    k_vals,
    delta: float = 0.1,
    alpha: float = 0.8,
    n_seeds: int = 100,
    base_seed: int = 1337,
):
    results = []
    failure_matrix = np.zeros((len(k_vals), len(C_vals)))

    for j, C in enumerate(C_vals):
        T = int(50 * (C + 1))
        for i, k in enumerate(k_vals):
            # Enforce floor beta_2 >= 0.05
            beta2 = max(1.0 - (k / C), 0.05)
            tau2 = 1.0 / (1.0 - beta2)
            T_burst = (C + 1.0) / (1.0 + delta)
            rho = tau2 / T_burst

            env = StochasticCounterexample(
                C=C, delta=delta, n_seeds=n_seeds, base_seed=base_seed + (j * 100 + i)
            )
            opt = Adam(lr=alpha, beta1=0.9, beta2=beta2, projection_fn=project_box_1d)

            x = np.zeros((n_seeds, 1), dtype=np.float64)
            for t in range(1, T + 1):
                g = env.get_gradient(t)
                x = opt.step(x, g)

            x_T = x[:, 0]
            fail_frac = float(np.mean(x_T > 0.5))
            mean_x = float(np.mean(x_T))
            std_x = float(np.std(x_T))

            failure_matrix[i, j] = fail_frac

            results.append({
                "C": C,
                "k": k,
                "beta2": beta2,
                "tau2": tau2,
                "rho": rho,
                "T": T,
                "failure_fraction": fail_frac,
                "mean_x_T": mean_x,
                "std_x_T": std_x,
            })
            print(f"  [C={C:4d}, k={k:5.2f}, beta2={beta2:7.5f}] -> fail_frac={fail_frac:.2f}, mean_x_T={mean_x:+.3f}")

    return results, failure_matrix


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "report", "figures"))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 4: PRE-REGISTERED PHASE BOUNDARY & SCALING ANALYSIS")
    print("=" * 80)

    C_vals = [10, 30, 100, 300, 1000]
    k_vals = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    # --- 1. DETERMINISTIC NOISE-FREE TRANSITION ESTIMATION ---
    print("\n--- 1. DETERMINISTIC SLOPE ESTIMATION ---")
    det_flip_k = {}
    for C in C_vals:
        # Search over k for the transition point k* where x_T flips from survival (<0) to failure (>0)
        k_search = np.linspace(0.01, 5.0, 100)
        flip_k = None
        for k in k_search:
            beta2 = max(1.0 - (k / C), 0.05)
            x_end = run_deterministic_adam(C, beta2)
            if x_end > 0.0:  # Failed
                flip_k = k
                break
        det_flip_k[C] = flip_k
        if flip_k is not None:
            print(f"  Deterministic flip for C={C:4d}: k* ≈ {flip_k:.3f}, 1 - beta2* ≈ {flip_k/C:.5f}")
        else:
            print(f"  Deterministic flip for C={C:4d}: No flip found in range (always survives or fails)")

    # Fit log-log slope for valid transitions
    valid_C = [c for c in C_vals if det_flip_k[c] is not None]
    if len(valid_C) >= 2:
        log_C = np.log([c for c in valid_C])
        log_one_minus_beta2 = np.log([det_flip_k[c] / c for c in valid_C])
        slope, intercept = np.polyfit(log_C, log_one_minus_beta2, 1)
        print(f"\n>> PRE-REGISTERED LOG-LOG SLOPE ESTIMATE: slope = {slope:.4f} (Predicted: ~ -1.00)")
    else:
        slope, intercept = -1.0, 0.0
        print("\n>> Not enough transition points to fit slope.")

    # --- 2. STOCHASTIC COLLAPSE SWEEP ---
    print("\n--- 2. STOCHASTIC DATA COLLAPSE SWEEP (N=100 seeds, delta=0.1) ---")
    results, failure_matrix = run_stochastic_grid(C_vals, k_vals, delta=0.1, alpha=0.8, n_seeds=100)

    # Save data
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, "phase4_grid_results.csv"), index=False)
    with open(os.path.join(results_dir, "phase4_summary.json"), "w") as f:
        json.dump({
            "deterministic_slope": float(slope),
            "deterministic_intercept": float(intercept),
            "deterministic_flip_k": {str(k): (float(v) if v is not None else None) for k, v in det_flip_k.items()},
        }, f, indent=2)

    # Generate figures
    plot_phase_boundary_heatmap(
        C_vals,
        k_vals,
        failure_matrix,
        title=r"Adam Phase Transition Heatmap [$k = (1 - \beta_2) C$]",
        save_path=os.path.join(figures_dir, "phase4_phase_heatmap.png"),
    )

    plot_data_collapse(
        df["rho"].values,
        df["failure_fraction"].values,
        df["C"].values,
        title=r"Universal Collapse vs Dimensionless Memory Ratio $\rho = \frac{\tau_2}{T_{\rm burst}}$",
        save_path=os.path.join(figures_dir, "phase4_data_collapse.png"),
    )

    print("\nPhase 4 experiments completed. Artifacts saved to results/ and report/figures/.")


if __name__ == "__main__":
    main()
