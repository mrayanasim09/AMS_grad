"""
Phase 3 Experiment: Reproduction of Reddi et al. Counterexample & Gamma_t Dynamics.
Compares Adam, AMSGrad (raw & debiased), and AdaGrad on both Deterministic and Stochastic settings.
Generates:
1. x_t trajectory curves showing Adam diverging to +1 and AMSGrad/AdaGrad converging to -1.
2. Cumulative regret curves R_t.
3. Gamma_t metric time-series showing Adam dipping below zero.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers import Adam, AMSGrad, AdaGrad
from src.benchmarks.counterexample import (
    project_box_1d,
    DeterministicCounterexample,
    StochasticCounterexample,
)
from src.diagnostics import ConvergenceTracker
from src.visualization.convergence_curves import (
    plot_counterexample_trajectories,
    plot_regret_curves,
)
from src.visualization.diagnostics_plotter import plot_gamma_metric


def run_experiment(env, optimizer_cls, opt_kwargs, T=1000, x1=0.0):
    tracker = ConvergenceTracker(x_star=-1.0)
    opt = optimizer_cls(projection_fn=project_box_1d, **opt_kwargs)
    
    n_seeds = getattr(env, "n_seeds", 1)
    x = np.full((n_seeds, 1), x1, dtype=np.float64)

    for t in range(1, T + 1):
        g = env.get_gradient(t)
        
        # Determine internal v / v_effective before or after step
        x_next = opt.step(x, g)
        
        v_raw = opt.state.get("v", g ** 2)
        v_eff = opt.state.get("v_effective", opt.state.get("v_tilde", opt.state.get("G", v_raw)))
        lr_val = opt.get_lr(t)
        
        tracker.record(x, g, v_raw, v_eff, lr_val)
        x = x_next

    return tracker


def main():
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "report", "figures"))
    os.makedirs(figures_dir, exist_ok=True)

    print("Running Phase 3 Counterexample Reproduction...")

    # --- 1. DETERMINISTIC REPRODUCTION ---
    C_det = 10
    T_det = 500
    alpha = 0.8
    env_det = DeterministicCounterexample(C=C_det, n_seeds=1)

    opt_configs = {
        "Adam (beta2=0.5, fail)": (Adam, {"lr": alpha, "beta1": 0.9, "beta2": 0.5}),
        "Adam (beta2=0.999, control)": (Adam, {"lr": alpha, "beta1": 0.9, "beta2": 0.999}),
        "AMSGrad (raw)": (AMSGrad, {"lr": alpha, "beta1": 0.9, "beta2": 0.5, "bias_correction": False}),
        "AdaGrad": (AdaGrad, {"lr": alpha}),
    }

    det_trajs = {}
    det_regrets = {}
    det_gammas = {}

    for name, (cls, kwargs) in opt_configs.items():
        tracker = run_experiment(env_det, cls, kwargs, T=T_det, x1=0.0)
        det_trajs[name] = tracker.get_trajectory()
        det_regrets[name] = tracker.compute_cumulative_regret()
        det_gammas[name] = tracker.compute_gamma()

    plot_counterexample_trajectories(
        det_trajs,
        title=f"Deterministic Counterexample (C={C_det}, alpha={alpha})",
        save_path=os.path.join(figures_dir, "phase3_deterministic_trajectories.png"),
    )
    plot_regret_curves(
        det_regrets,
        title=f"Deterministic Cumulative Regret (C={C_det})",
        save_path=os.path.join(figures_dir, "phase3_deterministic_regret.png"),
    )
    plot_gamma_metric(
        det_gammas,
        title=r"Deterministic $\Gamma_t$ Dynamics (First 100 steps)",
        save_path=os.path.join(figures_dir, "phase3_deterministic_gamma.png"),
        max_steps=100,
    )

    # --- 2. STOCHASTIC REPRODUCTION ---
    C_stoch = 20
    delta_stoch = 0.05
    T_stoch = 3000
    n_seeds = 30
    env_stoch = StochasticCounterexample(C=C_stoch, delta=delta_stoch, n_seeds=n_seeds, base_seed=42)

    stoch_trajs = {}
    stoch_regrets = {}

    for name, (cls, kwargs) in opt_configs.items():
        tracker = run_experiment(env_stoch, cls, kwargs, T=T_stoch, x1=0.0)
        stoch_trajs[name] = tracker.get_trajectory()
        stoch_regrets[name] = tracker.compute_cumulative_regret()

    plot_counterexample_trajectories(
        stoch_trajs,
        title=f"Stochastic Counterexample (C={C_stoch}, delta={delta_stoch}, N={n_seeds} seeds)",
        save_path=os.path.join(figures_dir, "phase3_stochastic_trajectories.png"),
    )
    plot_regret_curves(
        stoch_regrets,
        title=f"Stochastic Cumulative Regret (C={C_stoch}, delta={delta_stoch})",
        save_path=os.path.join(figures_dir, "phase3_stochastic_regret.png"),
    )

    print("Phase 3 Counterexample Reproduction completed successfully. Figures generated.")


if __name__ == "__main__":
    main()
