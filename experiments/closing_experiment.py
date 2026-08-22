"""
Closing Experiment: Stochastic Theory Resolution (Experiment 1).
Validates constant vs decreasing step-size dynamics for Adam and AMSGrad.
Generates 2x2 panel figure and saves source CSVs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.benchmarks.counterexample import StochasticCounterexample, project_box_1d
from src.optimizers import Adam, AMSGrad

os.makedirs("results/mechanism_probes", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)


def run_closing_experiment():
    print("=== 1. RUNNING CLOSING EXPERIMENT (C=20, delta=0.5, T=5000, N=100) ===")
    C = 20
    delta = 0.5
    T = 5000
    N = 100
    base_seed = 42

    # Environment instances
    env_c = StochasticCounterexample(C=C, delta=delta, n_seeds=N, base_seed=base_seed)
    env_d = StochasticCounterexample(C=C, delta=delta, n_seeds=N, base_seed=base_seed)

    # Optimizers
    opt_adam_c = Adam(lr=0.8, beta1=0.9, beta2=0.5, projection_fn=project_box_1d)
    opt_ams_c = AMSGrad(lr=0.8, beta1=0.9, beta2=0.5, bias_correction=False, projection_fn=project_box_1d)

    opt_adam_d = Adam(lr=lambda t: 0.8 / np.sqrt(t), beta1=0.9, beta2=0.5, projection_fn=project_box_1d)
    opt_ams_d = AMSGrad(lr=lambda t: 0.8 / np.sqrt(t), beta1=0.9, beta2=0.5, bias_correction=False, projection_fn=project_box_1d)

    # State tracking
    x_adam_c = np.zeros((N, 1))
    x_ams_c = np.zeros((N, 1))
    x_adam_d = np.zeros((N, 1))
    x_ams_d = np.zeros((N, 1))

    # Record trajectory samples for seeds 0..4
    traj_adam_c = []
    traj_ams_c = []
    traj_adam_d = []
    traj_ams_d = []

    for t in range(1, T + 1):
        g_c = env_c.get_gradient(t)
        g_d = env_d.get_gradient(t)

        x_adam_c = opt_adam_c.step(x_adam_c, g_c)
        x_ams_c = opt_ams_c.step(x_ams_c, g_c)
        x_adam_d = opt_adam_d.step(x_adam_d, g_d)
        x_ams_d = opt_ams_d.step(x_ams_d, g_d)

        if t % 5 == 0:
            traj_adam_c.append(x_adam_c[:5, 0].copy())
            traj_ams_c.append(x_ams_c[:5, 0].copy())
            traj_adam_d.append(x_adam_d[:5, 0].copy())
            traj_ams_d.append(x_ams_d[:5, 0].copy())

    df_dist = pd.DataFrame({
        "seed": np.arange(N),
        "adam_const_xT": x_adam_c[:, 0],
        "amsgrad_const_xT": x_ams_c[:, 0],
        "adam_dec_xT": x_adam_d[:, 0],
        "amsgrad_dec_xT": x_ams_d[:, 0],
    })
    df_dist.to_csv("results/mechanism_probes/closing_experiment_distributions.csv", index=False)
    print("Saved results/mechanism_probes/closing_experiment_distributions.csv")

    print(f"Adam (constant):     mean = {x_adam_c.mean():+.4f}, std = {x_adam_c.std():.4f}, frac(>0.5) = {np.mean(x_adam_c > 0.5):.2f}")
    print(f"AMSGrad (constant):  mean = {x_ams_c.mean():+.4f}, std = {x_ams_c.std():.4f}, frac(<-0.5) = {np.mean(x_ams_c < -0.5):.2f}")
    print(f"Adam (decreasing):   mean = {x_adam_d.mean():+.4f}, std = {x_adam_d.std():.4f}, frac(>0.5) = {np.mean(x_adam_d > 0.5):.2f}")
    print(f"AMSGrad (decreasing):mean = {x_ams_d.mean():+.4f}, std = {x_ams_d.std():.4f}, frac(<-0.9) = {np.mean(x_ams_d < -0.9):.2f}")

    return (
        df_dist,
        np.array(traj_adam_c),
        np.array(traj_ams_c),
        np.array(traj_adam_d),
        np.array(traj_ams_d),
    )


def run_robustness_check_1e6():
    print("\n=== 2. ROBUSTNESS CHECK (delta=0.1, T=10^6, N=20) ===")
    C = 20
    delta = 0.1
    T = 1000000
    N = 20

    env = StochasticCounterexample(C=C, delta=delta, n_seeds=N, base_seed=123)
    opt_ams = AMSGrad(lr=lambda t: 0.8 / np.sqrt(t), beta1=0.9, beta2=0.5, bias_correction=False, projection_fn=project_box_1d)
    opt_adam = Adam(lr=lambda t: 0.8 / np.sqrt(t), beta1=0.9, beta2=0.5, projection_fn=project_box_1d)

    x_ams = np.zeros((N, 1))
    x_adam = np.zeros((N, 1))

    chunk_size = 50000
    for chunk_start in range(1, T + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, T + 1)
        for t in range(chunk_start, chunk_end):
            g = env.get_gradient(t)
            x_ams = opt_ams.step(x_ams, g)
            x_adam = opt_adam.step(x_adam, g)

    df_rob = pd.DataFrame({
        "seed": np.arange(N),
        "amsgrad_1e6_xT": x_ams[:, 0],
        "adam_1e6_xT": x_adam[:, 0],
    })
    df_rob.to_csv("results/mechanism_probes/robustness_check_1e6.csv", index=False)
    print(f"T=10^6: AMSGrad mean = {x_ams.mean():.4f}, Adam mean = {x_adam.mean():.4f}")
    return df_rob


def plot_2x2_figure(df_dist, traj_ac, traj_amsc, traj_ad, traj_amsd):
    print("\n=== 3. GENERATING 2x2 PANEL FIGURE ===")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # Top-Left: Constant Step Size Distribution
    ax = axes[0, 0]
    bins = np.linspace(-1.05, 1.05, 22)
    ax.hist(df_dist["adam_const_xT"], bins=bins, alpha=0.6, color="tab:red", label=rf"Adam ($\mathrm{{mean}}={df_dist['adam_const_xT'].mean():+.2f}$)")
    ax.hist(df_dist["amsgrad_const_xT"], bins=bins, alpha=0.6, color="tab:blue", label=rf"AMSGrad ($\mathrm{{mean}}={df_dist['amsgrad_const_xT'].mean():+.2f}$)")
    ax.set_title(r"(a) Constant Step Size ($\alpha=0.8$): Diffuse Stationary Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Terminal Position $x_T$", fontsize=9)
    ax.set_ylabel("Seed Count", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Top-Right: Decreasing Step Size Distribution
    ax = axes[0, 1]
    ax.hist(df_dist["adam_dec_xT"], bins=bins, alpha=0.6, color="tab:red", label=rf"Adam ($\mathrm{{mean}}={df_dist['adam_dec_xT'].mean():+.2f}$)")
    ax.hist(df_dist["amsgrad_dec_xT"], bins=bins, alpha=0.6, color="tab:blue", label=rf"AMSGrad ($\mathrm{{mean}}={df_dist['amsgrad_dec_xT'].mean():+.2f}$)")
    ax.set_title(r"(b) Decreasing Step Size ($\alpha_t = 0.8/\sqrt{t}$): Asymptotic Convergence", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Terminal Position $x_T$", fontsize=9)
    ax.set_ylabel("Seed Count", fontsize=9)
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Bottom-Left: Sample Trajectories Constant
    ax = axes[1, 0]
    t_steps = np.arange(5, 5001, 5)
    for s in range(3):
        ax.plot(t_steps, traj_ac[:, s], color="tab:red", alpha=0.7, linewidth=1.2, label="Adam" if s == 0 else "")
        ax.plot(t_steps, traj_amsc[:, s], color="tab:blue", alpha=0.7, linewidth=1.2, label="AMSGrad" if s == 0 else "")
    ax.set_title(r"(c) Trajectories ($\alpha=0.8$): Persistent Diffusion on $[-1, 1]$", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Iteration $t$", fontsize=9)
    ax.set_ylabel(r"Coordinate $x_t$", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Bottom-Right: Sample Trajectories Decreasing
    ax = axes[1, 1]
    for s in range(3):
        ax.plot(t_steps, traj_ad[:, s], color="tab:red", alpha=0.7, linewidth=1.2, label="Adam" if s == 0 else "")
        ax.plot(t_steps, traj_amsd[:, s], color="tab:blue", alpha=0.7, linewidth=1.2, label="AMSGrad" if s == 0 else "")
    ax.set_title(r"(d) Trajectories ($\alpha_t = 0.8/\sqrt{t}$): AMSGrad Freezes at $x^*=-1$", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Iteration $t$", fontsize=9)
    ax.set_ylabel(r"Coordinate $x_t$", fontsize=9)
    ax.legend(fontsize=8, loc="center right")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    for path in [
        "results/mechanism_probes/stochastic_theory_resolution.pdf",
        "results/mechanism_probes/stochastic_theory_resolution.png",
        "report/figures/stochastic_theory_resolution.pdf",
        "report/figures/stochastic_theory_resolution.png",
    ]:
        fig.savefig(path, dpi=300 if path.endswith(".png") else None)
    plt.close(fig)
    print("Saved 2x2 panel figure.")


if __name__ == "__main__":
    df_d, tac, tamsc, tad, tamsd = run_closing_experiment()
    df_r = run_robustness_check_1e6()
    plot_2x2_figure(df_d, tac, tamsc, tad, tamsd)
