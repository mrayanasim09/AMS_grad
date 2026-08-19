"""
Convergence Curves Plotting (Loss, x_t position, Cumulative Regret).
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    loss_histories: Dict[str, List[float]],
    title: str = "Objective Loss vs Iteration",
    save_path: Optional[str] = None,
    log_scale: bool = True,
):
    plt.figure(figsize=(8, 5))
    for name, losses in loss_histories.items():
        iters = np.arange(1, len(losses) + 1)
        clean_losses = np.array(losses, dtype=np.float64)
        clean_losses = np.nan_to_num(clean_losses, nan=1e6, posinf=1e6)
        if log_scale:
            plt.semilogy(iters, np.maximum(clean_losses, 1e-8), label=name, linewidth=2)
        else:
            plt.plot(iters, clean_losses, label=name, linewidth=2)

    plt.xlabel("Iteration $t$", fontsize=12)
    plt.ylabel("Loss $f(x_t)$" + (" (log scale)" if log_scale else ""), fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_counterexample_trajectories(
    trajectories: Dict[str, np.ndarray],
    title: str = "1D Trajectory $x_t$ on Reddi Counterexample",
    save_path: Optional[str] = None,
):
    """
    Plots x_t over time for 1D counterexample showing divergence (+1) vs convergence (-1).
    """
    plt.figure(figsize=(9, 5))
    plt.axhline(1.0, color="r", linestyle=":", label="Suboptimal boundary (+1)")
    plt.axhline(-1.0, color="g", linestyle=":", label="Optimal boundary $x^*$ (-1)")

    for name, traj in trajectories.items():
        traj_arr = np.squeeze(traj) # Ensure shape is (T,) or (T, n_seeds)
        t = np.arange(1, len(traj_arr) + 1)
        if traj_arr.ndim > 1:
            mean = np.mean(traj_arr, axis=1)
            std = np.std(traj_arr, axis=1)
            p = plt.plot(t, mean, label=name, linewidth=2)
            plt.fill_between(t, mean - std, mean + std, alpha=0.15, color=p[0].get_color())
        else:
            plt.plot(t, traj_arr, label=name, linewidth=2)

    plt.ylim(-1.15, 1.15)
    plt.xlabel("Iteration $t$", fontsize=12)
    plt.ylabel("Parameter $x_t$", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center right", framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_regret_curves(
    regret_histories: Dict[str, np.ndarray],
    title: str = "Cumulative Regret $R_t$ vs Iteration",
    save_path: Optional[str] = None,
    log_log: bool = True,
):
    plt.figure(figsize=(8, 5))
    for name, R_t in regret_histories.items():
        R_arr = np.squeeze(R_t)
        if R_arr.ndim > 1:
            R_arr = np.mean(R_arr, axis=1)
        t = np.arange(1, len(R_arr) + 1)
        if log_log:
            plt.loglog(t, np.maximum(R_arr, 1e-6), label=name, linewidth=2)
        else:
            plt.plot(t, R_arr, label=name, linewidth=2)

    plt.xlabel("Iteration $t$", fontsize=12)
    plt.ylabel("Cumulative Regret $R_t$", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
