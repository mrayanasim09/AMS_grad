"""
Diagnostics & Phase Boundary Plotting.
Includes Gamma_t time series, (beta_2, C) phase transition heatmaps, and rho-data collapse plots.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_gamma_metric(
    gamma_dict: Dict[str, np.ndarray],
    title: str = r"OCO Metric $\Gamma_t = \frac{\sqrt{v_t}}{\alpha_t} - \frac{\sqrt{v_{t-1}}}{\alpha_{t-1}}$",
    save_path: Optional[str] = None,
    max_steps: int = 150,
):
    """
    Plots Gamma_t showing Adam dipping into negative territory (violating Gamma_t >= 0)
    while AdaGrad and AMSGrad remain strictly non-negative.
    """
    plt.figure(figsize=(9, 5))
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.7)

    for name, gamma in gamma_dict.items():
        if gamma.ndim > 1:
            gamma = gamma.mean(axis=tuple(range(1, gamma.ndim)))
        t = np.arange(1, min(len(gamma), max_steps) + 1)
        plt.plot(t, gamma[:max_steps], label=name, linewidth=2, alpha=0.85)

    plt.xlabel("Iteration $t$", fontsize=12)
    plt.ylabel(r"$\Gamma_t$", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_phase_boundary_heatmap(
    C_vals: List[int],
    k_vals: List[float],
    failure_matrix: np.ndarray,
    title: str = r"Adam Phase Boundary in $(k, C)$ Space [$k = (1-\beta_2)C$]",
    save_path: Optional[str] = None,
):
    """
    Heatmap of failure fraction across C and k.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(
        failure_matrix,
        origin="lower",
        cmap="coolwarm",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = plt.colorbar()
    cbar.set_label("Failure Fraction ($x_T > 0.5$)", fontsize=12)

    plt.xticks(np.arange(len(C_vals)), [str(c) for c in C_vals])
    plt.yticks(np.arange(len(k_vals)), [f"{k:.2f}" for k in k_vals])
    
    plt.xlabel("Burst Scale $C$", fontsize=12)
    plt.ylabel(r"Dimensionless Parameter $k = (1 - \beta_2) C$", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_data_collapse(
    rho_vals: np.ndarray,
    failure_fractions: np.ndarray,
    c_labels: np.ndarray,
    title: str = r"Universal Data Collapse vs $\rho = \frac{\tau_2}{T_{\rm burst}} = \frac{1+\delta}{(1-\beta_2)C}$",
    save_path: Optional[str] = None,
):
    """
    Plots failure fraction vs dimensionless ratio rho to test for universal collapse onto a sigmoidal curve.
    """
    plt.figure(figsize=(8, 5))
    unique_c = np.unique(c_labels)
    
    for c in unique_c:
        mask = (c_labels == c)
        plt.scatter(
            rho_vals[mask],
            failure_fractions[mask],
            label=f"$C = {c}$",
            alpha=0.8,
            s=50,
        )

    plt.xscale("log")
    plt.axvline(1.0, color="k", linestyle=":", label=r"Predicted transition $\rho^* \approx 1$")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel(r"Dimensionless Memory Ratio $\rho = \tau_2 / T_{\rm burst}$", fontsize=12)
    plt.ylabel("Failure Fraction ($x_T > 0.5$)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
