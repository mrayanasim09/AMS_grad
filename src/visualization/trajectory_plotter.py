"""
2D Contour Trajectory Plotting for Toy Optimization Landscapes.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_trajectories(
    landscape_obj,
    trajectories: Dict[str, np.ndarray],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    title: str = "Optimizer Trajectories",
    save_path: str = None,
    grid_points: int = 200,
):
    """
    Plot 2D contour lines of objective landscape with overlaid optimization paths.
    """
    x = np.linspace(x_range[0], x_range[1], grid_points)
    y = np.linspace(y_range[0], y_range[1], grid_points)
    X, Y = np.meshgrid(x, y)
    
    XY = np.stack([X, Y], axis=-1)
    Z = landscape_obj.evaluate(XY)

    plt.figure(figsize=(9, 7))
    
    # Log contour for steep functions
    levels = np.logspace(0, 3.5, 35) if np.max(Z) > 100 else 30
    plt.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.6)

    colors = {
        "SGD": "#1f77b4",
        "SGD (Momentum)": "#ff7f0e",
        "AdaGrad": "#2ca02c",
        "RMSProp": "#d62728",
        "Adam": "#9467bd",
        "AMSGrad": "#8c564b",
    }

    for name, traj in trajectories.items():
        color = colors.get(name, None)
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            markersize=3,
            linewidth=2,
            label=name,
            color=color,
            alpha=0.85,
        )
        # Start and end markers
        plt.plot(traj[0, 0], traj[0, 1], "ko", markersize=6)
        plt.plot(traj[-1, 0], traj[-1, 1], "r*", markersize=10)

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(r"$x_1$", fontsize=12)
    plt.ylabel(r"$x_2$", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
