"""
Phase 2 Experiment: 2D Toy Optimization Benchmarks.
Evaluates SGD, SGD (Momentum), AdaGrad, RMSProp, Adam, and AMSGrad on:
- Ill-conditioned Quadratic Bowl (kappa = 50)
- Rosenbrock Banana Valley
- Himmelblau (bounded multi-modal non-convex)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers import SGD, AdaGrad, RMSProp, Adam, AMSGrad
from src.benchmarks.toy_functions import QuadraticBowl, Rosenbrock, Himmelblau
from src.visualization.trajectory_plotter import plot_2d_trajectories
from src.visualization.convergence_curves import plot_loss_curves


def run_optimizer_on_landscape(opt_cls, opt_kwargs, landscape_obj, x0, steps=200):
    opt = opt_cls(**opt_kwargs)
    x = np.copy(x0)[None, :]  # Shape (1, 2)
    traj = [x[0].copy()]
    losses = [float(landscape_obj.evaluate(x)[0])]

    for _ in range(steps):
        grad = landscape_obj.grad(x)
        # Clip gradient for numerical safety on polynomial surfaces
        grad = np.clip(grad, -100.0, 100.0)
        x = opt.step(x, grad)
        traj.append(x[0].copy())
        losses.append(float(landscape_obj.evaluate(x)[0]))

    return np.array(traj), losses


def main():
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "report", "figures"))
    os.makedirs(figures_dir, exist_ok=True)

    print("Running Phase 2 Toy Benchmarks...")

    # 1. Quadratic Bowl
    quad_opts = {
        "SGD": (SGD, {"lr": 0.02}),
        "SGD (Momentum)": (SGD, {"lr": 0.02, "momentum": 0.9}),
        "AdaGrad": (AdaGrad, {"lr": 0.3}),
        "RMSProp": (RMSProp, {"lr": 0.05, "beta": 0.9}),
        "Adam": (Adam, {"lr": 0.1, "beta1": 0.9, "beta2": 0.999}),
        "AMSGrad": (AMSGrad, {"lr": 0.1, "beta1": 0.9, "beta2": 0.999}),
    }
    quad = QuadraticBowl(kappa=50.0)
    x0_quad = np.array([-2.5, 2.5])
    trajs_quad, losses_quad = {}, {}
    for name, (cls, kwargs) in quad_opts.items():
        tr, ls = run_optimizer_on_landscape(cls, kwargs, quad, x0_quad, steps=150)
        trajs_quad[name] = tr
        losses_quad[name] = ls

    plot_2d_trajectories(
        quad,
        trajs_quad,
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        title="2D Ill-Conditioned Quadratic Bowl Trajectories",
        save_path=os.path.join(figures_dir, "phase2_quadratic_trajectories.png"),
    )
    plot_loss_curves(
        losses_quad,
        title="Quadratic Bowl Loss vs Iteration",
        save_path=os.path.join(figures_dir, "phase2_quadratic_loss.png"),
    )

    # 2. Rosenbrock Banana
    rosen_opts = {
        "SGD": (SGD, {"lr": 0.001}),
        "SGD (Momentum)": (SGD, {"lr": 0.001, "momentum": 0.9}),
        "AdaGrad": (AdaGrad, {"lr": 0.1}),
        "RMSProp": (RMSProp, {"lr": 0.01, "beta": 0.9}),
        "Adam": (Adam, {"lr": 0.05, "beta1": 0.9, "beta2": 0.999}),
        "AMSGrad": (AMSGrad, {"lr": 0.05, "beta1": 0.9, "beta2": 0.999}),
    }
    rosen = Rosenbrock()
    x0_rosen = np.array([-1.2, 1.0])
    trajs_rosen, losses_rosen = {}, {}
    for name, (cls, kwargs) in rosen_opts.items():
        tr, ls = run_optimizer_on_landscape(cls, kwargs, rosen, x0_rosen, steps=300)
        trajs_rosen[name] = tr
        losses_rosen[name] = ls

    plot_2d_trajectories(
        rosen,
        trajs_rosen,
        x_range=(-2.0, 2.0),
        y_range=(-1.0, 3.0),
        title="Rosenbrock Banana Trajectories",
        save_path=os.path.join(figures_dir, "phase2_rosenbrock_trajectories.png"),
    )
    plot_loss_curves(
        losses_rosen,
        title="Rosenbrock Loss vs Iteration",
        save_path=os.path.join(figures_dir, "phase2_rosenbrock_loss.png"),
    )

    # 3. Himmelblau
    himmel_opts = {
        "SGD": (SGD, {"lr": 0.01}),
        "SGD (Momentum)": (SGD, {"lr": 0.01, "momentum": 0.9}),
        "AdaGrad": (AdaGrad, {"lr": 0.2}),
        "RMSProp": (RMSProp, {"lr": 0.05, "beta": 0.9}),
        "Adam": (Adam, {"lr": 0.1, "beta1": 0.9, "beta2": 0.999}),
        "AMSGrad": (AMSGrad, {"lr": 0.1, "beta1": 0.9, "beta2": 0.999}),
    }
    himmel = Himmelblau()
    x0_himmel = np.array([0.0, 0.0])
    trajs_himmel, losses_himmel = {}, {}
    for name, (cls, kwargs) in himmel_opts.items():
        tr, ls = run_optimizer_on_landscape(cls, kwargs, himmel, x0_himmel, steps=250)
        trajs_himmel[name] = tr
        losses_himmel[name] = ls

    plot_2d_trajectories(
        himmel,
        trajs_himmel,
        x_range=(-5.0, 5.0),
        y_range=(-5.0, 5.0),
        title="Himmelblau Bounded Non-Convex Trajectories",
        save_path=os.path.join(figures_dir, "phase2_himmelblau_trajectories.png"),
    )
    plot_loss_curves(
        losses_himmel,
        title="Himmelblau Loss vs Iteration",
        save_path=os.path.join(figures_dir, "phase2_himmelblau_loss.png"),
    )

    print("Phase 2 Toy Benchmarks finished successfully. Figures saved to report/figures/.")


if __name__ == "__main__":
    main()
