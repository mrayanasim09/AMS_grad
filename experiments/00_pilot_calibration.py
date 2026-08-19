"""
Phase 3 Pilot Calibration Script (Expanded analysis).
Investigates T in {500, 1000, 2000, 5000, 10000} and beta2 values.
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


def test_deterministic(T_vals=[500, 750, 1000]):
    print("\n=== DETERMINISTIC COUNTEREXAMPLE CALIBRATION (C=10, alpha=0.8, x1=0) ===")
    for T in T_vals:
        print(f"\n[ T = {T} ]")
        env = DeterministicCounterexample(C=10, n_seeds=1)
        
        # Adam fail (beta2=0.5)
        opt = Adam(lr=0.8, beta1=0.9, beta2=0.5, projection_fn=project_box_1d)
        x = np.zeros((1, 1))
        for t in range(1, T + 1):
            x = opt.step(x, env.get_gradient(t))
        print(f"  Adam (beta2=0.5)    : x_T = {float(x[0, 0]):+.4f}")

        # Adam control (beta2=0.999)
        opt = Adam(lr=0.8, beta1=0.9, beta2=0.999, projection_fn=project_box_1d)
        x = np.zeros((1, 1))
        for t in range(1, T + 1):
            x = opt.step(x, env.get_gradient(t))
        print(f"  Adam (beta2=0.999)  : x_T = {float(x[0, 0]):+.4f}")

        # AMSGrad (beta2=0.5)
        opt = AMSGrad(lr=0.8, beta1=0.9, beta2=0.5, bias_correction=False, projection_fn=project_box_1d)
        x = np.zeros((1, 1))
        for t in range(1, T + 1):
            x = opt.step(x, env.get_gradient(t))
        print(f"  AMSGrad (beta2=0.5) : x_T = {float(x[0, 0]):+.4f}")

        # AdaGrad
        opt = AdaGrad(lr=0.8, projection_fn=project_box_1d)
        x = np.zeros((1, 1))
        for t in range(1, T + 1):
            x = opt.step(x, env.get_gradient(t))
        print(f"  AdaGrad             : x_T = {float(x[0, 0]):+.4f}")


def test_stochastic(T_vals=[3000, 6000, 10000], deltas=[0.05, 0.1]):
    print("\n=== STOCHASTIC COUNTEREXAMPLE CALIBRATION (C=20, alpha=0.8, N=50 seeds) ===")
    for delta in deltas:
        for T in T_vals:
            print(f"\n[ delta = {delta}, T = {T} ]")
            env = StochasticCounterexample(C=20, delta=delta, n_seeds=50, base_seed=42)
            
            # Adam fail (beta2=0.5)
            opt_adam = Adam(lr=0.8, beta1=0.9, beta2=0.5, projection_fn=project_box_1d)
            x_adam = np.zeros((50, 1))
            for t in range(1, T + 1):
                x_adam = opt_adam.step(x_adam, env.get_gradient(t))
            x_a = x_adam[:, 0]
            print(f"  Adam (beta2=0.5)    : mean={np.mean(x_a):+.4f} +/- {np.std(x_a):.4f}, fail_frac={np.mean(x_a > 0.5):.2f}, fraction_pos={np.mean(x_a > 0.0):.2f}")

            # AMSGrad (beta2=0.5)
            opt_ams = AMSGrad(lr=0.8, beta1=0.9, beta2=0.5, bias_correction=False, projection_fn=project_box_1d)
            x_ams = np.zeros((50, 1))
            for t in range(1, T + 1):
                x_ams = opt_ams.step(x_ams, env.get_gradient(t))
            x_m = x_ams[:, 0]
            print(f"  AMSGrad (beta2=0.5) : mean={np.mean(x_m):+.4f} +/- {np.std(x_m):.4f}, fraction_neg={np.mean(x_m < 0.0):.2f}")

            # AdaGrad
            opt_ada = AdaGrad(lr=0.8, projection_fn=project_box_1d)
            x_ada = np.zeros((50, 1))
            for t in range(1, T + 1):
                x_ada = opt_ada.step(x_ada, env.get_gradient(t))
            x_d = x_ada[:, 0]
            print(f"  AdaGrad             : mean={np.mean(x_d):+.4f} +/- {np.std(x_d):.4f}, fraction_neg={np.mean(x_d < 0.0):.2f}")


if __name__ == "__main__":
    test_deterministic([500, 750, 1000])
    test_stochastic([3000, 6000, 10000], [0.05, 0.1])
