"""
Canonical Measurement Module for Deterministic Periodic Counterexamples.
Provides a single canonical measurement function for cycle-averaged metrics across all scripts,
experiments, notebooks, and tests.
"""

from typing import Tuple
import numpy as np
from src.benchmarks.counterexample import DeterministicCounterexample, project_box_1d
from src.optimizers import Adam


def measure_cycle_mean(
    C: int,
    alpha: float,
    beta2: float,
    T_cycles: int = 400,
    beta1: float = 0.9,
) -> Tuple[float, float]:
    """
    Measure cycle-averaged position x_bar and dwell fraction on final C-step cycle.

    Parameters
    ----------
    C : int
        Burst scale / cycle period.
    alpha : float
        Learning rate.
    beta2 : float
        Second-moment exponential moving average parameter.
    T_cycles : int
        Total number of cycles to run (warmup = T_cycles - 1, measurement = final cycle).
    beta1 : float
        First-moment exponential moving average parameter (default 0.9).

    Returns
    -------
    x_bar : float
        Mean coordinate x_t over the final C steps of the simulation.
    dwell : float
        Fraction of steps in the final C-step cycle where x_t > 0.5.
    """
    env = DeterministicCounterexample(C=C, n_seeds=1)
    opt = Adam(lr=alpha, beta1=beta1, beta2=beta2, projection_fn=project_box_1d)
    x = np.zeros((1, 1), dtype=np.float64)

    # Warmup phase: run for (T_cycles - 1) full cycles of length C
    T_warmup = (T_cycles - 1) * C
    for t in range(1, T_warmup + 1):
        g = env.get_gradient(t)
        x = opt.step(x, g)

    # Measurement phase: collect trajectory over the final C-step cycle
    cycle_positions = []
    for step in range(1, C + 1):
        t = T_warmup + step
        g = env.get_gradient(t)
        x = opt.step(x, g)
        cycle_positions.append(float(x[0, 0]))

    x_bar = float(np.mean(cycle_positions))
    dwell = float(np.mean(np.array(cycle_positions) > 0.5))
    return x_bar, dwell
