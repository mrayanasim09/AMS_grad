"""
Deterministic Counterexample Tests (Theorem 3 setting).
Guarantees:
1. Adam fails (x_T > 0.8) under beta2=0.5.
2. Adam control test: Adam converges to negative domain (x_T < -0.7) under beta2=0.999 (validates no sign errors).
3. AMSGrad succeeds (x_T < -0.8) under beta2=0.5.
4. AdaGrad succeeds (x_T < -0.8).
"""

import pytest
import numpy as np
from src.optimizers import Adam, AMSGrad, AdaGrad
from src.benchmarks.counterexample import (
    project_box_1d,
    DeterministicCounterexample,
)


DETERMINISTIC_TEST_CONFIG = {
    "C": 10,
    "T": 500,
    "x1": 0.0,
    "beta1": 0.9,
    "beta2_fail": 0.5,
    "beta2_control": 0.999,
    "alpha": 0.8,
}


def run_deterministic(optimizer_cls, opt_kwargs):
    cfg = DETERMINISTIC_TEST_CONFIG
    env = DeterministicCounterexample(C=cfg["C"], n_seeds=1)
    opt = optimizer_cls(projection_fn=project_box_1d, **opt_kwargs)
    
    x = np.full((1, 1), cfg["x1"], dtype=np.float64)
    for t in range(1, cfg["T"] + 1):
        g = env.get_gradient(t)
        x = opt.step(x, g)
    return float(x[0, 0])


def test_adam_deterministic_failure():
    """Under beta2=0.5 (tau2=2 << 10), Adam must diverge to suboptimal boundary x_T > 0.8."""
    cfg = DETERMINISTIC_TEST_CONFIG
    x_T = run_deterministic(
        Adam,
        {"lr": cfg["alpha"], "beta1": cfg["beta1"], "beta2": cfg["beta2_fail"]},
    )
    assert x_T > 0.8, f"Adam failed to diverge to +1: x_T = {x_T}"


def test_adam_deterministic_control():
    """Under beta2=0.999 (tau2=1000 >> 10), Adam must survive and converge to negative domain x_T < -0.7."""
    cfg = DETERMINISTIC_TEST_CONFIG
    x_T = run_deterministic(
        Adam,
        {"lr": cfg["alpha"], "beta1": cfg["beta1"], "beta2": cfg["beta2_control"]},
    )
    assert x_T < -0.7, f"Adam control test failed (expected x_T < -0.7, got {x_T})"


def test_amsgrad_deterministic_convergence():
    """Under beta2=0.5, AMSGrad (raw Algorithm 2) must converge to x_T < -0.8."""
    cfg = DETERMINISTIC_TEST_CONFIG
    x_T = run_deterministic(
        AMSGrad,
        {
            "lr": cfg["alpha"],
            "beta1": cfg["beta1"],
            "beta2": cfg["beta2_fail"],
            "bias_correction": False,
        },
    )
    assert x_T < -0.8, f"AMSGrad failed to converge to -1: x_T = {x_T}"


def test_adagrad_deterministic_convergence():
    """AdaGrad must converge to x_T < -0.8."""
    cfg = DETERMINISTIC_TEST_CONFIG
    x_T = run_deterministic(
        AdaGrad,
        {"lr": cfg["alpha"]},
    )
    assert x_T < -0.8, f"AdaGrad failed to converge to -1: x_T = {x_T}"
