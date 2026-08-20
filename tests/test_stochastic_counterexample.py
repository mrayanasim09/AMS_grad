"""
Stochastic Counterexample Tests (Section 3 setting).
Evaluates Adam divergence and AMSGrad/AdaGrad non-divergence across seeds.
Includes statistical self-test for the stochastic gradient generator.
"""

import pytest
import numpy as np
from src.optimizers import Adam, AMSGrad, AdaGrad
from src.benchmarks.counterexample import (
    project_box_1d,
    StochasticCounterexample,
)


STOCHASTIC_TEST_CONFIG = {
    "C": 20,
    "T": 5000,
    "x1": 0.0,
    "delta": 0.1,
    "beta1": 0.9,
    "beta2_fail": 0.5,
    "alpha": 0.8,
    "n_seeds": 50,
    "base_seed": 42,
}


def test_stochastic_environment_statistics():
    """Verify stochastic environment draws are unbiased (E[g] within 3-sigma) and seeds are independent."""
    cfg = STOCHASTIC_TEST_CONFIG
    n_seeds = 50
    n_steps = 2000  # Total 100,000 draws
    env = StochasticCounterexample(
        C=cfg["C"],
        delta=cfg["delta"],
        n_seeds=n_seeds,
        base_seed=cfg["base_seed"],
    )
    
    draws = np.array([env.get_gradient(t)[:, 0] for t in range(1, n_steps + 1)])  # (2000, 50)
    empirical_mean = float(np.mean(draws))
    
    p = (1.0 + cfg["delta"]) / (cfg["C"] + 1.0)
    theo_var = p * (cfg["C"] ** 2) + (1.0 - p) * 1.0 - (cfg["delta"] ** 2)
    std_err = np.sqrt(theo_var / draws.size)
    
    # 3-sigma gate
    assert abs(empirical_mean - cfg["delta"]) <= 3.0 * std_err, (
        f"Empirical E[g] = {empirical_mean:.4f} outside 3-sigma [{cfg['delta'] - 3*std_err:.4f}, {cfg['delta'] + 3*std_err:.4f}]"
    )
    
    # Pairwise cross-correlation check across seeds
    corrs = []
    for i in range(min(10, n_seeds)):
        for j in range(i + 1, min(10, n_seeds)):
            c = np.corrcoef(draws[:, i], draws[:, j])[0, 1]
            corrs.append(c)
    assert np.max(np.abs(corrs)) < 0.10, f"Seed cross-correlation too high: {np.max(np.abs(corrs)):.4f}"


def run_stochastic(optimizer_cls, opt_kwargs):
    cfg = STOCHASTIC_TEST_CONFIG
    env = StochasticCounterexample(
        C=cfg["C"],
        delta=cfg["delta"],
        n_seeds=cfg["n_seeds"],
        base_seed=cfg["base_seed"],
    )
    opt = optimizer_cls(projection_fn=project_box_1d, **opt_kwargs)
    
    x = np.full((cfg["n_seeds"], 1), cfg["x1"], dtype=np.float64)
    for t in range(1, cfg["T"] + 1):
        g = env.get_gradient(t)
        x = opt.step(x, g)
    return x[:, 0]


def test_adam_stochastic_failure():
    """Under beta2=0.5, Adam must exhibit net positive drift (mean(x_T) > 0.25 and >65% positive seeds)."""
    cfg = STOCHASTIC_TEST_CONFIG
    x_T_seeds = run_stochastic(
        Adam,
        {"lr": cfg["alpha"], "beta1": cfg["beta1"], "beta2": cfg["beta2_fail"]},
    )
    mean_x_T = float(np.mean(x_T_seeds))
    frac_pos = float(np.mean(x_T_seeds > 0.0))
    assert mean_x_T > 0.25, f"Adam stochastic mean not positive enough: mean(x_T) = {mean_x_T}"
    assert frac_pos >= 0.65, f"Adam positive seed fraction too low: {frac_pos}"


def test_amsgrad_stochastic_non_divergence():
    """Under beta2=0.5, AMSGrad must maintain non-positive mean and significantly lower failure than Adam."""
    cfg = STOCHASTIC_TEST_CONFIG
    x_T_seeds = run_stochastic(
        AMSGrad,
        {
            "lr": cfg["alpha"],
            "beta1": cfg["beta1"],
            "beta2": cfg["beta2_fail"],
            "bias_correction": False,
        },
    )
    mean_x_T = float(np.mean(x_T_seeds))
    frac_failed = float(np.mean(x_T_seeds > 0.5))
    assert mean_x_T <= 0.0, f"AMSGrad stochastic mean diverged positive: mean(x_T) = {mean_x_T}"
    assert frac_failed <= 0.35, f"AMSGrad failure fraction too high: {frac_failed}"


def test_adagrad_stochastic_convergence():
    """AdaGrad must strongly converge into the negative optimal domain."""
    cfg = STOCHASTIC_TEST_CONFIG
    x_T_seeds = run_stochastic(
        AdaGrad,
        {"lr": cfg["alpha"]},
    )
    mean_x_T = float(np.mean(x_T_seeds))
    frac_neg = float(np.mean(x_T_seeds < 0.0))
    assert mean_x_T < -0.5, f"AdaGrad stochastic convergence too weak: mean(x_T) = {mean_x_T}"
    assert frac_neg >= 0.85, f"AdaGrad negative fraction too low: {frac_neg}"
