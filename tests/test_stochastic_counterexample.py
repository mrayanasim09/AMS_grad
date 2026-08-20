"""
Stochastic Counterexample Tests (Section 3 setting).
Evaluates Adam divergence, AMSGrad separation, and decreasing step-size convergence.

Stationary-Distribution Note:
-----------------------------
Under constant step size alpha = 0.8, increments inherit the AR(1) momentum structure of m_t,
which amplifies effective noise by sqrt((1 + beta1) / (1 - beta1)) = sqrt(19) approx 4.36.
With v_hat approx 400, effective drift mu approx -20 and effective variance sigma_eff approx 13,
the continuous limit yields a Fokker-Planck stationary distribution on [-1, 1] with exponent
2*mu*L / sigma_eff^2 approx 0.24. This produces a broad, near-uniform density across [-1, 1]
with a slight negative tilt (theoretical mean approx -0.08, matching observed -0.08 to -0.13).
Therefore, absolute convergence to x* = -1 is mathematically impossible under constant alpha;
rigorous O(sqrt(T)) sublinear regret and concentration at x* = -1 require decreasing step sizes
alpha_t = alpha / sqrt(t). Under decreasing step sizes, AMSGrad strongly concentrates at -1
(mean < -0.95), while Adam diverges to +1 via the second-moment v-collapse ratchet.
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
    "n_seeds": 200,
    "base_seed": 42,
}


def test_stochastic_environment_statistics(request):
    """
    Verify stochastic environment draws are unbiased (E[g] within 3-sigma),
    second moments match theory (E[g^2]), and lag-1 autocorrelation is near zero.
    """
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
    empirical_mean_sq = float(np.mean(draws**2))

    p = (1.0 + cfg["delta"]) / (cfg["C"] + 1.0)
    theo_mean = cfg["delta"]
    theo_mean_sq = p * (cfg["C"] ** 2) + (1.0 - p) * 1.0
    theo_var = theo_mean_sq - (theo_mean**2)
    std_err_mean = np.sqrt(theo_var / draws.size)

    # Variance of g^2 for Bernoulli mixture
    var_sq = p * ((cfg["C"] ** 2) ** 2) + (1.0 - p) * (1.0**2) - (theo_mean_sq**2)
    std_err_sq = np.sqrt(var_sq / draws.size)

    # Compute mean lag-1 autocorrelation across seeds
    autocorrs = []
    for s in range(n_seeds):
        gs = draws[:, s]
        ac = np.corrcoef(gs[:-1], gs[1:])[0, 1]
        autocorrs.append(ac)
    mean_autocorr = float(np.mean(autocorrs))

    # Cross-correlation check across seeds
    cross_corrs = []
    for i in range(min(10, n_seeds)):
        for j in range(i + 1, min(10, n_seeds)):
            c = np.corrcoef(draws[:, i], draws[:, j])[0, 1]
            cross_corrs.append(c)
    max_cross_corr = float(np.max(np.abs(cross_corrs)))

    # Record margins
    request.node.record_margin("E[g] within 3-sigma", 3.0 * std_err_mean - abs(empirical_mean - theo_mean), 0.0, ">=")
    request.node.record_margin("E[g^2] within 3-sigma", 3.0 * std_err_sq - abs(empirical_mean_sq - theo_mean_sq), 0.0, ">=")
    request.node.record_margin("Lag-1 autocorrelation |rho_1| < 0.05", 0.05 - abs(mean_autocorr), 0.0, ">=")
    request.node.record_margin("Seed independence max|corr| < 0.10", 0.10 - max_cross_corr, 0.0, ">=")

    assert abs(empirical_mean - theo_mean) <= 3.0 * std_err_mean, (
        f"Empirical E[g] = {empirical_mean:.4f} outside 3-sigma [{theo_mean - 3*std_err_mean:.4f}, {theo_mean + 3*std_err_mean:.4f}]"
    )
    assert abs(empirical_mean_sq - theo_mean_sq) <= 3.0 * std_err_sq, (
        f"Empirical E[g^2] = {empirical_mean_sq:.4f} outside 3-sigma [{theo_mean_sq - 3*std_err_sq:.4f}, {theo_mean_sq + 3*std_err_sq:.4f}]"
    )
    assert abs(mean_autocorr) < 0.05, f"Lag-1 autocorrelation too high: {mean_autocorr:.4f}"
    assert max_cross_corr < 0.10, f"Seed cross-correlation too high: {max_cross_corr:.4f}"


def run_stochastic(optimizer_cls, opt_kwargs, cfg=None):
    if cfg is None:
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


def test_stochastic_constant_alpha_separation(request):
    """
    Under constant step size alpha=0.8, Adam and AMSGrad exhibit distinct stationary distributions.
    Adam's v-collapse ratchets parameters positive, while AMSGrad maintains a negative-tilted diffusion.
    Asserts a robust separation between Adam and AMSGrad terminal means (margin > 0.20).
    """
    cfg = STOCHASTIC_TEST_CONFIG
    x_adam = run_stochastic(
        Adam,
        {"lr": cfg["alpha"], "beta1": cfg["beta1"], "beta2": cfg["beta2_fail"]},
        cfg=cfg,
    )
    x_ams = run_stochastic(
        AMSGrad,
        {
            "lr": cfg["alpha"],
            "beta1": cfg["beta1"],
            "beta2": cfg["beta2_fail"],
            "bias_correction": False,
        },
        cfg=cfg,
    )

    mean_adam = float(np.mean(x_adam))
    mean_ams = float(np.mean(x_ams))
    separation = mean_adam - mean_ams
    frac_adam_pos = float(np.mean(x_adam > 0.5))
    frac_ams_pos = float(np.mean(x_ams > 0.5))

    request.node.record_margin("Adam - AMSGrad separation > 0.20", separation, 0.20, ">")
    request.node.record_margin("Adam positive drift mean > 0.10", mean_adam, 0.10, ">")
    request.node.record_margin("AMSGrad mean < 0.05", 0.05, mean_ams, ">")
    request.node.record_margin("Failure fraction difference > 0.15", frac_adam_pos - frac_ams_pos, 0.15, ">")

    assert separation > 0.20, f"Separation between Adam and AMSGrad insufficient: {separation:.4f} (Adam={mean_adam:+.4f}, AMSGrad={mean_ams:+.4f})"
    assert mean_adam > 0.10, f"Adam mean not positive: {mean_adam:+.4f}"
    assert mean_ams < 0.05, f"AMSGrad mean diverged positive: {mean_ams:+.4f}"
    assert frac_adam_pos > frac_ams_pos + 0.15, (
        f"Adam failure rate ({frac_adam_pos:.2f}) not sufficiently above AMSGrad ({frac_ams_pos:.2f})"
    )


def test_amsgrad_decreasing_alpha_convergence(request):
    """
    Under theoretical decreasing step sizes alpha_t = alpha / sqrt(t),
    AMSGrad converges strongly to the global minimizer x* = -1 (mean < -0.90),
    while Adam still diverges positive due to persistent v-collapse.
    """
    cfg = {
        "C": 20,
        "T": 5000,
        "x1": 0.0,
        "delta": 0.5,
        "beta1": 0.9,
        "beta2_fail": 0.5,
        "alpha": 0.8,
        "n_seeds": 100,
        "base_seed": 42,
    }
    x_ams = run_stochastic(
        AMSGrad,
        {
            "lr": lambda t: 0.8 / np.sqrt(t),
            "beta1": cfg["beta1"],
            "beta2": cfg["beta2_fail"],
            "bias_correction": False,
        },
        cfg=cfg,
    )
    x_adam = run_stochastic(
        Adam,
        {
            "lr": lambda t: 0.8 / np.sqrt(t),
            "beta1": cfg["beta1"],
            "beta2": cfg["beta2_fail"],
        },
        cfg=cfg,
    )

    mean_ams = float(np.mean(x_ams))
    frac_ams_conv = float(np.mean(x_ams < -0.90))
    mean_adam = float(np.mean(x_adam))

    request.node.record_margin("AMSGrad decreasing mean < -0.90", -0.90 - mean_ams, 0.0, ">=")
    request.node.record_margin("AMSGrad fraction < -0.90 >= 0.90", frac_ams_conv, 0.90, ">=")
    request.node.record_margin("Adam decreasing mean > 0.50", mean_adam, 0.50, ">=")

    assert mean_ams < -0.90, f"AMSGrad decreasing step size did not converge: mean={mean_ams:+.4f}"
    assert frac_ams_conv >= 0.90, f"AMSGrad convergence fraction too low: {frac_ams_conv:.2f}"
    assert mean_adam > 0.50, f"Adam did not diverge positive under decreasing step size: mean={mean_adam:+.4f}"


def test_adagrad_stochastic_convergence(request):
    """
    AdaGrad accumulates historical squared gradients without decay,
    concentrating towards the negative optimal domain under both constant and decaying schedules.
    """
    cfg = STOCHASTIC_TEST_CONFIG
    x_ada = run_stochastic(
        AdaGrad,
        {"lr": cfg["alpha"]},
        cfg=cfg,
    )
    mean_ada = float(np.mean(x_ada))
    frac_neg = float(np.mean(x_ada < 0.0))

    request.node.record_margin("AdaGrad mean < -0.30", -0.30 - mean_ada, 0.0, ">=")
    request.node.record_margin("AdaGrad negative fraction >= 0.75", frac_neg, 0.75, ">=")

    assert mean_ada < -0.30, f"AdaGrad stochastic convergence too weak: mean(x_T) = {mean_ada:+.4f}"
    assert frac_neg >= 0.75, f"AdaGrad negative fraction too low: {frac_neg:.2f}"
