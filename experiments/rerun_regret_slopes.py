"""
Re-run Table 4 regret slopes at δ=0.10, N=200 (matching §3.3 parameters).
Saves results to results/mechanism_probes/lambda_regret_slopes_delta010.json
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import json
from src.benchmarks.counterexample import StochasticCounterexample, project_box_1d
from src.optimizers import Adam, AMSGrad, AdaGrad

cfg = dict(C=20, delta=0.10, alpha=0.8, beta1=0.9, beta2=0.5, N=200, T=5000, base_seed=42)

def run_optimizer(opt_cls, opt_kwargs, cfg):
    """Return cumulative regret sequence averaged over N seeds."""
    env = StochasticCounterexample(C=cfg["C"], delta=cfg["delta"],
                                   n_seeds=cfg["N"], base_seed=cfg["base_seed"])
    opt = opt_cls(**opt_kwargs, projection_fn=project_box_1d)
    x = np.zeros((cfg["N"], 1))
    # x* = -1 for δ>0
    x_star = -1.0
    regret_seq = []
    for t in range(1, cfg["T"] + 1):
        g = env.get_gradient(t)
        loss = g[:, 0] * x[:, 0]        # f_t(x) = g_t * x
        loss_star = g[:, 0] * x_star
        regret_seq.append(float(np.mean(loss - loss_star)))
        x = opt.step(x, g)
    cumregret = np.cumsum(regret_seq)
    return cumregret

results = {}
for name, cls, kw in [
    ("Adam",    Adam,    {"lr": 0.8, "beta1": 0.9, "beta2": 0.5}),
    ("AMSGrad", AMSGrad, {"lr": 0.8, "beta1": 0.9, "beta2": 0.5, "bias_correction": False}),
    ("AdaGrad", AdaGrad, {"lr": 0.8}),
]:
    cr = run_optimizer(cls, kw, cfg)
    T = cfg["T"]
    t0 = int(0.1 * T)
    # fit log(R(t)) ~ lambda * log(t) over [t0, T]
    ts = np.arange(t0 + 1, T + 1)
    log_t = np.log(ts)
    log_R = np.log(np.maximum(cr[t0:], 1e-12))
    slope, intercept = np.polyfit(log_t, log_R, 1)
    results[name] = {"lambda": round(float(slope), 4),
                     "final_cumregret": round(float(cr[-1]), 2),
                     "delta": cfg["delta"], "N": cfg["N"]}
    print(f"{name:10s}: λ = {slope:.4f}  R(T) = {cr[-1]:.1f}")

out_path = "results/mechanism_probes/lambda_regret_slopes_delta010.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {out_path}")
