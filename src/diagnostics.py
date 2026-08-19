"""
Diagnostics & Convergence Tracking.
Logs optimizer internal states, computes Gamma_t matrix/metric, cumulative regret, and log-log slopes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class ConvergenceTracker:
    """
    Tracks trajectory history and computes OCO diagnostic metrics.
    """

    def __init__(self, x_star: float = -1.0):
        self.x_star = x_star
        self.reset()

    def reset(self) -> None:
        self.history: Dict[str, List[np.ndarray]] = {
            "x": [],
            "grad": [],
            "v": [],
            "v_effective": [],
            "lr": [],
            "loss": [],
        }

    def record(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        v: np.ndarray,
        v_effective: np.ndarray,
        lr: float,
        loss: Optional[float] = None,
    ) -> None:
        self.history["x"].append(np.copy(x))
        self.history["grad"].append(np.copy(grad))
        self.history["v"].append(np.copy(v))
        self.history["v_effective"].append(np.copy(v_effective))
        self.history["lr"].append(lr)
        if loss is not None:
            self.history["loss"].append(loss)

    def get_trajectory(self) -> np.ndarray:
        """Returns array of shape (T, ...)."""
        return np.array(self.history["x"])

    def compute_gamma(self) -> np.ndarray:
        """
        Compute Gamma_t = sqrt(v_t)/alpha_t - sqrt(v_{t-1})/alpha_{t-1}.
        Returns array of shape (T-1, ...).
        """
        v_arr = np.array(self.history["v_effective"])
        lr_arr = np.array(self.history["lr"])
        
        # metric_t = sqrt(v_t) / alpha_t
        if v_arr.ndim == 1 or (v_arr.ndim == 2 and v_arr.shape[1] == 1):
            sqrt_v = np.sqrt(v_arr)
        else:
            sqrt_v = np.sqrt(v_arr)

        metric = sqrt_v / lr_arr[:, None] if sqrt_v.ndim > 1 else sqrt_v / lr_arr
        gamma = metric[1:] - metric[:-1]
        return gamma

    def compute_cumulative_regret(self) -> np.ndarray:
        """
        Compute cumulative regret R_t = sum_{i=1}^t g_i * (x_i - x^*).
        Returns array of shape (T, ...).
        """
        x_arr = np.array(self.history["x"])
        g_arr = np.array(self.history["grad"])
        
        # Instantaneous regret: g_t * (x_t - x_star)
        instant_regret = g_arr * (x_arr - self.x_star)
        return np.cumsum(instant_regret, axis=0)

    def compute_windowed_regret_slope(self, min_frac: float = 0.1) -> float:
        """
        Fit log-log slope lambda = d(log R_t) / d(log t) over t in [min_frac * T, T].
        """
        R_t = self.compute_cumulative_regret()
        if R_t.ndim > 1:
            R_t = np.mean(R_t, axis=tuple(range(1, R_t.ndim)))

        T = len(R_t)
        start_idx = max(int(min_frac * T), 10)
        
        t_vals = np.arange(start_idx + 1, T + 1)
        r_vals = R_t[start_idx:]
        
        # Guard against non-positive regret
        valid = (r_vals > 1e-12) & (t_vals > 0)
        if np.sum(valid) < 5:
            return 0.0

        log_t = np.log(t_vals[valid])
        log_r = np.log(r_vals[valid])
        
        slope, _ = np.polyfit(log_t, log_r, 1)
        return float(slope)
