"""
AdaGrad Optimizer (Duchi et al., 2011).
Serves as the theoretical positive control for non-decreasing cumulative second moment (Gamma_t >= 0).
"""

from typing import Optional, Callable
import numpy as np
from .base import BaseOptimizer


class AdaGrad(BaseOptimizer):
    """
    AdaGrad optimizer with coordinate-wise cumulative sum of squared gradients.

    Update rule:
        G_t = G_{t-1} + g_t^2
        x_{t+1} = Pi_F(x_t - (lr_t / (sqrt(G_t) + eps)) * g_t)
    """

    def __init__(
        self,
        lr: float = 1e-2,
        eps: float = 1e-8,
        projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(lr=lr, projection_fn=projection_fn)
        self.eps = eps

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        lr_t = self.get_lr(self.t)

        if "G" not in self.state:
            self.state["G"] = np.zeros_like(grad)

        self.state["G"] += grad ** 2
        G_t = self.state["G"]

        # Epsilon outside sqrt
        step_dir = grad / (np.sqrt(G_t) + self.eps)
        x_next = x - lr_t * step_dir
        return self.project(x_next)
