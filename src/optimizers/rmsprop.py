"""
RMSProp Optimizer (Tieleman & Hinton, 2012).
Exponential moving average of squared gradients without bias correction.
"""

from typing import Optional, Callable
import numpy as np
from .base import BaseOptimizer


class RMSProp(BaseOptimizer):
    """
    RMSProp optimizer.

    Update rule:
        v_t = beta * v_{t-1} + (1 - beta) * g_t^2
        x_{t+1} = Pi_F(x_t - (lr_t / (sqrt(v_t) + eps)) * g_t)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta: float = 0.99,
        eps: float = 1e-8,
        projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(lr=lr, projection_fn=projection_fn)
        self.beta = beta
        self.eps = eps

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        lr_t = self.get_lr(self.t)

        if "v" not in self.state:
            self.state["v"] = np.zeros_like(grad)

        self.state["v"] = self.beta * self.state["v"] + (1.0 - self.beta) * (grad ** 2)
        v_t = self.state["v"]

        step_dir = grad / (np.sqrt(v_t) + self.eps)
        x_next = x - lr_t * step_dir
        return self.project(x_next)
