"""
Stochastic Gradient Descent (SGD) and SGD with Momentum.
"""

from typing import Optional, Callable
import numpy as np
from .base import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Vanilla SGD and SGD with classical momentum.

    Update rule:
        m_t = momentum * m_{t-1} + grad
        x_{t+1} = Pi_F(x_t - lr * m_t)
    """

    def __init__(
        self,
        lr: float = 1e-2,
        momentum: float = 0.0,
        projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(lr=lr, projection_fn=projection_fn)
        self.momentum = momentum

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        lr_t = self.get_lr(self.t)

        if "m" not in self.state:
            self.state["m"] = np.zeros_like(grad)

        if self.momentum > 0.0:
            self.state["m"] = self.momentum * self.state["m"] + grad
            update = lr_t * self.state["m"]
        else:
            update = lr_t * grad

        x_next = x - update
        return self.project(x_next)
