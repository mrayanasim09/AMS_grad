"""
Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014).
Includes exact bias correction for both first and second moments.
"""

from typing import Optional, Callable
import numpy as np
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam optimizer.

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_tilde_t = m_t / (1 - beta1^t)
        v_tilde_t = v_t / (1 - beta2^t)
        x_{t+1} = Pi_F(x_t - (lr_t / (sqrt(v_tilde_t) + eps)) * m_tilde_t)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(lr=lr, projection_fn=projection_fn)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        lr_t = self.get_lr(self.t)

        if "m" not in self.state:
            self.state["m"] = np.zeros_like(grad)
            self.state["v"] = np.zeros_like(grad)

        # Update biased first moment
        self.state["m"] = self.beta1 * self.state["m"] + (1.0 - self.beta1) * grad
        # Update biased second moment
        self.state["v"] = self.beta2 * self.state["v"] + (1.0 - self.beta2) * (grad ** 2)

        m_t = self.state["m"]
        v_t = self.state["v"]

        # Exact bias correction
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        m_tilde = m_t / bias_correction1
        v_tilde = v_t / bias_correction2

        # Save for diagnostic inspection
        self.state["m_tilde"] = m_tilde
        self.state["v_tilde"] = v_tilde

        # Parameter update
        step_dir = m_tilde / (np.sqrt(v_tilde) + self.eps)
        x_next = x - lr_t * step_dir
        return self.project(x_next)
