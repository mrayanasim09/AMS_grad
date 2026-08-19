"""
AMSGrad: On the Convergence of Adam and Beyond (Reddi, Kale & Kumar, ICLR 2018).
Guarantees non-decreasing effective second moment by taking coordinate-wise maximum.
"""

from typing import Optional, Callable
import numpy as np
from .base import BaseOptimizer


class AMSGrad(BaseOptimizer):
    """
    AMSGrad optimizer.

    Parameters
    ----------
    lr : float or callable
        Learning rate alpha.
    beta1 : float
        Exponential decay rate for first moment (default 0.9).
    beta2 : float
        Exponential decay rate for second moment (default 0.999).
    eps : float
        Numerical smoothing term outside sqrt (default 1e-8).
    bias_correction : bool
        If False (default, Reddi et al. Algorithm 2), operates on raw moments.
        If True (PyTorch convention), applies debiasing to moments.
    projection_fn : callable, optional
        Feasible domain projection operator Pi_F.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        bias_correction: bool = False,
        projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(lr=lr, projection_fn=projection_fn)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.bias_correction = bias_correction

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        lr_t = self.get_lr(self.t)

        if "m" not in self.state:
            self.state["m"] = np.zeros_like(grad)
            self.state["v"] = np.zeros_like(grad)
            self.state["v_hat"] = np.zeros_like(grad)

        # Update EMA moments
        self.state["m"] = self.beta1 * self.state["m"] + (1.0 - self.beta1) * grad
        self.state["v"] = self.beta2 * self.state["v"] + (1.0 - self.beta2) * (grad ** 2)

        # Monotonic running max of second moment
        self.state["v_hat"] = np.maximum(self.state["v_hat"], self.state["v"])

        m_t = self.state["m"]
        v_t = self.state["v"]
        v_hat_t = self.state["v_hat"]

        if self.bias_correction:
            # PyTorch style debiasing
            bc1 = 1.0 - (self.beta1 ** self.t)
            bc2 = 1.0 - (self.beta2 ** self.t)
            m_effective = m_t / bc1
            v_effective = v_hat_t / bc2
        else:
            # Raw Reddi et al. Algorithm 2
            m_effective = m_t
            v_effective = v_hat_t

        self.state["m_effective"] = m_effective
        self.state["v_effective"] = v_effective

        step_dir = m_effective / (np.sqrt(v_effective) + self.eps)
        x_next = x - lr_t * step_dir
        return self.project(x_next)
