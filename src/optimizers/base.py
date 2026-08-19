"""
Base Optimizer Interface.
Supports vectorized operations across multiple seeds (n_seeds, d) or single instances (d,).
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import numpy as np


class BaseOptimizer(ABC):
    """
    Abstract Base Class for pure NumPy optimizers.
    
    Parameters
    ----------
    lr : float or callable
        Learning rate alpha. Can be a scalar or a schedule callable lr(t).
    projection_fn : callable, optional
        Projection operator Pi_F(x) mapping points onto feasible domain F.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.lr = lr
        self.projection_fn = projection_fn
        self.t = 0
        self.state: Dict[str, Any] = {}

    def get_lr(self, t: int) -> float:
        """Compute current step size alpha_t."""
        if callable(self.lr):
            return self.lr(t)
        return self.lr

    def reset(self) -> None:
        """Reset optimizer internal step counter and moment states."""
        self.t = 0
        self.state.clear()

    @abstractmethod
    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Perform a single optimization update step.

        Parameters
        ----------
        x : np.ndarray
            Current parameter array of shape (..., d).
        grad : np.ndarray
            Current gradient array of shape (..., d).

        Returns
        -------
        x_next : np.ndarray
            Updated parameter array of shape (..., d), projected onto domain F if projection_fn is set.
        """
        pass

    def project(self, x: np.ndarray) -> np.ndarray:
        """Apply feasible set projection if configured."""
        if self.projection_fn is not None:
            return self.projection_fn(x)
        return x
