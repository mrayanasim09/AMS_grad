"""
2D Toy Optimization Landscapes.
Includes ill-conditioned Quadratic Bowl, Rosenbrock, Himmelblau (bounded non-convex), and Saddle Point (escape test).
"""

from typing import Tuple
import numpy as np


class QuadraticBowl:
    """Ill-conditioned quadratic bowl: f(x, y) = 0.5 * (x^2 + kappa * y^2)."""
    def __init__(self, kappa: float = 50.0):
        self.kappa = kappa

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        return 0.5 * (x ** 2 + self.kappa * (y ** 2))

    def grad(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        gx = x
        gy = self.kappa * y
        return np.stack([gx, gy], axis=-1)


class Rosenbrock:
    """Rosenbrock Banana function: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2."""
    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        return (1.0 - x) ** 2 + 100.0 * ((y - x ** 2) ** 2)

    def grad(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        gx = -2.0 * (1.0 - x) - 400.0 * x * (y - x ** 2)
        gy = 200.0 * (y - x ** 2)
        return np.stack([gx, gy], axis=-1)


class Himmelblau:
    """Himmelblau's function (bounded non-convex with 4 global minima)."""
    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        return (x ** 2 + y - 11.0) ** 2 + (x + y ** 2 - 7.0) ** 2

    def grad(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        gx = 4.0 * x * (x ** 2 + y - 11.0) + 2.0 * (x + y ** 2 - 7.0)
        gy = 2.0 * (x ** 2 + y - 11.0) + 4.0 * y * (x + y ** 2 - 7.0)
        return np.stack([gx, gy], axis=-1)


class SaddlePoint:
    """Saddle point landscape (unbounded below, used for saddle escape visualization)."""
    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        return xy[..., 0] ** 2 - xy[..., 1] ** 2

    def grad(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        return np.stack([2.0 * x, -2.0 * y], axis=-1)
