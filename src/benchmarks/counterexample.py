"""
Reddi et al. (ICLR 2018) Counterexample Benchmark.
Provides both Deterministic (Theorem 3) and Stochastic (Section 3) 1D Online Convex settings.
Fully vectorized across seeds with reproducible NumPy Generator spawning.
"""

from typing import Optional, Union, Tuple
import numpy as np


def project_box_1d(x: np.ndarray) -> np.ndarray:
    """Project coordinate onto feasible set F = [-1, 1]."""
    return np.clip(x, -1.0, 1.0)


class DeterministicCounterexample:
    """
    Deterministic periodic 1D counterexample from Reddi et al. (Theorem 3 setting).
    
    Gradient schedule:
        g_t = +C  if (t - 1) % C == 0
        g_t = -1  otherwise
        
    Domain: F = [-1, 1], Optimal point: x* = -1.
    """

    def __init__(self, C: int = 10, n_seeds: int = 1):
        self.C = C
        self.n_seeds = n_seeds
        self.x_star = -1.0

    def get_gradient(self, t: int) -> np.ndarray:
        """
        Compute gradient at step t (1-indexed).
        Returns array of shape (n_seeds, 1).
        """
        # Periodic condition: first step of every C-step cycle
        is_burst = ((t - 1) % self.C) == 0
        val = float(self.C) if is_burst else -1.0
        return np.full((self.n_seeds, 1), val, dtype=np.float64)

    def evaluate_loss(self, t: int, x: np.ndarray) -> np.ndarray:
        """Compute f_t(x) = g_t * x."""
        g = self.get_gradient(t)
        return g * x


class StochasticCounterexample:
    """
    Stochastic i.i.d. 1D counterexample from Reddi et al. (Section 3 setting).
    
    Gradient distribution:
        g_t = +C  with probability p = (1 + delta) / (C + 1)
        g_t = -1  with probability 1 - p
        
    Expected gradient: E[g_t] = delta > 0.
    Domain: F = [-1, 1], Optimal point: x* = -1.
    """

    def __init__(
        self,
        C: int = 20,
        delta: float = 0.05,
        n_seeds: int = 30,
        base_seed: int = 42,
    ):
        self.C = C
        self.delta = delta
        self.n_seeds = n_seeds
        self.p = (1.0 + delta) / (C + 1.0)
        self.x_star = -1.0
        
        # Initialize independent RNG generators per seed via SeedSequence
        ss = np.random.SeedSequence(base_seed)
        self.rngs = [np.random.default_rng(s) for s in ss.spawn(n_seeds)]

    def get_gradient(self, t: int) -> np.ndarray:
        """
        Sample gradient independently for each seed at step t.
        Returns array of shape (n_seeds, 1).
        """
        # Sample Bernoulli(p) across seeds
        uniform_draws = np.array([rng.uniform(0.0, 1.0) for rng in self.rngs])
        is_burst = (uniform_draws < self.p)
        grads = np.where(is_burst, float(self.C), -1.0)[:, None]
        return grads.astype(np.float64)

    def evaluate_loss(self, t: int, x: np.ndarray) -> np.ndarray:
        """Compute f_t(x) = g_t * x."""
        g = self.get_gradient(t)
        return g * x
