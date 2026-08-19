"""
Unit Tests for Optimizers.
Guarantees algorithmic correctness:
1. Exact bias correction at t=1: m_tilde_1 = g_1, v_tilde_1 = g_1^2 with beta1 != beta2.
2. Weak monotonicity of AMSGrad: v_hat_t >= v_hat_{t-1}.
3. Monotonic accumulator for AdaGrad: G_t >= G_{t-1}.
4. Projection operator Pi_F correctly constrains domain.
5. Vectorized operations support across (n_seeds, d).
"""

import pytest
import numpy as np
from src.optimizers import SGD, AdaGrad, RMSProp, Adam, AMSGrad
from src.benchmarks.counterexample import project_box_1d


class TestOptimizerUnitCorrectness:

    def test_adam_bias_correction_t1(self):
        """
        Verify exact debiasing at step t=1:
        m_tilde_1 = g_1 and v_tilde_1 = g_1^2 for asymmetric beta1 != beta2.
        """
        beta1, beta2 = 0.85, 0.42
        opt = Adam(lr=0.1, beta1=beta1, beta2=beta2)
        
        x = np.array([[0.5, -0.2]])
        grad = np.array([[3.0, -4.0]])
        
        x_next = opt.step(x, grad)
        
        # Verify internal unbiasing
        np.testing.assert_allclose(opt.state["m_tilde"], grad, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(opt.state["v_tilde"], grad ** 2, rtol=1e-7, atol=1e-7)

    def test_amsgrad_weak_monotonicity(self):
        """
        Verify that AMSGrad v_hat is strictly weakly monotonic (v_hat_t >= v_hat_{t-1})
        under oscillating large-then-small gradients.
        """
        opt = AMSGrad(lr=0.1, beta1=0.9, beta2=0.5, bias_correction=False)
        x = np.zeros((1, 1))
        
        # Oscillating gradients: large then small
        grads = [10.0, 0.1, 0.0, 5.0, 0.05, 12.0, 0.01]
        
        prev_v_hat = 0.0
        for g_val in grads:
            g = np.array([[g_val]])
            x = opt.step(x, g)
            curr_v_hat = opt.state["v_hat"][0, 0]
            assert curr_v_hat >= prev_v_hat, f"AMSGrad v_hat decreased: {curr_v_hat} < {prev_v_hat}"
            prev_v_hat = curr_v_hat

    def test_adagrad_monotonic_accumulator(self):
        """
        Verify AdaGrad accumulator G_t is non-decreasing.
        """
        opt = AdaGrad(lr=0.1)
        x = np.zeros((1, 2))
        
        grads = [np.array([[2.0, -1.0]]), np.array([[0.5, 0.0]]), np.array([[-3.0, 4.0]])]
        prev_G = np.zeros((1, 2))
        for g in grads:
            x = opt.step(x, g)
            curr_G = opt.state["G"]
            assert np.all(curr_G >= prev_G)
            prev_G = curr_G.copy()

    def test_projection_operator(self):
        """
        Verify that projection_fn constrains parameter updates to [-1, 1].
        """
        opt = SGD(lr=10.0, projection_fn=project_box_1d)
        x = np.array([[0.0]])
        grad = np.array([[-5.0]]) # Huge step towards +50
        
        x_next = opt.step(x, grad)
        assert x_next[0, 0] == 1.0, f"Expected 1.0, got {x_next[0, 0]}"

    def test_vectorization_shapes(self):
        """
        Verify that all optimizers seamlessly handle (n_seeds, d) tensors.
        """
        n_seeds, d = 25, 4
        opts = [
            SGD(lr=0.01),
            AdaGrad(lr=0.01),
            RMSProp(lr=0.01),
            Adam(lr=0.01),
            AMSGrad(lr=0.01),
        ]
        
        for opt in opts:
            x = np.random.randn(n_seeds, d)
            grad = np.random.randn(n_seeds, d)
            x_next = opt.step(x, grad)
            assert x_next.shape == (n_seeds, d), f"Optimizer {opt.__class__.__name__} failed shape preservation"
