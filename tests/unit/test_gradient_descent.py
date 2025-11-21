"""
Tests for the gradient descent implementation.
Run this file with pytest:
    pytest tests/test_gradient_descent.py -v
"""

import numpy as np
from rice_ml.supervised_learning.gradient_descent import GradientDescent1D, GradientDescentND


def test_gradient_descent_1d_converges():
    """Test that GradientDescent1D converges to the known minimum."""
    # f(w) = (w - 2)^2 + 1, derivative f'(w) = 2(w - 2)
    fprime = lambda w: 2 * (w - 2)
    gd = GradientDescent1D(fprime, alpha=0.8, tol=1e-3, max_iter=1000)
    history = gd.fit(fprime, w0=5.0)

    # Should converge near w = 2
    assert abs(history[-1] - 2.0) < 1e-2


def test_gradient_descent_nd_converges():
    """Test that GradientDescentND converges for a 2D quadratic."""
    # f(w) = w1^2 + w2^2, ∇f = [2w1, 2w2]
    grad = lambda w: np.array([2 * w[0], 2 * w[1]])

    gd = GradientDescentND(grad, alpha=0.1, max_iter=500)
    path = gd.fit(grad, np.array([5.0, -5.0]))

    # Final weights should be close to (0, 0)
    assert np.allclose(path[-1], np.array([0.0, 0.0]), atol=1e-2)


def test_learning_rate_affects_convergence_speed():
    """Test that a smaller learning rate converges slower."""
    grad = lambda w: 2 * (w - 2)

    gd_fast = GradientDescent1D(alpha=0.8)
    gd_slow = GradientDescent1D(alpha=0.1)

    w_fast = gd_fast.fit(grad, w0=5.0)
    w_slow = gd_slow.fit(grad, w0=5.0)

    assert len(w_slow) > len(w_fast)