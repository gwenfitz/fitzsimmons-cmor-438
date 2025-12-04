import numpy as np
from rice_ml.supervised_learning.gradient_descent import GradientDescent1D, GradientDescentND

# ----------------------
# 1D Gradient Descent
# ----------------------

def test_gradient_descent_1d_converges():
    """GradientDescent1D should converge to w = 2 for f(w) = (w-2)^2."""
    df = lambda w: 2 * (w - 2)
    gd = GradientDescent1D(df, alpha=0.8, tol=1e-6, max_iter=500)

    history = gd.fit(5.0)
    w_final = history[-1]

    assert np.isclose(w_final, 2.0, atol=1e-2)
    assert len(history) > 1


def test_gradient_descent_1d_small_lr_converges_slower():
    """A smaller alpha should require more iterations."""
    df = lambda w: 2 * (w - 2)

    fast = GradientDescent1D(df, alpha=0.8, tol=1e-6, max_iter=500)
    slow = GradientDescent1D(df, alpha=0.1, tol=1e-6, max_iter=500)

    fast_hist = fast.fit(5.0)
    slow_hist = slow.fit(5.0)

    assert len(slow_hist) > len(fast_hist)  # slower learning rate → slower convergence


def test_gradient_descent_1d_history_monotonic():
    """w values should move toward the minimum each step."""
    df = lambda w: 2 * (w - 2)
    gd = GradientDescent1D(df, alpha=0.5)

    history = gd.fit(5.0)

    # Distance to optimum should decrease
    distances = [abs(w - 2.0) for w in history]
    assert all(distances[i] >= distances[i+1] for i in range(len(distances)-1))


# ----------------------
# N-Dimensional
# ----------------------

def test_gradient_descent_nd_converges():
    """GradientDescentND should converge to [0,0] for f = w1^2 + w2^2."""
    grad = lambda w: np.array([2*w[0], 2*w[1]])

    gd = GradientDescentND(grad, alpha=0.1, max_iter=500)
    path = gd.fit(np.array([5.0, -5.0]))

    w_final = path[-1]
    assert np.allclose(w_final, np.array([0.0, 0.0]), atol=1e-2)
    assert len(path) > 1


def test_gradient_descent_nd_tolerance_stops_early():
    """When tolerance is loose, GD should stop quickly."""
    grad = lambda w: np.array([2*w[0], 2*w[1]])

    gd = GradientDescentND(grad, alpha=0.1, tol=1.0, max_iter=500)
    path = gd.fit(np.array([1.0, 1.0]))

    # Expect very early stopping because tol is large
    assert len(path) < 10


def test_gradient_descent_nd_path_monotonic():
    """Distance to optimum should shrink each step."""
    grad = lambda w: np.array([2*w[0], 2*w[1]])
    gd = GradientDescentND(grad, alpha=0.1)

    path = gd.fit(np.array([3.0, 4.0]))  # distance = 5

    distances = [np.linalg.norm(w) for w in path]
    assert all(distances[i] >= distances[i+1] for i in range(len(distances)-1))