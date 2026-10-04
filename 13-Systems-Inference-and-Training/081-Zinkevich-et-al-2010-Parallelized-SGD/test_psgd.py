import numpy as np

from psgd import (coupled_distance, dloss, eta_star, make_data, objective, simu_parallel_sgd, stationary_stats)


def test_losses_and_contraction():
    p, y = np.array([0.0, 3.0]), np.array([1.0, 1.0])
    assert np.allclose(dloss("huber", p, y), [-1.0, 1.0]) and np.allclose(dloss("squared", p, y), [-1.0, 2.0])
    X, yy = make_data(2000, d=64, seed=1)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    eta, lam = 0.5, 1e-2
    assert eta <= eta_star(X, lam)
    d = coupled_distance(X, yy, eta, lam, steps=300)
    assert all(b <= a * (1 - eta * lam) + 1e-9 for a, b in zip(d, d[1:]))           # Lemma 3, every step


def test_simu_parallel_sgd_partition_and_average():
    X, y = make_data(1000, d=32, seed=2)
    v, W, _ = simu_parallel_sgd(X, y, 10, 0.3, 1e-3)
    assert W.shape == (10, 32) and np.allclose(v, W.mean(0))
    v1, W1, _ = simu_parallel_sgd(X, y, 1, 0.3, 1e-3)
    assert W1.shape == (1, 32)


def test_averaging_cuts_variance():
    X, y = make_data(5000, d=32, seed=3)
    mean, var, W = stationary_stats(X, y, 0.5, 1e-3, runs=100, steps=1500)
    avg = W.reshape(10, 10, -1).mean(1)
    v10 = ((avg - mean) ** 2).sum(1).mean()
    assert v10 < var / 5                                                             # ~ var / 10
    assert objective(avg, X, y, 1e-3, "huber").mean() < objective(W, X, y, 1e-3, "huber").mean()
