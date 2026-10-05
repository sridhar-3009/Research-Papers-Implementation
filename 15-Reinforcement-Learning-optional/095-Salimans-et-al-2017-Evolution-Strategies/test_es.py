import numpy as np

import es as E


def test_centered_ranks_and_unbiased_gradient():
    assert np.allclose(E.centered_ranks(np.array([5.0, -1.0, 100.0])), [0.0, -0.5, 0.5])
    F = lambda P: -((P - 1) ** 2).sum(1)
    rng = np.random.default_rng(0)
    g = np.mean([E.es_gradient(F, np.zeros(5), 0.5, 100, rng, shaping=False)[0] for _ in range(100)], 0)
    assert np.allclose(g, 2.0, atol=0.15)


def test_parallel_workers_stay_identical_and_improve():
    F = lambda th: -np.sum((th - 1) ** 2)
    thetas, sent, grad_floats = E.parallel_es(F, np.zeros(20), workers=6, iters=60, alpha=0.02)
    assert all(np.array_equal(thetas[0], t) for t in thetas)
    assert F(thetas[0]) > F(np.zeros(20)) / 2 and sent * 20 == grad_floats


def test_cartpole_physics_and_es_learns():
    r, _ = E.cartpole_returns(np.zeros((3, 5)))
    assert np.all(r < 60)                                             # always pushing left falls quickly
    _, _, final = E.train_es_cartpole(iters=60)
    assert final > 300


def test_variance_grows_for_pg_not_es():
    out, true = E.estimator_variance_vs_T(Ts=(10, 300), n=3000)
    assert out[300]["PG var / true^2"] > 10 * out[10]["PG var / true^2"]
    assert out[300]["ES var / true^2"] < 2 * out[10]["ES var / true^2"]
    assert abs(out[300]["ES mean"] - true) < 0.05
