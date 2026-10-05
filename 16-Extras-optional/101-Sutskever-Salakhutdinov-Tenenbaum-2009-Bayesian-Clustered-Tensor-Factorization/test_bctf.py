import numpy as np

import bctf as B


def test_planted_data_and_split():
    T, co, cr = B.planted_data(n_obj=20, n_rel=3, seed=1)
    assert T.shape == (20, 3, 20) and abs(T.mean() - 0.3) < 0.01
    train, test = B.split(T, 0.1, 0.5, seed=0)
    assert len(test) == int(0.1 * T.size) and len(set(map(tuple, train)) & set(map(tuple, test))) == 0


def test_metrics():
    assert B.pr_auc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert B.adjusted_rand(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == 1.0
    assert abs(B.rmse(np.zeros(4), np.ones(4)) - 1) < 1e-12


def test_gaussian_conditional_sampler_matches_posterior_mean():
    rng = np.random.default_rng(0)
    design = rng.standard_normal((400, 2))
    v_true = np.array([1.5, -0.5])
    t = design @ v_true + 0.1 * rng.standard_normal(400)
    owner = np.zeros(400, int)
    draws = np.array([B.sample_vectors(design, t, owner, 1, np.zeros((1, 2)), np.ones((1, 2)), 0.01, rng)[0]
                      for _ in range(200)])
    assert np.allclose(draws.mean(0), v_true, atol=0.02)


def test_bayesian_beats_map_when_sparse():
    res, cl, n = B.compare(train_frac=0.03, seed=0, sweeps=30, burn=10)
    assert res["BTF"][0] < res["MAP"][0] - 0.2 and res["BCTF"][0] < res["MAP"][0] - 0.2
    assert res["BTF"][1] > res["block model (IRM-like)"][1]
