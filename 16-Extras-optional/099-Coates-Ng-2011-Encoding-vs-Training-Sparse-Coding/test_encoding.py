import numpy as np

import encoding as E


def test_lasso_fista_matches_kkt_conditions():
    rng = np.random.default_rng(0)
    D = E.unit_columns(rng.standard_normal((8, 12)))
    X = rng.standard_normal((5, 8))
    lam = 0.5
    S = E.lasso_fista(X, D, lam, iters=2000)
    G = 2 * (S @ D.T - X) @ D                                         # gradient of the squared error
    nz = np.abs(S) > 1e-6
    assert np.allclose(G[nz], -lam * np.sign(S[nz]), atol=1e-3)       # KKT on the support
    assert np.all(np.abs(G[~nz]) <= lam + 1e-3)                       # KKT off the support


def test_omp_recovers_sparse_signal_and_omp1_is_gain_shape_vq():
    rng = np.random.default_rng(1)
    D = E.unit_columns(rng.standard_normal((20, 30)))
    s_true = np.zeros(30); s_true[[3, 17]] = [1.5, -2.0]
    x = (D @ s_true)[None]
    S = E.omp(x, D, 2)
    assert np.allclose(S[0], s_true, atol=1e-6)
    S1 = E.omp(x, D, 1)
    j = np.argmax(np.abs(x @ D))
    assert np.count_nonzero(S1) == 1 and np.isclose(S1[0, j], (x @ D)[0, j])


def test_soft_threshold_and_polarity_split():
    D = np.eye(3)
    X = np.array([[0.6, -0.2, -1.0]])
    F = E.encode(X, D, "T", 0.25)
    assert np.allclose(F, [[0.35, 0, 0, 0, 0, 0.75]])


def test_whitening_and_dictionaries_are_unit_norm():
    images, y = E.load_digits_images()
    P = E.Pipeline(images[:300], n_sample=5000)
    C = np.cov(P.sample, rowvar=False)
    assert np.abs(np.diag(C) - 1).max() < 0.25                        # whitened: ~unit variance
    rng = np.random.default_rng(0)
    for name, kw in E.TRAINERS.items():
        D = E.train_dictionary(P.sample[:2000], 16, rng=rng, iters=3, **kw)
        assert np.allclose(np.linalg.norm(D, axis=0), 1)
