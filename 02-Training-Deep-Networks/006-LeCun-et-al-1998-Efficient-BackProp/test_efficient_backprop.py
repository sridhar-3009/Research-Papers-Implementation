"""Tests: each checks an equation, trick or claim of "Efficient BackProp".

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest

from efficient_backprop import (MLP, InputTransform, lecun_tanh, lecun_tanh_prime, lms_batch,
                                lms_hessian, power_method, stochastic_diag_lm_rates, train_batch,
                                train_sgd, two_gaussians)


def full_hessian(net, X, D, h=1e-5):
    w = net.get()
    H = np.zeros((w.size, w.size))
    for i in range(w.size):
        e = np.zeros_like(w); e[i] = h
        net.set(w + e); g1 = net.flat_gradient(X, D)
        net.set(w - e); g2 = net.flat_gradient(X, D)
        H[:, i] = (g1 - g2) / (2 * h)
    net.set(w)
    return (H + H.T) / 2


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(0)
    return MLP([4, 5, 3], rng=1), rng.normal(size=(8, 4)), rng.choice([-1.0, 1.0], (8, 3))


# ---- Section 4.4: the recommended sigmoid ----

def test_recommended_sigmoid_properties():
    # f(1) = 1, f(-1) = -1, and the second derivative is largest (in size) at x = 1
    assert lecun_tanh(1.0) == pytest.approx(1.0, abs=1e-3)
    assert lecun_tanh(-1.0) == pytest.approx(-1.0, abs=1e-3)
    x = np.linspace(0.01, 3, 3000)
    second = np.gradient(lecun_tanh_prime(x), x)
    assert x[np.argmax(np.abs(second))] == pytest.approx(0.99, abs=0.03)


# ---- Section 4.3: input transformation ----

def test_input_transform_centres_scales_and_decorrelates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)) @ np.array([[2, 1, 0], [0, 1, 0], [0, 0, 5]]) + 7
    Z = InputTransform().fit(X)(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-6)
    W = InputTransform(decorrelate=True).fit(X)(X)
    assert np.allclose(np.cov(W.T, bias=True), np.eye(3), atol=1e-6)     # uncorrelated, variance 1


# ---- Section 4.6: initialization ----

def test_lecun_init_gives_unit_variance_sums():
    # Eq. (16): with unit-variance inputs, weights of std m^-1/2 give weighted sums of std ~1
    rng = np.random.default_rng(0)
    net = MLP([400, 300], rng=0)
    Y = rng.normal(size=(2000, 400)) @ net.W[0]
    assert Y.std() == pytest.approx(1.0, abs=0.05)
    assert net.W[0].std() == pytest.approx(400 ** -0.5, rel=0.05)


# ---- Section 3: backprop ----

def test_gradient_matches_finite_differences(small_problem):
    net, X, D = small_problem
    w, g = net.get(), net.flat_gradient(X, D)
    fd = np.zeros_like(w)
    for i in range(w.size):
        e = np.zeros_like(w); e[i] = 1e-6
        net.set(w + e); up = net.cost(X, D)
        net.set(w - e); down = net.cost(X, D)
        fd[i] = (up - down) / 2e-6
    net.set(w)
    assert np.allclose(g, fd, atol=1e-7)


# ---- Section 7: Hessian information ----

def test_diag_hessian_is_the_gauss_newton_diagonal(small_problem):
    # Eqs. (54)-(56) give the diagonal of J^T J (the Gauss-Newton Hessian).
    net, X, D = small_problem
    w = net.get()
    hW, hb = net.diag_hessian(X)
    diag = np.concatenate([np.concatenate([a.ravel(), b]) for a, b in zip(hW, hb)])
    # Gauss-Newton diagonal = mean over patterns of sum over outputs of (d output / d w)^2
    J = np.zeros((len(X), 3, w.size))
    for i in range(w.size):
        e = np.zeros_like(w); e[i] = 1e-6
        net.set(w + e); up = net.predict(X)
        net.set(w - e); down = net.predict(X)
        J[:, :, i] = (up - down) / 2e-6
    net.set(w)
    assert np.allclose(diag, (J ** 2).sum(1).mean(0), atol=1e-6)


def test_power_method_finds_largest_eigenvalue(small_problem):
    net, X, D = small_problem
    true = np.abs(np.linalg.eigvalsh(full_hessian(net, X, D))).max()
    lam, _ = power_method(net, X, D, iters=200, rng=0)
    assert lam == pytest.approx(true, rel=1e-3)


# ---- Section 5: learning rates and eigenvalues ----

def test_lms_hessian_is_input_covariance():
    X, _ = two_gaussians(100, rng=0)
    Xb = np.c_[X, np.ones(100)]
    assert np.allclose(lms_hessian(X), Xb.T @ Xb / 100)                  # Eq. (29)
    cov = np.linalg.eigvalsh(np.cov(X.T, bias=True))
    assert cov == pytest.approx([0.036, 0.84], abs=0.04)                  # the paper's example


def test_learning_rate_limit_is_two_over_lambda_max():
    # Eq. (38): converges below 2/lambda_max, diverges above. Here lambda_max ~ 1.0
    # (the bias), so eta = 2.1 already diverges although 2.1 < 2.38 (the paper's bound).
    X, D = two_gaussians(100, rng=0)
    lam = np.linalg.eigvalsh(lms_hessian(X)).max()
    assert lam == pytest.approx(1.0, abs=0.01)
    for lr, diverges in ((1.5, False), (1.9, False), (2.1, True), (2.5, True)):
        h, _ = lms_batch(X, D, lr, epochs=40)
        assert (h[-1] > h[0]) == diverges


def test_non_centred_inputs_blow_up_the_condition_number():
    X, _ = two_gaussians(100, rng=0)
    cond = lambda A: (lambda ev: ev.max() / ev.min())(np.linalg.eigvalsh(lms_hessian(A)))
    assert cond(X + 3) > 20 * cond(X)                                     # Section 5.3


# ---- Section 4.1: stochastic vs batch ----

def test_stochastic_beats_batch_on_redundant_data():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 5))
    Dbase = np.sign(base @ rng.normal(size=(5, 2)))
    X, D = np.tile(base, (10, 1)), np.tile(Dbase, (10, 1))                # 10 identical copies
    s = train_sgd(MLP([5, 8, 2], rng=0), X, D, epochs=3, lr=0.02, rng=0)[-1]
    b = min(train_batch(MLP([5, 8, 2], rng=0), X, D, epochs=3, lr=lr)[-1] for lr in (0.1, 0.3, 1.0))
    assert s < b / 2


# ---- Section 9.1: stochastic diagonal Levenberg-Marquardt ----

def test_sdlm_rates_are_smaller_where_curvature_is_larger(small_problem):
    net, X, _ = small_problem
    rW, _ = stochastic_diag_lm_rates(net, X, eta=0.1, mu=0.01)
    hW, _ = net.diag_hessian(X)
    for r, h in zip(rW, hW):
        order = np.argsort(h.ravel())
        assert np.all(np.diff(r.ravel()[order]) <= 1e-12)                 # Eq. (61): more curvature, smaller step
