"""Tests: each checks an equation or claim of Sutskever et al. (2013).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest
import torch

from momentum import (RNN, Autoencoder, Momentum, addition_problem, mu_schedule, quadratic_run,
                      sparse_init_)


# ---- Eqs. (1)-(4) ----

@pytest.mark.parametrize("nesterov", [False, True])
def test_momentum_class_matches_the_equations(nesterov):
    # The torch optimizer must produce exactly the iterates of Eqs. (1)-(4).
    A, b = np.diag([3.0, 0.5]), np.array([1.0, -1.0])
    x = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    At, bt = torch.tensor(A), torch.tensor(b)
    opt = Momentum([x], lr=0.1, mu=0.9, nesterov=nesterov)
    for _ in range(20):
        opt.step(lambda: 0.5 * x @ At @ x + bt @ x)
    ref = quadratic_run(A, b, [2.0, -1.0], 0.1, 0.9, 20, nesterov)[-1]
    assert np.allclose(x.detach().numpy(), ref, atol=1e-10)


def test_theorem_2_1_nag_is_cm_with_reduced_momentum():
    # Along an eigendirection with curvature lambda, NAG = CM with mu (1 - lambda eps).
    lams, b, x0, eps, mu = np.array([10.0, 1.0, 0.1]), np.array([1.0, -2.0, 0.5]), [3.0, -1.0, 2.0], 0.05, 0.9
    nag = quadratic_run(np.diag(lams), b, x0, eps, mu, 60, nesterov=True)
    for i, lam in enumerate(lams):
        cm = quadratic_run(np.array([[lam]]), b[i:i + 1], [x0[i]], eps, mu * (1 - lam * eps), 60, nesterov=False)
        assert np.allclose(nag[:, i], cm[:, 0], atol=1e-12)


def test_nag_oscillates_less_than_cm_on_an_oblong_quadratic():
    # Section 2.1 / appendix Fig. 2
    A = np.diag([1.0, 100.0])
    cm = quadratic_run(A, np.zeros(2), [-10, 1], 0.01, 0.9, 100, False)
    nag = quadratic_run(A, np.zeros(2), [-10, 1], 0.01, 0.9, 100, True)
    flips = lambda p: np.sum(np.diff(np.sign(p[:, 1])) != 0)
    assert flips(nag) < flips(cm) / 10
    assert np.linalg.norm(nag[-1]) < np.linalg.norm(cm[-1])


def test_nag_can_diverge_where_cm_converges():
    # Reproduction finding: once eps * lambda > 1, NAG's effective momentum
    # mu (1 - lambda eps) turns NEGATIVE and NAG can blow up while CM still converges.
    A = np.diag([1.0, 100.0])                      # eps * lambda = 1.5 in the steep direction
    cm = quadratic_run(A, np.zeros(2), [-10, 1], 0.015, 0.95, 100, False)
    nag = quadratic_run(A, np.zeros(2), [-10, 1], 0.015, 0.95, 100, True)
    assert np.linalg.norm(cm[-1]) < 1 and np.linalg.norm(nag[-1]) > 1e6


def test_cm_and_nag_agree_for_tiny_learning_rates():
    # "CM and NAG become equivalent when eps is small"
    A = np.diag([1.0, 5.0])
    cm = quadratic_run(A, np.zeros(2), [1, 1], 1e-5, 0.9, 50, False)
    nag = quadratic_run(A, np.zeros(2), [1, 1], 1e-5, 0.9, 50, True)
    assert np.allclose(cm, nag, atol=1e-6)


# ---- Eq. (5) ----

def test_momentum_schedule():
    assert mu_schedule(0, 0.999) == 0.5
    assert mu_schedule(250, 0.999) == pytest.approx(0.75)
    assert mu_schedule(750, 0.999) == pytest.approx(0.875)
    assert mu_schedule(10 ** 7, 0.99) == 0.99                          # capped at mu_max
    seq = [mu_schedule(t, 0.999) for t in range(0, 100000, 250)]
    assert all(a <= b for a, b in zip(seq, seq[1:]))                   # never decreases


# ---- initializations ----

def test_sparse_initialization_has_15_inputs_per_unit():
    W = sparse_init_(torch.empty(784, 1000), 15)
    assert torch.all((W != 0).sum(0) == 15)
    assert W[W != 0].std().item() == pytest.approx(1.0, rel=0.05)


def test_esn_init_spectral_radius():
    r = RNN(2, 100, 1, radius=1.1)
    assert float(torch.linalg.eigvals(r.W_hh).abs().max()) == pytest.approx(1.1, rel=1e-4)


def test_autoencoder_shape():
    m = Autoencoder()
    assert [tuple(W.shape) for W in m.W][:4] == [(784, 1000), (1000, 500), (500, 250), (250, 30)]
    assert m(torch.rand(3, 784)).shape == (3, 784)


# ---- the addition problem ----

def test_addition_problem_targets():
    x, y = addition_problem(500, 50, torch.Generator().manual_seed(0))
    marks = x[:, :, 1] + 2 / 50                     # undo the centring
    assert torch.all(marks.sum(1).round() == 2)     # exactly two marked steps
    vals = x[:, :, 0] + 0.5
    assert torch.allclose((vals * marks).sum(1) - 1.0, y[:, 0], atol=1e-5)
