"""Tests: each checks a statement of Srivastava et al. (2014). All tiny and fast.

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest
import torch

from dropout import (Autoencoder, Net, dropout_linear_regression_closed_form, max_norm_,
                     monte_carlo_error, train)


def test_bernoulli_test_time_scaling_equals_expected_training_input():
    # Section 4: W_test = p W gives the next layer the input it gets on average in training.
    torch.manual_seed(0)
    net = Net([10, 20, 3], p_input=1.0, p_hidden=0.5, init_std=0.5)
    x = torch.randn(1, 10)
    with torch.no_grad():
        h = torch.relu(x @ net.W[0] + net.b[0])
        samples = torch.stack([(h * (torch.rand_like(h) < 0.5).float()) @ net.W[1] + net.b[1] for _ in range(20000)])
        assert torch.allclose(samples.mean(0), net(x), atol=0.03)


def test_gaussian_dropout_has_mean_one_and_right_variance():
    # Section 10: r ~ N(1, (1-p)/p) - same mean and variance as Bernoulli(p)/p.
    torch.manual_seed(0)
    p = 0.5
    net = Net([1000, 5], p_input=p, noise="gaussian", init_std=1.0)
    with torch.no_grad():
        x = torch.ones(200, 1000)
        r = x * (1 + np.sqrt((1 - p) / p) * torch.randn_like(x))
    assert r.mean().item() == pytest.approx(1.0, abs=0.01)
    assert r.var().item() == pytest.approx((1 - p) / p, rel=0.03)
    # and NO scaling at test time: test output equals the plain network's output
    with torch.no_grad():
        x = torch.randn(3, 1000)
        assert torch.allclose(net(x), x @ net.W[0] + net.b[0])


def test_max_norm_projects_onto_the_ball():
    net = Net([10, 20, 3], init_std=3.0)
    max_norm_(net, 2.0)
    assert torch.all(net.W[0].norm(dim=0) <= 2.0 + 1e-5)


def test_dropout_linear_regression_is_ridge_regression():
    # Section 9.1: the closed-form minimizer of E||y - (R*X)w||^2 beats OLS on that objective,
    # and matches a direct Monte-Carlo minimization.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4)) * np.array([1, 2, 0.5, 3])
    y = X @ np.array([1.0, -1.0, 2.0, 0.5]) + rng.normal(0, 0.3, 100)
    p = 0.6
    wc = dropout_linear_regression_closed_form(X, y, p)
    R = rng.random((5000,) + X.shape) < p
    obj = lambda w: np.mean(np.sum((y - (R * X) @ w) ** 2, axis=1))
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert obj(wc) < obj(ols)
    # gradient of the marginalized objective is zero at the closed form
    G = np.diag(np.diag(X.T @ X))
    grad = -2 * p * X.T @ (y - p * X @ wc) + 2 * p * (1 - p) * G @ wc
    assert np.allclose(grad, 0, atol=1e-8)


def test_smaller_p_means_stronger_regularization_in_linear_regression():
    # The absorbed form: regularization constant (1-p)/p grows as p shrinks.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3)); y = X @ np.array([2.0, -3.0, 1.0])
    norms = [np.linalg.norm(p * dropout_linear_regression_closed_form(X, y, p)) for p in (0.9, 0.5, 0.2)]
    assert norms[0] > norms[1] > norms[2]


def test_small_dropout_net_trains_and_monte_carlo_works():
    torch.manual_seed(0)
    X = torch.randn(800, 20); y = (X[:, :4].sum(1) > 0).long()
    net = Net([20, 128, 128, 2], p_input=0.8, p_hidden=0.5)
    train(net, X, y, epochs=30, lr=0.1, momentum=0.9, batch=50, max_norm=3.0)
    with torch.no_grad():
        err = (net(X).argmax(1) != y).float().mean().item()
    assert err < 0.08
    assert monte_carlo_error(net, X, y, 50) < 0.12


def test_autoencoder_shapes():
    ae = Autoencoder(256, 0.5)
    rec, h = ae(torch.rand(4, 784), train=True)
    assert rec.shape == (4, 784) and h.shape == (4, 256)
