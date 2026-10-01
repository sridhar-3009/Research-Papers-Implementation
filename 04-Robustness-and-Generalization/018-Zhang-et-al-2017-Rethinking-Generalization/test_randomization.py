"""Tests: each checks a statement or tool of Zhang et al. (2017). About a second in total.

Run with:  python3 -m pytest -q
"""

import pytest
import torch

from randomization import (MLP, SmallAlexNet, SmallInception, center_crop, corrupt_labels, count_params,
                           finite_sample_network, finite_sample_predict, gaussian_images, min_norm_interpolant,
                           per_image_whitening, rademacher_fit, random_pixels, sgd_least_squares, shuffle_pixels)

g = lambda s=0: torch.Generator().manual_seed(s)


# ---- Section 2.1: the randomizations ----

def test_label_corruption_rate():
    y = torch.zeros(100000, dtype=torch.long)
    for p in (0.0, 0.5, 1.0):
        changed = (corrupt_labels(y, p, generator=g()) != y).float().mean().item()
        assert changed == pytest.approx(p * 0.9, abs=0.01)       # a random class equals the old one 1/10 of the time


def test_shuffled_pixels_use_one_permutation_for_all_images():
    X = torch.rand(5, 3, 4, 4)
    S, perm = shuffle_pixels(X, g())
    assert torch.equal(S[0, 1].flatten(), X[0, 1].flatten()[perm])
    assert torch.equal(S[3, 2].flatten(), X[3, 2].flatten()[perm])        # same permutation everywhere
    T, _ = shuffle_pixels(X, perm=perm)
    assert torch.equal(S, T)                                               # reusable for the test set


def test_random_pixels_permute_each_image_differently():
    X = torch.arange(2 * 3 * 16, dtype=torch.float32).view(2, 3, 4, 4)
    X[1] = X[0]
    R = random_pixels(X, g())
    assert torch.equal(R[0].flatten(1).sort(1).values, X[0].flatten(1).sort(1).values)   # same pixel values...
    assert not torch.equal(R[0], R[1])                                                    # ...different orders
    assert torch.equal(R[0, 0].flatten().argsort(), R[0, 1].flatten().argsort())          # channels move together


def test_gaussian_images_match_the_dataset_statistics():
    X = torch.rand(2000, 3, 8, 8) * torch.tensor([1.0, 0.5, 0.2]).view(1, 3, 1, 1)
    G = gaussian_images(X, g())
    assert torch.allclose(G.mean((0, 2, 3)), X.mean((0, 2, 3)), atol=0.01)
    assert torch.allclose(G.std((0, 2, 3)), X.std((0, 2, 3)), rtol=0.03)


def test_preprocessing():
    X = torch.rand(4, 3, 32, 32)
    C = center_crop(X)
    assert C.shape == (4, 3, 28, 28) and torch.equal(C, X[..., 2:30, 2:30])
    W = per_image_whitening(C)
    assert torch.allclose(W.mean((1, 2, 3)), torch.zeros(4), atol=1e-5)
    assert torch.allclose(W.std((1, 2, 3), unbiased=False), torch.ones(4), atol=1e-4)


# ---- Appendix A / Table 1: the models ----

def test_parameter_counts_of_table_1():
    assert count_params(SmallInception()) == 1649402
    assert count_params(SmallInception(bn=False)) == 1649402            # "Inception w/o BatchNorm": same count
    assert count_params(SmallAlexNet()) == 1387786
    assert count_params(MLP(3)) == 1735178
    assert count_params(MLP(1)) == 1209866


def test_models_run_on_28x28_inputs():
    x = torch.randn(2, 3, 28, 28)
    for m in (SmallInception(), SmallAlexNet(), MLP(1)):
        assert m.eval()(x).shape == (2, 10)
    assert SmallInception().body(x).shape[-2:] == (7, 7)                 # "Mean Pooling 7x7 kernel (global)"


# ---- Section 4, Theorem 1 ----

def test_two_layer_relu_net_with_2n_plus_d_weights_fits_any_labels():
    X = torch.randn(200, 30)
    y = torch.randn(200)                                                  # arbitrary real "labels"
    a, b, w = finite_sample_network(X, y, g())
    assert a.numel() + b.numel() + w.numel() == 2 * 200 + 30
    assert torch.allclose(finite_sample_predict(a, b, w, X), y.double(), atol=1e-6)


def test_finite_sample_net_fits_random_class_labels_too():
    X = torch.randn(100, 5)
    y = torch.randint(0, 10, (100,)).float()
    a, b, w = finite_sample_network(X, y, g(1))
    assert torch.equal(finite_sample_predict(a, b, w, X).round().long(), y.long())


# ---- Section 2.2: Rademacher ----

def test_overparameterized_learner_has_rademacher_complexity_one():
    X = torch.randn(50, 200)                                              # d > n
    fit = lambda X, s: X.double() @ min_norm_interpolant(X, s)
    assert rademacher_fit(fit, X, trials=10, generator=g()) == pytest.approx(1.0)
    X_small = torch.randn(200, 3)                                         # d << n: can't fit random signs
    fit_small = lambda X, s: X.double() @ torch.linalg.lstsq(X.double(), s.double()[:, None]).solution[:, 0]
    assert rademacher_fit(fit_small, X_small, trials=10, generator=g()) < 0.3


# ---- Section 5: implicit regularization ----

def test_sgd_from_zero_converges_to_the_minimum_norm_solution():
    X = torch.randn(20, 60)
    y = torch.randn(20)
    w_kernel = min_norm_interpolant(X, y)
    w_sgd = sgd_least_squares(X, y, epochs=400, generator=g())
    assert torch.allclose(X.double() @ w_sgd, y.double(), atol=1e-4)     # fits exactly
    assert torch.allclose(w_sgd, w_kernel, atol=1e-3)                    # and it IS the min-norm solution
    other = w_kernel + (torch.eye(60, dtype=torch.float64) - torch.linalg.pinv(X.double()) @ X.double()) @ torch.randn(60, dtype=torch.float64)
    assert torch.allclose(X.double() @ other, y.double(), atol=1e-6)     # another exact fit...
    assert other.norm() > w_kernel.norm()                                 # ...with a bigger norm


def test_hessian_of_linear_least_squares_does_not_depend_on_w():
    # "the curvature of all optimal solutions is the same": for squared loss the Hessian is X^T X / n
    X = torch.randn(10, 30)
    y = torch.randn(10)
    loss = lambda w: 0.5 * ((X @ w - y) ** 2).mean()
    H1 = torch.autograd.functional.hessian(loss, torch.zeros(30))
    H2 = torch.autograd.functional.hessian(loss, torch.randn(30))
    assert torch.allclose(H1, H2) and torch.allclose(H1, X.T @ X / 10, atol=1e-5)
