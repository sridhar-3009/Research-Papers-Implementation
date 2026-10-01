"""Tests: each checks a statement or tool of Szegedy et al. (2013). A few seconds in total.

Run with:  python3 -m pytest -q
"""

import pytest
import torch
import torch.nn as nn

from adversarial import (AE400, amplify, conv_as_matrix, conv_operator_norm_fft, conv_operator_norm_power,
                         decay_penalty, distortion, fc_net, fc_operator_norm, gaussian_distort, lbfgs_attack,
                         lipschitz_upper_bound, minimal_adversarial, predict, random_direction, top_activating,
                         unit_direction)

torch.manual_seed(0)


def boundary_model(k=20.0):
    """2-D linear classifier: class 0 if x0 > 0.5, class 1 if x0 < 0.5."""
    m = nn.Linear(2, 2)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[k, 0.0], [-k, 0.0]]))
        m.bias.copy_(torch.tensor([-0.5 * k, 0.5 * k]))
    return m


# ---- Section 4.1: the attack ----

def test_attack_reaches_the_target_class_inside_the_box():
    m = boundary_model()
    x = torch.tensor([0.3, 0.5])
    assert predict(m, x) == 1
    z = lbfgs_attack(m, x, target=0, c=0.1)
    assert predict(m, z) == 0
    assert z.min() >= 0 and z.max() <= 1


def test_minimal_adversarial_finds_the_closest_point_over_the_boundary():
    # For a linear boundary the true minimal L2 perturbation is the distance to it: 0.5 - 0.3 = 0.2,
    # straight along x0. The attack must find (almost) exactly that.
    m = boundary_model()
    x = torch.tensor([0.3, 0.5])
    z, c = minimal_adversarial(m, x, target=0)
    r = z - x
    assert predict(m, z) == 0
    assert 0.2 <= r.norm().item() < 0.26
    assert abs(r[1].item()) < 1e-3                       # no wasted movement along x1


def test_bigger_c_means_smaller_perturbation():
    m = boundary_model()
    x = torch.tensor([0.3, 0.5])
    small_c = (lbfgs_attack(m, x, 0, c=0.01) - x).norm()
    big_c = (lbfgs_attack(m, x, 0, c=1.0) - x).norm()
    assert big_c < small_c


def test_distortion_measure_is_the_perturbation_stddev():
    x = torch.zeros(784)
    assert distortion(x, x + 0.06) == pytest.approx(0.06)


# ---- Section 4.2 baselines ----

def test_gaussian_noise_and_amplify():
    g = torch.Generator().manual_seed(0)
    x = torch.full((784,), 0.5)
    noisy = gaussian_distort(x, 0.1, generator=g)
    assert noisy.min() >= 0 and noisy.max() <= 1 and distortion(x, noisy) == pytest.approx(0.1, rel=0.1)
    x_adv = x + 0.06 * torch.randn(784, generator=g).sign() * 0.5
    amp = amplify(x, x_adv, 0.1)
    assert distortion(x, amp) == pytest.approx(0.1, rel=1e-3)
    cos = torch.nn.functional.cosine_similarity(amp - x, x_adv - x, dim=0)
    assert cos > 0.999                                    # same direction, just bigger


def test_paper_weight_decay_divides_by_layer_width():
    net = fc_net((100,))
    w1, w2 = [m.weight for m in net if isinstance(m, nn.Linear)]
    expected = 1e-5 * (w1 ** 2).sum() / 100 + 1e-6 * (w2 ** 2).sum() / 10
    assert torch.allclose(decay_penalty(net, [1e-5, 1e-6]), expected)


# ---- Section 3: units vs random directions ----

def test_top_activating_for_unit_and_random_direction():
    feats = torch.randn(50, 6)
    top = top_activating(feats, unit_direction(6, 2), k=5)
    assert torch.equal(top, torch.argsort(feats[:, 2], descending=True)[:5])
    v = random_direction(6, torch.Generator().manual_seed(1))
    assert v.norm().item() == pytest.approx(1.0)
    assert len(top_activating(feats, v, 5)) == 5


# ---- Section 4.3: operator norms ----

def test_fc_norm_is_largest_singular_value_and_bounds_a_relu_layer():
    W = torch.randn(30, 20)
    L = fc_operator_norm(W)
    assert L == pytest.approx(torch.linalg.svdvals(W)[0].item(), rel=1e-6)
    for _ in range(20):
        x, r = torch.randn(20), torch.randn(20) * 0.1
        assert (torch.relu(W @ (x + r)) - torch.relu(W @ x)).norm() <= L * r.norm() + 1e-5


def test_conv_norm_fourier_formula_equals_brute_force():
    w = torch.randn(4, 3, 3, 3)
    n = 6
    exact = torch.linalg.matrix_norm(conv_as_matrix(w, n), ord=2).item()
    assert conv_operator_norm_fft(w, n) == pytest.approx(exact, rel=1e-6)
    assert conv_operator_norm_power(w, n) == pytest.approx(exact, rel=1e-4)


def test_conv_norm_with_stride():
    w = torch.randn(4, 2, 3, 3)
    exact = torch.linalg.matrix_norm(conv_as_matrix(w, 8, stride=2), ord=2).item()
    assert conv_operator_norm_power(w, 8, stride=2) == pytest.approx(exact, rel=1e-4)


def test_lipschitz_bound_of_a_whole_network_holds():
    net = fc_net((50, 50), act="relu")
    lins = [m for m in net if isinstance(m, nn.Linear)]
    L = lipschitz_upper_bound([fc_operator_norm(m.weight) for m in lins])
    with torch.no_grad():
        for _ in range(20):
            x, r = torch.rand(1, 784), torch.randn(1, 784) * 0.01
            assert (net(x + r) - net(x)).norm() <= L * r.norm() + 1e-5


# ---- the models ----

def test_table_1_models_shapes():
    x = torch.rand(3, 1, 28, 28)
    for hidden in ((), (100, 100), (200, 200), (123, 456)):
        assert fc_net(hidden)(x).shape == (3, 10)
    ae = AE400()
    assert ae(x).shape == (3, 10) and ae.reconstruct(x).shape == (3, 784)
