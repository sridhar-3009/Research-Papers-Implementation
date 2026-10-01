"""Tests: each checks a statement of LeCun et al. (1998). All tiny (a couple of seconds).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lenet5 import (C3_TABLE, LeNet5, Subsample, conv2d_naive, distort, gauss_newton_diag, map_loss, mse_loss,
                    paper_eta, prepare, rbf_prototypes, sdlm_step, squash)

torch.manual_seed(0)


# ---- what a convolution is ----

def test_naive_convolution_matches_torch():
    rng = np.random.default_rng(0)
    x, w, b = rng.normal(size=(2, 9, 9)), rng.normal(size=(3, 2, 5, 5)), rng.normal(size=3)
    ours = conv2d_naive(x, w, b)
    theirs = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b))[0].numpy()
    assert ours.shape == (3, 5, 5) and np.allclose(ours, theirs)


def test_convolution_is_shift_equivariant():
    # Section II.A: "if the input image is shifted, the feature map output will be shifted by the same amount"
    x = torch.zeros(1, 1, 20, 20); x[0, 0, 5:9, 6:8] = 1.0
    w = torch.randn(4, 1, 5, 5)
    shifted = torch.roll(x, shifts=(3, 2), dims=(2, 3))
    out, out_s = F.conv2d(x, w), F.conv2d(shifted, w)
    assert torch.allclose(torch.roll(out, shifts=(3, 2), dims=(2, 3)), out_s, atol=1e-6)


# ---- the architecture: the paper's exact numbers ----

def test_layer_shapes():
    net = LeNet5()
    x = torch.randn(2, 1, 32, 32)
    shapes = []
    h = squash(net.C1(x)); shapes.append(h.shape[1:])
    h = net.S2(h); shapes.append(h.shape[1:])
    h = squash(net.C3(h)); shapes.append(h.shape[1:])
    h = net.S4(h); shapes.append(h.shape[1:])
    h = squash(net.C5(h)); shapes.append(h.shape[1:])
    assert [tuple(s) for s in shapes] == [(6, 28, 28), (6, 14, 14), (16, 10, 10), (16, 5, 5), (120, 1, 1)]
    assert net.features(x).shape == (2, 84) and net(x).shape == (2, 10)


def test_parameter_counts_match_section_II_B():
    net = LeNet5()
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(net.C1) == 156
    assert count(net.S2) == 12
    assert net.C3.n_params() == 1516
    assert count(net.S4) == 32
    assert count(net.C5) == 48120
    assert count(net.F6) == 10164
    assert net.n_trainable() == 60000                              # "only 60,000 trainable free parameters"


def test_connection_counts_match_section_II_B():
    # connections = (units in the layer) x (inputs per unit, including the bias)
    assert 28 * 28 * 6 * (25 + 1) == 122304                         # C1
    assert 14 * 14 * 6 * (4 + 1) == 5880                            # S2
    assert 10 * 10 * LeNet5().C3.n_params() == 151600               # C3
    assert 5 * 5 * 16 * (4 + 1) == 2000                             # S4


def test_c3_table_structure():
    sizes = [len(t) for t in C3_TABLE]
    assert sizes == [3] * 6 + [4] * 9 + [6]
    assert all(set(C3_TABLE[i]) == {i % 6, (i + 1) % 6, (i + 2) % 6} for i in range(6))   # contiguous triples


def test_c3_disconnected_weights_never_matter():
    net = LeNet5()
    x = torch.randn(3, 6, 14, 14)
    net.C3(x).sum().backward()
    g = net.C3.weight.grad
    assert torch.all(g[0, 3:] == 0) and torch.any(g[0, :3] != 0)    # map 0 only reads S2 maps 0, 1, 2
    before = net.C3(x).detach()
    with torch.no_grad():
        net.C3.weight[0, 5] += 100.0                                 # change a disconnected weight...
    assert torch.allclose(net.C3(x), before)                         # ...nothing changes


def test_subsampling_is_sum_times_coefficient_plus_bias():
    s = Subsample(1)
    with torch.no_grad():
        s.coef.fill_(0.5); s.bias.fill_(0.1)
    x = torch.arange(16.0).view(1, 1, 4, 4)
    expected = squash(0.5 * torch.tensor([[0 + 1 + 4 + 5, 2 + 3 + 6 + 7], [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]]) + 0.1)
    assert torch.allclose(s(x)[0, 0], expected)


def test_squashing_function_appendix_A():
    a = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(squash(a), torch.tensor([-1.0, 0.0, 1.0]), atol=1e-3)      # f(+-1) = +-1
    assert squash(torch.tensor(100.0)) == pytest.approx(1.7159, abs=1e-4)          # asymptote A
    # |f''| is largest near |a| = 1: the RBF targets +-1 sit where the sigmoid is most non-linear
    t = torch.linspace(0, 3, 3001, requires_grad=True)
    d1 = torch.autograd.grad(squash(t).sum(), t, create_graph=True)[0]
    d2 = torch.autograd.grad(d1.sum(), t)[0]
    assert abs(t[d2.abs().argmax()].item() - 1.0) < 0.05


# ---- the output layer and the loss ----

def test_prototypes_are_distinct_plus_minus_one_bitmaps():
    P = rbf_prototypes()
    assert P.shape == (10, 84) and set(P.unique().tolist()) == {-1.0, 1.0}
    d = torch.cdist(P, P)
    assert (d + torch.eye(10) * 1e9).min() > 0                        # all ten codes differ


def test_rbf_output_is_squared_distance_and_argmin_predicts():
    net = LeNet5()
    x = torch.randn(4, 1, 32, 32)
    h = net.features(x)
    assert torch.allclose(net(x), torch.cdist(h, net.prototypes) ** 2, atol=1e-3)
    assert torch.equal(net.predict(x), net(x).argmin(1))


def test_mse_collapses_with_learned_prototypes_but_map_does_not():
    # Section II.C: if the RBF centers can move, MSE has a trivial solution: all centers equal to a
    # constant F6 state. Then every penalty is 0 whatever the input. The MAP criterion stays positive.
    y_collapsed = torch.zeros(5, 10)
    labels = torch.tensor([0, 3, 5, 7, 9])
    assert mse_loss(y_collapsed, labels) == 0
    assert map_loss(y_collapsed, labels) > np.log(10) - 1e-6          # log(e^-j + 10) > log 10


def test_map_loss_prefers_separated_penalties():
    labels = torch.tensor([2])
    good = torch.full((1, 10), 50.0); good[0, 2] = 1.0              # correct class close, others far
    bad = torch.full((1, 10), 1.0)                                  # all equally close
    assert map_loss(good, labels) < map_loss(bad, labels)
    assert mse_loss(good, labels) == mse_loss(bad, labels)           # MSE can't tell them apart


# ---- data ----

def test_input_format():
    img = torch.zeros(1, 28, 28, dtype=torch.uint8); img[0, 10, 10] = 255
    x = prepare(img)
    assert x.shape == (1, 1, 32, 32)
    assert x.min().item() == pytest.approx(-0.1) and x.max().item() == pytest.approx(1.175)


def test_distortion_with_zero_strength_is_identity():
    x = prepare(torch.randint(0, 256, (3, 28, 28), dtype=torch.uint8))
    same = distort(x, max_shift=0, max_scale=0, max_squeeze=0, max_shear=0)
    assert torch.allclose(same, x, atol=1e-5)
    moved = distort(x)
    assert moved.shape == x.shape and not torch.allclose(moved, x)


# ---- Appendix C ----

def test_gauss_newton_diag_on_a_linear_layer():
    # For h = W u (no squashing), the GN diagonal of ||h - w||^2 in W_ij is 2 u_j^2.
    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.W = torch.nn.Parameter(torch.randn(3, 4))
        def features(self, u):
            return u @ self.W.T
    torch.manual_seed(1)
    u = torch.randn(1, 4)
    h = gauss_newton_diag(Lin(), u.repeat(4000, 1))[0]
    assert torch.allclose(h, 2 * u ** 2 * torch.ones(3, 1), rtol=0.1)


def test_sdlm_step_reduces_the_loss_and_schedule():
    net = LeNet5()
    x, y = torch.randn(8, 1, 32, 32), torch.randint(0, 10, (8,))
    h = gauss_newton_diag(net, x[:4])
    before = mse_loss(net(x), y).item()
    for _ in range(5):
        sdlm_step(net, mse_loss(net(x), y), h, eta=1e-4)     # 5e-4 overshoots when one noise batch is repeated
    assert mse_loss(net(x), y).item() < before
    assert [paper_eta(e) for e in (0, 1, 2, 5, 8, 12, 19)] == [5e-4, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5, 1e-5]
