"""Tests: each checks a statement of Krizhevsky, Sutskever & Hinton (2012). A few seconds in total.

Run with:  python3 -m pytest -q
"""

import pytest
import torch
import torch.nn.functional as F

from alexnet import (AlexNet, DivideOnPlateau, PaperDropout, SmallCifarNet, count_params, local_response_norm,
                     paper_sgd_step, pca_color_augment, random_crop_flip, rgb_pca, ten_crop)

torch.manual_seed(0)


@pytest.fixture(scope="module")
def net():
    return AlexNet().eval()


# ---- the architecture (Section 3.5, Figure 2) ----

def test_about_60_million_parameters(net):
    per_layer = {n: sum(p.numel() for p in m.parameters()) for n, m in net.named_children()
                 if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))}
    assert per_layer == {"conv1": 34944, "conv2": 307456, "conv3": 885120, "conv4": 663936, "conv5": 442624,
                         "fc6": 37752832, "fc7": 16781312, "fc8": 4097000}
    assert count_params(net) == 60965224                     # "60 million parameters"
    assert per_layer["fc6"] + per_layer["fc7"] + per_layer["fc8"] > 0.9 * count_params(net)   # almost all in the fc layers


def test_neuron_counts_of_figure_2(net):
    x = torch.randn(1, 3, 224, 224)
    sizes = []
    h = net.act(net.conv1(x)); sizes.append(h[0].numel())
    h = net.pool(net.norm(h))
    h = net.act(net.conv2(h)); sizes.append(h[0].numel())
    h = net.pool(net.norm(h))
    h = net.act(net.conv3(h)); sizes.append(h[0].numel())
    h = net.act(net.conv4(h)); sizes.append(h[0].numel())
    h = net.act(net.conv5(h)); sizes.append(h[0].numel())
    # paper: 253,440-186,624-64,896-64,896-43,264. The first number doesn't fit 96 maps of 55x55
    # (= 290,400); the other four match exactly.
    assert sizes == [290400, 186624, 64896, 64896, 43264]
    assert (224 - 11) / 4 + 1 == 54.25                        # why conv1 needs padding (or a 227 crop)
    assert net(x).shape == (1, 1000)


def test_two_gpu_split_in_conv4(net):
    # conv4's kernels are 3x3x192: each half only sees the 192 maps "on its own GPU".
    assert net.conv4.weight.shape == (384, 192, 3, 3) and net.conv3.weight.shape == (384, 256, 3, 3)
    x = torch.randn(1, 384, 13, 13)
    x2 = x.clone(); x2[:, 192:] = torch.randn(1, 192, 13, 13)          # change only "GPU 2"'s input maps
    y, y2 = net.conv4(x), net.conv4(x2)
    assert torch.allclose(y[:, :192], y2[:, :192])                     # GPU 1's outputs don't notice
    assert not torch.allclose(y[:, 192:], y2[:, 192:])


def test_paper_initialization(net):
    assert net.conv2.bias.eq(1).all() and net.fc7.bias.eq(1).all()
    assert net.conv1.bias.eq(0).all() and net.conv3.bias.eq(0).all() and net.fc8.bias.eq(0).all()
    assert abs(net.fc6.weight.std().item() - 0.01) < 1e-4


# ---- Section 3.1: ReLU ----

def test_relu_does_not_saturate():
    x = torch.tensor([0.5, 5.0, 50.0], requires_grad=True)
    g_relu = torch.autograd.grad(F.relu(x).sum(), x)[0]
    g_tanh = torch.autograd.grad(torch.tanh(x).sum(), x)[0]
    assert torch.equal(g_relu, torch.ones(3))                          # gradient stays 1 for any positive input
    assert g_tanh[2] < 1e-10 and g_tanh[1] < 1e-3                       # tanh's gradient vanishes


# ---- Section 3.3: local response normalization ----

def test_lrn_matches_formula_and_torch():
    a = torch.relu(torch.randn(2, 8, 4, 4)) * 30
    b = local_response_norm(a)
    i = 0                                                                # the edge: sum over maps 0..2 only
    expected = a[:, i] / (2 + 1e-4 * (a[:, 0:3] ** 2).sum(1)) ** 0.75
    assert torch.allclose(b[:, i], expected)
    i = 4                                                                # the middle: maps 2..6
    expected = a[:, i] / (2 + 1e-4 * (a[:, 2:7] ** 2).sum(1)) ** 0.75
    assert torch.allclose(b[:, i], expected)
    # torch's version divides alpha by n, so alpha * n there = our alpha
    assert torch.allclose(b, F.local_response_norm(a, size=5, alpha=5e-4, beta=0.75, k=2.0), atol=1e-6)


def test_lrn_big_activity_inhibits_neighbours():
    a = torch.ones(1, 5, 1, 1)
    quiet = local_response_norm(a)[0, 2, 0, 0]
    a[0, 1] = 200.0                                                       # a loud neighbour
    assert local_response_norm(a)[0, 2, 0, 0] < 0.8 * quiet


# ---- Section 3.4: overlapping pooling ----

def test_overlapping_and_plain_pooling_give_the_same_size():
    x = torch.randn(1, 96, 55, 55)
    assert torch.nn.MaxPool2d(3, 2)(x).shape == torch.nn.MaxPool2d(2, 2)(x).shape == (1, 96, 27, 27)
    small = torch.randn(2, 3, 32, 32)
    assert SmallCifarNet(overlap=True)(small).shape == SmallCifarNet(overlap=False)(small).shape == (2, 10)


# ---- Section 4.1: data augmentation ----

def test_crops_and_flips():
    img = torch.arange(3 * 256 * 256, dtype=torch.float32).view(3, 256, 256)
    g = torch.Generator().manual_seed(0)
    assert random_crop_flip(img, generator=g).shape == (3, 224, 224)
    crops = ten_crop(img)
    assert crops.shape == (10, 3, 224, 224)
    assert torch.equal(crops[0], img[:, :224, :224])                     # top-left corner
    assert torch.equal(crops[4], img[:, 16:240, 16:240])                 # center
    assert torch.equal(crops[5], crops[0].flip(-1))                      # its mirror image
    assert 32 * 32 * 2 == 2048                                           # "a factor of 2048"


def test_pca_color_shifts_every_pixel_by_the_same_rgb_vector():
    torch.manual_seed(1)
    mix = torch.tensor([[1.0, 0.9, 0.8], [0.0, 0.3, 0.1], [0.0, 0.0, 0.2]])      # correlated R, G, B
    imgs = (torch.randn(50 * 16, 3) @ mix).view(50, 4, 4, 3).permute(0, 3, 1, 2).contiguous()
    vals, vecs = rgb_pca(imgs)
    assert torch.allclose(vecs @ torch.diag(vals) @ vecs.T, torch.cov(imgs.permute(0, 2, 3, 1).reshape(-1, 3).T), atol=1e-4)
    out = pca_color_augment(imgs[0], vals, vecs)
    diff = out - imgs[0]
    assert torch.allclose(diff, diff[:, :1, :1].expand_as(diff))         # the same shift everywhere
    assert diff.abs().max() > 0


# ---- Section 4.2: dropout ----

def test_paper_dropout_matches_expectation_at_test_time():
    d = PaperDropout()
    x = torch.ones(200000)
    d.train()
    frac_kept = (d(x) != 0).float().mean().item()
    assert frac_kept == pytest.approx(0.5, abs=0.01)
    d.eval()
    assert torch.allclose(d(x), 0.5 * x)                                  # E[training output] = 0.5 x


# ---- Section 5: learning ----

def test_update_rule_equals_torch_sgd_with_momentum_and_weight_decay():
    torch.manual_seed(2)
    w1 = torch.randn(5, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_()
    A = torch.randn(5, 5)
    v = [torch.zeros(5)]
    opt = torch.optim.SGD([w2], lr=0.01, momentum=0.9, weight_decay=0.0005)
    for _ in range(20):
        w1.grad = None
        ((A @ w1) ** 2).sum().backward()
        paper_sgd_step([w1], v, lr=0.01)
        opt.zero_grad(); ((A @ w2) ** 2).sum().backward(); opt.step()
    assert torch.allclose(w1, w2, atol=1e-6)


def test_divide_learning_rate_by_10_on_plateau():
    s = DivideOnPlateau(lr=0.01, patience=2)
    lrs = [s.update(e) for e in (0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3)]
    assert lrs == pytest.approx([0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.0001])
