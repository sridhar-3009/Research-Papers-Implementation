"""Tests: each checks a statement of Simonyan & Zisserman (2015). A few seconds in total.

The full-size networks (up to 144M parameters) are only built on the "meta" device:
PyTorch creates the shapes but allocates no memory. Forward passes use shrunken copies.

Run with:  python3 -m pytest -q
"""

import random

import pytest
import torch

from vgg import (CONFIGS, PAPER_PARAMS_MILLIONS, VGG, count_params, dense_predict, init_from_A, multi_crop,
                 receptive_field, rescale_shorter_side, scale_jitter, to_fully_convolutional, weight_layers)

torch.manual_seed(0)


def tiny(name="A", **kw):
    """A shrunken VGG for fast tests: 64x64 input, channels / 16, fc 32."""
    return VGG(name, num_classes=10, input_size=64, fc=32, width_div=16, **kw)


# ---- Table 1 and Table 2 ----

def test_depths_of_table_1():
    assert {n: weight_layers(n) for n in CONFIGS} == {"A": 11, "A-LRN": 11, "B": 13, "C": 16, "D": 16, "E": 19}
    assert all(c.count("M") == 5 for c in CONFIGS.values())                       # five max-pool layers


def test_parameter_counts_of_table_2():
    for name, millions in PAPER_PARAMS_MILLIONS.items():
        n = count_params(name)
        assert round(n / 1e6) == millions, name
        built = VGG(name, device="meta")                                          # shapes only, no memory
        assert sum(p.numel() for p in built.parameters()) == n
    assert count_params("D") == 138357544                                         # VGG-16
    assert count_params("E") == 143667240                                         # VGG-19


def test_most_parameters_are_in_the_first_fc_layer():
    fc6 = (512 * 7 * 7 + 1) * 4096
    assert fc6 / count_params("D") > 0.74                                         # 102.8M of 138.4M


def test_width_doubles_after_each_pool_until_512():
    widths = [v for v in CONFIGS["E"] if isinstance(v, int)]
    assert sorted(set(widths)) == [64, 128, 256, 512]


# ---- Section 2.3: stacks of 3x3 filters ----

def test_receptive_fields_and_parameter_savings():
    assert receptive_field(2) == 5 and receptive_field(3) == 7
    C = 256
    three_3x3, one_7x7 = 3 * (3 * 3 * C * C), 7 * 7 * C * C
    assert three_3x3 == 27 * C * C and one_7x7 == 49 * C * C
    assert one_7x7 / three_3x3 == pytest.approx(1.81, abs=0.01)                    # "81% more"


def test_two_3x3_convs_really_see_a_5x5_window():
    convs = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 3, padding=1), torch.nn.Conv2d(1, 1, 3, padding=1))
    x = torch.zeros(1, 1, 11, 11, requires_grad=True)
    convs(x)[0, 0, 5, 5].backward()                                               # which inputs affect the center?
    touched = (x.grad[0, 0] != 0).nonzero()
    assert touched[:, 0].min() == 3 and touched[:, 0].max() == 7                   # rows 3..7 = 5 rows
    assert touched[:, 1].min() == 3 and touched[:, 1].max() == 7


def test_convs_preserve_size_and_pools_halve_it():
    net = VGG("D", device="meta")
    assert net.final_size == 7 and net.final_channels == 512                      # 224 / 2^5 = 7
    t = tiny("D")
    x = torch.randn(2, 3, 64, 64)
    assert t.features(x).shape == (2, 32, 2, 2) and t(x).shape == (2, 10)


def test_config_c_uses_1x1_convs():
    kernels = [m.kernel_size for m in tiny("C").conv_layers()]
    assert kernels.count((1, 1)) == 3 and kernels.count((3, 3)) == 10


# ---- Section 3.1: initialisation ----

def test_init_from_A_copies_first_four_convs_and_all_fc():
    a, e = tiny("A"), tiny("E")
    pairs = init_from_A(e, a)
    assert pairs == [(0, 0), (1, 2), (2, 4), (3, 5)]                              # matched by shape, in order
    for i, j in pairs:
        assert torch.equal(e.conv_layers()[j].weight, a.conv_layers()[i].weight)
    assert init_from_A(tiny("D"), a) == [(0, 0), (1, 2), (2, 4), (3, 5)]
    assert torch.equal(e.fc_layers()[2].weight, a.fc_layers()[2].weight)


def test_paper_init_is_std_0_1_as_written():
    net = tiny("A", init="paper")
    w = net.fc_layers()[0].weight
    assert w.std().item() == pytest.approx(0.1, rel=0.1)
    assert all(m.bias.abs().sum() == 0 for m in net.conv_layers())


# ---- Section 3.2: dense evaluation ----

def test_fully_convolutional_net_equals_original_on_training_size():
    net = tiny("B").eval()
    fcn = to_fully_convolutional(net).eval()
    x = torch.randn(3, 3, 64, 64)
    assert fcn(x).shape == (3, 10, 1, 1)
    assert torch.allclose(fcn(x)[:, :, 0, 0], net(x), atol=1e-5)


def test_dense_evaluation_works_on_bigger_images():
    fcn = to_fully_convolutional(tiny("B").eval()).eval()
    big = torch.randn(1, 3, 128, 96)
    assert fcn(big).shape == (1, 10, 3, 2)                                        # a 3x2 map of class scores
    p = dense_predict(fcn, big)
    assert p.shape == (1, 10) and torch.allclose(p.sum(1), torch.ones(1))


# ---- scales and crops ----

def test_rescale_and_scale_jitter():
    img = torch.rand(3, 300, 400)
    assert rescale_shorter_side(img, 256).shape == (3, 256, 341)
    rng = random.Random(0)
    sizes = set()
    for _ in range(20):
        crop, S = scale_jitter(img, rng=rng)
        assert crop.shape == (3, 224, 224) and 256 <= S <= 512
        sizes.add(S)
    assert len(sizes) > 5                                                         # really a different S each time


def test_multi_crop_is_50_per_scale():
    crops = multi_crop(torch.rand(3, 256, 300))
    assert crops.shape == (50, 3, 224, 224)
    assert torch.equal(crops[25], crops[0].flip(-1))
