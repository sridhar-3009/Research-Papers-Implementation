"""Tests: each checks a statement of He et al. (2016). A few seconds in total.

Big ImageNet nets are built inside `with torch.device("meta")`: shapes only, no memory.

Run with:  python3 -m pytest -q
"""

import pytest
import torch
import torch.nn as nn

from resnet import (PAPER_GFLOPS, BasicBlock, Bottleneck, CifarResNet, ImageNetResNet, ZeroPadShortcut, count_macs,
                    count_params, layer_responses, weight_layers)

torch.manual_seed(0)


def meta(depth, **kw):
    with torch.device("meta"):
        return ImageNetResNet(depth, **kw)


# ---- Table 1: ImageNet architectures ----

def test_depths_of_table_1():
    for d in (18, 34, 50, 101, 152):
        assert weight_layers(meta(d)) == d


def test_flops_of_table_1():
    for d, gflops in PAPER_GFLOPS.items():
        assert count_macs(meta(d)) / 1e9 == pytest.approx(gflops, rel=0.04), d
    assert count_macs(meta(152)) < 15.3e9                     # still cheaper than VGG-16 (15.3) and VGG-19 (19.6)


def test_parameter_counts_option_b():
    counts = {d: count_params(meta(d)) for d in (18, 34, 50, 101, 152)}
    assert counts == {18: 11689512, 34: 21797672, 50: 25557032, 101: 44549160, 152: 60192808}


def test_plain_and_residual_have_the_same_parameters_with_option_a():
    # "the residual networks have no extra parameter compared to their plain counterparts"
    assert count_params(meta(34, residual=False)) == count_params(meta(34, option="A"))
    assert count_params(CifarResNet(3, residual=False)) == count_params(CifarResNet(3))


def test_options_a_b_c_add_parameters_in_order():
    a, b, c = (count_params(meta(34, option=o)) for o in "ABC")
    assert a < b < c


def test_output_shape():
    net = ImageNetResNet(18).eval()
    assert net(torch.randn(1, 3, 224, 224)).shape == (1, 1000)


# ---- Section 4.2 / Table 6: CIFAR-10 networks ----

def test_cifar_depths_and_parameter_counts_of_table_6():
    table6 = {3: (20, 0.27), 5: (32, 0.46), 7: (44, 0.66), 9: (56, 0.85), 18: (110, 1.7), 200: (1202, 19.4)}
    for n, (depth, millions) in table6.items():
        if n == 200:
            with torch.device("meta"):
                net = CifarResNet(n)
        else:
            net = CifarResNet(n)
        assert net.depth == depth == weight_layers(net)
        digits = 2 if millions < 1 else 1
        assert round(count_params(net) / 1e6, digits) == millions, depth


def test_cifar_forward_and_map_sizes():
    net = CifarResNet(3).eval()
    x = torch.randn(2, 3, 32, 32)
    h = torch.relu(net.bn1(net.conv1(x)))
    sizes = []
    for b in net.blocks:
        h = b(h)
        sizes.append(tuple(h.shape[1:]))
    assert sizes[0] == (16, 32, 32) and sizes[3] == (32, 16, 16) and sizes[6] == (64, 8, 8)
    assert net(x).shape == (2, 10)


# ---- Section 3.1-3.2: the residual idea ----

def test_zero_residual_makes_a_block_an_identity():
    # "if an identity mapping were optimal, ... push the residual to zero": with the last BN's
    # scale at 0, F(x) = 0 and the block passes (non-negative) input straight through.
    blk = BasicBlock(16, 16).eval()
    nn.init.zeros_(blk.bn2.weight); nn.init.zeros_(blk.bn2.bias)
    x = torch.relu(torch.randn(2, 16, 8, 8))
    assert torch.allclose(blk(x), x)


def test_deeper_net_can_copy_a_shallower_one():
    # The construction argument: a deeper net = the shallow net + identity layers has the SAME output.
    shallow = CifarResNet(3).eval()
    deep = CifarResNet(5).eval()
    with torch.no_grad():
        deep.conv1.load_state_dict(shallow.conv1.state_dict()); deep.bn1.load_state_dict(shallow.bn1.state_dict())
        deep.fc.load_state_dict(shallow.fc.state_dict())
        for stage in range(3):
            for j in range(5):
                d = deep.blocks[stage * 5 + j]
                if j < 3:
                    d.load_state_dict(shallow.blocks[stage * 3 + j].state_dict())
                else:                                       # the extra blocks: residual = 0
                    nn.init.zeros_(d.bn2.weight); nn.init.zeros_(d.bn2.bias)
    x = torch.randn(4, 3, 32, 32)
    assert torch.allclose(deep(x), shallow(x), atol=1e-5)


def test_gradient_passes_through_the_shortcut():
    # y = F(x) + x  =>  dy/dx = dF/dx + 1: even if F's gradient is 0, the signal still flows.
    blk = BasicBlock(8, 8).eval()
    nn.init.zeros_(blk.bn2.weight)
    x = (torch.rand(1, 8, 4, 4) + 0.1).requires_grad_()
    blk(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_option_a_shortcut_pads_zeros_and_subsamples():
    s = ZeroPadShortcut(16, 32, 2)
    x = torch.randn(1, 16, 8, 8)
    y = s(x)
    assert y.shape == (1, 32, 4, 4)
    assert torch.equal(y[:, :16], x[:, :, ::2, ::2]) and torch.all(y[:, 16:] == 0)


def test_bottleneck_shapes():
    b = Bottleneck(256, 128, stride=2).eval()
    assert b(torch.randn(1, 256, 56, 56)).shape == (1, 512, 28, 28)
    assert b.conv1.stride == (2, 2) and b.conv2.stride == (1, 1)          # the paper's (v1) placement


def test_layer_responses_are_recorded_for_every_3x3_layer():
    r = layer_responses(CifarResNet(3), torch.randn(8, 3, 32, 32))
    assert len(r) == 18 and all(v > 0 for v in r)
