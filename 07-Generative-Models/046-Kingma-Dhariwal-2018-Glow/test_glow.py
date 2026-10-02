import math

import torch

from glow import (ActNorm, Coupling, FlowStep, Glow, InvConv1x1, Permute, bits_per_dim, dequantize,
                  gaussian_log_prob, squeeze2d, unsqueeze2d)


def jacobian_logdet(f, x):
    """log|det| of the full Jacobian of f at a single example x (1, C, H, W), via autograd."""
    J = torch.autograd.functional.jacobian(lambda v: f(v.view_as(x))[0].flatten(), x.flatten())
    return torch.linalg.slogdet(J)[1]


def randomise(module, std=0.2):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * std)
    return module


def test_squeeze_is_an_invertible_reshape():
    x = torch.randn(2, 3, 4, 6)
    y = squeeze2d(x)
    assert y.shape == (2, 12, 2, 3) and torch.equal(unsqueeze2d(y), x)
    assert torch.equal(y[0, :4, 0, 0], x[0, 0, :2, :2].flatten())               # one 2x2 block -> 4 channels


def test_actnorm_data_dependent_init_and_logdet():
    torch.manual_seed(0)
    a = ActNorm(3)
    x = torch.randn(64, 3, 4, 4) * torch.tensor([2.0, 0.5, 5.0]).view(1, 3, 1, 1) + 3
    y, ld = a(x)
    assert torch.allclose(y.mean((0, 2, 3)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(y.std((0, 2, 3)), torch.ones(3), atol=1e-3)
    x1 = torch.randn(1, 3, 4, 4)
    assert torch.allclose(jacobian_logdet(a, x1), ld, atol=1e-4)                  # h * w * sum log|s|
    assert torch.allclose(a.reverse(a(x1)[0]), x1, atol=1e-5)


def test_invertible_1x1_conv_plain_and_lu():
    torch.manual_seed(0)
    for lu in (False, True):
        c = InvConv1x1(4, lu=lu)
        x = torch.randn(1, 4, 3, 3)
        _, ld0 = c(x)
        assert abs(ld0.item()) < 1e-4                                            # rotation init: log|det| = 0
        randomise(c)
        y, ld = c(x)
        assert torch.allclose(jacobian_logdet(c, x), ld, atol=1e-4)               # h * w * log|det W|
        assert torch.allclose(c.reverse(y), x, atol=1e-4)
    p = Permute(5, "shuffle")
    x = torch.randn(2, 5, 2, 2)
    assert torch.equal(p.reverse(p(x)[0]), x)


def test_coupling_starts_as_near_identity_and_is_invertible():
    torch.manual_seed(0)
    for additive in (True, False):
        c = Coupling(4, 16, additive)
        x = torch.randn(2, 4, 3, 3)
        y, ld = c(x)
        if additive:
            assert torch.equal(y, x) and torch.all(ld == 0)                       # zero init: exactly identity
        else:
            assert torch.allclose(y[:, :2], torch.sigmoid(torch.tensor(2.0)) * x[:, :2], atol=1e-6)
            assert torch.equal(y[:, 2:], x[:, 2:])
        randomise(c)
        y, ld = c(x[:1])
        assert torch.allclose(jacobian_logdet(c, x[:1]), ld, atol=1e-4)
        assert torch.allclose(c.reverse(y), x[:1], atol=1e-5)


def test_full_glow_change_of_variables_and_inverse():
    torch.manual_seed(0)
    g = Glow(C=1, K=2, L=2, hidden=8)
    x = torch.randn(4, 1, 4, 4)
    g.log_prob(x)                                                                 # actnorm init
    randomise(g, 0.1)
    zs, logdet, logpz = g.encode(x[:1])
    flat = lambda v: torch.cat([z.flatten(1) for z in g.encode(v)[0]], 1)
    J = torch.autograd.functional.jacobian(lambda v: flat(v.view(1, 1, 4, 4))[0], x[:1].flatten())
    assert torch.allclose(torch.linalg.slogdet(J)[1], logdet[0], atol=1e-3)       # Eq. 7, all layers together
    assert torch.allclose(g.decode(zs), x[:1], atol=1e-4)                          # exact inverse


def test_temperature_and_learned_split_prior_sampling():
    torch.manual_seed(0)
    g = Glow(C=1, K=1, L=2, hidden=8)
    g.log_prob(torch.randn(8, 1, 4, 4))
    shape = g.top_shape(1, 4, 4)
    s0 = g.decode(n=3, shape=shape, temperature=0.0)
    assert torch.allclose(s0[0], s0[1]) and torch.allclose(s0[1], s0[2])         # T = 0: the mode, every time
    s1 = g.decode(n=500, shape=shape, temperature=1.0)
    s05 = g.decode(n=500, shape=shape, temperature=0.5)
    assert s05.std() < s1.std()


def test_dequantization_and_bits():
    x8 = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
    x, c = dequantize(x8)
    assert x.min() >= -0.5 and x.max() < 0.5 and abs(c - 48 * math.log(256)) < 1e-9
    x5, c5 = dequantize(x8, n_bits=5)
    assert abs(c5 - 48 * math.log(32)) < 1e-9
    uniform = torch.tensor(0.0)                                                   # a density of 1 on [-1/2, 1/2]^M
    assert abs(bits_per_dim(uniform, c, 48).item() - 8.0) < 1e-9
    assert torch.allclose(gaussian_log_prob(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)),
                          torch.tensor([-math.log(2 * math.pi)]))
