import math

import torch
import torch.nn.functional as F

from pixelcnnpp import (PixelCNNpp, bits_per_dim, dequantized_log_density, discretized_mix_logistic_log_prob,
                        n_params, sample_discretized_mix_logistic, softmax_log_prob, split_params, to_pm1)

LEVELS = to_pm1(torch.arange(256))


def gray_params(logits, mu, log_s):
    """Build a (1, 3K, 1, 1) parameter tensor for one gray pixel."""
    return torch.cat([logits, mu, log_s]).view(1, -1, 1, 1)


def test_single_channel_mixture_sums_to_one_and_matches_eq2():
    torch.manual_seed(0)
    K = 4
    l = gray_params(torch.randn(K), torch.rand(K) * 2 - 1, torch.randn(K) - 3)
    lp = torch.stack([discretized_mix_logistic_log_prob(v.view(1, 1, 1, 1), l, K)[0, 0, 0] for v in LEVELS])
    assert abs(lp.exp().sum().item() - 1) < 1e-4
    pi = F.softmax(l[0, :K, 0, 0], 0)
    mu, s = l[0, K:2 * K, 0, 0], l[0, 2 * K:, 0, 0].exp()
    k = 100
    x = LEVELS[k]
    eq2 = (pi * (torch.sigmoid((x + 1 / 255 - mu) / s) - torch.sigmoid((x - 1 / 255 - mu) / s))).sum()
    assert torch.allclose(lp[k].exp(), eq2, atol=1e-6)


def test_edges_keep_the_tails():
    """A logistic centred near 0 (the value 0 sits at -1): P(0) includes all mass below, so it beats P(1)."""
    l = gray_params(torch.zeros(1), torch.tensor([-1.2]), torch.tensor([-2.0]))
    p = lambda v: discretized_mix_logistic_log_prob(LEVELS[v].view(1, 1, 1, 1), l, 1).exp().item()
    assert abs(p(0) - torch.sigmoid(torch.tensor((-1 + 1 / 255 + 1.2) / math.exp(-2.0))).item()) < 1e-6
    assert p(0) > p(1) > p(2)


def test_rgb_coupling_and_normalisation_per_channel():
    torch.manual_seed(0)
    K, C = 3, 3
    l = torch.randn(1, n_params(C, K), 1, 1)
    l[:, K + K * C:K + 2 * K * C] -= 3                                             # small scales
    r, g = LEVELS[40], LEVELS[200]
    total = 0.0
    for b in LEVELS:                                                               # sum over blue for fixed r, g
        x = torch.stack([r, g, b]).view(1, 3, 1, 1)
        total += discretized_mix_logistic_log_prob(x, l, K).exp().item()
    # sum_b p(r, g, b) = p(r, g); compare with the explicit mixture over components of P(r) P(g|r)
    logit_pi, means, log_s, coeffs = split_params(l, C, K)
    pi = F.softmax(logit_pi[0, :, 0, 0], 0)
    bin_p = lambda x, m, s: torch.sigmoid((x + 1 / 255 - m) / s) - torch.sigmoid((x - 1 / 255 - m) / s)
    mr, mg = means[0, 0, :, 0, 0], means[0, 1, :, 0, 0] + coeffs[0, 0, :, 0, 0] * r
    sr, sg = log_s[0, 0, :, 0, 0].exp(), log_s[0, 1, :, 0, 0].exp()
    expect = (pi * bin_p(r, mr, sr) * bin_p(g, mg, sg)).sum().item()
    assert abs(total - expect) < 1e-6


def test_sampling_matches_the_probabilities():
    torch.manual_seed(0)
    K = 2
    l = gray_params(torch.tensor([0.0, 1.0]), torch.tensor([-0.5, 0.5]), torch.tensor([-2.5, -3.0]))
    s = sample_discretized_mix_logistic(l.expand(40000, -1, -1, -1), 1, K)[:, 0, 0, 0]
    idx = torch.round((s + 1) * 127.5).long()
    emp = torch.bincount(idx, minlength=256).float() / len(idx)
    lp = discretized_mix_logistic_log_prob(LEVELS.view(256, 1, 1, 1), l.expand(256, -1, -1, -1), K)[:, 0, 0]
    assert (emp - lp.exp()).abs().sum() < 0.08                                   # total variation < 0.04


def test_stable_log_prob_for_tiny_scales():
    l = gray_params(torch.zeros(1), torch.tensor([0.0]), torch.tensor([-7.0]))
    lp = discretized_mix_logistic_log_prob(LEVELS[200].view(1, 1, 1, 1), l, 1)
    assert torch.isfinite(lp).all() and lp.item() < -50


def test_network_is_causal_with_no_blind_spot():
    torch.manual_seed(0)
    m = PixelCNNpp(3, 1, 16, 2, 0.0).eval()
    x = to_pm1(torch.randint(0, 256, (1, 3, 8, 8))).requires_grad_(True)
    m(x)[0, :, 4, 4].sum().backward()
    seen = (x.grad.abs().sum(1)[0] > 0)
    expect = torch.zeros(8, 8, dtype=torch.bool)
    expect[:4] = True
    expect[4, :4] = True
    assert torch.equal(seen, expect)                                             # all of 'above', all of 'left'


def test_variants_shapes_and_class_conditioning():
    torch.manual_seed(0)
    x = to_pm1(torch.randint(0, 256, (2, 3, 8, 8)))
    for kw in ({}, {"shortcuts": False}, {"downsample": False}):
        assert PixelCNNpp(3, 1, 8, 3, 0.0, **kw)(x).shape == (2, n_params(3, 3), 8, 8)
    m = PixelCNNpp(3, 1, 8, 3, 0.0, n_classes=10).eval()
    h = F.one_hot(torch.tensor([1, 7]), 10).float()
    assert not torch.allclose(m(x, h), m(x, F.one_hot(torch.tensor([2, 2]), 10).float()))
    assert n_params(3, 10) == 100                                                 # 10 x (1 + 6 + 3)


def test_softmax_and_dequantized_ablations():
    torch.manual_seed(0)
    x = to_pm1(torch.randint(0, 256, (1, 1, 2, 2)))
    logits = torch.randn(1, 1, 256, 2, 2)
    lp = softmax_log_prob(x, logits)
    idx = torch.round((x + 1) * 127.5).long()
    assert torch.allclose(lp, F.log_softmax(logits, 2).gather(2, idx.unsqueeze(2)).squeeze(2).squeeze(1))
    l = gray_params(torch.zeros(1), torch.tensor([0.1]), torch.tensor([-3.0]))
    v = LEVELS[140].view(1, 1, 1, 1)
    u = (torch.rand(5000, 1, 1, 1) - 0.5) * (2 / 255)
    bound = dequantized_log_density(v + u, l.expand(5000, -1, -1, -1), 1).mean()
    exact = discretized_mix_logistic_log_prob(v, l, 1)[0, 0, 0]
    assert bound <= exact + 1e-4 and exact - bound < 0.05                         # Jensen: a slightly lower bound
    assert abs(bits_per_dim(-math.log(256) * 3072, 3072) - 8.0) < 1e-9
