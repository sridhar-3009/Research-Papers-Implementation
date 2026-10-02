import math

import torch
import torch.nn as nn

from vlae import (AFPrior, LocalPixelCNN, VLAE, WindowAR1d, code_lengths, iaf040, receptive_field,
                  window_mask)


def test_window_receptive_field_is_exactly_a_by_b():
    torch.manual_seed(0)
    for A, B in ((4, 2), (5, 3), (7, 4)):
        dec = LocalPixelCNN(1, 12, 0, 16, window=(A, B))
        rf = receptive_field(dec, pixel=(6, 6))
        expect = torch.zeros(12, 12, dtype=torch.int)
        left = (A - 1) // 2
        expect[6 - B:6, 6 - left:6 - left + A] = 1                                # block above
        expect[6, 6 - (A - 1) // 2:6] = 1                                         # pixels to the left
        assert torch.equal(rf, expect), (A, B)
        assert rf[6, 6] == 0 and rf[7:].sum() == 0                               # never itself or the future


def test_stacked_pixelcnn_is_causal_and_local():
    torch.manual_seed(0)
    dec = LocalPixelCNN(1, 12, 0, 16, window=None, layers=6)
    rf = receptive_field(dec, pixel=(9, 9))
    assert rf[9, 9] == 0 and rf[10:].sum() == 0 and rf[9, 10:].sum() == 0        # only rows above / pixels left
    assert rf[:3].sum() == 0                                                      # 6 layers of 3x3 can't see 7 rows up
    assert rf.sum() > 10


def test_grayscale_context_ignores_colour_of_the_window():
    torch.manual_seed(0)
    dec = LocalPixelCNN(3, 8, 0, 16, window=(3, 2), gray_context=True, likelihood="logistic")
    x = torch.rand(1, 3, 8, 8)
    y = x.clone()
    y[:, :, 2, 3] = y[:, :, 2, 3].mean()                                          # same gray, different colour
    with torch.no_grad():
        assert torch.allclose(dec.features(x)[..., 3, 3], dec.features(y)[..., 3, 3], atol=1e-6)


def test_window_ar1d_sees_only_its_window():
    torch.manual_seed(0)
    for w in (0, 1, 3, 16):
        dec = WindowAR1d(16, w, 0)
        x = torch.rand(1, 16, requires_grad=True)
        dec.logits(x)[0, 8].backward()
        seen = [] if x.grad is None else (x.grad[0].abs() > 0).nonzero().flatten().tolist()
        assert seen == list(range(max(0, 8 - w), 8))


def test_af_prior_density_and_inverse():
    torch.manual_seed(0)
    prior = AFPrior(3, T=2, hidden=16)
    for p in prior.parameters():
        nn.init.normal_(p, 0, 0.4)
    z = prior.sample(4)
    eps, logdet = prior.inverse(z)
    J = torch.autograd.functional.jacobian(lambda v: prior.inverse(v)[0], z[0])
    assert torch.allclose(torch.linalg.slogdet(J)[1], logdet[0], atol=1e-4)       # log|det d eps/dz| (flip: sign -1)
    r = torch.distributions.MultivariateNormal(torch.zeros(3), 9 * torch.eye(3))
    zs = r.sample((100000,))
    with torch.no_grad():
        total = (prior.log_prob(zs) - r.log_prob(zs)).exp().mean().item()
    assert abs(total - 1) < 0.05                                                  # a normalised density


def test_af_prior_equals_iaf_posterior_bound():
    """Eqs. 12-14: the bound with an AF prior equals the bound in eps-space with an IAF posterior
    q(eps|x) = q(z|x) |dz/d eps| and a decoder applied to f(eps)."""
    torch.manual_seed(0)
    prior = AFPrior(2, T=1, hidden=8)
    for p in prior.parameters():
        nn.init.normal_(p, 0, 0.5)
    q = torch.distributions.Normal(torch.tensor([0.3, -0.2]), torch.tensor([0.5, 0.8]))
    z = q.sample((5,))
    log_px_given_z = -(z ** 2).sum(-1)                                            # any decoder
    L_af = log_px_given_z + prior.log_prob(z) - q.log_prob(z).sum(-1)
    eps, logdet = prior.inverse(z)                                                # logdet = log|d eps/dz|
    log_q_eps = q.log_prob(z).sum(-1) - logdet                                    # IAF posterior density of eps
    L_iaf = log_px_given_z + (-0.5 * (math.log(2 * math.pi) * 2 + (eps ** 2).sum(-1))) - log_q_eps
    assert torch.allclose(L_af, L_iaf, atol=1e-5)


def test_bits_back_code_length_is_minus_elbo_and_refund_is_entropy():
    torch.manual_seed(0)
    rec, lp, lq = -torch.rand(100) * 50, -torch.rand(100) * 5, -torch.rand(100) * 3
    c = code_lengths(rec, lp, lq)
    assert abs(c["bits-back"] - (-(rec + lp - lq).mean().item())) < 1e-4
    assert abs(c["naive"] - c["bits-back"] - c["refund = H(q)"]) < 1e-4


def test_vlae_shapes_bound_and_kl():
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ELU())
    dec = LocalPixelCNN(1, 8, 4, 16, window=(3, 2))
    m = VLAE(enc, 32, 4, dec, prior="af", af_hidden=16)
    x = torch.bernoulli(torch.full((3, 1, 8, 8), 0.5))
    with torch.no_grad():
        b, ll, kl = m.elbo(x, 500), m.log_likelihood(x, 500), m.kl(x)
    assert b.shape == (3,) and torch.all(ll >= b - 0.05) and torch.all(kl > -0.5)


def test_discretized_logistic_reused_from_040():
    assert iaf040.discretized_logistic_log_prob is not None
    m = window_mask(5, 3)
    assert m.shape == (4, 5) and m.sum() == 3 * 5 + 2
