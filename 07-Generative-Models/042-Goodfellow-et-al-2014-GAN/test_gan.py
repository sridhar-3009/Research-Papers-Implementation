import math

import torch

from gan import (Discriminator, Generator, Maxout, d_loss, g_loss, interpolate, jsd, nearest_neighbours,
                 optimal_discriminator, parzen_log_likelihood, select_sigma, train_step, value, virtual_criterion)


def test_proposition_1_optimal_discriminator():
    """a log y + b log(1 - y) is maximised at y = a / (a + b), so D* = p_data / (p_data + p_g)."""
    y = torch.linspace(1e-4, 1 - 1e-4, 100001)
    for a, b in ((0.3, 0.7), (2.0, 0.5), (1.0, 1.0)):
        best = y[(a * y.log() + b * (1 - y).log()).argmax()]
        assert abs(best - a / (a + b)) < 1e-3
    p, q = torch.tensor([0.2, 0.5, 0.3]), torch.tensor([0.6, 0.2, 0.2])
    Dstar = optimal_discriminator(p, q)
    for _ in range(100):                                                         # any other D is worse
        D = torch.rand(3).clamp(0.01, 0.99)
        assert value(p, q, D) <= value(p, q, Dstar) + 1e-6


def test_theorem_1_criterion_is_minus_log4_plus_2jsd():
    torch.manual_seed(0)
    for _ in range(20):
        p, q = torch.rand(6), torch.rand(6)
        p, q = p / p.sum(), q / q.sum()
        assert torch.allclose(virtual_criterion(p, q), -math.log(4) + 2 * jsd(p, q), atol=1e-5)
        assert virtual_criterion(p, q) >= -math.log(4) - 1e-6
    assert torch.allclose(virtual_criterion(p, p), torch.tensor(-math.log(4)), atol=1e-6)
    assert torch.allclose(optimal_discriminator(p, p), torch.full_like(p, 0.5))
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])                   # disjoint supports
    assert torch.allclose(jsd(a, b), torch.tensor(math.log(2)), atol=1e-6)


def test_saturating_vs_non_saturating_gradient():
    """When D confidently rejects a fake (D(G(z)) = 0.01), the minimax loss gives a 99x smaller gradient."""
    l = torch.tensor(math.log(0.01 / 0.99), requires_grad=True)                   # logit with sigmoid = 0.01
    D = lambda x: x
    g_sat, = torch.autograd.grad(g_loss(D, l.view(1), saturating=True), l)
    g_ns, = torch.autograd.grad(g_loss(D, l.view(1), saturating=False), l)
    assert abs(g_sat.item() - (-0.01)) < 1e-6 and abs(g_ns.item() - (-0.99)) < 1e-6


def test_losses_match_definitions():
    torch.manual_seed(0)
    D = Discriminator(3, (8,), pieces=2, dropout=0.0, input_dropout=0.0)
    real, fake = torch.randn(5, 3), torch.randn(5, 3)
    p_r, p_f = torch.sigmoid(D(real)), torch.sigmoid(D(fake))
    assert torch.allclose(d_loss(D, real, fake), -(p_r.log().mean() + (1 - p_f).log().mean()), atol=1e-5)
    assert torch.allclose(g_loss(D, fake, True), (1 - p_f).log().mean(), atol=1e-5)
    assert torch.allclose(g_loss(D, fake, False), -p_f.log().mean(), atol=1e-5)


def test_maxout_and_shapes():
    m = Maxout(4, 3, pieces=2)
    x = torch.randn(7, 4)
    pieces = m.lin(x).view(7, 3, 2)
    assert torch.equal(m(x), pieces.max(-1).values)
    G, D = Generator(100, (1200, 1200), 784), Discriminator(784, (240, 240))
    z = G.sample_z(4)
    assert z.min() >= -1 and z.max() <= 1
    out = G(z)
    assert out.shape == (4, 784) and out.min() >= 0 and out.max() <= 1 and D(out).shape == (4,)
    assert interpolate(G, z[0], z[1], 5).shape == (5, 784)


def test_parzen_window_estimator():
    s = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    x = torch.tensor([[1.0, 0.0]])
    sigma = 0.7
    manual = math.log(0.5 * 2 * math.exp(-1 / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2))
    assert abs(parzen_log_likelihood(s, x, sigma).item() - manual) < 1e-5
    torch.manual_seed(0)
    samples, valid = torch.randn(2000, 2), torch.randn(500, 2)
    best, _ = select_sigma(samples, valid, [0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    assert best in (0.2, 0.3)                                                    # neither too sharp nor too wide
    exact = -0.5 * (2 * math.log(2 * math.pi) + (valid ** 2).sum(1)).mean()
    assert abs(parzen_log_likelihood(samples, valid, best).mean() - exact) < 0.15
    assert torch.equal(nearest_neighbours(s[:1] + 0.1, s), s[:1])


def test_algorithm_1_learns_a_1d_gaussian():
    torch.manual_seed(0)
    G = Generator(1, (32, 32), 1, out="linear")
    D = Discriminator(1, (64, 64), pieces=3, dropout=0.0, input_dropout=0.0)
    og = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
    for _ in range(1000):
        train_step(G, D, og, od, lambda m: 3 + 0.5 * torch.randn(m, 1), m=256)
    with torch.no_grad():
        x = G(G.sample_z(5000))
        d = torch.sigmoid(D(torch.linspace(2.5, 3.5, 5)[:, None]))
    assert abs(x.mean() - 3) < 0.3 and 0.3 < x.std() < 0.7
    assert torch.all((d > 0.35) & (d < 0.65))                                   # D near 1/2 on the data
