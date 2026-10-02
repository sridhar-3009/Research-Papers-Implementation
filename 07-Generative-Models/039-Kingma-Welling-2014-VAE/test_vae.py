import math

import torch

from vae import (VAE, hmc, kl_to_standard_normal, log_normal_diag, marginal_likelihood_appendix_d,
                 reparam_grad, reparameterize, score_function_grad, wake_sleep_losses)


class LinearGaussian:
    """p(z) = N(0, 1), p(x|z) = N(w z, s^2): everything is exact. p(x) = N(0, w^2 + s^2);
    posterior p(z|x) = N(v w x / s^2, v) with v = 1 / (1 + w^2 / s^2)."""
    kind, z_dim = "gaussian", 1

    def __init__(self, w=2.0, s=0.5):
        self.w, self.s = w, s
        self.v = 1 / (1 + w ** 2 / s ** 2)

    def log_px(self, x):
        return log_normal_diag(x, torch.zeros_like(x), torch.full_like(x, math.log(self.w ** 2 + self.s ** 2)))

    def enc(self, x):                                                              # the exact posterior
        return self.v * self.w * x / self.s ** 2, torch.full_like(x, math.log(self.v))

    def log_px_given_z(self, x, z):
        return log_normal_diag(x, self.w * z, torch.full_like(z, 2 * math.log(self.s)))

    def elbo_exact(self, x, m, logv):
        """Eq. 3 in closed form for q = N(m, e^logv)."""
        v = math.exp(logv)
        rec = -0.5 * math.log(2 * math.pi * self.s ** 2) - ((x - self.w * m) ** 2 + self.w ** 2 * v) / (2 * self.s ** 2)
        return rec - kl_to_standard_normal(torch.tensor([m]), torch.tensor([logv])).item()


def test_eq1_log_likelihood_is_bound_plus_kl_to_posterior():
    lg, x = LinearGaussian(), 1.3
    logpx = lg.log_px(torch.tensor([x])).item()
    pm, plv = (t.item() for t in lg.enc(torch.tensor([x])))
    assert abs(lg.elbo_exact(x, pm, plv) - logpx) < 1e-6                          # q = posterior: bound is tight
    m, lv = 0.2, math.log(0.3)
    post = torch.distributions.Normal(pm, math.exp(0.5 * plv))
    q = torch.distributions.Normal(m, math.exp(0.5 * lv))
    gap = torch.distributions.kl_divergence(q, post).item()
    assert abs(lg.elbo_exact(x, m, lv) + gap - logpx) < 1e-6                       # Eq. 1
    assert lg.elbo_exact(x, m, lv) < logpx


def test_closed_form_kl_appendix_b():
    mu, logvar = torch.randn(5, 3), torch.randn(5, 3)
    q = torch.distributions.Normal(mu, (0.5 * logvar).exp())
    ref = torch.distributions.kl_divergence(q, torch.distributions.Normal(0., 1.)).sum(-1)
    assert torch.allclose(kl_to_standard_normal(mu, logvar), ref, atol=1e-5)
    assert torch.allclose(kl_to_standard_normal(torch.zeros(3), torch.zeros(3)), torch.tensor(0.))


def test_reparameterization_gives_the_right_distribution_and_gradients():
    torch.manual_seed(0)
    mu, logvar = torch.tensor([1.5], requires_grad=True), torch.tensor([math.log(4.0)], requires_grad=True)
    z = reparameterize(mu.expand(200000), logvar.expand(200000))
    assert abs(z.mean().item() - 1.5) < 0.02 and abs(z.var().item() - 4.0) < 0.05
    (z ** 2).mean().backward()                                                     # E[z^2] = mu^2 + sigma^2
    assert abs(mu.grad.item() - 3.0) < 0.05                                        # d/dmu = 2 mu
    assert abs(logvar.grad.item() - 4.0) < 0.1                                     # d/dlogvar = sigma^2


def test_reparameterized_gradient_has_far_lower_variance_than_score_function():
    torch.manual_seed(0)
    f = lambda z: z ** 2
    sf = score_function_grad(f, 1.0, 0.0, 20000)
    rp = reparam_grad(f, 1.0, 0.0, 20000)
    assert abs(sf[:, 0].mean().item() - 2.0) < 0.15 and abs(rp[:, 0].mean().item() - 2.0) < 0.05   # both unbiased
    assert sf[:, 0].var() > 5 * rp[:, 0].var()


def test_estimators_a_and_b_agree_and_bound_is_below_likelihood():
    torch.manual_seed(0)
    m = VAE(12, 2, 16, init_std=0.3)
    x = torch.bernoulli(torch.full((4, 12), 0.5))
    with torch.no_grad():
        a, b = m.elbo(x, L=20000, estimator="A"), m.elbo(x, L=20000, estimator="B")
        ll = m.importance_log_likelihood(x, K=20000)
    assert torch.allclose(a, b, atol=0.05)
    assert (ll >= b - 0.02).all()


def test_sizes_and_shapes():
    m = VAE(784, 20, 500)
    enc = 784 * 500 + 500 + 2 * (500 * 20 + 20)
    dec = 20 * 500 + 500 + 500 * 784 + 784
    assert sum(p.numel() for p in m.parameters()) == enc + dec
    g = VAE(560, 2, 200, decoder="gaussian")                                       # Frey Face: 28 x 20
    assert g.sample(3).shape == (3, 560) and ((g.sample(3) > 0) & (g.sample(3) < 1)).all()
    assert VAE(784, 2, 50).manifold(5).shape == (25, 784)


def test_wake_sleep_losses_touch_the_right_parameters():
    torch.manual_seed(0)
    m = VAE(12, 2, 16, init_std=0.3)
    wake, sleep = wake_sleep_losses(m, torch.bernoulli(torch.full((8, 12), 0.5)))
    wake.backward()
    assert all(p.grad is None for p in m.enc.parameters()) and all(p.grad is not None for p in m.dec.parameters())
    m.zero_grad(set_to_none=True)
    sleep.backward()
    assert all(p.grad is None for p in m.dec.parameters()) and all(p.grad is not None for p in m.enc.parameters())


def test_hmc_and_appendix_d_estimator_on_an_exact_model():
    torch.manual_seed(0)
    lg = LinearGaussian()
    x = torch.tensor([[1.3], [-0.4]])
    lj = lambda z: log_normal_diag(z, torch.zeros_like(z), torch.zeros_like(z)) + lg.log_px_given_z(x, z)
    s, acc = hmc(lj, torch.zeros(2, 1), 600, leapfrog=4, step=0.2)
    pm = lg.enc(x)[0]
    assert acc > 0.6
    assert torch.allclose(s[100:].mean(0), pm, atol=0.05) and abs(s[100:].var(0).mean().item() - lg.v) < 0.03
    est, _ = marginal_likelihood_appendix_d(lg, x, L=300, burn=50, leapfrog=4, step=0.2)
    assert torch.allclose(est, lg.log_px(x), atol=0.05)


def test_aevb_learns():
    torch.manual_seed(0)
    protos = torch.bernoulli(torch.full((4, 20), 0.5))
    m = VAE(20, 2, 32, init_std=0.1)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    x = protos[torch.randint(0, 4, (256,))]
    start = m.elbo(x).mean().item()
    for _ in range(300):
        loss = -m.map_objective(x, N=256)
        opt.zero_grad(); loss.backward(); opt.step()
    end = m.elbo(x, L=10).mean().item()
    assert start < -13 and end > -3                                               # ln 4 = 1.39 is the best possible
