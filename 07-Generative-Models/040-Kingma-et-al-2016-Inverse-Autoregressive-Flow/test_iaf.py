import math

import torch
import torch.nn as nn

from iaf import (MADE, FlowVAE, IAFPosterior, IAFStep, LinearIAF, PlanarFlow, discretized_logistic_log_prob,
                 free_bits_kl)


def randomize(module, std=0.5):
    for p in module.parameters():
        nn.init.normal_(p, 0, std)
    return module


def test_made_is_strictly_autoregressive():
    torch.manual_seed(0)
    for layers in (0, 1, 2):
        made = randomize(MADE(5, 16, layers, context=3))
        h = torch.randn(3)
        for k in range(2):                                                       # both outputs (m and s)
            J = torch.autograd.functional.jacobian(lambda z: made(z, h)[k], torch.randn(5))
            assert torch.all(J.triu() == 0)                                      # output i ignores z_j for j >= i
            assert J.tril(-1).abs().sum() > 0                                    # ...but does use z_<i


def test_iaf_step_jacobian_is_triangular_with_sigma_on_the_diagonal():
    torch.manual_seed(0)
    for mode in ("gated", "affine", "location"):
        step = randomize(IAFStep(4, 16, 1, mode=mode))
        z = torch.randn(4)
        J = torch.autograd.functional.jacobian(lambda v: step(v)[0], z)
        _, logdet = step(z)
        assert torch.all(J.triu(1).abs() < 1e-7)                                  # Eq. 8: triangular
        assert torch.allclose(torch.logdet(J), logdet, atol=1e-5)                # log det = sum log sigma
        if mode == "location":
            assert abs(logdet.item()) < 1e-7                                     # volume preserving


def test_iaf_step_inverse_is_sequential_and_exact():
    torch.manual_seed(0)
    step = randomize(IAFStep(6, 32, 1, context=2), 0.3)
    z, h = torch.randn(3, 6), torch.randn(3, 2)
    with torch.no_grad():
        y, _ = step(z, h)
        assert torch.allclose(step.inverse(y, h), z, atol=1e-5)


def test_algorithm_1_density_integrates_to_one():
    """log q from Algorithm 1 must be a normalised density: E_{r}[q(z) / r(z)] = 1 for a broad proposal r."""
    torch.manual_seed(0)
    post = randomize(IAFPosterior(2, T=2, hidden=16), 0.4)
    mu, logvar = torch.zeros(2), torch.zeros(2)
    with torch.no_grad():
        z, _ = post(mu, logvar, eps=torch.randn(1, 2))
        r = torch.distributions.MultivariateNormal(torch.zeros(2), 9 * torch.eye(2))
        zs = r.sample((200000,))
        # density of an arbitrary point: invert the flow step by step (Eq. 5 read backwards)
        y = zs.clone()
        if len(post.steps) % 2 == 0:
            y = y.flip(-1)
        log_det = torch.zeros(len(y))
        for t in reversed(range(len(post.steps))):
            prev = post.steps[t].inverse(y)
            log_det = log_det + post.steps[t](prev)[1]
            y = prev.flip(-1) if t > 0 else prev
        log_q = -0.5 * (math.log(2 * math.pi) * 2 + (y ** 2).sum(-1)) - log_det
        total = (log_q - r.log_prob(zs)).exp().mean().item()
    assert abs(total - 1) < 0.03


def test_algorithm_1_log_q_matches_inversion():
    torch.manual_seed(1)
    post = randomize(IAFPosterior(3, T=1, hidden=16), 0.4)
    mu, logvar, eps = torch.randn(3), torch.randn(3) * 0.3, torch.randn(5, 3)
    with torch.no_grad():
        z, log_q = post(mu, logvar, eps=eps)
        z0 = post.steps[0].inverse(z)
        base = torch.distributions.Normal(mu, (0.5 * logvar).exp()).log_prob(z0).sum(-1)
        assert torch.allclose(log_q, base - post.steps[0](z0)[1], atol=1e-4)


def test_linear_iaf_gives_full_covariance():
    torch.manual_seed(0)
    lin = LinearIAF(3, context=1)
    nn.init.normal_(lin.L.bias, 0, 1.0)
    h = torch.ones(1)
    Lm = lin.matrix(h)
    sig2 = torch.tensor([1.0, 0.5, 2.0])
    with torch.no_grad():
        z, log_q = lin(torch.zeros(200000, 3), sig2.log().expand(200000, 3), h.expand(200000, 1))
    target = Lm @ torch.diag(sig2) @ Lm.T
    assert torch.allclose(torch.cov(z.T), target, atol=0.05)
    exact = torch.distributions.MultivariateNormal(torch.zeros(3), target).log_prob(z[:100])
    assert torch.allclose(log_q[:100], exact, atol=1e-4)                          # det L = 1


def _fit(make_q, steps=400):
    """Fit an unconditional q to N(0, [[1, .9], [.9, 1]]) by maximising E_q[log p - log q] = -KL(q || p)."""
    torch.manual_seed(0)
    target = torch.distributions.MultivariateNormal(torch.zeros(2), torch.tensor([[1.0, 0.9], [0.9, 1.0]]))
    mu, logvar = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    flow, sample = make_q(mu, logvar)
    opt = torch.optim.Adam([mu, logvar] + list(flow.parameters()), 2e-2)
    for _ in range(steps):
        z, log_q = sample(256)
        loss = (log_q - target.log_prob(z)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z, log_q = sample(20000)
        return (log_q - target.log_prob(z)).mean().item()                          # = KL(q || p)


def test_iaf_closes_the_gap_a_diagonal_gaussian_cannot():
    def diag(mu, logvar):
        def s(n):
            eps = torch.randn(n, 2)
            return mu + (0.5 * logvar).exp() * eps, torch.distributions.Normal(mu, (0.5 * logvar).exp()).log_prob(
                mu + (0.5 * logvar).exp() * eps).sum(-1)
        return nn.Module(), s

    def linear(mu, logvar):
        lin = LinearIAF(2, 1)
        return lin, lambda n: lin(mu.expand(n, 2), logvar.expand(n, 2), torch.ones(n, 1))

    kl_diag, kl_lin = _fit(diag), _fit(linear)
    assert abs(kl_diag - (-0.5 * math.log(1 - 0.81))) < 0.03                      # mean-field optimum: 0.830 nats
    assert kl_lin < 0.02


def test_free_bits_and_discretized_logistic():
    kl = torch.tensor([[0.1, 2.0], [0.3, 1.0]], requires_grad=True)               # batch 2, groups 2
    out = free_bits_kl(kl, lam=0.5)
    out.backward()
    assert abs(out.item() - (0.5 + 1.5)) < 1e-6
    assert torch.all(kl.grad[:, 0] == 0) and torch.all(kl.grad[:, 1] == 0.5)      # group 1 below lambda: no push
    x = torch.arange(256) / 256
    lp = discretized_logistic_log_prob(x, torch.tensor(0.4), torch.tensor(-3.0))
    assert abs(lp.exp().sum().item() - 1) < 1e-5 and lp.argmax().item() in (102, 101)


def test_flow_vae_bound_and_planar_flow():
    torch.manual_seed(0)
    x = torch.bernoulli(torch.full((4, 20), 0.5))
    for post in ("diag", "linear", "iaf", "planar"):
        m = FlowVAE(20, 4, 32, posterior=post, T=2, iaf_hidden=16, ctx=8)
        with torch.no_grad():
            b, ll = m.elbo(x, 2000), m.log_likelihood(x, 2000)
        assert b.shape == (4,) and torch.all(ll >= b - 0.05)
    pf = PlanarFlow(3, K=4)
    nn.init.normal_(pf.u, 0, 1); nn.init.normal_(pf.w, 0, 1)
    z = torch.randn(3)
    J = torch.autograd.functional.jacobian(lambda v: pf(v)[0], z)
    assert torch.allclose(torch.logdet(J), pf(z)[1], atol=1e-4)
