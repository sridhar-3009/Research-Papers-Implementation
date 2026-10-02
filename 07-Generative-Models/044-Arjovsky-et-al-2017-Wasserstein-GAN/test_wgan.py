import math

import torch
import torch.nn as nn

from wgan import (Critic, clip_weights, critic_objective, example1_distances, generator_loss, js, kl,
                  lipschitz_bound, max_grad_norm, tv, wasserstein_1d_hist, wasserstein_1d_samples,
                  wasserstein_dual_lp, wasserstein_lp, wgan_step)


def test_example_1_parallel_lines():
    for theta in (1.0, 0.1, 1e-6):
        d = example1_distances(theta)
        assert d["W"] == theta and d["JS"] == math.log(2) and d["KL"] == float("inf") and d["TV"] == 1.0
    assert example1_distances(0.0) == {"W": 0.0, "JS": 0.0, "KL": 0.0, "TV": 0.0}
    z = torch.rand(1000)                                                          # sample check of W = |theta|
    a, b = torch.stack([torch.zeros(1000), z], 1), torch.stack([torch.full((1000,), 0.3), z], 1)
    assert abs((a - b).norm(dim=1).mean().item() - 0.3) < 1e-6                     # the coupling (0,z) <-> (theta,z)


def test_divergences_on_disjoint_supports():
    p, q = torch.tensor([0.5, 0.5, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.5, 0.5])
    assert torch.allclose(js(p, q), torch.tensor(math.log(2)))
    assert tv(p, q) == 1.0 and kl(p, q) == float("inf")
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.allclose(wasserstein_1d_hist(p, q, x), torch.tensor(2.0))      # move every unit of mass by 2


def test_primal_equals_dual_kantorovich_rubinstein():
    torch.manual_seed(0)
    x = torch.linspace(0, 4, 9)
    p, q = torch.rand(9), torch.rand(9)
    p, q = (p / p.sum()).double(), (q / q.sum()).double()
    w_primal, plan = wasserstein_lp(p, q, x.double())
    w_dual, f = wasserstein_dual_lp(p, q, x.double())
    assert abs(w_primal - w_dual) < 1e-6                                           # Eq. 1 = Eq. 2
    assert abs(w_primal - wasserstein_1d_hist(p, q, x.double()).item()) < 1e-6
    assert torch.allclose(plan.sum(1), p, atol=1e-8) and torch.allclose(plan.sum(0), q, atol=1e-8)
    assert torch.all((f[1:] - f[:-1]).abs() <= (x[1] - x[0]) + 1e-6)              # the optimal f is 1-Lipschitz


def test_samples_w1_matches_shifted_gaussians():
    torch.manual_seed(0)
    a = torch.randn(20000)
    assert abs(wasserstein_1d_samples(a, a + 1.5).item() - 1.5) < 1e-5
    assert abs(wasserstein_1d_samples(a, torch.randn(20000) + 1.5).item() - 1.5) < 0.05


def test_theorem_3_gradient_with_a_linear_critic():
    """P_r = N(0, 1), P_theta = N(theta, 1): W = |theta|, optimal critic f(x) = -x (theta > 0), and
    grad W = -E[grad_theta f(g_theta(z))] = 1."""
    theta = torch.tensor(0.7, requires_grad=True)
    z = torch.randn(1000)
    f = lambda x: -x
    g, = torch.autograd.grad(generator_loss(f, theta + z), theta)
    assert abs(g.item() - 1.0) < 1e-6
    est = critic_objective(f, torch.randn(200000), 0.7 + torch.randn(200000))
    assert abs(est.item() - 0.7) < 0.02


def test_weight_clipping_bounds_the_lipschitz_constant():
    torch.manual_seed(0)
    f = Critic(2, (16, 16))
    for p in f.parameters():
        nn.init.normal_(p, 0, 1.0)
    before = lipschitz_bound(f)
    clip_weights(f, 0.01)
    assert all(p.abs().max() <= 0.01 for p in f.parameters())
    after = lipschitz_bound(f)
    assert after < before and after <= (0.01 * 16) * (0.01 * 16) * (0.01 * 4) + 1e-6   # ||W||_2 <= c sqrt(mn)
    assert max_grad_norm(f, torch.randn(100, 2)) <= after + 1e-6


def test_wgan_learns_example_1():
    class G1(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.tensor(1.0))

        def forward(self, z):
            return torch.stack([self.theta.expand_as(z), z], 1)

    torch.manual_seed(0)
    G, f = G1(), Critic(2, (32, 32))
    og, of = torch.optim.RMSprop(G.parameters(), 5e-3), torch.optim.RMSprop(f.parameters(), 5e-4)
    for _ in range(200):
        wgan_step(G, f, og, of, lambda m: torch.stack([torch.zeros(m), torch.rand(m)], 1), lambda m: torch.rand(m),
                  m=64, n_critic=5, clip=0.01)
    assert abs(G.theta.item()) < 0.1
