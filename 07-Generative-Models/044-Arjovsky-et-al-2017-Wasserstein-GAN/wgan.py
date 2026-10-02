"""Wasserstein GAN (Arjovsky, Chintala & Bottou, ICML 2017).

  Earth-Mover / Wasserstein-1:   W(P_r, P_g) = inf_{gamma in Pi(P_r, P_g)} E_{(x,y)~gamma} ||x - y||             (Eq. 1)
  Kantorovich-Rubinstein dual:   W(P_r, P_g) = sup_{||f||_L <= 1} E_{P_r}[f(x)] - E_{P_g}[f(x)]                  (Eq. 2)
  WGAN:  max_{w in W} E_{P_r}[f_w(x)] - E_z[f_w(g_theta(z))]   (a K-Lipschitz 'critic', W compact)            (Eq. 3)
  Theorem 3:  grad_theta W(P_r, P_theta) = -E_z[grad_theta f(g_theta(z))] for the optimal critic f
  Algorithm 1: n_critic = 5 critic steps (RMSProp, lr 5e-5) with weights clipped to [-c, c] (c = 0.01), then one
  generator step; batch m = 64. Example 1 ('parallel lines') shows why: W = |theta| is continuous, while JS, KL and
  TV jump.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------- distances
def tv(p, q):
    """Total variation for discrete distributions: sup_A |P(A) - Q(A)| = 1/2 sum |p - q|."""
    return 0.5 * (p - q).abs().sum()


def kl(p, q):
    m = p > 0
    if torch.any(q[m] == 0):
        return torch.tensor(float("inf"))
    return (p[m] * (p[m] / q[m]).log()).sum()


def js(p, q):
    """Jensen-Shannon with the usual 1/2 weights (max log 2). The paper writes it without the 1/2s but quotes
    log 2 for disjoint supports, which matches this convention."""
    m = (p + q) / 2
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein_1d_samples(a, b):
    """Exact W1 between two equal-size empirical 1-D distributions: match sorted samples."""
    return (a.sort().values - b.sort().values).abs().mean()


def wasserstein_1d_hist(p, q, x):
    """W1 between histograms p, q on sorted grid x: integral of |F_p - F_q|."""
    dx = torch.diff(x)
    return ((p.cumsum(0) - q.cumsum(0)).abs()[:-1] * dx).sum()


def wasserstein_lp(p, q, x):
    """Eq. 1 directly: a linear program over transport plans gamma (n x n) with marginals p and q.
    Returns (W, optimal plan). Small problems only (uses scipy)."""
    from scipy.optimize import linprog
    n = len(p)
    C = torch.cdist(x.view(n, -1), x.view(n, -1)).numpy().ravel()
    A = []
    for i in range(n):                                                           # row sums = p
        r = torch.zeros(n, n); r[i] = 1; A.append(r.ravel())
    for j in range(n):                                                           # column sums = q
        c = torch.zeros(n, n); c[:, j] = 1; A.append(c.ravel())
    res = linprog(C, A_eq=torch.stack(A).numpy(), b_eq=torch.cat([p, q]).numpy(), bounds=(0, None), method="highs")
    return float(res.fun), torch.tensor(res.x).view(n, n)


def wasserstein_dual_lp(p, q, x):
    """Eq. 2 directly: maximise sum f (p - q) over f with |f_i - f_j| <= |x_i - x_j| (1-Lipschitz on the grid)."""
    from scipy.optimize import linprog
    n = len(p)
    D = torch.cdist(x.view(n, -1), x.view(n, -1))
    rows, b = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                r = torch.zeros(n); r[i], r[j] = 1, -1
                rows.append(r); b.append(D[i, j])
    res = linprog(-(p - q).numpy(), A_ub=torch.stack(rows).numpy(), b_ub=torch.tensor(b).numpy(),
                  bounds=(None, None), method="highs")
    return float(-res.fun), torch.tensor(res.x)


def example1_distances(theta):
    """Example 1: P_0 = (0, Z), P_theta = (theta, Z), Z ~ U[0, 1]. Closed forms."""
    t = abs(theta)
    return {"W": t, "JS": 0.0 if t == 0 else math.log(2), "KL": 0.0 if t == 0 else float("inf"),
            "TV": 0.0 if t == 0 else 1.0}


# ---------------------------------------------------------------------------------------------------- networks
class Critic(nn.Module):
    """f_w: an MLP with NO sigmoid (a real-valued score, not a probability)."""

    def __init__(self, x_dim=2, hidden=(64, 64)):
        super().__init__()
        layers, d = [], x_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def clip_weights(module, c):
    """Algorithm 1, line 7: keep w in the compact box [-c, c]^l, which makes f_w K-Lipschitz for some K."""
    with torch.no_grad():
        for p in module.parameters():
            p.clamp_(-c, c)


def lipschitz_bound(module):
    """An upper bound on the Lipschitz constant of a ReLU MLP: the product of the layers' spectral norms."""
    K = 1.0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            K *= torch.linalg.matrix_norm(m.weight, ord=2).item()
    return K


def max_grad_norm(f, x):
    """An empirical lower estimate of the Lipschitz constant: max ||grad_x f(x)|| over the given points."""
    x = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(f(x).sum(), x)
    return g.norm(dim=-1).max().item()


# ---------------------------------------------------------------------------------------------------- losses
def critic_objective(f, real, fake):
    """E_r[f] - E_g[f]: the critic MAXIMISES this; with the Lipschitz constraint it estimates K * W."""
    return f(real).mean() - f(fake).mean()


def generator_loss(f, fake):
    """Algorithm 1, line 10: minimise -E_z[f(g(z))] (its gradient is Theorem 3's gradient of W)."""
    return -f(fake).mean()


def wgan_step(G, f, opt_g, opt_f, sample_real, sample_z, m=64, n_critic=5, clip=0.01):
    """One generator iteration of Algorithm 1. Returns the last critic estimate E_r[f] - E_g[f]."""
    for _ in range(n_critic):
        with torch.no_grad():
            fake = G(sample_z(m))
        obj = critic_objective(f, sample_real(m), fake)
        opt_f.zero_grad(); (-obj).backward(); opt_f.step()
        clip_weights(f, clip)
    loss = generator_loss(f, G(sample_z(m)))
    opt_g.zero_grad(); loss.backward(); opt_g.step()
    return obj.item()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
