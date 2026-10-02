"""A few-second tour of Inverse Autoregressive Flow.

  1. MADE masks: which inputs each output may see (the autoregressive property)
  2. A diagonal Gaussian can't capture correlation: the exact gap -1/2 log(1 - rho^2), and linear IAF closing it
  3. A curved ('banana') posterior: diagonal vs linear IAF vs IAF with 1, 2, 4 steps vs a planar flow
  4. Figure 1's toy: a VAE with a 2-D latent on four datapoints, diagonal vs IAF posteriors
  5. Why 'inverse': sampling is one parallel pass, evaluating an arbitrary point needs D sequential passes
"""

import math
import time

import torch
import torch.nn as nn

from iaf import MADE, FlowVAE, IAFPosterior, IAFStep, LinearIAF, PlanarFlow

torch.manual_seed(0)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. MADE masks (D = 4, one hidden layer of 6 units): which z_j can output i see?")
made = MADE(4, 6, 1)
conn = (made.out.mask[:4] @ made.hidden[0].mask).clamp(max=1).int()            # paths input -> output (m part)
for i in range(4):
    print(f"  output {i + 1}: sees z_j for j = {[j + 1 for j in range(4) if conn[i, j]] or 'none (constant)'}")
print("  -> strictly lower-triangular: m_i and s_i depend only on z_1..z_(i-1), so dz'_i/dz_i = sigma_i and the"
      " log-determinant is just sum log sigma (Eq. 8).")


# --------------------------------------------------------------------------------------------- helpers
def fit(target_log_prob, kind, steps=500, D=2, seed=0):
    """Fit an unconditional q by maximising E_q[log p(z) - log q(z)]; returns KL(q || p) (target is normalised)."""
    torch.manual_seed(seed)
    mu, logvar = nn.Parameter(torch.zeros(D)), nn.Parameter(torch.zeros(D))
    if kind == "diagonal":
        flow = nn.Module()

        def sample(n):
            eps = torch.randn(n, D)
            return mu + (0.5 * logvar).exp() * eps, -(0.5 * eps ** 2 + 0.5 * math.log(2 * math.pi) + 0.5 * logvar).sum(-1)
    elif kind == "linear IAF":
        flow = LinearIAF(D, 1)
        sample = lambda n: flow(mu.expand(n, D), logvar.expand(n, D), torch.ones(n, 1))
    elif kind.startswith("IAF"):
        flow = IAFPosterior(D, T=int(kind.split("T=")[1].rstrip(")")), hidden=32)
        sample = lambda n: flow(mu.expand(n, D), logvar.expand(n, D))
    else:                                                                        # planar, K = 8
        flow = PlanarFlow(D, K=8)

        def sample(n):
            eps = torch.randn(n, D)
            z, ld = flow(mu + (0.5 * logvar).exp() * eps)
            return z, -(0.5 * eps ** 2 + 0.5 * math.log(2 * math.pi) + 0.5 * logvar).sum(-1) - ld
    opt = torch.optim.Adam([mu, logvar] + list(flow.parameters()), 1e-2)
    for _ in range(steps):
        z, lq = sample(256)
        loss = (lq - target_log_prob(z)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z, lq = sample(50000)
        return (lq - target_log_prob(z)).mean().item(), sum(p.numel() for p in flow.parameters()) + 2 * D


# --------------------------------------------------------------------------------------------- 2
section("2. Target N(0, [[1, rho], [rho, 1]]): best KL(q || p) for a diagonal q vs linear IAF (Appendix A)")
print(f"  {'rho':>5} {'theory (diag)':>14} {'diagonal q':>11} {'linear IAF':>11}")
for rho in (0.5, 0.9, 0.99):
    tgt = torch.distributions.MultivariateNormal(torch.zeros(2), torch.tensor([[1, rho], [rho, 1.]]))
    n = 2000 if rho > 0.95 else 400                                              # the thin ridge needs longer
    print(f"  {rho:5.2f} {-0.5 * math.log(1 - rho * rho):14.3f} {fit(tgt.log_prob, 'diagonal', n)[0]:11.3f} "
          f"{fit(tgt.log_prob, 'linear IAF', n)[0]:11.3f}   ({n} steps)")
print("  -> the diagonal Gaussian is stuck at exactly -1/2 log(1 - rho^2) nats however long it trains; one linear IAF"
      " step (z = L y) reaches ~0.")

# --------------------------------------------------------------------------------------------- 3
section("3. A curved posterior: z1 ~ N(0, 1), z2 | z1 ~ N(z1^2 - 1, 0.3^2). KL(q || p) after 500 steps")
banana = lambda z: (torch.distributions.Normal(0., 1.).log_prob(z[..., 0]) +
                    torch.distributions.Normal(z[..., 0] ** 2 - 1, 0.3).log_prob(z[..., 1]))
print(f"  {'posterior':>16} {'KL (nats)':>10} {'params':>7}")
for kind in ("diagonal", "linear IAF", "planar (K=8)", "IAF (T=1)", "IAF (T=2)", "IAF (T=4)"):
    kl, n = fit(banana, kind, 500)
    print(f"  {kind:>16} {kl:10.3f} {n:7d}")
print("  -> no Gaussian (diagonal or full) can bend; IAF steps can, because each m_i, sigma_i is a nonlinear function"
      " of the earlier coordinates. More steps fit better.")

# --------------------------------------------------------------------------------------------- 4
section("4. Figure 1's toy: VAE, 2-D latent, four datapoints (one-hot over 4 blocks of 5 pixels)")
X = torch.zeros(4, 20)
for i in range(4):
    X[i, 5 * i:5 * i + 5] = 1
res = {}
for post in ("diag", "iaf"):
    torch.manual_seed(0)
    m = FlowVAE(20, 2, 64, posterior=post, T=4, iaf_hidden=32, ctx=16)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(1500):
        loss = -m.elbo(X, n=8).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        b, ll = m.elbo(X, n=5000).mean().item(), m.log_likelihood(X, n=5000).mean().item()
    res[post] = (b, ll)
    print(f"  {post:>5} posterior: bound {b:7.3f}, importance-sampled log p(x) {ll:7.3f}, gap {ll - b:5.3f} nats"
          f"   (best possible log p(x) = log 1/4 = {-math.log(4):.3f})")
print("  -> as in Figure 1 and Table 1: the more flexible posterior gives a tighter bound (smaller gap to log p(x)).")

# --------------------------------------------------------------------------------------------- 5
section("5. Sampling vs inverting one IAF step, D = 128 (batch of 64)")
step = IAFStep(128, 256, 1)
for p in step.parameters():
    nn.init.normal_(p, 0, 0.02)
z = torch.randn(64, 128)
with torch.no_grad():
    t = time.time(); y, _ = step(z); t_fwd = time.time() - t
    t = time.time(); z_back = step.inverse(y); t_inv = time.time() - t
print(f"  forward (sample, 1 pass): {1000 * t_fwd:6.2f} ms   inverse (density of a given z, 128 passes): "
      f"{1000 * t_inv:7.2f} ms   max error {(z_back - z).abs().max():.1e}")
print("  -> a VAE only needs the forward direction (sample z, and Algorithm 1 gives log q as a by-product), so IAF"
      " is cheap exactly where variational inference needs it.")

print(f"\n(total {time.time() - T0:.1f} s)")
