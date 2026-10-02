"""Improved Variational Inference with Inverse Autoregressive Flow (Kingma, Salimans, Jozefowicz, Chen,
Sutskever & Welling, NIPS 2016).

  normalizing flow:   z_0 ~ q(z_0|x),  z_t = f_t(z_{t-1}),  log q(z_T|x) = log q(z_0|x) - sum_t log|det dz_t/dz_{t-1}|  (Eq. 5)
  IAF step:           [m_t, s_t] = AutoregressiveNN_t(z_{t-1}, h)       (outputs i depend only on z_{<i})   (Eq. 12)
                      sigma_t = sigmoid(s_t),  z_t = sigma_t * z_{t-1} + (1 - sigma_t) * m_t               (Eqs. 13-14)
  log-density:        log q(z_T|x) = -sum_i (eps_i^2 / 2 + log(2 pi) / 2 + sum_{t=0..T} log sigma_{t,i})   (Eq. 11)
  - the Jacobian of each step is triangular with sigma_t on the diagonal, so log det = sum log sigma_t (Eq. 8);
  - sampling is ONE parallel pass per step; only the inverse (density of an arbitrary z) is sequential;
  - variable order is reversed between steps; s_t is initialised positive (a 'forget-gate bias');
  - linear IAF (Appendix A): z = L(x) y with unit lower-triangular L  -> a full-covariance Gaussian;
  - free bits (Appendix C.8, Eq. 15) and the discretized logistic likelihood (Appendix C.5);
  - planar flow (Eq. 6, Rezende & Mohamed 2015) for comparison.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


# ---------------------------------------------------------------------------------------------------- MADE
class MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, mask):
        super().__init__(n_in, n_out)
        self.register_buffer("mask", mask.float())                                 # n_out x n_in

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked autoencoder (Germain et al. 2015) with `n_out` outputs per input dimension, each output i depending
    only on inputs 1..i-1 (strictly autoregressive). An optional context h feeds the first hidden layer unmasked.
    n_hidden_layers = 0 gives a linear autoregressive map (the paper's 'IAF with 0 hidden layers')."""

    def __init__(self, D, hidden, n_hidden_layers=1, context=0, n_out=2):
        super().__init__()
        self.D, self.n_out = D, n_out
        deg_in = torch.arange(1, D + 1)
        layers, prev = [], deg_in
        for _ in range(n_hidden_layers):
            deg_h = torch.arange(hidden) % max(D - 1, 1) + 1                         # degrees 1..D-1, cycled
            layers.append(MaskedLinear(len(prev), hidden, deg_h[:, None] >= prev[None, :]))
            prev = deg_h
        deg_out = deg_in.repeat(n_out)
        self.hidden = nn.ModuleList(layers)
        self.out = MaskedLinear(len(prev), D * n_out, deg_out[:, None] > prev[None, :])
        self.ctx = nn.Linear(context, hidden if n_hidden_layers else D * n_out, bias=False) if context else None

    def forward(self, z, h=None):
        a = z
        for k, layer in enumerate(self.hidden):
            a = layer(a)
            if k == 0 and self.ctx is not None:
                a = a + self.ctx(h)
            a = F.elu(a)
        out = self.out(a)
        if not self.hidden and self.ctx is not None:
            out = out + self.ctx(h)
        return out.view(*z.shape[:-1], self.n_out, self.D).unbind(-2)


# ---------------------------------------------------------------------------------------------------- IAF
class IAFStep(nn.Module):
    """One IAF transformation.
    mode='gated'    : sigma = sigmoid(s + bias), z' = sigma z + (1 - sigma) m          (Eqs. 13-14, Algorithm 1)
    mode='affine'   : sigma = exp(s), z' = m + sigma z                                   (Eq. 10)
    mode='location' : sigma = 1, z' = z + m                      (Table 3's 'location-only', volume preserving)
    Returns (z', log det) with log det = sum_i log sigma_i."""

    def __init__(self, D, hidden=64, n_hidden_layers=1, context=0, mode="gated", forget_bias=1.5):
        super().__init__()
        self.made = MADE(D, hidden, n_hidden_layers, context, n_out=2)
        self.mode, self.bias = mode, forget_bias
        nn.init.zeros_(self.made.out.weight); nn.init.zeros_(self.made.out.bias)  # start near the identity

    def params(self, z, h=None):
        m, s = self.made(z, h)
        if self.mode == "gated":
            return m, torch.sigmoid(s + self.bias)
        if self.mode == "affine":
            return m, s.exp()
        return m, torch.ones_like(m)

    def forward(self, z, h=None):
        m, sig = self.params(z, h)
        if self.mode == "gated":
            out = sig * z + (1 - sig) * m
        else:
            out = m + sig * z
        return out, sig.log().sum(-1)

    def inverse(self, y, h=None):
        """Recover z from y = step(z): sequential, D passes (this is why IAF is fast to SAMPLE, slow to EVALUATE
        an arbitrary point, the reverse of an autoregressive density model)."""
        z = torch.zeros_like(y)
        for i in range(y.shape[-1]):
            m, sig = self.params(z, h)                                             # entry i only needs z_<i
            if self.mode == "gated":
                z[..., i] = (y[..., i] - (1 - sig[..., i]) * m[..., i]) / sig[..., i]
            else:
                z[..., i] = (y[..., i] - m[..., i]) / sig[..., i]
        return z


class IAFPosterior(nn.Module):
    """Algorithm 1. Given the encoder's (mu_0, log sigma_0^2, h): z_0 = mu_0 + sigma_0 * eps, then T IAF steps
    (variable order reversed between steps). Returns (z_T, log q(z_T|x))."""

    def __init__(self, D, T=2, hidden=64, n_hidden_layers=1, context=0, mode="gated", reverse=True):
        super().__init__()
        self.steps = nn.ModuleList([IAFStep(D, hidden, n_hidden_layers, context, mode) for _ in range(T)])
        self.reverse = reverse

    def forward(self, mu, logvar, h=None, eps=None):
        if eps is None:
            eps = torch.randn_like(mu)
        z = mu + (0.5 * logvar).exp() * eps
        log_q = -(0.5 * eps ** 2 + 0.5 * LOG2PI + 0.5 * logvar).sum(-1)             # log N(z_0; mu, sigma^2)
        for t, step in enumerate(self.steps):
            if self.reverse and t > 0:
                z = z.flip(-1)                                                     # volume preserving
            z, logdet = step(z, h)
            log_q = log_q - logdet                                                 # Eq. 5 / Eq. 11
        if self.reverse and len(self.steps) > 1 and len(self.steps) % 2 == 0:
            z = z.flip(-1)                                                         # undo an odd number of flips
        return z, log_q


class LinearIAF(nn.Module):
    """Appendix A: z = L y with L unit lower-triangular (from the encoder), y ~ N(mu, diag sigma^2).
    det L = 1, so log q(z) = log q(y): one step turns a diagonal Gaussian into ANY full-covariance Gaussian."""

    def __init__(self, D, context):
        super().__init__()
        self.D = D
        self.L = nn.Linear(context, D * D)
        nn.init.zeros_(self.L.weight); nn.init.zeros_(self.L.bias)

    def matrix(self, h):
        L = self.L(h).view(*h.shape[:-1], self.D, self.D).tril(-1)
        return L + torch.eye(self.D)

    def forward(self, mu, logvar, h, eps=None):
        if eps is None:
            eps = torch.randn_like(mu)
        y = mu + (0.5 * logvar).exp() * eps
        log_q = -(0.5 * eps ** 2 + 0.5 * LOG2PI + 0.5 * logvar).sum(-1)
        return (self.matrix(h) @ y.unsqueeze(-1)).squeeze(-1), log_q


class PlanarFlow(nn.Module):
    """Eq. 6 (Rezende & Mohamed 2015): f(z) = z + u tanh(w.z + b); a one-unit bottleneck per step.
    u is constrained so w.u >= -1 (invertibility). log|det| = log|1 + u.psi|, psi = (1 - tanh^2) w."""

    def __init__(self, D, K=8):
        super().__init__()
        self.u, self.w = nn.Parameter(torch.randn(K, D) * 0.01), nn.Parameter(torch.randn(K, D) * 0.01)
        self.b = nn.Parameter(torch.zeros(K))

    def forward(self, z, h=None):
        logdet = torch.zeros(z.shape[:-1])
        for k in range(len(self.b)):
            w, u = self.w[k], self.u[k]
            wu = (w * u).sum()
            u_hat = u + (-1 + F.softplus(wu) - wu) * w / (w * w).sum().clamp_min(1e-8)
            a = torch.tanh(z @ w + self.b[k])
            z = z + u_hat * a.unsqueeze(-1)
            psi = (1 - a ** 2).unsqueeze(-1) * w
            logdet = logdet + (1 + psi @ u_hat).abs().clamp_min(1e-8).log()
        return z, logdet


# ---------------------------------------------------------------------------------------------------- objectives
def free_bits_kl(kl_per_group, lam):
    """Eq. 15: kl_per_group (batch, K) holds KL(q(z_j|x) || p(z_j)) estimates for each group j. Each group's
    minibatch-mean KL is replaced by max(lambda, mean) -> no gradient pressure below lambda nats."""
    return torch.clamp(kl_per_group.mean(0), min=lam).sum()


def discretized_logistic_log_prob(x, mu, log_scale, bins=256):
    """Appendix C.5: P(x) = CDF(x + 1/bins) - CDF(x) under a logistic(mu, s); x in {0, 1/bins, ..., (bins-1)/bins}.
    The two edge bins take all the remaining mass (so the probabilities sum to exactly 1)."""
    s = log_scale.exp()
    upper = torch.sigmoid((x + 1 / bins - mu) / s)
    lower = torch.sigmoid((x - mu) / s)
    upper = torch.where(x >= (bins - 1) / bins - 1e-6, torch.ones_like(upper), upper)
    lower = torch.where(x <= 1e-6, torch.zeros_like(lower), lower)
    return (upper - lower).clamp_min(1e-12).log()


def log_standard_normal(z):
    return -0.5 * (LOG2PI + z ** 2).sum(-1)


# ---------------------------------------------------------------------------------------------------- a VAE
class FlowVAE(nn.Module):
    """A small MLP VAE whose posterior is diagonal ('diag'), linear IAF ('linear'), IAF ('iaf') or planar ('planar').
    Bernoulli decoder; the encoder also outputs a context h for the flow (Algorithm 1)."""

    def __init__(self, x_dim, z_dim=32, hidden=256, posterior="iaf", T=2, iaf_hidden=128, ctx=64):
        super().__init__()
        self.z_dim, self.posterior = z_dim, posterior
        self.enc = nn.Sequential(nn.Linear(x_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.mu, self.logvar, self.h = nn.Linear(hidden, z_dim), nn.Linear(hidden, z_dim), nn.Linear(hidden, ctx)
        self.dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, x_dim))
        if posterior == "iaf":
            self.flow = IAFPosterior(z_dim, T, iaf_hidden, 1, ctx)
        elif posterior == "linear":
            self.flow = LinearIAF(z_dim, ctx)
        elif posterior == "planar":
            self.flow = PlanarFlow(z_dim, K=T)

    def q_sample(self, x, n=1):
        a = self.enc(x)
        mu, logvar, h = self.mu(a), self.logvar(a), self.h(a)
        mu, logvar, h = (t.expand(n, *t.shape) for t in (mu, logvar, h))
        if self.posterior == "diag":
            eps = torch.randn_like(mu)
            return mu + (0.5 * logvar).exp() * eps, -(0.5 * eps ** 2 + 0.5 * LOG2PI + 0.5 * logvar).sum(-1)
        if self.posterior == "planar":
            eps = torch.randn_like(mu)
            z0 = mu + (0.5 * logvar).exp() * eps
            z, logdet = self.flow(z0)
            return z, -(0.5 * eps ** 2 + 0.5 * LOG2PI + 0.5 * logvar).sum(-1) - logdet
        return self.flow(mu, logvar, h)

    def log_weights(self, x, n=1):
        """log p(x, z) - log q(z|x) for n posterior samples: (n, B)."""
        z, log_q = self.q_sample(x, n)
        logits = self.dec(z)
        log_px = -F.binary_cross_entropy_with_logits(logits, x.expand_as(logits), reduction="none").sum(-1)
        return log_px + log_standard_normal(z) - log_q

    def elbo(self, x, n=1):
        return self.log_weights(x, n).mean(0)

    @torch.no_grad()
    def log_likelihood(self, x, n=128):
        """Table 1's right column: importance-sampled log p(x) with 128 samples."""
        return torch.logsumexp(self.log_weights(x, n), 0) - math.log(n)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
