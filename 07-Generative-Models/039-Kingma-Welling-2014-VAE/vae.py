"""Auto-Encoding Variational Bayes (Kingma & Welling, ICLR 2014): the variational auto-encoder.

  log p(x) = KL(q(z|x) || p(z|x)) + L(x)                                                     (Eq. 1)
  L(x)     = E_q[log p(x|z)] - KL(q(z|x) || p(z))                       the lower bound (ELBO)  (Eq. 3)
  z        = mu + sigma * eps,  eps ~ N(0, I)                         the reparameterization (Eq. 4)
  L_A      = 1/L sum_l [log p(x, z_l) - log q(z_l|x)]                generic SGVB estimator  (Eq. 6)
  L_B      = -KL (closed form) + 1/L sum_l log p(x|z_l)              second SGVB estimator   (Eq. 7)
  -KL      = 1/2 sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)       Gaussian case (App. B, Eq. 10)
  - encoder q(z|x): Gaussian MLP with one tanh hidden layer (App. C.2);
  - decoder p(x|z): Bernoulli MLP (binary data, App. C.1) or Gaussian MLP with sigmoid means (Frey Face);
  - baselines: the wake-sleep algorithm and Monte Carlo EM with Hybrid (Hamiltonian) Monte Carlo;
  - Appendix D's marginal likelihood estimator (posterior samples by HMC + a fitted density q(z)).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def log_normal_diag(x, mu, logvar):
    """log N(x; mu, diag(exp(logvar))), summed over the last dimension."""
    return -0.5 * (LOG2PI + logvar + (x - mu) ** 2 / logvar.exp()).sum(-1)


def log_standard_normal(z):
    return -0.5 * (LOG2PI + z ** 2).sum(-1)


def kl_to_standard_normal(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) in closed form (Appendix B): -1/2 sum (1 + log s^2 - mu^2 - s^2)."""
    return -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1)


def reparameterize(mu, logvar, eps=None):
    """Eq. 4 for the Gaussian: z = mu + sigma * eps. The randomness lives in eps, so z is differentiable in
    mu and sigma."""
    if eps is None:
        eps = torch.randn_like(mu)
    return mu + (0.5 * logvar).exp() * eps


class GaussianMLP(nn.Module):
    """App. C.2: h = tanh(W3 in + b3), mu = W4 h + b4, log sigma^2 = W5 h + b5.
    sigmoid_mean=True squashes mu into (0, 1) (the paper's Frey Face decoder)."""

    def __init__(self, n_in, n_hidden, n_out, sigmoid_mean=False):
        super().__init__()
        self.h, self.mu, self.logvar = nn.Linear(n_in, n_hidden), nn.Linear(n_hidden, n_out), nn.Linear(n_hidden, n_out)
        self.sigmoid_mean = sigmoid_mean

    def forward(self, v):
        h = torch.tanh(self.h(v))
        mu = self.mu(h)
        return (torch.sigmoid(mu) if self.sigmoid_mean else mu), self.logvar(h)


class BernoulliMLP(nn.Module):
    """App. C.1: y = sigmoid(W2 tanh(W1 z + b1) + b2); returns the logits W2 tanh(.) + b2."""

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.h, self.out = nn.Linear(n_in, n_hidden), nn.Linear(n_hidden, n_out)

    def forward(self, z):
        return self.out(torch.tanh(self.h(z)))


class VAE(nn.Module):
    def __init__(self, x_dim=784, z_dim=20, hidden=500, decoder="bernoulli", init_std=0.01):
        super().__init__()
        self.z_dim, self.kind = z_dim, decoder
        self.enc = GaussianMLP(x_dim, hidden, z_dim)                               # q_phi(z|x)
        self.dec = BernoulliMLP(z_dim, hidden, x_dim) if decoder == "bernoulli" else \
            GaussianMLP(z_dim, hidden, x_dim, sigmoid_mean=True)                    # p_theta(x|z)
        for p in self.parameters():                                                 # Section 5: N(0, 0.01)
            nn.init.normal_(p, 0.0, init_std)

    # ------------------------------------------------------------------------------------------- densities
    def log_px_given_z(self, x, z):
        """log p(x|z); x is broadcast against z's leading sample dimensions."""
        if self.kind == "bernoulli":
            logits = self.dec(z)
            return -F.binary_cross_entropy_with_logits(logits, x.expand_as(logits), reduction="none").sum(-1)
        mu, logvar = self.dec(z)
        return log_normal_diag(x, mu, logvar)

    def log_q(self, z, x):
        mu, logvar = self.enc(x)
        return log_normal_diag(z, mu, logvar)

    # ------------------------------------------------------------------------------------------- the bound
    def elbo(self, x, L=1, estimator="B"):
        """Per-datapoint SGVB estimate of the lower bound. 'B' = Eq. 7/10 (analytic KL), 'A' = Eq. 6."""
        mu, logvar = self.enc(x)
        eps = torch.randn(L, *mu.shape, device=x.device)
        z = reparameterize(mu, logvar, eps)                                         # L, B, J
        rec = self.log_px_given_z(x, z)                                             # L, B
        if estimator == "B":
            return rec.mean(0) - kl_to_standard_normal(mu, logvar)
        return (rec + log_standard_normal(z) - log_normal_diag(z, mu, logvar)).mean(0)

    def map_objective(self, x, N, L=1, estimator="B"):
        """Section 5: the minibatch bound (Eq. 8, per datapoint) plus the weight-decay prior p(theta) = N(0, I),
        which over the full dataset adds -1/2 |theta|^2, i.e. -1/2 |theta|^2 / N per datapoint."""
        prior = sum((p ** 2).sum() for p in self.parameters())
        return self.elbo(x, L, estimator).mean() - 0.5 * prior / N

    # ------------------------------------------------------------------------------------------- generation
    @torch.no_grad()
    def decode_mean(self, z):
        return torch.sigmoid(self.dec(z)) if self.kind == "bernoulli" else self.dec(z)[0]

    @torch.no_grad()
    def sample(self, n, mean=True):
        """Ancestral sampling: z ~ p(z) = N(0, I), then x ~ p(x|z) (or its mean)."""
        z = torch.randn(n, self.z_dim)
        m = self.decode_mean(z)
        if mean:
            return m
        if self.kind == "bernoulli":
            return torch.bernoulli(m)
        return m + (0.5 * self.dec(z)[1]).exp() * torch.randn_like(m)

    @torch.no_grad()
    def manifold(self, n=20):
        """Figure 4: a regular grid on the unit square pushed through the Gaussian inverse CDF, then decoded.
        Returns (n*n, x_dim) in row-major order. Only for z_dim = 2."""
        u = torch.linspace(0.5 / n, 1 - 0.5 / n, n)
        g = torch.distributions.Normal(0., 1.).icdf(u)
        z = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        return self.decode_mean(z)

    @torch.no_grad()
    def importance_log_likelihood(self, x, K=500):
        """A check that is NOT from this paper (Burda et al. 2016 / Rezende et al. 2014): with z_k ~ q(z|x),
        log p(x) ~ log mean_k p(x, z_k) / q(z_k|x). It is >= the ELBO in expectation and tends to log p(x)."""
        mu, logvar = self.enc(x)
        z = reparameterize(mu, logvar, torch.randn(K, *mu.shape))
        w = self.log_px_given_z(x, z) + log_standard_normal(z) - log_normal_diag(z, mu, logvar)
        return torch.logsumexp(w, 0) - math.log(K)


# ---------------------------------------------------------------------------------------------- gradient estimators
def score_function_grad(f, mu, log_sigma, n):
    """Section 2.2's 'naive' estimator of d/d(mu, log_sigma) E_{N(mu, sigma^2)}[f(z)]:
    f(z) * grad log q(z) per sample. Returns per-sample gradients (n, 2) so the variance can be measured."""
    s = math.exp(log_sigma)
    z = mu + s * torch.randn(n)
    fz = f(z)
    d_mu = (z - mu) / s ** 2                                                       # d log q / d mu
    d_ls = ((z - mu) ** 2) / s ** 2 - 1                                            # d log q / d log sigma
    return torch.stack([fz * d_mu, fz * d_ls], 1)


def reparam_grad(f, mu, log_sigma, n):
    """Section 2.4's estimator: z = mu + sigma eps, then differentiate f(z) through z. Per-sample (n, 2):
    each sample gets its own copy of (mu, log_sigma) so autograd returns per-sample gradients."""
    m = torch.full((n,), float(mu), requires_grad=True)
    ls = torch.full((n,), float(log_sigma), requires_grad=True)
    gm, gs = torch.autograd.grad(f(m + ls.exp() * torch.randn(n)).sum(), (m, ls))
    return torch.stack([gm, gs], 1)


# ---------------------------------------------------------------------------------------------- baselines
def wake_sleep_losses(model, x):
    """The wake-sleep algorithm (Hinton et al. 1995), the paper's main baseline, as two losses:
      wake  (updates theta): z ~ q(z|x) (no gradient into q), maximise log p(x, z);
      sleep (updates phi):   z ~ p(z), x' ~ p(x|z) (a 'dream'), maximise log q(z|x').
    They don't add up to one objective, which is the paper's criticism. Use separate optimizers."""
    with torch.no_grad():
        mu, logvar = model.enc(x)
        z = reparameterize(mu, logvar)
    wake = -(model.log_px_given_z(x, z) + log_standard_normal(z)).mean()
    with torch.no_grad():
        zd = torch.randn(len(x), model.z_dim)
        if model.kind == "bernoulli":
            xd = torch.bernoulli(torch.sigmoid(model.dec(zd)))
        else:
            m, lv = model.dec(zd)
            xd = m + (0.5 * lv).exp() * torch.randn_like(m)
    sleep = -model.log_q(zd, xd).mean()
    return wake, sleep


def hmc(log_prob, z, n_steps, leapfrog=10, step=0.05):
    """Hybrid (Hamiltonian) Monte Carlo on a batch of independent chains. log_prob(z) -> (B,).
    Returns (samples (n_steps, B, J), acceptance rate)."""
    def grad(zz):
        zz = zz.detach().requires_grad_(True)
        lp = log_prob(zz)
        return lp.detach(), torch.autograd.grad(lp.sum(), zz)[0]

    lp, g = grad(z)
    out, acc = [], 0.0
    for _ in range(n_steps):
        p0 = torch.randn_like(z)
        zn, p, gn = z.clone(), p0 + 0.5 * step * g, g
        for k in range(leapfrog):
            zn = zn + step * p
            lpn, gn = grad(zn)
            if k < leapfrog - 1:
                p = p + step * gn
        p = p + 0.5 * step * gn
        log_accept = (lpn - 0.5 * (p ** 2).sum(-1)) - (lp - 0.5 * (p0 ** 2).sum(-1))
        a = torch.rand_like(lp).log() < log_accept
        z = torch.where(a[:, None], zn, z)
        lp = torch.where(a, lpn, lp)
        g = torch.where(a[:, None], gn, g)
        out.append(z.clone()); acc += a.float().mean().item()
    return torch.stack(out), acc / n_steps


def log_joint(model, x):
    """z -> log p(x, z) = log p(z) + log p(x|z) for a fixed batch x (for HMC)."""
    return lambda z: log_standard_normal(z) + model.log_px_given_z(x, z)


def marginal_likelihood_appendix_d(model, x, L=100, burn=50, leapfrog=4, step=0.05):
    """Appendix D: (1) HMC samples from p(z|x); (2) fit a density q(z) (here: a full-covariance Gaussian per
    datapoint); (3) with fresh posterior samples, 1/p(x) ~ mean_l q(z_l) / (p(z_l) p(x|z_l)).
    Returns log p(x) estimates (B,) and the HMC acceptance rate."""
    lj = log_joint(model, x)
    with torch.no_grad():
        z0 = model.enc(x)[0]
    s1, acc = hmc(lj, z0, burn + L, leapfrog, step)
    s1 = s1[burn:]                                                                # L, B, J
    m = s1.mean(0)
    c = s1 - m
    cov = torch.einsum("lbi,lbj->bij", c, c) / (L - 1) + 1e-4 * torch.eye(s1.shape[-1])
    q = torch.distributions.MultivariateNormal(m, covariance_matrix=cov)
    s2, _ = hmc(lj, s1[-1], L, leapfrog, step)
    with torch.no_grad():
        log_ratio = q.log_prob(s2) - torch.stack([lj(s) for s in s2])            # L, B
    return -(torch.logsumexp(log_ratio, 0) - math.log(L)), acc


def count_params(m):
    return sum(p.numel() for p in m.parameters())
