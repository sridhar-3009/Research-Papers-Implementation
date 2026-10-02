"""Generative Adversarial Nets (Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville & Bengio,
NIPS 2014).

  min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]                         (Eq. 1)
  optimal discriminator for a fixed G:   D*_G(x) = p_data(x) / (p_data(x) + p_g(x))                    (Eq. 2)
  virtual criterion:  C(G) = max_D V(G, D) = -log 4 + 2 JSD(p_data || p_g)    -> minimised iff p_g = p_data (Eqs. 4-6)
  Algorithm 1: k discriminator ascent steps, then one generator step (k = 1 in the paper; momentum SGD).
  'Non-saturating' generator loss: maximise log D(G(z)) instead of minimising log(1 - D(G(z))) (Section 3).
  Evaluation: Gaussian Parzen-window log-likelihood of test data under generated samples, sigma by validation.
  Architecture (Section 5): generator = ReLU/sigmoid MLP with noise only at the input; discriminator = maxout MLP
  with dropout.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- networks
class Maxout(nn.Module):
    """Maxout unit (Goodfellow et al. 2013): k linear pieces per output, keep the largest."""

    def __init__(self, n_in, n_out, pieces=5):
        super().__init__()
        self.lin, self.n_out, self.pieces = nn.Linear(n_in, n_out * pieces), n_out, pieces

    def forward(self, x):
        return self.lin(x).view(*x.shape[:-1], self.n_out, self.pieces).max(-1).values


class Generator(nn.Module):
    """z (uniform noise, only at the bottom layer) -> ReLU layers -> sigmoid output in (0, 1) (or linear)."""

    def __init__(self, z_dim=100, hidden=(1200, 1200), x_dim=784, out="sigmoid"):
        super().__init__()
        layers, d = [], z_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, x_dim))
        if out == "sigmoid":
            layers.append(nn.Sigmoid())
        self.net, self.z_dim = nn.Sequential(*layers), z_dim

    def sample_z(self, n):
        return torch.rand(n, self.z_dim) * 2 - 1                                  # uniform on [-1, 1]

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Maxout MLP with dropout; returns the LOGIT of D(x) = P(x came from the data)."""

    def __init__(self, x_dim=784, hidden=(240, 240), pieces=5, dropout=0.5, input_dropout=0.2):
        super().__init__()
        layers, d = [nn.Dropout(input_dropout)], x_dim
        for h in hidden:
            layers += [Maxout(d, h, pieces), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------------------------------- losses
def d_loss(D, real, fake):
    """Minus the discriminator's objective (Algorithm 1): -[log D(x) + log(1 - D(G(z)))], averaged.
    With logits l: log D = logsigmoid(l), log(1 - D) = logsigmoid(-l)."""
    return -(F.logsigmoid(D(real)).mean() + F.logsigmoid(-D(fake)).mean())


def g_loss(D, fake, saturating=False):
    """saturating=True:  minimise log(1 - D(G(z)))  (the minimax loss of Eq. 1);
       saturating=False: minimise -log D(G(z))      (the paper's practical, non-saturating alternative)."""
    l = D(fake)
    return F.logsigmoid(-l).mean() if saturating else -F.logsigmoid(l).mean()


def train_step(G, D, opt_g, opt_d, sample_real, m=100, k=1, saturating=False):
    """One iteration of Algorithm 1: k discriminator steps, then one generator step. Returns the two losses."""
    for _ in range(k):
        real = sample_real(m)
        with torch.no_grad():
            fake = G(G.sample_z(m))
        ld = d_loss(D, real, fake)
        opt_d.zero_grad(); ld.backward(); opt_d.step()
    lg = g_loss(D, G(G.sample_z(m)), saturating)
    opt_g.zero_grad(); lg.backward(); opt_g.step()
    return ld.item(), lg.item()


# ---------------------------------------------------------------------------------------------------- theory
def optimal_discriminator(p_data, p_g):
    """Eq. 2 on any common grid of densities/probabilities."""
    return p_data / (p_data + p_g)


def value(p_data, p_g, D, dx=1.0):
    """V(G, D) = integral of p_data log D + p_g log(1 - D) (Eq. 3) on a grid with spacing dx."""
    eps = 1e-12
    return ((p_data * (D + eps).log() + p_g * (1 - D + eps).log()) * dx).sum()


def kl(p, q, dx=1.0):
    m = p > 0
    return (p[m] * (p[m] / q[m]).log() * dx).sum()


def jsd(p, q, dx=1.0):
    """Jensen-Shannon divergence: 1/2 KL(p || (p+q)/2) + 1/2 KL(q || (p+q)/2); 0 <= JSD <= log 2."""
    m = (p + q) / 2
    return 0.5 * kl(p, m, dx) + 0.5 * kl(q, m, dx)


def virtual_criterion(p_data, p_g, dx=1.0):
    """C(G) = V(G, D*_G) (Eq. 4)."""
    return value(p_data, p_g, optimal_discriminator(p_data, p_g + 1e-300), dx)


# ---------------------------------------------------------------------------------------------------- evaluation
def parzen_log_likelihood(samples, x, sigma, chunk=100):
    """Gaussian Parzen window: log p(x) = log mean_j N(x; s_j, sigma^2 I), evaluated for each row of x."""
    n, d = samples.shape
    out = []
    for k in range(0, len(x), chunk):
        sq = torch.cdist(x[k:k + chunk], samples) ** 2                            # B x n
        out.append(torch.logsumexp(-sq / (2 * sigma ** 2), 1) - math.log(n) - d * math.log(sigma * math.sqrt(2 * math.pi)))
    return torch.cat(out)


def select_sigma(samples, x_valid, sigmas):
    """Cross-validate sigma on held-out data (the paper's procedure)."""
    scores = {s: parzen_log_likelihood(samples, x_valid, s).mean().item() for s in sigmas}
    return max(scores, key=scores.get), scores


def nearest_neighbours(samples, train):
    """Figure 2's rightmost column: the training example closest (Euclidean) to each sample."""
    return train[torch.cdist(samples, train).argmin(1)]


def interpolate(G, z0, z1, steps=8):
    """Figure 3: decode points on the straight line between two noise vectors."""
    t = torch.linspace(0, 1, steps)[:, None]
    with torch.no_grad():
        return G((1 - t) * z0 + t * z1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
