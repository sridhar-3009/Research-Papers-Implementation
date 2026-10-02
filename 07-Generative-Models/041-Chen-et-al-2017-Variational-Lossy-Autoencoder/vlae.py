"""Variational Lossy Autoencoder (Chen, Kingma, Salimans, Duan, Dhariwal, Schulman, Sutskever & Abbeel, ICLR 2017).

  bits-back code length:  C(x) = E_q[log q(z|x) - log p(z) - log p(x|z)] = -L(x)                    (Eqs. 6-7)
                          = -log p(x) + KL(q(z|x) || p(z|x))  >=  H(data) + E KL(q || posterior)       (Eqs. 8-11)
  -> 'information preference': whatever the decoder p(x|z) can model WITHOUT z is modelled there, because
     routing it through z costs the extra KL(q || true posterior). Only the rest goes into z.
  -> lossy code by design (Section 3.1): a decoder p_local(x|z) = prod_i p(x_i | z, x_WindowAround(i)) models
     local statistics itself, so z keeps only what a small window can't see (global structure).
  -> autoregressive-flow (AF) prior (Section 3.2): z = f(eps), log p(z) = log u(eps) + log|det d eps/dz|;
     with eps = f^-1(z) it is the SAME bound as an IAF posterior, but with a deeper generator (Eqs. 12-14).

Contents: a 1-D windowed autoregressive decoder (for the toy), a 2-D PixelCNN whose receptive field is an exact
A x B window (Section 4.3 notation) or a stack of 3x3 masked convolutions (Section 4: 6 layers), an optional
grayscale context, the AF prior, the VLAE model, and code-length bookkeeping.
"""

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)

_spec = importlib.util.spec_from_file_location(
    "iaf040", Path(__file__).resolve().parents[1] / "040-Kingma-et-al-2016-Inverse-Autoregressive-Flow" / "iaf.py")
iaf040 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(iaf040)
MADE, discretized_logistic_log_prob = iaf040.MADE, iaf040.discretized_logistic_log_prob


def log_standard_normal(z):
    return -0.5 * (LOG2PI + z ** 2).sum(-1)


# ---------------------------------------------------------------------------------------------------- 1-D decoder
class WindowAR1d(nn.Module):
    """p(x|z) = prod_i Bernoulli(x_i | z, x_{i-w..i-1}) over a binary sequence.
    window = 0: factorized decoder (sees only z); window >= length: a full autoregressive decoder."""

    def __init__(self, length, window, z_dim, hidden=64):
        super().__init__()
        self.length, self.window = length, window
        self.conv = nn.Linear(window, hidden) if window > 0 else None                # applied to each window
        self.z = nn.Linear(z_dim, hidden * length) if z_dim else None
        self.pos = nn.Parameter(torch.zeros(hidden, length))
        self.out = nn.Sequential(nn.ReLU(), nn.Conv1d(hidden, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, 1, 1))
        self.hidden = hidden

    def logits(self, x, z=None):
        lead = x.shape[:-1]
        x = x.reshape(-1, 1, self.length)
        a = self.pos.expand(len(x), -1, -1)
        if self.conv is not None:                                                # shift right by one: x_<i only
            win = F.pad(x, (self.window, 0))[:, 0].unfold(-1, self.window, 1)[:, :self.length]   # B, L, w: x_{i-w..i-1}
            a = a + self.conv(win).transpose(1, 2)
        if z is not None and self.z is not None:
            a = a + self.z(z.reshape(-1, z.shape[-1])).view(-1, self.hidden, self.length)
        return self.out(a).view(*lead, self.length)

    def log_prob(self, x, z=None):
        lg = self.logits(x, z)
        return -F.binary_cross_entropy_with_logits(lg, x.expand_as(lg), reduction="none").sum(-1)


# ---------------------------------------------------------------------------------------------------- 2-D decoder
def window_mask(A, B):
    """Section 4.3: pixel i sees the A-wide, B-tall block directly above it plus (A-1)/2 pixels to its left.
    Returns a (B+1) x A kernel mask; the kernel's centre column sits on the pixel, its last row on the pixel's row."""
    m = torch.zeros(B + 1, A)
    m[:B] = 1
    m[B, : (A - 1) // 2] = 1
    return m


class MaskedConv2d(nn.Conv2d):
    def __init__(self, cin, cout, mask):
        kh, kw = mask.shape
        super().__init__(cin, cout, (kh, kw))
        self.register_buffer("mask", mask[None, None].float())
        self.kh, self.kw = kh, kw

    def forward(self, x):
        left = (self.kw - 1) // 2                                                  # kernel column 'left' = the pixel
        x = F.pad(x, (left, self.kw - 1 - left, self.kh - 1, 0))                   # rows above, columns both sides
        return F.conv2d(x, self.weight * self.mask, self.bias)


def pixelcnn_mask(k, include_centre):
    """Standard PixelCNN k x k mask restricted to rows above + left (mask 'A' excludes the centre, 'B' keeps it).
    Returned in the 'rows above only' layout used by MaskedConv2d: ((k+1)/2) x k."""
    m = torch.zeros((k + 1) // 2, k)
    m[:-1] = 1
    m[-1, : k // 2 + (1 if include_centre else 0)] = 1
    return m


class LocalPixelCNN(nn.Module):
    """p(x|z) = prod_i p(x_i | z, x_window(i)).
      window=(A, B): one masked conv with an exact A x B window, then 1x1 layers (receptive field is exactly it);
      window=None:   `layers` stacked 3x3 masked convs (mask A, then B): the paper's 6-layer, filter-3 decoder.
    gray_context=True feeds only the grayscale of the window (Figure 3d). likelihood: 'bernoulli' or 'logistic'.
    The global code z enters every layer as a learned spatial map."""

    def __init__(self, C=1, size=28, z_dim=32, channels=64, window=(5, 3), layers=6, gray_context=False,
                 likelihood="bernoulli", depth_1x1=3):
        super().__init__()
        self.C, self.size, self.gray, self.likelihood = C, size, gray_context, likelihood
        cin = 1 if gray_context else C
        if window is not None:
            convs = [MaskedConv2d(cin, channels, window_mask(*window))] + \
                    [nn.Conv2d(channels, channels, 1) for _ in range(depth_1x1)]
        else:
            convs = [MaskedConv2d(cin, channels, pixelcnn_mask(3, False))] + \
                    [MaskedConv2d(channels, channels, pixelcnn_mask(3, True)) for _ in range(layers - 1)]
        self.convs = nn.ModuleList(convs)
        self.zmap = nn.Linear(z_dim, channels * 7 * 7) if z_dim else None
        self.channels = channels
        self.out = nn.Conv2d(channels, C, 1)
        self.log_scale = nn.Parameter(torch.full((C,), -3.0))

    def features(self, x, z=None):
        h = x.mean(1, keepdim=True) if self.gray else x
        zf = None
        if z is not None and self.zmap is not None:
            zf = F.interpolate(self.zmap(z).view(-1, self.channels, 7, 7), size=(self.size, self.size), mode="nearest")
        for k, conv in enumerate(self.convs):
            h = conv(h)
            if zf is not None:
                h = h + zf
            h = F.elu(h)
        return self.out(h)

    def log_prob(self, x, z=None):
        """x: (..., C, H, W) with the same leading dims as z's (..., z_dim)."""
        lead = x.shape[:-3]
        xf = x.reshape(-1, *x.shape[-3:])
        o = self.features(xf, None if z is None else z.reshape(-1, z.shape[-1]))
        if self.likelihood == "bernoulli":
            lp = -F.binary_cross_entropy_with_logits(o, xf, reduction="none")
        else:
            lp = discretized_logistic_log_prob(xf, torch.sigmoid(o), self.log_scale[:, None, None])
        return lp.flatten(1).sum(-1).view(lead)

    @torch.no_grad()
    def sample(self, z):
        """Pixel-by-pixel ancestral sampling (binary or 8-bit)."""
        B = z.shape[0]
        x = torch.zeros(B, self.C, self.size, self.size)
        for r in range(self.size):
            for c in range(self.size):
                o = self.features(x, z)[:, :, r, c]
                if self.likelihood == "bernoulli":
                    x[:, :, r, c] = torch.bernoulli(torch.sigmoid(o))
                else:
                    u = torch.rand_like(o).clamp(1e-5, 1 - 1e-5)
                    v = torch.sigmoid(o) + self.log_scale.exp() * (u.log() - (1 - u).log())
                    x[:, :, r, c] = (v.clamp(0, 255 / 256) * 256).floor() / 256
        return x


def receptive_field(decoder, C=1, size=12, pixel=(6, 6)):
    """Which input pixels can influence the prediction at `pixel` (via autograd), as a size x size 0/1 map."""
    x = torch.rand(1, C, size, size, requires_grad=True)
    decoder.size = size
    o = decoder.features(x)
    o[0, :, pixel[0], pixel[1]].sum().backward()
    return (x.grad.abs().sum(1)[0] > 0).int()


# ---------------------------------------------------------------------------------------------------- AF prior
class AFPrior(nn.Module):
    """Section 3.2: z = f(eps), eps ~ N(0, I), f an autoregressive flow with z_i = eps_i * sigma_i(z_<i) + mu_i(z_<i).
    Density (parallel, one MADE pass per step):  eps_i = (z_i - mu_i(z_<i)) / sigma_i(z_<i),
      log p(z) = log N(eps; 0, I) - sum_i log sigma_i.
    Sampling is sequential over dimensions (only needed for generation, never for training)."""

    def __init__(self, D, T=1, hidden=128, n_hidden_layers=1):
        super().__init__()
        self.D = D
        self.mades = nn.ModuleList([MADE(D, hidden, n_hidden_layers, 0, n_out=2) for _ in range(T)])
        for m in self.mades:
            nn.init.zeros_(m.out.weight); nn.init.zeros_(m.out.bias)              # start as N(0, I)

    def _ms(self, made, z):
        mu, s = made(z)
        return mu, F.softplus(s + 0.5413) + 1e-3                                  # sigma = 1 at init

    def inverse(self, z):
        """z -> eps with the total log|det d eps/dz| (whitening, the IAF direction)."""
        logdet = torch.zeros(z.shape[:-1])
        for t, made in enumerate(reversed(self.mades)):
            mu, sig = self._ms(made, z)
            z = (z - mu) / sig
            logdet = logdet - sig.log().sum(-1)
            if t < len(self.mades) - 1:
                z = z.flip(-1)
        return z, logdet

    def log_prob(self, z):
        eps, logdet = self.inverse(z)
        return log_standard_normal(eps) + logdet

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.D)
        for t, made in enumerate(self.mades):
            if t > 0:
                z = z.flip(-1)
            eps, out = z, torch.zeros_like(z)
            for i in range(self.D):                                                # sequential: z_i needs z_<i
                mu, sig = self._ms(made, out)
                out[:, i] = eps[:, i] * sig[:, i] + mu[:, i]
            z = out
        return z


# ---------------------------------------------------------------------------------------------------- the model
class VLAE(nn.Module):
    """Encoder (MLP or conv) -> diagonal q(z|x); prior N(0, I) or AF; decoder: any module with log_prob(x, z)."""

    def __init__(self, encoder, enc_dim, z_dim, decoder, prior="af", af_steps=1, af_hidden=128):
        super().__init__()
        self.encoder, self.decoder, self.z_dim = encoder, decoder, z_dim
        self.mu, self.logvar = nn.Linear(enc_dim, z_dim), nn.Linear(enc_dim, z_dim)
        self.prior = AFPrior(z_dim, af_steps, af_hidden) if prior == "af" else None

    def log_prior(self, z):
        return log_standard_normal(z) if self.prior is None else self.prior.log_prob(z)

    def terms(self, x, n=1):
        """Per sample: (log p(x|z), log p(z), log q(z|x)), each (n, B)."""
        a = self.encoder(x)
        mu, logvar = self.mu(a), self.logvar(a)
        eps = torch.randn(n, *mu.shape)
        z = mu + (0.5 * logvar).exp() * eps
        log_q = -(0.5 * eps ** 2 + 0.5 * LOG2PI + 0.5 * logvar).sum(-1)
        xe = x.expand(n, *x.shape)
        return self.decoder.log_prob(xe, z), self.log_prior(z), log_q

    def elbo(self, x, n=1):
        rec, lp, lq = self.terms(x, n)
        return (rec + lp - lq).mean(0)

    def kl(self, x, n=16):
        """E_q[log q(z|x) - log p(z)]: the nats the code z carries for x (the 'lossy code length')."""
        _, lp, lq = self.terms(x, n)
        return (lq - lp).mean(0)

    @torch.no_grad()
    def log_likelihood(self, x, n=128):
        rec, lp, lq = self.terms(x, n)
        return torch.logsumexp(rec + lp - lq, 0) - math.log(n)


def code_lengths(rec, log_p_z, log_q):
    """Eqs. 5-7 (nats): the naive two-part code E[-log p(z) - log p(x|z)] and the bits-back code, which gives back
    the H(q) nats that the choice of z carries: E[log q - log p(z) - log p(x|z)] = -ELBO."""
    naive = (-log_p_z - rec).mean()
    return {"naive": naive.item(), "bits-back": (log_q - log_p_z - rec).mean().item(),
            "refund = H(q)": (-log_q).mean().item()}


# ---------------------------------------------------------------------------------------------------- toy data
def toy_data(n, L=16, p_copy=0.6):
    """One GLOBAL bit g ~ Bernoulli(1/2); each x_i copies x_(i-1) with prob p_copy (LOCAL structure), else is a
    fresh coin with P(1) = 0.85 if g else 0.15. A small window sees the copying but can only guess g."""
    g = (torch.rand(n) < 0.5).float()
    p = 0.15 + 0.7 * g
    x = torch.zeros(n, L)
    x[:, 0] = torch.bernoulli(p)
    for i in range(1, L):
        x[:, i] = torch.where(torch.rand(n) < p_copy, x[:, i - 1], torch.bernoulli(p))
    return x


def toy_true_log_prob(x, p_copy=0.6):
    """Exact log p(x): a mixture over g of two Markov chains."""
    out = []
    for g in (0.0, 1.0):
        p = 0.15 + 0.7 * g
        lp = torch.where(x[:, 0] == 1, torch.tensor(math.log(p)), torch.tensor(math.log(1 - p)))
        for i in range(1, x.shape[1]):
            fresh = torch.where(x[:, i] == 1, torch.tensor(p), torch.tensor(1 - p))
            same = (x[:, i] == x[:, i - 1]).float()
            lp = lp + torch.log(p_copy * same + (1 - p_copy) * fresh)
        out.append(lp + math.log(0.5))
    return torch.logsumexp(torch.stack(out), 0)


def train_toy(window, steps=400, seed=0, X=None, L=16):
    """A VLAE (N(0, I) prior, 4 latents) with a window-`window` decoder on the toy. Returns the model."""
    torch.manual_seed(seed)
    X = toy_data(20000, L) if X is None else X
    m = VLAE(nn.Sequential(nn.Linear(L, 32), nn.ELU()), 32, 4, WindowAR1d(L, window, 4, 32), prior="normal")
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(steps):
        loss = -m.elbo(X[torch.randint(0, len(X), (256,))]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())
