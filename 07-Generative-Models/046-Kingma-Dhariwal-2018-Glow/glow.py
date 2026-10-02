"""Glow: Generative Flow with Invertible 1x1 Convolutions (Kingma & Dhariwal, NeurIPS 2018).

  x <-> h_1 <-> ... <-> z,   log p(x) = log p(z) + sum_i log |det(dh_i / dh_{i-1})|                    (Eqs. 5-7)
  one step of flow = actnorm -> invertible 1x1 convolution -> affine coupling          (Figure 2a, Table 1)
    actnorm:     y = s * x + b per channel, data-dependent init (zero mean, unit variance); logdet h w sum log|s|
    1x1 conv:    y = W x per pixel, W initialised as a random rotation;  logdet h w log|det W|            (Eq. 9)
                 LU form W = P L (U + diag(s)):  logdet = h w sum log|s|                                 (Eqs. 10-11)
    coupling:    (log s, t) = NN(x_b); y_a = s * x_a + t; y_b = x_b; logdet sum log|s|; last conv zero-initialised
  multi-scale (Figure 2b): L levels of [squeeze 2x2 -> K steps -> split half off] (the last level keeps everything)
  continuous objective (Eq. 2): x~ = x + u, u ~ U(0, a); -log p(x~) + M log(1/a) is the code length in nats
  temperature T: sample z with its standard deviation multiplied by T (Section 6)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


# ---------------------------------------------------------------------------------------------------- reshaping
def squeeze2d(x):
    """(B, C, H, W) -> (B, 4C, H/2, W/2): each 2x2 spatial block becomes 4 channels (volume preserving)."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * 4, H // 2, W // 2)


def unsqueeze2d(x):
    B, C, H, W = x.shape
    x = x.view(B, C // 4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C // 4, H * 2, W * 2)


# ---------------------------------------------------------------------------------------------------- layers
class ActNorm(nn.Module):
    """Per-channel scale and bias; the first forward pass sets them so the outputs have zero mean, unit variance."""

    def __init__(self, C):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.register_buffer("initialised", torch.tensor(False))

    def forward(self, x):
        if not self.initialised:
            with torch.no_grad():
                mean = x.mean((0, 2, 3), keepdim=True)
                std = x.std((0, 2, 3), keepdim=True) + 1e-6
                self.loc.copy_(-mean / std)
                self.log_scale.copy_(-std.log())
                self.initialised.fill_(True)
        y = x * self.log_scale.exp() + self.loc
        return y, x.shape[2] * x.shape[3] * self.log_scale.sum()

    def reverse(self, y):
        return (y - self.loc) * torch.exp(-self.log_scale)


class InvConv1x1(nn.Module):
    """y = W x at every pixel. lu=True parameterises W = P L (U + diag(sign * exp(log|s|))) (Eqs. 10-11)."""

    def __init__(self, C, lu=False):
        super().__init__()
        W = torch.linalg.qr(torch.randn(C, C))[0]                                  # random rotation: log|det| = 0
        self.lu = lu
        if not lu:
            self.W = nn.Parameter(W)
        else:
            P, L, U = torch.linalg.lu(W)
            s = torch.diagonal(U)
            self.register_buffer("P", P)
            self.register_buffer("sign_s", torch.sign(s))
            self.register_buffer("lower_mask", torch.tril(torch.ones(C, C), -1))
            self.L = nn.Parameter(L)
            self.U = nn.Parameter(torch.triu(U, 1))
            self.log_s = nn.Parameter(s.abs().log())

    def weight(self):
        if not self.lu:
            return self.W, torch.linalg.slogdet(self.W)[1]
        C = self.L.shape[0]
        L = self.L * self.lower_mask + torch.eye(C)
        U = self.U * self.lower_mask.T + torch.diag(self.sign_s * self.log_s.exp())
        return self.P @ L @ U, self.log_s.sum()

    def forward(self, x):
        W, logdet = self.weight()
        return F.conv2d(x, W[:, :, None, None]), x.shape[2] * x.shape[3] * logdet

    def reverse(self, y):
        W, _ = self.weight()
        return F.conv2d(y, torch.inverse(W)[:, :, None, None])


class Permute(nn.Module):
    """RealNVP's channel reversal ('reverse') or a fixed random permutation ('shuffle'): log|det| = 0."""

    def __init__(self, C, mode="reverse"):
        super().__init__()
        perm = torch.arange(C - 1, -1, -1) if mode == "reverse" else torch.randperm(C)
        self.register_buffer("perm", perm)
        self.register_buffer("inv", torch.argsort(perm))

    def forward(self, x):
        return x[:, self.perm], torch.zeros(())

    def reverse(self, y):
        return y[:, self.inv]


class ZeroConv(nn.Module):
    """The last convolution of NN(): zero-initialised (so the coupling starts as the identity), with a learned
    per-channel output scale exp(3 * logs) as in the released code."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.logs = nn.Parameter(torch.zeros(1, cout, 1, 1))
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x) * torch.exp(3 * self.logs)


class Coupling(nn.Module):
    """Affine (or additive) coupling. NN = conv3x3 -> ReLU -> conv1x1 -> ReLU -> ZeroConv3x3 (Section 5).
    Scale: s = sigmoid(raw + 2), the released code's stable choice (Table 1 writes s = exp(log s));
    this keeps 0 < s < 1 and starts near s = 0.88."""

    def __init__(self, C, hidden=512, additive=False):
        super().__init__()
        self.additive = additive
        self.ca = C // 2
        cb = C - self.ca
        self.nn = nn.Sequential(nn.Conv2d(cb, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
                                ZeroConv(hidden, self.ca * (1 if additive else 2)))

    def _st(self, xb):
        h = self.nn(xb)
        if self.additive:
            return torch.ones_like(h), h
        raw, t = h[:, 0::2], h[:, 1::2]
        return torch.sigmoid(raw + 2.0), t

    def forward(self, x):
        xa, xb = x[:, :self.ca], x[:, self.ca:]
        s, t = self._st(xb)
        ya = s * xa + t
        return torch.cat([ya, xb], 1), s.log().flatten(1).sum(1)

    def reverse(self, y):
        ya, yb = y[:, :self.ca], y[:, self.ca:]
        s, t = self._st(yb)
        return torch.cat([(ya - t) / s, yb], 1)


class FlowStep(nn.Module):
    """actnorm -> permutation ('conv' = invertible 1x1, 'lu', 'reverse', 'shuffle') -> coupling."""

    def __init__(self, C, hidden=512, perm="conv", additive=False):
        super().__init__()
        self.actnorm = ActNorm(C)
        self.perm = InvConv1x1(C, lu=(perm == "lu")) if perm in ("conv", "lu") else Permute(C, perm)
        self.coupling = Coupling(C, hidden, additive)

    def forward(self, x):
        total = torch.zeros(x.shape[0])
        for layer in (self.actnorm, self.perm, self.coupling):
            x, ld = layer(x)
            total = total + ld
        return x, total

    def reverse(self, y):
        for layer in (self.coupling, self.perm, self.actnorm):
            y = layer.reverse(y)
        return y


def gaussian_log_prob(z, mean, log_std):
    return (-0.5 * (LOG2PI + 2 * log_std + ((z - mean) * torch.exp(-log_std)) ** 2)).flatten(1).sum(1)


class SplitPrior(nn.Module):
    """Split: half the channels leave as z_i with a learned N(mean, std) computed from the other half."""

    def __init__(self, C):
        super().__init__()
        self.conv = ZeroConv(C - C // 2, 2 * (C // 2))

    def params(self, h):
        p = self.conv(h)
        return p[:, 0::2], p[:, 1::2]


class Glow(nn.Module):
    """Multi-scale Glow (Figure 2b). Input (B, C, H, W) with H, W divisible by 2^L.
    perm: 'conv' | 'lu' | 'reverse' | 'shuffle'; additive: additive coupling (s = 1)."""

    def __init__(self, C=3, K=32, L=3, hidden=512, perm="conv", additive=False, squeeze=True):
        super().__init__()
        self.L, self.squeeze = L, squeeze
        self.levels, self.priors = nn.ModuleList(), nn.ModuleList()
        c = C
        for i in range(L):
            c = c * 4 if squeeze else c
            self.levels.append(nn.ModuleList([FlowStep(c, hidden, perm, additive) for _ in range(K)]))
            if i < L - 1:
                self.priors.append(SplitPrior(c))
                c = c // 2
        self.top_channels = c
        self.top = None                                                            # top prior N(0, I)

    def encode(self, x):
        """x -> (list of z per level, total log-determinant, total log p(z))."""
        logdet, logpz, zs = torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), []
        h = x
        for i, steps in enumerate(self.levels):
            if self.squeeze:
                h = squeeze2d(h)
            for step in steps:
                h, ld = step(h)
                logdet = logdet + ld
            if i < self.L - 1:
                c = h.shape[1] // 2
                z, h = h[:, :c], h[:, c:]
                mean, log_std = self.priors[i].params(h)
                logpz = logpz + gaussian_log_prob(z, mean, log_std)
                zs.append(z)
        logpz = logpz + gaussian_log_prob(h, torch.zeros_like(h), torch.zeros_like(h))
        zs.append(h)
        return zs, logdet, logpz

    def log_prob(self, x):
        """log p(x) = log p(z) + log|det dz/dx| (Eq. 6), per example."""
        _, logdet, logpz = self.encode(x)
        return logpz + logdet

    def decode(self, zs=None, n=None, shape=None, temperature=1.0):
        """Reverse pass. Pass zs (as from encode) to reconstruct, or (n, shape) to sample at temperature T:
        each Gaussian's standard deviation is multiplied by T (Section 6)."""
        h = zs[-1] if zs is not None else torch.randn(n, *shape) * temperature
        for i in reversed(range(self.L)):
            if i < self.L - 1:
                mean, log_std = self.priors[i].params(h)
                z = zs[i] if zs is not None else mean + torch.exp(log_std) * torch.randn_like(mean) * temperature
                h = torch.cat([z, h], 1)
            for step in reversed(self.levels[i]):
                h = step.reverse(h)
            if self.squeeze:
                h = unsqueeze2d(h)
        return h

    def top_shape(self, C, H, W):
        f = 2 ** self.L if self.squeeze else 1
        return (self.top_channels, H // f, W // f)


def dequantize(x_uint8, n_bits=8):
    """Eq. 2: x~ = x / 2^bits - 1/2 + U(0, 1/2^bits); returns x~ and the constant c = M log(2^bits) per image."""
    n_bins = 2 ** n_bits
    x = torch.floor(x_uint8.float() / 2 ** (8 - n_bits))                          # reduce bit depth (e.g. 5-bit)
    x = x / n_bins - 0.5 + torch.rand_like(x) / n_bins
    return x, x[0].numel() * math.log(n_bins)


def bits_per_dim(log_p, c, n_dims):
    """(-log p(x~) + c) / (M ln 2)."""
    return (-log_p + c) / (n_dims * math.log(2))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
