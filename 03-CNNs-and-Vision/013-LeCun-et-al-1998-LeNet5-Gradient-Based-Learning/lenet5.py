"""LeNet-5 - LeCun, Bottou, Bengio & Haffner (1998), Sections II-III and Appendices A, C.

Built exactly as described in the paper, layer by layer:

  INPUT 32x32 (background -0.1, ink 1.175)
  C1  conv 5x5, 6 maps 28x28            156 params    122,304 connections
  S2  subsample 2x2, 6 maps 14x14        12 params      5,880 connections
  C3  conv 5x5, 16 maps 10x10         1,516 params    151,600 connections  (sparse: Table I)
  S4  subsample 2x2, 16 maps 5x5         32 params      2,000 connections
  C5  conv 5x5, 120 maps 1x1         48,120 params     48,120 connections
  F6  full, 84 units                 10,164 params     10,164 connections
  OUTPUT  10 Euclidean RBF units, fixed +-1 prototypes = 7x12 bitmaps of the digits
                                     -------
                                      60,000 trainable parameters  ("only 60,000", Section II.A)

Every layer up to F6 uses f(a) = 1.7159 tanh(2a/3) (Eq. 6, Appendix A).
Loss: MSE (Eq. 8) or the discriminative MAP criterion (Eq. 9).
Training: stochastic diagonal Levenberg-Marquardt (Appendix C), see `sdlm_step`.

There is also a slow NumPy convolution (`conv2d_naive`) that shows what a convolution IS.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

A, S = 1.7159, 2.0 / 3.0


def squash(a):
    """Eq. 6: f(a) = A tanh(S a), A = 1.7159, S = 2/3. Chosen so f(1) = 1 and f(-1) = -1 (Appendix A)."""
    return A * torch.tanh(S * a)


# ---------------------------------------------------------------------------
# What a convolution is: loops, no library
# ---------------------------------------------------------------------------

def conv2d_naive(x, w, b):
    """x: (C_in, H, W) image, w: (C_out, C_in, k, k) kernels, b: (C_out,) biases.
    Output (C_out, H-k+1, W-k+1). Every output unit looks at a k x k window ("local receptive
    field") and ALL units of one output map use the SAME kernel ("shared weights")."""
    C_out, C_in, k, _ = w.shape
    H, W = x.shape[1] - k + 1, x.shape[2] - k + 1
    out = np.zeros((C_out, H, W))
    for o in range(C_out):                   # one feature map at a time
        for i in range(H):
            for j in range(W):               # slide the same kernel over every location
                out[o, i, j] = np.sum(x[:, i:i + k, j:j + k] * w[o]) + b[o]
    return out


# ---------------------------------------------------------------------------
# Table I: which S2 maps each C3 map reads
# ---------------------------------------------------------------------------

C3_TABLE = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 0), (5, 0, 1),                    # 6 maps: 3 contiguous
    (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 0), (4, 5, 0, 1), (5, 0, 1, 2),  # 6 maps: 4 contiguous
    (0, 1, 3, 4), (1, 2, 4, 5), (0, 2, 3, 5),                                           # 3 maps: 4 non-contiguous
    (0, 1, 2, 3, 4, 5),                                                                 # 1 map: all six
]


def c3_mask():
    """(16, 6) matrix: mask[o, i] = 1 if C3 map o is connected to S2 map i."""
    m = torch.zeros(16, 6)
    for o, ins in enumerate(C3_TABLE):
        m[o, list(ins)] = 1
    return m


# ---------------------------------------------------------------------------
# Output codes: 7x12 bitmaps of the digits, +1 = ink, -1 = background (Section II.B, Fig. 3)
# ---------------------------------------------------------------------------

_BITMAPS = {
    0: [".......", "..###..", ".#...#.", "#.....#", "#.....#", "#.....#",
        "#.....#", "#.....#", "#.....#", ".#...#.", "..###..", "......."],
    1: [".......", "...#...", "..##...", ".#.#...", "...#...", "...#...",
        "...#...", "...#...", "...#...", "...#...", ".#####.", "......."],
    2: [".......", "..###..", ".#...#.", "#.....#", "......#", ".....#.",
        "....#..", "...#...", "..#....", ".#.....", "#######", "......."],
    3: [".......", ".#####.", "......#", "......#", ".....#.", "..###..",
        ".....#.", "......#", "......#", "......#", ".#####.", "......."],
    4: [".......", ".....#.", "....##.", "...#.#.", "..#..#.", ".#...#.",
        "#######", ".....#.", ".....#.", ".....#.", ".....#.", "......."],
    5: [".......", "#######", "#......", "#......", "#####..", ".....#.",
        "......#", "......#", "......#", ".....#.", "#####..", "......."],
    6: [".......", "...###.", "..#....", ".#.....", "#......", "#.###..",
        "##...#.", "#.....#", "#.....#", ".#...#.", "..###..", "......."],
    7: [".......", "#######", "......#", ".....#.", ".....#.", "....#..",
        "....#..", "...#...", "...#...", "..#....", "..#....", "......."],
    8: [".......", "..###..", ".#...#.", ".#...#.", ".#...#.", "..###..",
        ".#...#.", "#.....#", "#.....#", ".#...#.", "..###..", "......."],
    9: [".......", "..###..", ".#...#.", "#.....#", "#.....#", ".#...##",
        "..###.#", "......#", ".....#.", "....#..", ".###...", "......."],
}


def rbf_prototypes():
    """(10, 84) tensor of +-1: each row is a digit drawn on a 7-wide, 12-tall bitmap."""
    rows = []
    for d in range(10):
        bm = _BITMAPS[d]
        assert len(bm) == 12 and all(len(r) == 7 for r in bm)
        rows.append([1.0 if c == "#" else -1.0 for r in bm for c in r])
    return torch.tensor(rows)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def _uniform_fan_in(t, fan_in):
    """Appendix A: weights uniform in [-2.4/F, 2.4/F], F = the unit's number of inputs."""
    with torch.no_grad():
        t.uniform_(-2.4 / fan_in, 2.4 / fan_in)


class Subsample(nn.Module):
    """S2 / S4: for each map, add the 4 inputs of a 2x2 block (non-overlapping),
    multiply by ONE trainable coefficient, add ONE trainable bias, squash.
    So each map has only 2 parameters."""

    def __init__(self, maps):
        super().__init__()
        self.coef = nn.Parameter(torch.empty(maps))
        self.bias = nn.Parameter(torch.empty(maps))
        _uniform_fan_in(self.coef, 4); _uniform_fan_in(self.bias, 4)

    def forward(self, x):
        s = F.avg_pool2d(x, 2) * 4                                  # sum of each 2x2 block
        return squash(self.coef.view(1, -1, 1, 1) * s + self.bias.view(1, -1, 1, 1))


class SparseConv(nn.Module):
    """C3: a 5x5 convolution from 6 maps to 16 maps, but map o only sees the S2 maps in
    C3_TABLE[o]. Done with a 0/1 mask on the kernel: masked weights are always multiplied
    by 0, so they get zero gradient and never matter."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(16, 6, 5, 5))
        self.bias = nn.Parameter(torch.empty(16))
        self.register_buffer("mask", c3_mask().view(16, 6, 1, 1))
        for o, ins in enumerate(C3_TABLE):                          # fan-in differs per map: 75, 100 or 150
            _uniform_fan_in(self.weight[o], 25 * len(ins)); _uniform_fan_in(self.bias[o:o + 1], 25 * len(ins))

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias)

    def n_params(self):
        return int(self.mask.sum().item()) * 25 + 16


class LeNet5(nn.Module):
    def __init__(self, learn_prototypes=False):
        super().__init__()
        self.C1 = nn.Conv2d(1, 6, 5)
        self.S2 = Subsample(6)
        self.C3 = SparseConv()
        self.S4 = Subsample(16)
        self.C5 = nn.Conv2d(16, 120, 5)
        self.F6 = nn.Linear(120, 84)
        for layer, fan_in in ((self.C1, 25), (self.C5, 400), (self.F6, 120)):
            _uniform_fan_in(layer.weight, fan_in); _uniform_fan_in(layer.bias, fan_in)
        # RBF centers: fixed by default ("chosen by hand and kept fixed")
        self.prototypes = nn.Parameter(rbf_prototypes(), requires_grad=learn_prototypes)

    def features(self, x):
        """Everything up to F6's state: (N, 84)."""
        x = squash(self.C1(x))          # (N, 6, 28, 28)
        x = self.S2(x)                  # (N, 6, 14, 14)
        x = squash(self.C3(x))          # (N, 16, 10, 10)
        x = self.S4(x)                  # (N, 16, 5, 5)
        x = squash(self.C5(x))          # (N, 120, 1, 1)
        return squash(self.F6(x.flatten(1)))   # (N, 84)

    def forward(self, x):
        """Eq. 7: y_i = sum_j (x_j - w_ij)^2, a PENALTY per class. Smallest = predicted class."""
        h = self.features(x)
        return ((h[:, None, :] - self.prototypes[None]) ** 2).sum(-1)     # (N, 10)

    def predict(self, x):
        return self(x).argmin(1)

    def n_trainable(self):
        """Count trainable parameters the way the paper does (C3's masked-out weights don't exist)."""
        n = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            n += self.C3.n_params() if name == "C3.weight" else (0 if name == "C3.bias" else p.numel())
        return n


# ---------------------------------------------------------------------------
# Loss functions (Section II.C)
# ---------------------------------------------------------------------------

def mse_loss(y, labels):
    """Eq. 8: E = mean over examples of y_{D_p}, the penalty of the correct class."""
    return y.gather(1, labels[:, None]).mean()


def map_loss(y, labels, j=1.0):
    """Eq. 9: E = mean[ y_D + log(e^{-j} + sum_i e^{-y_i}) ].
    The second term PULLS UP the penalties of the wrong classes (competition), and e^{-j}
    is a "rubbish class" that stops already-large penalties being pushed up further."""
    rubbish = torch.full_like(y[:, :1], -j)
    return (y.gather(1, labels[:, None]).squeeze(1) + torch.logsumexp(torch.cat([rubbish, -y], 1), 1)).mean()


# ---------------------------------------------------------------------------
# Data: the paper's input format and distortions
# ---------------------------------------------------------------------------

def prepare(images_uint8):
    """(N, 28, 28) uint8 MNIST -> (N, 1, 32, 32) float: pad to 32x32 so stroke end-points
    can reach the center of the top feature detectors; background -0.1, ink 1.175
    (mean ~0, variance ~1)."""
    x = images_uint8.float() / 255.0
    x = F.pad(x, (2, 2, 2, 2))
    return (-0.1 + 1.275 * x)[:, None]


def distort(x, max_shift=2.0, max_scale=0.15, max_squeeze=0.15, max_shear=0.3, generator=None):
    """Section III.B: random planar affine distortions: horizontal and vertical translation,
    scaling, squeezing (compress one axis while stretching the other) and horizontal shear.
    x: (N, 1, 32, 32) in the paper's input format."""
    N = x.shape[0]
    r = lambda: (torch.rand(N, generator=generator) * 2 - 1)
    scale = 1 + max_scale * r()
    squeeze = 1 + max_squeeze * r()
    shear = max_shear * r()
    tx, ty = max_shift * r() * 2 / 32, max_shift * r() * 2 / 32      # in affine_grid's [-1, 1] units
    theta = torch.zeros(N, 2, 3)
    theta[:, 0, 0] = 1 / (scale * squeeze)
    theta[:, 0, 1] = shear
    theta[:, 1, 1] = squeeze / scale
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    theta = theta.to(x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # sample in "ink" units so that outside pixels become background (-0.1)
    ink = (x + 0.1) / 1.275
    out = F.grid_sample(ink, grid, align_corners=False, padding_mode="zeros")
    return -0.1 + 1.275 * out


# ---------------------------------------------------------------------------
# Appendix C: stochastic diagonal Levenberg-Marquardt
# ---------------------------------------------------------------------------

def gauss_newton_diag(model, x, probes=1):
    """Estimate h_kk, the diagonal of the Gauss-Newton matrix of the MSE criterion, averaged
    over the examples in x. For one example the MSE loss is ||h - w_D||^2 with h = F6's state,
    whose Hessian in h is 2I. So GN = J^T (2I) J and
        h_kk = E_v[ (J^T v)_k^2 ]   with v ~ N(0, 2I)     (unbiased, one backward pass per probe)
    The paper gets its h_kk by back-propagating diagonal second derivatives (Appendix C);
    this is the same quantity without their extra layer-by-layer approximation."""
    params = [p for p in model.parameters() if p.requires_grad]
    h = [torch.zeros_like(p) for p in params]
    for i in range(len(x)):
        for _ in range(probes):
            out = model.features(x[i:i + 1])
            v = torch.randn_like(out) * np.sqrt(2.0)
            g = torch.autograd.grad((out * v).sum(), params)
            for hk, gk in zip(h, g):
                hk += gk ** 2
    return [hk / (len(x) * probes) for hk in h]


def sdlm_step(model, loss, h_diag, eta, mu=0.02):
    """One update with a separate learning rate per parameter:
        eps_k = eta / (mu + h_kk)          (Eq. 21 of Appendix C)
    Parameters with large curvature take small steps, flat ones take big steps.
    mu keeps the step bounded when h_kk is ~0."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params)
    with torch.no_grad():
        for p, g, hk in zip(params, grads, h_diag):
            p -= eta / (mu + hk) * g


def paper_eta(epoch):
    """The global learning-rate schedule of Section III.B (epoch counted from 0)."""
    for last, eta in ((2, 5e-4), (5, 2e-4), (8, 1e-4), (12, 5e-5)):
        if epoch < last:
            return eta
    return 1e-5
