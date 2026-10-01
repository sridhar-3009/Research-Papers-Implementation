"""Szegedy et al. (2013), "Intriguing properties of neural networks" - the tools of the paper.

  Section 3   top_activating          images that most excite a unit (e_i) or a random direction (v)
  Section 4.1 lbfgs_attack            minimize c*||r||^2 + loss(x + r, l) with x + r in [0, 1]  (L-BFGS-B)
              minimal_adversarial     line search over c: the smallest perturbation still classified as l
              distortion              the paper's measure sqrt(sum (x' - x)^2 / n)
  Section 4.2 gaussian_distort, amplify   the random-noise baseline and the "amplified to stddev 0.1" test
              decay_penalty           the paper's weight decay: lambda * sum(w^2) / k (k = units in the layer)
  Section 4.3 fc_operator_norm, conv_operator_norm_fft, conv_operator_norm_power, lipschitz_upper_bound
              the upper Lipschitz constant of each layer, ||phi(x) - phi(x + r)|| <= L ||r||
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Section 3: units vs random directions
# ---------------------------------------------------------------------------

def top_activating(features, direction, k=8):
    """features: (N, n) activations phi(x) of a held-out set. Returns the indices of the k inputs
    with the largest <phi(x), direction>. direction = e_i (one unit) or a random vector v."""
    scores = features @ direction
    return torch.argsort(scores, descending=True)[:k]


def unit_direction(n, i):
    e = torch.zeros(n)
    e[i] = 1.0
    return e


def random_direction(n, generator=None):
    v = torch.randn(n, generator=generator)
    return v / v.norm()


# ---------------------------------------------------------------------------
# Section 4.1: box-constrained L-BFGS
# ---------------------------------------------------------------------------

def lbfgs_attack(model, x, target, c, max_iter=200):
    """Minimize   c * ||r||^2 + CE(model(x + r), target)   subject to  0 <= x + r <= 1.
    (The paper writes c|r|; we use the squared L2 norm so the objective is smooth for L-BFGS.)
    x: (m,) tensor in [0, 1]. Returns x + r as a tensor."""
    model.eval()
    x0 = x.detach().double().cpu().numpy().ravel()
    shape = x.shape
    t = torch.tensor([target])

    def f_and_grad(z):
        zt = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        loss = c * ((zt - torch.tensor(x0, dtype=torch.float32)) ** 2).sum() + \
            F.cross_entropy(model(zt.view(1, *shape)), t)
        g, = torch.autograd.grad(loss, zt)
        return loss.item(), g.double().numpy()

    res = minimize(f_and_grad, x0, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * x0.size,
                   options={"maxiter": max_iter})
    return torch.tensor(res.x, dtype=torch.float32).view(shape)


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x.unsqueeze(0)).argmax(1).item()


def minimal_adversarial(model, x, target, c_start=1.0, search_steps=8, max_iter=200):
    """'line-search to find the [c] for which the minimizer r of [the penalized problem]
    satisfies f(x + r) = l'. A larger c punishes ||r|| more, so the SMALLEST perturbation
    comes from the LARGEST c that still succeeds (the paper says 'minimum c'; the logic needs the
    largest). We bracket c by factors of 10, then bisect in log-space.
    Returns (x_adv, c) or (None, None) if even a tiny c fails."""
    def attempt(c):
        z = lbfgs_attack(model, x, target, c, max_iter)
        return z, predict(model, z) == target

    c, best = c_start, None
    z, ok = attempt(c)
    if ok:                                              # grow c until it fails
        best = (z, c)
        for _ in range(6):
            z, ok = attempt(c * 10)
            if not ok:
                break
            c *= 10
            best = (z, c)
        lo, hi = c, c * 10
    else:                                               # shrink c until it succeeds
        for _ in range(6):
            c /= 10
            z, ok = attempt(c)
            if ok:
                best = (z, c)
                break
        if best is None:
            return None, None
        lo, hi = c, c * 10
    for _ in range(search_steps):                       # bisect between success (lo) and failure (hi)
        mid = (lo * hi) ** 0.5
        z, ok = attempt(mid)
        if ok:
            lo, best = mid, (z, mid)
        else:
            hi = mid
    return best


def distortion(x, x_adv):
    """Table 1's measure: sqrt( sum_i (x'_i - x_i)^2 / n ), n = number of pixels. This is the
    standard deviation of the perturbation, comparable to Gaussian noise of that stddev."""
    return ((x_adv - x) ** 2).mean().sqrt().item()


# ---------------------------------------------------------------------------
# Section 4.2: baselines
# ---------------------------------------------------------------------------

def gaussian_distort(x, std, generator=None):
    """The control: add Gaussian noise of the given stddev and clip to [0, 1]."""
    return (x + std * torch.randn(x.shape, generator=generator)).clamp(0, 1)


def amplify(x, x_adv, target_std=0.1):
    """Table 4, lower half: make the perturbation bigger, keeping its direction, so its stddev is
    target_std (the paper: 'from stddev 0.06 to 0.1'). Then clip to [0, 1].
    (The paper prints x + 0.1 (x'-x)/||x'-x||_2, which would be MUCH smaller; we follow the text.)"""
    r = x_adv - x
    return (x + r * (target_std / ((r ** 2).mean().sqrt() + 1e-12))).clamp(0, 1)


def decay_penalty(model, lambdas):
    """lambda * sum(w^2) / k for each Linear layer, k = number of units (outputs) in that layer."""
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return sum(lam * (m.weight ** 2).sum() / m.out_features for m, lam in zip(lins, lambdas))


# ---------------------------------------------------------------------------
# Section 4.3: spectral analysis
# ---------------------------------------------------------------------------

def fc_operator_norm(W):
    """||W|| = largest singular value. A ReLU layer max(0, Wx + b) has Lipschitz constant <= ||W||."""
    return torch.linalg.matrix_norm(W.detach().double(), ord=2).item()


def conv_operator_norm_fft(weight, n):
    """Eq. (1) for stride 1 on an n x n image (circular boundary): take the 2-D Fourier transform of
    every kernel w_{c,d}; at each frequency xi this gives a D x C matrix A(xi). The convolution's
    norm is the largest singular value over all frequencies (Parseval's formula)."""
    D, C, k, _ = weight.shape
    padded = torch.zeros(D, C, n, n, dtype=torch.float64)
    padded[:, :, :k, :k] = weight.detach().double()
    A = torch.fft.fft2(padded)                                   # (D, C, n, n), complex
    A = A.permute(2, 3, 0, 1)                                    # (n, n, D, C): one matrix per frequency
    return torch.linalg.svdvals(A).max().item()


def _circular_conv(x, weight, stride):
    k = weight.shape[-1]
    lo, hi = (k - 1) // 2, k // 2
    return F.conv2d(F.pad(x, (lo, hi, lo, hi), mode="circular"), weight, stride=stride)


def conv_operator_norm_power(weight, n, stride=1, iters=300, seed=0):
    """The same norm for ANY stride, by power iteration on A^T A, where A is the circular
    convolution and A^T is obtained with autograd (a vector-Jacobian product)."""
    g = torch.Generator().manual_seed(seed)
    w = weight.detach().double()
    x = torch.randn(1, w.shape[1], n, n, generator=g, dtype=torch.float64)
    sigma = 0.0
    for _ in range(iters):
        x = x / x.norm()
        x.requires_grad_(True)
        y = _circular_conv(x, w, stride)
        ATy, = torch.autograd.grad(y, x, grad_outputs=y)        # A^T A x
        sigma = y.norm().item()                                  # ||A x|| with ||x|| = 1
        x = ATy.detach()
    return sigma


def conv_as_matrix(weight, n, stride=1):
    """Brute force (small sizes only): the full matrix of the circular convolution."""
    C = weight.shape[1]
    cols = []
    for i in range(C * n * n):
        e = torch.zeros(C * n * n, dtype=torch.float64)
        e[i] = 1.0
        cols.append(_circular_conv(e.view(1, C, n, n), weight.double(), stride).flatten())
    return torch.stack(cols, 1)


def lipschitz_upper_bound(norms):
    """L = prod_k L_k: ||phi(x) - phi(x + r)|| <= L ||r||. Max-pooling and ReLU have L_k <= 1."""
    return float(np.prod(norms))


# ---------------------------------------------------------------------------
# The MNIST models of Table 1
# ---------------------------------------------------------------------------

def fc_net(hidden=(), act="sigmoid"):
    """FC10 (no hidden layer, a softmax classifier), FC100-100-10, FC200-200-10, FC123-456-10."""
    layers, n = [nn.Flatten()], 784
    for h in hidden:
        layers += [nn.Linear(n, h), nn.Sigmoid() if act == "sigmoid" else nn.ReLU()]
        n = h
    layers.append(nn.Linear(n, 10))
    return nn.Sequential(*layers)


class AE400(nn.Module):
    """AE400-10: a sparse autoencoder with 400 sigmoid units (trained first, then frozen) and a
    softmax classifier on top."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(784, 400)
        self.dec = nn.Linear(400, 784)
        self.cls = nn.Linear(400, 10)

    def encode(self, x):
        return torch.sigmoid(self.enc(x.flatten(1)))

    def reconstruct(self, x):
        return torch.sigmoid(self.dec(self.encode(x)))

    def forward(self, x):
        return self.cls(self.encode(x))
