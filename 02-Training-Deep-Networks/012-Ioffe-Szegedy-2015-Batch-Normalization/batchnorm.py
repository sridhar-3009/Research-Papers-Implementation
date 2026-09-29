"""Batch Normalization - Ioffe & Szegedy (2015) - written from scratch.

NumPy part (everything by hand, including the backward pass):
  bn_forward / bn_backward      Algorithm 1 and the chain-rule equations of Section 3
  population_stats              Algorithm 2, lines 10: E[x] and the unbiased Var[x] from many mini-batches
  fuse_for_inference            Algorithm 2, line 11: BN at test time is one linear map  y = a x + c
  bn_conv_forward / backward    Section 3.2: one (gamma, beta) per feature map, statistics over N, H, W
  normalize_outside_gradient    Section 2: why the normalization must be INSIDE the gradient step

PyTorch part (for the experiments; autograd computes gradients through our forward):
  BatchNorm                     a hand-written BN layer with training / inference modes
  MLP                           784-100-100-100-10 (Section 4.1), optionally with BN before each nonlinearity
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# NumPy: Algorithm 1 and its backward pass
# ---------------------------------------------------------------------------

def bn_forward(x, gamma, beta, eps=1e-5):
    """x: (m, d) mini-batch, one row per example. Every column (feature) is normalized separately.

        mu    = (1/m) sum_i x_i                 mini-batch mean
        var   = (1/m) sum_i (x_i - mu)^2        mini-batch variance (biased, divides by m)
        x_hat = (x_i - mu) / sqrt(var + eps)    normalize
        y     = gamma x_hat + beta              scale and shift
    """
    mu = x.mean(axis=0)
    var = ((x - mu) ** 2).mean(axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    cache = (x, mu, var, x_hat, gamma, eps)
    return y, cache


def bn_backward(dy, cache):
    """The chain rule written out in Section 3, one line per equation.
    dy = dl/dy, shape (m, d). Returns dl/dx, dl/dgamma, dl/dbeta."""
    x, mu, var, x_hat, gamma, eps = cache
    m = x.shape[0]
    std_inv = 1.0 / np.sqrt(var + eps)

    dx_hat = dy * gamma                                                          # dl/dx_hat
    dvar = (dx_hat * (x - mu)).sum(axis=0) * -0.5 * std_inv ** 3                 # dl/dsigma^2
    dmu = (dx_hat * -std_inv).sum(axis=0) + dvar * (-2 * (x - mu)).sum(axis=0) / m   # dl/dmu
    dx = dx_hat * std_inv + dvar * 2 * (x - mu) / m + dmu / m                    # dl/dx_i
    dgamma = (dy * x_hat).sum(axis=0)                                            # dl/dgamma
    dbeta = dy.sum(axis=0)                                                       # dl/dbeta
    return dx, dgamma, dbeta


def bn_backward_compact(dy, cache):
    """The same gradient, simplified. It shows what BN's backward pass does:
        dx = (gamma / sigma) * (dy - mean(dy) - x_hat * mean(dy * x_hat))
    i.e. it REMOVES from dy its mean and its component along x_hat. So the gradient
    reaching x can't change the batch mean or the batch scale: BN undoes those anyway."""
    x, mu, var, x_hat, gamma, eps = cache
    dx = gamma / np.sqrt(var + eps) * (dy - dy.mean(axis=0) - x_hat * (dy * x_hat).mean(axis=0))
    return dx, (dy * x_hat).sum(axis=0), dy.sum(axis=0)


# ---------------------------------------------------------------------------
# NumPy: Algorithm 2 (inference)
# ---------------------------------------------------------------------------

def population_stats(batches):
    """Algorithm 2, line 10: average the mini-batch statistics over many batches of size m.
        E[x]   = E_B[mu_B]
        Var[x] = m/(m-1) E_B[sigma_B^2]     (unbiased: sigma_B^2 divides by m, not m-1)"""
    mus = [b.mean(axis=0) for b in batches]
    vars_ = [b.var(axis=0) for b in batches]
    m = batches[0].shape[0]
    return np.mean(mus, axis=0), m / (m - 1) * np.mean(vars_, axis=0)


def fuse_for_inference(gamma, beta, mean, var, eps=1e-5):
    """Algorithm 2, line 11: at test time
        y = gamma (x - E[x]) / sqrt(Var[x] + eps) + beta = a x + c
    with a = gamma / sqrt(Var[x] + eps), c = beta - a E[x]. It can be folded into the previous layer's W."""
    a = gamma / np.sqrt(var + eps)
    return a, beta - a * mean


# ---------------------------------------------------------------------------
# NumPy: convolutional BN (Section 3.2)
# ---------------------------------------------------------------------------

def bn_conv_forward(x, gamma, beta, eps=1e-5):
    """x: (N, C, H, W). Each feature map (channel) is normalized using ALL its values in the batch:
    the effective mini-batch size is m' = N * H * W. gamma, beta have shape (C,)."""
    N, C, H, W = x.shape
    flat = x.transpose(0, 2, 3, 1).reshape(-1, C)                  # (N*H*W, C): every location is an "example"
    y, cache = bn_forward(flat, gamma, beta, eps)
    return y.reshape(N, H, W, C).transpose(0, 3, 1, 2), (cache, x.shape)


def bn_conv_backward(dy, cache):
    cache, (N, C, H, W) = cache
    dx, dgamma, dbeta = bn_backward(dy.transpose(0, 2, 3, 1).reshape(-1, C), cache)
    return dx.reshape(N, H, W, C).transpose(0, 3, 1, 2), dgamma, dbeta


# ---------------------------------------------------------------------------
# Section 2: normalizing OUTSIDE the gradient step blows up
# ---------------------------------------------------------------------------

def normalize_outside_gradient(u, target, lr=0.1, steps=100, inside=False):
    """The paper's example: x = u + b, x_hat = x - E[x], loss = mean((x_hat - target)^2).

    inside=False: the gradient step treats E[x] as a constant, so dl/db = mean(dl/dx_hat).
                  But b cancels in x_hat, so the loss never changes while b drifts forever.
    inside=True : the gradient accounts for E[x] = E[u] + b, so dl/db = 0 exactly: b stays put.
    Returns the history of b and of the loss."""
    b, bs, losses = 0.0, [], []
    for _ in range(steps):
        x = u + b
        x_hat = x - x.mean()
        losses.append(np.mean((x_hat - target) ** 2))
        dl_dxhat = 2 * (x_hat - target) / len(u)
        grad_b = dl_dxhat.sum() * (0.0 if inside else 1.0)   # d x_hat_i / d b = 1 - 1 = 0 when E[x] is included
        b -= lr * grad_b
        bs.append(b)
    return np.array(bs), np.array(losses)


# ---------------------------------------------------------------------------
# PyTorch: a hand-written BN layer and the Section 4.1 network
# ---------------------------------------------------------------------------

class BatchNorm(nn.Module):
    """y = gamma x_hat + beta, per feature (2-D input) or per channel (4-D input).

    training mode: normalize with the mini-batch statistics (Algorithm 1). Also keep running
                   averages of mu_B and sigma_B^2 so the test accuracy can be tracked during training
                   (the paper: "Using moving averages instead, we can track the accuracy").
    eval mode:     normalize with the running (population) statistics (Algorithm 2)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps, self.momentum = eps, momentum
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        dims = (0,) if x.dim() == 2 else (0, 2, 3)                  # conv: statistics over N, H, W
        shape = (1, -1) if x.dim() == 2 else (1, -1, 1, 1)
        if self.training:
            mu = x.mean(dims)
            var = x.var(dims, unbiased=False)
            m = x.numel() // x.shape[1]
            with torch.no_grad():
                self.running_mean.lerp_(mu, self.momentum)
                self.running_var.lerp_(var * m / (m - 1), self.momentum)   # unbiased, as in Algorithm 2
        else:
            mu, var = self.running_mean, self.running_var
        x_hat = (x - mu.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return self.gamma.view(shape) * x_hat + self.beta.view(shape)


class MLP(nn.Module):
    """Section 4.1: 784 -> 100 -> 100 -> 100 -> 10, sigmoid, small random Gaussian weights.
    bn=True computes z = g(BN(W u)): BN before each nonlinearity, and no bias
    (BN's beta replaces it, Section 3.2)."""

    def __init__(self, sizes=(784, 100, 100, 100, 10), act="sigmoid", bn=False, init_std=0.1):
        super().__init__()
        self.act = {"sigmoid": torch.sigmoid, "relu": torch.relu, "tanh": torch.tanh}[act]
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            hidden = i < len(sizes) - 2
            lin = nn.Linear(a, b, bias=not (bn and hidden))
            nn.init.normal_(lin.weight, 0.0, init_std)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
            self.linears.append(lin)
            if bn and hidden:
                self.bns.append(BatchNorm(b))
        self.bn = bn

    def forward(self, x, return_preact=False):
        """return_preact=True also returns the input of every sigmoid (Figure 1(b, c))."""
        preacts = []
        for i, lin in enumerate(self.linears[:-1]):
            z = lin(x)
            if self.bn:
                z = self.bns[i](z)
            preacts.append(z)
            x = self.act(z)
        out = self.linears[-1](x)
        return (out, preacts) if return_preact else out
