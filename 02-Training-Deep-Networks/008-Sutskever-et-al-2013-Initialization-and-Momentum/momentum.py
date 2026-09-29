"""Sutskever, Martens, Dahl & Hinton (2013): initialization and momentum.

  Eqs. (1)-(2)   classical momentum (CM):   v <- mu v - eps grad f(theta);        theta <- theta + v
  Eqs. (3)-(4)   Nesterov (NAG):           v <- mu v - eps grad f(theta + mu v); theta <- theta + v
  Theorem 2.1    on a quadratic, NAG = CM with momentum mu (1 - lambda eps) along each
                 eigendirection with curvature lambda
  Eq. (5)        the momentum schedule mu_t = min(1 - 2^(-1 - log2(floor(t/250) + 1)), mu_max)
  Section 3.1    sparse initialization (SI): each unit gets 15 random incoming weights ~ N(0,1)
  Section 4.1    echo-state-style RNN initialization: spectral radius 1.1, small input weights
  Section 4      the long-range "addition" problem of Hochreiter & Schmidhuber (1997)

The optimizers are written out by hand; PyTorch is only used for gradients.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# The two momentum methods, Eqs. (1)-(4)
# ---------------------------------------------------------------------------

class Momentum:
    """Classical momentum (nesterov=False) or Nesterov's accelerated gradient (True).

    step(loss_fn) evaluates the gradient where the method wants it:
      CM:  at theta           (Eq. 1)
      NAG: at theta + mu v    (Eq. 3): "look ahead" along the velocity first,
           then correct the velocity using the gradient found there.
    """

    def __init__(self, params, lr, mu=0.9, nesterov=True):
        self.params = [p for p in params]
        self.lr, self.mu, self.nesterov = lr, mu, nesterov
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self, loss_fn):
        with torch.no_grad():
            if self.nesterov:                              # move to theta + mu v
                for p, v in zip(self.params, self.v):
                    p.add_(self.mu * v)
        for p in self.params:
            p.grad = None
        loss = loss_fn()
        loss.backward()
        with torch.no_grad():
            for p, v in zip(self.params, self.v):
                if self.nesterov:
                    p.sub_(self.mu * v)                    # back to theta
                v.mul_(self.mu).sub_(self.lr * p.grad)     # Eq. (1) / (3)
                p.add_(v)                                  # Eq. (2) / (4)
        return float(loss)


def mu_schedule(t, mu_max, period=250):
    """Eq. (5): mu_t = min(1 - 2^(-1 - log2(floor(t/250) + 1)), mu_max).
    Starts at 0.5 and rises: 0.75 after 250 updates, ~0.83 after 500, 0.875 after 750,
    ... i.e. 1 - 1/(2 (floor(t/250) + 1)), capped at mu_max.
    `period` (250 in the paper, for 750,000 updates) can be shrunk for shorter runs
    so that mu still reaches mu_max = 0.99 / 0.995 before training ends."""
    return min(1 - 2 ** (-1 - math.log2(t // period + 1)), mu_max)


# ---------------------------------------------------------------------------
# Theorem 2.1 on a quadratic, in closed form (no autograd)
# ---------------------------------------------------------------------------

def quadratic_run(A, b, x0, lr, mu, steps, nesterov):
    """Minimize q(x) = x^T A x / 2 + b^T x with CM or NAG. Returns the path of x."""
    x, v = np.array(x0, float), np.zeros(len(x0))
    path = [x.copy()]
    for _ in range(steps):
        look = x + mu * v if nesterov else x
        v = mu * v - lr * (A @ look + b)
        x = x + v
        path.append(x.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def sparse_init_(W, n_nonzero=15, scale=1.0, gen=None):
    """Section 3.1 (Martens 2010): each unit (column of W, shape fan_in x fan_out)
    gets exactly n_nonzero incoming weights drawn from N(0, 1) * scale; all others 0.
    The total input to a unit then doesn't grow with the size of the layer below."""
    fan_in, fan_out = W.shape
    with torch.no_grad():
        W.zero_()
        for j in range(fan_out):
            idx = torch.randperm(fan_in, generator=gen)[:min(n_nonzero, fan_in)]
            W[idx, j] = torch.randn(len(idx), generator=gen) * scale
    return W


def esn_init(n_hidden, n_in, n_out, radius=1.1, in_scale=0.001, out_scale=0.1, fan_in=15, gen=None):
    """Table 4: hidden-to-hidden weights sparse (15 per unit) and rescaled so the
    spectral radius (largest |eigenvalue|) is exactly `radius` (1.1); input weights
    dense and tiny (N(0,1) * 0.001 for the tasks with distractor inputs); output
    weights N(0,1) * 0.1; hidden biases 0."""
    W_hh = sparse_init_(torch.empty(n_hidden, n_hidden), fan_in, 1.0, gen)
    rho = torch.linalg.eigvals(W_hh).abs().max()
    W_hh *= radius / rho
    W_xh = torch.randn(n_in, n_hidden, generator=gen) * in_scale
    W_hy = torch.randn(n_hidden, n_out, generator=gen) * out_scale
    return W_hh, W_xh, W_hy


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Autoencoder(torch.nn.Module):
    """The deep autoencoder of Hinton & Salakhutdinov (2006) used in Section 3:
    784-1000-500-250-30-250-500-1000-784, logistic units, linear 30-unit code layer,
    sparse initialization. (Martens 2010 and this paper use sigmoid outputs with
    cross-entropy for MNIST; the paper reports squared error.)"""

    SIZES = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]

    def __init__(self, sizes=None, si_scale=1.0, seed=0):
        super().__init__()
        sizes = sizes or self.SIZES
        gen = torch.Generator().manual_seed(seed)
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for a, c in zip(sizes[:-1], sizes[1:]):
            self.W.append(torch.nn.Parameter(sparse_init_(torch.empty(a, c), 15, si_scale, gen)))
            self.b.append(torch.nn.Parameter(torch.zeros(c)))
        self.code = len(sizes) // 2 - 1                 # index of the layer producing the code

    def forward(self, x):
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i != self.code and i != len(self.W) - 1:
                h = torch.sigmoid(h)
        return h                                         # logits of the reconstruction


def autoencoder_losses(model, x):
    """Returns (cross-entropy used for training, squared error reported in Table 1:
    per-example sum of squared reconstruction errors, averaged over the batch)."""
    logits = model(x)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
    se = ((torch.sigmoid(logits) - x) ** 2).sum() / len(x)
    return ce, se


class RNN(torch.nn.Module):
    """100 tanh units (Section 4), initialized as in Table 4."""

    def __init__(self, n_in, n_hidden=100, n_out=1, radius=1.1, in_scale=0.1, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        W_hh, W_xh, W_hy = esn_init(n_hidden, n_in, n_out, radius, in_scale, 0.1, 15, gen)
        self.W_hh, self.W_xh, self.W_hy = (torch.nn.Parameter(w) for w in (W_hh, W_xh, W_hy))
        self.b_h = torch.nn.Parameter(torch.zeros(n_hidden))
        self.b_y = torch.nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        """x: (batch, time, n_in). Returns the output at the LAST time step."""
        h = torch.zeros(x.shape[0], self.W_hh.shape[0])
        for t in range(x.shape[1]):
            h = torch.tanh(h @ self.W_hh + x[:, t] @ self.W_xh + self.b_h)
        return h @ self.W_hy + self.b_y


def addition_problem(batch, T, gen=None):
    """Hochreiter & Schmidhuber's addition problem: each step has a random value in
    [0, 1] and a marker; exactly two steps are marked (one in the first 10% of the
    sequence, one in the first half). Target: the SUM of the two marked values.
    Inputs are centred (Section 4.1: 'centering ... of both the inputs and the
    outputs [is] important'). An answer counts as wrong if its error exceeds 0.04."""
    vals = torch.rand(batch, T, generator=gen)
    marks = torch.zeros(batch, T)
    i1 = torch.randint(0, max(1, T // 10), (batch,), generator=gen)
    i2 = torch.randint(T // 10, T // 2, (batch,), generator=gen)
    marks[torch.arange(batch), i1] = 1
    marks[torch.arange(batch), i2] = 1
    target = vals[torch.arange(batch), i1] + vals[torch.arange(batch), i2]
    x = torch.stack([vals - 0.5, marks - 2 / T], dim=2)     # centred inputs
    return x, (target - 1.0)[:, None]                        # centred target (mean 1)
