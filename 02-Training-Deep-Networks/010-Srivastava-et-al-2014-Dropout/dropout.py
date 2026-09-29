"""Dropout, the full JMLR version (Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, 2014).

  Section 4     the dropout model:  r ~ Bernoulli(p);  y~ = r * y;  z = W y~ + b;  test: W_test = p W
  Section 5.1   max-norm: ||w|| <= c for each hidden unit's incoming weights (projected after each step)
  Section 6.5   other regularizers for comparison: L2, L1, max-norm
  Section 7.2   sparsity of hidden activations
  Section 9.1   dropout in linear regression = ridge regression with Gamma = diag(X^T X)^(1/2)
  Section 10    multiplicative Gaussian noise: r ~ N(1, sigma^2), sigma^2 = (1 - p)/p, no test-time scaling

p is the probability of RETAINING a unit (the paper's convention): p = 0.5 hidden, 0.8 input.
Masks are drawn explicitly - no nn.Dropout - so every step is visible.
"""

import numpy as np
import torch


class Net(torch.nn.Module):
    """ReLU (or logistic) feed-forward net with dropout on the input and every hidden layer.
    noise="bernoulli" (Section 4) or "gaussian" (Section 10)."""

    def __init__(self, sizes, p_input=1.0, p_hidden=1.0, act="relu", noise="bernoulli", init_std=0.01, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.W = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(a, b, generator=gen) * init_std) for a, b in zip(sizes[:-1], sizes[1:])])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(b)) for b in sizes[1:]])
        self.p = [p_input] + [p_hidden] * (len(sizes) - 2)
        self.act = torch.relu if act == "relu" else torch.sigmoid
        self.noise = noise

    def forward(self, x, train=False, return_hidden=False):
        h, hidden = x, []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            p = self.p[i]
            if p < 1:
                if train and self.noise == "bernoulli":
                    h = h * (torch.rand_like(h) < p).float()                   # r ~ Bernoulli(p)
                elif train and self.noise == "gaussian":
                    h = h * (1 + np.sqrt((1 - p) / p) * torch.randn_like(h))   # r ~ N(1, (1-p)/p)
                elif not train and self.noise == "bernoulli":
                    h = h * p                                                  # W_test = p W  (same thing)
                # gaussian: E[r] = 1, so no scaling at test time
            h = h @ W + b
            if i < len(self.W) - 1:
                h = self.act(h)
                hidden.append(h)
        return (h, hidden) if return_hidden else h


@torch.no_grad()
def max_norm_(net, c):
    """Section 5.1: project each hidden unit's incoming weight vector onto the ball ||w|| <= c."""
    for W in net.W[:-1]:
        n = W.norm(dim=0)
        W.mul_(torch.clamp(c / (n + 1e-12), max=1.0))


def train(net, X, y, epochs, lr=0.1, momentum=0.95, batch=100, max_norm=None, l2=0.0, l1=0.0,
          lr_decay=1.0, seed=0, eval_fn=None):
    """SGD with momentum on cross-entropy, with optional max-norm / L2 / L1.
    eval_fn(net) is called after every epoch; its results are returned."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    hist = []
    for epoch in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr * lr_decay ** epoch
        perm = torch.randperm(len(X), generator=gen).to(X.device)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = torch.nn.functional.cross_entropy(net(X[idx], train=True), y[idx])
            if l2:
                loss = loss + l2 * sum((W ** 2).sum() for W in net.W)
            if l1:
                loss = loss + l1 * sum(W.abs().sum() for W in net.W)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if max_norm:
                max_norm_(net, max_norm)
        if eval_fn is not None:
            hist.append(eval_fn(net))
    return hist


@torch.no_grad()
def error_rate(net, X, y):
    return float((net(X).argmax(1) != y).float().mean())


@torch.no_grad()
def monte_carlo_error(net, X, y, k, seed=0):
    """Section 7.5: average the predictions of k sampled dropout networks."""
    torch.manual_seed(seed)
    probs = sum(torch.softmax(net(X, train=True), 1) for _ in range(k)) / k
    return float((probs.argmax(1) != y).float().mean())


# ---------------------------------------------------------------------------
# Section 7.1-7.2: a one-hidden-layer ReLU autoencoder, for features and sparsity
# ---------------------------------------------------------------------------

class Autoencoder(torch.nn.Module):
    def __init__(self, n_hidden=256, p_hidden=1.0, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.W1 = torch.nn.Parameter(torch.randn(784, n_hidden, generator=gen) * 0.01)
        self.b1 = torch.nn.Parameter(torch.zeros(n_hidden))
        self.W2 = torch.nn.Parameter(torch.randn(n_hidden, 784, generator=gen) * 0.01)
        self.b2 = torch.nn.Parameter(torch.zeros(784))
        self.p = p_hidden

    def forward(self, x, train=False):
        h = torch.relu(x @ self.W1 + self.b1)
        if self.p < 1:
            h = h * (torch.rand_like(h) < self.p).float() if train else h * self.p
        return torch.sigmoid(h @ self.W2 + self.b2), h


# ---------------------------------------------------------------------------
# Section 9.1: marginalizing dropout in linear regression
# ---------------------------------------------------------------------------

def dropout_linear_regression_closed_form(X, y, p):
    """minimize E_R ||y - (R * X) w||^2 with R_ij ~ Bernoulli(p).
    The paper: = ||y - p X w||^2 + p (1 - p) ||Gamma w||^2, Gamma = diag(X^T X)^(1/2).
    Setting the gradient to 0:  (p^2 X^T X + p (1 - p) diag(X^T X)) w = p X^T y."""
    G = np.diag(np.diag(X.T @ X))
    return np.linalg.solve(p * p * X.T @ X + p * (1 - p) * G, p * X.T @ y)


def dropout_linear_regression_sgd(X, y, p, steps=200000, lr=1e-3, seed=0):
    """Minimize the SAME objective by actually sampling dropout masks (plain SGD on
    one random row and one random mask at a time)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    for t in range(steps):
        i = rng.integers(len(X))
        xr = X[i] * (rng.random(X.shape[1]) < p)
        w += lr * (y[i] - xr @ w) * xr
    return w
