"""Chapter 4 of Sutskever's thesis (2013) / Martens & Sutskever (2011): training a plain tanh RNN with
Hessian-free optimization and STRUCTURAL DAMPING.

  RNN                 x_t = W_hv v_t + W_hh h_{t-1} + b_h,  h_t = tanh(x_t),  o_t = W_oh h_t + b_o
                      sparse initialization of Section 4.4: each unit gets 15 nonzero incoming weights,
                      variance 1 for W_hv, 1/15 for W_hh, W_oh and the biases
  HFStructural        one HF step solves (G_f + lambda I + lambda mu G_S) d = -grad with conjugate gradient:
                        G_f v = J_o^T H_L J_o v      (Gauss-Newton of the loss through the outputs o)
                        G_S v = J_x^T diag(1 - h^2) J_x v
                      G_S is the Gauss-Newton matrix of a distance between the NEW and OLD hidden-state
                      sequences (Eq. 4.7): it penalizes parameter changes that would make the hidden states
                      change a lot, even if the parameter change itself is small.
                      Both products come from ONE jvp and ONE vjp of a function returning (o, x).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp


class RNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, seed=0, sparse=True):
        super().__init__()
        self.W_hv = nn.Parameter(torch.zeros(n_hidden, n_in))
        self.W_hh = nn.Parameter(torch.zeros(n_hidden, n_hidden))
        self.W_oh = nn.Parameter(torch.zeros(n_out, n_hidden))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.b_o = nn.Parameter(torch.zeros(n_out))
        self.h0 = nn.Parameter(torch.zeros(n_hidden))
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for W, var in ((self.W_hv, 1.0), (self.W_hh, 1 / 15), (self.W_oh, 1 / 15)):
                if sparse:
                    for r in range(W.shape[0]):
                        idx = torch.randperm(W.shape[1], generator=g)[:min(15, W.shape[1])]
                        W[r, idx] = torch.randn(len(idx), generator=g) * var ** 0.5
                else:
                    W.normal_(0, var ** 0.5 / 3, generator=g)
            self.b_h.normal_(0, (1 / 15) ** 0.5, generator=g)
            self.b_o.normal_(0, (1 / 15) ** 0.5, generator=g)

    def forward(self, vs):
        """vs: (T, N, n_in). Returns outputs o (T, N, n_out) and hidden pre-activations x (T, N, H)."""
        h = self.h0.expand(vs.shape[1], -1)
        outs, xs = [], []
        for v in vs:
            x = v @ self.W_hv.T + h @ self.W_hh.T + self.b_h
            h = torch.tanh(x)
            xs.append(x)
            outs.append(h @ self.W_oh.T + self.b_o)
        return torch.stack(outs), torch.stack(xs)


def loss_fn(o, target, mask, kind):
    """kind 'mse': linear outputs + squared error (continuous targets).
       kind 'ce' : softmax + cross-entropy (symbols). 'target' holds class ids then.
    mask (T, N) selects which time steps carry a target. Returns the mean over targeted steps."""
    if kind == "mse":
        err = ((o - target) ** 2).sum(-1) * 0.5
    else:
        err = F.cross_entropy(o.reshape(-1, o.shape[-1]), target.reshape(-1), reduction="none").view(mask.shape)
    return (err * mask).sum() / mask.sum()


class HFStructural:
    def __init__(self, model, kind, lam=0.1, mu=1 / 30, max_cg=300):
        """Thesis defaults: lambda = 0.1 and mu = 1/30 with structural damping (lambda = 0.3, mu = 0 without),
        at most 300 CG steps."""
        self.model, self.kind, self.lam, self.mu, self.max_cg = model, kind, lam, mu, max_cg
        self.names = [n for n, _ in model.named_parameters()]
        self.prev = None

    def _flat(self, d):
        return torch.cat([d[n].reshape(-1) for n in self.names])

    def _unflat(self, v):
        out, i = {}, 0
        for n, p in self.model.named_parameters():
            out[n] = v[i:i + p.numel()].view_as(p)
            i += p.numel()
        return out

    def objective(self, data):
        vs, target, mask = data
        o, _ = self.model(vs)
        return loss_fn(o, target, mask, self.kind)

    def grad(self, data):
        self.model.zero_grad()
        L = self.objective(data)
        L.backward()
        return L.item(), torch.cat([p.grad.reshape(-1) for p in self.model.parameters()])

    def curvature_product(self, data, v, structural=True):
        """(G_f + lambda mu G_S) v using one forward-mode and one reverse-mode pass."""
        vs, target, mask = data
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        fn = lambda prm: functional_call(self.model, prm, (vs,))
        (o, x), (Jo, Jx) = jvp(fn, (params,), (self._unflat(v),))
        m = mask[..., None] / mask.sum()
        if self.kind == "mse":
            Ho = Jo * m                                                    # Hessian of 0.5||o - t||^2 is I
        else:
            p = F.softmax(o, -1)
            Ho = (p * Jo - p * (p * Jo).sum(-1, keepdim=True)) * m         # softmax + cross-entropy Hessian
        if structural:
            # (D o e)'' = diag(1 - h^2): the Hessian of the "matching" distance for tanh units, w.r.t. x.
            # Scaled by lambda * mu and averaged over the N sequences, like the objective.
            h = torch.tanh(x)
            Hx = (1 - h ** 2) * Jx * (self.lam * self.mu / x.shape[1])
        else:
            Hx = torch.zeros_like(Jx)
        _, pull = vjp(fn, params)
        g, = pull((Ho, Hx))
        return self._flat(g)

    def step(self, grad_data, curv_data, structural=True):
        L0, g = self.grad(grad_data)
        A = lambda v: self.curvature_product(curv_data, v, structural) + self.lam * v
        x = 0.95 * self.prev if self.prev is not None else torch.zeros_like(g)
        r = -g - A(x)                                                      # residual of A x = -g
        d = r.clone()
        rr = r @ r
        phi = []                                                           # q(x) = g.x + 0.5 x.A.x after each CG step
        for i in range(self.max_cg):
            Ad = A(d)
            alpha = rr / (d @ Ad)
            x = x + alpha * d
            r = r - alpha * Ad
            phi.append((0.5 * (g @ x) - 0.5 * (x @ r)).item())             # A x = -g - r  =>  q = 0.5 g.x - 0.5 x.r
            k = max(10, int(0.1 * i))                                      # Martens (2010)'s stopping rule:
            if i > k and phi[-1] < 0 and (phi[-1] - phi[-1 - k]) / phi[-1] < k * 5e-4:
                break                                                      # little progress over the last k steps
            rr_new = r @ r
            if rr_new.sqrt() < 1e-10:
                break
            d = r + (rr_new / rr) * d
            rr = rr_new
        self.prev = x
        predicted = (g @ x + 0.5 * x @ A(x)).item()
        with torch.no_grad():
            for p, dp in zip(self.model.parameters(), self._unflat(x).values()):
                p += dp
            L1 = self.objective(grad_data).item()
        rho = (L1 - L0) / predicted if predicted < 0 else -1.0
        if rho < 0.25:
            self.lam *= 1.5
        elif rho > 0.75:
            self.lam *= 2 / 3
        if L1 > L0:                                                         # reject a bad step
            with torch.no_grad():
                for p, dp in zip(self.model.parameters(), self._unflat(x).values()):
                    p -= dp
            self.prev, L1 = None, L0
        return L0, L1, rho
