"""Sutskever, Martens & Hinton (2011), "Generating Text with Recurrent Neural Networks".

  CharRNN        Eqs. (1)-(2): h_t = tanh(W_hx x_t + W_hh h_{t-1} + b_h), o_t = W_oh h_t + b_o
  TensorRNN      Eqs. (3), (5): every character c has its OWN hidden-to-hidden matrix W_hh^(c)
  MRNN           Eqs. (7)-(9): the tensor factored as W_hh^(x) = W_hf diag(W_fx x) W_fh
                     f_t = diag(W_fx x_t) W_fh h_{t-1}       (multiplicative "factor" units)
                     h_t = tanh(W_hf f_t + W_hx x_t)
                     o_t = W_oh h_t + b_o
  effective_matrix   Eq. (10): the hidden-to-hidden matrix a character "synthesizes"
  sparse_init_       'each unit starts out with 15 nonzero connections' (Martens & Sutskever 2011)
  bits_per_char, sample, debag     evaluation, generation (Section 6), the debagging test (Section 5.4)
  HessianFree    a compact Hessian-free optimizer (Martens 2010): Gauss-Newton matrix-vector
                 products with jvp/vjp, conjugate gradient, Levenberg-Marquardt damping.
                 (The RNN-specific 'structural damping' of Martens & Sutskever 2011 is not included.)
"""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------

def sparse_init_(W, k=15, scale=1.0, generator=None):
    """Every unit (row) gets exactly k nonzero incoming weights ~ N(0, scale^2); the rest are 0.
    Sparse init keeps the units' inputs diverse without making the total input too big."""
    with torch.no_grad():
        W.zero_()
        rows, cols = W.shape
        for r in range(rows):
            idx = torch.randperm(cols, generator=generator)[:min(k, cols)]
            W[r, idx] = torch.randn(len(idx), generator=generator) * scale
    return W


class CharRNN(nn.Module):
    """The standard RNN of Eqs. (1)-(2). h_0 is a learned 'initial bias vector' h_init."""

    def __init__(self, vocab, hidden):
        super().__init__()
        self.M, self.H = vocab, hidden
        self.W_hx = nn.Linear(vocab, hidden, bias=True)          # includes b_h
        self.W_hh = nn.Linear(hidden, hidden, bias=False)
        self.W_oh = nn.Linear(hidden, vocab)
        self.h_init = nn.Parameter(torch.zeros(hidden))

    def step(self, x, h):
        return torch.tanh(self.W_hx(x) + self.W_hh(h))

    def forward(self, xs, h=None):
        """xs: (T, N, M) one-hot characters -> logits (T, N, M), final hidden state."""
        h = self.h_init.expand(xs.shape[1], -1) if h is None else h
        outs = []
        for x in xs:
            h = self.step(x, h)
            outs.append(self.W_oh(h))
        return torch.stack(outs), h


class TensorRNN(CharRNN):
    """Eqs. (3), (5): W_hh^(x_t) = sum_m x_t^(m) W_hh^(m). With one-hot x this simply selects
    the matrix of the current character: M full H x H matrices (too many parameters for big H)."""

    def __init__(self, vocab, hidden):
        super().__init__(vocab, hidden)
        del self.W_hh
        self.W_hh = nn.Parameter(torch.randn(vocab, hidden, hidden) / math.sqrt(hidden))

    def step(self, x, h):
        Whh_x = torch.einsum("nm,mij->nij", x, self.W_hh)          # one matrix per example
        return torch.tanh(self.W_hx(x) + torch.einsum("nij,nj->ni", Whh_x, h))


class MRNN(nn.Module):
    """Eqs. (7)-(9). F factors. Each factor f is a rank-one hidden-to-hidden matrix
    (outer product of a column of W_hf and a row of W_fh) whose GAIN is set by the input, W_fx x."""

    def __init__(self, vocab, hidden, factors, bias=True):
        super().__init__()
        self.M, self.H, self.F = vocab, hidden, factors
        self.W_fx = nn.Linear(vocab, factors, bias=False)
        self.W_fh = nn.Linear(hidden, factors, bias=False)
        self.W_hf = nn.Linear(factors, hidden, bias=False)
        self.W_hx = nn.Linear(vocab, hidden, bias=bias)            # eq. (8) has no bias; eq. (3) has b_h
        self.W_oh = nn.Linear(hidden, vocab)
        self.h_init = nn.Parameter(torch.zeros(hidden))

    def step(self, x, h):
        f = self.W_fx(x) * self.W_fh(h)                             # (7): diag(W_fx x) W_fh h
        return torch.tanh(self.W_hf(f) + self.W_hx(x))              # (8)

    def forward(self, xs, h=None):
        h = self.h_init.expand(xs.shape[1], -1) if h is None else h
        outs = []
        for x in xs:
            h = self.step(x, h)
            outs.append(self.W_oh(h))                               # (9)
        return torch.stack(outs), h

    def effective_matrix(self, c):
        """Eq. (6)/(10): W^(c) = W_hf diag(W_fx e_c) W_fh, the H x H matrix character c 'synthesizes'."""
        gains = self.W_fx.weight[:, c]                              # W_fx e_c
        return self.W_hf.weight @ torch.diag(gains) @ self.W_fh.weight

    def sparse_init(self, k=15, scale=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        for W in (self.W_fh.weight, self.W_hf.weight, self.W_hx.weight, self.W_fx.weight):
            sparse_init_(W, k, scale, g)
        return self


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Section 4: a generative model of text
# ---------------------------------------------------------------------------

def one_hot(ids, M):
    return F.one_hot(ids, M).float()


def nll(model, ids, skip=0):
    """Total -log P(x_{t+1} | x_<=t) in nats over a (T, N) id tensor, ignoring the first `skip`
    predictions ('predict only the last 200 timesteps ... at least 50 characters of context')."""
    logits, _ = model(one_hot(ids[:-1], model.M))
    lp = F.log_softmax(logits, -1).gather(-1, ids[1:, :, None])[..., 0]
    return -lp[skip:].sum(), lp[skip:].numel()


@torch.no_grad()
def bits_per_char(model, ids, skip=0):
    total, n = nll(model, ids, skip)
    return (total / n / math.log(2)).item()


@torch.no_grad()
def sample(model, prefix_ids, n, temperature=1.0, generator=None):
    """'We can sample from this conditional distribution to get the next character ... and provide
    it as the next input to the RNN.' prefix_ids: list of ids used to set the hidden state first."""
    h = None
    xs = one_hot(torch.tensor(prefix_ids)[:, None], model.M)
    logits, h = model(xs, h)
    out = []
    last = logits[-1, 0]
    for _ in range(n):
        p = F.softmax(last / temperature, -1)
        c = torch.multinomial(p, 1, generator=generator).item()
        out.append(c)
        logits, h = model(one_hot(torch.tensor([[c]]), model.M), h)
        last = logits[-1, 0]
    return out


@torch.no_grad()
def log_prob_of(model, ids):
    """Total log P(ids[1:] | ids[0]) of one sequence (natural log)."""
    total, _ = nll(model, torch.tensor(ids)[:, None])
    return -total.item()


def debag(model, words, encode, context_before=(), context_after=()):
    """Section 5.4: try every ordering of the bag of words, return the one with the highest
    log-probability under the character model (7 words -> 7! = 5040 orderings)."""
    best, best_lp = None, -float("inf")
    for perm in itertools.permutations(words):
        text = " ".join(list(context_before) + list(perm) + list(context_after))
        lp = log_prob_of(model, encode(text))
        if lp > best_lp:
            best, best_lp = perm, lp
    return best, best_lp


# ---------------------------------------------------------------------------
# Hessian-free optimization (Martens 2010), compact version
# ---------------------------------------------------------------------------

class HessianFree:
    """One HF step:
        1. gradient g of the loss (on a big batch),
        2. curvature: the Gauss-Newton matrix G = J^T H_L J, used only through products G v,
           computed with a forward-mode jvp (J v), the softmax/cross-entropy Hessian H_L, and a
           reverse-mode vjp (J^T u), on a smaller batch,
        3. conjugate gradient solves (G + lambda I) d = -g approximately (no matrix is ever formed),
        4. Levenberg-Marquardt: rho = actual / predicted reduction; lambda *= 2/3 if rho > 3/4,
           *= 3/2 if rho < 1/4 (Martens 2010's rule)."""

    def __init__(self, model, lam=10.0, cg_iters=50):
        self.model, self.lam, self.cg_iters = model, lam, cg_iters
        self.names = [n for n, _ in model.named_parameters()]
        self.prev_d = None                                          # CG warm start ('decay' of 0.95)

    def _flat(self, tensors):
        return torch.cat([t.reshape(-1) for t in tensors])

    def _unflat(self, v):
        out, i = {}, 0
        for n, p in self.model.named_parameters():
            out[n] = v[i:i + p.numel()].view_as(p)
            i += p.numel()
        return out

    def _params(self):
        return {n: p.detach() for n, p in self.model.named_parameters()}

    def loss_and_grad(self, ids):
        self.model.zero_grad()
        total, n = nll(self.model, ids)
        (total / n).backward()
        return (total / n).item(), self._flat([p.grad for p in self.model.parameters()])

    def gauss_newton_product(self, ids, v):
        """G v = J^T H_L J v / n, J = d logits / d params, H_L = diag(p) - p p^T per prediction."""
        xs = one_hot(ids[:-1], self.model.M)
        params = self._params()
        logits_fn = lambda prm: functional_call(self.model, prm, (xs,))[0]
        logits, Jv = jvp(logits_fn, (params,), (self._unflat(v),))
        p = F.softmax(logits, -1)
        HJv = p * Jv - p * (p * Jv).sum(-1, keepdim=True)
        _, pullback = vjp(logits_fn, params)
        JtHJv, = pullback(HJv / p[..., 0].numel())
        return self._flat([JtHJv[n] for n in self.names])

    def conjugate_gradient(self, Avp, b, x0):
        x = x0.clone()
        r = b - Avp(x)
        d = r.clone()
        rr = r @ r
        for _ in range(self.cg_iters):
            Ad = Avp(d)
            alpha = rr / (d @ Ad)
            x = x + alpha * d
            r = r - alpha * Ad
            rr_new = r @ r
            if rr_new.sqrt() < 1e-10:
                break
            d = r + (rr_new / rr) * d
            rr = rr_new
        return x

    def step(self, grad_ids, curv_ids):
        loss0, g = self.loss_and_grad(grad_ids)
        Avp = lambda v: self.gauss_newton_product(curv_ids, v) + self.lam * v
        x0 = 0.95 * self.prev_d if self.prev_d is not None else torch.zeros_like(g)
        d = self.conjugate_gradient(Avp, -g, x0)
        self.prev_d = d
        predicted = (g @ d + 0.5 * d @ Avp(d)).item()               # the quadratic model's change
        with torch.no_grad():
            for p, dp in zip(self.model.parameters(), self._unflat(d).values()):
                p += dp
        with torch.no_grad():
            total, n = nll(self.model, grad_ids)
        loss1 = (total / n).item()
        rho = (loss1 - loss0) / predicted if predicted != 0 else 0.0
        if rho > 0.75:
            self.lam *= 2 / 3
        elif rho < 0.25:
            self.lam *= 1.5
        if loss1 > loss0:                                          # reject a step that made things worse
            with torch.no_grad():
                for p, dp in zip(self.model.parameters(), self._unflat(d).values()):
                    p -= dp
            self.prev_d = None
            loss1 = loss0
        return loss0, loss1, rho
