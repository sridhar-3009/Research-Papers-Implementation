"""Long Short-Term Memory - Hochreiter & Schmidhuber (1997) - the ORIGINAL LSTM.

Note: the 1997 LSTM has NO forget gate (that was added by Gers, Schmidhuber & Cummins, 2000).
Its memory cell only adds:   s(t) = s(t-1) + y_in(t) * g(net_c(t)).

  Section 3.1   error_scaling            how an error is scaled after q steps back in time (Eq. 2)
  Appendix A.1  f, g, h                  logistic gates in [0,1]; g in [-2,2]; h in [-1,1]
                LSTM1997 (NumPy)         memory cell blocks with input and output gates, and the
                                         paper's TRUNCATED learning rule: forward-mode derivatives
                                         ds/dw carried along the constant error carrousel (CEC)
                LSTM1997Torch            the same network in PyTorch. With truncate=True the
                                         recurrent inputs are detached, which gives exactly the
                                         paper's truncated gradient (checked in the tests)
                VanillaRNN               a standard sigmoid RNN trained by full BPTT, the baseline
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Appendix A.1: the squashing functions
# ---------------------------------------------------------------------------

def f(x):
    """Logistic sigmoid, range [0, 1]: gates and output units."""
    return 1.0 / (1.0 + np.exp(-x))


def df(x):
    s = f(x)
    return s * (1 - s)


def g(x):
    """Squashes the cell input, range [-2, 2]: g(x) = 4 / (1 + e^-x) - 2."""
    return 4.0 * f(x) - 2.0


def dg(x):
    return 4.0 * df(x)


def h(x):
    """Scales the cell output, range [-1, 1]: h(x) = 2 / (1 + e^-x) - 1."""
    return 2.0 * f(x) - 1.0


def dh(x):
    return 2.0 * df(x)


# ---------------------------------------------------------------------------
# Section 3.1: why errors vanish in ordinary recurrent nets
# ---------------------------------------------------------------------------

def error_scaling(fprime_times_w):
    """Eq. (2) along ONE path: the error is multiplied by f'(net(t-m)) * w at every step back.
    fprime_times_w: the factors |f'(net) w| for m = 1..q. Returns their product.
    If every factor is < 1 the error vanishes exponentially in q; if > 1 it blows up."""
    return float(np.prod(np.abs(fprime_times_w)))


def max_sigmoid_factor(w):
    """max over net of |f'(net) w| for the logistic sigmoid = 0.25 |w| (f' <= 0.25).
    So with |w| < 4.0 the error is guaranteed to shrink at every step."""
    return 0.25 * abs(w)


# ---------------------------------------------------------------------------
# The 1997 LSTM in NumPy, with the paper's truncated learning rule
# ---------------------------------------------------------------------------

class LSTM1997:
    """One hidden layer of B memory-cell blocks with S cells each (C = B*S cells), K output units.

    Every gate and cell sees  z(t) = [x(t), y_c(t-1), 1]  (inputs, all cell outputs, a bias).
        y_in  = f(W_in z)                (one input gate per block)
        y_out = f(W_out z)               (one output gate per block)
        s     = s(t-1) + y_in * g(W_c z) (the CEC: a linear unit with self-weight 1.0)
        y_c   = y_out * h(s)
        y_k   = f(W_k [y_c, 1])          (logistic output units)
    """

    def __init__(self, n_in, n_blocks, block_size, n_out, init=0.1, in_gate_bias=None, out_gate_bias=None, seed=0):
        rng = np.random.default_rng(seed)
        self.I, self.B, self.S, self.K = n_in, n_blocks, block_size, n_out
        self.C = n_blocks * block_size
        Z = n_in + self.C + 1
        u = lambda *shape: rng.uniform(-init, init, shape)
        self.W_in, self.W_out, self.W_c, self.W_k = u(self.B, Z), u(self.B, Z), u(self.C, Z), u(n_out, self.C + 1)
        # gate biases (the last column of z is the constant 1). Negative biases keep gates closed at first,
        # one remedy for the "abuse problem" and for internal state drift (Section 4).
        if in_gate_bias is not None:
            self.W_in[:, -1] = in_gate_bias
        if out_gate_bias is not None:
            self.W_out[:, -1] = out_gate_bias

    def params(self):
        return [self.W_in, self.W_out, self.W_c, self.W_k]

    def rep(self, v):
        """Repeat a per-block value for each of the block's S cells."""
        return np.repeat(v, self.S)

    def forward(self, xs):
        """xs: (T, I). Returns a list of per-step dicts with every quantity the learning rule needs."""
        s = np.zeros(self.C)
        yc = np.zeros(self.C)
        steps = []
        for x in xs:
            z = np.concatenate([x, yc, [1.0]])
            n_in, n_out, n_c = self.W_in @ z, self.W_out @ z, self.W_c @ z
            y_in, y_out = f(n_in), f(n_out)
            s = s + self.rep(y_in) * g(n_c)
            yc = self.rep(y_out) * h(s)
            n_k = self.W_k @ np.concatenate([yc, [1.0]])
            steps.append(dict(z=z, n_in=n_in, n_out=n_out, n_c=n_c, y_in=y_in, y_out=y_out, s=s.copy(), yc=yc.copy(),
                              n_k=n_k, y=f(n_k)))
        return steps

    def truncated_gradient(self, xs, targets, mask):
        """Gradient of E = 1/2 sum_t mask(t) ||target(t) - y(t)||^2 with the paper's truncation:
        errors are NOT propagated back through y_u(t-1); they flow back in time ONLY through the
        internal states s (the CEC), where the factor is exactly 1.0 at every step.

        Implemented as in Appendix A.1, forward in time (RTRL-style, O(W) per step):
            dS_c[v, u]  += g'(net_c) * y_in * z_u        (derivative of s_v w.r.t. W_c[v, u])
            dS_in[v, u] += g(net_c) * f'(net_in) * z_u    (derivative of s_v w.r.t. its block's W_in[., u])
        and whenever an error arrives at time t:
            e_k   = f'(net_k)(y - target)                          output units
            e_c   = W_k^T e_k                                      at each cell output
            e_out = f'(net_out) * sum_{cells in block} h(s) e_c    output gate (no flow back in time)
            e_s   = y_out h'(s) e_c                                into the CEC
            grad W_c += e_s * dS_c,   grad W_in += sum over the block of e_s * dS_in"""
        grads = [np.zeros_like(p) for p in self.params()]
        gW_in, gW_out, gW_c, gW_k = grads
        Z = self.I + self.C + 1
        dS_c, dS_in = np.zeros((self.C, Z)), np.zeros((self.C, Z))
        loss = 0.0
        for st, tgt, m in zip(self.forward(xs), targets, mask):
            z = st["z"]
            dS_c += (dg(st["n_c"]) * self.rep(st["y_in"]))[:, None] * z[None, :]
            dS_in += (g(st["n_c"]) * self.rep(df(st["n_in"])))[:, None] * z[None, :]
            if not m:
                continue
            err = st["y"] - tgt
            loss += 0.5 * float(err @ err)
            e_k = df(st["n_k"]) * err
            gW_k += np.outer(e_k, np.concatenate([st["yc"], [1.0]]))
            e_c = self.W_k[:, :self.C].T @ e_k
            e_out = df(st["n_out"]) * (h(st["s"]) * e_c).reshape(self.B, self.S).sum(1)
            gW_out += np.outer(e_out, z)
            e_s = self.rep(st["y_out"]) * dh(st["s"]) * e_c
            gW_c += e_s[:, None] * dS_c
            gW_in += (e_s[:, None] * dS_in).reshape(self.B, self.S, Z).sum(1)
        return loss, grads

    def sgd_step(self, xs, targets, mask, lr):
        loss, grads = self.truncated_gradient(xs, targets, mask)
        for p, gr in zip(self.params(), grads):
            p -= lr * gr
        return loss


# ---------------------------------------------------------------------------
# The same network in PyTorch (fast, batched; for the experiments)
# ---------------------------------------------------------------------------

class LSTM1997Torch(nn.Module):
    """Identical parameters and equations to LSTM1997. Input (T, N, I) -> outputs (T, N, K).
    truncate=True: y_c(t-1) is detached where it enters the gates and cells, so autograd computes
    exactly the paper's truncated gradient; truncate=False gives the full BPTT gradient."""

    def __init__(self, n_in, n_blocks, block_size, n_out, truncate=True, init=0.1, in_gate_bias=None,
                 out_gate_bias=None, output_sigmoid=True, dtype=torch.float32, seed=0):
        super().__init__()
        ref = LSTM1997(n_in, n_blocks, block_size, n_out, init, in_gate_bias, out_gate_bias, seed)
        self.B, self.S, self.C = n_blocks, block_size, ref.C
        P = lambda a: nn.Parameter(torch.tensor(a, dtype=dtype))
        self.W_in, self.W_out, self.W_c, self.W_k = P(ref.W_in), P(ref.W_out), P(ref.W_c), P(ref.W_k)
        self.truncate, self.output_sigmoid = truncate, output_sigmoid

    def forward(self, xs, return_states=False, s0=None):
        """s0: optional initial internal state (N, C); zeros by default ('activations are reset')."""
        T, N, _ = xs.shape
        s = xs.new_zeros(N, self.C) if s0 is None else s0
        yc = xs.new_zeros(N, self.C)
        one = xs.new_ones(N, 1)
        outs, states = [], []
        for t in range(T):
            rec = yc.detach() if self.truncate else yc
            z = torch.cat([xs[t], rec, one], 1)
            y_in = torch.sigmoid(z @ self.W_in.T).repeat_interleave(self.S, 1)
            y_out = torch.sigmoid(z @ self.W_out.T).repeat_interleave(self.S, 1)
            s = s + y_in * (4 * torch.sigmoid(z @ self.W_c.T) - 2)          # CEC: s(t) = s(t-1) + y_in g(net_c)
            yc = y_out * (2 * torch.sigmoid(s) - 1)                          # y_c = y_out h(s)
            net_k = torch.cat([yc, one], 1) @ self.W_k.T
            outs.append(torch.sigmoid(net_k) if self.output_sigmoid else net_k)
            states.append(s)
        out = torch.stack(outs)
        return (out, torch.stack(states)) if return_states else out


class VanillaRNN(nn.Module):
    """A conventional fully recurrent sigmoid net, h(t) = sigmoid(W [x(t), h(t-1), 1]), trained with
    full BPTT. Section 3 predicts its errors vanish over long time lags."""

    def __init__(self, n_in, n_hidden, n_out, init=0.2, output_sigmoid=True, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        u = lambda *s: nn.Parameter((torch.rand(*s, generator=gen) * 2 - 1) * init)
        self.W, self.V = u(n_hidden, n_in + n_hidden + 1), u(n_out, n_hidden + 1)
        self.H, self.output_sigmoid = n_hidden, output_sigmoid

    def forward(self, xs):
        T, N, _ = xs.shape
        hdn, one, outs = xs.new_zeros(N, self.H), xs.new_ones(N, 1), []
        for t in range(T):
            hdn = torch.sigmoid(torch.cat([xs[t], hdn, one], 1) @ self.W.T)
            net = torch.cat([hdn, one], 1) @ self.V.T
            outs.append(torch.sigmoid(net) if self.output_sigmoid else net)
        return torch.stack(outs)
