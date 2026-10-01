"""Zaremba, Sutskever & Vinyals (2014), "Recurrent Neural Network Regularization".

  LSTMCell        Section 3.1 (Graves 2013 form): [i, f, o, g] = [sigm, sigm, sigm, tanh](T_{2n,4n}[h^{l-1}_t ; h^l_{t-1}])
                  c^l_t = f * c^l_{t-1} + i * g,   h^l_t = o * tanh(c^l_t)
  DeepLSTM        L layers, word embeddings of size n, softmax output. Three dropout modes:
                    "nonrecurrent"  the paper's recipe: D(.) only on h^{l-1}_t (the vertical, layer-to-layer
                                    connections, including the embedding input and the softmax input)
                    "naive"         dropout ALSO on the recurrent connection h^l_{t-1} (a fresh mask each step)
                    "none"          no dropout
  dropouts_on_path   Figure 3: how many times information from k steps ago gets corrupted
  run_epoch       stateful truncated BPTT exactly as in Section 4.1 (unroll 35 steps, batch 20, carry the
                  hidden state between minibatches, clip the gradient norm, SGD)
  perplexity, ensemble_perplexity
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """One affine map T_{2n,4n} on the concatenation [input from below ; own previous output]."""

    def __init__(self, n_in, n):
        super().__init__()
        self.n = n
        self.T = nn.Linear(n_in + n, 4 * n)

    def forward(self, x, h, c):
        z = self.T(torch.cat([x, h], -1))
        i, f, o, g = z.chunk(4, -1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        return o * torch.tanh(c), c


class DeepLSTM(nn.Module):
    def __init__(self, vocab, n, layers, dropout=0.0, mode="nonrecurrent", init=0.05):
        super().__init__()
        self.vocab, self.n, self.L, self.p, self.mode = vocab, n, layers, dropout, mode
        self.embed = nn.Embedding(vocab, n)
        self.cells = nn.ModuleList([LSTMCell(n, n) for _ in range(layers)])
        self.out = nn.Linear(n, vocab)
        for prm in self.parameters():
            nn.init.uniform_(prm, -init, init)                      # 'initialized uniformly in [-0.05, 0.05]'
        self.record = False

    def D(self, x):
        """The dropout operator: zero a random subset, scale the rest by 1/(1-p) (inverted dropout)."""
        return F.dropout(x, self.p, self.training)

    def init_state(self, batch):
        z = torch.zeros(batch, self.n, device=self.out.weight.device)
        return [(z, z) for _ in range(self.L)]

    def forward(self, tokens, state):
        """tokens: (T, N) word ids; state: list of (h, c) per layer. Returns logits (T, N, V), new state."""
        outs, self.trace = [], []
        state = list(state)
        for x in self.embed(tokens):                                 # x = h^0_t
            below = x
            for l, cell in enumerate(self.cells):
                h, c = state[l]
                vertical = self.D(below) if self.mode != "none" else below
                recurrent = self.D(h) if self.mode == "naive" else h
                if self.record:
                    self.trace.append((l, vertical.detach(), recurrent.detach(), h.detach()))
                h, c = cell(vertical, recurrent, c)
                state[l] = (h, c)
                below = h
            top = self.D(below) if self.mode != "none" else below   # dropout before the softmax layer too
            outs.append(self.out(top))
        return torch.stack(outs), state


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Figure 3: dropout on the path of information
# ---------------------------------------------------------------------------

def dropouts_on_path(L, k, mode):
    """Information enters at time t-k, travels k steps to the right (recurrent edges, possibly in the top
    layer) and L+1 steps up (input -> layer 1 -> ... -> layer L -> softmax). Count the dropout operators
    it passes. 'nonrecurrent': L + 1 whatever k is. 'naive': L + 1 + k: it grows with the time lag."""
    vertical = L + 1
    return vertical + (k if mode == "naive" else 0) if mode != "none" else 0


# ---------------------------------------------------------------------------
# Section 4.1: training on a long token stream
# ---------------------------------------------------------------------------

def batchify(ids, batch):
    """Cut the token stream into `batch` parallel streams: (len // batch, batch). 'successive minibatches
    sequentially traverse the training set'."""
    n = len(ids) // batch
    return ids[:n * batch].view(batch, n).T.contiguous()


def run_epoch(model, data, steps=35, lr=None, clip=5.0, device="cpu"):
    """One pass over `data` ((T_total, N)). Truncated BPTT over `steps`; the final hidden state of one
    minibatch is the initial state of the next (detached). With lr: plain SGD with gradient-norm clipping.
    Returns the perplexity exp(mean cross-entropy)."""
    train = lr is not None
    model.train(train)
    state = model.init_state(data.shape[1])
    total, count = 0.0, 0
    for s in range(0, data.shape[0] - 1, steps):
        x = data[s:s + steps].to(device)
        y = data[s + 1:s + 1 + steps].to(device)
        x = x[:len(y)]
        state = [(h.detach(), c.detach()) for h, c in state]
        with torch.set_grad_enabled(train):
            logits, state = model(x, state)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1))
        if train:
            model.zero_grad()
            (loss * len(y)).backward()          # sum over time, averaged over the batch ('normalized by minibatch size')
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
        total += loss.item() * y.numel()
        count += y.numel()
    return math.exp(total / count)


@torch.no_grad()
def ensemble_perplexity(models, data, steps=35, device="cpu"):
    """'Model averaging': average the predicted PROBABILITIES of several models, then score."""
    for m in models:
        m.eval()
    states = [m.init_state(data.shape[1]) for m in models]
    total, count = 0.0, 0
    for s in range(0, data.shape[0] - 1, steps):
        x, y = data[s:s + steps].to(device), data[s + 1:s + 1 + steps].to(device)
        x = x[:len(y)]
        probs = 0
        for k, m in enumerate(models):
            logits, states[k] = m(x, states[k])
            probs = probs + F.softmax(logits, -1) / len(models)
        total += -torch.log(probs.gather(-1, y[..., None])).sum().item()
        count += y.numel()
    return math.exp(total / count)
