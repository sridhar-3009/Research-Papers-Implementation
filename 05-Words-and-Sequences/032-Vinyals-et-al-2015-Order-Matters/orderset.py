"""Order Matters: Sequence to Sequence for Sets (Vinyals, Bengio & Kudlur, ICLR 2016).

Input sets - the Read-Process-and-Write model (Section 4, Figure 1):
    Read:     m_i = MLP(x_i)                                  the same small network for every element
    Process:  q_t = LSTM(q*_{t-1})                            an LSTM with no inputs, run for T steps   (3)
              e_{i,t} = f(m_i, q_t) = m_i . q_t                content-based attention                  (4)
              a_{i,t} = softmax_i(e_{i,t})                                                               (5)
              r_t = sum_i a_{i,t} m_i                                                                    (6)
              q*_t = [q_t ; r_t]                                                                         (7)
    Write:    a pointer network (Paper 031) started from q*_T, with optional 'glimpses' (extra attention
              reads before each pointer step).
  Nothing in Read or Process depends on the ORDER of the m_i, so q*_T is permutation invariant.
  The baseline (Table 1's 'Ptr-Net') instead reads the elements with an LSTM, in the given order.

Output sets - search over orders while training (Section 5.2, Eq. 9):
    theta* = argmax sum_i max_pi log p(Y_pi(X_i) | X_i)
  with (a) a uniform-prior pretraining phase (a random order each time) and (b) sampling pi in proportion to
  p(Y_pi | X) by ancestral (left-to-right) sampling instead of the max.
"""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- input sets
class ReadProcessWrite(nn.Module):
    def __init__(self, in_dim=1, d=128, steps=1, glimpses=1, encoder="set", mask_repeats=False):
        """encoder: 'set' (Read-Process-and-Write) or 'lstm' (the Ptr-Net baseline of Table 1).
        steps: P, the number of process steps (0 = the writer starts 'blind').
        glimpses: attention reads before each pointer step (0 or 1 in Table 1).
        mask_repeats: forbid pointing twice at the same element (off by default, as in the paper)."""
        super().__init__()
        assert encoder in ("set", "lstm")
        self.d, self.steps, self.glimpses, self.encoder, self.mask_repeats = d, steps, glimpses, encoder, mask_repeats
        self.read = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d))
        if encoder == "lstm":
            self.enc = nn.LSTMCell(d, d)
        else:
            self.proc = nn.LSTMCell(2 * d, d)                   # q_t = LSTM(q*_{t-1}): its only input is q*
        self.init = nn.Linear(2 * d, d)                         # q*_T (or the encoder state) -> writer state
        self.dec = nn.LSTMCell(d, d)
        self.go = nn.Parameter(torch.zeros(d))
        self.Wg = nn.Linear(d, d, bias=False)                   # glimpse query
        self.W1 = nn.Linear(d, d, bias=False)
        self.W2 = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)

    # --- Process: Eqs. (3)-(7) ---
    def process(self, M):
        """M (N, n, d) memories -> q*_T (N, 2d), and the attention weights of every step."""
        N = M.shape[0]
        q_star = M.new_zeros(N, 2 * self.d)
        h = c = M.new_zeros(N, self.d)
        attn = []
        for _ in range(self.steps):
            h, c = self.proc(q_star, (h, c))                    # (3)
            e = (M * h[:, None]).sum(-1)                        # (4) dot-product attention
            a = torch.softmax(e, -1)                            # (5)
            r = (a[..., None] * M).sum(1)                       # (6)
            q_star = torch.cat([h, r], -1)                      # (7)
            attn.append(a)
        return q_star, attn

    def encode(self, X):
        """X (N, n, in_dim) -> memories (N, n, d) and the writer's initial state."""
        M = self.read(X)
        if self.encoder == "set":
            q_star, _ = self.process(M)
            h0 = torch.tanh(self.init(q_star))
        else:                                                   # order-dependent LSTM encoder
            N, n, _ = M.shape
            h = c = M.new_zeros(N, self.d)
            states = []
            for i in range(n):
                h, c = self.enc(M[:, i], (h, c))
                states.append(h)
            M = torch.stack(states, 1)                          # Ptr-Net points at encoder states
            h0 = torch.tanh(self.init(torch.cat([h, c], -1)))
        return M, h0

    def pointer_logits(self, M, h, used=None):
        query = h
        for _ in range(self.glimpses):                          # 'glimpse': an attention read before pointing
            a = torch.softmax((M * self.Wg(query)[:, None]).sum(-1), -1)
            query = query + (a[..., None] * M).sum(1)           # keep the decoder state, add what was read
            # (replacing the query by the readout alone, as a literal reading suggests, stalled training of
            #  the set model in our runs: at init the read is ~ the mean memory, the same at every step)
        u = self.v(torch.tanh(self.W1(M) + self.W2(query)[:, None]))[..., 0]
        if used is not None and self.mask_repeats:
            u = u.masked_fill(used, float("-inf"))
        return u

    def log_prob(self, X, order):
        """X (N, n, in_dim), order (N, n) = the target sequence of indices. Teacher forcing."""
        M, h = self.encode(X)
        c = torch.zeros_like(h)
        x = self.go.expand(X.shape[0], -1)
        used = torch.zeros(order.shape, dtype=torch.bool)
        total = X.new_zeros(X.shape[0])
        idx = torch.arange(X.shape[0])
        for t in range(order.shape[1]):
            h, c = self.dec(x, (h, c))
            lp = F.log_softmax(self.pointer_logits(M, h, used), -1)
            total = total + lp[idx, order[:, t]]
            used = used.clone(); used[idx, order[:, t]] = True
            x = M[idx, order[:, t]]
        return total

    @torch.no_grad()
    def greedy(self, X):
        M, h = self.encode(X)
        c = torch.zeros_like(h)
        x = self.go.expand(X.shape[0], -1)
        n = X.shape[1]
        used = torch.zeros(X.shape[0], n, dtype=torch.bool)
        idx = torch.arange(X.shape[0])
        out = []
        for _ in range(n):
            h, c = self.dec(x, (h, c))
            k = self.pointer_logits(M, h, used).argmax(-1)
            out.append(k)
            used = used.clone(); used[idx, k] = True
            x = M[idx, k]
        return torch.stack(out, 1)


def sort_batch(N, n, generator=None):
    X = torch.rand(N, n, 1, generator=generator)
    return X, X[..., 0].argsort(1)


# ---------------------------------------------------------------------------------------------------- output sets
class SetLM(nn.Module):
    """An LSTM over a sequence of tokens: the chain rule p(y_1..y_n) = prod p(y_t | y_<t) (Eq. 8). Used to model
    an output SET by emitting its elements in some order. Token ids must be < vocab."""

    def __init__(self, vocab, d=64):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d)                   # the extra id is the start symbol
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.out = nn.Linear(d, vocab)
        self.start = vocab

    def log_probs(self, seqs):
        """seqs (N, n) -> per-step log-distributions (N, n, vocab) for predicting seqs[:, t]."""
        inp = torch.cat([torch.full_like(seqs[:, :1], self.start), seqs[:, :-1]], 1)
        h, _ = self.lstm(self.emb(inp))
        return F.log_softmax(self.out(h), -1)

    def seq_log_prob(self, seqs):
        return self.log_probs(seqs).gather(-1, seqs[..., None])[..., 0].sum(1)


def order_log_prob(model, items, order):
    """log p(Y_pi): items (N, n) token ids, order (N, n) a permutation per row."""
    return model.seq_log_prob(items.gather(1, order))


@torch.no_grad()
def best_order(model, items):
    """Eq. 9's inner max, exactly: try all n! orders (only for small n)."""
    N, n = items.shape
    perms = torch.tensor(list(itertools.permutations(range(n))))                   # (n!, n)
    scores = torch.stack([order_log_prob(model, items, p.expand(N, -1)) for p in perms], 1)
    return perms[scores.argmax(1)], scores


@torch.no_grad()
def sample_order(model, items, generator=None):
    """Section 5.2: sample pi by ancestral (left-to-right) sampling. At each step choose among the elements not yet
    emitted, with probability proportional to the model's p(element | prefix). Returns the order and its sampling
    probability q(pi) = prod_t p(y_pi_t | prefix) / Z_t, where Z_t sums p over the remaining elements. This equals
    p(Y_pi) / sum_pi' p(Y_pi') only when the model puts all its mass on the remaining elements (every Z_t = 1)."""
    N, n = items.shape
    idx = torch.arange(N)
    remaining = torch.ones(N, n, dtype=torch.bool)
    order, logq = [], items.new_zeros(N, dtype=torch.float)
    h = None
    x = torch.full((N, 1), model.start, dtype=torch.long)
    for _ in range(n):
        o, h = model.lstm(model.emb(x), h)
        p = F.softmax(model.out(o[:, -1]), -1).gather(1, items)                    # p(each element | prefix)
        p = p * remaining
        k = torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=generator)[:, 0]
        logq += torch.log(p[idx, k] / p.sum(1))
        order.append(k)
        remaining[idx, k] = False
        x = items[idx, k][:, None]
    return torch.stack(order, 1), logq


def eq9_step(model, opt, items, mode, generator=None):
    """One training step of Eq. 9 on a batch of sets. mode: 'given' (the order in `items`), 'uniform' (a random
    order: the pretraining phase), 'max' (exact max over orders) or 'sample' (ancestral sampling)."""
    N, n = items.shape
    if mode == "given":
        order = torch.arange(n).expand(N, -1)
    elif mode == "uniform":
        order = torch.stack([torch.randperm(n, generator=generator) for _ in range(N)])
    elif mode == "max":
        order, _ = best_order(model, items)
    else:
        order, _ = sample_order(model, items, generator)
    loss = -order_log_prob(model, items, order).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), order


# ---------------------------------------------------------------------------------------------------- ordering tasks
def reverse_words(words):
    return list(reversed(words))


def three_word_reversal(words, pad="<pad>"):
    """Section 5.1.1: reverse every block of 3 words ('This is a sentence .' -> 'a is This <pad> . sentence')."""
    words = list(words) + [pad] * (-len(words) % 3)
    return [w for k in range(0, len(words), 3) for w in reversed(words[k:k + 3])]


def dfs_linearize(t):
    """Figure 2, depth first: 'S NP DT !DT !NP VP VBZ !VBZ NP DT !DT NN !NN !NP !VP . !. !S' (words dropped)."""
    label, kids = t
    return [label] + [s for k in kids for s in dfs_linearize(k)] + ["!" + label]


def bfs_linearize(t):
    """Figure 2, breadth first: 'S LEV NP VP . LEV DT PAR VBZ NP LEV PAR PAR DT NN DONE'. Each level lists, for every
    node of the previous level, its children; groups are separated by PAR, trailing empty groups are dropped."""
    out, level = [t[0]], [t]
    while True:
        groups = [[k[0] for k in node[1]] for node in level]
        while groups and not groups[-1]:
            groups.pop()
        if not groups:
            return out + ["DONE"]
        out.append("LEV")
        for g, grp in enumerate(groups):
            if g:
                out.append("PAR")
            out += grp
        level = [k for node in level for k in node[1]]


def bfs_delinearize(seq):
    """Inverse of bfs_linearize (labels only)."""
    assert seq[-1] == "DONE"
    root = (seq[0], [])
    level, i = [root], 1
    while seq[i] != "DONE":
        assert seq[i] == "LEV"
        i += 1
        g, nxt = 0, []
        while seq[i] not in ("LEV", "DONE"):
            if seq[i] == "PAR":
                g += 1
            else:
                node = (seq[i], [])
                level[g][1].append(node)
                nxt.append(node)
            i += 1
        level = [k for node in level for k in node[1]]
    return root


def star_model(n_vars, K=10, peaky=1.0, generator=None):
    """Section 5.1.4: a star-shaped graphical model. The head y_0 ~ Cat(pi); every other y_j ~ Cat(P_j[y_0]).
    Distributions are softmax(peaky * Gaussian), so a large `peaky` makes them nearly deterministic."""
    g = generator
    head_p = torch.softmax(peaky * torch.randn(K, generator=g), 0)
    tables = torch.softmax(peaky * torch.randn(n_vars - 1, K, K, generator=g), -1)   # [j, head value, value]
    return head_p, tables


def star_sample(model, n_samples, generator=None):
    """Draw n_samples joint samples from a star model; returns integer values (n_samples, n_vars)."""
    head_p, tables = model
    head = torch.multinomial(head_p, n_samples, replacement=True, generator=generator)
    rest = torch.stack([torch.multinomial(tables[j, head], 1, generator=generator)[:, 0]
                        for j in range(tables.shape[0])], 1)
    return torch.cat([head[:, None], rest], 1)


def star_log_prob(model, values):
    """Exact log p(y) under the star model (the best any learner could do)."""
    head_p, tables = model
    lp = torch.log(head_p[values[:, 0]])
    for j in range(tables.shape[0]):
        lp = lp + torch.log(tables[j, values[:, 0], values[:, j + 1]])
    return lp


def star_tokens(values, head_first=True, K=10):
    """Turn variable values into tokens 'variable j has value v' = j*K + v, in head-first or head-last order."""
    n = values.shape[1]
    tok = torch.arange(n) * K + values
    return tok if head_first else torch.cat([tok[:, 1:], tok[:, :1]], 1)
