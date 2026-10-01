"""Pointer Networks (Vinyals, Fortunato & Jaitly, NIPS 2015) and the two baselines of Table 1.

  seq2seq (Sec. 2.1):   encoder LSTM reads P_1..P_n, then the decoder emits C_i with a softmax over a FIXED
                        dictionary of n_max + 1 classes (so one model per n).
  +attention (Sec. 2.2): u_ij = v^T tanh(W1 e_j + W2 d_i), a_i = softmax(u_i), d'_i = sum_j a_ij e_j;
                        [d_i ; d'_i] is used for the (fixed-size) softmax and fed to the next step.
  Ptr-Net (Sec. 2.3):   p(C_i | C_1..C_{i-1}, P) = softmax(u_i): the attention weights ARE the output distribution,
                        over the n inputs, so the dictionary grows with the input. The decoder input is the
                        coordinates of the previously pointed point, P_{C_{i-1}}.

The end token '<=' needs a place to point to: we prepend a learned 'end' state e_0, so index 0 = end and indices
1..n are the points (the paper's 1-based indices). The start token '=>' is a learned decoder input.
Hyperparameters (Sec. 4.1): 1-layer LSTM, 256 or 512 units, SGD lr 1.0, batch 128, uniform(-0.08, 0.08)
init, gradient norm clipped at 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

END = 0


class PtrNet(nn.Module):
    def __init__(self, hidden=256, mode="ptr", n_max=50, in_dim=2, embed=True):
        """mode: 'ptr' (Ptr-Net), 'seq2seq' or 'attention' (the baselines, with n_max + 1 output classes).
        embed: pass each point through a learned linear map to `hidden` dims before the LSTMs. The paper feeds
        the raw coordinates; with raw 2-d inputs and the small uniform init, short runs barely learn, so we embed
        by default (embed=False is the paper's literal setup)."""
        super().__init__()
        assert mode in ("ptr", "seq2seq", "attention")
        self.mode, self.h = mode, hidden
        self.emb = nn.Linear(in_dim, hidden) if embed else nn.Identity()
        in_dim = hidden if embed else in_dim
        self.enc = nn.LSTMCell(in_dim, hidden)
        dec_in = in_dim + (2 * hidden if mode == "attention" else 0)
        self.dec = nn.LSTMCell(dec_in, hidden)
        self.go = nn.Parameter(torch.zeros(in_dim))             # '=>' (start of output)
        self.e0 = nn.Parameter(torch.zeros(hidden))             # what the pointer points at to say '<=' (end)
        self.W1 = nn.Linear(hidden, hidden, bias=False)
        self.W2 = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)
        if mode != "ptr":
            self.out = nn.Linear(2 * hidden if mode == "attention" else hidden, n_max + 1)
        for p in self.parameters():
            nn.init.uniform_(p, -0.08, 0.08)                     # 'random uniform weight initialization'

    def encode(self, P):
        """P (N, n, 2) -> encoder states E (N, n + 1, h) with E[:, 0] = e_0 (end), and the final LSTM state."""
        N, n, _ = P.shape
        h = c = P.new_zeros(N, self.h)
        es = []
        for j in range(n):
            h, c = self.enc(P[:, j], (h, c))
            es.append(h)                                         # 'the state after the output gate'
        E = torch.cat([self.e0.expand(N, 1, -1), torch.stack(es, 1)], 1)
        return E, (h, c)

    def scores(self, E, W1E, d):
        return self.v(torch.tanh(W1E + self.W2(d)[:, None]))[..., 0]          # u_ij, (N, n + 1)

    def start(self, P):
        P = self.emb(P)                                          # (embedded) inputs; decoder inputs are copies
        E, state = self.encode(P)
        N = P.shape[0]
        S = {"P": P, "E": E, "W1E": self.W1(E), "state": state, "x": self.go.expand(N, -1)}
        if self.mode == "attention":
            S["feed"] = P.new_zeros(N, 2 * self.h)
        return S

    def step(self, S, mask=None):
        """One output step. Returns log p over the output dictionary (n + 1 for 'ptr', n_max + 1 otherwise)."""
        x = torch.cat([S["x"], S["feed"]], -1) if self.mode == "attention" else S["x"]
        h, c = self.dec(x, S["state"])
        S = dict(S, state=(h, c))
        if self.mode == "ptr":
            u = self.scores(S["E"], S["W1E"], h)
        elif self.mode == "attention":
            a = torch.softmax(self.scores(S["E"], S["W1E"], h)[:, 1:], -1)        # attend over the n points
            hd = torch.cat([h, (a[..., None] * S["E"][:, 1:]).sum(1)], -1)
            S["feed"] = hd
            u = self.out(hd)
        else:
            u = self.out(h)
        if mask is not None:
            u = u.masked_fill(~mask, float("-inf"))
        return F.log_softmax(u, -1), S

    def feed(self, S, idx):
        """Next decoder input: the coordinates of the chosen point P_{C_i} ('we simply copy the corresponding
        P_{C_{i-1}} as the input'); the end token feeds zeros."""
        P = S["P"]
        k = (idx - 1).clamp(min=0, max=P.shape[1] - 1)
        x = P[torch.arange(P.shape[0]), k] * (idx > 0)[:, None]
        return dict(S, x=x)

    def log_prob(self, P, C):
        """P (N, n, 2), C (N, m) 1-based targets followed by END (0) and padded with -1. Teacher forcing."""
        S = self.start(P)
        total = P.new_zeros(P.shape[0])
        for i in range(C.shape[1]):
            lp, S = self.step(S)
            tgt = C[:, i]
            valid = tgt >= 0
            total = total + lp.gather(1, tgt.clamp(min=0)[:, None])[:, 0] * valid
            S = self.feed(S, tgt.clamp(min=0))
        return total


def targets(seqs):
    """List of 1-based index sequences -> (N, m) tensor with END appended and -1 padding."""
    L = max(map(len, seqs)) + 1
    return torch.tensor([list(s) + [END] + [-1] * (L - len(s) - 1) for s in seqs])


@torch.no_grad()
def beam_search(model, P, beam=1, max_len=None, constraint=None):
    """Beam search for ONE point set P (n, 2). `constraint`: None (unconstrained, as for convex hull and Delaunay),
    or 'tour' (Sec. 4.4: 'only consider valid tours': each city once, end only after all cities).
    Returns the best output as a list of 1-based indices (without END)."""
    n = P.shape[0]
    max_len = max_len or 2 * n + 2
    S0 = model.start(P[None])
    hyps, done = [(0.0, [], S0)], []
    for _ in range(max_len):
        cand = []
        for score, seq, S in hyps:
            mask = None
            if constraint == "tour":
                mask = torch.ones(1, n + 1, dtype=torch.bool)
                mask[0, seq] = False
                mask[0, END] = len(seq) == n
            lp, S2 = model.step(S, mask)
            lp = lp[0]
            if model.mode != "ptr":
                lp = lp[:n + 1]                                 # the baselines can name classes beyond n
            top = lp.topk(min(beam, int(torch.isfinite(lp).sum())))
            for s, k in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((score + s, seq + [k], S2))
        cand.sort(key=lambda h: -h[0])
        hyps = []
        for score, seq, S in cand:
            if seq[-1] == END:
                done.append((score, seq[:-1]))
            else:
                hyps.append((score, seq, model.feed(S, torch.tensor([seq[-1]]))))
            if len(hyps) == beam:
                break
        if not hyps or (done and max(d[0] for d in done) >= hyps[0][0]):
            break
    best = max(done or [(h[0], h[1]) for h in hyps], key=lambda d: d[0])
    return best[1]
