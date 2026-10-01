"""Cho et al. (2014), "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine
Translation" - the GRU and the encoder-decoder, as described in Section 2 and Appendix A.

  GRUCell        Eqs. (5)-(8): r = sigm(W_r x + U_r h), z = sigm(W_z x + U_z h),
                 h~ = tanh(W x + U (r * h)),  h = z * h_prev + (1 - z) * h~
  Encoder        embeddings -> GRU over the source -> c = tanh(V h_N)            (Appendix A.1)
  Decoder        h'_0 = tanh(V' c); every step also sees c:                         (Appendix A.1.1)
                   z' = sigm(W'_z e(y_{t-1}) + U'_z h' + C_z c),  r' = sigm(W'_r e(y_{t-1}) + U'_r h' + C_r c)
                   h~' = tanh(W' e(y_{t-1}) + r' * (U' h' + C c))       (reset applied AFTER U', as written)
                   s' = O_h h' + O_y e(y_{t-1}) + O_c c,  s = maxout over pairs of s',  p = softmax(G s)
                 G is a product of two low-rank matrices
  EncoderDecoder log p(y | x) (Eq. 4 objective), greedy generation, sampling
  bleu           corpus BLEU-4 with the brevity penalty (Papineni et al. 2002): the metric of Table 1
  loglinear      Eq. (9): sum_n w_n f_n(f, e): how the RNN's phrase score enters a phrase-based SMT system
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


def orthogonal_(W):
    """'For the recurrent weight matrices, we first sampled from a white Gaussian distribution and used its
    left singular vectors matrix' (Saxe et al. 2014)."""
    with torch.no_grad():
        U, _, _ = torch.linalg.svd(torch.randn(W.shape[0], W.shape[0]))
        W.copy_(U[:, :W.shape[1]])
    return W


class GRUCell(nn.Module):
    """The paper's hidden unit. Input x (already embedded), state h. Biases omitted as in Appendix A."""

    def __init__(self, n_in, n):
        super().__init__()
        self.W = nn.Linear(n_in, 3 * n, bias=False)          # W, W_z, W_r stacked
        self.U_zr = nn.Linear(n, 2 * n, bias=False)
        self.U = nn.Linear(n, n, bias=False)

    def forward(self, x, h):
        wx, wz, wr = self.W(x).chunk(3, -1)
        uz, ur = self.U_zr(h).chunk(2, -1)
        z = torch.sigmoid(wz + uz)                            # update gate (6)
        r = torch.sigmoid(wr + ur)                            # reset gate (5)
        h_tilde = torch.tanh(wx + self.U(r * h))              # (8): reset BEFORE U
        return z * h + (1 - z) * h_tilde                      # (7)


class TanhCell(nn.Module):
    """The 'oft-used tanh unit without any gating' (the paper couldn't get meaningful results with it)."""

    def __init__(self, n_in, n):
        super().__init__()
        self.W, self.U = nn.Linear(n_in, n, bias=False), nn.Linear(n, n, bias=False)

    def forward(self, x, h):
        return torch.tanh(self.W(x) + self.U(h))


class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, n=1000, emb=100, maxout=500, rank=100, unit="gru"):
        super().__init__()
        cell = GRUCell if unit == "gru" else TanhCell
        self.n, self.unit = n, unit
        self.src_emb = nn.Embedding(src_vocab, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb, padding_idx=PAD)
        self.enc = cell(emb, n)
        self.V = nn.Linear(n, n, bias=False)                  # c = tanh(V h_N)
        self.V0 = nn.Linear(n, n, bias=False)                 # h'_0 = tanh(V' c)
        # decoder: input e(y_{t-1}) and the summary c
        self.dW = nn.Linear(emb, 3 * n, bias=False)           # W', W'_z, W'_r
        self.dC = nn.Linear(n, 3 * n, bias=False)             # C, C_z, C_r
        self.dU_zr = nn.Linear(n, 2 * n, bias=False)
        self.dU = nn.Linear(n, n, bias=False)
        self.O_h, self.O_y, self.O_c = nn.Linear(n, 2 * maxout), nn.Linear(emb, 2 * maxout, bias=False), nn.Linear(n, 2 * maxout, bias=False)
        self.G_r, self.G_l = nn.Linear(maxout, rank, bias=False), nn.Linear(rank, tgt_vocab)   # G = G_l G_r
        for p in self.parameters():                           # 'white Gaussian ... standard deviation 0.01'
            nn.init.normal_(p, 0, 0.01)
        for W in [m.weight for m in (self.dU, self.V, self.V0)] + [self.enc.U.weight]:
            orthogonal_(W)

    def encode(self, src):
        """src: (S, N) ids, PAD-padded at the end. Returns c (N, n). Padded steps keep the state unchanged."""
        h = torch.zeros(src.shape[1], self.n, device=src.device)
        for x, tok in zip(self.src_emb(src), src):
            h_new = self.enc(x, h)
            keep = (tok != PAD).float()[:, None]
            h = keep * h_new + (1 - keep) * h
        return torch.tanh(self.V(h))

    def decode_step(self, y_prev, h, c, cc):
        """One decoder step. cc = (C c, C_z c, C_r c), precomputed. Returns (logits, new h)."""
        e = self.tgt_emb(y_prev)
        wx, wz, wr = self.dW(e).chunk(3, -1)
        Cc, Czc, Crc = cc
        uz, ur = self.dU_zr(h).chunk(2, -1)
        z = torch.sigmoid(wz + uz + Czc)
        r = torch.sigmoid(wr + ur + Crc)
        if self.unit == "gru":
            h = z * h + (1 - z) * torch.tanh(wx + r * (self.dU(h) + Cc))
        else:
            h = torch.tanh(wx + self.dU(h) + Cc)
        s_prime = self.O_h(h) + self.O_y(e) + self.O_c(c)
        s = s_prime.view(*s_prime.shape[:-1], -1, 2).max(-1).values   # maxout: pairs -> one
        return self.G_l(self.G_r(s)), h

    def forward(self, src, tgt_in):
        """Teacher forcing. tgt_in: (T, N) starting with BOS. Returns logits (T, N, V)."""
        c = self.encode(src)
        h = torch.tanh(self.V0(c))
        cc = self.dC(c).chunk(3, -1)
        outs = []
        for y in tgt_in:
            logits, h = self.decode_step(y, h, c, cc)
            outs.append(logits)
        return torch.stack(outs)

    def log_prob(self, src, tgt):
        """log p(y | x) for each pair (Eq. 3-4). tgt: (T, N) = target tokens followed by EOS, PAD after."""
        tgt_in = torch.cat([torch.full_like(tgt[:1], BOS), tgt[:-1]])
        lp = F.log_softmax(self(src, tgt_in), -1).gather(-1, tgt[..., None])[..., 0]
        return (lp * (tgt != PAD)).sum(0)

    @torch.no_grad()
    def generate(self, src, max_len=30, sample=False, generator=None):
        c = self.encode(src)
        h = torch.tanh(self.V0(c))
        cc = self.dC(c).chunk(3, -1)
        y = torch.full((src.shape[1],), BOS, dtype=torch.long, device=src.device)
        out, done = [], torch.zeros_like(y, dtype=torch.bool)
        for _ in range(max_len):
            logits, h = self.decode_step(y, h, c, cc)
            y = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)[:, 0] if sample else logits.argmax(-1)
            y = torch.where(done, torch.full_like(y, PAD), y)
            out.append(y)
            done |= y == EOS
            if done.all():
                break
        return torch.stack(out)


# ---------------------------------------------------------------------------
# Section 3: the SMT side
# ---------------------------------------------------------------------------

def loglinear(features, weights):
    """Eq. (9) without the normalizer: score(f, e) = sum_n w_n f_n(f, e). The RNN Encoder-Decoder's
    log p(f | e) is just one more feature f_n, with its own weight tuned on a development set."""
    return sum(w * f for w, f in zip(weights, features))


def bleu(hypotheses, references, max_n=4):
    """Corpus BLEU (Papineni et al. 2002): geometric mean of modified n-gram precisions (n = 1..4) times a
    brevity penalty exp(1 - r/c) when the hypotheses are shorter than the references. Inputs: lists of token
    lists (one reference per sentence). Returns a number in [0, 100]."""
    clipped, totals = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp); ref_len += len(ref)
        for n in range(1, max_n + 1):
            h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            clipped[n - 1] += sum(min(c, r[g]) for g, c in h.items())        # 'modified' = clipped counts
            totals[n - 1] += max(len(hyp) - n + 1, 0)
    if min(totals) == 0 or min(clipped) == 0:
        return 0.0
    log_p = sum(math.log(c / t) for c, t in zip(clipped, totals)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(log_p)
