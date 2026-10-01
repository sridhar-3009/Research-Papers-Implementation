"""Bahdanau, Cho & Bengio (2015), "Neural Machine Translation by Jointly Learning to Align and Translate":
RNNsearch (attention) and its baseline RNNencdec, exactly as in Appendix A.

  GRU            s = (1 - z) * s_prev + z * s~,  s~ = tanh(W e + U (r * s_prev) [+ C c])   (A.1.1; note: this
                 paper's z is '1 - z' of Cho et al. 2014: z = 1 means TAKE the new state)
  BiEncoder      forward and backward GRUs over the source (shared embeddings E); annotation
                 h_j = [ forward h_j ; backward h_j ]   (Eq. 7)
  alignment      e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)   (A.1.2; U_a h_j is precomputed once)
                 alpha_ij = softmax_j(e_ij)                 (Eq. 6)
                 c_i = sum_j alpha_ij h_j                   (Eq. 5)  'an expected annotation'
  RNNsearch      decoder state s_i = GRU(s_{i-1}, y_{i-1}, c_i), s_0 = tanh(W_s backward-h_1);
                 output: t~ = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i, maxout over pairs, p = softmax(W_o t)
                 mode="encdec": the same decoder with a FIXED context (no attention): the RNNencdec baseline
  paper_init_    Appendix B.1: recurrent matrices orthogonal; W_a, U_a ~ N(0, 0.001^2); v_a and biases 0;
                 everything else ~ N(0, 0.01^2)
  beam_search, bleu   decoding and evaluation
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


class GRU(nn.Module):
    def __init__(self, m, n, ctx=0):
        super().__init__()
        self.W = nn.Linear(m, 3 * n, bias=False)            # W, W_z, W_r on the embedding
        self.U_zr = nn.Linear(n, 2 * n, bias=False)
        self.U = nn.Linear(n, n, bias=False)
        self.C = nn.Linear(ctx, 3 * n, bias=False) if ctx else None    # C, C_z, C_r on the context (decoder only)

    def forward(self, e, s, c=None):
        we, wz, wr = self.W(e).chunk(3, -1)
        cc, cz, cr = self.C(c).chunk(3, -1) if self.C is not None else (0, 0, 0)
        uz, ur = self.U_zr(s).chunk(2, -1)
        z = torch.sigmoid(wz + uz + cz)
        r = torch.sigmoid(wr + ur + cr)
        s_tilde = torch.tanh(we + self.U(r * s) + cc)
        return (1 - z) * s + z * s_tilde


class RNNsearch(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, m=620, n=1000, l=500, n_align=1000, mode="search"):
        super().__init__()
        self.n, self.mode = n, mode
        self.E_src = nn.Embedding(src_vocab, m)              # shared by the forward and backward RNNs
        self.E_tgt = nn.Embedding(tgt_vocab, m)
        self.fwd, self.bwd = GRU(m, n), GRU(m, n)
        self.W_s = nn.Linear(n, n)                           # s_0 = tanh(W_s backward-h_1)
        self.W_a = nn.Linear(n, n_align, bias=False)
        self.U_a = nn.Linear(2 * n, n_align, bias=False)
        self.v_a = nn.Linear(n_align, 1, bias=False)
        self.dec = GRU(m, n, ctx=2 * n)
        self.U_o, self.V_o, self.C_o = nn.Linear(n, 2 * l), nn.Linear(m, 2 * l, bias=False), nn.Linear(2 * n, 2 * l, bias=False)
        self.W_o = nn.Linear(l, tgt_vocab)

    # ---- encoder ----
    def encode(self, src):
        """src: (Tx, N) ids, PAD at the end. Returns annotations (Tx, N, 2n), mask (Tx, N), s_0 (N, n)."""
        mask = src != PAD
        e = self.E_src(src)
        N = src.shape[1]
        h, fw = torch.zeros(N, self.n, device=src.device), []
        for t in range(len(src)):
            h_new = self.fwd(e[t], h)
            h = torch.where(mask[t, :, None], h_new, h)
            fw.append(h)
        h, bw = torch.zeros(N, self.n, device=src.device), [None] * len(src)
        for t in reversed(range(len(src))):                  # padded steps (at the end) keep h = 0
            h = torch.where(mask[t, :, None], self.bwd(e[t], h), h)
            bw[t] = h
        ann = torch.cat([torch.stack(fw), torch.stack(bw)], -1)
        s0 = torch.tanh(self.W_s(torch.stack(bw)[0]))
        return ann, mask, s0

    # ---- attention ----
    def attend(self, s_prev, ann, Ua_h, mask):
        """e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j); alpha = softmax over valid j; c = sum alpha h."""
        e = self.v_a(torch.tanh(self.W_a(s_prev)[None] + Ua_h))[..., 0]          # (Tx, N)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(e, 0)
        return (alpha[..., None] * ann).sum(0), alpha

    def fixed_context(self, ann, mask):
        """RNNencdec: one context for every step: [forward h_Tx ; backward h_1] (the two full summaries)."""
        last = mask.sum(0) - 1
        fw_last = ann[last, torch.arange(ann.shape[1]), :self.n]
        return torch.cat([fw_last, ann[0, :, self.n:]], -1)

    def step(self, y_prev, s_prev, ann, Ua_h, mask, c_fixed=None):
        """One decoder step -> (log-probs (N, V), s_i, alpha (Tx, N) or None)."""
        if self.mode == "search":
            c, alpha = self.attend(s_prev, ann, Ua_h, mask)
        else:
            c, alpha = c_fixed, None
        e = self.E_tgt(y_prev)
        t = self.U_o(s_prev) + self.V_o(e) + self.C_o(c)                         # t~ (uses s_{i-1}, as written)
        t = t.view(*t.shape[:-1], -1, 2).max(-1).values                           # maxout
        s = self.dec(e, s_prev, c)
        return F.log_softmax(self.W_o(t), -1), s, alpha

    def forward(self, src, tgt_in, return_alpha=False):
        ann, mask, s = self.encode(src)
        Ua_h = self.U_a(ann)                                                      # precomputed once
        c_fixed = self.fixed_context(ann, mask) if self.mode != "search" else None
        outs, alphas = [], []
        for y in tgt_in:
            lp, s, a = self.step(y, s, ann, Ua_h, mask, c_fixed)
            outs.append(lp); alphas.append(a)
        lp = torch.stack(outs)
        return (lp, torch.stack(alphas) if self.mode == "search" else None) if return_alpha else lp

    def log_prob(self, src, tgt):
        tgt_in = torch.cat([torch.full_like(tgt[:1], BOS), tgt[:-1]])
        lp = self(src, tgt_in).gather(-1, tgt[..., None])[..., 0]
        return (lp * (tgt != PAD)).sum(0)


def orthogonal_(W):
    with torch.no_grad():
        q, _ = torch.linalg.qr(torch.randn(max(W.shape), max(W.shape)))
        W.copy_(q[:W.shape[0], :W.shape[1]])
    return W


def paper_init_(model):
    """Appendix B.1."""
    for name, p in model.named_parameters():
        if "bias" in name:
            nn.init.zeros_(p)
        else:
            nn.init.normal_(p, 0, 0.01)
    for gru in (model.fwd, model.bwd, model.dec):
        for W in (gru.U.weight, gru.U_zr.weight[:model.n], gru.U_zr.weight[model.n:]):
            orthogonal_(W)
    nn.init.normal_(model.W_a.weight, 0, 0.001)
    nn.init.normal_(model.U_a.weight, 0, 0.001)
    nn.init.zeros_(model.v_a.weight)                                              # => uniform attention at the start
    return model


# ---------------------------------------------------------------------------
# Decoding and evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(model, src, beam=12, max_len=60):
    """One source sentence (Tx, 1). Returns [(log_prob, tokens, alignments)] best first."""
    ann, mask, s = model.encode(src)
    Ua_h = model.U_a(ann)
    c_fixed = model.fixed_context(ann, mask) if model.mode != "search" else None
    beams, finished = [(0.0, [BOS], s, [])], []
    for _ in range(max_len):
        if not beams:
            break
        B = len(beams)
        y = torch.tensor([b[1][-1] for b in beams])
        S = torch.cat([b[2] for b in beams])
        cf = c_fixed.expand(B, -1) if c_fixed is not None else None
        lp, S2, alpha = model.step(y, S, ann.expand(-1, B, -1), Ua_h.expand(-1, B, -1), mask.expand(-1, B), cf)
        scores = torch.tensor([b[0] for b in beams])[:, None] + lp
        top = scores.flatten().topk(min(beam, scores.numel()))
        new = []
        for sc, idx in zip(top.values.tolist(), top.indices.tolist()):
            b, w = divmod(idx, lp.shape[1])
            al = beams[b][3] + ([alpha[:, b].tolist()] if alpha is not None else [])
            if w == EOS:
                finished.append((sc, beams[b][1][1:] + [EOS], al))
            else:
                new.append((sc, beams[b][1] + [w], S2[b:b + 1], al))
        beams = new[:beam]
        if len(finished) >= beam:
            break
    if not finished:
        finished = [(b[0], b[1][1:], b[3]) for b in beams]
    return sorted(finished, key=lambda x: -x[0])


def bleu(hypotheses, references, max_n=4):
    """Corpus BLEU-4 (Papineni et al. 2002)."""
    clipped, totals, hl, rl = [0] * max_n, [0] * max_n, 0, 0
    for hyp, ref in zip(hypotheses, references):
        hl += len(hyp); rl += len(ref)
        for n in range(1, max_n + 1):
            h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            clipped[n - 1] += sum(min(c, r[g]) for g, c in h.items())
            totals[n - 1] += max(len(hyp) - n + 1, 0)
    if min(totals) == 0 or min(clipped) == 0:
        return 0.0
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    return 100 * bp * math.exp(sum(math.log(c / t) for c, t in zip(clipped, totals)) / max_n)
