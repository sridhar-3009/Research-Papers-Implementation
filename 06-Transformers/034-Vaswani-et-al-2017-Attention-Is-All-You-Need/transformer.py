"""The Transformer (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, NIPS 2017), written
out piece by piece (no nn.Transformer).

  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V                                        (Eq. 1)
  MultiHead(Q, K, V) = Concat(head_1..head_h) W^O,  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
  FFN(x) = max(0, x W1 + b1) W2 + b2                                                         (Eq. 2)
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  Every sub-layer: LayerNorm(x + Dropout(Sublayer(x)))          ("post-LN", as in the paper)
  lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)                                   (Eq. 3)
  Label smoothing 0.1; beam 4 with length penalty alpha = 0.6; checkpoint averaging.

Shapes are batch-first: (N, T, d_model).
"""

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


# ---------------------------------------------------------------------------------------------------- attention
def attention(Q, K, V, mask=None):
    """Eq. 1. Q (..., Tq, d_k), K (..., Tk, d_k), V (..., Tk, d_v); mask (..., Tq, Tk) True where attending is
    ALLOWED. Returns (output, weights)."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, -1)
    return w @ V, w


class MultiHeadAttention(nn.Module):
    """h heads of size d_k = d_v = d_model / h (Section 3.2.2). One d_model x d_model matrix per projection holds
    all h heads' W_i side by side."""

    def __init__(self, d_model, h, dropout=0.0):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        self.W_Q, self.W_K, self.W_V, self.W_O = (nn.Linear(d_model, d_model) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.weights = None                                     # the last attention weights (for plots)

    def split(self, x):                                         # (N, T, d_model) -> (N, h, T, d_k)
        return x.view(x.shape[0], x.shape[1], self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        Q, K, V = self.split(self.W_Q(q)), self.split(self.W_K(k)), self.split(self.W_V(v))
        out, w = attention(Q, K, V, None if mask is None else mask[:, None])
        self.weights = w.detach()
        out = out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)       # Concat(head_1..head_h)
        return self.W_O(self.drop(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))                      # Eq. 2


def sinusoidal_positions(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float)[:, None]
    i = torch.arange(0, d_model, 2, dtype=torch.float)
    angle = pos / torch.pow(10000.0, i / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return pe


# ---------------------------------------------------------------------------------------------------- layers
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attn, self.ffn = MultiHeadAttention(d_model, h), FeedForward(d_model, d_ff)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.ln1(x + self.drop(self.attn(x, x, x, mask)))   # LayerNorm(x + Sublayer(x))
        return self.ln2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn, self.cross_attn = MultiHeadAttention(d_model, h), MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, self_mask, cross_mask):
        y = self.ln1(y + self.drop(self.self_attn(y, y, y, self_mask)))          # masked self-attention
        y = self.ln2(y + self.drop(self.cross_attn(y, memory, memory, cross_mask)))  # encoder-decoder attention
        return self.ln3(y + self.drop(self.ffn(y)))


def causal_mask(T, device=None):
    """True where position i may attend to position j, i.e. j <= i."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=512,
                 shared_vocab=True, positions="sinusoidal"):
        """shared_vocab: one embedding matrix for source, target and the pre-softmax layer (Section 3.4);
        needs src_vocab == tgt_vocab. positions: 'sinusoidal' or 'learned' (Table 3 row E)."""
        super().__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        if shared_vocab:
            assert src_vocab == tgt_vocab
            self.src_emb = self.tgt_emb
        else:
            self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        if positions == "sinusoidal":
            self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)
            self.pos = None
        else:
            self.pos = nn.Embedding(max_len, d_model)
        self.enc = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.dec = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.drop = nn.Dropout(dropout)
        self.out_bias = nn.Parameter(torch.zeros(tgt_vocab))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed(self, emb, x):
        """Embeddings times sqrt(d_model), plus positions, then dropout (Sections 3.4, 3.5, 5.4)."""
        P = self.pe[: x.shape[1]] if self.pos is None else self.pos(torch.arange(x.shape[1], device=x.device))
        return self.drop(emb(x) * math.sqrt(self.d_model) + P)

    def encode(self, src):
        mask = (src != PAD)[:, None, :]                         # (N, 1, S): don't attend to padding
        x = self.embed(self.src_emb, src)
        for layer in self.enc:
            x = layer(x, mask)
        return x, mask

    def decode(self, tgt_in, memory, src_mask):
        T = tgt_in.shape[1]
        self_mask = causal_mask(T, tgt_in.device)[None] & (tgt_in != PAD)[:, None, :]
        y = self.embed(self.tgt_emb, tgt_in)
        for layer in self.dec:
            y = layer(y, memory, self_mask, src_mask)
        return y @ self.tgt_emb.weight.T + self.out_bias        # pre-softmax layer shares the embedding matrix

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)


# ---------------------------------------------------------------------------------------------------- training
def noam_lr(step, d_model=512, warmup=4000):
    """Eq. 3: linear warm-up for `warmup` steps, then decay as 1/sqrt(step)."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def label_smoothed_loss(logits, target, eps=0.1, ignore=PAD):
    """Cross-entropy against (1 - eps) * one_hot + eps / K * uniform (Szegedy et al. 2016), averaged over the
    non-padding tokens."""
    lp = F.log_softmax(logits, -1)
    nll = -lp.gather(-1, target[..., None])[..., 0]
    smooth = -lp.mean(-1)
    loss = (1 - eps) * nll + eps * smooth
    keep = target != ignore
    return (loss * keep).sum() / keep.sum()


def average_checkpoints(state_dicts):
    """Section 6.1: average the parameters of the last few checkpoints."""
    out = collections.OrderedDict()
    for k in state_dicts[0]:
        out[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
    return out


def length_penalty(length, alpha=0.6):
    """GNMT's lp(Y) = ((5 + |Y|) / 6)^alpha; candidates are ranked by log p / lp."""
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search(model, src, beam=4, alpha=0.6, extra_len=50):
    """Beam search for ONE source sentence src (1, S). Max output length = input length + 50 (Section 6.1)."""
    memory, src_mask = model.encode(src)
    hyps, done = [(0.0, [BOS])], []
    for _ in range(src.shape[1] + extra_len):
        tgt = torch.tensor([h[1] for h in hyps], device=src.device)
        lp = F.log_softmax(model.decode(tgt, memory.expand(len(hyps), -1, -1), src_mask.expand(len(hyps), -1, -1))[:, -1], -1)
        cand = []
        for (score, seq), row in zip(hyps, lp):
            top = row.topk(min(beam, row.shape[-1]))
            cand += [(score + s, seq + [k]) for s, k in zip(top.values.tolist(), top.indices.tolist())]
        cand.sort(key=lambda c: -c[0])
        hyps = []
        for c in cand:
            (done if c[1][-1] == EOS else hyps).append(c)
            if len(hyps) == beam:
                break
        if not hyps:
            break
    pool = done or hyps
    best = max(pool, key=lambda c: c[0] / length_penalty(len(c[1]) - 1, alpha))
    seq = best[1][1:]
    return seq[:-1] if seq and seq[-1] == EOS else seq


# ---------------------------------------------------------------------------------------------------- byte-pair encoding
def learn_bpe(word_counts, n_merges):
    """Sennrich et al. (2016): start from characters (+ an end-of-word marker) and repeatedly merge the most
    frequent adjacent pair. Returns the list of merges, in order."""
    vocab = {tuple(w) + ("</w>",): c for w, c in word_counts.items()}
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for sym, c in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        new = {}
        for sym, c in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new[tuple(out)] = c
        vocab = new
    return merges


def apply_bpe(word, merges):
    """Split one word into subword units by applying the learned merges in order."""
    sym = list(word) + ["</w>"]
    for a, b in merges:
        i, out = 0, []
        while i < len(sym):
            if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(sym[i]); i += 1
        sym = out
    return sym


def count_params(model):
    return sum(p.numel() for p in model.parameters())
