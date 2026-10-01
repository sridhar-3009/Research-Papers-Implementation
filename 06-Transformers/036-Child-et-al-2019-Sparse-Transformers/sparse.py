"""Generating Long Sequences with Sparse Transformers (Child, Gray, Radford & Sutskever, 2019).

Factorized self-attention (Section 4): instead of every position attending to all earlier positions (n^2 work),
p heads each attend to a subset A^(m)_i of size ~ n^(1/p), chosen so that ANY earlier position j can reach i in at
most p + 1 attention steps. For p = 2, with stride l ~ sqrt(n):

  strided:  A1_i = {j : i - l < j <= i}  (the previous l positions)      A2_i = {j <= i : (i - j) mod l = 0}
  fixed:    A1_i = {j <= i : floor(j/l) = floor(i/l)}  (the same block)   A2_i = {j <= i : j mod l >= l - c}
            (the last c positions of every block act as 'summary' cells; we add j = i so that no row is empty)

Sparse Transformer block (Section 5.2), pre-activation residuals:
  a(H) = dropout(attention(norm(H)));  b(H) = dropout(ff(norm(H + a(H))));  H <- H + a(H) + b(H)
  ff uses GELU (approximated as x * sigmoid(1.702 x)); W2 and the attention output W_p are initialised scaled by
  1/sqrt(2N); embeddings N(0, 0.125/sqrt(d)); output logits initialised to 0.
Attention and feed-forward blocks can be recomputed in the backward pass (Section 5.4: gradient checkpointing).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------------------------------- patterns
def causal(n):
    i, j = torch.arange(n)[:, None], torch.arange(n)[None]
    return j <= i


def strided_patterns(n, l):
    i, j = torch.arange(n)[:, None], torch.arange(n)[None]
    A1 = (j <= i) & (j > i - l)
    A2 = (j <= i) & ((i - j) % l == 0)
    return A1, A2


def fixed_patterns(n, l, c):
    i, j = torch.arange(n)[:, None], torch.arange(n)[None]
    A1 = (j <= i) & (j // l == i // l)
    A2 = ((j <= i) & (j % l >= l - c)) | (j == i)   # + itself: early positions would otherwise attend to nothing
    return A1, A2


def reachable_in_two(A_first, A_second):
    """Can every earlier position j reach i by attending through A_first then A_second (path j -> a -> i with
    j in A_first[a], a in A_second[i]), or directly? Returns the boolean (n, n) reachability."""
    two = (A_second.float() @ A_first.float()) > 0
    return two | A_first | A_second


def entries(mask):
    return int(mask.sum())


# ---------------------------------------------------------------------------------------------------- attention
def masked_attention(q, k, v, mask):
    """Dense computation with a sparsity mask (simple, but still O(n^2) memory). q,k,v (..., n, d)."""
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v


def block_local_attention(q, k, v, l):
    """The fixed pattern's A1 without the n^2 matrix: reshape into n/l blocks of length l and attend causally
    inside each block. Work O(n l)."""
    *B, n, d = q.shape
    r = n // l
    rs = lambda x: x.reshape(*B, r, l, d)
    return masked_attention(rs(q), rs(k), rs(v), causal(l)).reshape(*B, n, d)


def strided_column_attention(q, k, v, l):
    """The strided pattern's A2 ('every l-th earlier position') as attention along the COLUMNS of an (n/l) x l
    matrix: transpose, then attend causally down each column (Section 5.5). Work O(n * n/l)."""
    *B, n, d = q.shape
    r = n // l
    cols = lambda x: x.reshape(*B, r, l, d).transpose(-2, -3)               # (..., l, r, d)
    out = masked_attention(cols(q), cols(k), cols(v), causal(r))
    return out.transpose(-2, -3).reshape(*B, n, d)


# ---------------------------------------------------------------------------------------------------- the model
def gelu_approx(x):
    return x * torch.sigmoid(1.702 * x)


class SparseAttention(nn.Module):
    """attention(X) = W_p . attend(X, S) with heads that each use one pattern. mode:
      'dense'       every head: full causal attention
      'interleave'  this layer uses pattern (layer_index mod p) for all heads (Eq. 6)
      'merged'      every head uses the union of the patterns (Eq. 7)
      'multihead'   heads split across the patterns (Eq. 8)"""

    def __init__(self, d, heads, patterns, mode, layer_index, n_layers):
        super().__init__()
        self.h, self.dh, self.mode, self.r = heads, d // heads, mode, layer_index
        self.W_qkv = nn.Linear(d, 3 * d)
        self.W_p = nn.Linear(d, d)
        nn.init.normal_(self.W_p.weight, 0, 0.125 / math.sqrt(d) / math.sqrt(2 * n_layers))
        self.patterns = patterns

    def head_masks(self, n):
        P = [p[:n, :n] for p in self.patterns]
        if self.mode == "dense":
            return [causal(n)] * self.h
        if self.mode == "interleave":
            return [P[self.r % len(P)]] * self.h
        if self.mode == "merged":
            u = P[0]
            for p in P[1:]:
                u = u | p
            return [u] * self.h
        return [P[i % len(P)] for i in range(self.h)]

    def forward(self, x):
        N, n, d = x.shape
        q, k, v = self.W_qkv(x).view(N, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        mask = torch.stack(self.head_masks(n))                                      # (h, n, n)
        out = masked_attention(q, k, v, mask)
        return self.W_p(out.transpose(1, 2).reshape(N, n, d))


class ResBlock(nn.Module):
    def __init__(self, d, heads, patterns, mode, layer_index, n_layers, dropout=0.0, ff_mult=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SparseAttention(d, heads, patterns, mode, layer_index, n_layers)
        self.W1, self.W2 = nn.Linear(d, ff_mult * d), nn.Linear(ff_mult * d, d)
        nn.init.normal_(self.W2.weight, 0, 0.125 / math.sqrt(ff_mult * d) / math.sqrt(2 * n_layers))
        nn.init.zeros_(self.W1.bias); nn.init.zeros_(self.W2.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, H):
        a = self.drop(self.attn(self.n1(H)))                                        # Eq. 12
        b = self.drop(self.W2(gelu_approx(self.W1(self.n2(H + a)))))                # Eq. 13
        return a + b                                                                # Eq. 14: resblock(H)


class SparseTransformerLM(nn.Module):
    def __init__(self, vocab, n_ctx, d=128, layers=4, heads=2, pattern="strided", stride=None, c=None,
                 mode="multihead", dropout=0.0, recompute=False):
        """pattern: 'dense', 'strided' or 'fixed'. Positions use the paper's 2-D 'attention embedding': the row
        (i // stride) and column (i % stride) of each position in a matrix of width = stride (Section 5.3)."""
        super().__init__()
        l = stride or int(round(math.sqrt(n_ctx)))
        self.l, self.recompute = l, recompute
        if pattern == "dense":
            pats, mode = [causal(n_ctx)], "dense"
        elif pattern == "strided":
            pats = list(strided_patterns(n_ctx, l))
        else:
            pats = list(fixed_patterns(n_ctx, l, c or max(1, l // 4)))
        self.emb = nn.Embedding(vocab, d)
        self.row, self.col = nn.Embedding(n_ctx // l + 1, d), nn.Embedding(l, d)
        for e in (self.emb,):
            nn.init.normal_(e.weight, 0, 0.125 / math.sqrt(d))
        for e in (self.row, self.col):
            nn.init.normal_(e.weight, 0, 0.125 / math.sqrt(d * 2))                  # n_emb = 2 attention embeddings
        self.blocks = nn.ModuleList([ResBlock(d, heads, pats, mode, r, layers, dropout) for r in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        nn.init.zeros_(self.out.weight)                                             # 'output logits initialised to 0'

    def forward(self, x):
        n = x.shape[1]
        pos = torch.arange(n)
        H = self.emb(x) + self.row(pos // self.l) + self.col(pos % self.l)          # Eq. 15
        for blk in self.blocks:
            H = H + (checkpoint(blk, H, use_reentrant=False) if self.recompute and self.training else blk(H))
        return self.out(self.norm(H))                                               # Eq. 11
