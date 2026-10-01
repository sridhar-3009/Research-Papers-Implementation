"""Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019): multi-query attention.

Multi-head attention (MHA) gives every head its own keys and values. Multi-query attention (MQA) keeps h query
heads but ONE shared key head and ONE shared value head: "remove the letter h from the einsum equations where it
represents the heads dimension of K, V, P_k or P_v". During incremental decoding the cached K and V tensors shrink
by a factor of h, and so does the memory traffic that dominates decoding time.

The functions below follow the paper's einsum code (Sections 2.2-3). We also include grouped-query attention
(Ainslie et al. 2023), the later generalisation: g key/value heads shared by h/g query heads each
(g = h is MHA, g = 1 is MQA).

Shapes: b batch, n query positions, m memory positions, d model width, h heads, k key size, v value size.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- the paper's code
def multihead_attention_batched(X, M, mask, P_q, P_k, P_v, P_o):
    """Section 2.3. X [b,n,d], M [b,m,d], mask [b,h,n,m] (0 or -inf), P_q/P_k [h,d,k], P_v/P_o [h,d,v]."""
    Q = torch.einsum("bnd,hdk->bhnk", X, P_q)
    K = torch.einsum("bmd,hdk->bhmk", M, P_k)
    V = torch.einsum("bmd,hdv->bhmv", M, P_v)
    logits = torch.einsum("bhnk,bhmk->bhnm", Q, K)              # (the 1/sqrt(k) can be folded into P_q)
    weights = torch.softmax(logits + mask, -1)
    O = torch.einsum("bhnm,bhmv->bhnv", weights, V)
    return torch.einsum("bhnv,hdv->bnd", O, P_o)


def multiquery_attention_batched(X, M, mask, P_q, P_k, P_v, P_o):
    """Section 3: identical, but P_k [d,k] and P_v [d,v] have no heads dimension."""
    Q = torch.einsum("bnd,hdk->bhnk", X, P_q)
    K = torch.einsum("bmd,dk->bmk", M, P_k)
    V = torch.einsum("bmd,dv->bmv", M, P_v)
    logits = torch.einsum("bhnk,bmk->bhnm", Q, K)
    weights = torch.softmax(logits + mask, -1)
    O = torch.einsum("bhnm,bmv->bhnv", weights, V)
    return torch.einsum("bhnv,hdv->bnd", O, P_o)


def multihead_self_attention_incremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    """Section 2.4: one decoding step. x [b,d], prev_K [b,h,m,k], prev_V [b,h,m,v]."""
    q = torch.einsum("bd,hdk->bhk", x, P_q)
    K = torch.cat([prev_K, torch.einsum("bd,hdk->bhk", x, P_k)[:, :, None]], 2)
    V = torch.cat([prev_V, torch.einsum("bd,hdv->bhv", x, P_v)[:, :, None]], 2)
    w = torch.softmax(torch.einsum("bhk,bhmk->bhm", q, K), -1)
    o = torch.einsum("bhm,bhmv->bhv", w, V)
    return torch.einsum("bhv,hdv->bd", o, P_o), K, V


def multiquery_self_attention_incremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    """Section 3: one decoding step with ONE cached key/value head. prev_K [b,m,k], prev_V [b,m,v]."""
    q = torch.einsum("bd,hdk->bhk", x, P_q)
    K = torch.cat([prev_K, torch.einsum("bd,dk->bk", x, P_k)[:, None]], 1)
    V = torch.cat([prev_V, torch.einsum("bd,dv->bv", x, P_v)[:, None]], 1)
    w = torch.softmax(torch.einsum("bhk,bmk->bhm", q, K), -1)
    o = torch.einsum("bhm,bmv->bhv", w, V)
    return torch.einsum("bhv,hdv->bd", o, P_o), K, V


# ---------------------------------------------------------------------------------------------------- cost model
def incremental_ratio(n, d, h, b, kind="mha"):
    """Sections 2.4.1 / 3.1 (with k = v = d/h, m = n): memory accessed / arithmetic over n decoding steps.
    MHA: Theta(n/d + 1/b).  MQA: Theta(1/d + n/(d h) + 1/b)."""
    if kind == "mha":
        return n / d + 1 / b
    return 1 / d + n / (d * h) + 1 / b


def kv_cache_numbers(b, n, h, k, layers, kv_heads):
    """How many numbers the K and V caches hold: 2 (K and V) * layers * b * n * kv_heads * k."""
    return 2 * layers * b * n * kv_heads * k


def attention_params(d, h, k, v, kv_heads):
    """P_q and P_o always have h heads; P_k, P_v have kv_heads (h for MHA, 1 for MQA)."""
    return d * h * k + d * kv_heads * k + d * kv_heads * v + d * h * v


def matching_ffn_width(d, h, k, d_ff, n_attention_layers, n_ffn_layers, kv_heads=1):
    """Section 4.1: widen every feed-forward layer so the MQA model has as many parameters as the MHA one."""
    saved = n_attention_layers * (attention_params(d, h, k, k, h) - attention_params(d, h, k, k, kv_heads))
    return d_ff + saved / (n_ffn_layers * 2 * d)


# ---------------------------------------------------------------------------------------------------- a decoder-only LM
class GroupedAttention(nn.Module):
    """Causal self-attention with h query heads and g key/value heads (g = h: MHA, g = 1: MQA), with an
    incremental mode that keeps a KV cache and, optionally, a local window (Section 4.1's 'local' models)."""

    def __init__(self, d, h, kv_heads, window=None):
        super().__init__()
        assert h % kv_heads == 0
        self.h, self.g, self.k, self.window = h, kv_heads, d // h, window
        self.W_q = nn.Linear(d, h * self.k, bias=False)
        self.W_k = nn.Linear(d, kv_heads * self.k, bias=False)
        self.W_v = nn.Linear(d, kv_heads * self.k, bias=False)
        self.W_o = nn.Linear(h * self.k, d, bias=False)

    def _attend(self, q, K, V, mask):
        """q [b,h,n,k]; K,V [b,g,m,k]; every group of h/g query heads shares one key/value head."""
        b, _, n, _ = q.shape
        q = q.view(b, self.g, self.h // self.g, n, self.k)
        logits = torch.einsum("bgrnk,bgmk->bgrnm", q, K) / math.sqrt(self.k)
        w = torch.softmax(logits.masked_fill(~mask, float("-inf")), -1)
        return torch.einsum("bgrnm,bgmk->bgrnk", w, V).reshape(b, self.h, n, self.k)

    def forward(self, x, cache=None):
        """x [b,n,d]. With cache = (K, V) of earlier positions, x holds only the NEW positions."""
        b, n, _ = x.shape
        q = self.W_q(x).view(b, n, self.h, self.k).transpose(1, 2)
        K = self.W_k(x).view(b, n, self.g, self.k).transpose(1, 2)
        V = self.W_v(x).view(b, n, self.g, self.k).transpose(1, 2)
        if cache is not None:
            K, V = torch.cat([cache[0], K], 2), torch.cat([cache[1], V], 2)
        m = K.shape[2]
        qpos = torch.arange(m - n, m)[:, None]
        kpos = torch.arange(m)[None]
        mask = kpos <= qpos
        if self.window is not None:
            mask &= kpos > qpos - self.window
        out = self._attend(q, K, V, mask)
        return self.W_o(out.transpose(1, 2).reshape(b, n, -1)), (K, V)


class Block(nn.Module):
    def __init__(self, d, h, kv_heads, d_ff, window=None):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = GroupedAttention(d, h, kv_heads, window)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))

    def forward(self, x, cache=None):
        a, cache = self.attn(self.ln1(x), cache)
        x = x + a
        return x + self.ffn(self.ln2(x)), cache


class DecoderLM(nn.Module):
    """A 'transformer-decoder' language model (Section 4.1's Billion-Word setup, scaled down). Pre-LN blocks,
    learned positions, tied input/output embeddings."""

    def __init__(self, vocab, d=128, h=8, kv_heads=8, d_ff=512, layers=2, max_len=512, window=None):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, kv_heads, d_ff, window) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x, caches=None, start=0):
        h = self.emb(x) + self.pos(torch.arange(start, start + x.shape[1]))
        new = []
        for i, blk in enumerate(self.blocks):
            h, c = blk(h, None if caches is None else caches[i])
            new.append(c)
        return self.ln(h) @ self.emb.weight.T, new

    @torch.no_grad()
    def generate(self, prompt, steps, incremental=True):
        """Greedy continuation. incremental=True reuses the KV cache (one new position per step);
        False recomputes the whole sequence every step (the slow, cache-free way)."""
        out = prompt
        logits, caches = self(prompt)
        for t in range(steps):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
            if incremental:
                logits, caches = self(nxt, caches, start=out.shape[1] - 1)
            else:
                logits, _ = self(out)
        return out

    def cache_numbers(self, b, n):
        a = self.blocks[0].attn
        return kv_cache_numbers(b, n, a.h, a.k, len(self.blocks), a.g)
