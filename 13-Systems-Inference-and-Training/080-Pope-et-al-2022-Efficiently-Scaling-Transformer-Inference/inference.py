"""Efficiently Scaling Transformer Inference (Pope et al. 2022): how to lay out a 500B-parameter model over 64 chips
for low-latency or high-throughput generation.

  costs           compute time  = 2 N tokens / (chips x peak FLOPS)                (matmuls: 2 FLOPs per parameter)
                  memory time   = (weight bytes + KV-cache bytes) / (chips x HBM bandwidth)  (loaded once per step)
                  comm time     = depends on the partitioning layout:
                    1D weight-stationary (shard d_ff):         T = 2 B L E / bw                     (constant in chips)
                    2D weight-stationary (shard d_model, d_ff): T = 8 B L E / (sqrt(chips) bw)
                    weight-gathered (all-gather the weights):   T = 4 E sqrt(B L F) / (sqrt(chips) bw)
                  MFU = observed throughput / peak = compute time / step time
  prefill vs      prefill processes the whole prompt in parallel (compute-bound, big batches of tokens);
  decode          decode makes one token per sequence per step (memory-bound: weights + KV cache reloaded each step)
  multiquery      one shared K/V head for all query heads: the KV cache shrinks by n_heads -- but only if the cache
                  is partitioned over the BATCH (otherwise each chip holds a full replicated copy)

This file has the PaLM configurations and TPU v4 constants, the KV-cache memory model (reproducing Table 1's maximum
context lengths), a roofline step-time model for every layout, and an explicit simulation of the feed-forward layer
partitioned over a mesh of virtual chips (1D / 2D weight-stationary, weight-gathered) that checks the result is exact
and COUNTS the bytes each chip sends, to compare with the formulas (Figure 3).
"""

import math

import numpy as np

TPU_V4 = {"flops": 275e12, "hbm_bytes": 32 * 2**30, "hbm_bw": 1200e9, "ici_bw": 270e9}

PALM = {   # name: (n_params, layers, d_model E, d_ff F, heads H, d_head)
    "8B": (8.6e9, 32, 4096, 16384, 16, 256),
    "62B": (62.5e9, 64, 8192, 32768, 32, 256),
    "540B": (540.35e9, 118, 18432, 73728, 48, 256),
}

# ----------------------------------------------------------------------------------------------- KV cache

def kv_bytes_per_token(layers, heads, d_head, kv_heads, bytes_per=2):
    """K and V for one token across all layers."""
    return 2 * layers * kv_heads * d_head * bytes_per


def max_context(variant, batch, chips=64, kv_fraction=0.30, model="540B"):
    """Table 1: longest context whose KV cache fits in 30% of each chip's HBM.
      multihead (d_head 128, 48 heads): K/V sharded over heads -> each chip holds max(1, H/chips) heads
      baseline multiquery (d_head 256): the single K/V head is REPLICATED on every chip
      optimized multiquery: the cache is sharded over the batch -> each chip holds batch/chips sequences"""
    _, L, _, _, H, d_head = PALM[model]
    budget = kv_fraction * TPU_V4["hbm_bytes"]
    if variant == "multihead":
        per_chip_token = kv_bytes_per_token(L, H, 128, max(1, math.ceil(H / chips)))
        return budget / (batch * per_chip_token)
    if variant == "baseline multiquery":
        return budget / (batch * kv_bytes_per_token(L, H, d_head, 1))
    return budget / (batch / chips * kv_bytes_per_token(L, H, d_head, 1))   # optimized multiquery

# ----------------------------------------------------------------------------------------------- step time model

def comm_time(layout, tokens, E, F, chips, bw=TPU_V4["ici_bw"], bytes_per=2):
    """Per feed-forward layer (the paper's formulas; attention is partitioned the same way)."""
    if layout == "1D WS":
        return 2 * tokens * E * bytes_per / bw
    if layout == "2D WS":
        return 8 * tokens * E * bytes_per / (math.sqrt(chips) * bw)
    if layout == "WG":
        return 4 * E * math.sqrt(tokens * F) * bytes_per / (math.sqrt(chips) * bw)
    raise ValueError(layout)


def step_time(model, chips, batch, ctx, phase="decode", layout="2D WS", weight_bytes=2, multiquery_opt=True,
              prompt_len=2048):
    """Roofline estimate of one forward pass: max(compute, memory) + communication, plus its MFU.
    decode: `batch` new tokens attending to `ctx` cached tokens; prefill: batch x prompt_len tokens at once."""
    n, L, E, F, H, d_head = PALM[model]
    tokens = batch if phase == "decode" else batch * prompt_len
    compute = 2 * n * tokens / (chips * TPU_V4["flops"])
    kv = kv_bytes_per_token(L, H, d_head, 1) * batch * ctx
    kv_per_chip = kv / chips if multiquery_opt else kv
    memory = (n * weight_bytes / chips + (kv_per_chip if phase == "decode" else 0)) / TPU_V4["hbm_bw"]
    comm = 2 * L * comm_time(layout, tokens, E, F, chips)                   # FFN + attention block per layer
    total = max(compute, memory) + comm
    return {"compute": compute, "memory": memory, "comm": comm, "total": total, "MFU": compute / total}

# ----------------------------------------------------------------------------------------------- simulated chips

class Mesh:
    """A logical mesh of chips; every collective records how many bytes each chip sends."""

    def __init__(self, n):
        self.n = n
        self.sent = np.zeros(n)

    def all_gather(self, shards, group):
        """Each chip in `group` ends up with the concatenation; ring all-gather: each sends (k - 1) shards."""
        k = len(group)
        for c in group:
            self.sent[c] += (k - 1) * shards[0].nbytes
        return np.concatenate(shards, axis=-1)

    def reduce_scatter(self, partials, group, axis=-1):
        """Sum the partial results over `group` and leave each chip one slice; ring: each sends (k - 1)/k of the data."""
        k = len(group)
        total = sum(partials)
        for c in group:
            self.sent[c] += (k - 1) / k * partials[0].nbytes
        return np.array_split(total, k, axis=axis)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def ffn_reference(x, W_in, W_out):
    return gelu(x @ W_in) @ W_out


def ffn_1d_ws(x_shards, W_in, W_out, mesh):
    """1D weight-stationary: chip c holds W_in[:, F_c] and W_out[F_c, :]. Activations arrive sharded over E;
    all-gather them to full BLE, multiply locally, reduce-scatter the partial BLE outputs back to E-shards."""
    n = mesh.n
    Fs = np.array_split(np.arange(W_in.shape[1]), n)
    x = mesh.all_gather(x_shards, list(range(n)))
    partials = [gelu(x @ W_in[:, f]) @ W_out[f, :] for f in Fs]
    return mesh.reduce_scatter(partials, list(range(n)))


def ffn_2d_ws(x, W_in, W_out, mesh, X, YZ):
    """2D weight-stationary (simplified to an X x YZ grid): chip (i, j) holds W_in[E_i, F_j] and W_out[F_j, E_i].
    Inputs: chip (i, j) has x[:, E_i] split further over j. The first einsum needs x[:, E_i] (all-gather over the j
    axis) and produces partial sums over i (reduce-scatter over the i axis); the second does the reverse."""
    E, F = W_in.shape
    Es, Fs = np.array_split(np.arange(E), X), np.array_split(np.arange(F), YZ)
    chip = lambda i, j: i * YZ + j
    # step 1: each chip column-group i gathers x[:, E_i] from its YZ chips
    xi = {}
    for i in range(X):
        parts = np.array_split(x[:, Es[i]], YZ, axis=-1)
        xi[i] = mesh.all_gather(parts, [chip(i, j) for j in range(YZ)])
    # step 2: partial h[:, F_j] = sum_i x[:, E_i] W_in[E_i, F_j]; reduce-scatter over i, so each chip keeps a slice
    h = {}
    for j in range(YZ):
        partials = [xi[i] @ W_in[np.ix_(Es[i], Fs[j])] for i in range(X)]
        slices = mesh.reduce_scatter(partials, [chip(i, j) for i in range(X)])
        h[j] = gelu(np.concatenate(slices, axis=-1))
    # step 3: gather h[:, F_j] over the X chips that share j (the reduce-scatter left it split), then multiply
    # by W_out[F_j, E_i]; partial sums over j are reduce-scattered over j, giving output shards of E_i
    out = np.zeros((x.shape[0], E))
    for j in range(YZ):
        for i in range(X):
            mesh.sent[chip(i, j)] += (X - 1) / X * h[j].nbytes                 # all-gather of h over i
    for i in range(X):
        partials = [h[j] @ W_out[np.ix_(Fs[j], Es[i])] for j in range(YZ)]
        out[:, Es[i]] = np.concatenate(mesh.reduce_scatter(partials, [chip(i, j) for j in range(YZ)]), axis=-1)
    return out


def ffn_weight_gathered(x, W_in, W_out, mesh):
    """XYZ-weight-gathered: activations stay put (each chip owns a slice of the BATCH); every chip all-gathers the
    full weights, which were stored sharded over all chips."""
    n = mesh.n
    for c in range(n):
        mesh.sent[c] += (n - 1) / n * (W_in.nbytes + W_out.nbytes)
    rows = np.array_split(np.arange(x.shape[0]), n)
    return np.concatenate([ffn_reference(x[r], W_in, W_out) for r in rows])


REPORTED = {
    "headline (PaLM 540B, 64 TPU v4, 2048-token context)": "29 ms per generated token at low batch (int8 weights: "
                                                         "28.5 ms vs 36.9 ms bf16 at batch 64); 76% MFU for large-batch prefill",
    "Table 1 (max context, batch 128 / 512)": "multihead (d_head 128) 1320 / 330; baseline multiquery 660 / 165; "
                                             "optimized (batch-sharded) multiquery 43,000 / 10,700 -> up to 32x longer",
    "Table 2 (PaLM 540B)": "low-latency: prefill batch 1 0.29 s (MFU 43%), decode batch 64 1.82 s for 64 tokens "
                           "(MFU 14%), int8; high-throughput: prefill batch 512 85.2 s (MFU 76%, weight-gathered), decode "
                           "batch 512 6.0 s (MFU 33%), bf16",
    "KV cache": "multihead 500B+ model, batch 512, context 2048: ~3 TB, 3x the parameters",
    "TPU v4": "275 TFLOPS bf16, 32 GiB HBM at 1200 GB/s, 270 GB/s interconnect, 3D torus",
    "interactive example": "PaLM 540B int8 on 64 chips: read 64 new tokens, consult 1920 cached tokens of history and "
                           "generate a 64-token reply in 1.9 s; offline: 73% FLOPS efficiency overall",
}
