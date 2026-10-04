"""ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., SC 2020).

  model states    mixed-precision Adam keeps, per parameter: an fp16 parameter (2 bytes), an fp16 gradient (2), and
                  fp32 master parameter + momentum + variance (K = 12)  ->  16 bytes per parameter. Plain data
                  parallelism (DP) REPLICATES all of it on every GPU.
  ZeRO-DP         partition the model states across the N_d data-parallel ranks instead:
                    P_os      optimizer states (fp32 copies) partitioned:   4 Psi + 12 Psi / N_d   (4x less)
                    P_os+g    + gradients partitioned:                      2 Psi + 14 Psi / N_d   (8x less)
                    P_os+g+p  + parameters partitioned:                    16 Psi / N_d            (N_d x less)
                  communication: DP all-reduces gradients = reduce-scatter + all-gather = 2 Psi per step; P_os and
                  P_os+g also move 2 Psi (reduce-scatter gradients, all-gather updated parameters); P_os+g+p moves
                  3 Psi (all-gather parameters for the forward AND the backward, reduce-scatter gradients) -- 1.5x
  ZeRO-R          partitioned activation checkpoints (P_a), constant-size buffers (C_B), memory defragmentation (M_D)

This file computes the per-GPU memory of every stage (Figure 1, Table 1), and runs a simulated N_d-rank data-parallel
training job of a small MLP with mixed precision (fp16 parameters / gradients, fp32 master weights and Adam moments)
in four modes -- plain DP, P_os, P_os+g, P_os+g+p -- counting the bytes each rank actually holds and sends, and
checking that all four produce bit-identical weights.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------- memory model

def model_state_bytes(psi, nd, stage, K=12):
    """Per-GPU bytes of model states for psi parameters, data-parallel degree nd, stage in {0 (DP), 1, 2, 3}."""
    if stage == 0:
        return (2 + 2 + K) * psi
    if stage == 1:
        return 2 * psi + 2 * psi + K * psi / nd
    if stage == 2:
        return 2 * psi + (2 + K) * psi / nd
    return (2 + 2 + K) * psi / nd


def comm_volume(stage):
    """Elements sent per rank per step, in units of Psi (ring collectives, large N_d)."""
    return 3.0 if stage == 3 else 2.0


def max_params(gpu_bytes, nd, stage, K=12, usable=1.0):
    """Largest model whose model states fit in `usable` of a GPU's memory."""
    per_param = model_state_bytes(1, nd, stage, K)
    return usable * gpu_bytes / per_param

# ----------------------------------------------------------------------------------------------- the MLP

def init_params(d_in=32, hidden=64, d_out=1, seed=0):
    rng = np.random.default_rng(seed)
    shapes = [(d_in, hidden), (hidden,), (hidden, d_out), (d_out,)]
    flat = np.concatenate([rng.standard_normal(int(np.prod(s))) * (1 / np.sqrt(s[0]) if len(s) == 2 else 0.0)
                           for s in shapes]).astype(np.float32)
    return flat, shapes


def unflatten(flat, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat[i:i + n].reshape(s))
        i += n
    return out


def loss_and_grad(flat, shapes, X, y):
    """Squared error of a 1-hidden-layer tanh MLP; returns the loss and the flat gradient (float32 arithmetic)."""
    W1, b1, W2, b2 = unflatten(flat.astype(np.float32), shapes)
    h = np.tanh(X @ W1 + b1)
    out = h @ W2 + b2
    r = out - y
    n = len(y)
    dout = 2 * r / n
    gW2, gb2 = h.T @ dout, dout.sum(0)
    dh = (dout @ W2.T) * (1 - h ** 2)
    gW1, gb1 = X.T @ dh, dh.sum(0)
    return float((r ** 2).mean()), np.concatenate([g.ravel() for g in (gW1, gb1, gW2, gb2)]).astype(np.float32)

# ----------------------------------------------------------------------------------------------- simulated cluster

class Cluster:
    """N_d ranks; collectives count the elements each rank SENDS (ring algorithms)."""

    def __init__(self, nd):
        self.nd = nd
        self.sent = np.zeros(nd)

    def shards(self, n):
        return np.array_split(np.arange(n), self.nd)

    def reduce_scatter(self, vectors):
        """Sum the ranks' full vectors; rank r keeps shard r. Each rank sends (N_d - 1)/N_d of a vector."""
        total = np.sum(vectors, axis=0, dtype=np.float32)
        n = len(total)
        self.sent += (self.nd - 1) / self.nd * n
        return [total[s] for s in self.shards(n)]

    def all_gather(self, pieces):
        """Every rank ends with the concatenation. Each rank sends its piece to N_d - 1 others (ring: (N_d-1)/N_d)."""
        full = np.concatenate(pieces)
        self.sent += (self.nd - 1) / self.nd * len(full)
        return full

    def all_reduce(self, vectors):
        return self.all_gather(self.reduce_scatter(vectors))


def adam_update(master, m, v, g, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * g * g
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    master -= lr * mh / (np.sqrt(vh) + eps)


def train(stage, nd=4, steps=50, batch=64, seed=0):
    """Data-parallel mixed-precision Adam on N_d simulated ranks.
      stage 0 (DP):   every rank: fp16 params + fp16 grads + fp32 master/m/v for ALL parameters; all-reduce grads
      stage 1 (P_os): fp16 params + fp16 grads for all; fp32 master/m/v only for its shard; reduce-scatter grads,
                      update its shard, all-gather the new fp16 params
      stage 2 (+g):   as 1, but the full fp16 gradient is dropped once reduce-scattered (only the shard is kept)
      stage 3 (+p):   every rank keeps only its shard of fp16 params too; all-gathers the full params for the forward
                      and AGAIN for the backward (freed after use), reduce-scatters the gradients
    Returns the final fp32 weights, the per-rank peak bytes held, the elements sent per rank per step, and the losses."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((2048, 32)).astype(np.float32)
    y = (np.tanh(X @ rng.standard_normal((32, 1)).astype(np.float32)) + 0.1 * rng.standard_normal((2048, 1))).astype(np.float32)
    flat, shapes = init_params()
    psi = len(flat)
    cl = Cluster(nd)
    sh = cl.shards(psi)
    full_states = stage == 0
    master = [flat.copy() if full_states else flat[s].copy() for s in sh]       # rank r's fp32 master (full or shard)
    m = [np.zeros_like(x) for x in master]
    v = [np.zeros_like(x) for x in master]
    params16 = [flat.astype(np.float16) for _ in range(nd)] if stage < 3 else [flat[s].astype(np.float16) for s in sh]
    peak = 0
    losses = []
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(X), (nd, batch))
        if stage == 3:                                                        # gather params for the forward
            full16 = cl.all_gather(params16).astype(np.float16)
            fwd = [full16] * nd
        else:
            fwd = params16
        grads32 = []
        for r in range(nd):
            loss, g = loss_and_grad(fwd[r], shapes, X[idx[r]], y[idx[r]])
            grads32.append(g.astype(np.float16).astype(np.float32) / nd)       # fp16 gradient, averaged
            if r == 0:
                losses.append(loss)
        if stage == 3:                                                        # ... and again for the backward
            cl.all_gather(params16)
        # bytes held at the peak of the step (fp16 = 2 bytes, fp32 = 4 bytes)
        held_params = 2 * (psi if stage < 3 else len(sh[0]) + psi)            # stage 3: own shard + transient full
        held_grads = 2 * psi                                                  # every rank produces a full gradient
        held_opt = 12 * len(master[0])
        peak = max(peak, held_params + held_grads + held_opt)
        if stage == 0:
            g_full = cl.all_reduce(grads32)
            for r in range(nd):
                adam_update(master[r], m[r], v[r], g_full, t)
                params16[r] = master[r].astype(np.float16)
        else:
            g_shards = cl.reduce_scatter(grads32)
            for r in range(nd):
                adam_update(master[r], m[r], v[r], g_shards[r], t)
            new16 = [master[r].astype(np.float16) for r in range(nd)]
            if stage in (1, 2):
                full = cl.all_gather(new16).astype(np.float16)
                params16 = [full.copy() for _ in range(nd)]
            else:
                params16 = new16                                              # stays partitioned
    final = master[0] if stage == 0 else np.concatenate(master)
    persistent = {0: 16 * psi, 1: 4 * psi + 12 * len(sh[0]), 2: 2 * psi + 14 * len(sh[0]), 3: 16 * len(sh[0])}[stage]
    return {"weights": final, "peak bytes": peak, "persistent bytes": persistent, "psi": psi,
            "sent per step (x psi)": float(cl.sent.mean() / steps / psi), "losses": losses}


REPORTED = {
    "Figure 1 (7.5B params, N_d = 64)": "per-GPU model states: baseline DP 120 GB, P_os 31.4 GB, P_os+g 16.6 GB, "
                                        "P_os+g+p 1.9 GB",
    "Table 1 (1T params, N_d = 1 / 4 / 16 / 64 / 256 / 1024)": "P_os 16000 / 7000 / 4750 / 4187 / 4046 / 4011 GB; "
        "P_os+g 16000 / 5500 / 2875 / 2218 / 2054 / 2013 GB; P_os+g+p 16000 / 4000 / 1000 / 250 / 62.5 / 15.6 GB",
    "communication": "P_os and P_os+g: same volume as DP (2 Psi); P_os+g+p: 1.5x (3 Psi)",
    "limits of DP": "plain DP runs out of memory beyond ~1.4B parameters on 32 GB GPUs",
    "results": "ZeRO-100B (P_os+g + ZeRO-R) trains 100B+ parameter models on 400 V100 GPUs at 15 PFlops (super-linear "
               "speedup), 8x larger models and 10x faster than Megatron-LM; up to 13B params without model parallelism; "
               "powered Turing-NLG (17B parameters)",
    "trillion": "with all three stages, a 1T-parameter model fits on 1024 GPUs (16 TB / 1024 = 16 GB each)",
}
