"""FlashAttention (Dao et al. 2022): EXACT attention computed tile by tile so that the N x N score matrix never goes
to slow GPU memory (HBM), only to fast on-chip SRAM. Same FLOPs (a bit more, from recomputation), far fewer memory
reads/writes -- and on GPUs, where attention is memory-bound, that is what decides the speed.

  standard (Alg. 0)   S = Q K^T -> HBM, P = softmax(S) -> HBM, O = P V -> HBM          Theta(N d + N^2) HBM accesses
  online softmax      for x = [x1, x2]:  m = max(m1, m2),  l = e^(m1-m) l1 + e^(m2-m) l2,
                      so softmax can be built one block of columns at a time, keeping only (m, l) per row
  FlashAttention      Alg. 1: blocks B_c = ceil(M / 4d), B_r = min(B_c, d); outer loop over K, V blocks, inner loop
                      over Q blocks; rescale the running output O_i with the new (m, l)   Theta(N^2 d^2 / M) accesses
  backward            store only O and the row statistics (m, l); RECOMPUTE the score blocks from Q, K, V (Alg. 4)
  block-sparse        skip zero blocks of a block mask: accesses scale with the fraction of non-zero blocks

"HBM" here is a counter: every element moved between the big arrays (Q, K, V, O, S, P, dO, ...) and the working
buffers ("SRAM") is counted, so the IO complexity can be measured exactly on a CPU.
"""

import math

import numpy as np
import torch


class HBM:
    """Counts elements read from / written to slow memory."""

    def __init__(self):
        self.reads = self.writes = 0

    def read(self, x):
        self.reads += x.numel()
        return x.clone()

    def write(self, dst, idx, x):
        self.writes += x.numel()
        dst[idx] = x

    @property
    def total(self):
        return self.reads + self.writes

# ----------------------------------------------------------------------------------------------- standard attention

def standard_attention(Q, K, V, hbm=None, scale=None):
    """Algorithm 0: materialise S and P in HBM (each step reads its inputs and writes its output)."""
    hbm = hbm or HBM()
    scale = scale if scale is not None else 1 / math.sqrt(Q.shape[1])
    N = Q.shape[0]
    S, P, O = torch.empty(N, N), torch.empty(N, N), torch.empty_like(Q)
    hbm.write(S, slice(None), hbm.read(Q) @ hbm.read(K).T * scale)          # read Q, K; write S
    s = hbm.read(S)                                                         # read S; write P
    hbm.write(P, slice(None), torch.softmax(s, -1))
    hbm.write(O, slice(None), hbm.read(P) @ hbm.read(V))                    # read P, V; write O
    return O, hbm

# ----------------------------------------------------------------------------------------------- online softmax

def online_softmax(x, block):
    """Softmax of a vector computed one block at a time with running (m, l), as in Section 3.1."""
    m, l = -math.inf, 0.0
    for s in range(0, len(x), block):
        xb = x[s:s + block]
        mb = float(xb.max())
        lb = float(torch.exp(xb - mb).sum())
        m_new = max(m, mb)
        l = math.exp(m - m_new) * l + math.exp(mb - m_new) * lb
        m = m_new
    return torch.exp(x - m) / l, m, l

# ----------------------------------------------------------------------------------------------- FlashAttention

def block_sizes(M, d):
    Bc = math.ceil(M / (4 * d))
    return Bc, min(Bc, d)


def flash_attention(Q, K, V, M, hbm=None, scale=None, block_mask=None):
    """Algorithm 1 (forward). M = SRAM size in elements. block_mask[i, j] = False skips the (Q_i, K_j) block
    (block-sparse FlashAttention, Algorithm 5). Returns O, the row statistics (m, l) and the counter."""
    hbm = hbm or HBM()
    scale = scale if scale is not None else 1 / math.sqrt(Q.shape[1])
    N, d = Q.shape
    Bc, Br = block_sizes(M, d)
    O, l, m = torch.zeros(N, d), torch.zeros(N), torch.full((N,), -math.inf)
    for j, c0 in enumerate(range(0, N, Bc)):
        Kj, Vj = hbm.read(K[c0:c0 + Bc]), hbm.read(V[c0:c0 + Bc])                   # line 6
        for i, r0 in enumerate(range(0, N, Br)):
            if block_mask is not None and not block_mask[i, j]:
                continue
            rows = slice(r0, r0 + Br)
            Qi, Oi, li, mi = hbm.read(Q[rows]), hbm.read(O[rows]), hbm.read(l[rows]), hbm.read(m[rows])   # line 8
            Sij = Qi @ Kj.T * scale                                                  # line 9 (on chip)
            mij = Sij.max(1).values                                                  # line 10
            Pij = torch.exp(Sij - mij[:, None])
            lij = Pij.sum(1)
            m_new = torch.maximum(mi, mij)                                           # line 11
            a, b = torch.exp(mi - m_new), torch.exp(mij - m_new)
            l_new = a * li + b * lij
            hbm.write(O, rows, ((li * a)[:, None] * Oi + b[:, None] * (Pij @ Vj)) / l_new[:, None])   # line 12
            hbm.write(l, rows, l_new)                                                # line 13
            hbm.write(m, rows, m_new)
    return O, m, l, hbm


def flash_backward(Q, K, V, O, dO, m, l, M, hbm=None, scale=None):
    """Algorithm 4 (backward) without storing S or P: each score block is RECOMPUTED from Q, K and the saved row
    statistics; D_i = rowsum(dO_i * O_i) replaces the softmax Jacobian term."""
    hbm = hbm or HBM()
    scale = scale if scale is not None else 1 / math.sqrt(Q.shape[1])
    N, d = Q.shape
    Bc, Br = block_sizes(M, d)
    dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)
    D = (dO * O).sum(1)                                                              # computed once
    hbm.reads += 2 * dO.numel()
    hbm.writes += D.numel()
    for c0 in range(0, N, Bc):
        cols = slice(c0, c0 + Bc)
        Kj, Vj = hbm.read(K[cols]), hbm.read(V[cols])
        dKj, dVj = torch.zeros_like(Kj), torch.zeros_like(Vj)
        for r0 in range(0, N, Br):
            rows = slice(r0, r0 + Br)
            Qi, dOi, dQi = hbm.read(Q[rows]), hbm.read(dO[rows]), hbm.read(dQ[rows])
            mi, li, Di = hbm.read(m[rows]), hbm.read(l[rows]), hbm.read(D[rows])
            Sij = Qi @ Kj.T * scale                                                  # recomputation
            Pij = torch.exp(Sij - mi[:, None]) / li[:, None]
            dVj += Pij.T @ dOi
            dPij = dOi @ Vj.T
            dSij = Pij * (dPij - Di[:, None])
            hbm.write(dQ, rows, dQi + dSij @ Kj * scale)
            dKj += dSij.T @ Qi * scale
        hbm.write(dK, cols, dKj)
        hbm.write(dV, cols, dVj)
    return dQ, dK, dV, hbm


def standard_backward_io(N, d):
    """Algorithm 3's HBM traffic in elements: read P, dO, V, Q, K, write/read dP, dS, write dQ, dK, dV
    (the forward also stored S and P)."""
    return (N * N + 2 * N * d) + (2 * N * d + N * N) + (2 * N * N) + (2 * N * N) + (N * N + 2 * N * d) + (N * N + 2 * N * d)


REPORTED = {
    "Theorem 2": "standard attention Theta(N d + N^2) HBM accesses; FlashAttention Theta(N^2 d^2 / M); no exact "
                 "attention algorithm can do asymptotically better over all SRAM sizes (Proposition 3)",
    "Figure 2 (GPT-2 medium attention, A100)": "standard: 66.6 GFLOPs, 40.3 GB HBM traffic, 41.7 ms; FlashAttention: "
                                               "75.2 GFLOPs (recomputation), 4.4 GB, 7.3 ms",
    "A100 memory": "HBM 40-80 GB at 1.5-2.0 TB/s; SRAM 192 KB per SM x 108 SMs at ~19 TB/s",
    "training speed": "BERT-large (seq 512) 15% faster than the MLPerf 1.1 record; GPT-2 (seq 1K) 3x faster than "
                      "HuggingFace / Megatron; Long Range Arena (1K-4K) 2.4x",
    "longer context": "first better-than-chance Transformers on Path-X (seq 16K, 61.4%) and Path-256 (seq 64K, "
                      "block-sparse); GPT-2 with 4K context trains faster than Megatron with 1K and gets 0.7 lower ppl",
    "memory": "linear in sequence length (no N x N matrix): up to 20x less memory than standard attention",
}
