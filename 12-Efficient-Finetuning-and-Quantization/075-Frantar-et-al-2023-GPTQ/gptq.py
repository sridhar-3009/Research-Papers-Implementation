"""GPTQ (Frantar et al. 2023): accurate post-training 3-4 bit WEIGHT quantization of large transformers, one linear layer
at a time, using second-order (Hessian) information to compensate each rounding error with the not-yet-quantized
weights.

  layer-wise problem   argmin_Q || W X - Q X ||^2      (W: d_row x d_col weights, X: d_col x n calibration inputs)
  Hessian              H = 2 X X^T  (the same for every row of W -- it depends only on the inputs)
  OBQ step (Eq. 2)     quantize weight q, then push its error onto the remaining weights F:
                         delta_F = - (w_q - quant(w_q)) / [H_F^-1]_qq * (H_F^-1)_{:, q}
                       and remove q from H^-1 by one Gaussian-elimination step (Eq. 3)
  GPTQ's three changes 1. quantize all rows in the SAME fixed column order (so H^-1 is updated once per column,
                          not once per weight: O(d_row d_col^2) instead of O(d_row d_col^3));
                       2. lazy batch updates: process B = 128 columns at a time, update the rest of W once per block;
                       3. Cholesky form: all the rows of H_F^-1 that are ever needed are rows of the upper Cholesky
                          factor of H^-1 (plus 1% dampening of the diagonal) -- numerically stable

This file has round-to-nearest (RTN), greedy OBQ (exact, slow; for small matrices), a naive column-by-column GPTQ that
applies Eqs. 2-3 literally, GPTQ as in Algorithm 1 (Cholesky + lazy blocks + optional grouping), and sequential
quantization of paper 073's tiny LLaMA with calibration data.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "lora073", Path(__file__).resolve().parent.parent / "073-Hu-et-al-2022-LoRA" / "lora.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

# ----------------------------------------------------------------------------------------------- quantization grid

def grid(w, bits):
    """Asymmetric min-max grid per row: returns (scale, zero) so that q = clamp(round(w/scale) + zero, 0, 2^bits - 1)."""
    maxq = 2 ** bits - 1
    lo = torch.minimum(w.min(dim=-1, keepdim=True).values, torch.zeros(1, dtype=w.dtype))
    hi = torch.maximum(w.max(dim=-1, keepdim=True).values, torch.zeros(1, dtype=w.dtype))
    scale = ((hi - lo) / maxq).clamp(min=1e-9)
    zero = torch.round(-lo / scale)
    return scale, zero


def quant(w, scale, zero, bits):
    """Round onto the grid and map back to real values."""
    q = torch.clamp(torch.round(w / scale) + zero, 0, 2 ** bits - 1)
    return scale * (q - zero)


def rtn(W, bits, groupsize=None):
    """Round-to-nearest: each row (or each group of `groupsize` columns of each row) gets its own min-max grid."""
    if groupsize is None:
        return quant(W, *grid(W, bits), bits)
    Q = torch.empty_like(W)
    for s in range(0, W.shape[1], groupsize):
        blk = W[:, s:s + groupsize]
        Q[:, s:s + groupsize] = quant(blk, *grid(blk, bits), bits)
    return Q


def hessian(X, damp=0.01):
    """H = 2 X X^T plus `damp` x mean(diag) on the diagonal (the paper's 1% dampening)."""
    H = 2 * X @ X.T
    return H + damp * torch.diag(H).mean() * torch.eye(H.shape[0], dtype=H.dtype)


def layer_error(W, Q, X):
    """The layer-wise objective ||W X - Q X||^2."""
    return float(((W - Q) @ X).pow(2).sum())

# ----------------------------------------------------------------------------------------------- OBQ and GPTQ

def obq(W, H, bits):
    """Optimal Brain Quantization: every row on its own, always quantizing the weight with the smallest
    (quant(w) - w)^2 / [H_F^-1]_qq next, and updating the remaining weights (Eqs. 2-3). O(d_row * d_col^3)."""
    W = W.clone().double()
    scale, zero = grid(W, bits)
    Q = torch.zeros_like(W)
    Hinv0 = torch.linalg.inv(H.double())
    for r in range(W.shape[0]):
        w, Hinv = W[r].clone(), Hinv0.clone()
        left = list(range(W.shape[1]))
        while left:
            idx = torch.tensor(left)
            qv = quant(w[idx], scale[r], zero[r], bits)
            score = (qv - w[idx]) ** 2 / torch.diag(Hinv)[idx]
            k = int(score.argmin())
            q = left[k]
            Q[r, q] = qv[k]
            err = (w[q] - Q[r, q]) / Hinv[q, q]
            w = w - err * Hinv[:, q]                                   # Eq. 2: compensate with the remaining weights
            w[q] = Q[r, q]
            Hinv = Hinv - torch.outer(Hinv[:, q], Hinv[q, :]) / Hinv[q, q]   # Eq. 3: remove q from H^-1
            left.pop(k)
    return Q.float()


def gptq_naive(W, H, bits):
    """GPTQ step 1 only: the same left-to-right column order for all rows, Eqs. 2-3 applied literally (one H^-1
    update per column, shared by all rows)."""
    W = W.clone().double()
    scale, zero = grid(W, bits)
    Hinv = torch.linalg.inv(H.double())
    Q = torch.zeros_like(W)
    for j in range(W.shape[1]):
        Q[:, j] = quant(W[:, j:j + 1], scale, zero, bits)[:, 0]
        err = (W[:, j] - Q[:, j]) / Hinv[j, j]
        W -= torch.outer(err, Hinv[j, :])                                # columns < j are already done
        W[:, j] = Q[:, j]
        Hinv -= torch.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]
    return Q.float()


def gptq(W, H, bits, blocksize=128, groupsize=None):
    """Algorithm 1: Cholesky form of H^-1, lazy batches of `blocksize` columns, optional grouping (the grid of each
    group of columns is computed from the CURRENT, already error-compensated weights when the group starts)."""
    W = W.clone().double()
    d_col = W.shape[1]
    Hinv = torch.linalg.cholesky(torch.linalg.inv(H.double()), upper=True)   # rows = all H_F^-1 rows we need
    Q = torch.zeros_like(W)
    scale, zero = grid(W, bits)
    for i in range(0, d_col, blocksize):
        j_end = min(i + blocksize, d_col)
        E = torch.zeros(W.shape[0], j_end - i, dtype=W.dtype)
        for j in range(i, j_end):
            if groupsize is not None and j % groupsize == 0:
                scale, zero = grid(W[:, j:j + groupsize], bits)
            Q[:, j] = quant(W[:, j:j + 1], scale, zero, bits)[:, 0]
            E[:, j - i] = (W[:, j] - Q[:, j]) / Hinv[j, j]
            W[:, j:j_end] -= torch.outer(E[:, j - i], Hinv[j, j:j_end])      # update inside the block
        W[:, j_end:] -= E @ Hinv[i:j_end, j_end:]                           # lazy update of all later columns
    return Q.float()

# ----------------------------------------------------------------------------------------------- whole model

GROUPS = (("attn", ("wq", "wk", "wv")), ("attn", ("wo",)), ("ffn", ("w1", "w3")), ("ffn", ("w2",)))


@torch.no_grad()
def layer_inputs(model, calib, block, sub, name):
    """Inputs to one linear layer when the (partly quantized) model runs on the calibration data: (d_in, n)."""
    store = {}
    lin = getattr(getattr(model.blocks[block], sub), name)
    def hook(mod, inp, out):                                            # must return None (else it replaces the output)
        store["x"] = inp[0].reshape(-1, inp[0].shape[-1])
    h = lin.register_forward_hook(hook)
    model(calib)
    h.remove()
    return store["x"].T.double()


@torch.no_grad()
def quantize_model(model, calib, bits, method="gptq", groupsize=None, blocksize=128):
    """Quantize every linear layer in every block, in forward order. Each layer's Hessian is computed from inputs
    produced by the ALREADY-QUANTIZED earlier layers, as in the paper. Embeddings and the output head stay in FP."""
    m = copy.deepcopy(model)
    for b in range(len(m.blocks)):
        for sub, names in GROUPS:
            X = layer_inputs(m, calib, b, sub, names[0])                  # names in a group share the same input
            H = hessian(X)
            for n in names:
                lin = getattr(getattr(m.blocks[b], sub), n)
                W = lin.weight.data.double()
                Q = rtn(W, bits, groupsize) if method == "rtn" else gptq(W, H, bits, blocksize, groupsize)
                lin.weight.data = Q.float()
    return m


@torch.no_grad()
def model_loss(model, rng=None, n=600):
    """Mean next-token cross-entropy on the output digits (the toy's 'perplexity' is exp of this)."""
    rng = rng or np.random.default_rng(999)
    seq = torch.cat([L.make_batch(t, n // 3, rng) for t in ("copy", "reverse", "sort")])
    return float(L.seq_loss(model, seq))


def task_accuracy(model):
    return float(np.mean([L.accuracy(model, t) for t in ("copy", "reverse", "sort")]))


def calibration_batch(n=128, seed=1):
    rng = np.random.default_rng(seed)
    return torch.cat([L.make_batch(t, n // 3, rng) for t in ("copy", "reverse", "sort")])


REPORTED = {
    "OPT WikiText2 perplexity (Table 3), 125M / 1.3B / 13B / 175B": "FP16 27.65 / 14.63 / 10.13 / 8.34; 4-bit RTN "
        "37.28 / 48.17 / 11.32 / 10.54, GPTQ 31.12 / 15.47 / 10.31 / 8.37; 3-bit RTN 1.3e3 / 1.3e4 / 3.4e3 / 7.3e3, "
        "GPTQ 53.85 / 20.97 / 11.61 / 8.68",
    "175B models": "quantized in ~4 GPU hours (OPT-175B, BLOOM-176B); 4-bit within <= 0.25 perplexity of FP16; "
                   "OPT-175B runs on ONE 80 GB A100 at 3 bits",
    "speed": "custom kernels (memory-bound generation): ~3.25x on A100, ~4.5x on A6000 vs FP16",
    "implementation": "blocks of B = 128 columns, 1% dampening, Cholesky form, 128 random 2048-token C4 segments for "
                      "calibration; grouping g1024 / g128 improves 3-bit further; 2-bit and ternary possible with groups",
    "vs OBQ": "arbitrary fixed order is about as good as greedy order on large layers; >1000x faster",
}
