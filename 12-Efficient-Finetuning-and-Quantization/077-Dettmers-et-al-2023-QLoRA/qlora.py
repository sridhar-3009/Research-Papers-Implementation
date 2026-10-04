"""QLoRA (Dettmers et al. 2023): fine-tune a 4-bit quantized, FROZEN base model through LoRA adapters, so a 65B model
fits on one 48 GB GPU (vs > 780 GB for 16-bit full fine-tuning) with no loss in quality.

  storage vs compute   weights are STORED in 4-bit NormalFloat (NF4), de-quantized to 16-bit on the fly for each
                       matmul; gradients flow through them to the LoRA adapters, which are the only trained weights
                         Y = X doubleDequant(c1, c2, W_NF4) + X L1 L2                                         (Eq. 5)
  NF4                  16 values = (averaged) quantiles of N(0, 1), normalised to [-1, 1], with an exact zero;
                       each block of 64 weights is scaled by its absmax -- every level is used equally often for
                       normally distributed weights ("information-theoretically optimal")              (Eq. 4)
  double quantization  the per-block FP32 absmax constants (32/64 = 0.5 bits per parameter) are themselves quantized
                       to 8 bits in blocks of 256: 8/64 + 32/(64 * 256) = 0.127 bits per parameter
  paged optimizers     optimizer states in unified memory, paged to CPU RAM during memory spikes (not simulated)

This file builds 4-bit data types (Int4, FP4 E2M1, FP4 E3M0, NF4), block-wise absmax quantization, double
quantization, a frozen de-quantized base model with LoRA (re-using paper 073's LoRA and tiny LLaMA), and the 65B
memory budget.
"""

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

_spec = importlib.util.spec_from_file_location(
    "lora073", Path(__file__).resolve().parent.parent / "073-Hu-et-al-2022-LoRA" / "lora.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

# ----------------------------------------------------------------------------------------------- 4-bit data types

def nf4_values(offset=0.9677083, bits=4):
    """NormalFloat-k: 2^(k-1) positive and 2^(k-1) - 1 negative quantiles of N(0, 1) (asymmetric so that 0 is exact
    and all 2^k codes are used), normalised to [-1, 1]. `offset` is the outermost probability used (as in
    bitsandbytes), chosen so the extreme quantile is finite. bits=4 reproduces the paper's Appendix E values."""
    N = Normal(0.0, 1.0)
    half = 2 ** (bits - 1)
    pos = N.icdf(torch.linspace(offset, 0.5, half + 1)[:-1])                   # 8 positive values for 4 bits
    neg = -N.icdf(torch.linspace(offset, 0.5, half)[:-1])                      # 7 negative values
    v = torch.cat([pos, torch.zeros(1), neg]).sort().values
    return v / v.abs().max()


def int4_values(bits=4):
    """Symmetric absmax Int-k: -(2^(k-1) - 1) ... 2^(k-1) - 1 (one code unused), scaled to [-1, 1]."""
    q = 2 ** (bits - 1) - 1
    return torch.arange(-q, q + 1, dtype=torch.float32) / q


def fp4_values(exp_bits, man_bits, bias):
    """A 4-bit float (1 sign bit): all values of (1 + m / 2^man) * 2^(e - bias), with subnormals for e = 0,
    normalised to [-1, 1]."""
    vals = set()
    for e in range(2 ** exp_bits):
        for m in range(2 ** man_bits):
            v = (m / 2 ** man_bits) * 2 ** (1 - bias) if e == 0 else (1 + m / 2 ** man_bits) * 2 ** (e - bias)
            vals |= {v, -v}
    v = torch.tensor(sorted(vals), dtype=torch.float32)
    return v / v.abs().max()


DATA_TYPES = {"Int4": int4_values(), "FP4 (E2M1)": fp4_values(2, 1, 1), "FP4 (E3M0)": fp4_values(3, 0, 3),
              "NF4": nf4_values()}
DATA_TYPES_3BIT = {"Int3": int4_values(3), "FP3 (E2M0)": fp4_values(2, 0, 1), "NF3": nf4_values(bits=3)}


def quantize_blockwise(W, values, blocksize=64):
    """Block-wise absmax quantization: each block of `blocksize` consecutive weights is divided by its absmax c and
    snapped to the nearest code value. Returns (codes, constants c) -- the storage format."""
    flat = W.reshape(-1, blocksize)
    c = flat.abs().amax(1, keepdim=True).clamp(min=1e-12)
    codes = (flat / c).unsqueeze(-1).sub(values).abs().argmin(-1)
    return codes, c.squeeze(1)


def dequantize_blockwise(codes, c, values, shape):
    return (values[codes] * c[:, None]).reshape(shape)


def fake_quant(W, values, blocksize=64):
    return dequantize_blockwise(*quantize_blockwise(W, values, blocksize), values, W.shape)


def double_quantize(c, blocksize=256, bits=8):
    """Quantize the positive FP32 constants c: subtract their mean, then symmetric 8-bit absmax quantization in blocks
    of 256 (the paper uses an 8-bit float; a uniform 8-bit grid gives the same bit count). Returns c reconstructed."""
    mean = c.mean()
    x = c - mean
    pad = (-len(x)) % blocksize
    xb = torch.cat([x, torch.zeros(pad)]).reshape(-1, blocksize)
    c2 = xb.abs().amax(1, keepdim=True).clamp(min=1e-12)
    qmax = 2 ** (bits - 1) - 1
    q = torch.round(xb / c2 * qmax).clamp(-qmax, qmax)
    return (q / qmax * c2).flatten()[:len(c)] + mean


def bits_per_parameter(blocksize=64, dq=False, dq_blocksize=256):
    """Overhead of the quantization constants on top of the 4-bit codes."""
    return 4 + (8 / blocksize + 32 / (blocksize * dq_blocksize) if dq else 32 / blocksize)

# ----------------------------------------------------------------------------------------------- QLoRA model

@torch.no_grad()
def quantize_model(model, values, blocksize=64, dq=False):
    """Replace every block linear's weight by its 4-bit (de-quantized) version; the base stays frozen in QLoRA."""
    m = copy.deepcopy(model)
    for b in m.blocks:
        for mod in (b.attn, b.ffn):
            for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
                lin = getattr(mod, name, None)
                if lin is None:
                    continue
                codes, c = quantize_blockwise(lin.weight.data, values, blocksize)
                if dq:
                    c = double_quantize(c)
                lin.weight.data = dequantize_blockwise(codes, c, values, lin.weight.shape)
    return m


def add_lora_everywhere(model, r=8, attention_only=False):
    """QLoRA's recommendation: adapters on ALL linear layers of each block (the paper found q, v only falls short)."""
    targets = ("wq", "wv") if attention_only else ("wq", "wk", "wv", "wo")
    L.add_lora(model, targets, r)
    if not attention_only:
        for i, b in enumerate(model.blocks):
            for j, name in enumerate(("w1", "w2", "w3")):
                setattr(b.ffn, name, L.LoRALinear(getattr(b.ffn, name), r, seed=10_000 * r + 100 * i + 50 + j))
    return model


def memory_budget_gb(n_params, method, lora_frac=0.004, act_gb=6.0):
    """Rough training memory: weights + gradients + Adam states (2 x FP32) for the trained parameters + activations."""
    if method == "16-bit full fine-tuning":
        weights, trained = 2 * n_params, n_params
        return weights + trained * (2 + 8) + act_gb * 1e9
    if method == "16-bit LoRA":
        weights, trained = 2 * n_params, lora_frac * n_params
    else:                                                                      # QLoRA: NF4 + DQ constants
        weights, trained = n_params * bits_per_parameter(dq=True) / 8, lora_frac * n_params
    return weights + trained * (2 + 2 + 8) + act_gb * 1e9                      # LoRA weights, grads, Adam


REPORTED = {
    "headline": "65B model fine-tuned on one 48 GB GPU (16-bit full fine-tuning: > 780 GB) with no degradation",
    "Guanaco": "99.3% of ChatGPT on the Vicuna benchmark (GPT-4 judged), 24 hours on one GPU; Guanaco-65B uses 41 GB, "
               "33B 21 GB, 13B 10 GB",
    "Table 2 (mean Pile-CC perplexity, 125M-13B models)": "Int4 34.34, FP4 (E2M1) 31.07, FP4 (E3M0) 29.48, NF4 + DQ 27.41",
    "Table 4 (mean 5-shot MMLU, LLaMA 7-65B)": "BF16 53.0, FP4 52.2, NF4 + DQ 53.1 (NF4 matches 16-bit; FP4 ~1 point behind)",
    "double quantization": "0.5 -> 0.127 bits per parameter of constants (saves 0.373 bits, ~3 GB on 65B)",
    "LoRA placement": "LoRA on ALL linear layers is needed to match full fine-tuning; r then matters little",
    "NF4 values": "[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, "
                  "0.4407, 0.5626, 0.7230, 1.0] (Appendix E)",
}
