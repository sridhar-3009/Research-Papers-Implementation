"""SmoothQuant (Xiao et al. 2023): accurate INT8 weight + INT8 activation (W8A8) post-training quantization of LLMs by
migrating the quantization difficulty from activations to weights.

  quantizer        X_int = round(X / Delta),   Delta = max|X| / (2^(N-1) - 1)               (Eq. 1, symmetric)
  granularity      per-tensor (one Delta), per-token (one per row of X), per-channel (one per column of X / row of W)
  the problem      LLM activations have a few channels ~100x larger than the rest, in FIXED channels; per-tensor or
                   per-token scales are set by those outliers, so normal channels keep only a few quantization levels
  the trick        Y = (X diag(s)^-1) (diag(s) W): divide activation channel j by s_j and multiply weight row j by s_j
                   s_j = max|X_j|^alpha / max|W_j|^(1-alpha)                                (Eq. 4, alpha = 0.5)
                   the division is folded into the preceding normalisation, so it costs nothing at run time

This file builds on paper 073's pre-trained tiny LLaMA (COPY / REVERSE / SORT of digits) and:
  * gives it LLM-like activation outliers: a few RMSNorm gain entries are multiplied by 150, the matching input
    columns of the next linear layers are divided by only sqrt(150), and the model is briefly re-trained -- so a few
    FIXED channels are ~100x larger than the rest while the weights stay fairly flat, as in real LLMs;
  * simulates (fake-)quantization of every linear layer's weights and input activations at any bit width and
    granularity, static (calibrated) or dynamic;
  * applies SmoothQuant (calibration, smoothing factors, folding into the norm gains and weights);
  * compares with an LLM.int8()-style mixed-precision decomposition (outlier channels kept in floating point).
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_spec = importlib.util.spec_from_file_location(
    "lora073", Path(__file__).resolve().parent.parent / "073-Hu-et-al-2022-LoRA" / "lora.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

# ----------------------------------------------------------------------------------------------- quantizers

def quantize(x, bits=8, dim=None, scale=None):
    """Symmetric uniform fake-quantization. dim=None: per-tensor; dim=-1: one scale per row (per-token for X,
    per-output-channel for W stored as out x in); dim=0: one scale per column (per-channel for X).
    scale: a pre-computed (static) step size. Returns the de-quantized tensor."""
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        m = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
        scale = m.clamp(min=1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def effective_levels(X, bits=8):
    """Section 3: with one per-tensor scale, channel j only uses 2^bits * max|X_j| / max|X| levels."""
    m = X.abs().amax(0)
    return (2 ** bits) * m / m.max()


class QuantLinear(nn.Module):
    """Wraps a Linear: fake-quantizes its weight (per-tensor or per-output-channel) and its input activation
    (per-tensor / per-token / per-channel; static with a calibrated scale or dynamic), or keeps outlier input channels
    in floating point (LLM.int8()-style decomposition, threshold `outlier`)."""

    def __init__(self, lin, w_bits=8, a_bits=8, w_gran="tensor", a_gran="tensor", static_scale=None, outlier=None):
        super().__init__()
        self.lin, self.a_bits, self.a_gran, self.static, self.outlier = lin, a_bits, a_gran, static_scale, outlier
        W = lin.weight.data
        self.Wq = W if w_bits is None else quantize(W, w_bits, None if w_gran == "tensor" else -1)

    def forward(self, x):
        if self.a_bits is None:
            xq = x
        elif self.outlier is not None:                                     # mixed precision decomposition
            big = (x.abs().amax(dim=tuple(range(x.dim() - 1))) > self.outlier)
            xq = torch.where(big, x, quantize(x * ~big, self.a_bits, -1))
        elif self.static is not None:
            xq = quantize(x, self.a_bits, scale=self.static)
        else:
            dim = {"tensor": None, "token": -1, "channel": 0}[self.a_gran]
            xq = quantize(x.reshape(-1, x.shape[-1]), self.a_bits, dim).reshape(x.shape)
        return xq @ self.Wq.T


LINEARS = (("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("attn", "wo"), ("ffn", "w1"), ("ffn", "w2"), ("ffn", "w3"))
# the linear layers whose inputs come straight from a normalisation (where outliers live and where s can be folded)
NORM_GROUPS = (("attn_norm", "attn", ("wq", "wk", "wv")), ("ffn_norm", "ffn", ("w1", "w3")))


def quantized_copy(model, calib=None, **kw):
    """Replace every block linear with a QuantLinear. For static activation quantization, `calib` (a batch of
    sequences) is run once to record each layer's max |input|."""
    m = copy.deepcopy(model)
    static = {}
    if kw.get("a_gran") == "static":
        stats = record_input_absmax(m, calib, per_channel=False)
        kw = dict(kw, a_gran="tensor")
        static = {k: v / (2 ** (kw.get("a_bits", 8) - 1) - 1) for k, v in stats.items()}
    for i, b in enumerate(m.blocks):
        for sub, name in LINEARS:
            mod = getattr(b, sub)
            setattr(mod, name, QuantLinear(getattr(mod, name), static_scale=static.get((i, sub, name)), **kw))
    return m

# ----------------------------------------------------------------------------------------------- calibration and smoothing

@torch.no_grad()
def record_input_absmax(model, calib, per_channel=True):
    """Run calibration data and record max |input| of every block linear (per input channel or per tensor)."""
    stats, hooks = {}, []
    for i, b in enumerate(model.blocks):
        for sub, name in LINEARS:
            lin = getattr(getattr(b, sub), name)

            def hook(mod, inp, out, key=(i, sub, name)):
                x = inp[0].reshape(-1, inp[0].shape[-1]).abs()
                v = x.amax(0) if per_channel else x.amax()
                stats[key] = torch.maximum(stats[key], v) if key in stats else v
            hooks.append(lin.register_forward_hook(hook))
    model(calib)
    for h in hooks:
        h.remove()
    return stats


def inject_outliers(model, rng, channels=(3, 17, 41), factor=150.0, compensate=0.5, steps=300, lr=1e-3):
    """Give the model LLM-like outlier channels: multiply a few RMSNorm gains by `factor` and divide the matching input
    columns of the following linear layers by only factor**compensate, so those channels carry a genuinely LARGE
    activation x weight product (as in real LLMs, whose weights stay fairly flat while a few activation channels are
    huge). The function changes, so the model is then re-trained briefly (all weights) to recover its accuracy."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for b in m.blocks:
            for norm, sub, names in NORM_GROUPS:
                getattr(b, norm).weight[list(channels)] *= factor
                for n in names:
                    getattr(getattr(b, sub), n).weight[:, list(channels)] /= factor ** compensate
    return L.pretrain(m, rng, steps=steps, lr=lr) if steps else m


@torch.no_grad()
def smooth(model, calib, alpha=0.5):
    """SmoothQuant: s_j = max|X_j|^alpha / max|W_j|^(1-alpha) per input channel j of each norm -> linear group (the
    max over the group's weights), then X / s is folded into the norm gain and s * W into the weights."""
    m = copy.deepcopy(model)
    stats = record_input_absmax(m, calib)
    for i, b in enumerate(m.blocks):
        for norm, sub, names in NORM_GROUPS:
            ax = stats[(i, sub, names[0])].clamp(min=1e-5)                 # all names read the same normalised input
            aw = torch.stack([getattr(getattr(b, sub), n).weight.abs().amax(0) for n in names]).amax(0).clamp(min=1e-5)
            s = (ax ** alpha / aw ** (1 - alpha)).clamp(min=1e-5)
            getattr(b, norm).weight.div_(s)
            for n in names:
                getattr(getattr(b, sub), n).weight.mul_(s)
    return m


@torch.no_grad()
def layer_error(model, qmodel, x, key=(0, "attn", "wq")):
    """Relative error of one linear layer's output under quantization (same input)."""
    stats = {}

    def grab(name):
        def hook(mod, inp, out):
            stats[name] = (inp[0], out)
        return hook
    i, sub, name = key
    h1 = getattr(getattr(model.blocks[i], sub), name).register_forward_hook(grab("fp"))
    h2 = getattr(getattr(qmodel.blocks[i], sub), name).register_forward_hook(grab("q"))
    model(x)
    qmodel(x)
    h1.remove()
    h2.remove()
    ref = stats["fp"][1]
    out_q = getattr(getattr(qmodel.blocks[i], sub), name)(stats["fp"][0])
    return float((out_q - ref).norm() / ref.norm())


def task_accuracy(model, tasks=("copy", "reverse", "sort")):
    return float(np.mean([L.accuracy(model, t) for t in tasks]))


def calibration_batch(n=128, seed=1):
    rng = np.random.default_rng(seed)
    return torch.cat([L.make_batch(t, n // 3, rng) for t in ("copy", "reverse", "sort")])


REPORTED = {
    "Table 1 (OPT avg acc, 6.7B / 13B / 30B / 66B / 175B)": "FP16 64.9 / 65.6 / 67.9 / 69.5 / 71.6; INT8 per-tensor "
        "39.9 / 33.0 / 32.8 / 33.1 / 32.3; per-token 42.5 / 33.0 / 33.1 / 32.9 / 31.7; per-channel 64.8 / 65.6 / 68.0 / "
        "69.4 / 71.4 (but per-channel activation scales do not fit INT8 matrix-multiply kernels)",
    "Table 3 (OPT-175B, 7-task average / WikiText ppl)": "FP16 66.9% / 10.99; W8A8 35.5% / 93080; ZeroQuant 35.8%; "
        "LLM.int8() 66.7% / 11.10; Outlier Suppression 36.0%; SmoothQuant O1 66.5%, O2 66.4%, O3 66.8% / 11.17",
    "outliers": "~100x larger than most activations, a few fixed channels (OPT-13B: > 70)",
    "alpha": "0.5 for OPT and BLOOM; 0.75 for GLM-130B (~30% outliers); calibration on 512 Pile sentences",
    "efficiency": "up to 1.56x speedup and 2x memory reduction (FasterTransformer); 530B model on one 8-GPU node",
}
