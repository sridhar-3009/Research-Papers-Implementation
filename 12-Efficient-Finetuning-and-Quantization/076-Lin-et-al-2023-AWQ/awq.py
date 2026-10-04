"""AWQ: Activation-aware Weight Quantization (Lin et al. 2023/2024): low-bit WEIGHT-only quantization that protects the
~1% of weight channels that matter most -- the ones multiplied by LARGE ACTIVATIONS -- by scaling them up before
quantization (an equivalent transformation), instead of keeping them in FP16.

  observation 1   keeping 0.1-1% of input channels in FP16 rescues INT3 quantization, but only if they are chosen by
                  ACTIVATION magnitude (not by weight magnitude, not at random) -- Table 1
  observation 2   for one weight w and input x, Q(w s) (x / s) has error  Delta' * RoundErr(ws/Delta') * x / s; the
                  rounding error (~0.25) and Delta' (~Delta) barely change, so scaling a salient channel by s > 1 cuts
                  its relative error by ~1/s -- until s is so large that it changes the group maximum and hurts the
                  other channels (Table 2: best at s = 2)
  the method      s = s_X^alpha (s_X = mean |activation| per input channel), alpha grid-searched in [0, 1] to minimise
                  || Q(W diag(s)) (diag(s)^-1 X) - W X ||   (Eq. 4-5); 1/s is folded into the previous operator;
                  optional weight clipping search; no back-propagation or regression -> little over-fitting

This file re-uses paper 074's tiny LLaMA with LLM-like activation outliers (a few channels ~250x larger) and paper
075's min-max group quantizer and GPTQ, and implements: mixed precision (keep p% channels in FP), scaling salient
channels by a fixed s, the AWQ search (with the scale folded into norms / previous linear layers), clipping, and a
calibration-distribution test.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _here / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SQ = _load("sq074", "074-Xiao-et-al-2023-SmoothQuant/smoothquant.py")
G = _load("gptq075", "075-Frantar-et-al-2023-GPTQ/gptq.py")
L = SQ.L

# Each group: the linear layers sharing one input, and where that input's per-channel scale can be folded:
#   ("norm", name)      divide the RMSNorm gain
#   ("rows", sub, name) divide the OUTPUT rows of an earlier linear layer (its output IS this input, up to linear ops)
GROUPS = ((("attn", ("wq", "wk", "wv")), ("norm", "attn_norm")),
          (("attn", ("wo",)), ("rows", "attn", "wv")),             # attention output = softmax-weighted sums of V
          (("ffn", ("w1", "w3")), ("norm", "ffn_norm")),
          (("ffn", ("w2",)), ("rows", "ffn", "w3")))               # w2's input = silu(w1 x) * (w3 x)


def wq(W, bits, group):
    """Weight fake-quantization: asymmetric min-max per row, in groups of `group` input columns (075's RTN)."""
    return G.rtn(W, bits, group)


def group_input_stats(model, calib, b, sub, name):
    """Calibration inputs (n x d_in) of one linear layer of block b."""
    return G.layer_inputs(model, calib, b, sub, name).T.float()

# ----------------------------------------------------------------------------------------------- Table 1 / Table 2 tools

@torch.no_grad()
def quantize_keep_fp(model, calib, bits, group, frac, how):
    """Quantize every block linear, but keep a fraction `frac` of INPUT channels in floating point, chosen by
    activation magnitude ('act'), weight-column norm ('weight') or at random ('random')."""
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(0)
    for b in range(len(m.blocks)):
        for (sub, names), _ in GROUPS:
            X = group_input_stats(m, calib, b, sub, names[0])
            for n in names:
                lin = getattr(getattr(m.blocks[b], sub), n)
                W = lin.weight.data
                k = max(1, int(round(frac * W.shape[1])))
                if how == "act":
                    keep = X.abs().mean(0).topk(k).indices
                elif how == "weight":
                    keep = W.norm(dim=0).topk(k).indices
                else:
                    keep = torch.randperm(W.shape[1], generator=g)[:k]
                Q = wq(W, bits, group)
                Q[:, keep] = W[:, keep]
                lin.weight.data = Q
    return m


@torch.no_grad()
def scale_salient(model, calib, bits, group, s, frac=0.01):
    """Table 2: multiply the top-`frac` activation channels' weights by s (and divide their inputs by s) before
    quantizing. Returns the model and the share of groups whose quantization step Delta changed."""
    m = copy.deepcopy(model)
    changed, total = 0, 0
    for b in range(len(m.blocks)):
        for (sub, names), _ in GROUPS:
            X = group_input_stats(m, calib, b, sub, names[0])
            k = max(1, int(round(frac * X.shape[1])))
            scale = torch.ones(X.shape[1])
            scale[X.abs().mean(0).topk(k).indices] = s
            for n in names:
                lin = getattr(getattr(m.blocks[b], sub), n)
                W = lin.weight.data
                for c in range(0, W.shape[1], group):                       # how many group maxima change?
                    d0 = W[:, c:c + group].amax(1) - W[:, c:c + group].amin(1)
                    d1 = (W * scale)[:, c:c + group].amax(1) - (W * scale)[:, c:c + group].amin(1)
                    changed += int((~torch.isclose(d0, d1)).sum())
                    total += d0.numel()
                lin.weight.data = wq(W * scale, bits, group) / scale       # Q(w s) applied to x / s
    return m, changed / total

# ----------------------------------------------------------------------------------------------- AWQ

def awq_scale(W_list, X, bits, group, grid=20):
    """Search s = s_X^alpha over alpha in {0, 1/grid, ..., 1}, minimising sum over the group's layers of
    || Q(W diag(s)) diag(s)^-1 X^T - W X^T ||^2. s is normalised so that sqrt(max * min) = 1."""
    sx = X.abs().mean(0).clamp(min=1e-6)
    ref = [X @ W.T for W in W_list]
    best = (float("inf"), 0.0, torch.ones_like(sx))
    for i in range(grid + 1):
        a = i / grid
        s = sx ** a
        s = s / (s.max() * s.min()).sqrt()
        err = sum(float(((X / s) @ wq(W * s, bits, group).T - r).pow(2).sum()) for W, r in zip(W_list, ref))
        if err < best[0]:
            best = (err, a, s)
    return best[2], best[1], best[0]


def clip_search(W, X, bits, group, ratios=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)):
    """Weight clipping: for each row group, shrink the [min, max] range by a ratio and keep the ratio with the lowest
    output error on the calibration inputs."""
    W = W.clone()
    for c in range(0, W.shape[1], group):
        blk, x = W[:, c:c + group], X[:, c:c + group]
        best = None
        for r in ratios:
            lo, hi = blk.amin(1, keepdim=True) * r, blk.amax(1, keepdim=True) * r
            q = wq(blk.clamp(lo, hi), bits, None)
            e = ((x @ q.T - x @ blk.T) ** 2).sum(0)                          # per output row
            best = (e, blk.clamp(lo, hi)) if best is None else (
                torch.minimum(best[0], e), torch.where((e < best[0])[:, None], blk.clamp(lo, hi), best[1]))
        W[:, c:c + group] = best[1]
    return W


@torch.no_grad()
def awq_model(model, calib, bits, group, clip=False, grid=20):
    """Apply AWQ to every group of every block, in forward order (later groups see the already-quantized model),
    folding 1/s into the norm gain or into the output rows of the previous linear layer. Returns the model and the
    chosen alphas."""
    m = copy.deepcopy(model)
    alphas = []
    for b in range(len(m.blocks)):
        blk = m.blocks[b]
        for (sub, names), fold in GROUPS:
            X = group_input_stats(m, calib, b, sub, names[0])
            lins = [getattr(getattr(blk, sub), n) for n in names]
            s, a, _ = awq_scale([l.weight.data for l in lins], X, bits, group, grid)
            alphas.append(a)
            if fold[0] == "norm":
                getattr(blk, fold[1]).weight.div_(s)
            else:
                getattr(getattr(blk, fold[1]), fold[2]).weight.div_(s[:, None])
            for l in lins:
                W = l.weight.data * s
                if clip:
                    W = clip_search(W, X / s, bits, group)
                l.weight.data = wq(W, bits, group)
    return m, alphas


@torch.no_grad()
def rtn_model(model, bits, group):
    m = copy.deepcopy(model)
    for b in m.blocks:
        for (sub, names), _ in GROUPS:
            for n in names:
                lin = getattr(getattr(b, sub), n)
                lin.weight.data = wq(lin.weight.data, bits, group)
    return m


def outlier_model(seed=0):
    """074's recipe: pre-train the tiny LLaMA, then give it 3 fixed outlier activation channels (~250x)."""
    base = L.pretrain(L.new_model(seed), np.random.default_rng(seed))
    return SQ.inject_outliers(base, np.random.default_rng(seed + 2))


def task_batch(tasks, n=126, seed=1):
    rng = np.random.default_rng(seed)
    return torch.cat([L.make_batch(t, n // len(tasks), rng) for t in tasks])


REPORTED = {
    "Table 1 (INT3-g128 WikiText ppl, OPT-1.3B / 6.7B / 13B)": "FP16 14.62 / 10.86 / 10.13; RTN 119.00 / 23.54 / "
        "46.04; keep 1% FP16 chosen by activation 16.91 / 11.39 / 10.43; by weight norm 98.55 / 22.37 / 48.96; random "
        "119.76 / 23.54 / 44.87",
    "Table 2 (OPT-6.7B, scale the 1% salient channels by s)": "s = 1 / 1.25 / 1.5 / 2 / 4: ppl 23.54 / 12.87 / 12.48 / "
        "11.92 / 12.36; share of groups whose Delta changes 0% / 2.8% / 4.4% / 8.2% / 21.2%",
    "Table 3 (INT3-g128, OPT-1.3B / 6.7B / 13B / 30B)": "RTN 119.47 / 23.54 / 46.04 / 18.80; 1% FP16 16.91 / 11.39 / "
        "10.43 / 9.85; s = 2 18.63 / 11.92 / 10.80 / 10.32; AWQ 16.32 / 11.39 / 10.56 / 9.77",
    "search": "s = s_X^alpha, alpha grid over [0, 1] (20 points), plus weight clipping; calibration from the "
              "pre-training data; less over-fitting than GPTQ (only per-channel mean magnitudes are measured)",
    "calibration": "AWQ needs ~10x less calibration data than GPTQ (16 vs 192 sequences); with a mismatched calibration "
                   "domain (PubMed <-> Enron) AWQ's perplexity rises 0.5-0.6, GPTQ's 2.3-4.9; AWQ + GPTQ combine (INT2)",
    "LLaMA / Llama-2 (Table 4)": "AWQ beats RTN and GPTQ (with and without reordering) from 7B to 70B, INT3 and INT4",
    "TinyChat": "W4A16 kernels: 3.2-3.3x faster than Hugging Face FP16 on desktop and mobile GPUs; Llama-2-70B on a "
                "64 GB Jetson Orin",
}
