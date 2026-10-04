"""SmoothQuant in ~25 seconds: the INT8 quantizer and why outliers waste its levels, a tiny LLaMA given LLM-like
activation outliers, W8A8 with per-tensor / per-token / per-channel scales vs an LLM.int8()-style decomposition vs
SmoothQuant (Tables 1 and 3), the migration strength alpha (Figure 10), and why smoothing is free."""

import time

import numpy as np
import torch

from smoothquant import (REPORTED, L, calibration_batch, effective_levels, inject_outliers, layer_error, quantize,
                         quantized_copy, record_input_absmax, smooth, task_accuracy)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The INT8 quantizer (Eq. 1) and what an outlier does to it")
x = torch.tensor([0.3, -1.2, 0.05, 2.0])
d = x.abs().max() / 127
print(f"  x = [0.3, -1.2, 0.05, 2.0]: Delta = max|x|/127 = {d:.4f}; round(x/Delta) = {(x / d).round().int().tolist()}; "
      f"back: {[round(v, 4) for v in quantize(x).tolist()]}")
x2 = torch.cat([x, torch.tensor([200.0])])
print(f"  add one outlier 200: Delta = {200 / 127:.3f} -> the small values become "
      f"{[round(v, 3) for v in quantize(x2).tolist()[:4]]} (0.3 and 0.05 vanish, -1.2 and 2.0 are badly rounded)")

section("2. A tiny LLaMA with LLM-like activation outliers")
t = time.time()
base = L.pretrain(L.new_model(), np.random.default_rng(0))
model = inject_outliers(base, np.random.default_rng(2))
calib = calibration_batch()
print(f"  pre-trained on copy / reverse / sort, then given 3 outlier channels and re-trained ({time.time() - t:.0f}s): "
      f"task accuracy {task_accuracy(model):.1%}")
X = {}
hook = model.blocks[0].attn.wq.register_forward_hook(lambda m, i, o: X.setdefault("x", i[0].detach()))
model(calib)
hook.remove()
Xf = X["x"].reshape(-1, X["x"].shape[-1])
cmax = Xf.abs().amax(0)
top = cmax.topk(3).indices.tolist()
print(f"  input of block 0's Wq: channel max |x| median {cmax.median():.2f}, but channels {sorted(top)} reach "
      f"{', '.join(f'{v:.0f}' for v in cmax.topk(3).values.tolist())} (~{cmax.max() / cmax.median():.0f}x)")
persist = (Xf[:, top].abs() > 10 * cmax.median()).float().mean().item()
print(f"  they are large in {persist:.0%} of tokens (outliers persist in FIXED channels); weights stay flat: max |W| per input"
      f" column median {model.blocks[0].attn.wq.weight.detach().abs().amax(0).median():.2f}")
lv = effective_levels(Xf)
print(f"  per-tensor INT8: a normal channel keeps only ~{lv.median():.1f} of the 256 levels (2^8 * m_j / m)")

section("3. W8A8 post-training quantization (Tables 1 and 3)")
rows = [("floating point", model, None)]
for label, kw in (("per-tensor activations, dynamic", dict(a_gran="tensor")),
                  ("per-tensor activations, static (calibrated)", dict(a_gran="static")),
                  ("per-token activations, dynamic", dict(a_gran="token")),
                  ("per-channel activations (not INT8-GEMM friendly)", dict(a_gran="channel")),
                  ("LLM.int8()-style: outlier channels kept in FP", dict(a_gran="token", outlier=6.0))):
    rows.append((label, quantized_copy(model, calib, **kw), kw))
sq = smooth(model, calib, 0.5)
rows.append(("SmoothQuant a=0.5 + per-token dynamic (O1)", quantized_copy(sq, calib, a_gran="token"), None))
rows.append(("SmoothQuant a=0.5 + per-tensor static (O3)", quantized_copy(sq, calib, a_gran="static"), None))
print("    method                                              task accuracy   Wq output error (block 0)")
res = {}
for label, m, _ in rows:
    res[label] = task_accuracy(m)
    err = "" if m is model else f"{layer_error(model if 'SmoothQuant' not in label else sq, m, calib):.3f}"
    print(f"    {label:50s}    {res[label]:6.1%}          {err}")
print(f"  -> naive W8A8 collapses ({res['per-tensor activations, static (calibrated)']:.1%}); per-token scales help only partly "
      f"({res['per-token activations, dynamic']:.1%});")
print(f"     per-channel activation scales work ({res['per-channel activations (not INT8-GEMM friendly)']:.1%}) but an INT8 matrix"
      " multiply cannot apply a different scale to each")
print("     input channel. SmoothQuant gets per-channel-like accuracy with the cheapest scheme (one static scale per")
print(f"     tensor): {res['SmoothQuant a=0.5 + per-tensor static (O3)']:.1%}. The paper (OPT-175B): W8A8 35.5% -> SmoothQuant-O3 66.8% (FP16 66.9%).")

section("4. Smoothing is exact in floating point and moves the outliers into the weights")
x = calib[:8]
print(f"  max |output difference| smoothed vs original (no quantization): {(sq(x) - model(x)).abs().max().item():.1e}")
s_in = record_input_absmax(sq, calib)[(0, "attn", "wq")]
print(f"  block 0 Wq input after smoothing: max |x| per channel ranges {s_in.min():.2f} .. {s_in.max():.2f} "
      f"(before: {cmax.min():.2f} .. {cmax.max():.0f})")
w_after = sq.blocks[0].attn.wq.weight.detach().abs().amax(0)
print(f"  its weight columns now range {w_after.min():.2f} .. {w_after.max():.2f} (before "
      f"{model.blocks[0].attn.wq.weight.detach().abs().amax(0).min():.2f} .. {model.blocks[0].attn.wq.weight.detach().abs().amax(0).max():.2f})")
print("  the 1/s is folded into the RMSNorm gain and s into the weights offline: no extra work at run time")

section("5. Migration strength alpha (Figure 10): s_j = max|X_j|^a / max|W_j|^(1-a), per-tensor static")
alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
for bits in (8, 6):
    acc = {a: task_accuracy(quantized_copy(smooth(model, calib, a), calib, a_gran="static", w_bits=bits, a_bits=bits))
           for a in alphas}
    print(f"    W{bits}A{bits}: " + "   ".join(f"a={a}: {v:5.1%}" for a, v in acc.items()))
print("  -> at 8 bits any smoothing is enough here; at 6 bits the trade-off appears: a = 0 leaves the activations hard,")
print("     a = 1 pushes all the difficulty into the weights, a ~ 0.5 balances the two (the paper's default).")

section("6. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
