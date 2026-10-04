"""AWQ in ~25 seconds: why scaling a salient weight up shrinks its rounding error, which channels are salient (Table 1:
by activation, not by weight), scaling the salient channels by a fixed s (Table 2), and the AWQ search vs RTN, mixed
precision and GPTQ on a tiny LLaMA with LLM-like activation outliers (Table 3)."""

import time

import numpy as np
import torch

from awq import (REPORTED, G, awq_model, group_input_stats, outlier_model, quantize_keep_fp, rtn_model, scale_salient,
                 task_batch, wq)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Why scaling up helps (Section 3.2): Q(w s)(x / s)")
w = torch.tensor([[0.07, -0.9, 0.42, 0.55, -0.31, 0.88, 0.12, -0.6]])
x = torch.tensor([50.0, 1, 1, 1, 1, 1, 1, 1])                                # channel 0 has a big activation
for s in (1.0, 2.0, 4.0):
    sc = torch.ones(8)
    sc[0] = s
    q = wq(w * sc, 3, None) / sc
    d = (w * sc).max() - (w * sc).min()
    print(f"    s = {s}: group range {d.item():.3f} (Delta {d.item() / 7:.4f}); salient weight 0.07 -> {q[0, 0].item():.4f}; "
          f"output error |(Q - w) x| = {((q - w) @ x).abs().item():.3f}")
print("  -> Delta stays the same (0.07 x s is still far from the group's max), so the salient weight's rounding error,")
print("     multiplied by its big activation, drops (0.070 -> 0.057 -> 0.006 here; ~1/s on average). If s grew until")
print("     w s became the group's max, Delta would grow and every other weight in the group would get worse.")

section("2. A tiny LLaMA with LLM-like activation outliers (from paper 074)")
t = time.time()
model = outlier_model()
calib = task_batch(("copy", "reverse", "sort"))
X = group_input_stats(model, calib, 0, "attn", "wq")
top = X.abs().mean(0).topk(3)
print(f"  pre-trained, given 3 outlier channels, re-trained ({time.time() - t:.0f}s): accuracy {G.task_accuracy(model):.1%}")
print(f"  block 0 attention input: mean |x| median {X.abs().mean(0).median():.2f}, channels {sorted(top.indices.tolist())} "
      f"{', '.join(f'{v:.0f}' for v in top.values.tolist())}")
W = model.blocks[0].attn.wq.weight.data
print(f"  their weight columns are SMALL (norm {W[:, top.indices].norm(dim=0).mean():.2f} vs median "
      f"{W.norm(dim=0).median():.2f}): judged by weights they look unimportant")

section("3. Keep 3 of 64 input channels (~5%) in floating point, quantize the rest (Table 1)")
for bits in (3, 4):
    r = rtn_model(model, bits, 16)
    row = [f"RTN {G.task_accuracy(r):5.1%}"]
    for how in ("act", "weight", "random"):
        row.append(f"by {how}: {G.task_accuracy(quantize_keep_fp(model, calib, bits, 16, 0.05, how)):5.1%}")
    print(f"    INT{bits}, groups of 16:  " + "   ".join(row))
print("  -> channels chosen by ACTIVATION magnitude rescue the model; chosen by weight norm or at random they do nothing")
print("     (the paper, OPT-6.7B INT3-g128: RTN 23.54 ppl, 1% by activation 11.39, by weight 22.37, random 23.54).")

section("4. Scale the salient channels by a fixed s instead (Table 2), INT4-g16")
print("    s       groups whose Delta changed    accuracy")
for s in (1.0, 1.5, 2.0, 4.0, 8.0):
    q, ch = scale_salient(model, calib, 4, 16, s, frac=0.05)
    print(f"    {s:<5}          {ch:5.1%}                  {G.task_accuracy(q):6.1%}")
print("  -> more groups change their Delta as s grows (the paper: 2.8% at s = 1.25 ... 21.2% at s = 4). Honest difference:")
print("     here the best s is large (8), not 2, because our salient weights are ~12x SMALLER than the others, so a")
print("     small s barely lifts them off zero -- one fixed s cannot suit every model, hence AWQ's search.")

section("5. AWQ: search s = s_X^alpha per layer group (Table 3)")
print("    method                                  INT3-g16 accuracy (loss)     INT4-g16 accuracy (loss)")
rows = {}
for name, f in (("RTN", lambda b: rtn_model(model, b, 16)),
                ("keep 3 channels in FP (by activation)", lambda b: quantize_keep_fp(model, calib, b, 16, 0.05, "act")),
                ("AWQ (scale search)", lambda b: awq_model(model, calib, b, 16)[0]),
                ("AWQ + weight clipping", lambda b: awq_model(model, calib, b, 16, clip=True)[0]),
                ("GPTQ (paper 075)", lambda b: G.quantize_model(model, calib, b, "gptq", 16))):
    res = [f(b) for b in (3, 4)]
    rows[name] = [(G.task_accuracy(m), G.model_loss(m)) for m in res]
    print(f"    {name:40s}   {rows[name][0][0]:6.1%} ({rows[name][0][1]:.3f})          {rows[name][1][0]:6.1%} ({rows[name][1][1]:.3f})")
_, alphas = awq_model(model, calib, 4, 16)
print(f"  alphas chosen for the 8 layer groups (INT4): {alphas}")
print("  -> AWQ recovers nearly all the accuracy with a hardware-friendly uniform format (no FP channels), like keeping")
print("     the salient channels in FP. Honest note: GPTQ does badly here -- the huge outlier inputs dominate its")
print("     Hessian and their tiny weights are rounded early with nothing correlated left to absorb the error (the")
print("     'act-order' variant, in experiments.py, quantizes those columns first). The paper finds AWQ >= GPTQ and")
print("     less prone to over-fitting the calibration data (experiments.py E3 tests that).")

section("6. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
