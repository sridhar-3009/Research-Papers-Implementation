"""GPipe in ~3 seconds: the pipeline schedule and its bubble, a REAL micro-batched pipeline (gradients identical to
ordinary back-propagation), re-materialisation's memory saving, and normalised throughput vs partitions K and
micro-batches M for balanced (Transformer-like) and imbalanced (AmoebaNet-like) layers (Table 2)."""

import time

import numpy as np
import torch

from gpipe import (REPORTED, bubble_fraction, gpipe_step, make_model, normalised_throughput, reference_grads, simulate,
                   split_stages)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The schedule: K = 4 stages, M = 4 micro-batches (a backward pass costs 2 forwards)")
total, busy, tl = simulate([1, 1, 1, 1], 4, backward_ratio=2.0, remat=False)
scale = 2
for k in range(4):
    row = [" "] * int(total * scale)
    for (kk, kind, m, s, e) in tl:
        if kk == k:
            for t in range(int(s * scale), int(e * scale)):
                row[t] = f"{kind}{m}"[0] if kind == "F" else "b"
    marks = "".join(row)
    print(f"    stage {k}: |{marks}|")
print(f"  (F = forward slot, b = backward slot, blank = idle; time runs left to right)")
print(f"  idle fraction measured {1 - busy.mean() / total:.3f} = formula (K-1)/(M+K-1) = {bubble_fraction(4, 4):.3f}")
print("    M \\ K        2        4        8")
for M in (1, 4, 8, 32):
    print(f"    {M:3d}        " + "   ".join(f"{bubble_fraction(K, M):6.1%}" for K in (2, 4, 8)))
print("  -> with M >= 4K the bubble is small (the paper: 'negligible'), so K devices give nearly K-fold speed")

section("2. A REAL pipeline: 12-layer MLP, K = 4 stages, M = 8 micro-batches of a 64-example batch")
model = make_model(L=12, d=64)
x, y = torch.randn(64, 64), torch.randn(64, 1)
loss_ref, g_ref = reference_grads(model, x, y)
for remat in (False, True):
    model.zero_grad()
    loss, kept = gpipe_step(split_stages(model, 4), x, y, M=8, remat=remat)
    err = max((p.grad - g).abs().max().item() for p, g in zip(model.parameters(), g_ref))
    print(f"    re-materialisation {'ON ' if remat else 'OFF'}: loss {loss:.6f} (full batch {loss_ref:.6f}), max gradient difference "
          f"{err:.1e}, activation values kept after the forward phase: {kept:,}")
print("  -> accumulating the micro-batch gradients gives EXACTLY the full-batch gradient: GPipe is synchronous,")
print("     no staleness, so the optimiser sees the same update for any K and M")
L, N, K, M, d = 12, 64, 4, 8, 64
print(f"  memory: re-materialisation keeps only each stage's input per micro-batch (16,384 values instead of 114,752,")
print(f"  7x less) plus, while re-computing, one stage's activations for one micro-batch ({(2 * L // K) * (N // M) * d:,} values)")
print("  -- the paper's O(N + (L/K)(N/M)) instead of O(N L)")

section("3. Normalised throughput vs K and M (Table 2; 1.0 = K = 2, M = 1)")
uniform = [1.0] * 48
rng = np.random.default_rng(0)
imbalanced = list(np.exp(rng.normal(0, 2.0, 48)))
paper = {"Transformer-like (uniform layers)": {1: (1, 1.07, 1.3), 4: (1.7, 3.2, 4.8), 32: (1.8, 3.4, 6.3)},
         "AmoebaNet-like (imbalanced layers)": {1: (1, 1.13, 1.38), 4: (1.07, 1.26, 1.72), 32: (1.21, 1.84, 3.48)}}
for name, costs in (("Transformer-like (uniform layers)", uniform), ("AmoebaNet-like (imbalanced layers)", imbalanced)):
    print(f"  {name}:   ours (paper) for K = 2 / 4 / 8")
    for M in (1, 4, 32):
        ours = [normalised_throughput(costs, K, M) for K in (2, 4, 8)]
        print(f"    M = {M:2d}:  " + "   ".join(f"{o:5.2f} ({p})" for o, p in zip(ours, paper[name][M])))
print(f"  -> balanced layers: near-linear speedup once M >> K; imbalanced layers (our heaviest layer holds "
      f"{max(imbalanced) / sum(imbalanced):.0%} of the work)")
print("     cap the speedup -- the slowest stage sets the pace. Honest note: our M = 1 rows stay at 1.0 for every K (a")
print("     one-micro-batch pipeline cannot overlap), while the paper's grow because it enlarged the batch when more")
print("     partitions freed memory. The trends match; individual cells differ (e.g. our M = 4 is slower for the")
print("     uniform model and faster for the imbalanced one), since we model compute only, not the real kernels.")

section("4. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
