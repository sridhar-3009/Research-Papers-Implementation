"""Efficiently Scaling Transformer Inference in ~2 seconds: the KV cache and multiquery attention (Table 1), a
feed-forward layer partitioned over virtual chips (1D / 2D weight-stationary, weight-gathered) with exact results and
counted communication (Section 3.2, Figure 3), and a roofline cost model of PaLM 540B on TPU v4 for prefill vs decode
(Table 2) and the latency-cost trade-off (Figure 1)."""

import math
import time

import numpy as np

from inference import (PALM, REPORTED, TPU_V4, Mesh, ffn_1d_ws, ffn_2d_ws, ffn_reference, ffn_weight_gathered,
                       kv_bytes_per_token, max_context, step_time)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Compute, memory and the KV cache for PaLM 540B")
n, L, E, F, H, dh = PALM["540B"]
print(f"  {n / 1e9:.0f}B parameters, {L} layers, d_model {E}, d_ff {F}, {H} heads; one token costs 2N = {2 * n / 1e12:.2f} TFLOPs")
mh = kv_bytes_per_token(L, H, 128, H)
mq = kv_bytes_per_token(L, H, dh, 1)
print(f"  KV cache per token: multihead (48 heads x d_head 128) {mh / 1e6:.2f} MB, multiquery (1 head x 256) {mq / 1e6:.3f} MB "
      f"({mh / mq:.0f}x smaller)")
print(f"  multihead, batch 512, context 2048: {mh * 512 * 2048 / 1e12:.1f} TB of KV cache vs {n * 2 / 1e12:.2f} TB of bf16 weights "
      "(the paper: '3TB, 3 times the model')")
print("  longest context whose KV cache fits in 30% of 64 chips' HBM (Table 1):")
print("    variant                        batch 128      batch 512")
paper = {"multihead": (1320, 330), "baseline multiquery": (660, 165), "optimized multiquery": (43000, 10700)}
for v, (p128, p512) in paper.items():
    print(f"    {v:28s}  {max_context(v, 128):8,.0f} ({p128:,})  {max_context(v, 512):7,.0f} ({p512:,})   ours (paper)")
print("  -> multiquery only helps if the K/V cache is sharded over the BATCH; with the usual head sharding its single")
print("     K/V head is replicated on every chip and it does worse than multihead. Batch sharding: 32-64x longer context.")

section("2. A feed-forward layer split over virtual chips: exact, and the bytes each chip sends")
rng = np.random.default_rng(0)
E_, F_, B = 512, 2048, 64
x = rng.standard_normal((B, E_))
Wi, Wo = rng.standard_normal((E_, F_)) / 22, rng.standard_normal((F_, E_)) / 45
ref = ffn_reference(x, Wi, Wo)
print(f"  tokens x d_model = {B} x {E_} (BLE = {x.nbytes / 1024:.0f} KiB), d_ff = {F_}; sent per chip in units of BLE:")
print("    chips   1D weight-stationary   2D weight-stationary   2D formula 8/sqrt(n)   max |error|")
for nchips, X in ((16, 2), (64, 4), (256, 8)):
    m1, m2 = Mesh(nchips), Mesh(nchips)
    o1 = np.concatenate(ffn_1d_ws(np.array_split(x, nchips, axis=-1), Wi, Wo, m1), axis=-1)
    o2 = ffn_2d_ws(x, Wi, Wo, m2, X, nchips // X)
    err = max(np.abs(o1 - ref).max(), np.abs(o2 - ref).max())
    print(f"    {nchips:4d}          {m1.sent.mean() / x.nbytes:5.2f}                 {m2.sent.mean() / x.nbytes:5.2f}"
          f"                  {8 / math.sqrt(nchips):5.2f}            {err:.0e}")
print("  -> 1D sends ~2 x BLE per chip however many chips you add (communication stops shrinking); 2D falls like")
print("     1/sqrt(chips), so latency keeps improving with more chips (X = sqrt(n)/2, YZ = 2 sqrt(n) is optimal)")
print("  weight-gathered (activations stay, weights move) vs 2D weight-stationary, 64 chips:")
for tokens in (16, 256, 4096, 65536):
    xt = rng.standard_normal((tokens, E_))
    m2, m3 = Mesh(64), Mesh(64)
    ffn_2d_ws(xt, Wi, Wo, m2, 4, 16)
    ffn_weight_gathered(xt, Wi, Wo, m3)
    best = "weight-stationary" if m2.sent.mean() < m3.sent.mean() else "weight-gathered"
    print(f"    {tokens:6d} tokens per batch: 2D WS {m2.sent.mean() / 1e6:8.2f} MB/chip   WG {m3.sent.mean() / 1e6:6.2f} MB/chip"
          f"   -> {best}")
print("  -> moving weights costs the same at any batch size, moving activations grows with it: big prefill batches")
print("     switch to weight-gathered layouts (Figure 3)")

section("3. Roofline model of PaLM 540B on 64 TPU v4 chips (Table 2 settings)")
print("    scenario                                         compute  memory   comm     total    MFU   | paper")
cases = (("low-latency prefill, batch 1, int8, 2D WS", "prefill", 1, "2D WS", 1, "0.29 s, MFU 43%", 1),
         ("low-latency decode, batch 64, int8, 2D WS", "decode", 64, "2D WS", 1, "1.82 s / 64 tok, MFU 14%", 64),
         ("high-throughput prefill, batch 512, bf16, WG", "prefill", 512, "WG", 2, "85.2 s, MFU 76%", 1),
         ("high-throughput decode, batch 512, bf16, 2D WS", "decode", 512, "2D WS", 2, "6.0 s / 64 tok, MFU 33%", 64))
for label, phase, b, layout, wb, paper_s, steps in cases:
    r = step_time("540B", 64, b, 2048, phase, layout, wb)
    print(f"    {label:47s} {r['compute'] * 1e3:7.1f}  {r['memory'] * 1e3:6.1f}  {r['comm'] * 1e3:6.1f}  "
          f"{r['total'] * steps:7.2f} s  {r['MFU']:4.0%}  | {paper_s}")
print("  (times in ms per step; 'total' x 64 steps for decode)")
print("  -> prefill is compute-bound (high MFU), decode is memory-bound: every step reloads all weights for only")
print("     `batch` tokens, so MFU is low at small batch. Our roofline ignores many real overheads and is 1.2-3x")
print("     faster than the measured times -- read it as a lower bound that explains the trends, not the numbers.")
d16 = step_time("540B", 64, 64, 2048, "decode", "2D WS", 2)["total"]
d8 = step_time("540B", 64, 64, 2048, "decode", "2D WS", 1)["total"]
print(f"  int8 weights halve the weight-loading time: decode step {d16 * 1e3:.1f} -> {d8 * 1e3:.1f} ms in our model "
      "(the paper: 36.9 -> 28.5 ms/token at batch 64)")

section("4. The latency-cost trade-off for decoding (Figure 1, right)")
print("    chips  batch   ms per token   chip-ms per token   fits in HBM?")
for chips in (32, 64, 128, 256):
    for b in (1, 16, 64, 256, 1024):
        r = step_time("540B", chips, b, 2048, "decode", "2D WS", 1)
        mem = n * 1 + mq * b * 2048
        fits = mem < chips * TPU_V4["hbm_bytes"]
        if b in (1, 64, 1024):
            print(f"    {chips:5d}  {b:5d}   {r['total'] * 1e3:10.2f}     {r['total'] * 1e3 * chips / b:12.3f}        {'yes' if fits else 'NO'}")
print("  -> more chips lower latency but raise cost per token; bigger batches lower cost per token at almost the")
print("     same latency until compute catches up -- so low-latency serving batches decode requests (the paper")
print("     pairs batch-1 prefill with batch-64 decode)")

section("5. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
