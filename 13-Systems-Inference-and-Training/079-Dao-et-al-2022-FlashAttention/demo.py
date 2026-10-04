"""FlashAttention in ~10 seconds: online softmax by hand, exactness of the tiled forward and the recomputing backward,
measured slow-memory (HBM) traffic vs sequence length N and SRAM size M (Theorem 2), the memory footprint,
block-sparse attention, and an honest CPU timing note."""

import math
import time

import numpy as np
import torch

from flash import (REPORTED, block_sizes, flash_attention, flash_backward, online_softmax, standard_attention,
                   standard_backward_io)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Online softmax: one block at a time with a running max m and sum l")
x = torch.tensor([1.0, 3.0, 2.0, 5.0])
m, l = -math.inf, 0.0
print("  x = [1, 3, 2, 5] in blocks of 2:")
for s in (0, 2):
    xb = x[s:s + 2]
    mb, lb = float(xb.max()), float(torch.exp(xb - xb.max()).sum())
    m_new = max(m, mb)
    l = math.exp(m - m_new) * l + math.exp(mb - m_new) * lb
    print(f"    block {list(map(int, xb.tolist()))}: block max {mb:.0f}, block sum {lb:.4f} -> running m = {m_new:.0f}, "
          f"l = e^({m:.0f} - {m_new:.0f}) l_old + e^({mb:.0f} - {m_new:.0f}) {lb:.4f} = {l:.4f}")
    m = m_new
p, _, _ = online_softmax(x, 2)
print(f"  softmax = e^(x - {m:.0f}) / {l:.4f} = {[round(v, 4) for v in p.tolist()]}  (torch: "
      f"{[round(v, 4) for v in torch.softmax(x, 0).tolist()]})")

section("2. Exact, forward and backward")
torch.manual_seed(0)
N, d, M = 512, 32, 8192
Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
O_std, h_std = standard_attention(Q, K, V)
O_fl, mrow, lrow, h_fl = flash_attention(Q, K, V, M)
Bc, Br = block_sizes(M, d)
print(f"  N = {N}, d = {d}, SRAM M = {M} elements -> blocks B_c = ceil(M/4d) = {Bc}, B_r = min(B_c, d) = {Br}")
print(f"  forward: max |O_flash - O_standard| = {(O_fl - O_std).abs().max():.1e}")
Qr, Kr, Vr = (t.clone().requires_grad_() for t in (Q, K, V))
out = torch.softmax(Qr @ Kr.T / math.sqrt(d), -1) @ Vr
dO = torch.randn(N, d)
out.backward(dO)
dQ, dK, dV, h_bw = flash_backward(Q, K, V, O_fl, dO, mrow, lrow, M)
print(f"  backward with RECOMPUTED scores (only O, m, l saved): max error vs autograd dQ {(dQ - Qr.grad).abs().max():.1e}, "
      f"dK {(dK - Kr.grad).abs().max():.1e}, dV {(dV - Vr.grad).abs().max():.1e}")
print(f"  HBM traffic (elements): forward standard {h_std.total:,} vs flash {h_fl.total:,}; backward standard "
      f"{standard_backward_io(N, d):,} vs flash {h_bw.total:,}")

section("3. HBM traffic vs N and SRAM size M (Theorem 2: Theta(Nd + N^2) vs Theta(N^2 d^2 / M)), d = 32")
print("    N       M        standard      FlashAttention   ratio    N^2 d^2 / M")
rows = {}
for n in (256, 512, 1024):
    for mm in (2048, 8192, 32768):
        Qn, Kn, Vn = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)
        _, hs = standard_attention(Qn, Kn, Vn)
        _, _, _, hf = flash_attention(Qn, Kn, Vn, mm)
        rows[(n, mm)] = (hs.total, hf.total)
        print(f"    {n:5d}  {mm:6d}   {hs.total:11,d}    {hf.total:11,d}     {hs.total / hf.total:5.1f}x   {n * n * d * d // mm:10,d}")
print(f"  -> doubling N multiplies FlashAttention's traffic by ~{rows[(1024, 32768)][1] / rows[(512, 32768)][1]:.1f} (N^2) and 4x more SRAM "
      f"divides it by ~{rows[(1024, 8192)][1] / rows[(1024, 32768)][1]:.1f} (1/M).")
print("     With a tiny SRAM (M = 2048, only 16 x 32 blocks) it is WORSE than standard attention: the d^2/M factor must be")
print("     small, which real GPUs give (A100: 192 KB of SRAM per SM, ~96k fp16 values, vs head dims of 64-128).")

section("4. Memory footprint: no N x N matrix")
for n in (4096, 16384, 65536):
    print(f"    N = {n:6d}, d = 64: standard keeps S and P ({2 * n * n * 4 / 2**30:7.2f} GiB in fp32); FlashAttention keeps O, m, l "
          f"({(n * 64 + 2 * n) * 4 / 2**20:6.1f} MiB)")
print("  -> linear instead of quadratic in N (the paper: up to 20x less memory than exact baselines)")

section("5. Block-sparse FlashAttention: skip zero blocks")
n, mm = 1024, 8192
Qn, Kn, Vn = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)
bc, br = block_sizes(mm, d)
tr, tc = math.ceil(n / br), math.ceil(n / bc)
_, _, _, dense = flash_attention(Qn, Kn, Vn, mm)
for keep in (1.0, 0.5, 0.25, 0.1):
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(tr, tc, generator=g) < keep
    mask[torch.arange(tr), torch.clamp(torch.arange(tr) * tc // tr, max=tc - 1)] = True     # every row keeps a block
    _, _, _, h = flash_attention(Qn, Kn, Vn, mm, block_mask=mask)
    print(f"    {mask.float().mean():5.1%} of blocks kept: HBM traffic {h.total:10,d} ({h.total / dense.total:5.1%} of dense)")
print("  -> traffic scales with the fraction of non-zero blocks (Proposition 4: Theta(Nd + N^2 d^2 s / M))")

section("6. Honest timing note (CPU)")
n = 1024
Qn, Kn, Vn = torch.randn(n, 64), torch.randn(n, 64), torch.randn(n, 64)
t = time.time()
for _ in range(5):
    torch.softmax(Qn @ Kn.T / 8, -1) @ Vn
t_std = (time.time() - t) / 5
t = time.time()
flash_attention(Qn, Kn, Vn, 32768)
t_fl = time.time() - t
print(f"  N = 1024: one-shot PyTorch attention {1000 * t_std:.1f} ms vs our Python-loop FlashAttention {1000 * t_fl:.1f} ms")
print("  -> on a CPU (big caches, Python loops) tiling does not pay; the speed-up comes from a FUSED GPU kernel that")
print("     keeps blocks in SRAM. What we can reproduce exactly is the reduced memory traffic and memory footprint.")

section("7. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
