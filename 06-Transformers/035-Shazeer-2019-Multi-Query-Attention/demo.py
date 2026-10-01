"""A light tour of multi-query attention (Shazeer 2019). About 10 seconds.

    python3 demo.py
"""

import time

import torch
import torch.nn.functional as F

from mqa import DecoderLM, incremental_ratio, kv_cache_numbers, matching_ffn_width

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. What decoding has to re-read every step: the K/V cache (the paper's translation model)")
b, n, h, k, layers = 1024, 128, 8, 128, 6                       # batch, length, heads, key size, decoder layers
for name, g in (("multi-head  (8 K/V heads)", 8), ("multi-query (1 K/V head) ", 1)):
    nums = kv_cache_numbers(b, n, h, k, layers, g)
    print(f"  {name}: {nums:>12,} numbers = {nums * 2 / 2 ** 30:5.2f} GiB in 16-bit")
print("-> every decoding step reads the whole cache again; sharing one key/value head makes it h = 8 x smaller.")

line("2. The memory-access / arithmetic ratio of incremental decoding (Sections 2.4.1, 3.1)")
for nn_, bb in ((128, 1), (128, 128), (1024, 128)):
    mha, mqa = incremental_ratio(nn_, 1024, 8, bb, "mha"), incremental_ratio(nn_, 1024, 8, bb, "mqa")
    print(f"  n = {nn_:4d}, d = 1024, b = {bb:3d}: multi-head {mha:.3f}   multi-query {mqa:.3f}")
print("-> hardware does ~100x more arithmetic than memory traffic per second, so a ratio near 1 means waiting on")
print("   memory. Batching kills the 1/b term; only multi-query also shrinks the n/d term (to n/(d h)).")

line("3. Equal parameter counts: widen the feed-forward layers (Section 4.1)")
print(f"  translation model: d_ff 4096 -> {matching_ffn_width(1024, 8, 128, 4096, 18, 12):.0f}   (paper: 5440)")
print(f"  language model   : d_ff 8192 -> {matching_ffn_width(1024, 8, 128, 8192, 6, 6):.0f}   (paper: 9088)")
print("-> each attention layer saves 2 d k (h - 1) parameters in P_k and P_v; spread them over the FFNs.")

line("4. Decoding speed on this CPU: 64 sequences, 120 new tokens, 4 layers, d = 256")
for g, name in ((8, "multi-head"), (2, "grouped (2 K/V heads)"), (1, "multi-query")):
    m = DecoderLM(50, d=256, h=8, kv_heads=g, d_ff=1024, layers=4).eval()
    p = torch.randint(0, 50, (64, 8))
    t0 = time.time(); m.generate(p, 120, incremental=True); dt = time.time() - t0
    print(f"  {name:22s}: {1000 * dt / 120:5.2f} ms per step   cache {m.cache_numbers(64, 128):>10,} numbers")
m = DecoderLM(50, d=256, h=8, kv_heads=8, d_ff=1024, layers=4).eval()
t0 = time.time(); m.generate(torch.randint(0, 50, (64, 8)), 120, incremental=False)
print(f"  multi-head, NO cache (recompute everything each step): {1000 * (time.time() - t0) / 120:5.2f} ms per step")
print("-> caching keys/values turns each step into 'one new position'; multi-query then shrinks what each")
print("   step must read. (On TPUs the paper saw 46 -> 3.8 microseconds per decoded token.)")

line("5. Quality: multi-head vs multi-query (parameter-matched) on a copy task, 300 updates")
for g, d_ff in ((4, 64), (1, 64 + int(round(matching_ffn_width(32, 4, 8, 64, 1, 1) - 64)))):
    torch.manual_seed(0)
    m = DecoderLM(12, d=32, h=4, kv_heads=g, d_ff=d_ff, layers=2)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(300):
        x = torch.randint(3, 12, (32, 8))
        seq = torch.cat([x, x], 1)
        loss = F.cross_entropy(m(seq[:, :-1])[0][:, 7:].reshape(-1, 12), seq[:, 8:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    params = sum(p.numel() for p in m.parameters())
    print(f"  {'multi-head ' if g == 4 else 'multi-query'} ({params:,} parameters, d_ff {d_ff}): copy loss {loss.item():.4f}")
print("-> both learn the task; the paper found multi-query only slightly worse in quality (BLEU 26.5 vs 26.7 dev)")
print("   and much better than shrinking the number of heads or the key size.")
