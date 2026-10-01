"""A light tour of Sparse Transformers (Child et al. 2019). About 6 seconds.

    python3 demo.py
"""

import math
import time

import torch
import torch.nn.functional as F

from sparse import (SparseTransformerLM, block_local_attention, causal, entries, fixed_patterns, masked_attention,
                    reachable_in_two, strided_column_attention, strided_patterns)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Figure 3: which positions does output 27 read? (a 6 x 6 'image', stride 6)")
n, l = 36, 6
A1, A2 = strided_patterns(n, l)
F1, F2 = fixed_patterns(n, l, 2)
for name, (a, b) in (("strided", (A1, A2)), ("fixed (c = 2)", (F1, F2))):
    print(f"  {name}: head 1 | head 2")
    for r in range(n // l):
        row = lambda m: " ".join("@" if r * l + c == 27 else "#" if m[27, r * l + c] else "." for c in range(l))
        print(f"     {row(a)}   |   {row(b)}")
print("-> strided: the previous 6 cells + the same column above. fixed: the current block + the summary cells")
print("   (last c of every block). Either way, two hops reach EVERY earlier cell:")
print("  strided reachable in 2 steps:", torch.equal(reachable_in_two(A1, A2), causal(n)),
      "| fixed (block, then summary):", torch.equal(reachable_in_two(F1, F2), causal(n)))

line("2. How many query-key pairs are computed? (stride l = sqrt(n))")
for N in (1024, 4096, 16384):
    L = int(math.sqrt(N))
    a, b = strided_patterns(N, L) if N <= 4096 else (None, None)
    sparse = entries(a | b) if a is not None else N * 2 * L                      # too big to build: ~ 2 n sqrt(n)
    mark = " " if a is not None else "~"
    print(f"  n = {N:6d}: dense {N * (N + 1) // 2:>12,}   strided {mark}{sparse:>11,}   ({N * (N + 1) // 2 / sparse:5.1f}x fewer)")
print("-> O(n sqrt(n)) instead of O(n^2): 16,384-long sequences become affordable.")

line("3. Computing a pattern WITHOUT the n x n matrix (Section 5.5)")
q, k, v = torch.randn(3, 4, 1024, 32).unbind(0)
L = 32
fa1, _ = fixed_patterns(1024, L, 4)
_, sa2 = strided_patterns(1024, L)
for name, fast, mask in (("block-local (reshape into blocks)", block_local_attention, fa1),
                         ("strided column (transpose)", strided_column_attention, sa2)):
    t0 = time.time(); a = fast(q, k, v, L); t_fast = time.time() - t0
    t0 = time.time(); b = masked_attention(q, k, v, mask); t_mask = time.time() - t0
    print(f"  {name:34s}: equal {torch.allclose(a, b, atol=1e-4)}, {1000 * t_fast:5.1f} ms vs masked dense "
          f"{1000 * t_mask:5.1f} ms")
print("-> same numbers, but only the needed blocks are ever computed (the paper wrote GPU kernels for this).")

line("4. A periodic task: every row of 8 repeats the row above (64 positions, 2 layers, 200 updates)")
V, n, l = 8, 64, 8
for pat in ("dense", "strided", "fixed"):
    torch.manual_seed(0)
    m = SparseTransformerLM(V, n, d=64, layers=2, heads=2, pattern=pat, stride=l, c=2)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(200):
        x = torch.randint(0, V, (32, l)).repeat(1, n // l)
        loss = F.cross_entropy(m(x[:, :-1])[:, l - 1:].reshape(-1, V), x[:, l:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  {pat:7s}: loss {loss.item():.3f}   (guessing: {math.log(V):.3f})")
print("-> the strided pattern reads 'the same column, one row up' directly, like dense attention; the fixed")
print("   pattern must squeeze a whole row through 2 summary cells, and learns far more slowly. That is the")
print("   paper's finding: strided suits images/periodic data, fixed suits text (where strided failed).")
