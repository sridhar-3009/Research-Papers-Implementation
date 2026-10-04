"""FlashAttention experiments (Dao et al. 2022): measured IO on the toy, and real GPU benchmarks when available.

  E1  IO complexity fit: N in {256 ... 4096}, d in {16, 32, 64, 128}, M in {2^12 ... 2^17}: fit log(HBM) against
      log N, log d, log M (expect exponents 2, 2, -1 for FlashAttention; 2, 0, 0 for standard at large N).
  E2  Backward pass: HBM traffic and extra FLOPs from recomputation vs storing P (Figure 2 left: 66.6 vs 75.2 GFLOPs).
  E3  Block size (Figure 2 middle): fix M, vary B_c, B_r freely; traffic vs block size.
  E4  Block-sparse patterns (local + global "butterfly"-like): traffic and exactness vs a dense masked reference.
  E5  Real benchmark (needs a CUDA GPU): torch.nn.functional.scaled_dot_product_attention with the flash backend vs
      the math backend vs a naive implementation; runtime and peak memory for N in {512 ... 16384}, fwd and fwd+bwd.

!! HEAVY for E5 (CUDA GPU); E1-E4 take seconds to minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import flash as F_

HERE = Path(__file__).parent


def e1(a):
    rows = []
    for N in a.Ns:
        for d in a.ds:
            for M in a.Ms:
                if M < 4 * d or M > N * d:
                    continue
                Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
                rows.append({"N": N, "d": d, "M": M, "flash": F_.flash_attention(Q, K, V, M)[3].total,
                             "standard": F_.standard_attention(Q, K, V)[1].total})
    X = np.array([[1, math.log(r["N"]), math.log(r["d"]), math.log(r["M"])] for r in rows])
    out = {"rows": rows}
    for key in ("flash", "standard"):
        coef, *_ = np.linalg.lstsq(X, np.log([r[key] for r in rows]), rcond=None)
        out[f"{key} exponents (N, d, M)"] = coef[1:].tolist()
    return out


def e2(a):
    out = {}
    for N in a.Ns[:3]:
        d, M = 64, 2 ** 15
        Q, K, V, dO = (torch.randn(N, d) for _ in range(4))
        O, m, l, hf = F_.flash_attention(Q, K, V, M)
        _, _, _, hb = F_.flash_backward(Q, K, V, O, dO, m, l, M)
        fwd_flops = 4 * N * N * d
        out[N] = {"flash fwd+bwd HBM": hf.total + hb.total,
                  "standard fwd+bwd HBM": F_.standard_attention(Q, K, V)[1].total + F_.standard_backward_io(N, d),
                  "standard FLOPs (fwd 4N^2d + bwd 8N^2d)": 12 * N * N * d,
                  "flash FLOPs (+ recompute QK^T and P in bwd)": 12 * N * N * d + 2 * N * N * d}
    return out


def e3(a):
    N, d = 1024, 64
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
    out = {}
    original = F_.block_sizes
    for bc, br in ((16, 16), (32, 32), (64, 64), (128, 64), (256, 64), (512, 64)):
        F_.block_sizes = lambda M, d, bc=bc, br=br: (bc, br)
        out[f"Bc={bc}, Br={br}"] = F_.flash_attention(Q, K, V, 0)[3].total
    F_.block_sizes = original
    return out


def e4(a):
    N, d, M = 1024, 32, 8192
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
    bc, br = F_.block_sizes(M, d)
    tr, tc = math.ceil(N / br), math.ceil(N / bc)
    out = {}
    for width in (1, 2, 4):
        mask = torch.zeros(tr, tc, dtype=torch.bool)
        for i in range(tr):
            c = i * br // bc
            mask[i, max(0, c - width + 1):c + 1] = True                     # local window
            mask[i, 0] = True                                               # global first block
        O, _, _, h = F_.flash_attention(Q, K, V, M, block_mask=mask)
        full = torch.zeros(N, N, dtype=torch.bool)
        for i in range(tr):
            for j in range(tc):
                if mask[i, j]:
                    full[i * br:(i + 1) * br, j * bc:(j + 1) * bc] = True
        S = (Q @ K.T / math.sqrt(d)).masked_fill(~full, -math.inf)
        ref = torch.softmax(S, -1) @ V
        out[f"window {width}"] = {"kept blocks": float(mask.float().mean()), "HBM": h.total,
                                  "max error vs masked dense": float((O - ref).abs().max())}
    return out


def e5(a):
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    from torch.nn.attention import SDPBackend, sdpa_kernel
    out = {}
    for N in a.gpu_Ns:
        q, k, v = (torch.randn(8, 16, N, 64, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
        row = {}
        for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("math", SDPBackend.MATH)):
            try:
                with sdpa_kernel(backend):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t = time.time()
                    for _ in range(5):
                        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                        o.sum().backward()
                    torch.cuda.synchronize()
                    row[name] = {"ms fwd+bwd": (time.time() - t) / 5 * 1000, "peak GB": torch.cuda.max_memory_allocated() / 1e9}
            except RuntimeError as e:
                row[name] = {"error": str(e)[:80]}
        out[N] = row
    return out


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.Ns, a.ds, a.Ms, a.gpu_Ns = (256, 512, 1024, 2048, 4096), (16, 32, 64, 128), tuple(2 ** k for k in range(12, 18)), (512, 1024, 2048, 4096, 8192, 16384)
    if a.quick:
        a.Ns, a.ds, a.Ms, a.gpu_Ns = (128, 256, 512), (16,), (1024, 4096), (512,)
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except (ImportError, AssertionError) as e:
                    Rs[name] = {"skipped": str(e)}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
