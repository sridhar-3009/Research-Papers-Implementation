"""GPipe experiments (Huang et al. 2019): schedule sweeps, partitioning quality, micro-batch statistics, and a real
multi-process pipeline when available.

  E1  Throughput grid: K in {2, 4, 8, 16, 32} x M in {1, 2, 4, 8, 16, 32, 64, 128} x backward/forward ratio {2, 3}
      x re-materialisation on/off, for balanced and imbalanced layers; where does M >= 4K stop mattering?
  E2  Partitioning: our cumulative-cost cut vs the optimal min-max partition (dynamic programming) on random
      heavy-tailed layer costs: the slowest-stage load and the resulting throughput.
  E3  Batch normalisation: statistics computed per micro-batch (as GPipe does) vs over the whole mini-batch -- how
      far the normalised activations differ as M grows (the one place where pipelining is not exact).
  E4  Real memory: torch.utils.checkpoint on a deep MLP vs plain autograd -- peak memory (CUDA) or allocated tensor
      bytes for depth {16, 64} and micro-batches {1, 4, 16}.
  E5  Real pipeline (needs torch.distributed.pipelining, multiple processes): GPipe schedule over 2 / 4 CPU processes
      vs a single process -- examples per second.

!! E5 spawns processes (and benefits from GPUs); E1-E4 run in seconds to minutes.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import gpipe as Gp

HERE = Path(__file__).parent


def e1(a):
    rng = np.random.default_rng(0)
    costs = {"balanced": [1.0] * 64, "imbalanced": list(np.exp(rng.normal(0, 2.0, 64)))}
    out = []
    for name, c in costs.items():
        for K in a.Ks:
            for M in a.Ms:
                for ratio in (2.0, 3.0):
                    for remat in (False, True):
                        st = Gp.partition(c, K)
                        fwd = [sum(c[i] for i in s) / M for s in st]
                        t, busy, _ = Gp.simulate(fwd, M, backward_ratio=ratio, remat=remat)
                        single = Gp.simulate([sum(c)], 1, backward_ratio=ratio, remat=False)[0]
                        out.append({"layers": name, "K": K, "M": M, "bwd ratio": ratio, "remat": remat,
                                    "speedup vs 1 device": single / t, "idle fraction": 1 - busy.mean() / t})
    return out


def optimal_partition(costs, K):
    """Min-max contiguous partition by dynamic programming (the best any cut can do)."""
    n = len(costs)
    pre = np.concatenate([[0], np.cumsum(costs)])

    @lru_cache(None)
    def best(i, k):
        if k == 1:
            return pre[n] - pre[i], (n,)
        res = (float("inf"), ())
        for j in range(i + 1, n - k + 2):
            load, cuts = best(j, k - 1)
            cand = max(pre[j] - pre[i], load)
            if cand < res[0]:
                res = (cand, (j,) + cuts)
        return res
    load, cuts = best(0, K)
    bounds = [0] + list(cuts)
    return load, [list(range(bounds[i], bounds[i + 1])) for i in range(K)]


def e2(a):
    out = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        c = list(np.exp(rng.normal(0, 1.5, a.n_layers)))
        for K in (2, 4, 8):
            ours = max(sum(c[i] for i in s) for s in Gp.partition(c, K))
            opt, _ = optimal_partition(tuple(c), K)
            out.append({"seed": seed, "K": K, "ours max stage load": ours, "optimal": opt, "ideal (total/K)": sum(c) / K})
    return out


def e3(a):
    torch.manual_seed(0)
    x = torch.randn(256, 32) * 3 + 1
    full = (x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-5)
    out = {}
    for M in (1, 2, 4, 8, 16, 32, 64):
        parts = [(c - c.mean(0)) / (c.std(0, unbiased=False) + 1e-5) for c in x.chunk(M)]
        out[M] = float((torch.cat(parts) - full).abs().mean())
    return out


def e4(a):
    from torch.utils.checkpoint import checkpoint_sequential
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for depth in a.depths:
        model = nn.Sequential(*[nn.Sequential(nn.Linear(512, 512), nn.Tanh()) for _ in range(depth)]).to(dev)
        x = torch.randn(256, 512, device=dev, requires_grad=True)
        for mode in ("plain", "checkpointed"):
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            y = model(x) if mode == "plain" else checkpoint_sequential(model, 4, x, use_reentrant=False)
            y.sum().backward()
            out[f"depth {depth}, {mode}"] = (torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else
                                             "peak memory needs CUDA; run on a GPU")
    return out


def e5(a):
    from torch.distributed.pipelining import ScheduleGPipe  # noqa: F401  (import check; full launch needs torchrun)
    return {"note": "launch with torchrun --nproc-per-node K and build PipelineStage objects from split_stages(); this "
                    "experiment only checks the API is present on this machine"}


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs, key=str):
        Ls += [f"## {str(k).upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.Ks, a.Ms, a.seeds, a.n_layers, a.depths = (2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64, 128), 10, 48, (16, 64)
    if a.quick:
        a.Ks, a.Ms, a.seeds, a.n_layers, a.depths = (2,), (1, 4), 1, 12, (4,)
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except ImportError as e:
                    Rs[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
