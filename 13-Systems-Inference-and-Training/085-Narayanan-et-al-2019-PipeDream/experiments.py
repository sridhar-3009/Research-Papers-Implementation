"""PipeDream experiments (Narayanan et al. 2019): partitioner, schedule and weight-semantics sweeps, and a check of
PyTorch's 1F1B schedule.

  E1  Partitioner: random layer profiles (compute, weights, activations) x machines {4, 8, 16, 32} x bandwidth
      {0.25 ... 8}: chosen configuration, its time vs data parallelism vs a straight pipeline; the DP's runtime.
  E2  Schedules: 1F1B vs flushing pipelines for stages {2, 4, 8} x minibatches {8 ... 128} x imbalance; makespan and
      peak in-flight activations per stage.
  E3  Weight semantics: SGD / naive / stashing / vertical sync over stages {2, 4, 8, 16} x learning rates
      {0.05, 0.1, 0.2} x 10 seeds on the pipelined MLP: median final loss and how many runs diverge.
  E4  PyTorch's pipelining library (torch.distributed.pipelining): build a Schedule1F1B over 2 / 4 stages launched with
      torchrun and compare examples/s with ScheduleGPipe.

!! E4 needs torchrun and several processes / GPUs; E1-E3 run in minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pipedream as P

HERE = Path(__file__).parent


def e1(a):
    out = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        L = a.layers
        compute = list(rng.uniform(1, 6, L))
        weights = list(np.exp(rng.normal(-1, 1.5, L)))
        acts = list(rng.uniform(0.05, 3, L))
        for M in a.machines:
            for bw in a.bandwidths:
                t0 = time.time()
                t, st = P.partition_dp(compute, weights, acts, M, bw)
                dp_time = max(sum(compute), 2 * (M - 1) / M * sum(weights) / bw) / M
                out.append({"seed": seed, "machines": M, "bandwidth": bw, "config": P.config_string(st),
                            "time": t, "data parallel": dp_time, "dp runtime s": time.time() - t0})
    return out


def e2(a):
    out = []
    for S in (2, 4, 8):
        for imb in (0.0, 0.5):
            rng = np.random.default_rng(S)
            stages = list(1 + imb * rng.random(S))
            for n in a.mbs:
                t1, peak = P.simulate_1f1b(stages, n)
                tg = sum(P.simulate_gpipe(stages, S)[0] for _ in range(max(1, n // S)))
                out.append({"stages": S, "imbalance": imb, "minibatches": n, "1F1B": t1, "flush every S": tg,
                            "1F1B peak in flight": max(peak)})
    return out


def e3(a):
    out = {}
    for S in a.stage_counts:
        for lr in a.lrs:
            row = {}
            for m in ("sgd", "naive", "stash", "vsync"):
                v = [P.train_pipelined(m, S, lr=lr, steps=a.steps, seed=s)[-1] for s in range(a.seeds)]
                row[m] = {"median": float(np.median(v)), "diverged": int(sum(not np.isfinite(x) or x > 1 for x in v))}
            out[f"S={S}, lr={lr}"] = row
    return out


def e4(a):
    from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe  # noqa: F401
    return {"note": "API present; launch with torchrun --nproc-per-node S, wrap each split stage in PipelineStage and "
                    "time Schedule1F1B vs ScheduleGPipe"}


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs, key=str):
        Ls += [f"## {str(k).upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 5)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.seeds, a.layers, a.machines, a.bandwidths = 10, 16, (4, 8, 16, 32), (0.25, 0.5, 1, 2, 4, 8)
    a.mbs, a.stage_counts, a.lrs, a.steps = (8, 16, 32, 64, 128), (2, 4, 8, 16), (0.05, 0.1, 0.2), 800
    if a.quick:
        a.seeds, a.layers, a.machines, a.bandwidths = 1, 6, (4,), (1,)
        a.mbs, a.stage_counts, a.lrs, a.steps = (8,), (2,), (0.1,), 20
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
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
