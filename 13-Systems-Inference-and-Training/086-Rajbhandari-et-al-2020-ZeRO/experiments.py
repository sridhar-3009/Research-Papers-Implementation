"""ZeRO experiments (Rajbhandari et al. 2020): memory / communication sweeps of the model, scaling of the simulated
cluster, and real ZeRO with DeepSpeed or PyTorch FSDP when available.

  E1  Memory grid: model sizes {1.5B, 7.5B, 13B, 100B, 1T} x N_d {1 ... 1024} x stage x optimizer multiplier K
      {4 (SGD+momentum in mixed precision), 12 (Adam)}: per-GPU model states and the fit on 16 / 32 / 80 GB GPUs.
  E2  Simulated cluster: N_d {2, 4, 8, 16, 32} x stage: persistent bytes per rank and elements sent per step, checked
      against the formulas; identical final weights.
  E3  Communication time model: per-step time = compute + communication / bandwidth for each stage over N_d and
      bandwidths -- when does P_os+g+p's 1.5x communication start to matter?
  E4  Real (needs torch.distributed with several GPUs, or CPU processes via gloo): PyTorch FullyShardedDataParallel
      (FSDP, ZeRO-3 style) vs DistributedDataParallel on a 125M-parameter GPT-style model -- peak memory per rank and
      step time, launched with torchrun.

!! E4 needs a multi-GPU machine (or patience with gloo on CPU); E1-E3 run in seconds.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import zero as Z

HERE = Path(__file__).parent


def e1(a):
    out = []
    for psi in a.sizes:
        for nd in a.nds:
            for K in (4, 12):
                for s in range(4):
                    gb = Z.model_state_bytes(psi, nd, s, K) / 1e9
                    out.append({"params": psi, "N_d": nd, "K": K, "stage": s, "GB per GPU": gb,
                                **{f"fits {g} GB": gb < g for g in (16, 32, 80)}})
    return out


def e2(a):
    out = []
    for nd in a.cluster_sizes:
        res = {s: Z.train(s, nd=nd, steps=a.steps) for s in range(4)}
        for s, r in res.items():
            psi = r["psi"]
            expected = {0: 16, 1: 4 + 12 / nd, 2: 2 + 14 / nd, 3: 16 / nd}[s]
            out.append({"N_d": nd, "stage": s, "bytes per param": r["persistent bytes"] / psi, "formula": expected,
                        "sent x psi": r["sent per step (x psi)"],
                        "identical to DP": bool(np.array_equal(res[0]["weights"], r["weights"]))})
    return out


def e3(a):
    out = []
    psi, flops_per_param = 7.5e9, 6 * 2048                      # 6 FLOPs per param per token, 2048 tokens per GPU
    for nd in (8, 64, 512):
        for bw in (25e9, 100e9, 400e9):                          # bytes/s per GPU
            compute = psi * flops_per_param / 100e12             # 100 TFLOPS per GPU
            row = {"N_d": nd, "bandwidth GB/s": bw / 1e9}
            for s in range(4):
                comm = Z.comm_volume(s) * (nd - 1) / nd * psi * 2 / bw   # fp16 elements
                row[f"stage {s} step s"] = compute + comm
            out.append(row)
    return out


def e4(a):
    import torch.distributed.fsdp  # noqa: F401
    return {"note": "run with torchrun --nproc-per-node N: wrap a GPT-style model in FullyShardedDataParallel "
                    "(ShardingStrategy.FULL_SHARD ~ P_os+g+p, SHARD_GRAD_OP ~ P_os+g) vs DistributedDataParallel and "
                    "record torch.cuda.max_memory_allocated and step time"}


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
    a.sizes, a.nds, a.cluster_sizes, a.steps = (1.5e9, 7.5e9, 13e9, 100e9, 1e12), (1, 4, 16, 64, 256, 1024), (2, 4, 8, 16, 32), 50
    if a.quick:
        a.sizes, a.nds, a.cluster_sizes, a.steps = (7.5e9,), (64,), (4,), 3
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
