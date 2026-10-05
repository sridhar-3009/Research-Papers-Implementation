"""MRE experiments: the paper's Tables 1-5 with 5 runs each, plus a matrix-size sweep.

  E1  Table 1: mod-12 arithmetic with 30 / 60 / 90 held out, 5 runs; matrix size N in {2, 3, 4, 5, 6}.
  E2  Table 2: family trees with 10 / 20 / 30 held out, 5 runs.
  E3  Table 3 + the discriminative ablation: +1, +4, +6, +10 held out, Eq. 2 vs discriminative higher-order cost.
  E4  Table 4: has_father, has_aunt, has_sister, has_nephew via higher_oppsex, 5 runs.
  E5  Table 5: incremental learning for the same 8 relations, 5 runs.

Each run takes 0.2-2 s; the full set takes a few minutes; none were run in full when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mre as M

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    out = {}
    for n in ((30,) if quick else (30, 60, 90)):
        for N in ((4,) if quick else (2, 3, 4, 5, 6)):
            errs = [M.run_arithmetic(n, seed=s, N=N)[0] for s in range(2 if quick else 5)]
            out[f"held={n} N={N}"] = errs
            print(f"  E1 {n} held out, {N}x{N}: {errs} (mean {np.mean(errs):.1f})")
    return out


def e2(quick):
    out = {}
    for n in ((10,) if quick else (10, 20, 30)):
        errs = [M.run_family(n, seed=s)[0] for s in range(2 if quick else 5)]
        out[n] = errs
        print(f"  E2 {n} held out: {errs} (mean {np.mean(errs):.1f})")
    return out


def e3(quick):
    out = {}
    for k in ((4,) if quick else (1, 4, 6, 10)):
        seeds = range(2 if quick else 5)
        out[f"+{k}"] = {"eq2": [M.higher_order_arithmetic(k, seed=s)[0] for s in seeds],
                        "discriminative": [M.higher_order_arithmetic(k, seed=s, discriminative_higher=True)[0] for s in seeds]}
        print(f"  E3 +{k}: {out[f'+{k}']}")
    return out


def e4(quick):
    out = {}
    for r in (("has_sister",) if quick else ("has_father", "has_aunt", "has_sister", "has_nephew")):
        out[r] = [M.higher_order_family(r, seed=s)[0] for s in range(2 if quick else 5)]
        print(f"  E4 {r}: {out[r]}")
    return out


def e5(quick):
    out = {}
    for k in ((6,) if quick else (1, 4, 6, 10)):
        out[f"+{k}"] = [M.higher_order_arithmetic(k, seed=s, incremental=True)[0] for s in range(2 if quick else 5)]
        print(f"  E5 +{k}: {out[f'+{k}']}")
    for r in (("has_sister",) if quick else ("has_father", "has_aunt", "has_sister", "has_nephew")):
        out[r] = [M.higher_order_family(r, seed=s, incremental=True)[0] for s in range(2 if quick else 5)]
        print(f"  E5 {r}: {out[r]}")
    return out


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        print(json.dumps(json.loads(OUT.read_text()), indent=1) if OUT.exists() else "no results.json yet")
        return
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, fn in EXPS.items():
        if a.only and name != a.only:
            continue
        t = time.time()
        print(f"--- {name}")
        res[name] = fn(a.quick)
        print(f"    ({time.time() - t:.1f} s)")
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
