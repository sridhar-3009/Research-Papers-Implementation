"""Hidden-technical-debt experiments (Sculley et al. 2015): the toys of hidden_debt.py swept over seeds and settings.

  E1  Direct feedback loop: policy {greedy, eps 0.05/0.1/0.2, UCB, greedy + isolated slice 5/10/20%} x launch-log size
      {10, 30, 100 views per item} x 50 seeds -- share of seeds stuck, CTR, error of the final CTR estimates.
  E2  Hidden loop between two systems: A's product quality from -2 to 2 -- B's learned weight, B's A/B lift, and the
      rank correlation between B's review ranking learned under one A and the ranking under another.
  E3  Monitoring sensitivity: how big a producer bug (share of a slice broken x size of the unit change) is needed
      before overall bias, sliced bias, and the data tests fire? (Detection curves, 20 seeds.)
  E4  Reproducibility: spread of SGD weights and accuracy over 50 shuffle seeds; float32 vs float64 sum differences
      for array sizes 1e3 ... 1e7.

All are small numpy runs (seconds to a few minutes); none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import hidden_debt as H

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    seeds = range(5 if quick else 50)
    policies = [("greedy", 0.0, 0.1), ("eps", 0.0, 0.1), ("ucb", 0.0, 0.1), ("greedy", 0.1, 0.1)]
    if not quick:
        policies += [("eps", 0.0, 0.05), ("eps", 0.0, 0.2), ("greedy", 0.05, 0.1), ("greedy", 0.2, 0.1)]
    out = {}
    for init in ((30,) if quick else (10, 30, 100)):
        for pol, hold, eps in policies:
            rs = [H.direct_feedback_loop(pol, eps=eps, holdout=hold, initial_views=init,
                                         rounds=300 if quick else 1000, seed=s) for s in seeds]
            r = {"stuck": float(np.mean([x["share of rounds on best item"] < 0.1 for x in rs])),
                 "ctr": float(np.mean([x["ctr"] for x in rs])),
                 "estimate error": float(np.mean([np.abs(x["estimates"] - x["true"]).mean() for x in rs]))}
            key = f"init={init} {pol} eps={eps} hold={hold}"
            out[key] = r
            print(f"  E1 {key}: " + ", ".join(f"{k} {v:.4f}" for k, v in r.items()))
    return out


def e2(quick):
    out = {}
    for q in ((-1.0, 1.0) if quick else np.linspace(-2, 2, 9)):
        r = H.two_systems(float(q), n=20000 if quick else 100000)
        out[f"{q:+.1f}"] = r
        print(f"  E2 A quality {q:+.1f}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    return out


def e3(quick):
    X, y, c, names = H.world()
    w = H.fit_logreg(X, y)
    schema = H.make_schema(X)
    base = H.sliced_bias(w, *H.world(seed=99)[:3], names)
    out = {}
    for frac in ((0.5, 1.0) if quick else (0.1, 0.25, 0.5, 1.0)):
        for scale in ((0.0, 10.0) if quick else (0.0, 0.5, 2.0, 10.0, 100.0)):
            hits = {"overall bias": 0, "sliced bias": 0, "data tests": 0}
            seeds = range(3 if quick else 20)
            for s in seeds:
                Xb, yb, cb, _ = H.world(seed=s + 1)
                m = (cb == names.index("IN")) & (np.random.default_rng(s).random(len(yb)) < frac)
                Xb[m, 0] *= scale
                sb = H.sliced_bias(w, Xb, yb, cb, names)
                hits["overall bias"] += abs(sb["ALL"] - base["ALL"]) > 0.02
                hits["sliced bias"] += max(abs(sb[n] - base[n]) for n in names) > 0.02
                hits["data tests"] += bool(H.data_tests(schema, Xb))
            r = {k: v / len(seeds) for k, v in hits.items()}
            out[f"frac={frac} scale={scale}"] = r
            print(f"  E3 broken share {frac} x{scale}: " + ", ".join(f"{k} {v:.2f}" for k, v in r.items()))
    return out


def e4(quick):
    ws = np.array(H.sgd_runs(seeds=tuple(range(5 if quick else 50)), n=2000 if quick else 4000))
    out = {"sgd weight std": ws.std(0).tolist()}
    print(f"  E4 SGD weight std over shuffle seeds: {np.round(ws.std(0), 3)}")
    for n in ((10 ** 3, 10 ** 5) if quick else (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)):
        x = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        a, b = np.sum(x, dtype=np.float32), np.sum(np.sort(x), dtype=np.float32)
        out[f"sum n={n}"] = float(abs(a - b))
        print(f"  E4 n={n}: float32 pairwise vs sorted differ by {abs(a - b):.2e}")
    return out


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


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
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
