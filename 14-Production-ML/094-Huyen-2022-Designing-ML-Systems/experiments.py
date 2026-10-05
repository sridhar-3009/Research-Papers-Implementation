"""Experiments for the Designing-ML-Systems techniques: the demo's single runs swept over seeds and settings.

  E1  Weak supervision vs hand-label budgets {50, 100, 300, 1000} over 10 seeds: accuracy of the classifier trained
      on majority-vote labels, on the hand labels, and on all true labels.
  E2  Class imbalance at positive rates {10%, 1%, 0.1%} (200k examples): ROC-AUC vs PR-AUC, recall with class weights.
  E3  Active learning over 20 seeds and 20 rounds: final accuracy of random vs uncertainty sampling.
  E4  Testing in production: A/B vs interleaving detection rate for {100 ... 5,000} users x 200 trials; Thompson
      sampling vs A/B/n regret over 20 seeds.
  E5  Continual learning: stateless vs stateful daily updates over a 60-day drifting stream.

All are small numpy runs (seconds to ~10 minutes for E4); none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dmls as D

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    out = {}
    for gold in ((100,) if quick else (50, 100, 300, 1000)):
        rs = [D.weak_supervision(seed=s, gold=gold) for s in range(2 if quick else 10)]
        out[gold] = {k: float(np.mean([r[k] for r in rs])) for k in ("classifier on majority vote",
                                                                      f"classifier on {gold} hand labels",
                                                                      "classifier on all true labels")}
        print(f"  E1 gold={gold}: {out[gold]}")
    return out


def e2(quick):
    out = {}
    for rate in ((0.01,) if quick else (0.1, 0.01, 0.001)):
        r = D.class_imbalance(n=40000 if quick else 200000, rate=rate)
        out[rate] = r
        print(f"  E2 rate={rate}: ROC-AUC {r['model ROC-AUC']:.3f}, PR-AUC {r['model PR-AUC']:.3f}, recall weighted "
              f"{r['recall @0.5 class-weighted']:.2f}")
    return out


def e3(quick):
    out = {}
    for s in range(2 if quick else 20):
        r = D.active_learning(seed=s, rounds=10 if quick else 20)
        out[s] = {k: r[k][-1] for k in ("random", "uncertainty")}
    print(f"  E3 final accuracy (mean): random {np.mean([v['random'] for v in out.values()]):.3f}, uncertainty "
          f"{np.mean([v['uncertainty'] for v in out.values()]):.3f}")
    return out


def e4(quick):
    out = {}
    for u in ((200,) if quick else (100, 200, 500, 1000, 2000, 5000)):
        out[f"users={u}"] = D.ab_vs_interleaving(users=u, trials=10 if quick else 200)
        print(f"  E4 {u} users: {out[f'users={u}']}")
    out["bandit"] = D.bandit_vs_ab(reps=3 if quick else 20)
    print(f"  E4 bandit: {out['bandit']}")
    return out


def e5(quick):
    out = D.stateless_vs_stateful(days=15 if quick else 60)
    print(f"  E5 {out}")
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
