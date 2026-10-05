"""Evolution-strategies experiments (Salimans et al. 2017) at laptop scale.

  E1  CartPole data efficiency: ES (population 10 / 20 / 40 pairs, sigma 0.05 / 0.1 / 0.2) vs REINFORCE over 10 seeds:
      environment steps to reach an average training return of 475, and final greedy return.
  E2  Frame-skip (section 4.4): ES and REINFORCE with frame_skip {1, 2, 4} over 10 seeds -- final return after 100
      updates (on CartPole coarser control also makes the task itself harder, so compare the methods' SPREAD).
  E3  Fitness shaping and mirrored sampling ablation on CartPole with a 4-8-1 MLP policy: {ranks, raw} x {mirrored, plain}.
  E4  Scaling (Figure 1 analogue): wall-clock time per ES update vs population size on this machine, and the ideal
      linear speedup if each evaluation ran on its own core.
  E5  Section 3.1 variance table for T up to 10,000 and three noise levels.

E1-E3 take a few minutes; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import es as E

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def _reach(curve, target=475):
    return next((s for s, r in curve if r >= target), None)


def e1(quick):
    out = {}
    seeds = range(2 if quick else 10)
    for n_pairs, sigma in (((20, 0.1),) if quick else ((10, 0.1), (20, 0.1), (40, 0.1), (20, 0.05), (20, 0.2))):
        rs = [E.train_es_cartpole(iters=60 if quick else 150, n_pairs=n_pairs, sigma=sigma, seed=s) for s in seeds]
        out[f"ES pairs={n_pairs} sigma={sigma}"] = {"steps to 475": [_reach(r[1]) for r in rs],
                                                     "final": [r[2] for r in rs]}
    rs = [E.train_reinforce_cartpole(iters=60 if quick else 150, seed=s) for s in seeds]
    out["REINFORCE"] = {"steps to 475": [_reach(r[1]) for r in rs], "final": [r[2] for r in rs]}
    for k, v in out.items():
        print(f"  E1 {k}: final {np.mean(v['final']):.0f}, steps to 475 {v['steps to 475']}")
    return out


def e2(quick):
    out = {}
    for fs in ((1, 2) if quick else (1, 2, 4)):
        es_f = [E.train_es_cartpole(iters=40 if quick else 100, frame_skip=fs, seed=s)[2] for s in range(2 if quick else 10)]
        pg_f = [E.train_reinforce_cartpole(iters=40 if quick else 100, frame_skip=fs, seed=s)[2] for s in range(2 if quick else 10)]
        out[fs] = {"ES": es_f, "REINFORCE": pg_f}
        print(f"  E2 frame_skip={fs}: ES {np.mean(es_f):.0f}, REINFORCE {np.mean(pg_f):.0f}")
    return out


def e3(quick):
    out = {}
    for shaping in (True, False):
        for anti in (True, False):
            finals = []
            for s in range(2 if quick else 10):
                theta0 = 0.3 * np.random.default_rng(s).standard_normal(E.n_params(8))
                F = lambda P, s=s: E.cartpole_returns(P, seed=s, hidden=8)[0]
                th, _ = E.es_optimize(F, theta0, iters=30 if quick else 100, shaping=shaping, antithetic=anti, seed=s)
                finals.append(float(E.cartpole_returns(np.repeat(th[None], 10, 0), seed=99, hidden=8)[0].mean()))
            out[f"ranks={shaping} mirrored={anti}"] = finals
            print(f"  E3 ranks={shaping} mirrored={anti}: {np.mean(finals):.0f}")
    return out


def e4(quick):
    out = {}
    for pairs in ((10, 40) if quick else (5, 10, 20, 40, 80, 160)):
        t = time.time()
        E.train_es_cartpole(iters=3, n_pairs=pairs)
        per = (time.time() - t) / 3
        out[pairs] = per
        print(f"  E4 population {2 * pairs}: {per * 1000:.1f} ms per update on one process")
    return out


def e5(quick):
    out = {}
    for noise in ((0.3,) if quick else (0.1, 0.3, 1.0)):
        res, true = E.estimator_variance_vs_T(Ts=(10, 100, 1000) if quick else (10, 100, 1000, 10000), noise=noise)
        out[noise] = res
        print(f"  E5 noise={noise}: " + ", ".join(f"T={T} PG {v['PG var / true^2']:.0f} ES {v['ES var / true^2']:.1f}"
                                                  for T, v in res.items()))
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
