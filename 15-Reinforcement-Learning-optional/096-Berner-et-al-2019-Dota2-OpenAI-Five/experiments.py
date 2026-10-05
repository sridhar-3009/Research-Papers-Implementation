"""OpenAI Five ablations at toy scale (PPO + GAE on CartPole), mirroring Figure 5 and Appendices M and O.

  E1  Batch size (Figure 5a): episodes per version {4, 8, 16, 32, 64, 128} x 10 seeds -- versions to reach a mean
      training return of 200.
  E2  Staleness (Figure 5b): {0, 1, 2, 4, 8, 16, 32} versions x 10 seeds.
  E3  Sample reuse (Figure 5c): {1, 2, 4, 8, 16} x 10 seeds.
  E4  Surgery vs restart over 5 stages and 10 seeds; and the recurrent-style widening scale {0.001 ... 1} vs the
      largest change in action probabilities.
  E5  Team spirit tau {0, 0.3, 0.5, 0.75, 1} over 20 seeds: mean team reward over the first 100 steps.

E1-E4 take a few minutes each; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import five as F

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def _versions(seeds, target=200, **kw):
    return [F.train_ppo(updates=120, seed=s, target=target, **kw)[1] for s in seeds]


def e1(quick):
    seeds = range(2 if quick else 10)
    out = {}
    for n in ((8, 32) if quick else (4, 8, 16, 32, 64, 128)):
        out[n] = _versions(seeds, n_envs=n)
        print(f"  E1 batch {n}: versions to 200 {out[n]}")
    return out


def e2(quick):
    seeds = range(2 if quick else 10)
    out = {}
    for k in ((0, 8) if quick else (0, 1, 2, 4, 8, 16, 32)):
        out[k] = _versions(seeds, staleness=k)
        print(f"  E2 staleness {k}: {out[k]}")
    return out


def e3(quick):
    seeds = range(2 if quick else 10)
    out = {}
    for k in ((1, 8) if quick else (1, 2, 4, 8, 16)):
        out[k] = _versions(seeds, reuse=k)
        print(f"  E3 reuse {k}: {out[k]}")
    return out


def e4(quick):
    out = {"surgery vs restart": [F.surgery_vs_restart(stages=3 if quick else 5, seed=s) for s in range(1 if quick else 10)]}
    rng = np.random.default_rng(0)
    p = F.init_mlp(6, 16, 2, rng)
    X = rng.standard_normal((1000, 6))
    out["recurrent scale"] = {sc: F.max_policy_change(p, F.surgery_widen_recurrent(p, 32, rng, sc), X)
                              for sc in (0.001, 0.01, 0.03, 0.1, 0.3, 1.0)}
    print(f"  E4 recurrent-style widening: {out['recurrent scale']}")
    return out


def e5(quick):
    out = {}
    for tau in ((0.0, 1.0) if quick else (0.0, 0.3, 0.5, 0.75, 1.0)):
        curves = [F.team_spirit(tau, seed=s)[0] for s in range(2 if quick else 20)]
        out[tau] = float(np.mean([np.mean(c[:100]) for c in curves]))
        print(f"  E5 tau={tau}: mean team reward over the first 100 steps {out[tau]:.3f}")
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
