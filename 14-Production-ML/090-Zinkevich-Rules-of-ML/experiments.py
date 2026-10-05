"""Rules-of-ML experiments: the demo's checks swept over seeds and settings, to see how robust each effect is.

  E1  Rule 1: heuristic's share of the ML gain over 20 seeds and taste strengths {0.5, 1.5, 3} -- when does the
      'about 50%' rule of thumb hold?
  E2  Rule 21: examples {100 ... 100k} x feature sets {30, 330, 330 + 1000 triple crosses} x L2 -- the crossover
      point where more feature weights start paying off.
  E3  Rule 30 / Rule 34: sampling rate {0.01 ... 1} with and without weights; held-out share {0.1%, 1%, 5%} --
      calibration error and log-loss, 10 seeds.
  E4  Rule 33 / Rule 37: optimism of random vs temporal validation as the size of daily effects grows; the skew
      decomposition as the joined table drifts more.

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

import rules as R

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    out = {}
    for s in range(3 if quick else 20):
        r = R.rule1_heuristic_vs_ml(seed=s)
        out[s] = r["heuristic share of the ML gain"]
    print(f"  E1 heuristic share of the ML gain: mean {np.mean(list(out.values())):.2f}, "
          f"range {min(out.values()):.2f}-{max(out.values()):.2f}")
    return out


def e2(quick):
    sizes = (300, 3000) if quick else (100, 300, 1000, 3000, 10000, 30000, 100000)
    out = {str(k): v for k, v in R.rule21_features_vs_data(sizes=sizes).items()}
    for n, v in out.items():
        print(f"  E2 n={n}: " + ", ".join(f"{k} {x:.3f}" for k, x in v.items()))
    return out


def e3(quick):
    out = {}
    for keep in ((0.3,) if quick else (0.01, 0.05, 0.1, 0.3, 1.0)):
        r = R.rule30_importance_weighting(n=20000 if quick else 60000, keep=keep)
        out[f"keep={keep}"] = {k: v["mean prediction"] - v["true rate"] for k, v in r.items()}
        print(f"  E3 keep={keep}: calibration error " + ", ".join(f"{k} {v:+.3f}" for k, v in out[f'keep={keep}'].items()))
    for h in ((0.01,) if quick else (0.001, 0.01, 0.05)):
        r = R.rule34_filter_holdout(holdout=h)
        k = "trained on 1% held-out traffic"
        out[f"holdout={h}"] = {"examples": r[k]["examples"], "log-loss": r[k]["log-loss"],
                               "oracle log-loss": r["oracle: all traffic labelled"]["log-loss"]}
        print(f"  E3 holdout={h}: {out[f'holdout={h}']}")
    return out


def e4(quick):
    out = {}
    for s in range(2 if quick else 10):
        r = R.rule33_temporal_split(seed=s)
        out[f"rule33 seed {s}"] = {k: v["actual log-loss on future days"] - v["estimate log-loss"] for k, v in r.items()}
        print(f"  E4 seed {s}: optimism (actual - estimate) " +
              ", ".join(f"{k} {v:+.3f}" for k, v in out[f'rule33 seed {s}'].items()))
        a = R.rule37_skew(seed=s)
        out[f"rule37 seed {s}"] = a
        print(f"     skew: " + " -> ".join(f"{k} {v:.3f}" for k, v in a.items()))
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
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
