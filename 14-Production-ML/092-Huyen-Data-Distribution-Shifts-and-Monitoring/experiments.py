"""Distribution-shift experiments (after Huyen's post): how robust are the demo's conclusions?

  E1  Shift monitors over 20 seeds and shift strengths (covariate mean shift 0 ... 20 years, label-shift positive
      share 0.2 ... 1.0, concept-drift mixing 0 ... 1): detection rate of input KS, prediction KS, and accuracy drop.
  E2  Importance weighting: target shift size x model mis-specification -- when do weights help, and how small does
      the effective sample size get before they hurt?
  E3  Alert policies: window {1, 2, 3, 6, 12, 24} h x z {3, 4, 5} x duration {1, 2, 3} over 20 simulated fortnights
      with outages of random size and length -- false alarms per week vs detection delay (the trade-off curve).
  E4  Retraining cadence: retrain every {1, 3, 7, 14} days with each strategy on a stream with 2 concept drifts;
      average log-loss over the stream.
  E5  Feedback loops: exploration rate {0, 1%, 5%, 10%, 20%} x 200 rounds: CTR over time, diversity, regret.

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

import drift as D

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    out = {}
    for s in range(2 if quick else 20):
        t = D.shift_table(n=2000, seed=s)
        out[s] = {k: {"input alarm": v["inputs KS p"] < 0.05, "prediction alarm": v["predictions KS p"] < 0.05,
                      "accuracy drop": v["accuracy source"] - v["accuracy target"]} for k, v in t.items()}
    for k in ("source", "covariate", "label", "concept"):
        ia = np.mean([out[s][k]["input alarm"] for s in out])
        pa = np.mean([out[s][k]["prediction alarm"] for s in out])
        ad = np.mean([out[s][k]["accuracy drop"] for s in out])
        print(f"  E1 {k:10s}: input alarm {ia:.2f}, prediction alarm {pa:.2f}, mean accuracy drop {ad:+.3f}")
    return out


def e2(quick):
    out = {}
    for s in range(2 if quick else 20):
        r = D.importance_weighting(seed=s)
        out[s] = {k: v["target log-loss"] for k, v in r.items() if "target log-loss" in v}
    gain = np.mean([out[s]["unweighted"] - out[s]["weights from a domain classifier"] for s in out])
    print(f"  E2 mean log-loss gain from estimated weights: {gain:+.3f}")
    return out


def e3(quick):
    out = {}
    rng = np.random.default_rng(0)
    runs = [(int(rng.integers(24 * 8, 24 * 12)), int(rng.integers(2, 12)), float(rng.uniform(0.04, 0.15)))
            for _ in range(3 if quick else 20)]
    for win in ((3, 6) if quick else (1, 2, 3, 6, 12, 24)):
        for z in ((4.0,) if quick else (3.0, 4.0, 5.0)):
            for dur in ((1, 3) if quick else (1, 2, 3)):
                fa, delays = 0, []
                for i, (start, length, dip) in enumerate(runs):
                    s, _ = D.stream(dip_start=start, dip_hours=length, dip=dip, seed=i)
                    a = D.window_alerts(s, win, z=z, duration=dur)
                    hit = [h for h in a if start <= h < start + length + win]
                    fa += len(a) - len(hit)
                    delays.append(hit[0] - start if hit else np.nan)
                key = f"win={win} z={z} dur={dur}"
                out[key] = {"false alarms per week": fa / len(runs), "detected": float(np.mean(~np.isnan(delays))),
                            "median delay": float(np.nanmedian(delays)) if not np.all(np.isnan(delays)) else None}
                print(f"  E3 {key}: {out[key]}")
    return out


def e4(quick):
    out = {}
    for d in ((25,) if quick else (21, 22, 23, 25, 27, 29)):
        out[d] = D.retraining_strategies(eval_days=(d,))[d]
        print(f"  E4 day {d}: " + ", ".join(f"{k} {v:.3f}" for k, v in out[d].items()))
    return out


def e5(quick):
    out = {}
    for e in ((0.0, 0.1) if quick else (0.0, 0.01, 0.05, 0.1, 0.2)):
        rs = [D.feedback_recommender(explore=e, rounds=40 if quick else 200, seed=s) for s in range(2 if quick else 10)]
        out[e] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        print(f"  E5 explore={e}: " + ", ".join(f"{k} {v:.3f}" for k, v in out[e].items()))
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
