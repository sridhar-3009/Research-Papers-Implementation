"""ML Test Score experiments (Breck et al. 2017): how reliable are the automated tests themselves?

  E1  Detection vs bug severity: for the skew / upstream-unit / degraded-model bugs, sweep the size of the bug (unit
      factor 1.01 ... 100, weight noise 0.02 ... 1.0) x 20 seeds -- the detection rate of each test that should
      catch it (Data 1 / Monitor 2 / Monitor 3 / Monitor 7 / Infra 4 / Infra 6).
  E2  False alarms: 50 healthy seeds -- how often does each test fire with NO bug (the paper: alert thresholds need
      careful tuning to balance false positives and false negatives)?
  E3  Threshold tuning: for the schema test and the canary bias limit, sweep the threshold and trace the trade-off
      between false-alarm rate (E2 setup) and detection rate on a small bug.
  E4  Staleness: the age-vs-quality curve over 104 weeks for 3 drift speeds, and the tolerable age at gap < 0.01.

All are small numpy runs (a few seconds to a few minutes); none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mltestscore as M

HERE = Path(__file__).parent
OUT = HERE / "results.json"
COLS = M.BASE_SPEC["features"]


def _fires(seed, unit=1.0, serving_unit=1.0, noise=0.0, out_max=0.01, shift_max=0.25, bias_max=0.05):
    tr, va, live = M.raw_world(seed=seed), M.raw_world(seed=seed + 1), M.raw_world(seed=seed + 2)
    live["spend_raw"] = live["spend_raw"] * unit
    model = M.train(M.BASE_SPEC, tr)
    if noise:
        model.w = model.w + np.random.default_rng(seed).normal(0, noise, len(model.w))
    schema = M.make_schema(tr)
    Xs = M.featurize(live, COLS, serving=True)
    Xs[:, 0] = np.log1p(live["spend_raw"] * serving_unit)
    skew = np.abs(Xs - M.featurize(live, COLS)).max() > 1e-9
    bias = float(model.predict(Xs).mean() - live["y"].mean())
    reg = M.Registry()
    reg.push(M.train(M.BASE_SPEC, tr), va)
    return {"schema (Data 1 / Monitor 2)": bool(M.check_schema(schema, live, out_max, shift_max)),
            "skew (Monitor 3)": bool(skew), "served bias (Monitor 7)": abs(bias) > bias_max,
            "validation (Infra 4)": reg.push(model, va)[0] == "VETOED"}


def e1(quick):
    seeds = range(3 if quick else 20)
    out = {}
    sweeps = [("upstream unit", "unit", (1.01, 1.1, 2.0, 100.0)), ("serving unit", "serving_unit", (1.01, 1.1, 2.0, 100.0)),
              ("weight noise", "noise", (0.02, 0.1, 0.3, 1.0))]
    for name, kw, vals in sweeps:
        for v in (vals[1:3] if quick else vals):
            rates = {}
            for s in seeds:
                for k, f in _fires(s, **{kw: v}).items():
                    rates[k] = rates.get(k, 0) + f / len(seeds)
            out[f"{name}={v}"] = rates
            print(f"  E1 {name}={v}: " + ", ".join(f"{k} {r:.2f}" for k, r in rates.items()))
    return out


def e2(quick):
    seeds = range(5 if quick else 50)
    rates = {}
    for s in seeds:
        for k, f in _fires(s + 1000).items():
            rates[k] = rates.get(k, 0) + f / len(seeds)
    print("  E2 false-alarm rates: " + ", ".join(f"{k} {r:.2f}" for k, r in rates.items()))
    return rates


def e3(quick):
    seeds = range(3 if quick else 30)
    out = {}
    for shift_max in ((0.05, 0.25) if quick else (0.02, 0.05, 0.1, 0.25, 0.5)):
        fa = np.mean([_fires(s + 2000, shift_max=shift_max, out_max=1.0)["schema (Data 1 / Monitor 2)"] for s in seeds])
        det = np.mean([_fires(s, unit=1.1, shift_max=shift_max, out_max=1.0)["schema (Data 1 / Monitor 2)"] for s in seeds])
        out[f"schema shift_max={shift_max}"] = {"false alarm": float(fa), "detect x1.1": float(det)}
        print(f"  E3 schema mean-shift limit {shift_max}: false alarms {fa:.2f}, detects x1.1 units {det:.2f}")
    for bias_max in ((0.02, 0.05) if quick else (0.01, 0.02, 0.05, 0.1)):
        fa = np.mean([_fires(s + 2000, bias_max=bias_max)["served bias (Monitor 7)"] for s in seeds])
        det = np.mean([_fires(s, noise=0.3, bias_max=bias_max)["served bias (Monitor 7)"] for s in seeds])
        out[f"bias_max={bias_max}"] = {"false alarm": float(fa), "detect noise 0.3": float(det)}
        print(f"  E3 bias limit {bias_max}: false alarms {fa:.2f}, detects weight noise 0.3 {det:.2f}")
    return out


def e4(quick):
    out = {}
    ages = (0, 12, 52) if quick else (0, 4, 8, 12, 26, 52, 78, 104)
    for drift in ((0.03,) if quick else (0.01, 0.03, 0.1)):
        rows = {}
        m = M.train(M.BASE_SPEC, M.raw_world(seed=0))
        for a in ages:
            # the drift speed scales time: t = a * drift / 0.03 in the world's own units
            t = a * drift / 0.03
            w = M.raw_world(t=t, seed=100 + a)
            fresh = M.train(M.BASE_SPEC, M.raw_world(t=t, seed=200 + a))
            X = M.featurize(w, COLS)
            rows[a] = M.log_loss(m.predict(X), w["y"]) - M.log_loss(fresh.predict(X), w["y"])
        tol = max([a for a, g in rows.items() if g < 0.01], default=0)
        out[str(drift)] = {"gap by age": rows, "tolerable": tol}
        print(f"  E4 drift {drift}: gaps " + ", ".join(f"{a}w {g:+.3f}" for a, g in rows.items()) + f"; tolerable {tol} weeks")
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
