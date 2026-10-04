"""Technical-debt experiments (Sculley et al. 2014): each warning swept over seeds and strengths, so the size of the
effect is measured rather than shown once.

  E1  CACE: correlation between x0 and x1 in {0, 0.3, 0.7, 0.95} x 10 seeds -- how far do the other weights move when
      x0 is removed? (Entanglement should grow with correlation.)
  E2  Legacy features: share of old products {0.1 ... 0.9} x number of products -- loss after the old numbers stop
      being populated, vs a model trained on new numbers only; and what leave-one-group-out ablation reports.
  E3  Hidden feedback loop: quality boost {0.2, 0.4, 0.6, 0.8} x 10 seeds -- size of the prediction-bias spike in the
      improvement week and the drift of the x_week weight over 20 weeks.
  E4  Correction cascade / thresholds / correlations: 20 seeds each, mean and spread of the damage, plus the effect of
      the mitigations (retrain the correction, learn the threshold on held-out data, causal-only features).

All are small numpy runs (seconds to a minute each); none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import debt as D

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    seeds = range(3 if quick else 10)
    out = {}
    for rho in (0.0, 0.3, 0.7, 0.95):
        shifts = []
        for s in seeds:
            rng = np.random.default_rng(s)
            n = 5000 if quick else 20000
            z = rng.standard_normal((n, 2))
            x1 = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
            X = np.column_stack([z[:, 0], x1, z[:, 1], rng.standard_normal(n)])
            y = (rng.random(n) < D.sigmoid(1.2 * z[:, 0] - 0.8 * z[:, 1])).astype(float)
            base = D.fit_logreg(X, y)
            w = D.fit_logreg(X[:, 1:], y)
            shifts.append(float(np.abs(w[:3] - base[1:4]).max()))
        out[str(rho)] = {"mean shift": float(np.mean(shifts)), "std": float(np.std(shifts))}
        print(f"  E1 rho={rho:.2f}: other weights move by {np.mean(shifts):.3f} +- {np.std(shifts):.3f}")
    return out


def e2(quick):
    out = {}
    for n_products in ((50,) if quick else (50, 100, 300)):
        for share in ((0.3, 0.7) if quick else (0.1, 0.3, 0.5, 0.7, 0.9)):
            def world(sample_seed=None, old_populated=True):
                rng = np.random.default_rng(0)
                q = rng.standard_normal(n_products)
                old = rng.random(n_products) < share
                rng = np.random.default_rng(1000 if sample_seed is None else sample_seed + 1000)
                pid = rng.integers(0, n_products, 20000)
                new_oh = np.eye(n_products)[pid]
                old_oh = new_oh * old[pid][:, None] if old_populated else np.zeros_like(new_oh)
                return np.hstack([old_oh, new_oh]), (rng.random(20000) < D.sigmoid(2 * q[pid])).astype(float)
            Xtr, ytr = world()
            Xte, yte = world(1)
            Xc, yc = world(1, False)
            both = D.fit_logreg(Xtr, ytr, lr=5.0)
            new = D.fit_logreg(Xtr[:, n_products:], ytr, lr=5.0)
            _, abl = D.ablation(Xtr, ytr, {"old": range(n_products)}, Xte, yte, lr=5.0)
            r = {"both": D.log_loss(both, Xte, yte), "both after cleanup": D.log_loss(both, Xc, yc),
                 "new only": D.log_loss(new, Xte[:, n_products:], yte), "ablation old": abl["old"]}
            out[f"{n_products}/{share}"] = r
            print(f"  E2 products={n_products} old share={share}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    return out


def e3(quick):
    out = {}
    for boost in ((0.6,) if quick else (0.2, 0.4, 0.6, 0.8)):
        spikes, drifts = [], []
        for s in range(2 if quick else 10):
            ctr, bias, w = D.feedback_loop(weeks=8 if quick else 20, quality_boost=boost, seed=s)
            spikes.append(bias[3])
            drifts.append(float(np.ptp(w[1:])))
        out[str(boost)] = {"bias spike": float(np.mean(spikes)), "x_week weight range": float(np.mean(drifts))}
        print(f"  E3 boost={boost}: bias spike {np.mean(spikes):+.3f}, x_week weight range {np.mean(drifts):.3f}")
    return out


def e4(quick):
    seeds = range(3 if quick else 20)
    rows = {"cascade damage": [], "fixed-threshold precision": [], "re-learned precision": [], "correlated acc": [],
            "causal acc": []}
    for s in seeds:
        cc = D.correction_cascade(seed=s)
        rows["cascade damage"].append(cc["A' with a v2 + OLD correction"] - cc["A' with a v1 + correction"])
        td = D.threshold_drift(seed=s)
        rows["fixed-threshold precision"].append(td["v2 with v1's fixed threshold"])
        rows["re-learned precision"].append(td["v2 precision (re-learned)"])
        cb = D.correlation_break(seed=s)
        rows["correlated acc"].append(cb["accuracy after decoupling"])
        rows["causal acc"].append(cb["causal-only model after"])
    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in rows.items()}
    for k, v in out.items():
        print(f"  E4 {k}: {v['mean']:.3f} +- {v['std']:.3f}")
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
