"""BCTF experiments on planted relational data (the paper's real datasets are not downloaded here).

  E1  Density sweep: training fraction {1, 0.3, 0.1, 0.05, 0.03, 0.01} x 5 seeds -- RMSE / PR-AUC of MAP, BTF, BCTF and
      the block model (where does the Bayesian advantage appear?).
  E2  Dimensionality d {2, 3, 5, 8} at 10% density: MAP overfits more as d grows; do BTF / BCTF stay stable?
  E3  Cluster recovery: adjusted Rand index of BCTF's object / relation partitions vs truth, over densities and seeds,
      and vs the inverse-Gamma prior scale (fraction of the MAP-vector variance {0.01, 0.05, 0.2}).
  E4  Sampler length: sweeps {20, 60, 150} -- does more MCMC improve prediction or cluster recovery?

Each comparison takes ~2-5 s; the full set takes several minutes; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import bctf as B

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    out = {}
    for frac in ((0.1,) if quick else (1.0, 0.3, 0.1, 0.05, 0.03, 0.01)):
        runs = [B.compare(train_frac=frac, seed=s, sweeps=30 if quick else 60)[0] for s in range(1 if quick else 5)]
        out[frac] = {k: [float(np.mean([r[k][0] for r in runs])), float(np.mean([r[k][1] for r in runs]))] for k in runs[0]}
        print(f"  E1 frac={frac}: " + ", ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in out[frac].items()))
    return out


def e2(quick):
    out = {}
    for d in ((3,) if quick else (2, 3, 5, 8)):
        res = B.compare(train_frac=0.1, d=d, seed=0, sweeps=30 if quick else 60)[0]
        out[d] = {k: list(v) for k, v in res.items()}
        print(f"  E2 d={d}: " + ", ".join(f"{k} {v[0]:.3f}" for k, v in res.items()))
    return out


def e3(quick):
    out = {}
    T, co_t, cr_t = B.planted_data()
    mean = T.mean()
    for frac in ((0.1,) if quick else (1.0, 0.1, 0.03)):
        train, test = B.split(T, 0.1, frac)
        t = T[tuple(train.T)] - mean
        init = B.fit_map(train, t, 60, 6, 3)
        v = float(np.var(np.hstack([init[0], init[2]])))
        for scale in ((0.05,) if quick else (0.01, 0.05, 0.2)):
            _, co, cr = B.run_mcmc(train, t, 60, 6, 3, True, sweeps=30 if quick else 60, burn=10, init=init, test=test,
                                   beta=scale * v)
            out[f"frac={frac} scale={scale}"] = [B.adjusted_rand(co, co_t), B.adjusted_rand(cr, cr_t)]
            print(f"  E3 frac={frac} prior scale={scale}: object ARI {out[f'frac={frac} scale={scale}'][0]:.2f}, "
                  f"relation ARI {out[f'frac={frac} scale={scale}'][1]:.2f}")
    return out


def e4(quick):
    out = {}
    for sweeps in ((20,) if quick else (20, 60, 150)):
        res, cl, _ = B.compare(train_frac=0.1, seed=0, sweeps=sweeps, burn=sweeps // 3)
        out[sweeps] = {"BCTF": list(res["BCTF"]), "object ARI": cl["BCTF objects"][1]}
        print(f"  E4 sweeps={sweeps}: BCTF {res['BCTF'][0]:.3f} / {res['BCTF'][1]:.3f}, object ARI {cl['BCTF objects'][1]:.2f}")
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
