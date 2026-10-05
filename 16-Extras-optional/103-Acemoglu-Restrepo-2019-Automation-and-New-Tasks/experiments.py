"""Automation-and-new-tasks experiments: the task model's comparative statics and the decomposition's reliability.

  E1  Automation's wage effect over capital productivity A_K {0.5 ... 8} for sigma {0.5, 0.8, 1.0, 1.5}: the A_K
      threshold where automation starts to raise wages (the 'so-so' boundary).
  E2  Decomposition accuracy vs shock size: planted vs estimated task content, displacement and
      reinstatement for annual shocks of 0.5x / 1x / 2x / 4x.
  E3  Sensitivity to the assumed sigma {0.6, 0.8, 1.0, 1.2} when the truth is 0.8 (the paper's Appendix robustness).
  E4  Moving-average window {1, 3, 5, 10} years and the share of industry-years with both kinds of shock {0, 0.1, 0.3,
      0.5}: how biased are the displacement / reinstatement estimates (the paper's 'lower bounds')?

All are instant numpy computations; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e4
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import tasks as T

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    out = {}
    for s in ((0.8,) if quick else (0.5, 0.8, 1.0, 1.5)):
        AKs = np.geomspace(0.5, 8, 5 if quick else 25)
        wages = [T.effects("automation", sigma=s, AK=a)["wage (d ln W)"] for a in AKs]
        cross = next((float(a) for a, w in zip(AKs, wages) if w > 0), None)
        out[s] = {"A_K": AKs.tolist(), "wage effect": wages, "so-so boundary A_K": cross}
        print(f"  E1 sigma={s}: automation starts raising wages at A_K ~ {cross}")
    return out


def e2(quick):
    out = {}
    for m in ((1.0,) if quick else (0.5, 1.0, 2.0, 4.0)):
        d = T.synthetic_industries(auto_rate=0.010 * m, new_rate=0.008 * m)
        dec = T.decompose(d)
        pd, pr = T.planted_split(d)
        out[m] = {"task content est": float(dec["task content"].mean()), "planted": float(T.planted_task_content(d).mean()),
                  "disp est": float(dec["displacement"].mean()), "disp planted": float(pd.mean()),
                  "rein est": float(dec["reinstatement"].mean()), "rein planted": float(pr.mean())}
        print(f"  E2 shocks x{m}: " + ", ".join(f"{k} {v * 100:+.3f}" for k, v in out[m].items()))
    return out


def e3(quick):
    d = T.synthetic_industries()
    out = {s: float(T.decompose(d, sigma_assumed=s)["task content"].mean()) for s in ((0.8, 1.0) if quick else (0.6, 0.8, 1.0, 1.2))}
    print(f"  E3 assumed sigma -> task content (%/yr): {{{', '.join(f'{k}: {v * 100:+.3f}' for k, v in out.items())}}}")
    return out


def e4(quick):
    out = {}
    for both in ((0.0, 0.3) if quick else (0.0, 0.1, 0.3, 0.5)):
        d = T.synthetic_industries(both_prob=both)
        pd, pr = T.planted_split(d)
        for w in ((5,) if quick else (1, 3, 5, 10)):
            dec = T.decompose(d, window=w)
            out[f"both={both} window={w}"] = {"disp ratio": float(dec["displacement"].mean() / pd.mean()),
                                              "rein ratio": float(dec["reinstatement"].mean() / pr.mean())}
            print(f"  E4 both={both} window={w}: estimated / planted displacement "
                  f"{out[f'both={both} window={w}']['disp ratio']:.2f}, reinstatement {out[f'both={both} window={w}']['rein ratio']:.2f}")
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
