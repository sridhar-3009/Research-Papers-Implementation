"""Model-card experiments: how stable are disaggregated numbers, and how much evaluation data does a card need?

  E1  Confidence-interval width vs group size: bootstrap 95% CI width of FPR / FNR for groups of 50 ... 5,000
      examples -- how small can an intersectional group be before the card's numbers are mostly noise?
  E2  Seeds: re-run the smiling toy over 20 seeds -- does 'older men have the highest FDR' hold every time?
  E3  Toxicity toy over 5 seeds: mean BPSN AUC of the targeted terms for v1 and the mitigated v2 -- is the
      improvement stable?
  E4  Write every card to disk as Markdown (cards/ folder) so they can be read and diffed across versions.

All are small numpy runs (seconds); none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import modelcard as M

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    rng = np.random.default_rng(0)
    out = {}
    for n in ((50, 500) if quick else (50, 100, 200, 500, 1000, 2000, 5000)):
        y = (rng.random(n) < 0.5).astype(int)
        p = np.where(rng.random(n) < 0.85, y, 1 - y)
        ci = M.bootstrap_ci(y, p, reps=500)
        out[n] = {k: ci[k][1] - ci[k][0] for k in ("FPR", "FNR")}
        print(f"  E1 n={n}: CI width FPR {out[n]['FPR']:.3f}, FNR {out[n]['FNR']:.3f}")
    return out


def e2(quick):
    hits = []
    for s in range(2 if quick else 20):
        _, rows, _ = M.smiling_card(seed=s, reps=100)
        r = {k: v[0] for k, v in rows.items()}
        hits.append(max((k for k in r if " " in k), key=lambda k: r[k]["FDR"]) == "male old")
    print(f"  E2 'older men have the highest FDR' held in {np.mean(hits):.0%} of seeds")
    return {"share": float(np.mean(hits))}


def e3(quick):
    out = {}
    for s in range(1 if quick else 5):
        t = M.toxicity_versions(seed=s, pinned_draws=5)
        out[s] = {k: float(np.mean([v["per term"][x]["BPSN AUC"] for x in M.TARGETED])) for k, v in t.items()}
        print(f"  E3 seed {s}: mean BPSN AUC of targeted terms {out[s]}")
    return out


def e4(quick):
    d = HERE / "cards"
    d.mkdir(exist_ok=True)
    card, _, _ = M.smiling_card(reps=200 if quick else 1000)
    (d / "smiling.md").write_text(card.to_markdown())
    print(f"  E4 wrote {d / 'smiling.md'}")
    return {"written": [str(d / "smiling.md")]}


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
