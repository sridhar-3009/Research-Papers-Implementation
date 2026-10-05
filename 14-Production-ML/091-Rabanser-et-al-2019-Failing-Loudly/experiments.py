"""Failing Loudly experiments on 8x8 digits (and optionally MNIST): the paper's full grid at laptop scale.

  E1  Table 1a: all 14 DR / test pairs x all 10 shift types (s/m/l gn, s/m/l img, ko, adv, m_img+ko, oz+m_img) x
      delta {0.1, 0.5, 1.0} x sample sizes {10, 20, 50, 100, 200, 400} x 5 splits x 20 draws.
  E2  Table 1b / 2: per-shift and per-strength detection for the best univariate (BBSDs) and multivariate methods;
      and per-fraction delta.
  E3  Latent size K in {4, 8, 16, 32} for PCA / SRP / UAE / TAE (the paper uses 32 on 784-3072 dims).
  E4  Malignancy: across all shifts, does 'accuracy on the domain classifier's top-k anomalous samples' track the true
      accuracy on the shifted target? Correlation over shifts, for k in {10, 20, 50}.
  E5  (optional, needs MNIST via sklearn.datasets.fetch_openml, ~55 MB download) the same pipeline on MNIST with a
      larger MLP, sizes up to 1,000 -- closer to the paper's setting.

E1 is the long one (~30-60 minutes on a laptop); E2-E4 take minutes; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

import shiftdetect as S

warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful")
HERE = Path(__file__).parent
OUT = HERE / "results.json"
ALL_SHIFTS = ["s_gn", "m_gn", "l_gn", "s_img", "m_img", "l_img", "ko", "adv", "m_img+ko", "oz+m_img"]


def _setup(split):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = S.load_data(seed=split)
    return S.Reducers(Xtr, ytr, seed=split), Xva, yva, Xte, yte


def e1(quick):
    splits = range(1 if quick else 5)
    sizes = (10, 50) if quick else (10, 20, 50, 100, 200, 400)
    shifts = ["l_gn", "ko"] if quick else ALL_SHIFTS
    deltas = (0.5,) if quick else (0.1, 0.5, 1.0)
    reps = 3 if quick else 20
    acc = {f"{m}-{t}": np.zeros(len(sizes)) for m, t in S.METHODS}
    count = 0
    for sp in splits:
        red, Xva, yva, Xte, yte = _setup(sp)
        for sh in shifts:
            for d in deltas:
                Xs, _ = S.apply_shift(Xte, yte, sh, d, red, seed=sp)
                count += 1
                for m, t in S.METHODS:
                    acc[f"{m}-{t}"] += [S.detection_rate(red, Xva, Xs, m, t, n, reps, 100, seed=sp) for n in sizes]
    out = {k: (v / count).tolist() for k, v in acc.items()}
    for k, v in out.items():
        print(f"  E1 {k:14s} " + " ".join(f"{x:.2f}" for x in v))
    return {"sizes": list(sizes), "rates": out}


def e2(quick):
    red, Xva, yva, Xte, yte = _setup(0)
    sizes = (10, 50) if quick else (10, 20, 50, 100, 200, 400)
    out = {}
    for sh in (["m_img"] if quick else ALL_SHIFTS):
        for d in ((0.5,) if quick else (0.1, 0.5, 1.0)):
            Xs, _ = S.apply_shift(Xte, yte, sh, d, red)
            for m, t in (("BBSDs", "univ"), ("UAE", "mmd")):
                key = f"{sh} delta={d} {m}-{t}"
                out[key] = [S.detection_rate(red, Xva, Xs, m, t, n, 3 if quick else 20, 100) for n in sizes]
                print(f"  E2 {key}: {out[key]}")
    return out


def e3(quick):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = S.load_data()
    out = {}
    for K in ((8,) if quick else (4, 8, 16, 32)):
        red = S.Reducers(Xtr, ytr, K=K)
        rates = {}
        for m in ("PCA", "SRP", "UAE", "TAE"):
            rates[m] = float(np.mean([S.detection_rate(red, Xva, S.apply_shift(Xte, yte, sh, 0.5, red)[0], m, "univ",
                                                       100, 3 if quick else 20) for sh in ("m_gn", "m_img", "adv")]))
        out[K] = rates
        print(f"  E3 K={K}: {rates}")
    return out


def e4(quick):
    red, Xva, yva, Xte, yte = _setup(0)
    out = {}
    for k in ((20,) if quick else (10, 20, 50)):
        true_acc, est_acc = [], []
        for sh in (["m_img", "ko"] if quick else ALL_SHIFTS):
            Xs, ys = S.apply_shift(Xte, yte, sh, 0.5, red)
            true_acc.append(float((red.clf.proba(Xs).argmax(1) == ys).mean()))
            n = min(200, len(ys))
            est_acc.append(S.most_anomalous(Xva[:200], Xs[:n], ys[:n], red, top=k)[2])
        r = float(np.corrcoef(true_acc, est_acc)[0, 1])
        out[k] = {"true": true_acc, "top-k estimate": est_acc, "correlation": r}
        print(f"  E4 top-{k}: correlation between true target accuracy and top-k anomalous accuracy {r:.2f}")
    return out


def e5(quick):
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        return {"skipped": "sklearn missing"}
    print("  E5 downloading MNIST via fetch_openml (needs internet, ~55 MB) ...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X, y = X / 255.0, y.astype(int)
    idx = np.random.default_rng(0).permutation(len(y))[: 6000 if quick else 30000]
    n = len(idx)
    tr, va, te = idx[: n // 2], idx[n // 2: 3 * n // 4], idx[3 * n // 4:]
    red = S.Reducers(X[tr], y[tr], K=32)
    out = {}
    for sh in ("l_gn", "ko", "adv"):                                   # image transforms here are written for 8x8
        Xs, _ = S.apply_shift(X[te], y[te], sh, 0.5, red)
        out[sh] = [S.detection_rate(red, X[va], Xs, "BBSDs", "univ", m, 5, 50) for m in (10, 100, 1000)]
        print(f"  E5 {sh}: {out[sh]}")
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
        if name == "e5" and not a.only:
            continue                                                  # opt-in: needs a download
        t = time.time()
        print(f"--- {name}")
        res[name] = fn(a.quick)
        print(f"    ({time.time() - t:.1f} s)")
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
