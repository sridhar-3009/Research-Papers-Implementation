"""Encoding-vs-training experiments on 8x8 digits (and optionally CIFAR-10 if it is on disk).

  E1  The full Table 1 analogue with hyperparameter search, as in the paper: for each (dictionary, encoder) pair take
      the best of lambda {0.5, 0.75, 1.0, 1.25, 1.5} (SC), k {1, 2, 5, 10} (OMP), alpha {0.1, 0.25, 0.5, 1.0} (T),
      over 5 label draws.
  E2  Dictionary size d {16, 48, 128, 256}: the paper's 'bigger dictionaries help' (81.5% at d = 6000 on CIFAR) with the
      cheap OMP-1 + soft-threshold combination.
  E3  Labelled examples per class {5, 10, 20, 50, 100}: when does feature learning beat raw pixels?
  E4  Cost: wall-clock time of each training and encoding method per 10,000 patches.
  E5  (optional) CIFAR-10 with 6x6 colour patches if `cifar-10-batches-py/` is next to this file (not downloaded here).

E1 takes a few minutes; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

import encoding as E

HERE = Path(__file__).parent
OUT = HERE / "results.json"
GRID = {"SC": [("SC", l) for l in (0.5, 0.75, 1.0, 1.25, 1.5)], "OMP": [("OMP", k) for k in (1, 2, 5, 10)],
        "T": [("T", a) for a in (0.1, 0.25, 0.5, 1.0)]}


def e1(quick):
    images, y = E.load_digits_images()
    P = E.Pipeline(images)
    rng = np.random.default_rng(0)
    out = {}
    for tn, kw in E.TRAINERS.items():
        D = E.train_dictionary(P.sample[:8000], 48, rng=rng, **kw)
        for en, opts in GRID.items():
            opts = opts[:2] if quick else opts
            best = max((E.few_label_accuracy(P.features(D, m, pa), y, splits=2 if quick else 5), f"{m}={pa}")
                       for m, pa in opts)
            out[f"{tn} / {en}"] = best
            print(f"  E1 {tn:6s} / {en:3s}: {best[0]:.3f} ({best[1]})")
    return out


def e2(quick):
    images, y = E.load_digits_images()
    P = E.Pipeline(images)
    out = {}
    for d in ((16, 48) if quick else (16, 48, 128, 256)):
        D = E.train_dictionary(P.sample[:8000], d, rng=np.random.default_rng(0), method="OMP", k=1)
        out[d] = E.few_label_accuracy(P.features(D, "T", 0.25), y)
        print(f"  E2 d={d}: OMP-1 dictionary + soft threshold {out[d]:.3f}")
    return out


def e3(quick):
    images, y = E.load_digits_images()
    P = E.Pipeline(images)
    D = E.train_dictionary(P.sample[:8000], 48, rng=np.random.default_rng(0), method="RP")
    F = P.features(D, "T", 0.25)
    raw = images.reshape(len(y), -1)
    out = {}
    for n in ((5, 20) if quick else (5, 10, 20, 50, 100)):
        out[n] = {"raw": E.few_label_accuracy(raw, y, per_class=n), "RP + T": E.few_label_accuracy(F, y, per_class=n)}
        print(f"  E3 {n} labels/class: raw {out[n]['raw']:.3f}, RP + T {out[n]['RP + T']:.3f}")
    return out


def e4(quick):
    images, y = E.load_digits_images()
    P = E.Pipeline(images)
    X = P.sample[:10000]
    out = {}
    for tn, kw in E.TRAINERS.items():
        t = time.time(); D = E.train_dictionary(X, 48, rng=np.random.default_rng(0), **kw); out[f"train {tn}"] = time.time() - t
    for en, (m, pa) in E.ENCODERS.items():
        t = time.time(); E.encode(X, D, m, pa); out[f"encode {en}"] = time.time() - t
    for k, v in out.items():
        print(f"  E4 {k}: {v * 1000:.0f} ms")
    return out


def e5(quick):
    path = HERE / "cifar-10-batches-py"
    if not path.exists():
        print("  E5 skipped: put the CIFAR-10 python batches in ./cifar-10-batches-py to run it")
        return {"skipped": True}
    with open(path / "data_batch_1", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    imgs = d[b"data"].reshape(-1, 3, 32, 32).mean(1)[:2000 if quick else 10000] / 255.0
    y = np.array(d[b"labels"])[:len(imgs)]
    P = E.Pipeline(imgs, p=6)
    D = E.train_dictionary(P.sample[:20000], 200, rng=np.random.default_rng(0), method="OMP", k=1)
    acc = E.few_label_accuracy(P.features(D, "T", 0.25), y, per_class=100)
    print(f"  E5 CIFAR-10 grey, OMP-1 + T, 100 labels/class: {acc:.3f}")
    return {"accuracy": acc}


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
