"""Supervised-contrastive experiments on 8x8 digits (and optionally on a larger dataset).

  E1  Losses x seeds: cross-entropy, SupCon L_out, SupCon L_in, self-supervised (SimCLR) over 5 seeds at 100 and 200
      epochs -- mean and spread of test accuracy (is the SupCon / cross-entropy gap bigger than seed noise?).
  E2  Temperature tau {0.05, 0.1, 0.2, 0.5, 1.0} for L_out and L_in (the paper's Fig. 4d: lower is better until
      training becomes unstable).
  E3  Batch size {32, 64, 128, 256} for L_out vs cross-entropy (the paper's Fig. 4b: both benefit from larger batches).
  E4  Robustness: the corruption table over 5 seeds for cross-entropy and L_out.
  E5  Hyperparameter sensitivity: learning rate {3e-4, 1e-3, 3e-3, 1e-2} -- spread of
      accuracy per method (the paper: SupCon is less sensitive).

Each training run takes ~3-7 s; E1-E5 take a few minutes; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import supcon as S

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def run(method, Xtr, ytr, Xte, yte, seed=0, epochs=100, lr=3e-3, tau=0.1, batch=128):
    if method == "ce":
        p = S.train_cross_entropy(Xtr, ytr, epochs=epochs, lr=lr, seed=seed, batch=batch)
        return ("ce", p), S.ce_accuracy(p, Xte, yte)
    p, _ = S.train_contrastive(Xtr, ytr, method, tau=tau, epochs=epochs, lr=lr, seed=seed, batch=batch)
    W, b = S.linear_probe(p, Xtr, ytr)
    return ("probe", (p, W, b)), S.probe_accuracy(p, W, b, Xte, yte)


def e1(quick):
    Xtr, ytr, Xte, yte = S.load_digits_split()
    out = {}
    for epochs in ((30,) if quick else (100, 200)):
        for m in ("ce", "out", "in", "self"):
            accs = [run(m, Xtr, ytr, Xte, yte, seed=s, epochs=epochs)[1] for s in range(1 if quick else 5)]
            out[f"{m} epochs={epochs}"] = accs
            print(f"  E1 {m} epochs={epochs}: {np.mean(accs):.3f} +- {np.std(accs):.3f}")
    return out


def e2(quick):
    Xtr, ytr, Xte, yte = S.load_digits_split()
    out = {}
    for tau in ((0.1, 0.5) if quick else (0.05, 0.1, 0.2, 0.5, 1.0)):
        for m in ("out", "in"):
            out[f"{m} tau={tau}"] = run(m, Xtr, ytr, Xte, yte, tau=tau, epochs=30 if quick else 100)[1]
            print(f"  E2 {m} tau={tau}: {out[f'{m} tau={tau}']:.3f}")
    return out


def e3(quick):
    Xtr, ytr, Xte, yte = S.load_digits_split()
    out = {}
    for batch in ((64,) if quick else (32, 64, 128, 256)):
        for m in ("ce", "out"):
            out[f"{m} batch={batch}"] = run(m, Xtr, ytr, Xte, yte, batch=batch, epochs=30 if quick else 100)[1]
            print(f"  E3 {m} batch={batch}: {out[f'{m} batch={batch}']:.3f}")
    return out


def e4(quick):
    Xtr, ytr, Xte, yte = S.load_digits_split()
    out = {}
    for s in range(1 if quick else 5):
        models = {m: run(m, Xtr, ytr, Xte, yte, seed=s, epochs=30 if quick else 100)[0] for m in ("ce", "out")}
        for kind, levels in (("noise", (0.2, 0.4)), ("blur", (0.8, 1.2)), ("shift", (0.5, 1.0))):
            for l in levels:
                Xc = S.corrupt(Xte, kind, l, seed=s)
                for m, (k, mod) in models.items():
                    a = S.ce_accuracy(mod, Xc, yte) if k == "ce" else S.probe_accuracy(*mod, Xc, yte)
                    out.setdefault(f"{m} {kind} {l}", []).append(a)
    for k, v in out.items():
        print(f"  E4 {k}: {np.mean(v):.3f}")
    return out


def e5(quick):
    Xtr, ytr, Xte, yte = S.load_digits_split()
    out = {}
    for m in ("ce", "out"):
        accs = [run(m, Xtr, ytr, Xte, yte, lr=lr, epochs=30 if quick else 100)[1]
                for lr in ((1e-3, 3e-3) if quick else (3e-4, 1e-3, 3e-3, 1e-2))]
        out[m] = accs
        print(f"  E5 {m}: accuracies over learning rates {np.round(accs, 3)}, spread {np.ptp(accs):.3f}")
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
