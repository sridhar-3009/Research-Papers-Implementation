"""Experiments for the Malicious-Use-of-AI toys (defender's side).

  E1  Cost-benefit sensitivity: tailoring cost x tailored success probability {2%, 5%, 10%} -- how robust is the
      'expansion' conclusion to the model's assumptions?
  E2  Adversarial robustness curve: FGSM eps {0, 0.025, ..., 0.3} for standard vs adversarially trained models with
      training eps {0.05, 0.1, 0.2}; clean-accuracy cost of each.
  E3  Poisoning: random flip rates {0, 10, ..., 50%} -- accuracy without a defence and with loss-based
      sanitisation (dropping as many points as were flipped).
  E4  Detecting targeted poisoning: a per-class audit (compare each class's training-label frequency and the model's
      confusion on a small trusted clean set) -- does a 100-image trusted set reveal the 7 -> 1 attack?

All are small numpy runs (seconds); none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import security as S

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def e1(quick):
    out = {}
    for pt in ((0.05,) if quick else (0.02, 0.05, 0.1)):
        for c in ((20, 0.2) if quick else (20, 5, 1, 0.2, 0.05)):
            r = S.attack_economics(c, p_tailored=pt)
            out[f"p_tailored={pt} cost={c}"] = r
            print(f"  E1 p_tailored={pt} cost={c}: tailored {r['tailored']:.1%}, harm {r['expected harm']:.0f}")
    return out


def e2(quick):
    Xtr, ytr, Xte, yte = S.load_digits()
    out = {}
    models = {"standard": S.train_softmax(Xtr, ytr)}
    for te in ((0.1,) if quick else (0.05, 0.1, 0.2)):
        models[f"adv-trained eps={te}"] = S.train_softmax(Xtr, ytr, adv_eps=te)
    for name, (W, b) in models.items():
        out[name] = [S.accuracy(W, b, S.fgsm(W, b, Xte, yte, e), yte) for e in np.arange(0, 0.31, 0.1 if quick else 0.025)]
        print(f"  E2 {name}: {np.round(out[name], 3)}")
    return out


def e3(quick):
    Xtr, ytr, Xte, yte = S.load_digits()
    rng = np.random.default_rng(0)
    out = {}
    for f in ((0.3,) if quick else (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
        yp = S.poison_labels(ytr, f, rng)
        out[f"random {f}"] = {"none": S.accuracy(*S.train_softmax(Xtr, yp), Xte, yte),
                              "sanitised": S.accuracy(*S.sanitize(Xtr, yp, max(f, 0.05))[0], Xte, yte)}
        print(f"  E3 random flips {f}: {out[f'random {f}']}")
    return out


def e4(quick):
    Xtr, ytr, Xte, yte = S.load_digits()
    rng = np.random.default_rng(0)
    yt = S.poison_labels(ytr, 1.0, rng, targeted=True)
    W, b = S.train_softmax(Xtr, yt)
    trusted = rng.choice(len(yte), 100, replace=False)
    pred = (Xte[trusted] @ W + b).argmax(1)
    per_class = {c: float((pred[yte[trusted] == c] == c).mean()) for c in range(10) if (yte[trusted] == c).any()}
    label_freq = np.bincount(yt, minlength=10) / len(yt)
    print(f"  E4 per-class accuracy on 100 trusted images: {per_class}")
    print(f"  E4 training-label frequencies: {np.round(label_freq, 3)}")
    return {"per_class_trusted_accuracy": per_class, "label_freq": label_freq.tolist()}


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
