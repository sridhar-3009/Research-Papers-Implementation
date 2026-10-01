"""Reproduce the main experiments of Sutskever's thesis (2013), Chapters 3-4, at small scale.

  E1  Figure 4.1: a 100-unit tanh RNN trained with HF, with and without structural damping, on the
      pathological problems (addition, multiplication, xor, temporal order, 3-bit temporal order,
      random permutation, 5-bit and 20-bit memorization) for T = 30, 50, 100. Success = < 1% of
      10,000 test sequences wrong. Records the number of HF iterations to success.
      Thesis: HF solves them for T up to 200; structural damping is essential for memorization at T > 50.
  E2  Chapter 7's message as a control: the same addition problem with plain SGD + momentum
      (well-initialized), to compare with HF.
  E3  Chapter 3: RTRBM vs the same model with W' = 0 (no recurrence: an independent RBM per frame)
      on 30x30 bouncing-balls videos, 400 hidden units, CD-10 -> CD-25. Next-frame mean squared
      prediction error per pixel. Thesis: RTRBM 0.007 vs TRBM 0.04.

Thesis HF settings: gradient on 10,000 sequences, curvature on 1,000, at most 300 CG steps,
lambda0 = 0.1 and mu = 1/30 with structural damping (lambda0 = 0.3 without).

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick         # fewer sequences / problems / steps, ~30-60 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hf_rnn import RNN, HFStructural, loss_fn
from rtrbm import RTRBM
from tasks import PROBLEMS, addition, bouncing_balls, error_rate

HERE = Path(__file__).parent


def run_hf(problem, T, structural, a, seed=0):
    rng = np.random.default_rng(seed)
    make = PROBLEMS[problem]
    X, Y, M, kind = make(T, 4, rng)
    n_out = {"random permutation": 100, "temporal order": 4, "3-bit temporal order": 8}.get(
        problem, X.shape[-1] if "memorization" in problem else 1)          # mse problems: one linear output
    model = RNN(X.shape[-1], 100, n_out, seed=seed)
    lam0 = 0.1 if structural else 0.3
    if problem == "random permutation":
        lam0 *= T                                                          # 'we initialized lambda to 0.1 T'
    hf = HFStructural(model, kind, lam=lam0, mu=1 / 30 if structural else 0.0, max_cg=a.max_cg)
    test = make(T, a.test_seqs, np.random.default_rng(10_000 + seed))
    for it in range(1, a.max_iters + 1):
        g = make(T, a.grad_seqs, rng)[:3]
        c = tuple(t[:, :a.grad_seqs // 10] for t in g)
        hf.step(g, c, structural)
        if it % 5 == 0:
            err = error_rate(model, test)
            if err < 0.01:
                return it, err
    return None, error_rate(model, test)


def e1(a):
    out = {}
    for problem in a.problems:
        for T in a.lengths:
            for sd in (True, False):
                its, err = run_hf(problem, T, sd, a)
                key = f"{problem} | T={T} | {'structural' if sd else 'plain'} HF"
                out[key] = {"iterations to success": its, "final test error": err}
                print(f"  {key}: {out[key]}", flush=True)
    return out


def e2(a):
    """SGD + momentum (Chapter 7) on the addition problem."""
    out = {}
    for T in a.lengths:
        rng = np.random.default_rng(0)
        model = RNN(2, 100, 1)
        opt = torch.optim.SGD(model.parameters(), lr=a.sgd_lr, momentum=0.9)
        test = addition(T, a.test_seqs, np.random.default_rng(10_000))
        solved = None
        for step in range(1, a.sgd_steps + 1):
            X, Y, M, kind = addition(T, 100, rng)
            o, _ = model(X)
            loss = loss_fn(o, Y, M, kind)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 500 == 0 and error_rate(model, test) < 0.01:
                solved = step
                break
        out[f"T={T}"] = {"sgd steps to success": solved, "final test error": error_rate(model, test)}
        print(f"  E2 T={T}: {out[f'T={T}']}", flush=True)
    return out


def train_rtrbm(model, a, rng, k_switch=1000):
    steps = a.rtrbm_steps
    for step in range(steps):
        vs = bouncing_balls(a.video_len, res=a.res, rng=rng)
        lr = a.rtrbm_lr * (1 - step / steps)                               # 'linearly reduced towards 0'
        grads = model.gradient(vs, k=10 if step < k_switch else 25)
        if not hasattr(model, "vel"):
            model.vel = [np.zeros_like(p) for p in model.params()]
        for p, g, v in zip(model.params(), grads, model.vel):
            v *= 0.9
            v += lr * g / len(vs)                                          # 'the gradient was divided by the length'
            p += v                                                          # gradient ASCENT on log-likelihood
        if model.recurrent is False:
            model.Wp[:] = 0.0
        if step % 1000 == 0:
            label = "recurrent" if model.recurrent else "W'=0"
            print(f"    RTRBM ({label}) step {step}", flush=True)
    return model


def e3(a):
    out = {}
    for recurrent in (True, False):
        rng = np.random.default_rng(0)
        m = RTRBM(a.res * a.res, a.hidden, seed=0)
        m.recurrent = recurrent
        if not recurrent:
            m.Wp[:] = 0.0
        train_rtrbm(m, a, rng)
        errs = []
        for k in range(20):
            vs = bouncing_balls(a.video_len, res=a.res, rng=np.random.default_rng(1000 + k))
            errs.append(((m.predict_next(vs) - vs[1:]) ** 2).mean())
        name = "RTRBM" if recurrent else "no recurrence (W' = 0)"
        out[name] = float(np.mean(errs))
        print(f"  E3 {name}: next-frame MSE per pixel {out[name]:.4f}", flush=True)
    return out


EXPS = {"e1": e1, "e2": e2, "e3": e3}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    if R.get("e1"):
        L += ["## E1: HF on the pathological problems (Figure 4.1)", "", "| problem / length / method | HF iterations to success | test error |", "|---|---|---|"]
        L += [f"| {k} | {v['iterations to success']} | {100 * v['final test error']:.1f}% |" for k, v in R["e1"].items()] + [""]
    if R.get("e2"):
        L += ["## E2: SGD + momentum on the addition problem", ""] + [f"- {k}: {v}" for k, v in R["e2"].items()] + [""]
    if R.get("e3"):
        L += ["## E3: bouncing balls, next-frame squared error per pixel (thesis: RTRBM 0.007, TRBM 0.04)", ""]
        L += [f"- {k}: {v:.4f}" for k, v in R["e3"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--lengths", nargs="+", type=int, default=[30, 50, 100])
    ap.add_argument("--grad-seqs", type=int, default=10000)
    ap.add_argument("--test-seqs", type=int, default=10000)
    ap.add_argument("--max-cg", type=int, default=300)
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--sgd-lr", type=float, default=0.01)
    ap.add_argument("--sgd-steps", type=int, default=100000)
    ap.add_argument("--res", type=int, default=30)
    ap.add_argument("--video-len", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=400)
    ap.add_argument("--rtrbm-steps", type=int, default=100000)
    ap.add_argument("--rtrbm-lr", type=float, default=0.01)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.problems = ["addition", "temporal order", "5-bit memorization"]
        a.lengths, a.grad_seqs, a.test_seqs, a.max_cg, a.max_iters = [30], 1000, 1000, 100, 60
        a.sgd_steps, a.res, a.video_len, a.hidden, a.rtrbm_steps = 5000, 15, 30, 100, 2000
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
