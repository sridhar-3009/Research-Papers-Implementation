"""Parallelized SGD experiments (Zinkevich et al. 2010): sweeps on the synthetic hashed-feature task and a real
sparse text dataset.

  E1  Figures 1-3 extended: machines k in {1, 2, 5, 10, 20, 50, 100} x lambda {1e-3, 1e-6} x loss {Huber, squared,
      logistic} x eta {0.1, 0.5, 1.0}: relative objective and test RMSE after 250 / 1000 / 2000 examples per machine.
  E2  Theory: stationary spread vs eta (Theorem 10 predicts ~eta), the averaged error vs k (variance ~1/k, bias
      floor), and 'halve eta, double T' to halve the error.
  E3  Alternatives with the same data: (a) Mann et al.-style averaging of EXACT per-machine solutions (full-batch
      solver on each shard), (b) synchronous distributed mini-batch SGD (communicates every step), (c)
      SimuParallelSGD -- final objective and number of communication rounds.
  E4  Real data: RCV1 (scikit-learn `fetch_rcv1`, ~800k documents, 47k tf-idf features, CCAT vs rest), logistic
      loss, 1 / 10 / 100 machines; relative objective and test error.

!! HEAVY for E4 (downloads RCV1, ~650 MB in memory); E1-E3 take minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import psgd as P

HERE = Path(__file__).parent


def e1(a):
    Xa, ya = P.make_data(a.m + a.m_test, seed=3)
    Xtr, ytr, Xte, yte = P.split(Xa, ya, a.m_test)
    out = []
    for kind in a.kinds:
        for lam in (1e-3, 1e-6):
            for eta in a.etas:
                v1, _, _ = P.simu_parallel_sgd(Xtr, ytr, 1, eta, lam, kind)
                ref = P.objective(v1, Xtr, ytr, lam, kind)
                ref_rmse = np.sqrt(((Xte @ v1 - yte) ** 2).mean())
                for k in a.ks:
                    _, _, snaps = P.simu_parallel_sgd(Xtr, ytr, k, eta, lam, kind, snapshots=set(a.steps))
                    for s, w in snaps.items():
                        out.append({"loss": kind, "lambda": lam, "eta": eta, "k": k, "examples per machine": s,
                                    "relative objective": float(P.objective(w, Xtr, ytr, lam, kind) / ref),
                                    "relative test RMSE": float(np.sqrt(((Xte @ w - yte) ** 2).mean()) / ref_rmse)})
    return out


def e2(a):
    X, y = P.make_data(a.m // 4, seed=1)
    lam = 1e-3
    c_min = P.objective(P.full_batch_minimiser(X, y, lam, iters=3000), X, y, lam, "huber")
    out = {}
    for eta, steps in ((1.0, 2000), (0.5, 4000), (0.25, 8000), (0.125, 16000)):
        mean, var, W = P.stationary_stats(X, y, eta, lam, runs=a.runs, steps=steps)
        row = {"spread": var, "single chain suboptimality": float((P.objective(W, X, y, lam, "huber") - c_min).mean())}
        for k in (k_ for k_ in (1, 4, 16, 64) if k_ <= a.runs):
            n = (a.runs // k) * k
            avg = W[:n].reshape(-1, k, W.shape[1]).mean(1)
            row[f"average of {k}: suboptimality"] = float((P.objective(avg, X, y, lam, "huber") - c_min).mean())
        out[f"eta={eta}, T={steps}"] = row
    return out


def e3(a):
    X, y = P.make_data(a.m, seed=4)
    lam, eta, k = 1e-3, 0.5, 10
    c_min = P.objective(P.full_batch_minimiser(X, y, lam, iters=3000), X, y, lam, "huber")
    shards = np.array_split(np.random.default_rng(0).permutation(len(y)), k)
    exact = np.mean([P.full_batch_minimiser(X[s], y[s], lam, iters=1500) for s in shards], axis=0)
    rng = np.random.default_rng(1)
    w = np.zeros(X.shape[1])
    rounds = len(y) // (k * 10)
    for _ in range(rounds):                                            # mini-batch of 10 per machine per round
        idx = rng.integers(0, len(y), size=k * 10)
        w -= eta * (X[idx].T @ P.dloss("huber", X[idx] @ w, y[idx]) / len(idx) + lam * w)
    v, _, _ = P.simu_parallel_sgd(X, y, k, eta, lam)
    sub = lambda u: float(P.objective(u, X, y, lam, "huber") - c_min)
    return {"average of exact shard solutions (1 round, expensive solver)": {"suboptimality": sub(exact), "rounds": 1},
            "synchronous mini-batch SGD": {"suboptimality": sub(w), "rounds": rounds},
            "SimuParallelSGD": {"suboptimality": sub(v), "rounds": 1}}


def e4(a):
    from sklearn.datasets import fetch_rcv1
    rcv = fetch_rcv1()
    X = rcv.data.tocsr()
    y = np.where(rcv.target[:, list(rcv.target_names).index("CCAT")].toarray().ravel() > 0, 1.0, -1.0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(X.shape[0])
    tr, te = perm[:-a.rcv_test], perm[-a.rcv_test:]
    out = {}
    for k in a.ks:
        T = len(tr) // k
        W = np.zeros((k, X.shape[1]))
        parts = tr[: k * T].reshape(k, T)
        for t in range(min(T, a.rcv_steps)):
            rows = parts[:, t]
            xb = X[rows]
            p = np.array([xb[i].dot(W[i]) for i in range(k)]).ravel()          # each machine's prediction
            g = P.dloss("logistic", p, y[rows])
            for i in range(k):
                W[i] *= (1 - a.eta * 1e-6)
                W[i, xb[i].indices] -= a.eta * g[i] * xb[i].data
        v = W.mean(0)
        pred = np.sign(X[te] @ v)
        out[k] = {"test error": float((pred != y[te]).mean())}
    return out


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 5)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.m, a.m_test, a.ks, a.kinds, a.etas = 200000, 20000, (1, 2, 5, 10, 20, 50, 100), ("huber", "squared", "logistic"), (0.1, 0.5, 1.0)
    a.steps, a.runs, a.rcv_test, a.rcv_steps, a.eta = (250, 1000, 2000), 256, 100000, 5000, 0.5
    if a.quick:
        a.m, a.m_test, a.ks, a.kinds, a.etas = 4000, 500, (1, 10), ("huber",), (0.5,)
        a.steps, a.runs, a.rcv_test, a.rcv_steps = (100,), 16, 1000, 20
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except ImportError as e:
                    Rs[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
