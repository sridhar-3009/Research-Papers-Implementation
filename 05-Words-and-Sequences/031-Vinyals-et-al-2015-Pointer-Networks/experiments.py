"""Reproduce Pointer Networks (Vinyals, Fortunato & Jaitly 2015): Table 1 (convex hull), Sec. 4.3 (Delaunay),
Table 2 (TSP).

  E1  Table 1, convex hull. LSTM / LSTM+attention / Ptr-Net trained and tested on n = 50; LSTM trained on n = 5 and
      n = 10; one Ptr-Net trained on n uniform in 5..50 and tested on n = 5, 10, 50, 100, 200, 500.
      Paper: n = 50: LSTM 1.9% (area FAIL), +attention 38.9% (99.7%), Ptr-Net 72.6% (99.9%);
             Ptr-Net 5-50 -> n=5 92.0%, 10 87.0%, 50 69.6%, 100 50.3%, 200 22.1%, 500 1.3% (area 99.2%).
  E2  Delaunay, Ptr-Net trained per n. Paper: n = 5 accuracy 80.7% / triangle coverage 93.0%; n = 10: 22.6% / 81.3%;
      n = 50: 0% / 52.8%.
  E3  Table 2, TSP (decoding constrained to valid tours). Ptr-Net trained on optimal tours for n = 5 and n = 10;
      on A1 and on A3 tours for n = 50; one Ptr-Net trained on 5..20 and tested on 5..50.
      Paper: n = 5 2.12 (optimal 2.12), n = 10 2.88 (2.87); n = 50: A1-trained 6.42 (A1 itself 6.46),
      A3-trained 6.09 (A3 5.79); 5-20-trained -> n = 30 4.72 (A3 4.60), n = 50 7.66.

Setup (Sec. 4.1): 1-layer LSTM with 256 (or 512) units, SGD lr 1.0, batch 128, uniform(-0.08, 0.08) init, gradient
norm clipped at 2, 1M training pairs. Deviations, all flags:
  * inputs pass through a learned linear embedding (--no-embed for the literal setup; see ptrnet.py);
  * exact TSP labels (Held-Karp) are used up to n = --exact-max (default 12; the paper went to 20, which needs
    O(2^20 * 20^2) per example). For the '5-20' model, sizes above that are labelled by A3 (Christofides + 2-opt);
  * A1 / A2 / A3 are our stand-ins for the paper's three GitHub solvers (greedy edge, nearest neighbour + 2-opt,
    Christofides + 2-opt).

!! HEAVY (1M examples per model, many models). Not run on the author's laptop.
       python3 experiments.py --quick          # 20k examples, small models, ~1 hour on CPU
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ptrnet import PtrNet, beam_search, targets
from tasks import (A1, A2, A3, convex_hull, delaunay, delaunay_metrics, held_karp, hull_metrics, is_valid_tour,
                   sample_points, tour_length)

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- data
def hull_target(P):
    return convex_hull(P)


def delaunay_target(P):
    return [i for tri in delaunay(P) for i in tri]                 # triangles flattened, 3 pointers each


def tsp_target(label, exact_max):
    def f(P):
        if label == "optimal" or (label == "mixed" and len(P) <= exact_max):
            return held_karp(P)
        return {"A1": A1, "A3": A3, "mixed": A3}[label](P)
    return f


def make_data(task, sizes, n_examples, seed, target):
    """A fixed training set (the paper generated 1M pairs), each example with n drawn from `sizes`."""
    rng = np.random.default_rng(seed)
    data = []
    for k in range(n_examples):
        n = int(rng.choice(sizes))
        P = sample_points(n, rng)
        data.append((P, target(P)))
        if (k + 1) % 100000 == 0:
            print(f"    {task}: {k + 1} examples", flush=True)
    return data


def batches(data, bs, rng):
    """Batches of equal n (so no padding is needed on the input side)."""
    by_n = {}
    for i, (P, _) in enumerate(data):
        by_n.setdefault(len(P), []).append(i)
    chunks = []
    for idx in by_n.values():
        idx = list(rng.permutation(idx))
        chunks += [idx[i:i + bs] for i in range(0, len(idx), bs)]
    rng.shuffle(chunks)
    for c in chunks:
        yield torch.tensor([data[i][0] for i in c], dtype=torch.float), targets([data[i][1] for i in c])


# ---------------------------------------------------------------------------------------------------- training
def train(data, a, mode="ptr", n_max=50, tag=""):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = PtrNet(a.hidden, mode=mode, n_max=n_max, embed=not a.no_embed).to(DEV)
    opt = (torch.optim.SGD(m.parameters(), lr=a.lr) if a.opt == "sgd" else torch.optim.Adam(m.parameters(), lr=a.lr))
    for ep in range(a.epochs):
        tot = cnt = 0
        for P, C in batches(data, a.batch, rng):
            loss = -m.log_prob(P.to(DEV), C.to(DEV)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)         # 'L2 gradient clipping of 2.0'
            opt.step()
            tot += loss.item(); cnt += 1
        print(f"    {tag} {mode} epoch {ep + 1}: loss {tot / cnt:.3f}", flush=True)
    return m.eval().cpu()


def decode(m, Ps, beam, constraint=None):
    return [beam_search(m, torch.tensor(P, dtype=torch.float), beam=beam, constraint=constraint) for P in Ps]


def test_set(n, k, seed=123):
    rng = np.random.default_rng(seed + n)
    return [sample_points(n, rng) for _ in range(k)]


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    out = {}
    d50 = make_data("hull", [50], a.n_train, 0, hull_target)
    T50 = test_set(50, a.n_test)
    for mode, name in (("seq2seq", "LSTM"), ("attention", "+ATTENTION"), ("ptr", "PTR-NET")):
        m = train(d50, a, mode, n_max=50, tag="hull n=50")
        out[f"{name}, trained 50, n=50"] = hull_metrics(T50, decode(m, T50, a.beam))
    for n in (5, 10):
        m = train(make_data("hull", [n], a.n_train, n, hull_target), a, "seq2seq", n_max=n, tag=f"hull n={n}")
        T = test_set(n, a.n_test)
        out[f"LSTM, trained {n}, n={n}"] = hull_metrics(T, decode(m, T, a.beam))
    m = train(make_data("hull", list(range(5, 51)), a.n_train, 1, hull_target), a, "ptr", tag="hull n=5-50")
    for n in (5, 10, 50, 100, 200, 500):
        T = test_set(n, a.n_test if n <= 100 else a.n_test // 5)
        out[f"PTR-NET, trained 5-50, n={n}"] = hull_metrics(T, decode(m, T, a.beam))
        print("  E1", n, out[f"PTR-NET, trained 5-50, n={n}"], flush=True)
    return out


def e2(a):
    out = {}
    for n in (5, 10, 50):
        m = train(make_data("delaunay", [n], a.n_train, n, delaunay_target), a, "ptr", tag=f"delaunay n={n}")
        T = test_set(n, a.n_test)
        preds = []
        for seq in decode(m, T, a.beam):
            preds.append([tuple(seq[i:i + 3]) for i in range(0, len(seq) - 2, 3)])
        out[f"n={n}"] = delaunay_metrics(T, preds)
        print("  E2", n, out[f"n={n}"], flush=True)
    return out


def e3(a):
    out = {}

    def row(m, n, label):
        T = test_set(n, a.n_test_tsp)
        tours = decode(m, T, a.beam_tsp, constraint="tour")
        assert all(is_valid_tour(t, n) for t in tours)
        r = {"Ptr-Net": float(np.mean([tour_length(P, t) for P, t in zip(T, tours)]))}
        for name, algo in (("A1", A1), ("A2", A2), ("A3", A3)):
            r[name] = float(np.mean([tour_length(P, algo(P)) for P in T]))
        if n <= a.exact_max:
            r["optimal"] = float(np.mean([tour_length(P, held_karp(P)) for P in T]))
        out[label] = r
        print("  E3", label, r, flush=True)

    for n in (5, 10):
        m = train(make_data("tsp", [n], a.n_train, n, tsp_target("optimal", a.exact_max)), a, tag=f"tsp n={n}")
        row(m, n, f"n={n} (optimal-trained)")
    for lab in ("A1", "A3"):
        m = train(make_data("tsp", [50], a.n_train, 50, tsp_target(lab, a.exact_max)), a, tag=f"tsp n=50 {lab}")
        row(m, 50, f"n=50 ({lab}-trained)")
    m = train(make_data("tsp", list(range(5, 21)), a.n_train, 7, tsp_target("mixed", a.exact_max)), a, tag="tsp 5-20")
    for n in (5, 10, 20, 25, 30, 40, 50):
        row(m, n, f"n={n} (5-20 trained)")
    return out


def report(R, a):
    L = ["# Results", "", f"hidden {a.hidden}, {a.n_train} training pairs per model" + (" (QUICK run)" if a.quick else ""), ""]
    for k, title in (("e1", "E1: convex hull (Table 1)"), ("e2", "E2: Delaunay (Sec. 4.3)"), ("e3", "E3: TSP tour lengths (Table 2)")):
        if R.get(k):
            L += [f"## {title}", "", "| setting | result |", "|---|---|"]
            L += [f"| {s} | {json.dumps(v)} |" for s, v in R[k].items()] + [""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--opt", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=1_000_000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-test-tsp", type=int, default=200)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--beam-tsp", type=int, default=10)
    ap.add_argument("--exact-max", type=int, default=12)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.hidden, a.n_train, a.n_test, a.n_test_tsp, a.epochs, a.opt, a.lr = 128, 20000, 200, 50, 5, "adam", 1e-2
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
