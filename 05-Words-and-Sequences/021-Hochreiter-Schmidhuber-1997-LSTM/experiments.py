"""Reproduce the main experiments of Hochreiter & Schmidhuber (1997), Section 5, with the 1997 LSTM
(no forget gate, truncated gradient) against a conventional RNN trained with full BPTT.

  E1  Experiment 1 (Table 1): embedded Reber grammar, next-symbol prediction. Success = all legal
      next symbols predicted (outputs > 0.5 exactly for legal symbols) on test strings.
      Paper: LSTM succeeds in ~97-100% of trials; RTRL/BPTT/Elman rarely.
  E2  Experiment 2a (Table 2): remember the first symbol across p = 4, 10, 100 steps.
      Paper: RTRL 79% at p = 4, 0% at p = 10; BPTT 0% at p >= 10; LSTM 100% even at p = 100.
  E4  Experiment 4 (Table 7): adding problem with T = 100 (minimal lag 50).
      Paper: LSTM, after ~74,000 sequences, gets 1 out of 2560 test sequences wrong (error > 0.04).
  E5  Experiment 5: multiplication problem, T = 100.
  E6  Experiment 6a: temporal order of two symbols ~40 steps apart (4 classes).
  E7  Extra: the same adding problem with a MODERN LSTM (with a forget gate, full BPTT, Adam),
      torch.nn.LSTM, to compare with the 1997 design.

The paper trains ON-LINE (one sequence at a time, plain SGD). That is what --batch 1 does; it is
faithful but slow in Python. Larger --batch is faster.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # small budgets, ~20-40 minutes
       python3 experiments.py --only e4
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lstm import LSTM1997Torch, VanillaRNN
from tasks import adding_problem, embedded_reber, multiplication, noise_free_2a, temporal_order

HERE = Path(__file__).parent


def pad(seqs):
    """List of (T_i, d) arrays -> (T_max, N, d) tensor and a (T_max, N) mask of real steps."""
    T = max(len(s) for s in seqs)
    X = np.zeros((T, len(seqs), seqs[0].shape[1]))
    M = np.zeros((T, len(seqs)), bool)
    for i, s in enumerate(seqs):
        X[:len(s), i] = s
        M[:len(s), i] = True
    return torch.tensor(X, dtype=torch.float32), torch.tensor(M)


def lstm(n_in, blocks, size, n_out, init, in_bias=None, out_bias=None, seed=0):
    return LSTM1997Torch(n_in, blocks, size, n_out, truncate=True, init=init, in_gate_bias=in_bias,
                         out_gate_bias=out_bias, seed=seed)


# ---------------------------------------------------------------------------

def e1(a):
    """4 blocks of size 1 (as in the paper's '4-1' architecture), gate biases -1, -2, -3, -4."""
    out = {}
    for name, make in (("LSTM 1997", lambda s: lstm(7, 4, 1, 7, 0.2, out_bias=np.array([-1., -2., -3., -4.]), seed=s)),
                       ("RNN (BPTT)", lambda s: VanillaRNN(7, 12, 7, seed=s))):
        results = []
        for trial in range(a.trials):
            rng = np.random.default_rng(trial)
            net = make(trial)
            opt = torch.optim.SGD(net.parameters(), lr=0.5)
            solved_at = None
            for n in range(1, a.max_seqs + 1, a.batch):
                batch = [embedded_reber(rng) for _ in range(a.batch)]
                X, M = pad([b[1] for b in batch])
                Y, _ = pad([b[2] for b in batch])
                loss = 0.5 * (((net(X) - Y) ** 2).sum(-1) * M).sum() / a.batch
                opt.zero_grad(); loss.backward(); opt.step()
                if n % (50 * a.batch) < a.batch:
                    test = [embedded_reber(np.random.default_rng(10_000 + k)) for k in range(256)]
                    Xt, Mt = pad([b[1] for b in test]); Yt, _ = pad([b[2] for b in test])
                    with torch.no_grad():
                        ok = (((net(Xt) > 0.5).float() == Yt).all(-1) | ~Mt).all(0).float().mean().item()
                    if ok == 1.0:
                        solved_at = n
                        break
            results.append(solved_at)
            print(f"  E1 {name} trial {trial}: solved after {solved_at} sequences", flush=True)
        out[name] = {"success rate": float(np.mean([r is not None for r in results])),
                     "mean sequences (successful)": float(np.mean([r for r in results if r])) if any(results) else None}
    return out


def e2(a):
    out = {}
    for p in (4, 10, 100):
        for name, make in (("LSTM 1997", lambda s: lstm(p + 1, 1, 1, p + 1, 0.2, seed=s)),
                           ("RNN (BPTT)", lambda s: VanillaRNN(p + 1, 3, p + 1, seed=s))):
            wins = []
            for trial in range(a.trials):
                rng = np.random.default_rng(trial)
                net = make(trial)
                opt = torch.optim.SGD(net.parameters(), lr=1.0)
                done = None
                for n in range(1, a.max_seqs + 1):
                    X, Y = noise_free_2a(p, rng)
                    loss = 0.5 * ((net(torch.tensor(X, dtype=torch.float32)[:, None]) [:, 0] - torch.tensor(Y, dtype=torch.float32)) ** 2).sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                    if n % 100 == 0:
                        with torch.no_grad():
                            good = all(
                                (net(torch.tensor(noise_free_2a(p, np.random.default_rng(k))[0], dtype=torch.float32)[:, None])[-1, 0]
                                 - torch.tensor(noise_free_2a(p, np.random.default_rng(k))[1][-1])).abs().max() < 0.25
                                for k in range(20))
                        if good:
                            done = n
                            break
                wins.append(done)
            out[f"p={p} {name}"] = {"success rate": float(np.mean([w is not None for w in wins])),
                                    "mean sequences (successful)": float(np.mean([w for w in wins if w])) if any(wins) else None}
            print(f"  E2 p={p} {name}: {out[f'p={p} {name}']}", flush=True)
    return out


def regression_task(make_seq, net, a, lr, tol=0.04):
    """Train on fresh random sequences; report test sequences with |error| > tol out of 2560."""
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    log = []
    for n in range(0, a.max_seqs, a.batch):
        batch = [make_seq(rng) for _ in range(a.batch)]
        X, M = pad([b[0] for b in batch])
        lengths = M.sum(0) - 1
        y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        out = net(X)[lengths, torch.arange(a.batch), 0]                          # error only at the sequence end
        loss = 0.5 * ((out - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        log.append(loss.item())
    test_rng = np.random.default_rng(99)
    test = [make_seq(test_rng) for _ in range(2560)]
    wrong = 0
    with torch.no_grad():
        for s in range(0, 2560, 256):
            X, M = pad([b[0] for b in test[s:s + 256]])
            out = net(X)[M.sum(0) - 1, torch.arange(len(test[s:s + 256])), 0]
            wrong += ((out - torch.tensor([b[1] for b in test[s:s + 256]])).abs() > tol).sum().item()
    return {"wrong out of 2560": int(wrong), "final train loss": float(np.mean(log[-50:]))}


def e4(a):
    T = a.T
    make = lambda r: adding_problem(T, r)
    return {"LSTM 1997": regression_task(make, lstm(2, 2, 2, 1, 0.1, in_bias=np.array([-3., -6.])), a, lr=0.5),
            "RNN (BPTT)": regression_task(make, VanillaRNN(2, 8, 1, init=0.1), a, lr=0.5)}


def e5(a):
    make = lambda r: multiplication(a.T, r)
    return {"LSTM 1997": regression_task(make, lstm(2, 2, 2, 1, 0.1, in_bias=np.array([-3., -6.])), a, lr=0.1)}


def e6(a):
    net = lstm(8, 2, 2, 4, 0.1, in_bias=np.array([-2., -4.]), out_bias=np.array([-2., -4.]))
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    rng = np.random.default_rng(0)
    for n in range(0, a.max_seqs, a.batch):
        batch = [temporal_order(rng) for _ in range(a.batch)]
        X, M = pad([b[0] for b in batch])
        Y = torch.nn.functional.one_hot(torch.tensor([b[1] for b in batch]), 4).float()
        out = net(X)[M.sum(0) - 1, torch.arange(a.batch)]
        loss = 0.5 * ((out - Y) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    test = [temporal_order(np.random.default_rng(10_000 + k)) for k in range(2560)]
    X, M = pad([b[0] for b in test])
    with torch.no_grad():
        pred = net(X)[M.sum(0) - 1, torch.arange(2560)].argmax(1)
    return {"wrong out of 2560": int((pred != torch.tensor([b[1] for b in test])).sum())}


class ModernLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm, self.out = nn.LSTM(2, 16), nn.Linear(16, 1)

    def forward(self, x):
        return torch.sigmoid(self.out(self.lstm(x)[0]))


def e7(a):
    """The same adding problem with torch.nn.LSTM (forget gate) and full BPTT."""
    net = ModernLSTM()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    for n in range(0, a.max_seqs, a.batch):
        batch = [adding_problem(a.T, rng) for _ in range(a.batch)]
        X, M = pad([b[0] for b in batch])
        out = net(X)[M.sum(0) - 1, torch.arange(a.batch), 0]
        loss = 0.5 * ((out - torch.tensor([b[1] for b in batch], dtype=torch.float32)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    test = [adding_problem(a.T, np.random.default_rng(99 + k)) for k in range(2560)]
    X, M = pad([b[0] for b in test])
    with torch.no_grad():
        out = net(X)[M.sum(0) - 1, torch.arange(2560), 0]
    return {"wrong out of 2560": int(((out - torch.tensor([b[1] for b in test])).abs() > 0.04).sum())}


EXPS = {"e1": e1, "e2": e2, "e4": e4, "e5": e5, "e6": e6, "e7": e7}


def report(R, a):
    L = ["# Results", "", f"trials {a.trials}, budget {a.max_seqs} sequences, batch {a.batch}" + (" (QUICK run)" if a.quick else ""), ""]
    titles = {"e1": "E1: embedded Reber grammar (Table 1)", "e2": "E2: task 2a, remember the first symbol (Table 2)",
              "e4": f"E4: adding problem, T = {a.T} (paper: 1 of 2560 wrong)", "e5": f"E5: multiplication, T = {a.T}",
              "e6": "E6: temporal order (6a)", "e7": "E7: modern LSTM (forget gate, Adam) on the adding problem"}
    for k, title in titles.items():
        if R.get(k):
            L += [f"## {title}", ""] + [f"- {name}: {v}" for name, v in R[k].items()] + [""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.trials = a.trials or (2 if a.quick else 10)
    a.max_seqs = a.max_seqs or (20_000 if a.quick else 200_000)
    if a.quick:
        a.batch = max(a.batch, 16)
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
