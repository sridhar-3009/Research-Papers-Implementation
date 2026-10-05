"""AlphaGo experiments at small scale: the pipeline's ablations on solvable Connect-K boards.

  E1  Search strength vs simulations {10, 25, 50, 100, 200} x lambda {0, 0.25, 0.5, 0.75, 1}: perfect-move rate on
      decisive positions, and score against the perfect player over 100 games (Figure 4b analogue).
  E2  Priors: SL vs RL policy as the PUCT prior (the paper: SL priors worked better), and c_puct {0.5, 1, 2, 5, 10}.
  E3  Value-network data: all positions vs one-per-game at equal size, and one-per-game at 2x / 5x / 10x the size.
  E4  RL self-play: iterations {0 ... 200} and opponent pool on/off -- win rate vs the SL policy.
  E5  A harder board: Connect-4 on 4 x 5 (a draw with perfect play; solving takes ~15 s and ~2.5M positions).

E1-E4 take a few minutes; E5 longer; none were run when this file was written.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import alphago as A

HERE = Path(__file__).parent
OUT = HERE / "results.json"


def build(g, quick, seed=0):
    rng = np.random.default_rng(seed)
    X, M, Y, B = A.expert_games(g, 500 if quick else 1500, rng)
    sl = A.train_policy(A.Net(X.shape[1], g.C, seed=seed), X, M, Y, epochs=60 if quick else 100, lr=1e-2)
    rp = A.RolloutPolicy().fit(g, B[:1000 if quick else 2000], list(Y[:1000 if quick else 2000]))
    rl = A.rl_self_play(g, sl, iters=30 if quick else 60, lr=3e-3, seed=seed)
    Xv, Zv, _ = A.self_play_positions(g, rl, 2000 if quick else 8000, np.random.default_rng(seed + 2), True)
    vn = A.train_value(Xv, Zv, epochs=40)
    _, _, _, Bt = A.expert_games(g, 100 if quick else 300, np.random.default_rng(seed + 9))
    return sl, rp, rl, vn, Bt


def e1(quick):
    g = A.ConnectK()
    sl, rp, rl, vn, Bt = build(g, quick)
    out = {}
    for sims in ((25,) if quick else (10, 25, 50, 100, 200)):
        for lam in ((0.0, 1.0) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)):
            a = A.mcts_agent(sl, vn, rp, sims=sims, lam=lam)
            rate = A.optimal_move_rate(a, g, Bt[:200 if quick else 600])
            score = A.match(g, a, A.perfect_agent, 10 if quick else 100, np.random.default_rng(1))
            out[f"sims={sims} lambda={lam}"] = {"perfect-move rate": rate, "score vs perfect": score}
            print(f"  E1 sims={sims} lambda={lam}: perfect-move rate {rate:.3f}, score vs perfect {score:.2f}")
    return out


def e2(quick):
    g = A.ConnectK()
    sl, rp, rl, vn, Bt = build(g, quick)
    out = {}
    for name, prior in (("SL prior", sl), ("RL prior", rl)):
        for c in ((5.0,) if quick else (0.5, 1.0, 2.0, 5.0, 10.0)):
            rate = A.optimal_move_rate(A.mcts_agent(prior, vn, rp, sims=50, lam=0.5, c_puct=c), g, Bt[:200 if quick else 600])
            out[f"{name} c_puct={c}"] = rate
            print(f"  E2 {name} c_puct={c}: perfect-move rate {rate:.3f}")
    return out


def e3(quick):
    g = A.ConnectK()
    sl, rp, rl, vn, Bt = build(g, quick)
    XT, ZT, _ = A.self_play_positions(g, rl, 2000, np.random.default_rng(3), True)
    XA, ZA, _ = A.self_play_positions(g, rl, 500 if quick else 1500, np.random.default_rng(1), False)
    out = {}
    v = A.train_value(XA, ZA, epochs=20)
    out["all positions"] = {"n": len(ZA), "train": A.mse(v, XA, ZA), "test": A.mse(v, XT, ZT)}
    for mult in ((1,) if quick else (1, 2, 5, 10)):
        XB, ZB, _ = A.self_play_positions(g, rl, mult * len(ZA), np.random.default_rng(10 + mult), True)
        v = A.train_value(XB, ZB, epochs=20)
        out[f"one per game x{mult}"] = {"n": len(ZB), "train": A.mse(v, XB, ZB), "test": A.mse(v, XT, ZT)}
    for k, r in out.items():
        print(f"  E3 {k}: n={r['n']}, train {r['train']:.3f}, test {r['test']:.3f}")
    return out


def e4(quick):
    g = A.ConnectK()
    rng = np.random.default_rng(0)
    X, M, Y, _ = A.expert_games(g, 500 if quick else 1500, rng)
    sl = A.train_policy(A.Net(X.shape[1], g.C, seed=0), X, M, Y, epochs=60 if quick else 100, lr=1e-2)
    out = {}
    for iters in ((10, 30) if quick else (0, 20, 60, 120, 200)):
        for pool in ((5,) if quick else (5, 10 ** 9)):
            rl = A.rl_self_play(g, sl, iters=iters, lr=3e-3, pool_every=pool) if iters else sl
            s = A.match(g, A.policy_agent(rl), A.policy_agent(sl), 100 if quick else 400, np.random.default_rng(2))
            out[f"iters={iters} pool={'on' if pool < 1e6 else 'off'}"] = s
            print(f"  E4 iters={iters} pool={'on' if pool < 1e6 else 'off'}: score vs SL {s:.3f}")
    return out


def e5(quick):
    g = A.ConnectK(4, 5, 4)
    t = time.time()
    v = g.solve(g.empty())
    print(f"  E5 Connect-4 on 4x5: value {v}, {g.solve.cache_info().currsize:,} positions, {time.time() - t:.1f} s")
    if quick:
        return {"value": v}
    sl, rp, rl, vn, Bt = build(g, False)
    rates = {lam: A.optimal_move_rate(A.mcts_agent(sl, vn, rp, sims=100, lam=lam), g, Bt[:400]) for lam in (0.0, 0.5, 1.0)}
    print(f"  E5 perfect-move rates by lambda: {rates}")
    return {"value": v, "rates": rates}


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
