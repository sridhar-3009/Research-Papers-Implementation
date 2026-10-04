"""InstructGPT experiments (Ouyang et al. 2022) at toy scale.

  E1  When does RL help? Grow the reward model (more ranked prompts: 300 ... 8000; wider/deeper RM) and measure its
      pairwise accuracy and the utility of PPO vs SFT. (Our demo's RM, 66% accurate, was too weak to beat SFT.)
  E2  Section 3.5: the RM loss with all C(K,2) pairs of a prompt as ONE batch element vs the same pairs shuffled
      into independent examples: validation accuracy and log loss over epochs (the paper: shuffling overfits).
  E3  The alignment tax: gamma in {0, 0.1, 0.3, 1, 3} for PPO-ptx: instruction utility vs pre-training benchmark.
  E4  KL coefficient beta in {0, 0.01, 0.05, 0.2}: KL to SFT vs RM score vs true utility.
  E5  SFT data size (50 ... 2000 demonstrations): utility and benchmark (the cost of the alignment tax grows?).

!! HEAVY. Not run on the author's laptop (minutes per run on CPU).
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import instructgpt as I

HERE = Path(__file__).parent


def rm_acc(rm, test):
    hits = []
    for p, ranked in test:
        sc = I.score(rm, [p] * len(ranked), ranked).tolist()
        u = [I.labeler_utility(*p, r) for r in ranked]
        hits += [(sc[i] > sc[j]) == (u[i] > u[j]) for i, j in itertools.combinations(range(len(ranked)), 2) if u[i] != u[j]]
    return float(np.mean(hits))


def base_and_sft(a):
    base = I.pretrain(a.pre_steps)
    return base, I.sft(base, a.sft_steps, a.n_demos)


def e1(a):
    base, s = base_and_sft(a)
    test = I.collect_rankings(s, 200, seed=77, temperature=1.0)
    out = {"SFT utility": I.evaluate(s)["utility"]}
    for n in a.rank_sizes:
        rm = I.train_reward_model(s, I.collect_rankings(s, n), epochs=a.rm_epochs)
        pol, _ = I.ppo(s, rm, beta=0.05, iters=a.ppo_iters)
        out[f"{n} ranked prompts"] = {"RM accuracy": rm_acc(rm, test), "PPO utility": I.evaluate(pol)["utility"]}
        print("  E1", n, out[f"{n} ranked prompts"], flush=True)
    return out


def e2(a):
    base, s = base_and_sft(a)
    train, test = I.collect_rankings(s, a.n_rank), I.collect_rankings(s, 200, seed=77, temperature=1.0)
    out = {}
    for mode in ("per-prompt batch element (paper)", "shuffled independent pairs"):
        torch.manual_seed(0)
        rm = I.LM(d=s.body.tok.embedding_dim, scalar=True); rm.body.load_state_dict(s.body.state_dict())
        opt = torch.optim.AdamW(rm.parameters(), 3e-4)
        pairs = [(p, r[i], r[j]) for p, r in train for i, j in itertools.combinations(range(len(r)), 2)]
        curve = []
        for ep in range(a.rm_epochs):
            if mode.startswith("per-prompt"):
                for i in range(0, len(train), 8):
                    loss = torch.stack([I.ranking_loss(rm, p, r) for p, r in train[i:i + 8]]).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            else:
                random.Random(ep).shuffle(pairs)
                for i in range(0, len(pairs), 64):
                    b = pairs[i:i + 64]
                    rw = I.score(rm, [x[0] for x in b], [x[1] for x in b]); rl = I.score(rm, [x[0] for x in b], [x[2] for x in b])
                    loss = -F.logsigmoid(rw - rl).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            rm.eval(); curve.append(rm_acc(rm, test)); rm.train()
        out[mode] = curve
        print("  E2", mode, curve, flush=True)
    return out


def e3(a):
    base, s = base_and_sft(a)
    rm = I.train_reward_model(s, I.collect_rankings(s, a.n_rank))
    out = {"SFT": I.evaluate(s)}
    for g in a.gammas:
        pol, _ = I.ppo(s, rm, beta=0.05, gamma_ptx=g, iters=a.ppo_iters)
        out[f"gamma={g}"] = I.evaluate(pol)
        print("  E3", g, out[f"gamma={g}"], flush=True)
    return out


def e4(a):
    base, s = base_and_sft(a)
    rm = I.train_reward_model(s, I.collect_rankings(s, a.n_rank))
    out = {}
    for b in a.betas:
        pol, hist = I.ppo(s, rm, beta=b, iters=a.ppo_iters)
        out[f"beta={b}"] = {"KL": hist[-1][2], "RM score": hist[-1][0], "utility": I.evaluate(pol)["utility"]}
        print("  E4", b, out[f"beta={b}"], flush=True)
    return out


def e5(a):
    base = I.pretrain(a.pre_steps)
    out = {}
    for n in a.demo_sizes:
        out[n] = I.evaluate(I.sft(base, a.sft_steps, n))
        print("  E5", n, out[n], flush=True)
    return out


def report(R, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.pre_steps, a.sft_steps, a.n_demos, a.n_rank, a.rm_epochs, a.ppo_iters = 3000, 600, 200, 3000, 5, 200
    a.rank_sizes, a.gammas, a.betas, a.demo_sizes = (300, 1000, 3000, 8000), (0.0, 0.1, 0.3, 1.0, 3.0), (0.0, 0.01, 0.05, 0.2), (50, 200, 800, 2000)
    if a.quick:
        a.pre_steps, a.sft_steps, a.n_demos, a.n_rank, a.rm_epochs, a.ppo_iters = 5, 3, 8, 6, 1, 1
        a.rank_sizes, a.gammas, a.betas, a.demo_sizes = (6,), (0.3,), (0.05,), (8,)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
