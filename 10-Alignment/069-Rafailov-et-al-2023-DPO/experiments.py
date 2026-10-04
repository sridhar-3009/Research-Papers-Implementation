"""DPO experiments (Rafailov et al. 2023): the reward-KL frontier sweep of Figure 2 on the toy, data-size and
label-noise studies, the unweighted ablation, and a real DPO fine-tune of a small open LM.

  E1  Frontier (Figure 2 left): DPO beta in {0.05, 0.1, 0.3, 1, 3}, RLHF beta in the same grid, Unlikelihood
      alpha in {0.05, 0.1, 0.5, 1}, Preferred-FT seeds, Best-of-N; several seeds; distance to the exact frontier.
  E2  How much preference data does DPO need? 250 ... 16000 pairs, DPO vs RLHF (RM trained on the same pairs).
  E3  Label noise: a fraction (0 ... 40%) of preference labels flipped at random.
  E4  Ablation: DPO vs its gradient WITHOUT the sigma(r_l - r_w) weight (the naive version that degenerates, Table 3),
      and DPO without the reference model.
  E5  Real DPO: fine-tune a small open LM (Hugging Face `transformers`, e.g. Qwen2.5-0.5B-Instruct) with full
      (no LoRA) DPO on a slice of Anthropic HH, beta = 0.1, RMSprop lr 1e-6 with 150 warm-up steps, batch 64 (gradient
      accumulation); report the implicit-reward accuracy and mean margin on held-out pairs.

!! HEAVY for E5 (GPU + model and dataset download); E1-E4 take minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import dpo as D

HERE = Path(__file__).parent


def efficiency(e):
    return e["reward"] / D.optimal_reward_at_kl(max(e["KL"], 1e-3))


def e1(a):
    out = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        ref = D.train_sft(rng, steps=a.sft_steps, seed=seed)
        data = D.make_preferences(rng, ref, a.pairs)
        rm = D.train_reward_model(data)
        for beta in a.betas:
            for name, pi in (("DPO", D.train_offline(ref, data, "dpo", beta=beta, steps=a.steps, seed=seed)),
                             ("RLHF", D.train_rlhf(ref, rm, beta=beta, steps=a.steps, seed=seed, rng=rng))):
                e = D.evaluate(pi, ref, rng)
                out.append({"method": name, "beta": beta, "seed": seed, **e, "efficiency": efficiency(e)})
        for alpha in (0.05, 0.1, 0.5, 1.0):
            e = D.evaluate(D.train_offline(ref, data, "unlikelihood", alpha=alpha, steps=a.steps, seed=seed), ref, rng)
            out.append({"method": "Unlikelihood", "alpha": alpha, "seed": seed, **e, "efficiency": efficiency(e)})
        e = D.evaluate(D.train_offline(ref, data, "preferred_ft", steps=a.steps, seed=seed), ref, rng)
        out.append({"method": "Preferred-FT", "seed": seed, **e, "efficiency": efficiency(e)})
        for n in (4, 16, 64):
            e = D.best_of_n(ref, rm, rng, n)
            out.append({"method": f"Best-of-{n}", "seed": seed, **e, "efficiency": efficiency(e)})
    return {"runs": out, "frontier": {b: D.optimal_frontier(b) for b in a.betas}}


def e2(a):
    rng = np.random.default_rng(0)
    ref = D.train_sft(rng, steps=a.sft_steps)
    out = {}
    for n in a.sizes:
        data = D.make_preferences(rng, ref, n)
        e_d = D.evaluate(D.train_offline(ref, data, "dpo", beta=1.0, steps=a.steps), ref, rng)
        e_r = D.evaluate(D.train_rlhf(ref, D.train_reward_model(data), beta=1.0, steps=a.steps, rng=rng), ref, rng)
        out[n] = {"DPO": {**e_d, "efficiency": efficiency(e_d)}, "RLHF": {**e_r, "efficiency": efficiency(e_r)}}
    return out


def e3(a):
    rng = np.random.default_rng(1)
    ref = D.train_sft(rng, steps=a.sft_steps)
    x, yw, yl = D.make_preferences(rng, ref, a.pairs)
    out = {}
    for flip in (0.0, 0.1, 0.2, 0.3, 0.4):
        f = torch.tensor(rng.random(len(x)) < flip)[:, None]
        data = (x, torch.where(f, yl, yw), torch.where(f, yw, yl))
        e = D.evaluate(D.train_offline(ref, data, "dpo", beta=1.0, steps=a.steps), ref, rng)
        out[f"flip {flip}"] = {**e, "efficiency": efficiency(e)}
    return out


def e4(a):
    rng = np.random.default_rng(2)
    ref = D.train_sft(rng, steps=a.sft_steps)
    data = D.make_preferences(rng, ref, a.pairs)
    x, yw, yl = data
    out = {}
    for variant in ("dpo", "unweighted", "no_ref"):
        torch.manual_seed(0)
        pi = D.copy_of(ref).train()
        opt = torch.optim.Adam(pi.parameters(), lr=1e-3)
        with torch.no_grad():
            rw, rl = D.seq_logp(ref, x, yw), D.seq_logp(ref, x, yl)
        for _ in range(a.steps):
            i = torch.randint(0, len(x), (64,))
            lw, ll = D.seq_logp(pi, x[i], yw[i]), D.seq_logp(pi, x[i], yl[i])
            if variant == "dpo":
                loss = D.dpo_loss(lw, ll, rw[i], rl[i], 1.0)[0]
            elif variant == "unweighted":                                       # same direction, constant weight
                loss = -(lw - ll).mean()
            else:                                                               # pi_ref = uniform
                loss = D.dpo_loss(lw, ll, torch.zeros(64), torch.zeros(64), 1.0)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        e = D.evaluate(pi.eval(), ref, rng)
        out[variant] = {**e, "efficiency": efficiency(e)}
    return out


def e5(a):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    pi = AutoModelForCausalLM.from_pretrained(a.model).to(dev)
    ref = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    ds = load_dataset("Anthropic/hh-rlhf", split=f"train[:{a.hh_pairs}]")
    test = load_dataset("Anthropic/hh-rlhf", split=f"test[:{a.hh_test}]")

    def split(ex):                                          # shared prompt, then the two different replies
        c, r = ex["chosen"], ex["rejected"]
        k = c.rfind("\n\nAssistant:") + len("\n\nAssistant:")
        return c[:k], c[k:], r[r.rfind("\n\nAssistant:") + len("\n\nAssistant:"):]

    def logp(model, prompt, reply):
        ids = tok(prompt + reply, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
        n = len(tok(prompt).input_ids)
        out = model(ids).logits[0, n - 1:-1].log_softmax(-1)
        return out.gather(1, ids[0, n:, None]).sum()

    opt = torch.optim.RMSprop(pi.parameters(), lr=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 150))
    accum, step = 64, 0
    for i, ex in enumerate(ds):
        p, w, l = split(ex)
        with torch.no_grad():
            rw, rl = logp(ref, p, w), logp(ref, p, l)
        loss = D.dpo_loss(logp(pi, p, w)[None], logp(pi, p, l)[None], rw[None], rl[None], a.beta)[0] / accum
        loss.backward()
        if (i + 1) % accum == 0:
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
    pi.eval()
    acc, margins = [], []
    with torch.no_grad():
        for ex in test:
            p, w, l = split(ex)
            m = a.beta * ((logp(pi, p, w) - logp(ref, p, w)) - (logp(pi, p, l) - logp(ref, p, l)))
            acc.append(float(m > 0))
            margins.append(float(m))
    return {"steps": step, "implicit-reward accuracy (held-out)": float(np.mean(acc)), "mean margin": float(np.mean(margins))}


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
    a.seeds, a.sft_steps, a.pairs, a.steps = 3, 600, 2000, 400
    a.betas, a.sizes = (0.05, 0.1, 0.3, 1.0, 3.0), (250, 1000, 4000, 16000)
    a.model, a.hh_pairs, a.hh_test, a.beta = "Qwen/Qwen2.5-0.5B-Instruct", 6400, 500, 0.1
    if a.quick:
        a.seeds, a.sft_steps, a.pairs, a.steps = 1, 50, 100, 10
        a.betas, a.sizes = (1.0,), (100,)
        a.model, a.hh_pairs, a.hh_test = "sshleifer/tiny-gpt2", 2, 2
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    R[name] = fn(a)
                except ImportError as e:
                    R[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
