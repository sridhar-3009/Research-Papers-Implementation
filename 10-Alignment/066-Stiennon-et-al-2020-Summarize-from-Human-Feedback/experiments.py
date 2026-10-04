"""Experiments for 'Learning to summarize from human feedback' (Stiennon et al. 2020): toy studies of the paper's
analyses and a real reward model on the released comparisons.

  E1  Figure 5 (toy): PPO with beta in {0, 0.01, 0.03, 0.1, 0.3, 1}: KL from SFT vs RM score vs TRUE quality.
  E2  Figure 6 (toy): reward-model accuracy vs number of comparisons (250 ... 8000) and model width (32 ... 128).
  E3  Toy: best-of-n against the RM (n = 1 ... 64) vs PPO, at matched KL where possible.
  E4  Section 4.4 (toy): optimise a ROUGE-like reward (unigram overlap with the reference) instead of the RM; judge
      both by the labelers' true quality.
  E5  Real RM: the released `openai/summarize_from_feedback` comparisons (Hugging Face `datasets`): a small LM plus
      scalar head trained with the Bradley-Terry loss; held-out accuracy.

!! HEAVY. Not run on the author's laptop (E1-E4 take minutes on CPU; E5 needs a GPU).
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import rlhf as R

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def setup(a, n_cmp=None):
    sft = R.train_sft(steps=a.sft_steps)
    data = R.collect_comparisons(sft, (n_cmp or a.n_cmp) + 400)
    rm = R.train_reward_model(sft, data[:-400], steps=a.rm_steps)
    return sft, rm, data[-400:]


def e1(a):
    sft, rm, _ = setup(a)
    out = {}
    for beta in a.betas:
        pol, hist = R.ppo(sft, rm, beta, iters=a.ppo_iters)
        ev = R.evaluate_policy(pol, rm)
        out[beta] = {"KL": hist[-1][2], "RM score": ev["RM score"], "true quality": ev["true quality"],
                     "preferred to reference": ev["preferred to reference"]}
        print("  E1", beta, out[beta], flush=True)
    return out


def e2(a):
    out = {"comparisons": {}, "width": {}}
    sft = R.train_sft(steps=a.sft_steps)
    data = R.collect_comparisons(sft, max(a.sizes) + 1000)
    test = data[-1000:]
    for n in a.sizes:
        out["comparisons"][n] = R.rm_accuracy(R.train_reward_model(sft, data[:n], steps=a.rm_steps), test)
    for d in a.widths:                                                            # a fresh SFT + RM of each width
        sft_d = R.train_sft(steps=a.sft_steps, d=d)
        out["width"][d] = R.rm_accuracy(R.train_reward_model(sft_d, data[:max(a.sizes)], steps=a.rm_steps), test)
    return out


def e3(a):
    sft, rm, _ = setup(a)
    out = {}
    rng, gen = random.Random(5), torch.Generator().manual_seed(5)
    posts = [R.make_post(rng) for _ in range(300)]
    for n in a.ns:
        best_q = []
        for p, top in posts:
            cands = R.sample(sft, [p] * n, 1.0, gen)
            with torch.no_grad():
                scores = R.reward(rm, [p] * n, cands)
            best_q.append(R.true_quality(p, top, cands[int(scores.argmax())]))
        out[f"best-of-{n}"] = float(np.mean(best_q))
    pol, _ = R.ppo(sft, rm, 0.2, iters=a.ppo_iters)
    out["PPO beta=0.2"] = R.evaluate_policy(pol, rm)["true quality"]
    return out


def e4(a):
    """Optimise a ROUGE-1-like reward (unigram overlap with ONE noisy reference per post) vs the learned RM."""
    sft, rm, _ = setup(a)

    def rouge(posts, tops, summaries):
        out = []
        for p, top, s in zip(posts, tops, summaries):
            ref = R.reference_summary(top, random.Random(hash(tuple(p)) % 10**9))   # a fixed reference per post
            out.append(len(set(s) & set(ref)) / len(set(ref)) - 0.1 * max(0, len(s) - len(ref)))
        return torch.tensor(out)
    res = {}
    for name, fn in (("optimise ROUGE-like overlap", rouge), ("optimise reward model", None)):
        pol, _ = R.ppo(sft, rm, 0.1, iters=a.ppo_iters, reward_fn=fn)
        ev = R.evaluate_policy(pol, rm)
        res[name] = {"true quality": ev["true quality"], "preferred to reference": ev["preferred to reference"]}
    return res


def e5(a):
    import datasets
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    ds = datasets.load_dataset("openai/summarize_from_feedback", "comparisons")
    train, val = ds["train"].shuffle(seed=0).select(range(a.n_real)), ds["validation"].select(range(a.n_real_val))
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModel.from_pretrained(a.model).to(DEV); head = nn.Linear(base.config.hidden_size, 1).to(DEV)

    def score(posts, sums):
        enc = tok([f"POST: {p}\nTL;DR: {s}" for p, s in zip(posts, sums)], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(DEV)
        h = base(**enc).last_hidden_state
        return head(h[torch.arange(len(posts)), enc["attention_mask"].sum(1) - 1]).squeeze(-1)
    opt = torch.optim.AdamW(list(base.parameters()) + list(head.parameters()), 1e-5)
    for s in range(0, len(train), a.bs):
        b = train.select(range(s, min(s + a.bs, len(train))))
        posts = [r["info"]["post"] for r in b]
        c = [r["summaries"][r["choice"]]["text"] for r in b]; rj = [r["summaries"][1 - r["choice"]]["text"] for r in b]
        loss = -F.logsigmoid(score(posts, c) - score(posts, rj)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    base.eval(); hits = []
    with torch.no_grad():
        for r in val:
            p = r["info"]["post"]
            hits.append(float(score([p], [r["summaries"][r["choice"]]["text"]]) > score([p], [r["summaries"][1 - r["choice"]]["text"]])))
    return {"held-out accuracy": float(np.mean(hits)), "paper": "6.7B RM approaches single-labeler agreement"}


def report(R_, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R_):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R_[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.sft_steps, a.rm_steps, a.n_cmp, a.ppo_iters = 1500, 1200, 8000, 400
    a.betas, a.sizes, a.widths, a.ns = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0), (250, 500, 1000, 2000, 4000, 8000), (32, 64, 128), (1, 2, 4, 8, 16, 64)
    a.model, a.n_real, a.n_real_val, a.bs = "Qwen/Qwen2.5-0.5B", 20000, 2000, 8
    if a.quick:
        a.sft_steps, a.rm_steps, a.n_cmp, a.ppo_iters = 3, 2, 32, 2
        a.betas, a.sizes, a.widths, a.ns = (0.1,), (16,), (32,), (1, 2)
        a.model, a.n_real, a.n_real_val, a.bs = "sshleifer/tiny-gpt2", 4, 4, 2
    path = HERE / "results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    res[name] = fn(a)
                except ImportError as e:
                    res[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(res, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(res, a)


if __name__ == "__main__":
    main()
