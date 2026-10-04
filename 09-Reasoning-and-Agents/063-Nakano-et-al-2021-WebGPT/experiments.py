"""WebGPT experiments (Nakano et al. 2021): a real reward model on the released comparisons, plus toy studies.

  E1  Real RM: OpenAI's released `openai/webgpt_comparisons` (Hugging Face `datasets`): question + answer (+ quotes)
      pairs with human preference scores. Fine-tune a small LM with a scalar head on the Bradley-Terry loss (ties as
      0.5), report held-out pairwise accuracy (for scale: human labelers agree with each other 73% of the time).
  E2  Real best-of-n: an open instruction model answers held-out ELI5-style questions from the comparison set; pick the
      RM's favourite among n = 1, 4, 16 samples (the RM trained in E1); report the mean RM score of the chosen answer.
  E3  Toy (Figure 7): RM accuracy vs number of comparisons (250 ... 16,000): about +x% per doubling?
  E4  Toy (Figures 5, 8): best-of-n true quality vs n for RMs trained on 500 / 3,000 / 16,000 comparisons: where is the
      peak, and does a better RM push it to larger n?
  E5  Toy (Section 5.1): RL with KL coefficient beta = 0, 0.03, 0.1, 0.3, 1.0: RM score vs true quality.

!! HEAVY. Not run on the author's laptop (E1-E2 need a GPU and downloads).
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from webgpt import (answer_features, best_of_n, labeler_compare, make_web, policy_episode, rl_finetune,
                    train_reward_model, true_quality)

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BC = {"precise": 0.7, "top": 0.6, "find": 0.6, "fill": 1.0}


# ---------------------------------------------------------------------------------------------------- real RM
def rm_model(name):
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModel.from_pretrained(name).to(DEV)
    head = nn.Linear(base.config.hidden_size, 1).to(DEV)

    def score(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
        h = base(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        return head(h[torch.arange(len(texts)), last]).squeeze(-1)
    return base, head, score


def as_text(question, answer, quotes):
    refs = " ".join(f"[{i + 1}] {t}: {e}" for i, (t, e) in enumerate(zip(quotes.get("title", []), quotes.get("extract", []))))
    return f"Question: {question}\nQuotes: {refs}\nAnswer: {answer}"


def e1(a):
    import datasets
    import torch.nn.functional as F
    ds = datasets.load_dataset("openai/webgpt_comparisons", split="train").shuffle(seed=0)
    n_test = min(a.n_test, len(ds) // 10)
    test, train = ds.select(range(n_test)), ds.select(range(n_test, min(len(ds), n_test + a.n_train)))
    base, head, score = rm_model(a.model)
    opt = torch.optim.AdamW(list(base.parameters()) + list(head.parameters()), a.lr)
    for epoch in range(a.epochs):
        for s in range(0, len(train), a.bs):
            b = train.select(range(s, min(s + a.bs, len(train))))
            ra = score([as_text(r["question"]["full_text"], r["answer_0"], r["quotes_0"]) for r in b])
            rb = score([as_text(r["question"]["full_text"], r["answer_1"], r["quotes_1"]) for r in b])
            y = torch.tensor([0.5 if r["score_0"] == r["score_1"] else float(r["score_0"] > r["score_1"]) for r in b], device=DEV)
            loss = F.binary_cross_entropy_with_logits(ra - rb, y)
            opt.zero_grad(); loss.backward(); opt.step()
    base.eval()
    hits = []
    with torch.no_grad():
        for r in test:
            if r["score_0"] == r["score_1"]:
                continue
            ra = score([as_text(r["question"]["full_text"], r["answer_0"], r["quotes_0"])])
            rb = score([as_text(r["question"]["full_text"], r["answer_1"], r["quotes_1"])])
            hits.append(float((ra > rb).item()) == float(r["score_0"] > r["score_1"]))
    torch.save({"base": base.state_dict(), "head": head.state_dict()}, HERE / "rm.pt")
    return {"held-out pairwise accuracy": float(np.mean(hits)), "n test pairs": len(hits)}


def e2(a):
    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ds = datasets.load_dataset("openai/webgpt_comparisons", split="train").shuffle(seed=1).select(range(a.n_bon))
    tok = AutoTokenizer.from_pretrained(a.policy)
    pol = AutoModelForCausalLM.from_pretrained(a.policy).to(DEV).eval()
    base, head, score = rm_model(a.model)
    if (HERE / "rm.pt").exists():
        sd = torch.load(HERE / "rm.pt", map_location=DEV)
        base.load_state_dict(sd["base"]); head.load_state_dict(sd["head"])
    base.eval()
    out = {}
    for n in a.ns:
        best_scores = []
        for r in ds:
            q = r["question"]["full_text"]
            ids = tok(f"Question: {q}\nAnswer:", return_tensors="pt").to(DEV)
            with torch.no_grad():
                g = pol.generate(**ids, max_new_tokens=a.max_new, do_sample=True, temperature=1.0, num_return_sequences=n,
                                 pad_token_id=tok.eos_token_id)
                answers = [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True) for x in g]
                s = score([as_text(q, ans, {}) for ans in answers]).tolist()
            best_scores.append(max(s))
        out[n] = float(np.mean(best_scores))
    return {"mean RM score of the chosen answer": out, "RM trained": (HERE / "rm.pt").exists()}


# ---------------------------------------------------------------------------------------------------- toys
def toy_data(n, seed=0):
    web, truth = make_web()
    rng = np.random.default_rng(seed)
    FA, FB, Y = [], [], []
    for _ in range(n):
        topic = f"topic{rng.integers(40)}"
        a = policy_episode(web, topic, BC, rng)[:2]; b = policy_episode(web, topic, BC, rng)[:2]
        FA.append(answer_features(*a, topic)); FB.append(answer_features(*b, topic))
        Y.append(labeler_compare(true_quality(*a, topic, truth), true_quality(*b, topic, truth), rng))
    return web, truth, np.array(FA), np.array(FB), np.array(Y)


def e3(a):
    web, truth, FA, FB, Y = toy_data(max(a.sizes) + 2000)
    test = slice(max(a.sizes), None)
    out = {}
    for n in a.sizes:
        w = train_reward_model(FA[:n], FB[:n], Y[:n])
        mask = Y[test] != 0.5
        out[n] = float(np.mean(((FA[test] @ w > FB[test] @ w) == (Y[test] > 0.5))[mask]))
    return out


def e4(a):
    out = {}
    for n_cmp in a.rm_sizes:
        web, truth, FA, FB, Y = toy_data(n_cmp, seed=1)
        w = train_reward_model(FA, FB, Y)
        rng = np.random.default_rng(2)
        row = {}
        for n in a.ns_toy:
            qs = []
            for i in range(a.n_q):
                topic = f"topic{i % 40}"
                cands = [policy_episode(web, topic, BC, rng)[:2] for _ in range(n)]
                best = best_of_n(cands, lambda c: float(answer_features(*c, topic) @ w))
                qs.append(true_quality(*best, topic, truth))
            row[n] = float(np.mean(qs))
        out[f"RM on {n_cmp} comparisons"] = row
        print("  E4", n_cmp, row, flush=True)
    return out


def e5(a):
    web, truth, FA, FB, Y = toy_data(3000, seed=3)
    w = train_reward_model(FA, FB, Y)
    rm = lambda t, q, topic: float(answer_features(t, q, topic) @ w)
    out = {}
    for beta in a.betas:
        hist, theta = rl_finetune(web, truth, BC, rm, beta, steps=a.rl_steps, lr=0.3)
        out[beta] = {"RM score": hist[-1][0], "true quality": hist[-1][1], "theta": theta}
        print("  E5", beta, out[beta], flush=True)
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
    a.model, a.policy, a.n_train, a.n_test, a.lr, a.bs, a.epochs = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 17000, 1500, 1e-5, 8, 1
    a.n_bon, a.ns, a.max_new = 200, (1, 4, 16), 200
    a.sizes, a.rm_sizes, a.ns_toy, a.n_q, a.betas, a.rl_steps = (250, 500, 1000, 2000, 4000, 8000, 16000), (500, 3000, 16000), (1, 4, 16, 64, 256, 1024), 400, (0.0, 0.03, 0.1, 0.3, 1.0), 400
    if a.quick:
        a.model, a.policy, a.n_train, a.n_test, a.bs = "sshleifer/tiny-gpt2", "sshleifer/tiny-gpt2", 4, 20, 2
        a.n_bon, a.ns, a.max_new = 1, (1, 2), 8
        a.sizes, a.rm_sizes, a.ns_toy, a.n_q, a.betas, a.rl_steps = (50,), (100,), (1, 4), 10, (0.0,), 3
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
