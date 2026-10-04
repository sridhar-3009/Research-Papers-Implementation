"""Zero-shot-CoT experiments (Kojima et al. 2022) with open models, plus toy studies.

  E1  Table 1 / Figure 3: zero-shot vs zero-shot-CoT (two-stage, greedy) on GSM8K, MultiArith (Hugging Face
      `datasets`), and locally generated Last Letter Concatenation (4 names) and Coin Flip (4 flips), for a ladder of
      open models (Hugging Face `transformers`).
  E2  Table 4: all 16 trigger sentences (instructive / misleading / irrelevant) on MultiArith with the largest model.
  E3  Table 2: self-consistency: sample N reasoning paths (T = 0.7), majority vote, N = 1, 5, 10, 20.
  E4  Toy: how much 'reasoning text' must the pre-training corpus contain for the trigger to work? Vary the share
      of 'T' documents (5% ... 60%) at a fixed budget.
  E5  Toy emergence: the same corpus, models of several sizes: zero-shot vs zero-shot-CoT by size.

!! HEAVY. Not run on the author's laptop (E1-E3 need a GPU and model downloads).
       python3 experiments.py --quick
       python3 experiments.py --only e4
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from zeroshot import (TABLE_4, cleanse, answer_prompt, reasoning_prompt, self_consistency, toy_answer, toy_problem,
                      train_toy, zero_shot_prompt)

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = ["Elon Musk", "Larry Page", "Sergey Brin", "Bill Gates", "Ada Lovelace", "Alan Turing", "Grace Hopper",
         "Linus Torvalds", "Marie Curie", "Isaac Newton", "Emmy Noether", "Barbara Liskov", "Donald Knuth", "Tim Cook"]
PEOPLE = ["Ka", "Sherrie", "Jamey", "Vernell", "Delfina", "Paulette", "Mira", "Tom", "Ravi", "Lena"]


# ---------------------------------------------------------------------------------------------------- data
def last_letter(n, rng):
    out = []
    for _ in range(n):
        words = rng.sample([w for n in NAMES for w in n.split()], 4)               # 4 words, as in the paper
        q = f'Take the last letters of each words in "{" ".join(words)}" and concatenate them.'
        out.append((q, "".join(w[-1] for w in words), "letters"))
    return out


def coin_flip(n, rng):
    out = []
    for _ in range(n):
        heads, parts = True, ["A coin is heads up."]
        for p in rng.sample(PEOPLE, 4):
            flip = rng.random() < 0.5
            heads ^= flip
            parts.append(f"{p} {'flips' if flip else 'does not flip'} the coin.")
        out.append((" ".join(parts) + " Is the coin still heads up? Note that \"flip\" here means \"reverse\".",
                    "yes" if heads else "no", "yes/no"))
    return out


def hf_math(name, n):
    import datasets
    if name == "gsm8k":
        ds = datasets.load_dataset("gsm8k", "main", split="test")
        return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", ""), "number") for r in ds][:n]
    ds = datasets.load_dataset("ChilleD/MultiArith", split="test")
    return [(r["question"], str(r["final_ans"]).rstrip("0").rstrip(".") if "." in str(r["final_ans"]) else str(r["final_ans"]),
             "number") for r in ds][:n]


# ---------------------------------------------------------------------------------------------------- models
class HF:
    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()

    def __call__(self, prompt, max_new=128, temperature=0.0):
        ids = self.tok(prompt, return_tensors="pt").to(DEV)
        kw = dict(do_sample=temperature > 0, temperature=temperature) if temperature > 0 else dict(do_sample=False)
        with torch.no_grad():
            g = self.model.generate(**ids, max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id, **kw)
        return self.tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\nQ:")[0]


def score(gen, rows, trigger, a, temperature=0.0):
    ok = 0
    for q, truth, fmt in rows:
        if trigger is None:
            pred = cleanse(gen(zero_shot_prompt(q, fmt), max_new=16), fmt)
        else:
            x1 = reasoning_prompt(q, trigger)
            z = gen(x1, max_new=a.max_new, temperature=temperature)
            pred = cleanse(gen(answer_prompt(x1, z, fmt), max_new=16), fmt)
        ok += (pred or "").lower() == truth.lower()
    return ok / len(rows)


def datasets_for(a):
    rng = random.Random(0)
    sets = {"last letter": last_letter(a.n, rng), "coin flip": coin_flip(a.n, rng)}
    for name in ("gsm8k", "multiarith"):
        try:
            sets[name] = hf_math(name, a.n)
        except Exception as e:
            print("  (skip", name, type(e).__name__, ")")
    return sets


def e1(a):
    sets = datasets_for(a)
    out = {}
    for mname in a.models:
        gen = HF(mname)
        out[mname] = {s: {"zero-shot": score(gen, rows, None, a), "zero-shot-CoT": score(gen, rows, "Let's think step by step.", a)}
                      for s, rows in sets.items()}
        print("  E1", mname, out[mname], flush=True)
    return out


def e2(a):
    rows = datasets_for(a).get("multiarith") or datasets_for(a)["coin flip"]
    gen = HF(a.models[-1])
    return {f"{c}: {t or '(zero-shot)'}": {"ours": score(gen, rows, t or None, a), "paper": acc} for c, t, acc in TABLE_4}


def e3(a):
    rows = datasets_for(a).get("gsm8k") or datasets_for(a)["coin flip"]
    gen = HF(a.models[-1])
    out = {}
    for n_paths in a.paths:
        ok = 0
        for q, truth, fmt in rows:
            x1 = reasoning_prompt(q)
            answers = []
            for _ in range(n_paths):
                z = gen(x1, max_new=a.max_new, temperature=0.7)
                answers.append(cleanse(gen(answer_prompt(x1, z, fmt), max_new=16), fmt))
            ok += self_consistency(answers) == truth
        out[n_paths] = ok / len(rows)
        print("  E3", n_paths, out[n_paths], flush=True)
    return out


def toy_eval(m, trigger, k, n):
    rng, ok = random.Random(3), 0
    for _ in range(n):
        q, run = toy_problem(rng, k)
        ok += toy_answer(m, q, trigger)[0] == str(run[-1])
    return ok / n


def e4(a):
    out = {}
    for share in a.shares:
        direct = (1 - share) * 0.9
        mix = (direct, share, (1 - share - direct) / 2, (1 - share - direct) / 2)
        m, _ = train_toy(a.toy_steps, mix=mix)
        out[f"{share:.0%} reasoning text"] = {"zero-shot k=5": toy_eval(m, None, 5, a.n_toy),
                                              "zero-shot-CoT k=5": toy_eval(m, "T", 5, a.n_toy)}
        print("  E4", share, out[f"{share:.0%} reasoning text"], flush=True)
    return out


def e5(a):
    out = {}
    for d, layers in a.sizes:
        m, _ = train_toy(a.toy_steps, d=d, layers=layers)
        out[f"d={d}, layers={layers}"] = {"zero-shot k=5": toy_eval(m, None, 5, a.n_toy),
                                          "zero-shot-CoT k=5": toy_eval(m, "T", 5, a.n_toy)}
        print("  E5", d, layers, out[f"d={d}, layers={layers}"], flush=True)
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
    a.models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
    a.n, a.max_new, a.paths = 300, 256, (1, 5, 10, 20)
    a.shares, a.sizes, a.toy_steps, a.n_toy = (0.05, 0.15, 0.3, 0.6), [(16, 1), (32, 2), (64, 2), (128, 3)], 3000, 300
    if a.quick:
        a.models, a.n, a.max_new, a.paths = ["sshleifer/tiny-gpt2"], 2, 8, (1, 2)
        a.shares, a.sizes, a.toy_steps, a.n_toy = (0.6,), [(16, 1)], 3, 3
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
