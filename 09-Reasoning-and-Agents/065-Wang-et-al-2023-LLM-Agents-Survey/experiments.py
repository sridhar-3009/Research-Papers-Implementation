"""Agent-design experiments following the survey's taxonomy (Wang et al. 2023).

  E1  Planning strategies (single path, self-consistency, tree search with self-evaluation, environment feedback,
      Reflexion) over a grid of step accuracy p in {0.6 ... 0.95} and task length L in {2 ... 40}: success and LLM calls
      (the survey's call for reporting efficiency alongside success).
  E2  Tree search vs judge accuracy q in {0.5 ... 1.0}: self-evaluation is only as good as the evaluator.
  E3  Hybrid memory with UNIQUE distractors and a small capacity: does importance-aware eviction keep the key?
      (vs FIFO eviction).
  E4  A real LLM (Hugging Face `transformers`) on the Game of 24: single-path chain of thought vs self-consistency
      (majority over 5 samples) vs propose-and-evaluate breadth-first search (Tree-of-Thoughts style), with a
      Python checker as the environment; success and number of model calls.

!! HEAVY (E4 needs a GPU and a model download). E1-E3 are quick.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import itertools
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from agentkit import (HybridMemory, evaluate, key_door_episode, reflexion, self_consistency, single_path,
                      tree_of_thoughts, with_environment_feedback)

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STRATEGIES = {"single path": (single_path, {}), "self-consistency": (self_consistency, {"n_paths": 5}),
              "tree search": (tree_of_thoughts, {"breadth": 3}), "env feedback": (with_environment_feedback, {"retries": 2}),
              "reflexion": (reflexion, {"trials": 3})}


def e1(a):
    out = {}
    for p in a.ps:
        for L in a.Ls:
            out[f"p={p}, L={L}"] = {n: dict(zip(("success", "calls"), evaluate(f, p, L, n=a.n, **kw))) for n, (f, kw) in STRATEGIES.items()}
    return out


def e2(a):
    return {q: evaluate(tree_of_thoughts, 0.8, 6, n=a.n, q=q, breadth=3) for q in a.qs}


class FIFOMemory(HybridMemory):
    def write(self, text, importance=1):
        self.t += 1
        self.long.append({"text": text, "t": self.t, "importance": importance})
        if len(self.long) > self.capacity:
            self.long.pop(0)


def e3(a):
    out = {}
    for cap in a.caps:
        rng = random.Random(0)
        uniq = tuple(f"saw thing{i}" for i in range(1000))
        res = {}
        for name, cls in (("importance-aware eviction", HybridMemory), ("FIFO eviction", FIFOMemory)):
            res[name] = float(np.mean([key_door_episode(cls(8, capacity=cap), 100, rng, distractors=uniq) for _ in range(a.n_mem)]))
        out[cap] = res
    return out


def solve24_check(expr, nums):
    try:
        used = sorted(int(x) for x in re.findall(r"\d+", expr))
        return used == sorted(nums) and abs(eval(expr) - 24) < 1e-6
    except Exception:
        return False


def e4(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
    calls = {"n": 0}

    def gen(prompt, n=1, temp=0.0, max_new=60):
        calls["n"] += n
        ids = tok(prompt, return_tensors="pt").to(DEV)
        kw = dict(do_sample=True, temperature=temp, num_return_sequences=n) if temp > 0 else dict(do_sample=False)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **kw)
        return [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True) for x in g]

    rng = random.Random(0)
    puzzles = []
    while len(puzzles) < a.n_24:
        nums = [rng.randint(1, 9) for _ in range(4)]
        if any(abs(eval(f"(({a_}{o1}{b}){o2}{c}){o3}{d}") - 24) < 1e-6                 # solvable left to right
               for a_, b, c, d in itertools.permutations(nums) for o1, o2, o3 in itertools.product("+-*", repeat=3)):
            puzzles.append(nums)
    prompt = lambda nums: f"Use the numbers {nums} each exactly once with + - * / and parentheses to make 24.\nExpression:"
    res = {}
    for name in ("single path", "self-consistency", "propose + evaluate"):
        calls["n"], wins = 0, 0
        for nums in puzzles:
            if name == "single path":
                exprs = gen(prompt(nums))
            elif name == "self-consistency":
                cands = [e.strip().split("\n")[0] for e in gen(prompt(nums), n=5, temp=0.7)]
                exprs = [max(set(cands), key=cands.count)]
            else:                                                                 # propose many, verify with the checker
                exprs = [e.strip().split("\n")[0] for e in gen(prompt(nums), n=10, temp=0.9)]
                exprs = [e for e in exprs if solve24_check(e, nums)] or exprs[:1]
            wins += solve24_check(exprs[0].strip().split("\n")[0], nums)
        res[name] = {"success": wins / len(puzzles), "calls per puzzle": calls["n"] / len(puzzles)}
    return res


def report(R, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 5)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.ps, a.Ls, a.n, a.qs, a.caps, a.n_mem = (0.6, 0.7, 0.8, 0.9, 0.95), (2, 5, 10, 20, 40), 3000, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), (5, 10, 20, 50), 500
    a.model, a.n_24 = "Qwen/Qwen2.5-7B-Instruct", 100
    if a.quick:
        a.ps, a.Ls, a.n, a.qs, a.caps, a.n_mem = (0.8,), (3,), 50, (0.8,), (5,), 20
        a.model, a.n_24 = "sshleifer/tiny-gpt2", 2
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
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
