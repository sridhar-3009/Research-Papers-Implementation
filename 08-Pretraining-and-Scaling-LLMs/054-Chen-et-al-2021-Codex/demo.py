"""Codex's evaluation ideas in a few seconds: pass@k (biased vs unbiased), execution vs BLEU, temperature vs k,
picking one sample by mean log-prob, whitespace tokens, and the paper's numbers. No language model is trained here:
section 3 uses REAL programs (correct and buggy variants of synthetic problems, actually executed) with a SIMULATED
model probability over them."""

import random
import time
from math import comb
from pathlib import Path

import numpy as np

from codex import (MINI_HUMANEVAL, TABLE_1, bleu, check_correctness, code_loss, pass_at_k, pass_at_k_naive,
                   plain_tokens, rank, softmax, nucleus, synthetic_problem, whitespace_tokens, BUILDING_BLOCKS)

T0 = time.time()
rng = np.random.default_rng(0); random.seed(0)


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------------------------------- 1
section("1. pass@k: the unbiased estimator vs 1 - (1 - c/n)^k (Section 2.1, Appendix A)")
print("  expected value over c ~ Binomial(n, p), computed exactly; truth = 1 - (1 - p)^k")
print("    p      n    k   truth   unbiased  naive")
for p, n, k in [(0.05, 200, 100), (0.1, 20, 10), (0.1, 100, 10), (0.3, 20, 5)]:
    w = [comb(n, c) * p ** c * (1 - p) ** (n - c) for c in range(n + 1)]
    ub = sum(wi * pass_at_k(n, c, k) for c, wi in enumerate(w))
    nv = sum(wi * pass_at_k_naive(n, c, k) for c, wi in enumerate(w))
    print(f"  {p:4.2f} {n:5d} {k:4d}   {1 - (1 - p) ** k:.4f}  {ub:.4f}   {nv:.4f}")
print("  -> the unbiased form matches the truth exactly; the naive form is always low, and worst when k is close to n")

# ---------------------------------------------------------------------------------------------------- 2
section("2. Functional correctness vs BLEU (Figure 8): execute, don't compare text")
p = MINI_HUMANEVAL[3]                                                     # the paper's Appendix B x_or_y example
cands = {"reference": p["canonical"],
         "correct, rewritten": "    return x if n > 1 and all(n % d for d in range(2, n)) else y\n",
         "wrong (paper's completion 6)": "    for i in range(2,n-1):\n        if (n % i == 0):\n            return y\n    return x\n",
         "wrong (off-by-one sqrt)": p["canonical"].replace("int(n ** 0.5) + 1", "int(n ** 0.5)"),
         "wrong (paper's completion 8)": "    if n == x:\n        return x\n    elif n == y:\n        return y\n    else:\n        return n\n"}
for name, c in cands.items():
    r = check_correctness(p["prompt"], c, p["test"], p["entry_point"])
    print(f"  {name:30s} BLEU vs reference {bleu(c, p['canonical']):.2f}   tests: {r}")
print("  -> the off-by-one program has near-perfect BLEU yet fails (it calls 4 a prime); the rewritten correct one scores low")

# ---------------------------------------------------------------------------------------------------- 3
section("3. Temperature vs k (Figure 5), on 40 executed synthetic problems with a simulated model")


def variants(pr, r):
    """Real candidate programs: the reference plus plausible mistakes. Execution decides which are right."""
    lines = pr["canonical"].splitlines(keepends=True)
    body, ret = lines[:-1], lines[-1]
    out = [pr["canonical"]]
    for _ in range(5):
        b = list(body)
        kind = r.choice(["drop", "swap", "replace", "dup"])
        i = r.randrange(len(b))
        if kind == "drop" and len(b) > 1: b.pop(i)
        elif kind == "swap" and len(b) > 1: j = r.randrange(len(b)); b[i], b[j] = b[j], b[i]
        elif kind == "replace": b[i] = f"    {r.choice(BUILDING_BLOCKS)[1]}\n"
        else: b.insert(i, b[i])
        out.append("".join(b) + ret)
    out.append("    return s\n")
    return list(dict.fromkeys(out))


r = random.Random(1)
problems = []
for i in range(40):
    pr = synthetic_problem(r.choice([2, 3, 4]), r)
    cs = variants(pr, r)
    ok = np.array([check_correctness(pr["prompt"], c, pr["test"], pr["entry_point"]) == "passed" for c in cs])
    logits = rng.normal(0, 1.5, len(cs)); logits[0] += rng.normal(0.5, 1.5)   # the model favours the reference a bit
    problems.append((cs, ok, logits))
print(f"  {sum(len(c) for c, _, _ in problems)} candidate programs executed in a subprocess each; "
      f"{sum(o.sum() for _, o, _ in problems)} pass (some 'mistakes', e.g. swapping commuting steps, are still correct)")
print(f"  greedy (T = 0): pass@1 = {np.mean([o[np.argmax(l)] for _, o, l in problems]):.3f}")
print("    T     pass@1  pass@10  pass@100   (n = 200 samples per problem, nucleus p = 0.95)")
best = {}
for T in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5):
    res = []
    for cs, ok, lg in problems:
        q = nucleus(softmax(lg, T), 0.95)
        c = int(ok[rng.choice(len(cs), size=200, p=q)].sum())
        res.append([pass_at_k(200, c, k) for k in (1, 10, 100)])
    m = np.mean(res, 0)
    for k, v in zip((1, 10, 100), m):
        if v > best.get(k, (0, 0))[1]: best[k] = (T, v)
    print(f"  {T:4.1f}   {m[0]:.3f}   {m[1]:.3f}    {m[2]:.3f}")
print(f"  best T: pass@1 at T = {best[1][0]}, pass@10 at T = {best[10][0]}, pass@100 at T = {best[100][0]}")
print("  -> low temperature for one try, higher temperature (diversity) when many tries are allowed, as in the paper.")
print("     Caveat: each toy problem has only ~6 candidates, so even T = 2.5 never produces garbage; a real model's")
print("     samples degrade at high T, which is why the paper's best T for pass@100 is 0.8, not 'as high as possible'")

# ---------------------------------------------------------------------------------------------------- 4
section("4. Choosing ONE sample without tests: mean vs sum log-prob (Figure 7), simulated token log-probs")
trials = {"random": [], "mean log-prob": [], "sum log-prob": [], "oracle (unit tests)": []}
for _ in range(2000):
    samples = []
    for _ in range(10):
        correct = rng.random() < 0.3
        length = int(rng.integers(5, 15)) if correct else int(rng.integers(2, 25))  # wrong ones: often truncated/short
        lp = rng.normal(-0.45 if correct else -0.55, 0.5, length)
        samples.append((lp, correct))
    trials["random"].append(rank(samples, "random")[1])
    trials["mean log-prob"].append(rank(samples, "mean")[1])
    trials["sum log-prob"].append(rank(samples, "sum")[1])
    trials["oracle (unit tests)"].append(any(s[1] for s in samples))
for k, v in trials.items():
    print(f"  {k:20s} {np.mean(v):.3f}")
print("  -> sum log-prob is pulled toward SHORT samples (fewer negative terms), so it gains much less than the mean;")
print("     in the paper it is even slightly worse than random. (Simulated: correct samples get a slightly higher")
print("     per-token log-prob by assumption, -0.45 vs -0.55; all numbers here follow from that assumption.)")

# ---------------------------------------------------------------------------------------------------- 5
section("5. Whitespace tokens (Section 3.2): runs of spaces become one token")
files = list((Path(__file__).resolve().parents[2]).rglob("*.py"))[:200]
plain = ws = 0
for f in files:
    code = f.read_text(errors="ignore")
    plain += len(plain_tokens(code)); ws += len(whitespace_tokens(code))
print(f"  {len(files)} Python files from this repository: {plain:,} tokens -> {ws:,} ({1 - ws / plain:.0%} fewer)")
print("  (the paper: ~30% fewer BPE tokens; our split is word-level, so the base count and saving differ)")

# ---------------------------------------------------------------------------------------------------- 6
section("6. The paper's numbers")
print("  HumanEval pass@1 / @10 / @100 (%)  (Table 1)")
for name in ("GPT-Neo 2.7B", "GPT-J 6B", "TabNine", "Codex-85M", "Codex-300M", "Codex-2.5B", "Codex-12B"):
    print(f"    {name:13s} {TABLE_1[name][0]:6.2f} {TABLE_1[name][1]:6.2f} {TABLE_1[name][2]:6.2f}")
print("  Codex-S-12B: 37.7% pass@1; 77.5% within 100 samples; 44.5% picking one sample by mean log-prob")
print("  test loss (N / 5.92e7)^-0.13: " + ", ".join(f"{n}: {code_loss(v):.3f}" for n, v in
                                                   (("12M", 12e6), ("300M", 3e8), ("12B", 12e9))) + " nats/token")
print(f"\nTotal time: {time.time() - T0:.1f}s")
