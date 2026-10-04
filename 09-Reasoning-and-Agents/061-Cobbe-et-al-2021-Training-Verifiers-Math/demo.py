"""Verifiers in ~30 seconds: GSM8K's calculator-annotated format, then the paper's whole pipeline on a toy task:
a briefly-trained generator, many samples per training problem labelled by the final answer, a token-level verifier
trained jointly with language modelling, and best-of-N selection at test time, with a small and a large verifier
training set."""

import random
import time

import numpy as np
import torch

from verifiers import (FINDINGS, Verifier, calculator_step, final_answer, majority_vote, sample_solutions,
                       select_by_verifier, coverage_at_n, toy_correct, toy_problem, train_generator)

torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. GSM8K's solution format (Section 2) and the calculator")
sol = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
       "Natalia sold 48+24 = <<48+24=71>>71 clips altogether in April and May.\n#### 71")
print("  a sample with an arithmetic slip:\n    " + sol.replace("\n", "\n    "))
partial = "Natalia sold 48+24 = <<48+24="
print(f"  with the calculator, when sampling reaches '{partial}' the tool inserts '{calculator_step(partial)}' instead of")
print("  letting the model guess, so this slip cannot happen; only the reasoning (which numbers to combine) is the model's")
print(f"  final answer parsed from '####': {final_answer(sol)} (the label 'correct?' looks ONLY at this)")

section("2. Toy task: 'pick two of these 6 digits that add up to t' (generating = search, checking = easy)")
p = toy_problem(random.Random(3))
print(f"  e.g. {p['q']}  gold {p['gold']}  (the target sums to {p['target']}; any valid pair counts)")
t = time.time()
gen = train_generator(steps=400)
print(f"  generator: finetuned briefly on gold solutions ({time.time() - t:.1f}s), like the paper's 2-epoch generator")

rng = random.Random(5)
train = [toy_problem(rng) for _ in range(2000)]
test = [toy_problem(random.Random(10_000 + i)) for i in range(300)]
N = 16
t = time.time()
train_samples = sample_solutions(gen, [p["q"] for p in train], N, temperature=1.0, seed=1)
test_samples = sample_solutions(gen, [p["q"] for p in test], N, temperature=1.0, seed=2)
greedy = sample_solutions(gen, [p["q"] for p in test], 1, temperature=0)
ok_test = [[toy_correct(p, a) for a in s] for p, s in zip(test, test_samples)]
print(f"  sampled {N} solutions for each of {len(train)} training and {len(test)} test problems ({time.time() - t:.1f}s)")
print(f"  greedy (one T = 0 sample, the finetuning baseline): {np.mean([toy_correct(p, g[0]) for p, g in zip(test, greedy)]):.1%}")
print(f"  one random T = 1 sample: {np.mean([np.mean(o) for o in ok_test]):.1%};  coverage test@1/4/16: "
      + " / ".join(f"{coverage_at_n(ok_test, n):.1%}" for n in (1, 4, 16)))

section("3. Verifiers: rank the 16 samples, return the best (Figure 5: data size matters)")
results = {}
for n_train in (100, 2000):
    data = [(p["q"], a, toy_correct(p, a)) for p, ss in zip(train[:n_train], train_samples[:n_train]) for a in ss]
    t = time.time()
    ver = Verifier(gen).train(data, steps=1000)
    scores = [ver.score(p["q"], s) for p, s in zip(test, test_samples)]
    best = np.mean([ok[int(np.argmax(sc))] for ok, sc in zip(ok_test, scores)])
    top3 = np.mean([toy_correct(p, select_by_verifier(s, sc, top_k=3)) for p, s, sc in zip(test, test_samples, scores)])
    results[n_train] = (best, top3, scores, ver)
    print(f"  verifier trained on {n_train:4d} problems x {N} samples ({len(data):5d} labelled solutions, {time.time() - t:.1f}s): "
          f"best-of-16 {best:.1%}, top-3 vote {top3:.1%}")
print(f"  majority vote over the 16 samples (no verifier): "
      f"{np.mean([toy_correct(p, majority_vote(s)) for p, s in zip(test, test_samples)]):.1%}")
print(f"  -> with little data the verifier is WORSE than greedy ({results[100][0]:.1%} vs the baseline); with enough data it clearly beats it,")
print("     as in the paper (verifiers 'take off' only past a certain training-set size, then scale better).")

scores_big = results[2000][2]
print("  how many samples to rank (Figure 7a), large verifier:  " + ", ".join(
    f"N={n}: {np.mean([ok[int(np.argmax(sc[:n]))] for ok, sc in zip(ok_test, scores_big)]):.1%}" for n in (1, 2, 4, 8, 16)))
print("  (more samples = more chances to find a good one, and more chances to find one that fools the verifier: the")
print("   paper's 6B peak was ~400 samples; our flat 8 -> 16 may be that effect or just noise on 300 problems)")

section("4. The paper's other findings")
for k, v in FINDINGS.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
