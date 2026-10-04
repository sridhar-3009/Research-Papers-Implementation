"""InstructGPT's three steps in ~20 seconds, with real tiny networks: a pre-trained base model that continues text but
does not follow instructions, SFT on a few demonstrations, a reward model from K-way rankings with the paper's
per-prompt loss, and PPO vs PPO-ptx (pre-training gradients mixed in) with the 'alignment tax' measured on the
pre-training distribution."""

import itertools
import time

import numpy as np
import torch

from instructgpt import (REPORTED, collect_rankings, evaluate, labeler_utility, pretrain, ppo, respond, score, sft,
                         train_reward_model, prompt_tokens)

torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


def show(name, r):
    print(f"    {name:34s} utility {r['utility']:5.2f}   exact {r['exact']:5.1%}   pre-training benchmark {r['benchmark']:5.1%}")


section("1. A pre-trained base model: good at its training text, not at following instructions")
t = time.time()
base = pretrain(800)
print(f"  'web text': 80% periodic letter patterns (the 'public NLP benchmark' is predicting them), 20% instruction-")
print(f"  like text followed by unrelated words, never by the right answer ({time.time() - t:.1f}s)")
show("base model", evaluate(base))
print("  e.g. 'SORT d b c a ->'", " ".join(respond(base, [prompt_tokens("SORT", list("dbca"))], 0.0)[0]),
      "  (it continues text instead of doing the task: 'misaligned' with the user)")

section("2. Step 1: supervised fine-tuning on 200 labeler demonstrations")
s = sft(base, 300, 200)
show("SFT", evaluate(s))
print("  -> instruction following appears, but the benchmark drops: fine-tuning has an 'alignment tax'")

section("3. Step 2: reward model from K-way rankings (K = 4..9), all C(K,2) pairs of a prompt in one batch element")
t = time.time()
rankings = collect_rankings(s, 1000)
rm = train_reward_model(s, rankings)
test = collect_rankings(s, 150, seed=77, temperature=1.0)
hits = []
for p, ranked in test:
    sc = score(rm, [p] * len(ranked), ranked).tolist()
    u = [labeler_utility(*p, r) for r in ranked]
    hits += [(sc[i] > sc[j]) == (u[i] > u[j]) for i, j in itertools.combinations(range(len(ranked)), 2) if u[i] != u[j]]
n_pairs = sum(len(r) * (len(r) - 1) // 2 for _, r in rankings)
print(f"  1000 ranked prompts -> {n_pairs} comparisons, but only {sum(len(r) for _, r in rankings)} RM forward passes "
      f"({time.time() - t:.1f}s)")
print(f"  held-out pairwise accuracy against the labelers' true utility: {np.mean(hits):.1%}")

section("4. Step 3: PPO (per-token KL to SFT) vs PPO-ptx (+ gamma x pre-training log-likelihood)")
res = {"SFT": evaluate(s)}
for name, g in (("PPO", 0.0), ("PPO-ptx (gamma = 0.3)", 0.3)):
    t = time.time()
    pol, hist = ppo(s, rm, beta=0.05, gamma_ptx=g, iters=40)
    res[name] = evaluate(pol)
    print(f"  {name}: {time.time() - t:.1f}s, KL to SFT {hist[-1][2]:.2f} nats")
for k, v in res.items():
    show(k, v)
print("  -> PPO-ptx keeps the instruction-following of SFT and recovers the pre-training benchmark: mixing in")
print("     pre-training gradients pays most of the alignment tax, as in the paper.")
print("  -> honest result: our PPO does NOT improve instruction following beyond SFT. The reward model is only")
print(f"     {np.mean(hits):.0%} accurate while the SFT policy already answers {res['SFT']['exact']:.0%} of prompts exactly, so the RM")
print("     cannot tell it how to do better (longer PPO slowly made it worse). In paper 066's toy, where the RM's")
print("     signal exceeded the policy's own competence, RL helped; InstructGPT's 6B RM on 33k prompts is far stronger.")

section("5. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
