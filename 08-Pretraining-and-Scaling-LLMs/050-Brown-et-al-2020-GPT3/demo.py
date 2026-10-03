"""A ~8-second tour of GPT-3.

  1. Table 2.1 and the compute: 12 L d^2 parameters, 6 N D FLOPs, petaflop/s-days
  2. Table 2.2: sampling by quality, not size
  3. In-context learning, measured exactly: every sequence has its own random token distribution, so the only way to
     predict well is to learn it from the context. We compare a small transformer with the Bayes-optimal learner.
  4. Zero-, one- and few-shot prompts, and how multiple-choice answers are scored
  5. Data hygiene: the Pareto quality filter, MinHash fuzzy de-duplication, 13-gram contamination checks
"""

import importlib.util
import math
import random
import time
from pathlib import Path

import torch

torch.set_num_threads(2)
from gpt3 import (PFS_DAY, TABLE_2_1, arithmetic_example, bayes_optimal_nll, build_prompt, choose, dedup,  # noqa
                  dirichlet_sequences, epochs_elapsed, estimated_jaccard, is_dirty, jaccard, minhash, ngram_set,
                  param_estimate, pareto_keep, shingles, training_flops)

spec = importlib.util.spec_from_file_location(
    "gpt2_049", Path(__file__).resolve().parents[1] / "049-Radford-et-al-2019-GPT-2" / "gpt2.py")
gpt2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpt2)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. Table 2.1: sizes and training compute (all trained on 300B tokens)")
print(f"  {'model':>6} {'layers':>6} {'d_model':>7} {'paper n_params':>14} {'12Ld^2 + emb.':>13} {'6ND (pf/s-days)':>15}")
for name, (n, L, d, *_rest) in TABLE_2_1.items():
    print(f"  {name:>6} {L:6d} {d:7d} {n / 1e9:13.3f}B {param_estimate(L, d) / 1e9:12.3f}B "
          f"{training_flops(n, 300e9) / PFS_DAY:15.1f}")
print("  -> 175B x 300B tokens x 6 = 3.15e23 FLOPs = ~3,640 petaflop/s-days. The bigger models saw FEWER tokens per")
print("     parameter than usual practice, following Kaplan et al.'s scaling laws (paper 051).")

# --------------------------------------------------------------------------------------------- 2
section("2. Table 2.2: epochs of each dataset during 300B training tokens (weight x 300B / size)")
for k, v in epochs_elapsed().items():
    print(f"  {k:24s} {v:5.2f} epochs")
print("  -> high-quality sets are repeated 2-3x, Common Crawl is seen less than once. (Computed from the paper's")
print("     ROUNDED weights, which sum to 101%; the paper's own epoch column (WebText2 2.9, Wikipedia 3.4) differs a bit.)")

# --------------------------------------------------------------------------------------------- 3
section("3. In-context learning: predicting tokens from a distribution that is new in every sequence")
V, T, ALPHA = 10, 48, 0.3
x_test = dirichlet_sequences(2000, T, V, ALPHA, generator=torch.Generator().manual_seed(1))
bayes = bayes_optimal_nll(x_test, V, ALPHA)[:, 1:].mean(0)
curves = {}
for d in (8, 32):
    torch.manual_seed(0)
    m = gpt2.GPT2(vocab=V, n_ctx=T, d=d, layers=1, heads=2)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    for _ in range(800):
        loss = m.loss(dirichlet_sequences(64, T, V, ALPHA))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lg = m(x_test)
        nll = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, V), x_test[:, 1:].reshape(-1),
                                                reduction="none").view(2000, T - 1)
    curves[d] = nll.mean(0)
print(f"  {'tokens seen':>11} {'d = 8 model':>11} {'d = 32 model':>12} {'Bayes-optimal':>13}   (nats; uniform = {math.log(V):.3f})")
for t in (0, 1, 3, 7, 15, 31, 46):
    print(f"  {t + 1:11d} {curves[8][t]:11.3f} {curves[32][t]:12.3f} {bayes[t]:13.3f}")
print("  -> no weights change at test time, yet predictions improve with every token seen: the network has learned an")
print("     ESTIMATION PROCEDURE that runs inside its forward pass, close to the Bayes-optimal one. That is what")
print("     'in-context learning' means. Here the wider model tracks the optimum more closely, echoing the paper's")
print("     finding that larger models use their context better.")

# --------------------------------------------------------------------------------------------- 4
section("4. Zero-, one- and few-shot prompts (Figure 2.1), on Section 3.9's arithmetic task")
rng = random.Random(0)
demos = [tuple(s.replace("Q: ", "").replace("A: ", "") for s in arithmetic_example(rng)) for _ in range(3)]
query = arithmetic_example(rng)[0].replace("Q: ", "")
for k, name in ((0, "zero-shot"), (1, "one-shot"), (3, "few-shot (K = 3)")):
    print(f"  --- {name} ---")
    for line in build_prompt("Answer the arithmetic question.", demos, query, k, sep="\n", arrow=" ").split("\n"):
        print(f"    {line}")
table = {("ctx", "Paris"): (-2.0, 1), ("ctx", "the city of light"): (-5.0, 4),
         ("Answer:", "Paris"): (-6.0, 1), ("Answer:", "the city of light"): (-12.0, 4)}
lp = lambda c, s: table[(c, s)]
for mode in ("sum", "per_token", "unconditional"):
    print(f"  multiple choice, scored by {mode:13s}: picks {['Paris', 'the city of light'][choose(lp, 'ctx', ['Paris', 'the city of light'], mode)]!r}")
print("  -> length normalisation stops long answers being penalised for having more tokens; dividing by")
print("     P(answer | 'Answer:') removes answers that are just generically probable (used for ARC, OBQA, RACE).")
print("  GPT-3 175B few-shot (Table 3.9): 2-digit addition 100%, 3-digit 80.4%, 4-digit 25.5%, 5-digit 9.3%.")

# --------------------------------------------------------------------------------------------- 5
section("5. Data hygiene")
r = random.Random(0)
for score in (0.99, 0.9, 0.5, 0.1):
    kept = sum(pareto_keep(score, r) for _ in range(20000)) / 20000
    print(f"  Pareto(alpha = 9) filter, classifier score {score:4.2f}: kept {100 * kept:5.1f}% of such documents")
a = "the quick brown fox jumps over the lazy dog again and again in the park today"
b = a.replace("today", "tonight")
c = "a completely unrelated note about baking bread with flour water and salt"
print(f"  near-duplicate pair: true Jaccard {jaccard(shingles(a), shingles(b)):.2f}, MinHash estimate (10 hashes) "
      f"{estimated_jaccard(minhash(a), minhash(b)):.2f}; dedup keeps {len(dedup([a, b, c]))} of 3 documents")
train = ngram_set(a + " " + c, 13)
print(f"  13-gram contamination: test example copied from training -> dirty = {is_dirty(a, train)}; "
      f"a new sentence -> dirty = {is_dirty('another fresh test question that nobody wrote down before in this corpus anywhere', train)}")
print("  -> the paper found a bug left some overlaps in; Section 4 re-scored 'clean' subsets and found little effect on")
print("     most benchmarks.")

print(f"\n(total {time.time() - T0:.1f} s)")
