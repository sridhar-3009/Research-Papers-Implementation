"""Zero-shot chain of thought in ~25 seconds: the two-stage pipeline (reason, then extract the answer) on a tiny model
'pre-trained' on text where different trigger phrases precede different continuations: answer directly, reason
step by step, irrelevant filler, misleading noise. No exemplars at test time; only the trigger changes."""

import random
import time

import torch

from zeroshot import (TABLE_1, TABLE_2_MULTIARITH, TABLE_2_PALM_GSM8K, TABLE_4, TOY_TRIGGERS, answer_prompt, cleanse,
                      reasoning_prompt, self_consistency, toy_answer, toy_problem, train_toy, zero_shot_prompt)

torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The two prompts (Figure 2)")
q = "On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. How many punches did he throw?"
x1 = reasoning_prompt(q)
z = "In one minute, Joe throws 25 punches. In three minutes, Joe throws 3 * 25 = 75 punches. In five rounds, Joe throws 5 * 75 = 375 punches."
print("  1st prompt (reasoning extraction):\n    " + x1.replace("\n", "\n    "))
print(f"  model writes z: '{z}'")
print("  2nd prompt (answer extraction) ends with: '... " + answer_prompt(x1, z)[-60:].replace("\n", " ") + "'")
print(f"  model writes ' 375.' -> cleansed answer {cleanse(' 375.')}  (baseline prompt: '{zero_shot_prompt(q)[-35:]}')")

section("2. A toy 'pre-trained' model: which trigger makes it reason?")
print("  corpus after each question 'Q3729=': 30% '>1;' (direct), 60% 'T302>1;' (step-by-step text), 5% 'R...>1;'")
print("  (irrelevant filler), 5% 'X581>4;' (misleading noise). One model, 2000 steps; no exemplars at test time.")
t = time.time()
model, loss = train_toy(steps=2000)
print(f"  trained in {time.time() - t:.1f}s, final loss {loss:.3f}")
rows = {}
for name, trig in (("zero-shot (answer trigger only)", None), ("T: " + TOY_TRIGGERS["T"], "T"),
                   ("R: " + TOY_TRIGGERS["R"], "R"), ("X: " + TOY_TRIGGERS["X"], "X")):
    accs = []
    for k in (2, 4, 6):
        rng, ok = random.Random(3), 0
        for _ in range(150):
            qq, run = toy_problem(rng, k)
            ok += toy_answer(model, qq, trig)[0] == str(run[-1])
        accs.append(ok / 150)
    rows[name] = accs
print("    prompt                                          k=2     k=4     k=6   (chance 10%)")
for name, accs in rows.items():
    print(f"    {name:44s} " + "  ".join(f"{a:6.1%}" for a in accs))
example_q, _ = toy_problem(random.Random(11), 5)
ans, zz = toy_answer(model, example_q, "T")
print(f"  e.g. stage 1: '{example_q}T' -> reasoning '{zz}'; stage 2: '{example_q}T{zz}>' -> answer {ans}")
print("  -> as in Table 4: the instructive trigger makes the same model reason and wins on multi-step problems;")
print("     an irrelevant trigger is no better than the baseline; a misleading one is worse (even on easy k = 2).")

section("3. Self-consistency: sample several reasoning paths, vote (Table 2's PaLM + self-consistency)")
rng, g = random.Random(21), torch.Generator().manual_seed(0)
greedy = vote = single = 0
n = 60
for _ in range(n):
    qq, run = toy_problem(rng, 6)
    greedy += toy_answer(model, qq, "T")[0] == str(run[-1])
    answers = [toy_answer(model, qq, "T", temperature=1.0, gen=g)[0] for _ in range(8)]
    vote += self_consistency(answers) == str(run[-1])
    single += sum(a == str(run[-1]) for a in answers) / len(answers)
print(f"  k = 6, {n} problems: greedy reasoning {greedy / n:.1%}; one sampled path (T = 1) {single / n:.1%}; "
      f"majority vote of 8 sampled paths {vote / n:.1%}")
print("  -> voting lifts the sampled paths a lot, but here it does not beat GREEDY: sampling at T = 1 makes each")
print("     path much worse, and this tiny model's errors are not diverse enough for the vote to recover. Self-")
print("     consistency pays off when individual samples are nearly as good as greedy and their errors disagree.")

section("4. The paper's numbers")
print("  Table 1 (text-davinci-002), zero-shot -> zero-shot-CoT:")
for k_ in ("MultiArith", "GSM8K", "Last Letter (4 words)", "Coin Flip (4 times)", "SingleEq", "CommonsenseQA"):
    a, b = TABLE_1[k_]
    print(f"    {k_:22s} {a:5.1f} -> {b:5.1f}")
print("  Table 4 (MultiArith): " + "; ".join(f"{c[:5]} '{t[:28]}' {a}" for c, t, a in TABLE_4[:2] + TABLE_4[9:10] + TABLE_4[13:14]))
print("  Table 2 MultiArith: " + ", ".join(f"{k_} {v}" for k_, v in TABLE_2_MULTIARITH.items()))
print("  Table 2 PaLM 540B GSM8K: " + ", ".join(f"{k_} {v}" for k_, v in TABLE_2_PALM_GSM8K.items()))
print(f"\nTotal time: {time.time() - T0:.1f}s")
