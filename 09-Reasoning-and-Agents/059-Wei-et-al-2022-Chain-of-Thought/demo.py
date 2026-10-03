"""Chain of thought in ~25 seconds. Prompting a real 100B+ model is not possible on a laptop, so the CORE claim is
tested in miniature: on a multi-step task (sum of k digits mod 10), a tiny Transformer that writes the intermediate
steps before its answer vs one that answers directly, plus the paper's two ablations (same number of tokens as dots;
the same steps AFTER the answer). Each format gets its own model trained for the same number of steps."""

import time

import torch

from cot import (EXEMPLARS, FORMATS, PRIOR_BEST_GSM8K, TABLE_2_GSM8K, cot_prompt, evaluate_toy, extract_answer,
                 standard_prompt, train_toy)

torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Standard vs chain-of-thought prompts (Figure 1)")
q = "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"
print("  --- standard (one exemplar shown) ---")
print("  " + standard_prompt(EXEMPLARS[:1], q).replace("\n", "\n  "))
print("  --- chain of thought ---")
print("  " + cot_prompt(EXEMPLARS[:1], q).replace("\n", "\n  "))
out = " The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9."
print(f"  a chain-of-thought completion (the paper's Figure 1) -> extracted answer: {extract_answer(out)}")

section("2. The toy: sum of k digits mod 10, trained on 2-6 digits")
print("  e.g. Q3729= : standard '1;'   chain of thought '302>1;' (running sums 3, 3+7=0, 0+2=2, 2+9=1)")
print("        dots '...>1;' (same length, no content)   after '1>302;' (same steps, but after the answer)")
results = {}
for fmt in FORMATS:
    t = time.time()
    model, loss = train_toy(fmt, steps=600)
    accs = {}
    for k in (2, 4, 6, 8):
        acc, chain_ok, samples = evaluate_toy(model, fmt, k)
        accs[k] = acc
        if fmt == "chain of thought" and k in (6, 8):
            print(f"    chain of thought, k={k}: e.g. {samples[0][0]}{samples[0][1]} (correct answer {samples[0][2]})"
                  + (f"; correct answers with fully correct chains: {chain_ok:.0%}" if chain_ok is not None else ""))
    results[fmt] = accs
    print(f"  {fmt:26s} trained 600 steps ({time.time() - t:.1f}s), final loss {loss:.3f}")

print("\n  accuracy by number of digits k (chance = 10%; k = 8 is LONGER than anything seen in training)")
print("    format                       k=2     k=4     k=6     k=8")
for fmt, accs in results.items():
    print(f"    {fmt:26s} " + "  ".join(f"{accs[k]:6.1%}" for k in (2, 4, 6, 8)))
print("  -> writing the intermediate steps BEFORE the answer turns a task the direct model mostly fails (beyond 2")
print("     digits) into one it solves perfectly; the same number of extra tokens without content (dots) does not")
print("     help, and the same steps written AFTER the answer do not help: as in the paper's ablation (Figure 5).")
print(f"     (In our toy the dots even HURT at k = 2, {results[FORMATS[2]][2]:.1%} vs {results[FORMATS[0]][2]:.1%}: the answer sits farther from the digits; the paper")
print("     found dots about equal to the baseline.)")
print("  -> honest limits: the toy models were TRAINED on each format (the paper only PROMPTS a frozen model), and")
print("     chain of thought fails at k = 8: the model never saw chains that long, writes too few steps, and its")
print("     few 'correct' answers there come with wrong chains (lucky guesses), so it does not generalise in length.")

section("3. The paper's GSM8K numbers (Table 2): standard -> chain of thought")
for k, (s, c) in TABLE_2_GSM8K.items():
    print(f"  {k:30s} {s:5.1f}% -> {c:5.1f}%  ({c / s:.1f}x)")
print(f"  prior best (finetuned GPT-3 175B + verifier): {PRIOR_BEST_GSM8K}%; small models (< ~100B) gain nothing or lose")
print(f"\nTotal time: {time.time() - T0:.1f}s")
