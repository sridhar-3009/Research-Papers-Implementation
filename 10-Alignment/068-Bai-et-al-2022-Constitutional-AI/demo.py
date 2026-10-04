"""Constitutional AI in ~2 seconds: the critique -> revision loop on text, harm vs number of revisions (with and without
critiques), AI-feedback labels (soft, chain-of-thought, clamped) and RL from AI feedback compared with RL from
crowdworkers who reward evasive refusals. The language models are SIMULATED (see cai.py); the prompts, label
processing, preference model and RL objective follow the paper."""

import random
import time

import numpy as np

from cai import (CONSTITUTION, FEEDBACK_PRINCIPLES, HARMFUL_TOPICS, PRINCIPLE_BIASES, REPORTED, SimulatedModel,
                 clamp_label, critique_revision_chain, feedback_model_label, harm_score, make_comparisons, respond,
                 rl_against_pm, summarise_policy, train_preference_model)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Stage 1 (SL-CAI): critique and revise under randomly drawn principles")
chain = critique_revision_chain(SimulatedModel(seed=3), HARMFUL_TOPICS[0], 2, True, random.Random(3))
print(f"  Human: How do I {HARMFUL_TOPICS[0]}?")
for i, r in enumerate(chain):
    print(f"  {'initial' if i == 0 else f'revision {i}'}: {r}")
print(f"  e.g. principle: CritiqueRequest: {CONSTITUTION[0][0][:70]}...")
print("  The final revisions (plus helpful samples) are the supervised fine-tuning data.")

section("2. Harmful answers vs number of revisions (Figures 5 and 7), 400 red-team prompts")
rows = {}
for crit in (True, False):
    h = np.zeros(5)
    for i in range(400):
        ch = critique_revision_chain(SimulatedModel(seed=i), HARMFUL_TOPICS[i % 6], 4, crit, random.Random(i))
        h += [harm_score(x) for x in ch]
    rows[crit] = h / 400
print("    revisions                        0       1       2       3       4")
for crit, h in rows.items():
    print(f"    {'critique, then revise' if crit else 'revise directly':28s} " + "  ".join(f"{x:5.1%}" for x in h))
print("  -> each revision removes more harm; writing a critique first helps (the simulated reviser fixes harm 70% of")
print("     the time after a critique vs 45% without; the paper measured the same direction for large models)")

section("3. Stage 2: AI feedback labels from a multiple-choice question")
print(f"  'Consider the following conversation ... {FEEDBACK_PRINCIPLES[0][:60]}... (A) ... (B) ... The answer is:'")
rng = random.Random(0)
for name, kw in (("soft label (normalised log-probs)", {}), ("chain of thought", {"cot": True})):
    ps = [feedback_model_label(True, "explain", "evasive", rng, accuracy=0.8, **kw) for _ in range(500)]
    print(f"    {name:34s}: P(explained refusal preferred over evasive) mean {np.mean(ps):.2f}, values in "
          f"[{min(ps):.2f}, {max(ps):.2f}]")
print(f"  CoT labels are near 0 or 1, so the paper clamps them: 0.98 -> {clamp_label(0.98)}, 0.02 -> {clamp_label(0.02)}")

section("4. RL from AI feedback vs RL from (evasiveness-rewarding) crowdworkers")
styles = {False: ["helpful", "evasive", "explain", "lecture"], True: ["comply", "evasive", "explain", "lecture"]}
human_help = lambda a, b: feedback_model_label(False, a, b, rng, accuracy=0.85)
crowd_harm = lambda a, b: feedback_model_label(True, a, b, rng, accuracy=0.85, principle_bias={"evasive": 1.5})
ai_soft = lambda a, b: feedback_model_label(True, a, b, rng, accuracy=0.8, principle_bias=random.choice(PRINCIPLE_BIASES))
ai_cot = lambda a, b: feedback_model_label(True, a, b, rng, accuracy=0.9, cot=True, principle_bias=random.choice(PRINCIPLE_BIASES))
ai_cot_clamped = lambda a, b: clamp_label(ai_cot(a, b))
ai_single = lambda a, b: feedback_model_label(True, a, b, rng, accuracy=0.8, principle_bias=PRINCIPLE_BIASES[1])
print("  preference model: Bradley-Terry with soft targets on 4000 comparisons (half helpfulness from humans, half")
print("  harmlessness from the labeler below); policy: RL against it with a KL penalty (beta = 0.1)")
print("    harmlessness labels from          true utility  harmful compliance  evasive  explains  lectures  "
      "PM reward gap (explain - comply)")
init = {False: np.zeros(5), True: np.zeros(5)}
for name, lab in (("crowdworkers (reward evasion)", crowd_harm), ("AI feedback, soft, 4 principles", ai_soft),
                  ("AI feedback, soft, 1 'polite' principle", ai_single),
                  ("AI feedback, CoT (hard)", ai_cot), ("AI feedback, CoT clamped 40-60%", ai_cot_clamped)):
    w = train_preference_model(make_comparisons(rng, 4000, lab, human_help, styles))
    s = summarise_policy(rl_against_pm(w, init, beta=0.1))
    gap = w[5 + 2] - w[5 + 0]
    print(f"    {name:39s}{s['true utility']:5.2f}        {s['harmful compliance']:5.1%}         "
          f"{s['evasive on harmful']:5.1%}   {s['explains on harmful']:5.1%}    {s['lectures on harmful']:5.1%}      {gap:5.2f}")
print("  -> crowdworkers who reward curt refusals produce an EVASIVE assistant; AI feedback guided by the constitution")
print("     yields one that refuses harmful requests BY EXPLAINING (harmless and non-evasive), with higher true utility.")
print("  -> one principle alone carries its own bias (this 'polite' one over-rates lecturing), which the policy then")
print("     learns; sampling a different principle for each label averages such biases out (the paper's ensembling).")
print("  -> here hard CoT labels gave the same preference model as soft labels (both rank the styles correctly, and a")
print("     Bradley-Terry fit on many labels averages out the 0/1 noise); CLAMPING to 40-60% caps the model's confidence")
print("     (reward gap 0.55 vs ~4.7), which makes the RL policy far less peaked. Honest note: in this toy that only costs")
print("     a little utility; we do not reproduce the paper's finding that unclamped CoT labels made RL-CAI 'more extreme'.")

section("5. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
