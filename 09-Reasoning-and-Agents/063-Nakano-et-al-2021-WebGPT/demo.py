"""WebGPT in ~15 seconds: the text browser of Table 1, then the paper's training recipe on a toy web: a stochastic
browsing policy (standing in for the behaviour-cloned model), a simulated human labeler, a reward model trained on
comparisons, rejection sampling (best-of-n) and RL with a KL penalty, both judged by the labeler's TRUE preferences."""

import time

import numpy as np

from webgpt import (RESULTS, SCALING, Browser, LocalWeb, answer_features, best_of_n, elo_preference, format_answer,
                    labeler_compare, make_web, policy_episode, rl_finetune, train_reward_model, true_quality)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The text browser (Table 1): what the model sees and the commands it issues")
web = LocalWeb([{"title": "Tides", "domain": "encyclopedia.org", "authority": 1.5,
                 "text": "Tides are the rise and fall of sea levels. They are caused mainly by the Moon's gravity. "
                         "The Sun also contributes, less strongly. Two high tides happen about every 25 hours."},
                {"title": "Beach diary", "domain": "someblog.net", "text": "I love the beach. Tides are weird."}])
b = Browser(web, "Why are there tides?")
for cmd in ("Search why tides", "Clicked on link 0", "Find in page: Moon", "Quote: They are caused mainly by the Moon's gravity.", "End: Answer"):
    obs = b.step(cmd)
    print(f"  > {cmd}")
print("  final state shown to the model:\n  " + b.summary().replace("\n", "\n  "))
print("  answer phase: " + format_answer("Mostly the Moon's gravity pulls the oceans [1].", b.quotes).replace("\n", " | "))

section("2. Toy web: 40 topics x (reliable encyclopedia page, unreliable blog with a WRONG fact)")
web, truth = make_web()
rng = np.random.default_rng(0)
bc = {"precise": 0.7, "top": 0.6, "find": 0.6, "fill": 1.0}
print(f"  'BC' policy: precise query {bc['precise']}, click top result {bc['top']}, use Find {bc['find']}, "
      f"~{bc['fill']} filler sentences")
text, quotes, _, _ = policy_episode(web, "topic5", bc, rng)
print(f"  e.g. '{text}'  quotes: {[q['domain'] + ': ' + q['extract'] for q in quotes]}  (truth: {truth['topic5']})")
print("  labeler's true utility: correct +2 (+0.5 if supported by a quote), wrong -1, +0.4 per filler up to 2,")
print("  -0.6 per filler beyond 2 (a little detail is liked, verbosity is not)")


def episode(topic, theta=bc):
    t, q, _, _ = policy_episode(web, topic, theta, rng)
    return t, q


t = time.time()
FA, FB, Y = [], [], []
for _ in range(3000):
    topic = f"topic{rng.integers(40)}"
    a, b_ = episode(topic), episode(topic)
    FA.append(answer_features(*a, topic)); FB.append(answer_features(*b_, topic))
    Y.append(labeler_compare(true_quality(*a, topic, truth), true_quality(*b_, topic, truth), rng))
w = train_reward_model(FA[:2500], FB[:2500], Y[:2500])
held = [(fa @ w > fb @ w) == (y > 0.5) for fa, fb, y in zip(FA[2500:], FB[2500:], Y[2500:]) if y != 0.5]
print(f"\n  3000 labelled comparisons ({time.time() - t:.1f}s); linear reward model weights "
      f"[cites, supported, encyclopedia, filler] = {np.round(w[1:], 2)}; held-out accuracy {np.mean(held):.1%}")
print(f"  (the RM score is Elo-like: a difference of 1 means {elo_preference(1.0):.0%} preference)")
rm = lambda t_, q_, topic: float(answer_features(t_, q_, topic) @ w)

section("3. Rejection sampling: best-of-n against the reward model (Figures 5 and 8)")
print("    n      RM score of chosen   TRUE quality   preferred to BC (simulated labeler)")
bon = {}
base = [true_quality(*episode(f"topic{i % 40}"), f"topic{i % 40}", truth) for i in range(200)]
for n in (1, 4, 16, 64, 256):
    scores, quals = [], []
    for i in range(200):
        topic = f"topic{i % 40}"
        best = best_of_n([episode(topic) for _ in range(n)], lambda c: rm(*c, topic))
        scores.append(rm(*best, topic)); quals.append(true_quality(*best, topic, truth))
    pref = np.mean([1 / (1 + np.exp(-(q - qb))) for q, qb in zip(quals, base)])
    bon[n] = np.mean(quals)
    print(f"    {n:3d}         {np.mean(scores):6.2f}            {np.mean(quals):6.2f}           {pref:.1%}")
print("  -> picking the RM's favourite among more samples first helps a lot, then the true quality turns DOWN while")
print("     the RM score keeps rising: with enough samples the search finds the RM's blind spot (it thinks longer is")
print("     always better). That is reward over-optimisation, and why best-of-n has a best n.")

section("4. RL against the reward model, with and without the KL penalty (Section 5.1)")
rl = {}
for beta in (0.0, 0.3):
    t = time.time()
    hist, theta = rl_finetune(web, truth, bc, rm, beta, steps=300, lr=0.3)
    print(f"  beta = {beta}: RM score {hist[0][0]:.2f} -> {hist[-1][0]:.2f}, TRUE quality {hist[0][1]:.2f} -> "
          f"{hist[-1][1]:.2f}  (final policy: top {theta['top']:.2f}, find {theta['find']:.2f}, "
          f"filler {theta['fill']:.1f}; {time.time() - t:.1f}s)")
    rl[beta] = hist[-1][1]
print("  -> without the KL penalty RL learns the good habits (top result, Find) AND hacks the reward model with very")
print("     long answers (filler at its cap), so the RM score climbs while true quality falls. With the KL penalty the")
print(f"     policy stays close enough to BC to keep answers short, and true quality reaches {rl[0.3]:.2f}, close to the")
print(f"     best best-of-n ({max(bon.values()):.2f}). Honest difference: in the paper rejection sampling clearly beat RL")
print("     (68% vs 58% preferred over BC); our 4-parameter policy is far easier to optimise than a 175B model with PPO.")

section("5. The paper's numbers")
for k, v in RESULTS.items():
    print(f"  {k}: {v}")
for k, v in SCALING.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
