"""The survey's agent framework in ~3 seconds: build agents from profile / memory / planning / action modules, then
measure how each PLANNING strategy and MEMORY structure trades success against cost, using a simulated LLM whose
per-step accuracy p is known (so simulations can be checked against closed-form probabilities)."""

import random
import time

from agentkit import (CHALLENGES, FRAMEWORK, HybridMemory, UnifiedMemory, analytic, evaluate, generated_profiles,
                      key_door_episode, reflexion, self_consistency, single_path, tree_of_thoughts,
                      with_environment_feedback)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The unified framework (Figure 2)")
for k, v in FRAMEWORK.items():
    print(f"  {k:28s} {' | '.join(v)}")
print("  e.g. LLM-generated profiles: " + "; ".join(generated_profiles(2, random.Random(1))))

section("2. Planning strategies on a 6-step task, simulated LLM right on each step with p = 0.8")
print("  (judging a step, as Tree of Thoughts does, is right with q = 0.8; 'calls' = LLM calls per task)")
p, L = 0.8, 6
rows = [("single path (CoT, plan-then-execute)", single_path, {}),
        ("multi-path: self-consistency, 5 chains", self_consistency, {"n_paths": 5}),
        ("multi-path: tree search, 3 proposals + self-evaluation", tree_of_thoughts, {"breadth": 3}),
        ("feedback from the ENVIRONMENT (ReAct-style), 2 retries", with_environment_feedback, {"retries": 2}),
        ("feedback from the MODEL across trials (Reflexion), 3 trials", reflexion, {"trials": 3})]
print("    strategy                                                    success    LLM calls")
for name, f, kw in rows:
    s, c = evaluate(f, p, L, **kw)
    print(f"    {name:58s}  {s:6.1%}     {c:5.1f}")
print(f"  closed forms: single path p^L = {analytic('single_path', p, L):.1%}; environment feedback "
      f"(1-(1-p)^3)^L = {analytic('environment_feedback', p, L, retries=2):.1%}")
print("  -> one chain fails as soon as one step does; extra chains only help if the RIGHT final answer is the most")
print("     common one; self-evaluation helps as far as the judge is accurate; real feedback from the environment is the")
print("     strongest signal per call. Cost differs a lot too: an evaluation must report both (Section 4).")

section("3. How the strategies scale with task length (p = 0.9)")
print("    steps   single path   self-consistency   tree search   env. feedback   reflexion")
for L_ in (2, 5, 10, 20):
    vals = [evaluate(f, 0.9, L_, n=1000, **kw)[0] for _, f, kw in rows]
    print(f"    {L_:5d}     " + "        ".join(f"{v:6.1%}" for v in vals))
print("  -> long tasks are where open-loop planning collapses (0.9^20 = 12%) and feedback becomes essential.")
print("     Caveat: our environment says exactly WHICH step failed and lets the agent retry it for free; real")
print("     environments give noisier, delayed signals, so these feedback numbers are an upper bound.")

section("4. Memory structures: recall a fact seen before `gap` unrelated observations (key-door task)")
rng = random.Random(0)
print("    gap    unified memory (context window of 8)    hybrid memory (window 8 + long-term store)")
for gap in (2, 5, 10, 50, 200):
    u = sum(key_door_episode(UnifiedMemory(8), gap, rng) for _ in range(200)) / 200
    h = sum(key_door_episode(HybridMemory(8, capacity=50), gap, rng) for _ in range(200)) / 200
    print(f"    {gap:4d}            {u:6.1%}                                   {h:6.1%}")
print("  -> a unified (short-term only) memory forgets anything that falls out of the window; a hybrid memory keeps")
print("     it in long-term storage and READS it back by relevance + importance. Writing merges duplicates (only 4")
print("     distinct distractors here), so the store never overflows; with unique distractors and a small capacity,")
print("     eviction of the least important records would protect the key.")

section("5. Open challenges named by the survey")
print("  " + "; ".join(CHALLENGES))
print(f"\nTotal time: {time.time() - T0:.1f}s")
