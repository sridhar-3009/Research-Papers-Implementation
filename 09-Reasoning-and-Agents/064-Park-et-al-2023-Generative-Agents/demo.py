"""Generative agents in ~2 seconds: the memory stream's retrieval score worked out by hand, a reflection tree, and a
25-agent toy town where two pieces of news (a candidacy and a Valentine's party) spread only through conversations
that share RETRIEVED memories, averaged over 5 seeds, with retrieval ablations. The LLM calls of the paper are
replaced by transparent stand-ins (see agents.py)."""

import time

import numpy as np

from agents import DECAY, EMERGENCE, TRUESKILL, MemoryStream, reflect, reflection_depth, run_town

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Retrieval = recency + importance + relevance, each min-max scaled (Section 4.1)")
ms = MemoryStream()
ms.add("Klaus ate breakfast in his dorm", t=0, importance=1)
ms.add("Klaus talked with Maria about his research on gentrification", t=10, importance=6)
ms.add("Klaus saw that the library desk is free", t=22, importance=2)
query = "What is Klaus passionate about? research"
total, (rec, imp, rel) = ms.scores(query, t=24)
print(f"  query: '{query}' at hour 24; recency = {DECAY}^(hours since last access), then min-max scaled")
print("    memory                                                     recency  importance  relevance  total")
for m, r, i, l, s in zip(ms.records, rec, imp, rel, total):
    print(f"    {m.text[:58]:58s}  {r:5.2f}     {i:5.2f}      {l:5.2f}    {s:4.2f}")
print(f"  raw recencies {[round(DECAY ** (24 - m.last_access), 3) for m in ms.records]}: tiny differences, stretched to")
print("  [0, 1] by the scaling, so recency is a strong force even though 0.995 per hour decays slowly")

section("2. Reflection (Section 4.2): triggered when recent importance exceeds 150; reflections cite evidence")
ms = MemoryStream()
for i, txt in enumerate(["Klaus is reading a book on gentrification", "Klaus is writing his research paper",
                         "Klaus asked the librarian about gentrification research", "Klaus discussed research with Maria"] * 8):
    ms.add(txt, t=i, importance=5)
print(f"  32 observations of importance 5 -> summed {ms.importance_since_reflection} > 150: reflect = {ms.should_reflect()}")
level1 = reflect("Klaus", ms, t=40)
for r in level1:
    print(f"    reflection (depth {reflection_depth(r)}): {r.text}")
for i in range(32):
    ms.add("Klaus is passionate about gentrification research", t=41 + i, importance=5)
level2 = reflect("Klaus", ms, t=80)
depth = max(reflection_depth(r) for r in level2)
print(f"  second round: max depth {depth}. " + ("Reflections cited earlier reflections: a reflection tree." if depth > 1 else
      "Here retrieval surfaced only observations, so no deeper level formed; in the paper reflections often cite\n"
      "  earlier reflections (Figure 7), which only happens when those are retrieved as evidence."))

section("3. A 25-agent town over 2 game days: does news spread through retrieved memories? (Section 7.1)")
print("  agent0 is running for mayor; agent5 throws a Valentine's party at the cafe on day 2, 17:00-19:00.")
print("  In a conversation each speaker mentions the first NEWS item among its top 30 retrieved memories; each")
print("  morning an agent plans its day from its top 10 memories for 'what should I do today'. 5 seeds.")
configs = {"full retrieval (1, 1, 1)": {}, "no importance (1, 0, 1)": {"weights": (1, 0, 1)},
           "no recency (0, 1, 1)": {"weights": (0, 1, 1)}, "no relevance (1, 1, 0)": {"weights": (1, 1, 0)},
           "full, no reflection": {"reflect": False}}
print("    condition                    knew of candidacy   knew of party   came to party   density start -> end")
for name, kw in configs.items():
    rs = [run_town(seed=s, **kw) for s in range(5)]
    mean = lambda k: np.mean([r[k] for r in rs])
    d0, d1 = np.mean([r["density"][0] for r in rs]), np.mean([r["density"][1] for r in rs])
    print(f"    {name:28s}     {mean('mayor'):5.1f} / 25       {mean('party'):5.1f} / 25      {mean('attended'):5.1f}         "
          f"{d0:.3f} -> {d1:.3f}")
print("  -> news travels only as far as retrieval surfaces it: without the IMPORTANCE term, days-old news never")
print("     outranks fresh, mundane perceptions, so nothing spreads; knowing about the party is not enough to come:")
print("     the memory must also be retrieved when the agent plans its day (the paper: 13 knew, 5 of 12 invited came).")
print("  -> dropping recency or relevance makes news spread MORE here: both favour fresh, situation-specific")
print("     perceptions ('X is at the cafe') that compete with old news for the speaker's attention.")
print("  -> honest notes: our 'LLM' is a stand-in (keyword importance, hashed embeddings, a fixed rule for what to")
print("     say), so the numbers describe this toy, not Smallville; reflection changes nothing here because our")
print("     conversations and planner only use observations (in the paper reflections shape dialogue and plans).")

section("4. The paper's numbers")
for k, (mu, sd) in TRUESKILL.items():
    print(f"  TrueSkill believability, {k:45s} mu = {mu:5.2f} (sigma {sd})")
for k, v in EMERGENCE.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
