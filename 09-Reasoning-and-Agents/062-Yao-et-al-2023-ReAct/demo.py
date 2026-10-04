"""ReAct in ~25 seconds: (1) the Thought / Action / Observation loop on a small local 'Wikipedia', driven by a
scripted model to show the format; (2) four tiny models (Standard, CoT, Act, ReAct) trained by imitation on a toy
encyclopedia, tested when the WORLD CHANGES after training (grounding vs hallucination), and when the search tool
fails (the paper's ReAct -> CoT back-off)."""

import random
import time

import torch

from react import DECISION_MAKING, TABLE_1, TABLE_2, ToyWorld, WikiEnv, react_loop, react_then_cot, run_toy, train_toy

torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The ReAct loop (Figure 1d) on a local Wikipedia (fictional pages, scripted thoughts)")
pages = {"Velmora": "Velmora is a 1998 drama film. It was directed by Ana Kord. It won two awards.",
         "Ana Kord": "Ana Kord is a film director and writer. She was born in Brisk. She studied in Tallin.",
         "Brisk": "Brisk is a coastal town. Its population is 12,400."}
script = iter([" I need to find who directed Velmora, then where that person was born.\nAction 1: search[Velmora]",
               " Velmora was directed by Ana Kord. I need to find where Ana Kord was born.\nAction 2: search[Ana Kord]",
               " Ana Kord was born in Brisk. So the answer is Brisk.\nAction 3: finish[Brisk]"])
answer, transcript = react_loop(lambda prompt, stop: next(script), "Where was the director of Velmora born?", WikiEnv(pages))
print("  " + transcript.strip().replace("\n", "\n  "))
print(f"  -> answer {answer}. Thoughts plan and extract; actions fetch facts; observations come from the environment.")

section("2. A toy encyclopedia: 200 films -> 120 directors -> 20 cities; 'where was the director of film F born?'")
world = ToyWorld()
print("  trajectories the four models imitate (film F0):")
for m in ("standard", "cot", "act", "react"):
    print(f"    {m:8s} " + " ".join(world.trajectory("F0", m)))
print("  (tool users see random 'counterfactual' pages half the time in training, so they must READ observations;")
print("   observation tokens are written by the environment, never trained as model outputs)")
models = {}
for m in ("standard", "cot", "act", "react"):
    t = time.time()
    models[m], loss = train_toy(world, m, steps=500)
    print(f"  trained {m:8s} ({time.time() - t:.1f}s, final loss {loss:.3f})")


def accuracy(method, films):
    return sum(run_toy(world, models[method], f, method)[0] == world.answer(f) for f in films) / len(films)


old = dict(world.director)
rows = {m: [accuracy(m, world.films[:100])] for m in models}
rng = random.Random(5)
for f in world.films[100:]:                                                   # the world changes after training
    world.director[f] = f"P{rng.randrange(120)}"
for m in models:
    rows[m].append(accuracy(m, world.films[100:]))
example_film = world.films[100]
for m in ("cot", "react"):
    print(f"  {m} on updated {example_film} (truth {world.answer(example_film)}): " + " ".join(run_toy(world, models[m], example_film, m)[1]))
world.director.update(old)
world.outage = True                                                           # the search tool is down
for m in models:
    rows[m].append(accuracy(m, world.films[:100]))
outage_backoff = guessed = 0
for f in world.films[:100]:
    ans, _ = run_toy(world, models["react"], f, "react")
    guessed += ans is not None
    cot_ans, _ = run_toy(world, models["cot"], f, "cot")
    outage_backoff += react_then_cot(ans, [cot_ans])[0] == world.answer(f)
world.outage = False
print("\n    method      facts as trained   facts CHANGED after training   search tool down   (chance 5%)")
for m, (a, b, c) in rows.items():
    print(f"    {m:10s}      {a:6.1%}                 {b:6.1%}                     {c:6.1%}")
print(f"    ReAct -> CoT back-off (when ReAct returns no answer)               {outage_backoff / 100:6.1%}")
print(f"    (with the tool down, ReAct still wrote an evidence-free answer in {guessed} of 100 cases, so the back-off never")
print("     fired there: a back-off rule only helps when the agent admits it is stuck)")
print("  -> closed-book models (Standard, CoT) answer from memory: perfect on what they memorised, but they confidently")
print("     state STALE facts when the world changes (the 'hallucination' failure of Table 2). Tool users read the")
print("     current encyclopedia and stay right. When the tool fails, ReAct gets stuck; backing off to CoT recovers")
print("     the memorised answers: the paper's motivation for combining internal and external knowledge.")
print("  -> honest gap: Act = ReAct here. Our toy has one fixed plan; the paper's thoughts pay off when the model must")
print("     decide what to search next, recover from bad searches and synthesise the answer (ReAct > Act in Table 1).")

section("3. The paper's numbers")
print("  Table 1 (PaLM-540B), HotpotQA EM / FEVER accuracy:")
for k, (h, f) in TABLE_1.items():
    print(f"    {k:22s} {h:5.1f} / {f:5.1f}")
print("  Table 2 (HotpotQA, ReAct % / CoT %): " + "; ".join(f"{k} {a}/{b}" for k, (a, b) in TABLE_2.items()))
print("  decision making: " + ", ".join(f"{k} {v}" for k, v in DECISION_MAKING.items()))
print(f"\nTotal time: {time.time() - T0:.1f}s")
