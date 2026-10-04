# The code, explained simply

How the code in this folder implements generative agents (Park et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `agents.py` | the memory stream (records, retrieval score, recency refresh), the reflection trigger, questions, insights with citations and the reflection tree, day planning from retrieved memories, a 25-agent town with perception, conversations and plans, `run_town`, the paper's numbers. LLM calls are replaced by transparent stand-ins |
| `experiments.py` | LLM-backed memory, LLM-judged interview ablations, toy parameter sweeps (heavy, not run here) |
| `demo.py` | the retrieval score worked out by hand, reflection, the town with retrieval ablations over 5 seeds, the paper's numbers (~1 second) |
| `test_agents.py` | 4 quick tests (~0.1 seconds) |

**Run it** (from `09-Reasoning-and-Agents/064-Park-et-al-2023-Generative-Agents`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~1 second
python3 experiments.py --quick
```

---

## 2. `agents.py`

### Stand-ins for the LLM
| Name | Stands in for |
|---|---|
| `rate_importance(text)` | the 1–10 poignancy prompt (keyword scores, clipped to 1–10) |
| `embed(text)` | an embedding model (a hashed bag of crudely stemmed words, unit length) |
| `salient_questions(records, n, exclude)` | "3 most salient high-level questions" (the most frequent content words) |
| the insight text in `reflect` | "5 high-level insights (because of …)" |

### The memory stream
- **`Memory`:** text, created / last-access time, kind (observation, reflection, plan), importance, embedding, evidence (pointers).
- **`MemoryStream(weights)`:**
  - **`add`:** appends a memory; observations add to `importance_since_reflection`.
  - **`scores(query, t)`:** min-max-scaled recency (0.995^(t − last access)), importance and relevance (cosine), plus the weighted total.
  - **`retrieve(query, t, k)`:** the top k by total score; refreshes `last_access`.
  - **`should_reflect()`:** whether the summed importance exceeds 150.
- **`reflect(name, stream, t)`:**
  1. generate questions from the 100 most recent records;
  2. retrieve 5 pieces of evidence for each;
  3. add a reflection citing them (as record numbers and pointers);
  4. reset the counter.
- **`reflection_depth(m)`:** 0 for observations; 1 + the deepest evidence for reflections.

### Planning and the town
- **`plan_day(agent, t)`:** the agent's routine (hourly blocks), plus any event whose memory is among the top `plan_k` retrieved for "what should X do today", inserted at its time. The plan itself is stored as a memory.
- **`Town(n_agents, weights, plan_k, reflect, chat_prob, chat_k)`:** 8 places, each agent with a routine and starting acquaintances (`known`).
  - **`step_hour`:** plan at midnight; move everyone; every agent **perceives** itself and up to 4 others at its place; co-located agents may **converse**; agents reflect when triggered.
  - **`converse(a, b, place)`:** both become acquainted. Each speaker retrieves its top `chat_k` memories for "chatting with B at the place" and passes on the first **news** item (not a routine perception). Both store "was talking with".
  - **`knows(keyword)`, `density()`:** measurements.
- **`run_town(seed, hours, **kw)`:** seeds the candidacy (agent0) and the party (agent5, day 2, 17:00–19:00 at the café), runs the simulation, and counts who knew, who came (changed their plan to be at the café at 17:00 when their routine had them elsewhere), density and number of reflections.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | `install_llm` swaps the stand-ins for a real LLM (the paper's importance prompt, salient questions); a 6-agent town for one day; spread and sample reflections |
| `e2` | interview questions answered with full memory, observations only, or no memory; pairwise LLM judging; win rates per condition |
| `e3` | sweeps of the recency decay, conversation retrieval size and reflection threshold; 10 seeds each |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_importance_and_embedding_standins` | importance ordering; unit embeddings; related texts are closer |
| `test_retrieval_score_is_sum_of_minmax_scaled_terms` | exact recency and importance scaling; total = sum; the best memory is retrieved and refreshed |
| `test_reflection_trigger_and_tree` | 30 × 6 > 150 triggers; 3 questions; cited reflections; counter reset; depths |
| `test_town_runs_and_measures` | a 30-hour run produces sensible counts and a density that only grows |

---

## 5. Try it yourself

1. Set `DECAY = 0.9`. How fast is news forgotten, and how far does it spread?
2. Make conversations share the top **reflection** instead of the top observation. Do reflections now matter?
3. Let `plan_day` also read reflections (e.g. "agent3 keeps coming back to the party"). Does attendance rise?
4. Give importance a weight of 3. Does the town become obsessed with the same two stories?
5. Use `install_llm` from `experiments.py` with any small instruction model and compare its importance ratings with the stand-in's.
