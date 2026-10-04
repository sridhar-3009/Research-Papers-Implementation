# The code, explained simply

How the code in this folder turns the survey's framework (Wang et al. 2023) into runnable modules.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `agentkit.py` | profile builders; `UnifiedMemory` and `HybridMemory` (read, write, de-duplication, eviction), memory reflection; `SimLLM` (a simulated LLM with step accuracy p and judge accuracy q, counting calls); five planning strategies; closed forms; an evaluator; a key-door memory task; the survey's taxonomy and challenges |
| `experiments.py` | strategy × accuracy × length grids, judge-accuracy sweep, eviction policies, a real-LLM Game of 24 (only E4 is heavy) |
| `demo.py` | the framework, planning success vs cost, scaling with task length, memory structures (~0.1 seconds) |
| `test_agentkit.py` | 5 quick tests (~0.1 seconds) |

**Run it** (from `09-Reasoning-and-Agents/065-Wang-et-al-2023-LLM-Agents-Survey`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~0.1 seconds
python3 experiments.py --quick
```

---

## 2. `agentkit.py`

### Profile and memory
| Name | What it does |
|---|---|
| `handcrafted_profile`, `generated_profiles`, `dataset_aligned_profiles` | the survey's three profile methods |
| `UnifiedMemory(window)` | keeps only the last `window` records; `read` returns matching records or the latest ones |
| `HybridMemory(window, capacity)` | short-term window plus a long-term store |
| `reflect_memories(texts)` | repeated subjects become summary memories |

`HybridMemory` in more detail:
- **`write`** merges duplicates (refreshing their time and importance) and, on overflow, evicts the **least important, oldest** record.
- **`read`** ranks by recency (0.99^Δt) + word overlap + importance/10.

### The simulated LLM and planning strategies
- **`SimLLM(p, q)`:** `propose(correct)` is right with probability p; `judge(step, correct)` is right with probability q; `calls` counts every call.

| Strategy | Behaviour |
|---|---|
| `single_path` | one chain; stops at the first wrong step |
| `self_consistency(n_paths)` | n chains; plurality vote over final outcomes (wrong chains give scattered answers) |
| `tree_of_thoughts(breadth)` | per step, `breadth` proposals; keep the first the judge accepts |
| `with_environment_feedback(retries)` | the environment reports a failed step; retry it |
| `reflexion(trials)` | after a failed attempt, store a lesson for that step; it is solved from memory in later trials |

- **`analytic(...)`:** p^L for a single path; (1 − (1 − p)^(r+1))^L with feedback.
- **`evaluate(strategy, p, L, n)`:** (success rate, LLM calls per task).

### The memory task
- **`key_door_episode(memory, gap, rng)`:** write the key note (importance 8), write `gap` distractors (importance 1), then succeed if reading "key door" recovers the key.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | every strategy over p ∈ {0.6 … 0.95} × L ∈ {2 … 40}: success and calls |
| `e2` | tree search success vs judge accuracy q ∈ {0.5 … 1.0} |
| `e3` | importance-aware vs FIFO eviction for capacities 5 … 50 with unique distractors |
| `e4` | a real instruction LLM on Game-of-24 puzzles: single chain vs self-consistency vs propose-and-verify (a Python checker as the environment) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_profiles` | the three profile builders |
| `test_memory_structures_and_operations` | window overflow; de-duplication; eviction keeps the important record and drops the least important, oldest one; retrieval; reflection |
| `test_planning_strategies_match_closed_forms` | single path ≈ p^L; feedback ≈ the closed form; perfect and hopeless models |
| `test_reflexion_learns_across_trials` | 2 steps, 2 trials, p = 0.5: success 0.625, computed exactly |
| `test_key_door_memory_task` | unified memory fails after 20 distractions; hybrid memory always recalls |

---

## 5. Try it yourself

1. Make the environment feedback noisy: report a failure only 70% of the time. How much of its advantage survives?
2. Let `self_consistency` vote over **steps** instead of final answers. Does it beat tree search?
3. Give `tree_of_thoughts` a judge with q = 0.6. Is it still better than a single path?
4. Run the key-door task with unique distractors, capacity 10 and FIFO eviction. When is the key lost?
5. Add a "human feedback" strategy: a human fixes one wrong step per task at a cost of 10 calls. Where does it sit on the success-vs-cost table?
