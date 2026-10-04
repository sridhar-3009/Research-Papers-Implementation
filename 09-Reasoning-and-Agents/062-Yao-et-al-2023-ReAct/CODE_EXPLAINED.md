# The code, explained simply

How the code in this folder implements ReAct (Yao et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `react.py` | `WikiEnv` (search / lookup / finish on local pages), `react_loop` (Thought / Action / Observation with any `generate` function), trajectory ablations, the back-off rules, a toy encyclopedia world with imitation training and an environment-in-the-loop runner, the paper's tables |
| `experiments.py` | HotpotQA with an open model and live Wikipedia; toy flaky-search, training-size and back-off studies (heavy, not run here) |
| `demo.py` | a scripted ReAct transcript, then 4 toy models under stale facts and a tool outage, plus the paper's numbers (~22 seconds) |
| `test_react.py` | 4 quick tests (~1 second) |

**Run it** (from `09-Reasoning-and-Agents/062-Yao-et-al-2023-ReAct`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~22 seconds
python3 experiments.py --quick
```

---

## 2. `react.py`

### The environment and the loop
- **`WikiEnv(pages)`:**
  - `search(entity)`: the first 5 sentences of the page, or "Could not find … Similar: [...]" (via difflib);
  - `lookup(key)`: the next sentence of the current page containing `key` (it remembers its position);
  - `step(action)`: parses `search[...]`, `lookup[...]` or `finish[...]` and returns (observation, done).
- **`react_loop(generate, question, env, exemplars, max_steps)`:**
  1. at step i, the model continues from "Thought i:" until it would write "Observation i:";
  2. its text is split into the thought and the "Action i:";
  3. the environment's reply is appended as "Observation i: …";
  4. it stops at `finish[...]` or after `max_steps`, and returns (answer or None, transcript).

### Baselines and back-off
| Name | What it does |
|---|---|
| `ablate(trajectory, keep)` | builds Standard / CoT / Act / ReAct from one trajectory by deleting parts |
| `react_then_cot(react_answer, cot_sc_answers)` | uses the CoT-SC majority if ReAct gave no answer |
| `cot_then_react(cot_sc_answers, react_fn)` | runs ReAct if the CoT-SC majority has fewer than n/2 votes |
| `majority(answers)` | (most common answer, its count) |

### The toy world
- **`ToyWorld(seed)`:** films → directors → cities.
  - `page(entity)` returns `[entity, "dir", P]` or `[entity, "born", C]` (empty during an `outage`);
  - `answer(film)`.
- **`ToyWorld.trajectory(film, method, rng)`:** the 4 trajectory formats. With an `rng`, tool-using methods get counterfactual pages half the time.
- **`train_toy(world, method, steps)`:** a tiny LLaMA (paper 056) trained by imitation. The loss skips observation tokens.
- **`run_toy(world, model, film, method)`:** greedy decoding. When the model writes "O" after a search, the **environment** appends the page.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | HotpotQA dev (question only), an open 7B instruction model, `LiveWiki` (real search / lookup / finish via the Wikipedia API); the four prompts are built from the paper's Figure 1 trajectory (`EXEMPLAR`, `render`); exact match |
| `e2` | toy flaky search (30% outages) with one harness retry; Act vs ReAct |
| `e3` | toy fine-tuning curve: training on 25 … 200 films, accuracy on changed facts, all four methods |
| `e4` | toy back-off: ReAct, CoT, ReAct → CoT and CoT-SC → ReAct under 30% stale facts and 30% outages |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_wiki_env_actions` | search hit / miss with similar titles; lookup iterates matches, then "No more results"; finish; invalid actions |
| `test_react_loop_with_a_scripted_model` | a 3-step loop: correct answer; observations come from the environment; 3 thoughts |
| `test_ablations_and_backoff` | what each ablation keeps; majority; both back-off rules; Table 1 ordering |
| `test_toy_world_trajectories_and_run` | trajectory formats, outage pages, a short training run |

---

## 5. Try it yourself

1. Train the tool users **without** counterfactual pages (call `world.trajectory(f, m)` without `rng`). How do they do on changed facts?
2. Make 20% of film pages missing from the encyclopedia, and teach a "search the director's name directly" fallback in the ReAct trajectories only. Does ReAct now beat Act?
3. In `demo.py`, let the CoT model also back off to ReAct when the world may have changed. Which rule wins in which situation?
4. Plug a real model into `react_loop` (any `generate(prompt, stop)` function) with the `WikiEnv` pages from the demo.
5. Count how often a toy agent repeats the same search twice in a row: the paper's "repetitive loop" failure.
