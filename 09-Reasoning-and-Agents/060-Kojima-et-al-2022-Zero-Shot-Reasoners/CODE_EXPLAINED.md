# The code, explained simply

How the code in this folder implements zero-shot chain of thought (Kojima et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `zeroshot.py` | the two-stage prompts, answer triggers by format, answer cleansing, the full pipeline for any `generate` function, self-consistency, Tables 1/2/4, and a toy "pre-trained" model whose corpus uses trigger tokens (direct / reasoning / irrelevant / misleading) |
| `experiments.py` | open models on GSM8K, MultiArith, Last Letter and Coin Flip; all 16 triggers; self-consistency; toy corpus-share and size sweeps (heavy, not run here) |
| `demo.py` | the two prompts, the toy model under 4 prompts, self-consistency, the paper's numbers (~20 seconds) |
| `test_zeroshot.py` | 5 quick tests (~1.5 seconds) |

**Run it** (from `09-Reasoning-and-Agents/060-Kojima-et-al-2022-Zero-Shot-Reasoners`):
```
python3 -m pytest -q             # ~1.5 seconds
python3 demo.py                  # ~20 seconds
python3 experiments.py --quick
```

---

## 2. `zeroshot.py`

### The method
| Name | What it does |
|---|---|
| `reasoning_prompt(q, trigger)` | "Q: q\nA: Let's think step by step." |
| `answer_prompt(x1, z, fmt)` | stage-1 prompt + reasoning z + the format's answer trigger |
| `zero_shot_prompt(q, fmt)` | the baseline: an answer trigger straight away |
| `cleanse(text, fmt)` | the first number / first letter A–E / first yes-no / first word |
| `zero_shot_cot(generate, q, fmt)` | calls `generate` twice; returns (answer, z, raw stage-2 output) |
| `self_consistency(answers)` | the most common non-empty answer |
| `TABLE_1`, `TABLE_2_*`, `TABLE_4` | the paper's numbers (`TABLE_4` holds all 16 triggers) |

### The toy
- **`toy_document(rng, mix)`:** a question "Q…=" followed by one of four continuations:

| Trigger | Continuation | Role |
|---|---|---|
| `>` | the answer directly | direct answering |
| `T` | running sums, then `>` and the answer | reasoning text |
| `R` | filler dots, then the answer | irrelevant |
| `X` | random digits and a random answer | misleading |

  `mix` sets the shares; the default is 30 / 60 / 5 / 5%.
- **`train_toy(steps, mix, …)`:** a tiny LLaMA (paper 056); the loss covers the continuation only.
- **`toy_generate`:** greedy or temperature sampling until ">" or ";".
- **`toy_answer(model, q, trigger)`:**
  - no trigger: the baseline (append ">", read one digit);
  - with a trigger: stage 1 generates reasoning after the trigger, then stage 2 appends ">" and reads the answer.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 1 / Figure 3: a model ladder × {GSM8K, MultiArith, Last Letter (4 words), Coin Flip (4 flips)} × {zero-shot, zero-shot-CoT} |
| `e2` | Table 4: all 16 triggers on MultiArith, next to the paper's numbers |
| `e3` | self-consistency with 1, 5, 10 and 20 sampled paths at T = 0.7 |
| `e4` | toy: the share of reasoning text in the corpus (5–60%) vs the trigger's effect |
| `e5` | toy: model size vs the trigger's effect |

- **Datasets:** `last_letter` and `coin_flip` generate those datasets locally; `hf_math` downloads GSM8K and MultiArith.
- **Models:** `HF` wraps a Hugging Face causal LM as `generate(prompt, max_new, temperature)`.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_two_stage_prompts` | the exact prompt texts; stage 2 contains stage 1 |
| `test_cleansing_takes_the_first_fitting_piece` | 375 from "375 and 376", commas and "$", B from "B, C, and D", yes/no |
| `test_pipeline_with_a_fake_model_and_self_consistency` | two calls, the second containing the reasoning and the answer trigger; majority vote |
| `test_table_4_ordering` | "Let's think step by step." is best; every instructive trigger beats every misleading or irrelevant one |
| `test_toy_corpus_and_pipeline_shapes` | the toy target; all four continuation types appear; a short training run works |

---

## 5. Try it yourself

1. Change the toy corpus to 45% reasoning text and train for 2000 steps. How much does the trigger's effect shrink?
2. Make the misleading documents ("X") 30% of the corpus. Does the baseline get worse too?
3. Use temperature 0.5 instead of 1.0 for self-consistency. Does voting now beat greedy?
4. Plug a real model into `zero_shot_cot` (any function `generate(prompt) -> text`) and try a few triggers from `TABLE_4`.
5. Add a fifth trigger to the toy corpus that precedes *wrong* running sums but the right answer. What happens to accuracy when it is used as the prompt?
