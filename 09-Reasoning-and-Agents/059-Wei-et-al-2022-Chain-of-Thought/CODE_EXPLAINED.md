# The code, explained simply

How the code in this folder implements chain-of-thought prompting (Wei et al. 2022) and its miniature test.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `cot.py` | standard and chain-of-thought prompt builders, the Figure 1 exemplar and the 8 GSM8K exemplars of Table 20, answer extraction, the toy multi-step task in 4 formats (standard, chain of thought, dots, after the answer), chain checking, toy training and evaluation (with paper 056's LLaMA), the paper's numbers |
| `experiments.py` | GSM8K with open models (standard vs CoT, ablations, robustness) and toy emergence / difficulty sweeps (heavy, not run here) |
| `demo.py` | the two prompt types, 4 toy models (one per format), accuracy by number of steps, the paper's numbers (~24 seconds) |
| `test_cot.py` | 6 quick tests (~1 second) |

**Run it** (from `09-Reasoning-and-Agents/059-Wei-et-al-2022-Chain-of-Thought`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~24 seconds
python3 experiments.py --quick
```

---

## 2. `cot.py`

### Prompting
| Name | What it is |
|---|---|
| `EXEMPLARS` | Figure 1's tennis-ball exemplar and Table 20's first exemplar |
| `GSM8K_EXEMPLARS` | the 8 exemplars of Table 20 (annotator A), as {question, chain, answer} |
| `standard_prompt(exemplars, q)` | "Q: … A: The answer is X." for each exemplar, then "Q: q A:" |
| `cot_prompt(exemplars, q)` | the same, but each answer is "<chain> The answer is X." |
| `extract_answer(text)` | the number after the last "answer is" (or the last number); commas and a trailing ".0" removed |

### The toy task
- **`toy_item(rng, fmt, k)`:** k random digits; the target is their sum mod 10. Returns (question "Q…=", answer string, answer, running sums). The four formats:

| Format | Answer string for Q3729= |
|---|---|
| standard | `1;` |
| chain of thought | `302>1;` (running sums before ">") |
| variable compute (dots) | `...>1;` (same length, no content) |
| reasoning after answer | `1>302;` |

- **`toy_answer(fmt, completion)`:** reads the predicted digit (after ">" for the CoT and dots formats).
- **`chain_is_correct(completion, run)`:** true if every running sum before ">" is right.

### Training and evaluation
- **`train_toy(fmt, steps, k_range, d, layers, …)`:**
  - builds a tiny LLaMA (paper 056) over the 16-symbol vocabulary;
  - trains it on a single format;
  - the loss covers only the answer part (the question is masked).
- **`evaluate_toy(model, fmt, k, n)`:**
  - greedy decoding until ";";
  - returns the answer accuracy, the fraction of correct answers whose chain is also correct, and two samples.

### The paper's numbers
`TABLE_2_GSM8K`, `PRIOR_BEST_GSM8K`, `ERROR_ANALYSIS`.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | GSM8K test with Hugging Face open base models (Qwen2.5 0.5B → 7B by default): standard vs CoT with the 8 Table 20 exemplars, greedy |
| `e2` | equation-only, dots-as-long-as-the-equation, and answer-then-chain exemplars (`ablation_exemplars`, `after_answer_prompt`) |
| `e3` | 3 shuffled exemplar orders, plus a terser chain style |
| `e4` | toy prompting setting: models of several sizes pre-trained on single-format documents; 2 in-context exemplars choose the format; standard vs CoT accuracy per size |
| `e5` | toy gain of chain of thought vs number of steps k = 1 … 6 |

`generate` stops each completion at the next "\nQ:".

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_prompts_differ_only_by_the_chain` | same questions and answers; only the chain text differs |
| `test_answer_extraction` | "The answer is 11", commas, "$", ".0", no number |
| `test_toy_formats_have_the_promised_shapes` | each format's structure; dots have the same length as the chain; the answer is the true sum mod 10 |
| `test_toy_training_runs_and_learns_something` | a short training run lowers the loss; evaluation returns samples |
| `test_paper_numbers` | PaLM 17.9 → 56.9 (> 3×); every model gains |
| `test_gsm8k_exemplars_are_consistent` | 8 exemplars; each chain contains its answer |

---

## 5. Try it yourself

1. Train the standard format for 3000 steps instead of 600. Does it eventually solve k = 4? (Direct answers are learnable, just much slower.)
2. Train chain of thought on k = 2 … 8 and test k = 10. Does training on longer chains fix length generalisation?
3. Write a fifth format, "equation only": `3+7+2+9>1;`. Where does it land between standard and chain of thought?
4. Make the steps *wrong but consistent* (e.g. running sums mod 9) while the answer stays mod 10. What happens to accuracy?
5. Shrink the model to d = 16, 1 layer. Does chain of thought still help? (A small-scale analogue of "emergence".)
