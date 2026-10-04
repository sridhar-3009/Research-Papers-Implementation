# The code, explained simply

How the code in this folder implements verifiers for maths word problems (Cobbe et al. 2021).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `verifiers.py` | GSM8K helpers (final-answer parsing, calculator), coverage@N, verifier ranking with top-k voting, majority vote, a toy search task with a correctness checker, a short-trained generator, batched sampling, and a `Verifier` (generator-initialised, token-level value head, joint LM loss) |
| `experiments.py` | toy ablations (data size, token/solution level, joint, generator vs verifier, samples, top-k) and the real GSM8K pipeline (heavy, not run here) |
| `demo.py` | the GSM8K format, generator, sampling and coverage, small vs large verifier, voting, the number of samples (~25 seconds) |
| `test_verifiers.py` | 4 quick tests (~1 second) |

**Run it** (from `09-Reasoning-and-Agents/061-Cobbe-et-al-2021-Training-Verifiers-Math`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~25 seconds
python3 experiments.py --quick
```

---

## 2. `verifiers.py`

### GSM8K
| Name | What it does |
|---|---|
| `final_answer(solution)` | the number after "####" (commas removed) |
| `calculator_fix(text)` | rewrites every <<expr=x>> with the true value of expr |
| `calculator_step(partial)` | if the text ends with "<<expr=", returns the value to insert during sampling |
| `strip_annotations(text)` | removes the <<…>> annotations |

### Selection
| Name | What it does |
|---|---|
| `coverage_at_n(correct_lists, n)` | test@N: any of the first n samples correct |
| `select_by_verifier(answers, scores, top_k)` | top_k = 1 takes the best-scored answer; larger top_k lets the top_k vote |
| `majority_vote(answers)` | the most common non-empty answer |

### The toy
- **`toy_problem(rng)`:** 6 digits and a target t that two of them sum to (at most 2 valid pairs).
  - the question is "Q398259t7=", a gold solution "2+5;";
  - `toy_correct(p, sol)` checks the sum and that both digits are available.
- **`train_generator(steps)`:** a tiny LLaMA (paper 056) fine-tuned on gold solutions, with the loss on the solution only. It is trained briefly on purpose.
- **`sample_solutions(gen, questions, n, temperature)`:** n samples per question, batched over equal-length questions; T = 0 means greedy.
- **`Verifier(generator, token_level=True, joint=True)`:**
  - copies the generator's weights and adds a linear value head;
  - **`train(data, steps)`:** binary cross-entropy against the 0/1 label at every solution token (token-level) or only at the last token, plus the LM loss if `joint`;
  - **`score(q, solutions)`:** the value logit at each solution's last token.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 5: greedy vs verifier best-of-16 for 50 … 4000 training problems, 3 seeds |
| `e2` | Figure 6a/b: token- vs solution-level, joint vs verification-only |
| `e3` | Figure 6c: generator trained 300 / 600 steps × verifier initialised from the weak / strong generator |
| `e4` | Figure 7: best-of-N for N = 1 … 64, top-k voting for k = 1 … 16, majority vote |
| `e5` | real GSM8K: 2-epoch generator, 20 samples per training problem labelled by the final answer, a token-level joint verifier (LM + linear head on the last hidden state), test@1 vs majority vote vs verifier vs coverage |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_gsm8k_format_and_calculator` | "#### 72" parsing; calculator correction and insertion; stripping annotations |
| `test_selection_rules` | coverage@1/@2; best-scored pick; top-3 vote; majority vote |
| `test_toy_problem_and_checker` | gold solutions are valid; wrong sums, missing digits and bad formats are rejected |
| `test_generator_sampling_and_verifier_run` | sampling shapes; both token-level/joint and solution-level/verification-only verifiers train and score |

---

## 5. Try it yourself

1. Train the generator for 600 steps instead of 400. Greedy improves; does the verifier's advantage shrink?
2. Rank 64 samples instead of 16 (increase `N`). Do you see the paper's drop from adversarial samples?
3. Compare `Verifier(gen, token_level=False)` with the default on 2000 problems.
4. Initialise the verifier from a *fresh* model instead of the generator. How much worse is it?
5. Switch the toy to "sum of k digits mod 10" chains (paper 059's task) and watch the verifier find the length shortcut.
