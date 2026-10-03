# The code, explained simply

How the code in this folder implements Codex's evaluation methods (Chen et al. 2021).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `codex.py` | pass@k (stable, exact-binomial and naive), the execution harness, stop sequences, temperature/nucleus sampling, sample ranking, whitespace tokens, BLEU, the code-loss law, Table 1, the 13 Appendix C building blocks and synthetic problems, 4 mini HumanEval-style problems |
| `experiments.py` | small code LMs on the Python standard library: loss law, initialisation, whitespace BPE, Codex-S on synthetic problems, HumanEval (heavy, not run here) |
| `demo.py` | estimator bias, BLEU vs execution, temperature vs k on executed programs, mean vs sum selection, whitespace savings, the paper's numbers (~1 second) |
| `test_codex.py` | 9 quick tests (~0.5 seconds) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/054-Chen-et-al-2021-Codex`):
```
python3 -m pytest -q             # ~0.5 seconds
python3 demo.py                  # ~1 second
python3 experiments.py --quick
```

---

## 2. `codex.py`

### pass@k
| Function | What it is |
|---|---|
| `pass_at_k(n, c, k)` | Figure 3: 1 − ∏_{i=n−c+1}^{n}(1 − k/i); 1 if n − c < k |
| `pass_at_k_comb` | 1 − C(n−c,k)/C(n,k) with exact integers (for checking) |
| `pass_at_k_naive` | 1 − (1 − c/n)^k (biased low) |

### Running programs
- **`check_correctness(prompt, completion, test, entry_point, timeout)`:**
  1. joins prompt + completion + test code + `check(entry_point)`;
  2. `exec`s it in a **forked child process**;
  3. returns `passed`, `failed: <exception>`, `timed out` or `failed: crashed`.
- **This is only hang/crash protection, not a sandbox.** Don't run untrusted model output with it on a machine you care about.
- **`truncate(completion)`:** cuts at the first of `\nclass`, `\ndef`, `\n#`, `\nif`, `\nprint`.

### Sampling helpers
- **`softmax(logits, T)`:** softmax with temperature.
- **`nucleus(probs, top_p)`:** keeps the smallest top set with mass ≥ top_p, then renormalises.
- **`sample`:** one draw; greedy if T = 0.
- **`rank(samples, by)`:** picks one of [(token log-probs, passed)] by mean, sum or random.

### Tokens, BLEU and paper numbers
- **`plain_tokens`, `whitespace_tokens`:** the same word/symbol split, except that runs of spaces become one token (up to 24 spaces each) instead of one token per space. Joining either list rebuilds the code exactly.
- **`bleu(candidate, reference)`:** BLEU-4 on whitespace tokens with add-one smoothing and the brevity penalty.
- **`code_loss(N)`:** (N/5.92e7)^−0.13.
- **`TABLE_1`:** the paper's HumanEval numbers.

### Problems
- **`BUILDING_BLOCKS`:** Appendix C's 13 (description, code line) pairs.
- **`synthetic_problem(n_blocks, rng)`:**
  1. chains n blocks into a docstring and a body;
  2. executes the reference on 5 strings to write the unit test;
  3. returns `{prompt, canonical, test, entry_point, blocks}`.
- **`MINI_HUMANEVAL`:** 4 small problems in HumanEval's format, including the paper's `x_or_y`.

---

## 3. `experiments.py`

**Corpus:**
- the `.py` files of the local standard library (`--site` adds site-packages);
- filtered as in Section 3.1 (size, line lengths, alphanumeric fraction, exact duplicates).

**Model:** paper 049's `GPT2` and `ByteBPE`.

**Pieces:**
- **`train_lm`:**
  - AdamW (0.9, 0.95, 1e-8), weight decay 0.1;
  - linear warm-up, then cosine decay;
  - gradient clipping.
- **`sample_completions`:**
  - batched sampling with temperature and nucleus 0.95;
  - stop-sequence detection;
  - returns each sample's text and its token log-probs (cut to the truncated text).
- **`evaluate`:** samples n completions per problem and executes them all.
- **`codex_s`:** supervised fine-tuning on synthetic problems.
  - prompts are left-padded;
  - the loss is masked to the solution;
  - the learning rate is 1/10 of pre-training's.

| Function | Reproduces |
|---|---|
| `e1` | loss vs non-embedding parameters; fitted α and N_c (Figure 4) |
| `e2` | learning curves from a text-pretrained init vs from scratch (Section 3.2) |
| `e3` | token counts: a text BPE on code, with and without whitespace-run tokens |
| `e4` | Codex-S: pass@k vs T, selection heuristics, BLEU of passing vs failing samples, pass rate vs chain length |
| `e5` | HumanEval pass@1 (T = 0.2) and pass@10/100 (T = 0.8) for the E1 models |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_stable_estimator_equals_binomial_form` | product form = binomial form; pass@1 = c/n |
| `test_estimator_is_unbiased_and_naive_is_biased_low` | exact expectation over Binomial(n, p) |
| `test_execution_harness_pass_fail_timeout_crash` | pass, assertion, syntax error, timeout |
| `test_stop_sequences` | truncation at `\ndef` / `\nprint`, no-op otherwise |
| `test_nucleus_keeps_smallest_top_set` | (0.5, 0.3, 0.15, 0.05) at p = 0.9 keeps three |
| `test_whitespace_runs_shrink_indented_code` | lossless; < 60% of the tokens on indented code |
| `test_bleu_can_prefer_a_wrong_program` | wrong-but-similar scores above right-but-different |
| `test_building_blocks_and_synthetic_problems` | 13 blocks; generated references pass their tests |
| `test_paper_numbers` | Table 1 entries; pass@k increases with k; the loss law |

---

## 5. Try it yourself

1. Use `pass_at_k` to draw the bias of `pass_at_k_naive` as n grows from k to 10k, for p = 0.05 and k = 10. When is it within 1%?
2. Add a fifth problem to `MINI_HUMANEVAL` and write three wrong completions with high BLEU.
3. In `demo.py` section 3, give each problem 30 candidates (more random edits). Does very high T start to hurt pass@100?
4. Chain 1 to 6 building blocks and measure how often a random single-block edit of the reference still passes. Some operations commute; which ones?
5. Change `max_run` in `whitespace_tokens` to 4 or 8. How much of the saving remains?
