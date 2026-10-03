# The code, explained simply

How the code in this folder implements the methods described in the GPT-4 Technical Report (OpenAI 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

The report gives no architecture or training details, so the code implements its **methods**:
- prediction;
- calibration;
- contamination checking;
- rubric-based rewards.

It does not implement GPT-4.

---

## 1. The files

| File | What it is |
|---|---|
| `gpt4.py` | L(C) = aC^b + c fitting and prediction; the capability metric (−mean log pass rate, the eligibility filter, difficulty buckets, the α·C^−k fit); ECE and reliability tables; the Appendix C substring contamination check; a keyword stand-in RBRM with a reward table; a Hindsight Neglect item generator; the report's numbers |
| `experiments.py` | a real model family with a pre-registered loss prediction, capability prediction on addition, inverse-scaling probe, calibration before/after reward fine-tuning, contamination detection rates (heavy, not run here) |
| `demo.py` | real tiny-model loss prediction, a simulated capability prediction, a Hindsight Neglect example, ECE, the contamination check, the RBRM, reported numbers (~5 seconds) |
| `test_gpt4.py` | 7 quick tests (~0.1 seconds) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/057-OpenAI-2023-GPT-4-Technical-Report`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~5 seconds
python3 experiments.py --quick
```

---

## 2. `gpt4.py`

### Loss and capability prediction
| Function | What it does |
|---|---|
| `fit_power_law_offset(C, L)` | grid over c in [0, min L); for each c, a least-squares line of log(L − c) on log C; keeps the best error in L-space; returns (a, b, c) |
| `predict(params, C)` | a·C^b + c |
| `mean_log_pass_rate(p)` | −mean(log p); asserts every p > 0 |
| `eligible_problems(counts)` | the problems every model solves at least once |
| `difficulty_buckets(pass_rates, 6, 15)` | drops the 15 hardest, splits the rest into 6 buckets, easiest first |
| `fit_capability(C, metric)` | a log-log line, giving (α, k) |

### Calibration
- **`expected_calibration_error(conf, correct, n_bins)`:** Σ (n_bin/n)·|accuracy − confidence|.
- **`reliability_table`:** per-bin (low, high, count, mean confidence, accuracy).

### Contamination (Appendix C)
- **`strip`:** lower-case, then keep only letters and digits.
- **`sample_substrings`:** k random substrings of the given length, or the whole text if it is short.
- **`is_contaminated(eval_text, training_texts, rng, k=3, length=50)`:** true if any sampled substring occurs in any stripped training text.

### Rule-based reward model
- **`RUBRIC`:** the report's four classes:
  - A: a refusal in the desired style;
  - B: a refusal in an undesired style;
  - C: disallowed content;
  - D: a safe non-refusal.
- **`REWARD[(harmful?, class)]`:**
  - on harmful prompts: A +1, B +0.3, C −1, D 0;
  - on safe prompts: D +1, and refusals are penalised.
- **`toy_rbrm_classify`:** a **keyword** stand-in for the zero-shot GPT-4 classifier.
- **`rbrm_reward(harmful, response)`:** looks up the reward for the classified response.

### Other helpers
- **`hindsight_neglect_item(rng)`:** a bet with a known expected value whose outcome contradicts it. The label is Y if EV > 0, else N.
- **Reported numbers:** `TABLE_1` (exams), `TABLE_2` (benchmarks), `CALIBRATION_ECE`, `SAFETY`.

---

## 3. `experiments.py`

**Models:** paper 056's `LLaMA` at byte level. `family(a)` gives (width, depth, steps) with roughly constant tokens per parameter, growing about 4× in compute per member.

| Function | Reproduces |
|---|---|
| `e1` | trains the family; **before** the largest run, fits L(C) on the others and writes `e1_preregistered.json`; then trains the largest and reports the relative error |
| `e2` | an addition corpus; per problem, n samples at T = 1 give pass counts; the eligibility filter; −mean log pass rate fitted on the smaller models and predicted for the largest, overall and per digit-count bucket |
| `e3` | Hindsight Neglect items scored few-shot (Y vs N after "Answer: ") with the E1 checkpoints; accuracy vs size |
| `e4` | a 4-choice addition task: ECE and accuracy of the base model, then after REINFORCE on correctness |
| `e5` | plants verbatim / re-formatted / lightly edited eval sentences in the training text; detection rates for lengths 20/50/100 and k = 1/3/10, plus false positives on absent sentences |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_power_law_with_offset_recovers_and_extrapolates` | b and c recovered; 1.70 predicted at 10⁴× extrapolation |
| `test_capability_metric_selection_and_fit` | eligibility, metric value, α and k, bucket sizes and order |
| `test_ece` | 0 for perfect calibration, 0.4 for overconfidence; calibrated vs overconfident simulations |
| `test_contamination_substring_check` | catches a re-formatted copy; ignores unrelated text; misses a paraphrase |
| `test_rbrm_rewards` | classes A/B/C; harmful prompts reward refusal, safe prompts reward help |
| `test_hindsight_neglect_labels_follow_expected_value` | the label follows EV; the outcome always misleads |
| `test_reported_numbers` | MMLU 86.4; beats SOTA on all but DROP |

---

## 5. Try it yourself

1. In the demo's loss-prediction section, add one more, larger model. Does the prediction made from the original five still hold?
2. Fit the demo's runs **without** the c term (a pure power law). How far off is the prediction now?
3. In the capability simulation, make one problem's exponent different (k = 0.1). How does the per-bucket prediction change?
4. Change `length` in `is_contaminated` from 50 to 20. When does the paraphrase start being (wrongly or rightly) flagged?
5. Write a harder RBRM case: a response that refuses **and** then gives the disallowed content anyway. What should the reward be, and does `toy_rbrm_classify` get it right?
