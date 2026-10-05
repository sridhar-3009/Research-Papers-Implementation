# The code, explained simply

How the code in this folder implements the new parts of *Hidden Technical Debt in ML Systems* (Sculley et al. 2015).
Read [EXPLAINED.md](EXPLAINED.md) first; the 2014 debts (CACE, legacy features, cascades, thresholds) live in [087](../087-Sculley-et-al-2014-High-Interest-Credit-Card-Technical-Debt/).

---

## 1. The files

| File | What it is |
|---|---|
| `hidden_debt.py` | a feedback-loop simulator with greedy / ε-greedy / UCB / isolated-slice policies, a two-system page model, a feature registry plus config resolver / diff / validator / dependency closure, typed `Probability` and `LogOdds` values, sliced prediction bias, schema-based data tests, action limits, reproducibility probes, and a tiny end-to-end pipeline whose ML share is measured |
| `experiments.py` | the same toys swept over policies, seeds, bug sizes and array sizes (E1–E4) |
| `demo.py` | all seven parts with their numbers (under 1 second) |
| `test_hidden_debt.py` | 5 quick tests (~0.2 seconds) |

**Run it** (from `14-Production-ML/088-Sculley-et-al-2015-Hidden-Technical-Debt`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `hidden_debt.py`

### Feedback loops
| Name | What it does |
|---|---|
| `direct_feedback_loop(policy, items, rounds, users_per_round, eps, holdout, initial_views)` | true CTRs ~ U(0.02, 0.12); starts from a launch log of `initial_views` per item; each round picks one item (greedy, ε-greedy, or UCB with width √(2 p̂ ln t / n)) and logs clicks only for it. `holdout` sends a share of users to random items (the isolated slice) |
| `page_clicks(rng, n, product_q, review_q)` | buy probability σ(−2 + 1.5 q_p + 1.2 q_r (1 − 0.8 σ(3 q_p))): reviews matter less when the product is good |
| `two_systems(product_quality)` | B logs random reviews, fits its review → click model, and runs its own A/B test (top review vs random), all under A's product quality |

### Configuration
| Name | What it does |
|---|---|
| `FEATURES` | a feature registry: dependencies, serving availability, bad dates, first available date, substitutes, exclusions, memory needs, deprecation, mirroring the paper's examples A, B, D, Q/R, Z |
| `resolve(name, configs)` | follows the `parent` chain; children override; `features+` / `features-` edit the inherited list |
| `diff(a, b)` | `+ feature`, `- feature`, `~ setting: old -> new` lines |
| `closure(names)` | depth-first search over `deps`: the transitive closure |
| `validate(cfg)` | returns every violated assertion, including settings not in `KNOWN_SETTINGS` (typos) |
| `consumers_of(source, configs)` | which configs depend on a source, directly or transitively |

### Smells, monitoring, reproducibility, glue
| Name | What it does |
|---|---|
| `Probability`, `LogOdds` | float subclasses: a probability checks its range, and comparing the two types raises `TypeError`; `LogOdds.to_probability()` converts explicitly |
| `pod_bug` | share of decisions that flip when a consumer compares log-odds with a probability threshold |
| `world(broken=(country, 'zeros' / 'units'))` | traffic from 4 countries where one country's up-stream feature can break |
| `sliced_bias` | mean prediction − observed rate, overall and per slice |
| `make_schema`, `data_tests` | per-feature 0.1–99.9 percentile range, mean, std, zero rate; tests for out-of-range share, mean shift and zero-rate jump |
| `action_limit(scores, threshold, limit)` | the share that would be acted on, and whether it exceeds the limit |
| `float_order_sums`, `sgd_runs` | the same float32 numbers summed in three orders; SGD under different shuffle seeds |
| `pipeline_*`, `run_pipeline`, `code_fraction` | an end-to-end pipeline from raw text lines to a monitored decision; `code_fraction` counts non-blank, non-comment source lines of the ML functions vs the rest with `inspect` |

`REPORTED` holds the paper's claims, checked against the PDF text.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | policy × ε × holdout × launch-log size, 50 seeds: share stuck, CTR, error of the final estimates |
| `e2` | A's product quality from −2 to 2: B's learned weight, A/B lift, page CTR |
| `e3` | detection rate of overall bias, sliced bias and data tests vs the share of the slice broken and the unit scale |
| `e4` | SGD weight spread over 50 shuffle seeds; float32 summation-order error for 1e3 … 1e7 numbers |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_direct_loop_greedy_gets_stuck_more_than_ucb` | greedy is stuck in > 30 points more seeds than UCB |
| `test_hidden_loop_between_two_systems` | B's learned weight drops by > 0.3 when A improves |
| `test_config_resolve_diff_validate_and_closure` | inheritance, diffs, bad-date / serving / unknown-setting checks, a clean base, the correct closure |
| `test_typed_values_and_monitoring` | typed comparison and range errors; a broken slice shows in its bias but not in others; data tests fire on a broken world and stay quiet on a healthy one |
| `test_reproducibility_and_pipeline` | same seed gives identical weights, a different seed doesn't; the pipeline drops garbage lines; ML lines < other lines |

---

## 5. Try it yourself

1. Change the true CTRs halfway through `direct_feedback_loop`. How long does each policy take to notice?
2. Make B in `two_systems` choose reviews with its learned model, and let A retrain on clicks too: a two-way hidden loop.
3. Add a "transitively deprecated" check: warn when a feature depends on a deprecated source.
4. Give `Probability` arithmetic that keeps the type (e.g. averaging probabilities) and see where `float` leaks back in.
5. Slice prediction bias by two dimensions at once (country × device) and count how many slices you need before the false-alarm rate gets annoying.
