# The code, explained simply

How the code in this folder implements the ML Test Score rubric (Breck et al. 2017) and an automated version of its tests.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `mltestscore.py` | the 28-test rubric and its scoring rule, a toy production ML system, automated implementations of 26 tests, `run_suite` with 11 injectable bugs, and `score_suite` |
| `experiments.py` | detection rate vs bug size, false alarms, threshold trade-offs, staleness at several drift speeds (E1–E4) |
| `demo.py` | the rubric, a scoring example, the full suite on the clean system, the bug-injection matrix, and close-ups of five tests (~5 seconds) |
| `test_mltestscore.py` | 5 quick tests (~0.5 seconds) |

**Run it** (from `14-Production-ML/089-Breck-et-al-2017-ML-Test-Score`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `mltestscore.py`

### Rubric and score
| Name | What it does |
|---|---|
| `RUBRIC` | the 28 test names from Tables I–IV, 7 per section |
| `INTERPRETATION` | Table V as (upper bound, text) pairs |
| `ml_test_score(status)` | maps none / manual / automated to 0 / 0.5 / 1, sums per section, takes the minimum, looks up Table V |
| `score_suite(R)` | an automated test that exists counts 1 (pass or fail); the two process tests count 0.5 |

### The toy system
| Name | What it does |
|---|---|
| `raw_world(n, t, seed)` | users from US / IN / BR / NG; spend (log-normal dollars), visits, tenure, noise, age; label from a logistic model where the visits effect grows with time t and NG reacts differently to spend |
| `feature_spend_training`, `feature_spend_serving(bug)` | two code paths for the same feature; the serving one can read cents |
| `featurize(world, cols, serving, serving_bug)` | builds the feature matrix through either path |
| `Model(spec)` | standardised logistic regression with an optional per-step `monitor` hook, `version_ops` (the op version the model needs), and `explain(x)` for one example |
| `log_loss`, `auc`, `train(spec, world)` | metrics, plus training from a spec |

### Tests, by section
| Section | Functions |
|---|---|
| Data | `make_schema` / `check_schema` (range and mean-shift checks), `feature_value` (leave-one-feature-out), `FEATURE_COST` + `feature_cost_check` (latency per unit gain), `POLICY` + `meta_requirements`, `privacy_deletion_check`, `unit_test_feature_code` |
| Model | `offline_online_correlation` (degraded models; offline log-loss vs buy rate among the targeted top 30%), `tune_hyperparameters`, `staleness_curve`, `baseline_comparison`, `slice_quality` (absolute and incremental bounds), `inclusion_check` |
| Infra | `reproducible`, `spec_unit_tests` (one step, overfit 16 examples, checkpoint restore), the integration subset run inside `run_suite`, `Registry.push` (bless or veto) / `Registry.rollback`, `Model.explain`, `canary` (op versions, finite predictions, traffic ramp with a bias limit) |
| Monitoring | `dependency_notifications`, `check_schema` on serving data, `training_serving_skew`, `staleness_alert`, `numeric_monitor` / `dead_relu_fraction`, `perf_regression` (dramatic jump vs median of the last 5; slow leak vs the launch level), `calibration_by_slice` |

### `run_suite(spec, bugs)`
Builds the train / validation / live worlds, applies the injected bugs, runs every test in rubric order, and returns `{section: [(test, passed or None, detail)]}`.

| Bug | What changes |
|---|---|
| `skew` | serving reads cents |
| `schema` | upstream sends cents |
| `forbidden` | adds `age` |
| `nondeterministic` | time-based seeds |
| `stale` | serving model is 40 weeks old |
| `newop` | the model needs op v2 |
| `nan` | lr = 1e6 |
| `deletion` | deleted ids remain in training data |
| `depchange` | `spend_raw` v3 → v4 |
| `slowleak` | latency × 1 → 1.6 |
| `badmodel` | noise added to the candidate model's weights |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | detection rates of the schema, skew, served-bias and validation tests as the unit factor (1.01 … 100) or weight noise (0.02 … 1.0) grows |
| `e2` | false-alarm rate of each test on 50 healthy seeds |
| `e3` | the false-alarm vs detection trade-off for the schema mean-shift limit and the bias limit |
| `e4` | the stale − fresh log-loss gap by age for 3 drift speeds, and the tolerable age |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_scoring_rule_uses_minimum_and_table_v` | section sums, the minimum, Table V lookups, 28 tests |
| `test_clean_system_passes_infra_and_monitoring` | the clean system passes all Infra and Monitoring tests; the score is 6.5 |
| `test_skew_vs_upstream_change_are_told_apart` | serving-code bugs show as skew; upstream changes show in the schema test but not as skew |
| `test_registry_canary_and_unit_tests` | a degraded model is vetoed and the serving version stays; the canary refuses an op-v2 model; the spec unit tests pass |
| `test_monitors` | the NaN monitor fires on divergence; dramatic and slow-leak alarms; dead-ReLU fraction |

---

## 5. Try it yourself

1. Make the scheduler real: run `run_suite` in a loop over simulated days with drifting worlds, and log which alarms fire on which day.
2. Add a second slice dimension (e.g. new vs returning users) to `slice_quality` and see how many slices you need before false alarms become a problem.
3. Write a "golden test" (compare to a saved model) and change the seed. Watch it break for no real reason, which is the paper's warning.
4. Implement a feature store: one function computes `spend` for both training and serving. Which bug in the matrix becomes impossible?
5. Score a real project you know with `ml_test_score`. Which section is your minimum?
