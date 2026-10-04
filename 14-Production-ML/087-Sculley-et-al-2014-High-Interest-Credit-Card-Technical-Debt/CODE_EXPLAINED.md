# The code, explained simply

How the code in this folder turns the warnings of Sculley et al. (2014) into measurable experiments.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `debt.py` | a tiny logistic-regression learner and one function per warning: CACE, legacy product numbers plus ablation, a weekly CTR feedback loop, a correction cascade, threshold drift, correlation break, prediction bias |
| `experiments.py` | each warning swept over seeds and strengths (E1–E4) |
| `demo.py` | all six experiments with their numbers (~5 seconds) |
| `test_debt.py` | 4 quick tests (~1 second) |

**Run it** (from `14-Production-ML/087-Sculley-et-al-2014-High-Interest-Credit-Card-Technical-Debt`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `debt.py`

### The learner
| Name | What it does |
|---|---|
| `sigmoid(z)` | 1 / (1 + e^(−z)), clipped so it never overflows |
| `fit_logreg(X, y, l2, iters, lr)` | adds a bias column and runs full-batch gradient descent on log-loss + L2 (the bias is not penalised) |
| `predict`, `log_loss`, `accuracy` | probability, mean log-loss, accuracy at a threshold |

### One function per warning
| Function | Builds | Returns |
|---|---|---|
| `cace_data`, `cace_experiment` | 4 features from 2 hidden causes (x1 is a noisy copy of x0's cause); refits after removing x0, adding a noisy copy of x0, raising L2, blanking 30% of x0 | for each change: the max shift of the other features' weights, and the mean change in predictions |
| `product_number_world(n_products, seed, sample_seed, old_populated)` | each product has a quality; 70% are "old" and get an old one-hot number as well as a new one. `seed` fixes the products, `sample_seed` draws train/test traffic; `old_populated=False` is the world after the cleanup | features, labels, which rows are old products |
| `ablation(X, y, groups, Xv, yv, **fit_kw)` | leave-one-group-out retraining | the full model's validation log-loss and the increase when each group is removed |
| `feedback_loop(weeks, users, items, improve_week, quality_boost)` | each week every user is shown the headline with the highest *estimated* affinity; at `improve_week` the estimates get less noisy. Features: that estimate and x_week (clicked last week). Last week's model is scored on this week's traffic before retraining | CTR, prediction bias and the x_week weight per week |
| `correction_cascade` | a v1 (x0), correction a′ on a's logit plus x1, x2 for labels with an extra x2 effect, a v2 (x0, x1) | log-loss of A and A′ for each combination |
| `threshold_drift` | the first threshold reaching 90% validation precision for v1; v2 retrained with positives up-sampled 3× and L2 0.02 | precisions with the fixed and re-learned thresholds |
| `correlation_break` | x_proxy = x_cause + 0.05 noise in training, independent in production | weights, accuracy before and after, causal-only accuracy |
| `prediction_bias(w, X, y)` | mean prediction − observed rate | a float (≈ 0 when calibrated) |

`REPORTED` holds the paper's claims, checked against the PDF text.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | how far the other weights move when x0 is removed, for correlations 0, 0.3, 0.7, 0.95 between x0 and x1 |
| `e2` | the legacy-feature cleanup for 50/100/300 products and old-product share 0.1–0.9, with ablation |
| `e3` | the feedback loop for improvement sizes 0.2–0.8 over 20 weeks: the bias spike and the range of the x_week weight |
| `e4` | 20 seeds of cascade damage, fixed vs re-learned threshold precision, correlated vs causal accuracy |

Results go to `results.json`; `--report-only` prints them.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_logreg_learns_and_prediction_bias_is_small` | the learner recovers the signs of the true weights and is calibrated (bias < 0.01) |
| `test_cace_removing_a_feature_moves_the_others` | removing x0 shifts another weight by > 0.3; changing L2 by > 0.1 |
| `test_legacy_feature_cleanup_hurts_only_the_model_that_used_it` | the cleanup raises the two-scheme model's loss; the new-only model's loss is unchanged |
| `test_cascade_threshold_and_correlation_failures` | improving a hurts A′; the fixed threshold loses precision while the re-learned one keeps > 85%; decoupling hurts the correlated model more than the causal one |

---

## 5. Try it yourself

1. Retrain the correction in `correction_cascade` after a v2. Does A′ end up better than with a v1?
2. Add a second-level correction a″ on top of a′ and measure how an improvement to a propagates.
3. Slice prediction bias by old vs new products in the legacy-feature world after the cleanup. Which slice raises the alarm?
4. Make the feedback loop's click probability depend on x_week as well (habit formation) and see whether the slow drift becomes clearer.
5. Add an "undeclared consumer": a second model that uses the CTR model's prediction as a feature, then improve the CTR model.
