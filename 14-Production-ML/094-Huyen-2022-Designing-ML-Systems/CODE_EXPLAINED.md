# The code, explained simply

How the code in this folder turns techniques from *Designing Machine Learning Systems* (Huyen 2022) into experiments.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `dmls.py` | one function per technique from chapters 4, 5, 6, 7 and 9, plus shared helpers (weighted logistic regression, ROC-AUC, PR-AUC, log-loss) |
| `experiments.py` | the same techniques over seeds and settings (E1–E5) |
| `demo.py` | every experiment with its numbers (~4 seconds) |
| `test_dmls.py` | 5 quick tests (~1 second) |

**Run it** (from `14-Production-ML/094-Huyen-2022-Designing-ML-Systems`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `dmls.py`

### Chapter 4: training data
| Function | What it does |
|---|---|
| `reservoir_sample`, `reservoir_uniformity` | Algorithm R, and the inclusion frequency of each item over many runs |
| `stratified_vs_random` | 3 strata (70% / 25% / 5%, positive rates 2% / 10% / 60%); std of the estimated rate over 500 samples of 1,000 |
| `weak_supervision(gold)` | 60 features (4 informative); 6 LFs with (coverage, accuracy) from (0.3, 0.90) to (0.7, 0.62); majority vote; an accuracy-weighted vote with accuracies estimated from agreement with the majority; classifiers on each label source vs `gold` hand labels vs all labels |
| `class_imbalance` | ~1.4% positives; accuracy of "always negative", ROC-AUC, PR-AUC, recall / precision at 0.5 with and without class weights, mean prediction |
| `active_learning` | a 2-D off-centre boundary; start with 20 labels, add 20 per round at random or by smallest \|p − 0.5\| |

### Chapter 5: features
| Function | What it does |
|---|---|
| `leakage_feature_selection` | 200 rows, 5,000 noise features, random labels; top-20 correlated features chosen on all rows vs on training rows |
| `leakage_duplicates` | 30% near-duplicate rows; 1-NN accuracy with a random split vs a split by original item |
| `hashing_collisions` | uniform random buckets for 10,000 values; the measured collision share vs 1 − (1 − 1/B)^(n−1) |

### Chapter 6: evaluation
| Function | What it does |
|---|---|
| `expected_calibration_error` | 10 equal-width bins; returns the ECE and the per-bin (predicted, observed) rows |
| `naive_bayes_gaussian`, `calibration_demo` | 6 noisy copies of one signal → overconfident naive Bayes; Platt scaling fitted on a separate quarter of the data; ECE, log-loss and AUC before and after |
| `baselines_and_behaviour` | labels from historically biased decisions (−0.8 for group g); baselines; the model with and without g; invariance (flip g) and directional (+0.5 income) tests |

### Chapter 7: deployment
| Function | What it does |
|---|---|
| `batch_vs_online` | 2,000 users whose intent drifts each hour; 10% make a request each hour; nightly-snapshot scores vs fresh scores; accuracy and prediction counts |

### Chapter 9: testing in production
| Function | What it does |
|---|---|
| `ab_sample_size(p0, lift, alpha, power)` | the two-proportion z-test formula |
| `ab_vs_interleaving(users, trials)` | rankers A (noise 0.6) and B (noise 0.3) over true relevance. A/B: a t-test on clicks per user. Team-draft interleaving: a binomial test on per-user preferences (ties dropped) |
| `bandit_vs_ab` | 3 arms; A/B/n with an even split for half the traffic, then ship the leader, vs Thompson sampling with Beta posteriors; regret in clicks |
| `stateless_vs_stateful` | 30 days of drifting coefficients; daily from-scratch training on a 14-day window (300 iterations) vs fine-tuning yesterday's model on today (50 iterations); next-day log-loss and example-passes |

`REPORTED` records the source (public table of contents and summaries; no quotations) and the chapter topics.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | weak supervision vs hand-label budgets 50 to 1,000 over 10 seeds |
| `e2` | class imbalance at 10% / 1% / 0.1% positives |
| `e3` | active learning over 20 seeds and 20 rounds |
| `e4` | A/B vs interleaving for 100 to 5,000 users × 200 trials; bandit regret over 20 seeds |
| `e5` | stateless vs stateful over 60 days |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_reservoir_is_uniform_and_hashing_formula` | reservoir inclusion ≈ k/n; hashing collisions match the formula |
| `test_weak_supervision_and_imbalance` | weak labels beat 100 hand labels by > 5 points; the accuracy paradox; PR-AUC < ROC-AUC; class weights raise recall and inflate predictions |
| `test_leakage_inflates_accuracy` | leaky feature selection > 0.7 on noise; honest < 0.62; duplicates inflate 1-NN by > 8 points |
| `test_calibration_and_behaviour` | Platt cuts ECE ≥ 5× with AUC unchanged; ECE by hand (0.25); the invariance test catches the biased model; the directional test passes |
| `test_production_testing` | sample size ≈ 14,750; Thompson regret < A/B/n regret; stateful uses > 10× less compute |

---

## 5. Try it yourself

1. Replace class weights with **threshold tuning** on validation data. Does recall match, and does calibration survive?
2. Implement a proper label model: estimate LF accuracies by EM (Dawid–Skene) and compare with majority vote.
3. Add **time leakage** to chapter 5: a feature computed with information from after the label time.
4. Add edge deployment: quantise the logistic weights to int8 and measure the accuracy change (see 074–078).
5. Make the interleaving click model position-biased in a way that favours one ranker, and check whether team draft stays fair.
