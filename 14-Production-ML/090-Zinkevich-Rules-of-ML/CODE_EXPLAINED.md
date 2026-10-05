# The code, explained simply

How the code in this folder checks the measurable rules from Zinkevich's *Rules of Machine Learning*.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `rules.py` | all 43 rule titles (`RULES`), a weighted logistic-regression learner, and one experiment function per measurable rule (1, 10, 21, 24, 30, 33, 34, 36, 37, 40) |
| `experiments.py` | the same checks over seeds and settings (E1–E4) |
| `demo.py` | the rule list and every experiment (~9 seconds) |
| `test_rules.py` | 5 quick tests (~1.5 seconds) |

**Run it** (from `14-Production-ML/090-Zinkevich-Rules-of-ML`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `rules.py`

### Shared helpers
| Name | What it does |
|---|---|
| `fit_logreg(X, y, l2, iters, lr, weights)` | full-batch gradient descent on **weighted** log-loss (the weights are needed for Rule 30) |
| `predict`, `log_loss`, `auc`, `spearman`, `one_hot` | metrics and encoding |

### One function per rule
| Function | Toy and measurement |
|---|---|
| `rule1_heuristic_vs_ml` | 40 apps with quality and category; users prefer their category. Install rate of the top pick for random, "most installed", a model with app one-hot + category-match feature, and the oracle |
| `rule10_coverage_monitor` | daily data; feature 0 is populated 90% → 60% from day 18 (missing = 0). Coverage and log-loss per day; alert when coverage moves > 5 points from its 7-day median |
| `rule21_features_vs_data` | 3 categorical columns × 10 values; the label has main and pairwise effects. 30 one-hot vs 30 + 300 crossed features at several data sizes, scored on a 20k test set |
| `rule24_delta(a, b, k)` | position-weighted symmetric difference of the top-k sets, normalised to [0, 1] |
| `rule30_importance_weighting` | negatives kept with probability `keep`: all data vs dropped vs weighted 1/keep; mean prediction, log-loss, AUC |
| `daily_world`, `rule33_temporal_split` | per-day effects plus day one-hot features; the random-split estimate vs the later-days estimate, each compared with actual performance on future days |
| `rule34_filter_holdout` | spam depends on content and sender reputation; the old filter (reputation) blocks 75% of spam. The content model is trained on shown traffic, on a 1% unfiltered holdout, or on all traffic |
| `rule36_position` | click = σ(relevance + slot effect); the old ranker places popular docs high. Doc-only vs doc + slot one-hot; Spearman correlation of doc weights with true relevance |
| `rule37_skew(log_at_serving)` | a merchant rating joined from a table that is refreshed before serving. Log-loss at train / holdout / next day / live, with and without training on logged serving features |
| `rule40_ensemble` | two calibrated base models (on different features) stacked by logistic regression on their logits; checks non-negative weights and monotonicity when one base score is raised |

`REPORTED` holds the guide's numbers (50%, 2%, 90% → 60%, 10/3, 1% and 74%, Rule 21's sizes, YouTube, the three skew components), checked against the web page.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | heuristic share of the ML gain over 20 seeds |
| `e2` | Rule 21 at 7 data sizes from 100 to 100k |
| `e3` | calibration error vs sampling rate (with and without weights); held-out share 0.1% / 1% / 5% |
| `e4` | optimism of the random vs temporal estimates and the skew decomposition over 10 seeds |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_all_43_rules_listed` | rules 1 to 43, in order |
| `test_rule30_weighting_restores_calibration` | weighting matches the true rate within 0.01; unweighted is > 1.5× too high |
| `test_rule24_delta_bounds` | 0 for the same lists, 1 for disjoint, in between when one item leaves; reordering inside the top-k is 0 |
| `test_rule34_and_rule36` | the shown-traffic model under-predicts spam; the holdout model is calibrated; the position feature improves relevance recovery |
| `test_rule37_jump_is_at_live_and_logging_removes_it` | the next-day → live jump is > 0.05 and vanishes with logging; ensemble monotonicity |

---

## 5. Try it yourself

1. Add a "positional feature crossed with document" (which Rule 36 says to avoid) and check whether relevance recovery gets worse.
2. Make Rule 10's monitor also track the feature's mean, and see whether that alerts as fast as coverage.
3. In `rule30_importance_weighting`, also down-sample by a **feature** (e.g. keep only 30% of mobile traffic). Does weighting still fix calibration on each slice?
4. Implement Rule 28: a model with only doc_id × query features, plus a newly added app. Does it ever recommend the new app?
5. Build Rule 15's policy layer: a quality ranker plus a separate spam filter applied after ranking. Compare with a single model that learns both.
