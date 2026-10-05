# The code, explained simply

How the code in this folder turns Chip Huyen's post on data distribution shifts and monitoring into experiments.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `drift.py` | weighted logistic regression, a disease-prediction world with the three shift types, monitors, importance weighting, label-shift prior estimation, an hourly stream with window alerts, retraining strategies after concept drift, and a feedback-loop recommender |
| `experiments.py` | the same studies over seeds and settings (E1–E5) |
| `demo.py` | everything above with its numbers (~1 second) |
| `test_drift.py` | 4 quick tests (~1 second) |

**Run it** (from `14-Production-ML/092-Huyen-Data-Distribution-Shifts-and-Monitoring`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `drift.py`

| Name | What it does |
|---|---|
| `fit_logreg(X, y, l2, iters, lr, weights, w0)` | weighted L2 logistic regression; `w0` warm-starts it (fine-tuning) |
| `cancer_world(n, kind)` | features (age scaled, biomarker), label from σ(0.08(age − 60) + 1.2 · marker). `covariate`: ages N(65, 8). `label`: resampled to 40% of the positive share. `concept`: same inputs, label from σ(0.15(age − 55)) |
| `monitor_signals`, `shift_table` | KS on inputs (Bonferroni over 2 features), KS on predicted probabilities, accuracy, mean prediction vs true rate, for each target kind |
| `importance_weighting` | x_s ~ N(0, 1), x_t ~ N(1.5, 0.7), quadratic truth, linear model. Weights are the domain classifier's odds (it uses x and x²) vs the true normal-density ratio; also reports the effective sample size |
| `label_shift_prior` | confusion matrix on source validation, predicted-class shares on target, q = C⁻¹μ, then Bayes-rule correction of the probabilities |
| `stream` | 14 days of hourly conversion (200 users per hour, daily sine cycle) with an outage |
| `window_alerts(series, window, z, duration, seasonal)` | window mean vs the same hours a week earlier (noise √2 · SE) or vs the reference week's mean (noise SE); alert after `duration` consecutive hours with \|z\| > threshold |
| `cumulative_vs_sliding` | running mean vs 6-hour sliding mean |
| `drifting_days`, `retraining_strategies` | 30 days, with feature 1's effect flipping sign on day 20; stale / scratch on all / fine-tune 60 steps on the last 2 days / scratch from the drift point |
| `feedback_recommender(explore)` | 200 items, Beta(2, 20) true CTRs, a launch-popularity prior, greedy picks with small per-user noise, optional random exposure; reports diversity, top-item share, CTR, and the rank of the true best item |

`REPORTED` holds the post's claims (the 60 of 96 outages, shift definitions, tests, monitoring and alerting advice, TikTok's randomised exposure), checked against the web page.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | the shift table over 20 seeds: alarm rates and accuracy drops per shift type |
| `e2` | importance-weighting gain over 20 seeds |
| `e3` | window × z × duration over 20 random outages: false alarms per week, detection rate, median delay |
| `e4` | retraining strategies on several evaluation days |
| `e5` | exploration rate 0–20% over 200 rounds and 10 seeds |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_shift_types_and_monitors` | no false input alarm without shift; covariate shift is detected but stays calibrated; label shift miscalibrates; concept drift is invisible to inputs but drops accuracy |
| `test_adaptation_without_labels` | estimated weights lower target log-loss; the confusion-matrix prior beats the naive estimate; the prior correction lowers log-loss |
| `test_windows_and_cumulative` | the seasonal 6-hour window catches the outage with fewer false alarms than the flat one; the sliding mean moves ≥ 10× more than the cumulative mean |
| `test_retraining_and_feedback` | fine-tune < scratch < stale on log-loss; exploration raises diversity ≥ 10× |

---

## 5. Try it yourself

1. Add a **feature change**: a new feature appears on day 15 (missing before). How should training handle the earlier days?
2. Make concept drift gradual (the coefficient changes a little each day) and compare retraining every day vs every week.
3. Combine label shift and covariate shift. Which correction should you apply first?
4. Add a delayed-label monitor: labels arrive 3 days late. How much later is concept drift detected than with instant labels?
5. Implement the post's popularity-bucket hit rate: split items into popularity quintiles and measure how often each bucket is recommended.
