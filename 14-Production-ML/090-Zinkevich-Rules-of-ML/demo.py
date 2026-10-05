"""Rules of Machine Learning in ~15 seconds: the 43 rules by phase, and the ones that make a measurable claim, each
checked with a small logistic-regression experiment."""

import time

import numpy as np

import rules as R

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("The 43 rules")
for phase, rs in R.RULES.items():
    print(f"  {phase}")
    for k, v in rs.items():
        print(f"    #{k:2d} {v}")

section("Rule 1: don't be afraid to launch without ML (an app store)")
r = R.rule1_heuristic_vs_ml()
for k in ("random", "heuristic: most-installed app", "ML model", "oracle"):
    print(f"    install rate of the top recommendation, {k:30s} {r[k]:.3f}")
print(f"  -> the heuristic gets {r['heuristic share of the ML gain']:.0%} of the way from random to the ML model "
      f"(the guide's rule of thumb: about 50%)")

section("Rule 10: watch for silent failures (feature coverage 90% -> 60% on day 18)")
cov, loss, alerts = R.rule10_coverage_monitor()
for d in (0, 10, 17, 18, 19, 29):
    print(f"    day {d:2d}: coverage {cov[d]:.2f}   log-loss {loss[d]:.3f}")
print(f"  -> nothing crashed; log-loss rose from ~{np.mean(loss[:18]):.3f} to ~{np.mean(loss[18:]):.3f}; the coverage "
      f"monitor first alerted on day {alerts[0]}")

section("Rule 21: learnable feature weights grow with the amount of data")
res = R.rule21_features_vs_data()
print("    examples    30 one-hot features    + 300 crosses (330)     test log-loss")
for n, v in res.items():
    win = "small model" if v["30 features"] < v["330 features"] else "crossed model"
    print(f"    {n:8d}    {v['30 features']:.3f}                  {v['330 features']:.3f}                  -> {win}")
print("  -> with 300 examples the crosses overfit; with thousands they win: add feature weights as data grows")

section("Rule 24: measure the delta between models (position-weighted top-10 difference)")
rng = np.random.default_rng(0)
s = rng.standard_normal(200)
for name, other in (("same model", s), ("small retrain noise", s + 0.1 * rng.standard_normal(200)),
                    ("big change", s + 1.0 * rng.standard_normal(200)), ("reversed", -s)):
    print(f"    {name:20s} delta {R.rule24_delta(s, other):.3f}")
print("  -> 0 = identical top-10, 1 = disjoint; a large delta means look closely before launching")

section("Rule 30: importance-weight sampled data (negatives kept with probability 30%)")
for k, v in R.rule30_importance_weighting().items():
    print(f"    {k:32s} mean prediction {v['mean prediction']:.3f} (true {v['true rate']:.3f})   log-loss "
          f"{v['log-loss']:.3f}   AUC {v['AUC']:.3f}")
print("  -> dropping negatives doubles the predicted rate; weight 10/3 restores calibration exactly. Ranking (AUC) is")
print("     unchanged either way -- it is the probabilities that break")

section("Rule 33: test on data AFTER the training cutoff")
for k, v in R.rule33_temporal_split().items():
    print(f"    {k:46s} estimate log-loss {v['estimate log-loss']:.3f} / AUC {v['estimate AUC']:.3f}   actual on "
          f"future days {v['actual log-loss on future days']:.3f} / {v['actual AUC on future days']:.3f}")
print("  -> a random split lets the model memorise each day's effect, so its estimate is too optimistic; a")
print("     time-based validation set predicts what happens after launch far better")

section("Rule 34: filtering -- hold out 1% of traffic unfiltered for clean labels")
r = R.rule34_filter_holdout()
print(f"    filter v1 blocks {r['v1 blocks share of spam']:.1%} of spam ({r['v1 blocks share of good mail']:.1%} of good"
      f" mail); with 1% held out it still blocks {r['blocking with 1% held out']:.1%}")
for k in ("trained on shown (filtered) traffic", "trained on 1% held-out traffic", "oracle: all traffic labelled"):
    v = r[k]
    print(f"    {k:36s} {v['examples']:7d} examples   mean prediction {v['mean prediction']:.3f} (true "
          f"{v['true spam rate']:.3f})   AUC {v['AUC']:.3f}   log-loss {v['log-loss']:.3f}")
print("  -> labels from filtered traffic are biased by the old filter (most of the spam it saw was never shown), so")
print("     the new model badly under-predicts spam; 2,000 clean held-out labels nearly match the oracle")

section("Rule 36: avoid feedback loops with positional features")
r = R.rule36_position()
for k in ("popularity (old ranker)", "doc-only model", "doc + position model (served at slot 0)"):
    print(f"    rank correlation with true relevance, {k:42s} {r[k]:.3f}")
print(f"    learned position effects (slots 1-9 vs slot 0): {np.round(r['learned position effects'], 2)}")
print("  -> without a position feature the model credits documents for the slot the old ranker gave them; with it,")
print("     the position effect is absorbed separately and the document scores track relevance")

section("Rule 37: measure training/serving skew (with Rules 29, 31)")
for label, kw in (("joined table refreshed between training and serving", {}),
                  ("Rule 29: train on features logged at serving time", {"log_at_serving": True})):
    v = R.rule37_skew(**kw)
    print(f"    {label}")
    print(f"      log-loss: train {v['train']:.3f} -> holdout {v['holdout']:.3f} -> next day {v['next day']:.3f} -> "
          f"live {v['live']:.3f}")
print("  -> the first three agree and the jump is next-day -> live: an engineering skew, not overfitting or drift.")
print("     Logging serving features removes the jump (live 0.575 vs 0.631) -- even though offline log-loss looks")
print("     WORSE, because the logged (refreshed) ratings are noisier than the ones the labels depended on")

section("Rule 40: keep ensembles simple")
r = R.rule40_ensemble()
print(f"    base A AUC {r['AUC base A']:.3f}, base B AUC {r['AUC base B']:.3f}, ensemble AUC {r['AUC ensemble']:.3f}; "
      f"ensemble weights {np.round(r['ensemble weights'], 2)}")
print(f"    share of examples whose ensemble score went DOWN when base A's score went up: "
      f"{r['share of examples whose ensemble score went DOWN when base A went up']:.0%}")
print("  -> an ensemble that only takes calibrated base-model scores with non-negative weights is monotone in each")
print("     base model")

section("Numbers from the guide (verified against the web page)")
for k, v in R.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
