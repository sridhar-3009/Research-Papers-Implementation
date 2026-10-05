"""Data distribution shifts and monitoring in ~3 seconds: the three kinds of shift and which monitor sees each,
adapting without labels (importance weighting, label-shift prior estimation), window length / seasonality / alert
duration trade-offs, retraining strategies after concept drift, and a degenerate feedback loop."""

import time

import numpy as np

import drift as D

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Covariate shift, label shift, concept drift -- what each monitor sees (model trained on source)")
print("    target          inputs KS p   predictions KS p   accuracy (source -> target)   mean prediction vs true rate")
for k, v in D.shift_table().items():
    print(f"    {k:14s}  {v['inputs KS p']:9.1e}   {v['predictions KS p']:13.1e}      {v['accuracy source']:.3f} -> "
          f"{v['accuracy target']:.3f}             {v['mean prediction']:.3f} vs {v['true rate']:.3f}")
print("  -> covariate shift: inputs and predictions move, but the model stays calibrated (P(Y|X) did not change);")
print("     label shift: inputs and predictions move, accuracy even RISES, but the model now over-predicts disease")
print("     2x; concept drift: inputs and predictions are IDENTICAL -- only accuracy (which needs labels) reveals it")

section("2a. Covariate shift with a mis-specified model: importance weighting from unlabelled target data")
r = D.importance_weighting()
for k in ("unweighted", "weights from a domain classifier", "true density ratio"):
    print(f"    {k:34s} target log-loss {r[k]['target log-loss']:.3f}   accuracy {r[k]['target accuracy']:.3f}")
e = r["effective sample size (of 6000)"]
print(f"  -> the domain-classifier weights recover the true-ratio result (log-loss 0.713 -> 0.567); the price is a")
print(f"     smaller effective sample: {e['estimated weights']:.0f} of 6000 (true weights: {e['true weights']:.0f})")

section("2b. Label shift: estimate the new disease rate WITHOUT labels (confusion-matrix method)")
r = D.label_shift_prior()
print(f"    source rate {r['source positive rate']:.3f}; true target rate {r['true target positive rate']:.3f}; "
      f"naive (mean prediction) {r['naive estimate (mean prediction)']:.3f}; confusion-matrix estimate "
      f"{r['confusion-matrix estimate']:.3f}")
print(f"    after Bayes-rule prior correction: log-loss {r['log-loss before']:.3f} -> "
      f"{r['log-loss after prior correction']:.3f}, accuracy {r['accuracy before']:.3f} -> {r['accuracy after']:.3f}")

section("3. Time scale: an outage drops the hourly conversion rate by 0.12 for 6 hours on day 9")
s, rate = D.stream()
cum, slide = D.cumulative_vs_sliding(s)
h0 = 24 * 9
print(f"  3 hours into the outage: cumulative mean moved {cum[h0 + 3] - cum[h0 - 1]:+.4f}, 6-hour sliding mean "
      f"{slide[h0 + 3] - slide[h0 - 1]:+.4f}  -> cumulative statistics hide it")
print("    window   compare with          alert after   false alarms (7 days)   detection delay (hours)")
for win in (1, 3, 6, 24):
    for sea in (True, False):
        for dur in (1, 3):
            a = D.window_alerts(s, win, seasonal=sea, duration=dur)
            hit = [h for h in a if h0 <= h < h0 + 6 + win]
            fa = len(a) - len(hit)
            print(f"    {win:3d} h    {'same hours last week' if sea else 'last week mean   ':20s}  {dur} h           "
                  f"{fa:3d}                     {hit[0] - h0 if hit else 'missed'}")
print("  -> 1-hour windows are too noisy to catch a 0.12 drop at z = 4; comparing with the same hours last week")
print("     removes the daily-cycle false alarms (11 -> 0 for 6-hour windows); requiring 3 hours in a row cuts false")
print("     alarms further at the cost of delay; a 24-hour window compared with last week's same hours dilutes a")
print("     6-hour outage below the threshold and misses it")

section("4. Retraining after concept drift on day 20 (log-loss on the evaluation day)")
res = D.retraining_strategies()
print("    day    stale    scratch on all data    fine-tune on last 2 days    from the drift point")
for d, v in res.items():
    print(f"    {d:3d}    {v['stale']:.3f}    {v['scratch, all data']:.3f}                  "
          f"{v['fine-tune on last 2 days']:.3f}                       {v['from the drift point']:.3f}")
print("  -> retraining from scratch on everything dilutes the new concept with 20 days of old data; fine-tuning on")
print("     recent data or training from the drift point both adapt -- but the latter needs to KNOW when it began")

section("5. Degenerate feedback loop: a recommender trained on its own clicks")
for e in (0.0, 0.1):
    r = D.feedback_recommender(explore=e)
    print(f"    exploration {e:.0%}: distinct items shown {r['distinct items shown (last round)']:.1%}, most-shown item "
          f"gets {r['share of impressions on the most-shown item']:.0%} of impressions, CTR {r['CTR (last round)']:.3f} "
          f"(best {r['best possible CTR']:.3f}), true best item ranked #{r['rank of true best item by estimate'] + 1} "
          f"by the system")
print("  -> without exploration, recommendations collapse onto one item and the system's estimates of every other")
print("     item freeze (the true best is ranked #7); randomising 10% of traffic (as TikTok does for new videos)")
print("     keeps 22% of the catalogue in play and moves the true best to #2 -- closer, not solved, after 40 rounds")

section("What the post says (checked against the web page)")
for k, v in D.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
