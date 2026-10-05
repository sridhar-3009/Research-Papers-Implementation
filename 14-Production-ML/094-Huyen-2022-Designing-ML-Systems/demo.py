"""Designing Machine Learning Systems in ~10 seconds: one small experiment per technique from chapters 4-7 and 9 of
Chip Huyen's book -- sampling, weak supervision, class imbalance, active learning, leakage, hashing, calibration,
baselines and behavioural tests, batch vs online prediction, A/B testing, interleaving, bandits, stateful training."""

import time

import numpy as np

import dmls as D

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("Ch. 4  Sampling")
f = D.reservoir_uniformity(trials=5000)
print(f"  reservoir sampling (k = 10 from a stream of 100, 5000 runs): each item's inclusion rate is "
      f"{f.min():.3f}-{f.max():.3f} (target k/n = 0.100) -- uniform without knowing n in advance")
r = D.stratified_vs_random()
print(f"  estimating a positive rate of {r['truth']:.4f} with 1,000 samples: std {r['random: std of estimate']:.4f} "
      f"(simple random) vs {r['stratified: std of estimate']:.4f} (stratified)")

section("Ch. 4  Weak supervision: 6 labelling functions instead of hand labels")
r = D.weak_supervision()
print(f"  LFs cover {r['LF coverage (any LF fires)']:.1%} of examples; majority-vote labels are "
      f"{r['majority-vote label accuracy']:.1%} correct, accuracy-weighted vote {r['weighted-vote label accuracy']:.1%}")
print(f"  LF accuracies estimated without labels {r['estimated LF accuracies']} vs true {r['true LF accuracies']}")
for k in ("classifier on majority vote", "classifier on weighted vote", "classifier on 100 hand labels",
          "classifier on all true labels"):
    print(f"    {k:32s} test accuracy {r[k]:.3f}")
print("  -> ~3,000 noisy programmatic labels beat 100 hand labels (0.857 vs 0.747) and nearly match full supervision;")
print("     honest note: our agreement-based accuracy estimates are biased upward, so the weighted vote did not beat")
print("     plain majority vote")

section("Ch. 4  Class imbalance (about 1% positives)")
r = D.class_imbalance()
print(f"  'always negative' is {r['accuracy of ' + chr(39) + 'always negative' + chr(39)]:.1%} accurate and useless")
print(f"  a real model: ROC-AUC {r['model ROC-AUC']:.3f} looks strong, PR-AUC {r['model PR-AUC']:.3f} "
      f"(a random scorer gets {r['random-scorer PR-AUC (= positive rate)']:.3f}) shows how hard the positives are")
print(f"  at threshold 0.5: recall {r['recall @0.5 unweighted']:.2f} unweighted vs {r['recall @0.5 class-weighted']:.2f} "
      f"with class weights (precision {r['precision @0.5 class-weighted']:.3f}); but class weights inflate the mean "
      f"prediction from {r['mean prediction unweighted']:.3f} to {r['mean prediction class-weighted']:.3f}")
print("  -> report PR-AUC / recall for rare classes; reweighting moves the threshold for you but breaks calibration")

section("Ch. 4  Active learning: which examples to label next")
r = D.active_learning()
print("    labels        " + "  ".join(f"{n:5d}" for n in r["labels"]))
for k in ("random", "uncertainty"):
    print(f"    {k:12s}  " + "  ".join(f"{v:.3f}" for v in r[k]))
print("  -> labelling the examples the model is least sure about pulls slightly ahead from 60 labels on (0.946 at 100")
print("     labels vs 0.940 at random); the gain is small on this easy 2-D problem and random is even ahead at 40")

section("Ch. 5  Data leakage")
r = D.leakage_feature_selection()
print(f"  labels are pure noise (true accuracy 50%); choose 20 of 5,000 features by correlation with the label:")
print(f"    selected using ALL data, then split: test accuracy {r['selected on ALL data (leak)']:.2f}   <- leak")
print(f"    selected using training data only:   test accuracy {r['selected on training data']:.2f}")
r = D.leakage_duplicates()
print(f"  30% near-duplicate rows, 1-nearest-neighbour classifier: random split {r['random split (copies leak)']:.3f} vs "
      f"split by item {r['split by item (no leak)']:.3f}")

section("Ch. 5  The hashing trick: 10,000 categories into B buckets")
for B, v in D.hashing_collisions().items():
    print(f"    B = {B:7d}: share of values sharing a bucket {v['measured']:.3f} (formula 1-(1-1/B)^(n-1) = {v['formula']:.3f})")

section("Ch. 6  Calibration: naive Bayes on 6 correlated copies of one signal")
r = D.calibration_demo()
print(f"  ECE {r['ECE raw']:.3f} -> {r['ECE after Platt']:.3f} after Platt scaling; log-loss {r['log-loss raw']:.3f} -> "
      f"{r['log-loss after Platt']:.3f}; AUC {r['AUC raw']:.3f} -> {r['AUC after Platt']:.3f} (ranking unchanged)")
print(f"  Platt slope {r['Platt slope']:.2f}: the raw log-odds were ~{1 / r['Platt slope']:.1f}x too extreme -- naive Bayes "
      f"counted the same evidence 6 times")
print("    bin          raw: predicted / observed     after Platt: predicted / observed")
raw = {lo: (p, o) for lo, hi, n, p, o in r["reliability raw"]}
cal = {lo: (p, o) for lo, hi, n, p, o in r["reliability after Platt"]}
for lo in sorted(set(raw) | set(cal)):
    a = f"{raw[lo][0]:.2f} / {raw[lo][1]:.2f}" if lo in raw else "   --    "
    b = f"{cal[lo][0]:.2f} / {cal[lo][1]:.2f}" if lo in cal else "   --    "
    print(f"    [{lo:.1f}, {lo + 0.1:.1f})   {a}                   {b}")

section("Ch. 6  Baselines and behavioural tests (loan model trained on historically biased decisions)")
r = D.baselines_and_behaviour()
for k in ("baseline: random", "baseline: majority class", "baseline: income > 0", "model with protected attribute",
          "model without it"):
    print(f"    {k:32s} accuracy {r[k]:.3f}")
print(f"  invariance test (flip only the protected attribute): {r['invariance: decisions that flip']:.1%} of decisions "
      f"flip -> FAIL, the model learned the historical bias")
print(f"  directional test (+0.5 income must not lower approval): violations "
      f"{r['directional (model with protected attribute): share where +income lowers approval']:.1%} -> pass")

section("Ch. 7  Batch vs online prediction (intent drifts during the day)")
r = D.batch_vs_online()
print(f"  nightly batch scores: accuracy {r['batch accuracy']:.3f}, {r['batch predictions computed']} predictions computed")
print(f"  online at request:    accuracy {r['online accuracy']:.3f}, {r['online predictions computed']} predictions "
      f"computed")
print("  -> fresh features are worth 6 points here; online costs one prediction per request (here more than batch,")
print("     because active users return several times) and needs a low-latency path")

section("Ch. 9  Testing in production")
print(f"  A/B sample size to detect a CTR lift 10% -> 11% (alpha 0.05, power 0.8): {D.ab_sample_size():,.0f} users per arm")
for u in (200, 500, 1000):
    r = D.ab_vs_interleaving(users=u, trials=30)
    print(f"  {u:5d} users: A/B detects the better ranker in {r['A/B: share of trials detecting B']:.0%} of trials, "
          f"interleaving in {r['interleaving: share detecting B']:.0%}")
print("  -> interleaving (each user sees both rankers mixed) is more sensitive at small samples here; the gap closes by")
print("     1,000 users in this toy")
r = D.bandit_vs_ab(reps=10)
print(f"  arms with CTR 10% / 11% / 13%, 20,000 users: A/B/n-then-ship loses {r['A/B/n then ship: regret (clicks lost)']:.0f}"
      f" clicks, Thompson sampling {r['Thompson sampling: regret']:.0f}; Thompson ends on the best arm in "
      f"{r['Thompson picks the best arm at the end']:.0%} of runs")
r = D.stateless_vs_stateful()
print(f"  daily updates on a drifting world: stateless (14-day window from scratch) log-loss "
      f"{r['stateless (14-day window from scratch) log-loss']:.4f} with {r['stateless example-passes']:,} example-passes;")
print(f"  stateful (fine-tune on today only) {r['stateful (fine-tune on today) log-loss']:.4f} with "
      f"{r['stateful example-passes']:,} -> {r['stateless example-passes'] / r['stateful example-passes']:.0f}x less compute")

section("About the source")
for k, v in D.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
