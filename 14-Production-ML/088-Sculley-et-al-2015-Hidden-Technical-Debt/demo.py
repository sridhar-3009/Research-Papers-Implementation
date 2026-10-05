"""Hidden technical debt in ~2 seconds: the new pieces of Sculley et al. (2015) -- a direct feedback loop and its
bandit / randomisation fixes, a hidden loop between two disjoint systems, a configuration system that follows the
paper's principles, typed values vs plain floats, sliced prediction-bias monitoring with data tests and action limits,
reproducibility debt, and how much of a tiny end-to-end pipeline is actually ML code."""

import time

import numpy as np

import hidden_debt as H

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Direct feedback loop: the model chooses its own training data (20 seeds)")
print("    policy                                mean CTR   best possible   rounds on best item   seeds stuck (<10%)")
for label, pol, h in (("greedy supervised", "greedy", 0.0), ("epsilon-greedy (eps = 0.1)", "eps", 0.0),
                      ("UCB bandit", "ucb", 0.0), ("greedy + 10% isolated random slice", "greedy", 0.1)):
    r = [H.direct_feedback_loop(pol, holdout=h, seed=s) for s in range(20)]
    share = [x["share of rounds on best item"] for x in r]
    print(f"    {label:36s}  {np.mean([x['ctr'] for x in r]):.4f}     {np.mean([x['best possible ctr'] for x in r]):.4f}"
          f"          {np.mean(share):.2f}                 {np.mean(np.array(share) < 0.1):.0%}")
print("  -> greedy locks onto whatever looked best in the launch log in most seeds; randomisation or a bandit finds")
print("     the best item far more often. Honest note: the CTR cost here is tiny, because several items are almost")
print("     as good as the best -- the debt is what the system can no longer LEARN, not this week's CTR. The isolated")
print("     random slice costs CTR directly (0.1036): exploration is paid for by the users who see random items")

section("2. Hidden feedback loop between two disjoint systems (product picker A, review picker B)")
print("    A shows products of mean quality   B's learned review weight   B's A/B lift (top review)   page CTR")
for q in (-1.0, 0.0, 1.0):
    r = H.two_systems(q)
    print(f"    {q:+.1f}                               {r[chr(66) + chr(39) + 's learned review weight']:.3f}"
          f"                       {r[chr(66) + chr(39) + 's A/B lift']:+.3f}                      {r['page CTR']:.3f}")
print("  -> B's code and data pipeline never changed, yet when A improves, B's model changes (weight 0.93 -> 0.55")
print("     -> 0.27: users who see better products need less persuasion from reviews) and the size of B's measured")
print("     A/B win moves too (+0.095 / +0.131 / +0.104) -- B's past experiment results are stale after A ships")

section("3. Configuration debt: configs as small diffs, visual diffs, automatic assertions")
CONFIGS = {
    "base": {"features": ["query", "user_country", "ctr_7d"], "l2": 1e-3, "learning_rate": 0.1,
             "train_from": 1010, "train_to": 1130, "memory_gb": 8},
    "exp_A": {"parent": "base", "features+": ["feature_A"], "train_from": 901},
    "exp_B": {"parent": "base", "features+": ["feature_B", "feature_D"], "train_from": 920},
    "exp_C": {"parent": "base", "features+": ["feature_Q", "feature_R", "feature_Z"], "learing_rate": 0.05},
    "exp_D": {"parent": "exp_A", "features+": ["old_product_id"], "features-": ["ctr_7d"], "train_from": 1001},
}
base = H.resolve("base", CONFIGS)
for name in CONFIGS:
    cfg = H.resolve(name, CONFIGS)
    print(f"  {name}: {len(cfg['features'])} features")
    if name != "base":
        for ln in H.diff(base, cfg):
            print(f"      diff vs base  {ln}")
    for p in H.validate(cfg) or ["OK"]:
        print(f"      check         {p}")
print(f"  transitive closure of data dependencies of exp_B: {sorted(H.closure(H.resolve('exp_B', CONFIGS)['features']))}")
print(f"  configs that break if geo_db is turned off: {H.consumers_of('geo_db', CONFIGS)}")
print(f"  configs that break if legacy_db is turned off: {H.consumers_of('legacy_db', CONFIGS)}")
print("  -> every one of the paper's configuration examples (badly logged dates, missing early data, a feature not")
print("     available in serving, mutual exclusion, memory needs) becomes an automatic check; a misspelled setting")
print("     ('learing_rate') is caught as unused instead of silently ignored")

section("4. Plain-old-data type smell")
flip, meant, actual = H.pod_bug()
print(f"  the model switches from probabilities to log-odds; a consumer still uses 'score > 0.8':")
print(f"    intended to act on {meant:.1%} of items, now acts on {actual:.1%}; {flip:.1%} of decisions silently flip")
try:
    H.LogOdds(1.2) > H.Probability(0.8)
except TypeError as e:
    print(f"  with typed values the same comparison raises: TypeError: {e}")
print(f"  explicit conversion works: LogOdds(1.2).to_probability() = {H.LogOdds(1.2).to_probability():.3f} > 0.8 -> "
      f"{H.LogOdds(1.2).to_probability() > H.Probability(0.8)}")

section("5. Monitoring: sliced prediction bias, data tests, action limits")
X, y, country, names = H.world()
w = H.fit_logreg(X, y)
schema = H.make_schema(X)
print("    up-stream producer state         bias ALL   " + "   ".join(f"{n:>6s}" for n in names) + "   data tests")
for b in (None, ("IN", "zeros"), ("IN", "units")):
    Xb, yb, cb, _ = H.world(broken=b, seed=1)
    sb = H.sliced_bias(w, Xb, yb, cb, names)
    label = "healthy" if b is None else f"{b[0]} feature 0 -> {b[1]}"
    fails = H.data_tests(schema, Xb)
    print(f"    {label:32s} {sb['ALL']:+.3f}   " + "   ".join(f"{sb[n]:+.3f}" for n in names)
          + f"   {'; '.join(fails) if fails else 'pass'}")
print("  -> when one country's producer breaks, the overall bias barely moves (-0.005) but the IN slice shows it;")
print("     input-data tests against a schema learned from training data flag the feature before the model sees it")
for thr in (0.9, 0.5):
    rate, alert = H.action_limit(H.predict(w, X), thr, limit=0.25)
    print(f"  action limit 25%: a spam marker with threshold {thr} would act on {rate:.1%} -> "
          f"{'ALERT, stop and page a human' if alert else 'ok'}")

section("6. Reproducibility debt")
seq, pair, sort, exact = H.float_order_sums()
print(f"  the same 20,000 float32 numbers summed three ways: sequential {seq:.4f}, pairwise {pair:.4f}, sorted {sort:.4f}"
      f" (float64: {exact:.4f})")
w0, w0b, w1 = H.sgd_runs()
print(f"  SGD with shuffle seed 0 twice: identical = {np.array_equal(w0, w0b)}; with seed 1 the weights differ by up to "
      f"{np.abs(w0 - w1).max():.2f}")
print("  -> parallel reductions sum in a different order, and data order changes SGD's answer: fix seeds, log data")
print("     order, and expect small differences even then")

section("7. How much of a pipeline is 'ML code'? (Figure 1)")
r = H.run_pipeline()
ml, other = H.code_fraction()
print(f"  our tiny end-to-end pipeline ran: {r}")
print(f"  lines of code: learner + training {ml}, everything else (ingest, validate, features, threshold, serve,")
print(f"  monitor) {other} -> ML is {ml / (ml + other):.0%} of it")
print("  -> even in a toy with no real infrastructure, ML is a minority; the paper's mature systems are at most 5% ML")

section("What the paper says (verified against the PDF)")
for k, v in H.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
