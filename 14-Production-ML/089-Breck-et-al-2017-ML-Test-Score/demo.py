"""The ML Test Score in ~10 seconds: the 28-test rubric and its scoring rule, an automated implementation of the
tests run against a toy production ML system, the score it earns, and a bug-injection matrix -- for each kind of
production bug the paper describes, which of the tests catch it."""

import time

import numpy as np

import mltestscore as M

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The rubric: 28 tests, 7 per section")
for sec, tests in M.RUBRIC.items():
    print(f"  {sec}:")
    for i, t in enumerate(tests, 1):
        print(f"    {sec} {i}: {t}")

section("2. Scoring: 0.5 manual, 1 automated, sum per section, final = MINIMUM (Table V)")
example = {"Data": ["automated"] * 3 + ["manual"] * 2 + ["none"] * 2,          # 3 + 1 = 4
           "Model": ["automated"] * 2 + ["manual"] * 2 + ["none"] * 3,         # 2 + 1 = 3
           "Infra": ["automated"] * 5 + ["none"] * 2,                          # 5
           "Monitor": ["manual"] * 3 + ["none"] * 4}                           # 1.5
sums, final, text = M.ml_test_score(example)
print(f"  a hypothetical team: section sums {sums}")
print(f"  ML Test Score = min = {final} -> \"{text}\"")
print("  -> excellent infrastructure testing (5) cannot make up for weak monitoring (1.5): the weakest area sets the")
print("     score")

section("3. Running the automated tests against our toy production system (no bugs injected)")
R = M.run_suite()
for sec, rows in R.items():
    for name, passed, detail in rows:
        mark = "PASS" if passed else ("n/a " if passed is None else "FAIL")
        print(f"  [{mark}] {sec:7s} {name:52s} {detail[:70]}")
sums, final, text = M.score_suite(R)
print(f"\n  rubric status: 26 tests automated (1 point each), 2 process tests done by hand (0.5 each)")
print(f"  section sums {sums} -> ML Test Score {final}: \"{text}\"")
print("  -> the rubric scores HAVING the tests, not passing them: our suite already exposes two real problems in the")
print("     toy system -- the useless 'noise' feature (Data 2 / Data 3) and a poor NG slice (Model 6, the paper's")
print("     'global accuracy improved but one country dropped' case). Honest caveat: 'automated' here means a script")
print("     a scheduler COULD run repeatedly; nothing in this folder actually runs on a schedule")

section("4. Bug-injection matrix: which tests catch which production bug?")
clean = {(s, n) for s, rows in R.items() for n, p, d in rows if p is False}
BUGS = {"skew": "serving code reads cents (training/serving skew)",
        "schema": "upstream starts sending spend in cents",
        "forbidden": "someone adds the forbidden 'age' feature",
        "nondeterministic": "training uses a fresh random seed each run",
        "stale": "the serving model is 40 weeks old",
        "newop": "model needs op v2, the server only has v1",
        "nan": "learning rate 1e6 (training diverges)",
        "deletion": "deleted users still in the training data",
        "depchange": "an upstream dependency is upgraded",
        "slowleak": "serving latency creeps up 60% over 40 days",
        "badmodel": "the candidate model is degraded"}
for b, desc in BUGS.items():
    Rb = M.run_suite(bugs=[b])
    caught = [f"{s} {M.RUBRIC[s].index(n) + 1}" for s, rows in Rb.items() for n, p, d in rows
              if p is False and (s, n) not in clean]
    print(f"  {desc:46s} caught by: {', '.join(caught) if caught else 'NOTHING'}")
print("  -> every bug is caught by at least one test, often by several layers (e.g. serving skew: Monitor 3 names the")
print("     feature, the canary halts at 1% traffic, Monitor 7 sees the bias). Note what distinguishes them: the")
print("     cents bug in the SERVING CODE trips Monitor 3 (skew) but not the schema test, while the same change made")
print("     UPSTREAM trips the schema tests (Data 1 / Monitor 2) but not Monitor 3, because both code paths agree")

section("5. A few of the tests up close")
tr, va = M.raw_world(seed=0), M.raw_world(seed=1)
off, on, r = M.offline_online_correlation(M.BASE_SPEC, tr)
print("  Model 2 -- intentionally degraded models (weight noise 0 ... 1.0):")
print(f"    offline log-loss  {np.round(off, 3)}")
print(f"    online buy rate   {np.round(on, 3)}   correlation {r:.2f}")
curve = M.staleness_curve(M.BASE_SPEC)
print("  Model 4 -- staleness curve (log-loss of the week-0 model vs a freshly retrained one):")
for a, (stale, fresh) in curve.items():
    print(f"    age {a:2d} weeks: stale {stale:.3f}   fresh {fresh:.3f}   gap {stale - fresh:+.3f}")
model = M.train(M.BASE_SPEC, tr)
rep, fails = M.slice_quality(model, va, M.BASE_SPEC["features"])
print(f"  Model 6 -- per-country log-loss: " + ", ".join(f"{k} {v:.3f}" for k, v in rep.items()) + f"  -> {fails}")
print("  Monitor 7 -- calibration: mean prediction vs observed rate per probability bin")
for lo, hi, n, mp, obs in M.calibration_by_slice(model, va, M.BASE_SPEC["features"]):
    print(f"    [{lo:.1f}, {hi:.1f}): n = {n:5d}   predicted {mp:.3f}   observed {obs:.3f}")
ex = model.explain(M.featurize(va, M.BASE_SPEC["features"])[0])
print("  Infra 5 -- one example, step by step:")
for f, z, c in zip(M.BASE_SPEC["features"], ex["standardised"], ex["contributions"]):
    print(f"    {f:7s} standardised {z:+.3f} x weight -> {c:+.3f}")
print(f"    bias {ex['bias']:+.3f}; logit {ex['logit']:+.3f}; probability {ex['probability']:.3f}")
print(f"  Infra 2 -- spec unit tests: {M.spec_unit_tests(M.BASE_SPEC)}")

section("What the paper says (verified against the PDF)")
for k, v in M.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
