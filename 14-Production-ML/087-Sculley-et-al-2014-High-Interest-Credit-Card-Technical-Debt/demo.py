"""Technical debt in ~5 seconds: each warning of Sculley et al. (2014) as a tiny measurable experiment -- entanglement
(CACE), a legacy feature that silently breaks a model, a hidden feedback loop caught by prediction-bias monitoring, a
correction cascade, a fixed threshold that drifts, and a correlation that stops correlating."""

import time

import numpy as np

from debt import (REPORTED, ablation, cace_experiment, correction_cascade, correlation_break, feedback_loop,
                  fit_logreg, log_loss, product_number_world, threshold_drift)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. CACE -- Changing Anything Changes Everything")
base, out = cace_experiment()
print(f"  base model weights on x0..x3: {np.round(base[:4], 3)}  (x1 is a noisy copy of x0's cause, x3 is weakly causal)")
print("    change                          max shift of OTHER weights   mean |change| in predictions")
for k, (dw, dp) in out.items():
    print(f"    {k:32s} {dw:10.3f}                   {dp:8.3f}")
print("  -> dropping x0 rewrites the weights of x1..x3 (x1 was correlated with x0 and absorbs its credit); a")
print("     hyper-parameter change and missing values in x0 also move the others. Adding x4 (a noisy copy of x0)")
print("     barely moved x1..x3 -- it took its credit from x0 instead, which this column does not count")

section("2. Underutilized (legacy) feature: old and new product numbers")
Xtr, ytr, _ = product_number_world()
Xte, yte, _ = product_number_world(sample_seed=1)
Xte_clean, yte_clean, _ = product_number_world(sample_seed=1, old_populated=False)
P = Xtr.shape[1] // 2
FIT = dict(lr=5.0)                                     # one-hot columns are sparse: a larger step converges in 400 steps
both = fit_logreg(Xtr, ytr, **FIT)
new_only = fit_logreg(Xtr[:, P:], ytr, **FIT)
print("    model                          test log-loss   after old numbers stop being populated")
print(f"    old + new product numbers      {log_loss(both, Xte, yte):8.3f}        {log_loss(both, Xte_clean, yte_clean):8.3f}")
print(f"    new product numbers only       {log_loss(new_only, Xte[:, P:], yte):8.3f}        "
      f"{log_loss(new_only, Xte_clean[:, P:], yte_clean):8.3f}")
full, abl = ablation(Xtr, ytr, {"old numbers": range(P), "new numbers": range(P, 2 * P)}, Xte, yte, **FIT)
print(f"  leave-one-group-out ablation (validation log-loss {full:.3f}; increase when the group is removed):")
for k, v in abl.items():
    print(f"    without {k:12s} {v:+.3f}")
print("  -> the model split each old product's credit between its old and new number; when someone stops populating")
print("     the old numbers it gets WORSE than a model that never used them ('This will not be a good day'), while")
print("     the new-numbers-only model is unaffected. Honest note: the ablation says the redundant old numbers help")
print("     a little (removing them costs log-loss), so ablation alone would NOT have flagged them -- the danger is")
print("     the unstable upstream signal, not low offline value")

section("3. Hidden feedback loop (x_week = the user's clicks last week)")
ctr, bias, w_week = feedback_loop()
print("    week   CTR     prediction bias of last week's model   weight on x_week after retraining")
for i in range(len(ctr)):
    b = "   --" if np.isnan(bias[i]) else f"{bias[i]:+.3f}"
    print(f"    {i:3d}   {ctr[i]:.3f}   {b:>8s}                              {w_week[i]:+.3f}")
print("  -> the recommender improves in week 3; CTR jumps and last week's model under-predicts (bias ~ -0.17);")
print("     from week 4 x_week itself has shifted (users clicked more), so the model's inputs depend on its own past")
print("     outputs and the weight on x_week keeps moving. Prediction-bias monitoring is what flags the change")

section("4. Correction cascade: a' = correction learned on top of model a")
cc = correction_cascade()
for k, v in cc.items():
    print(f"    {k:32s} log-loss {v:.3f}")
print("  -> improving a (v2) helps problem A but makes A' WORSE until the correction is retrained: an improvement")
print("     deadlock")

section("5. Fixed thresholds in a changing system")
td = threshold_drift()
print(f"    v1: threshold {td['v1 threshold']:.3f} chosen for 90% precision -> precision {td['v1 precision']:.3f}")
print(f"    v2 (retrained, rebalanced) with v1's fixed threshold -> precision {td['v2 with v1' + chr(39) + 's fixed threshold']:.3f}")
print(f"    v2 with a threshold re-learned on held-out data ({td['v2 re-learned threshold']:.3f}) -> precision "
      f"{td['v2 precision (re-learned)']:.3f}")
print("  -> hand-set thresholds silently change meaning when the model is updated; learn them on held-out data")

section("6. Correlations that stop correlating")
cb = correlation_break()
print(f"    weights (cause, proxy) = {np.round(cb['weights (cause, proxy)'], 2)}")
print(f"    accuracy in training world {cb['accuracy before']:.3f}; after the world decouples them "
      f"{cb['accuracy after decoupling']:.3f}; causal-only model {cb['causal-only model after']:.3f}")
print("  -> the model split credit between the cause and its proxy; when they decouple it degrades")

section("What the paper says (verified against the PDF)")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
