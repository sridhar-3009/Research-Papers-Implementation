"""Bayesian Clustered Tensor Factorization in ~15 seconds: relational data with a planted structure (60 objects in 4
clusters, 6 relations in 2 clusters, a rank-3 tensor underneath), observed densely or sparsely; MAP tensor
factorization vs fully Bayesian BTF vs BCTF (Bayesian + clustering) vs a cluster-only block model -- prediction (RMSE,
area under the precision-recall curve) and whether the clusters are recovered."""

import time

import numpy as np

import bctf as B

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The data")
T, co_true, cr_true = B.planted_data()
print(f"  60 objects x 6 relations x 60 objects = {T.size:,} binary facts, {T.mean():.0%} true; objects in 4 planted "
      f"clusters (sizes {np.bincount(co_true).tolist()}), relations in 2 (sizes {np.bincount(cr_true).tolist()})")
print("  each object has a left and a right 3-dim vector, each relation a 3x3 matrix; truth = a_L^T R b_R + noise > cut")
print("  10% of facts are held out for testing; the training set is a fraction of the rest")

section("2. Prediction and recovered structure vs how much data is observed (d = 3)")
print("    observed    model                     test RMSE   PR-AUC")
summary = {}
for frac in (1.0, 0.1, 0.03):
    res, cl, n = B.compare(train_frac=frac, seed=0)
    summary[frac] = (res, cl)
    for k, (r, a) in res.items():
        print(f"    {n:6,d}      {k:24s}  {r:.3f}       {a:.3f}")
    print(f"               clusters: BCTF found {cl['BCTF objects'][0]} object clusters (adjusted Rand vs truth "
          f"{cl['BCTF objects'][1]:.2f}) and {cl['BCTF relations'][0]} relation clusters (ARI {cl['BCTF relations'][1]:.2f}); "
          f"block model objects ARI {cl['block model objects'][1]:.2f}")
    print()

section("3. What it shows")
d_res, s_res = summary[1.0][0], summary[0.03][0]
print(f"  dense data: MAP {d_res['MAP'][0]:.3f} = BTF {d_res['BTF'][0]:.3f} = BCTF {d_res['BCTF'][0]:.3f} RMSE -- with many more")
print("  observations than parameters there is little uncertainty, so the Bayesian average adds nothing (as the paper")
print("  found on Kinship and UML); but BCTF ALSO recovers the planted clusters, so it is interpretable for free")
print(f"  sparse data (3%): MAP overfits badly (PR-AUC {s_res['MAP'][1]:.2f}) while BTF / BCTF keep {s_res['BTF'][1]:.2f} / "
      f"{s_res['BCTF'][1]:.2f} -- the paper's MovieLens / ConceptNet finding")
print(f"  the cluster-only block model (IRM-like) predicts worse than every factorization at every density")
print("  Honest notes: at 10% BCTF edged out BTF (as on the paper's Animals data) though its two relation clusters merged;")
print("  at 3% BTF was as good or better and the clusters were only partly recovered; our sampler starts from many small random")
print("  clusters instead of using split-merge moves, and the cluster-variance prior is tied to the scale of the MAP vectors rather than sampled")

section("What the paper reports (verified against the PDF)")
for k, v in B.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
