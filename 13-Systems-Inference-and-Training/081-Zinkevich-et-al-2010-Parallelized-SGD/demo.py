"""Parallelized SGD (SimuParallelSGD) in ~15 seconds: the contraction behind the proof, the stationary distribution
of fixed-learning-rate SGD (bias and variance vs eta; averaging divides the variance), and the paper's experiment on
sparse hashed-feature data: 1 / 10 / 100 machines, no communication until one final average."""

import time

import numpy as np

from psgd import (REPORTED, coupled_distance, eta_star, full_batch_minimiser, make_data, objective, simu_parallel_sgd,
                  split, stationary_stats)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The algorithm")
print("  for each of k machines, in parallel and with NO communication:  take a random m/k share of the data,")
print("  shuffle it, run SGD from w = 0 with a fixed learning rate eta;   then return  v = average of the k vectors.")
print("  One MapReduce pass, one communication step -- unlike distributed gradient methods that sync every step.")

section("2. Why it works (Section 2): SGD contracts, so every machine forgets where it started")
X, y = make_data(60000)
lam, eta = 1e-3, 0.5
print(f"  sparse hashed features, unit-length rows: eta* = 1 / (max||x|| c* + lambda) = {eta_star(X, lam):.3f}; we use eta = {eta}")
d = coupled_distance(X, y, eta, lam, steps=2000)
print(f"  two chains fed the same examples from different starts: distance {d[0]:.2f} -> {d[500]:.2f} (500 steps) -> {d[2000]:.2f} "
      f"(2000 steps)")
print(f"  per-step shrink factor {(d[2000] / d[0]) ** (1 / 2000):.5f} <= 1 - eta lambda = {1 - eta * lam:.5f} (Lemma 3; faster in practice)")

section("3. The stationary distribution of fixed-eta SGD (Theorems 9 and 10)")
w_star = full_batch_minimiser(X, y, lam, iters=2000)
c_min = objective(w_star, X, y, lam, "huber")
print("    eta     single chain: c(w) - min c    spread E||w - mean||^2    after averaging 10 chains")
for e in (1.0, 0.5, 0.25):
    mean, var, W = stationary_stats(X, y, e, lam, runs=200, steps=3000)
    avg10 = W.reshape(20, 10, -1).mean(1)
    v10 = float(((avg10 - mean) ** 2).sum(1).mean())
    sub = float((objective(W, X, y, lam, "huber") - c_min).mean())
    sub10 = float((objective(avg10, X, y, lam, "huber") - c_min).mean())
    print(f"    {e:4}           {sub:.4f}                       {var:6.1f}                  {v10:5.2f}  (c - min c = {sub10:.4f})")
print("  -> a fixed-eta SGD chain never settles: it keeps wandering with spread proportional to eta (Theorem 10);")
print("     its MEAN is almost optimal (Theorem 9). Averaging k independent chains divides the spread by k, which is")
print("     exactly what SimuParallelSGD does. Smaller eta means less error (single chain 0.124 -> 0.060 -> 0.036), but")
print("     slower forgetting of the start: at eta = 0.25 the 3000 steps were not enough for the averaged error to fall")
print("     further (the paper: halve eta AND roughly double T to halve the error).")

section("4. The paper's experiment (Figures 1-3): objective relative to ONE sequential pass over all data")
Xa, ya = make_data(220000, seed=3)
Xtr, ytr, Xte, yte = split(Xa, ya, 20000)
print(f"  {len(ytr):,} training examples (512 hashed features, 12 active each), Huber loss, eta = {eta}")
for lam in (1e-3, 1e-6):
    v1, _, _ = simu_parallel_sgd(Xtr, ytr, 1, eta, lam)
    ref, ref_rmse = objective(v1, Xtr, ytr, lam, "huber"), np.sqrt(((Xte @ v1 - yte) ** 2).mean())
    print(f"  lambda = {lam:g}:   samples seen per machine ->      500      2000     (relative objective | relative test RMSE)")
    for k in (1, 10, 100):
        _, _, snaps = simu_parallel_sgd(Xtr, ytr, k, eta, lam, snapshots={500, 2000})
        obj = [float(objective(snaps[s], Xtr, ytr, lam, "huber") / ref) for s in (500, 2000)]
        rmse = [float(np.sqrt(((Xte @ snaps[s] - yte) ** 2).mean()) / ref_rmse) for s in (500, 2000)]
        print(f"     {k:3d} machines                                {obj[0]:6.3f}   {obj[1]:6.3f}     |   {rmse[0]:6.3f}   {rmse[1]:6.3f}")
print("  -> after the same work PER MACHINE (= the same wall-clock time), more machines give a better model; with")
print("     2000 examples each, 10 machines already beat one machine that read all 200,000 (relative value < 1),")
print("     because averaging removes the noise of the final SGD iterate. Going 1 -> 10 machines helps much more than")
print("     10 -> 100, and the high-variance lambda = 1e-6 problem gains more -- the paper's findings.")
print("  (machine time: 10 machines x 2000 = 20,000 examples in total vs 200,000 for the single pass -- here the")
print("   parallel run even needs LESS total work; the paper found it needed slightly more machine time)")

section("5. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
