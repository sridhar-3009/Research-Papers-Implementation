"""PipeDream in ~5 seconds: the dynamic-programming partitioner that mixes pipeline and data parallelism (it rediscovers
the paper's '7-1' layout for a VGG-like network), 1F1B scheduling vs flushing pipelines, and what weight stashing and
vertical sync do to the gradient -- trained for real on a small pipelined MLP."""

import time

import numpy as np

from pipedream import (REPORTED, config_string, noam, partition_dp, simulate_1f1b, simulate_gpipe, train_pipelined)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Partitioning by dynamic programming (Section 3.2)")
compute = [4, 4, 6, 6, 6, 5, 5, 5, 3, 3, 1, 1]                     # VGG-like: heavy convolutions first ...
weights = [0.01, 0.04, 0.1, 0.15, 0.3, 0.6, 0.6, 0.6, 1, 1, 40, 8]  # ... huge fully-connected weights last
acts = [3, 3, 2, 2, 1.5, 1, 1, 0.8, 0.5, 0.5, 0.05, 0.01]
print("  12 layers: convolution layers have most of the compute but few weights; the last fully-connected layers hold")
print("  most of the weights (40 + 8 of 52 units) but little compute -- like VGG-16")
print("    machines  network   PipeDream config   time/minibatch   data-parallel time   speedup   NOAM")
for M in (4, 8, 16):
    for bw, net in ((0.5, "slow"), (4.0, "fast")):
        t, st = partition_dp(compute, weights, acts, M, bw)
        tdp = max(sum(compute), 2 * (M - 1) / M * sum(weights) / bw) / M
        print(f"    {M:5d}     {net:5s}     {config_string(st):12s}      {t:8.3f}          {tdp:8.3f}         {tdp / t:5.2f}x   {noam(st)}")
print("  -> on a slow network the DP replicates the convolution layers (data parallel, cheap to synchronise) and puts")
print("     the weight-heavy fully-connected layers on ONE machine so their 48 units of weights are never all-reduced:")
print("     '7-1' on 8 machines -- the configuration the paper reports for VGG-16 (5.12x over BSP on cluster B, 95% less")
print("     communication). On a fast network plain data parallelism ('8') is already best.")

section("2. 1F1B scheduling: no flushes, fewer minibatches in flight (Section 3.3)")
stages = [1.0] * 4
t1, peak1 = simulate_1f1b(stages, 16)
tg_one, _ = simulate_gpipe(stages, 4)
print(f"  4 balanced stages, 16 minibatches, backward = 2 x forward:")
print(f"    PipeDream 1F1B (updates never stop the pipe):          {t1:5.0f} time units; minibatches held per stage {peak1}")
print(f"    flush after every 4 (a GPipe mini-batch of 4 micro-batches): {4 * tg_one:5.0f} time units; held per stage [4, 4, 4, 4]")
print("  -> 1F1B keeps every stage busy in steady state and stage k holds at most (stages - k) minibatches (NOAM = 4");
print("     for the input stage); a synchronous pipeline drains and refills at every update")

section("3. Weight versions: naive pipelining vs weight stashing vs vertical sync (Section 3.4)")
print("  an MLP split into S stages trained with the 1F1B steady-state staleness: stage k's forward reads the weights")
print("  from S - k updates ago. Median final loss over 5 seeds ('diverged' = all 5 blew up):")
print("    stages  lr     plain SGD   naive      weight stashing   vertical sync")
for S, lr in ((4, 0.1), (4, 0.2), (8, 0.1), (8, 0.2)):
    res = {}
    for m in ("sgd", "naive", "stash", "vsync"):
        v = [train_pipelined(m, S, lr=lr, steps=600, seed=s)[-1] for s in range(5)]
        med = float(np.median(v))
        res[m] = f"{med:.4f}" if np.isfinite(med) and med < 1 else "diverged"
    print(f"    {S:4d}    {lr:4}   {res['sgd']:9s}   {res['naive']:9s}  {res['stash']:9s}         {res['vsync']}")
print("  -> pipelining costs some accuracy at this step budget (stale weights), more with more stages.")
print("     Honest notes: (1) we do NOT reproduce naive pipelining failing -- on this small smooth MLP its mismatched")
print("     forward/backward weights hurt no more than stashing's staleness (at 8 stages, lr 0.2 naive is even better);")
print("     weight stashing's guarantee is that each stage's gradient is a VALID gradient, which the paper needed on")
print("     large CNNs and RNNs. (2) vertical sync makes EVERY stage S - 1 updates stale and diverges at these rates,")
print("     whereas the paper found it unnecessary (and left it off).")

section("4. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
