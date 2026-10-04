"""TensorFlow OSDI 2016 in ~10 seconds: a sharded sparse embedding built from Part / Gather / Stitch dataflow ops,
sampled softmax, checkpoint-based fault tolerance, and asynchronous vs synchronous vs backup-worker replication
(Section 4.4, Figures 7-8) in an event-driven simulation."""

import time

import numpy as np

from tfsys import (REPORTED, T, backup_worker_curve, full_softmax_loss, run_with_failures, sampled_softmax_grad,
                   sharded_embedding, sparse_update, train_replicated)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. A sharded sparse embedding as plain dataflow (Part -> Gather -> Stitch, Figure 3)")
rng = np.random.default_rng(0)
G = T.Graph()
ids = G.placeholder("ids")
emb, shard_vars, table = sharded_embedding(G, ids, n_rows=100_000, dim=16, shards=4, rng=rng)
batch_ids = rng.integers(0, 100_000, 32)
out = T.Session(G).run(emb, {ids: batch_ids.astype(float)})
print(f"  100,000 x 16 embedding split over 4 PS shards (row r on shard r mod 4); a batch of 32 ids is partitioned by")
print(f"  shard, gathered on each shard, and stitched back in order: equals the dense lookup? {np.allclose(out, table[batch_ids])}")
print(f"  rows read: 32 of 100,000 ({32 * 16 * 4 / 1024:.0f} KiB instead of {100_000 * 16 * 4 / 2**20:.1f} MiB)")
touched = sparse_update(G, "emb_shard0", (batch_ids[batch_ids % 4 == 0] // 4), rng.standard_normal(((batch_ids % 4 == 0).sum(), 16)), 0.1)
print(f"  the gradient is sparse too: shard 0 updates only its {touched} gathered rows (a ScatterSub on the PS)")

section("2. Sampled softmax (the paper: 512 of 800,000 words -> 78x less softmax work)")
V, d = 20_000, 32
W_true = rng.standard_normal((V, d)) / np.sqrt(d)
H = rng.standard_normal((4096, d))
Y = (H @ W_true.T + 2 * rng.gumbel(size=(4096, V))).argmax(1)
for mode in ("full softmax", "sampled softmax (512)"):
    W = np.zeros((V, d))
    t = time.time()
    for step in range(150):
        idx = rng.integers(0, len(Y), 128)
        h, y = H[idx], Y[idx]
        if mode.startswith("full"):
            logits = h @ W.T
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            p[np.arange(len(y)), y] -= 1
            W -= 2.0 * p.T @ h / len(y)
        else:
            _, classes, gW, _ = sampled_softmax_grad(h, W, y, 512, rng)
            W[classes] -= 2.0 * gW
    print(f"    {mode:22s} full-softmax loss after 150 steps {full_softmax_loss(H[:1024], W, Y[:1024]):.3f} "
          f"(start {np.log(V):.3f}), {time.time() - t:.2f}s; rows touched per step: {'all ' + str(V) if mode.startswith('full') else 'at most 640'}")
print(f"  -> sampled softmax touches ~{V // 640}x fewer output rows per step (the paper: 800,000 / 512 classes, 78x less "
      "work). Honest note:")
print("     on this small vocabulary it learns more slowly per step; the paper reports throughput, not convergence")

section("3. Fault tolerance: Save / Restore as graph operations (Section 4.3)")
print("    checkpoint every    wall time (1000 steps, 2 failures, a save costs 2 steps)    steps redone")
for every in (10, 50, 100, 250, 500):
    r = run_with_failures(ckpt_every=every)
    print(f"    {every:6d} steps                 {r['wall time']:7.0f}                                          {r['wasted steps']:4d}")
print("  -> frequent checkpoints waste time saving; rare ones waste time redoing work after a failure -- an interval")
print("     in between is cheapest (the paper leaves the policy to the user: checkpoints are just ops)")

section("4. Backup workers (Figure 8): 50 workers, the step uses the first 50 gradients of 50 + b")
curve = backup_worker_curve()
print("    backups b    median step    normalised speedup t(0)/t(b) x 50/(50+b)")
for b, v in curve.items():
    print(f"    {b:4d}          {v['median step']:.3f}         {v['normalised speedup']:6.3f}")
best_t = min(curve, key=lambda b: curve[b]["median step"])
best_s = max(curve, key=lambda b: curve[b]["normalised speedup"])
print(f"  -> shortest step with {best_t} backups, best value for the extra machines with {best_s} (the paper: 4 and 3, 9.5%);")
print("     beyond that, extra workers mostly add parameter-server traffic. Honest note: our straggler model (6% of")
print("     workers 30% slower, plus PS cost per message) was TUNED to give this shape.")

section("5. Asynchronous vs synchronous training (20 workers, logistic regression, simulated wall clock)")
print("    scheme                         updates   mean staleness   loss at t = 5      20      40")
for label, mode, b in (("asynchronous", "async", 0), ("synchronous", "sync", 0), ("synchronous + 3 backups", "sync", 3)):
    r = train_replicated(mode, b=b, wall=40)
    at = lambda t_: [l for tt, l in r["curve"] if tt <= t_][-1]
    print(f"    {label:28s}   {r['updates']:5d}        {r['mean staleness']:5.1f}          {at(5):.3f}   {at(20):.3f}   {at(40):.3f}")
print("  -> async makes many small updates on stale parameters (~19 updates old); sync makes few big, fresh ones but")
print("     waits for stragglers -- backups recover most of that. Honest note: on this easy convex problem staleness")
print("     barely hurts, so async is slightly ahead; the paper reports 'encouraging results with synchronous")
print("     replication' and, citing prior work, that it may converge to higher accuracy than asynchronous training.")

section("6. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
