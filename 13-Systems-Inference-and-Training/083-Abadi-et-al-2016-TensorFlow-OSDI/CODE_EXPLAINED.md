# The code, explained simply

How the code in this folder implements the mechanisms of TensorFlow OSDI 2016.
Read [EXPLAINED.md](EXPLAINED.md) first.

The code builds on paper 082's mini-TensorFlow (`minitf.py`), loaded with `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `tfsys.py` | Part / Gather / Stitch / LocalIndex kernels, a sharded embedding builder, sparse ScatterSub, sampled softmax, a worker step-time model, synchronous steps with backups, the Figure 8 curve, an event-driven async/sync trainer, and checkpoint/restart simulation |
| `experiments.py` | backup sweeps, step-time percentiles, learning-rate sensitivity, sampled softmax grid, real `tf.distribute` |
| `demo.py` | embedding, sampled softmax, checkpoints, backup workers, async vs sync (~3 seconds) |
| `test_tfsys.py` | 3 quick tests (~0.1 seconds) |

**Run it** (from `13-Systems-Inference-and-Training/083-Abadi-et-al-2016-TensorFlow-OSDI`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~3 seconds
python3 experiments.py --quick
```

---

## 2. `tfsys.py`

### Sparse embedding (dataflow)
- **Kernels added to the mini-TensorFlow:**
  - `Part(ids, shards, which)` keeps the ids with id mod shards = which;
  - `LocalIndex` maps id → id // shards (the row index within a shard);
  - `Gather(table, idx)` reads those rows;
  - `Stitch(ids, shards, *parts)` writes each shard's rows back to the positions of its ids.
- **`sharded_embedding(G, ids, n_rows, dim, shards, rng)`:** creates one Variable per shard (rows s, s + shards, …) placed on `ps:s`, wires Part → LocalIndex → Gather per shard, and Stitch at the end.
- **`sparse_update`:** `np.subtract.at` on only the touched rows of a shard variable.

### Sampled softmax
- **`sampled_softmax_grad(h, W, y, k, rng)`:** take the union of the true labels and k uniform samples, softmax over those classes only, and return the loss, the touched classes, the gradient for those rows of W, and the gradient for h. With uniform sampling, the log-Q correction is a constant.
- **`full_softmax_loss`:** the true loss, for evaluation.

### Replication
| Name | What it does |
|---|---|
| `worker_times(rng, n, …)` | lognormal jitter, where each worker is a straggler with probability `p_straggle` (slower by `slow`); the defaults were tuned for the Figure 8 shape |
| `sync_step_time(rng, n, b, ps_cost)` | the n-th fastest of n + b worker times, plus `ps_cost` × (n + b) server traffic |
| `backup_worker_curve(n, max_b)` | median step per b, and normalised speed-up t(0)/t(b) × n/(n + b) |
| `train_replicated(mode, n_workers, b, lr, wall)` | **async:** a heap of (finish time, worker, parameters read, version read); each event applies its gradient (computed on the parameters it read) and records staleness = updates since the read. **Sync:** each step averages n gradients on the current parameters and advances time by the n-th fastest of n + b workers. |

### Checkpoints
- **`run_with_failures(total_steps, ckpt_every, fail_at, ckpt_cost)`:** counts wall time including save costs, and redone steps after each failure (rolling back to the last checkpoint).

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | best b for time and for normalised speed-up across worker counts, straggler models and server cost |
| `e2` | p50/p90/p99 step times and gradients/s, sync vs async |
| `e3` | learning rate × {async, sync, sync + 3} final loss and staleness |
| `e4` | vocabulary × number of samples: time, loss, work ratio |
| `e5` | `tf.distribute.MirroredStrategy` vs one device: examples/s for a small Keras MLP |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_sharded_embedding_equals_dense_lookup_and_sparse_update` | the Part/Gather/Stitch output equals table[ids], including repeated ids; ScatterSub changes only the touched rows |
| `test_sampled_softmax_touches_few_rows_and_is_finite` | ≤ k + batch classes, which include all labels; gradient shapes |
| `test_backups_checkpoints_and_replication` | backups shorten the median step; checkpoint rollback arithmetic; async makes more updates with staleness > 0, sync has 0; both reduce the loss |

---

## 5. Try it yourself

1. Give async training a much larger learning rate. When does staleness start to hurt?
2. Make stragglers correlated (a whole machine slow for many steps). How do backups cope?
3. Add log-uniform (Zipf) sampling to sampled softmax with the log-Q correction.
4. Checkpoint asynchronously (concurrently with training) and measure how inconsistent the snapshots are.
5. Shard the embedding by contiguous ranges instead of id mod shards, and measure load imbalance with Zipf-distributed ids.
