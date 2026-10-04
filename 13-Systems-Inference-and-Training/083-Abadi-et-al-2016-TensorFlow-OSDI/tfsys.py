"""TensorFlow: A System for Large-Scale Machine Learning (Abadi et al., OSDI 2016) -- the parameter-server ideas
expressed as ordinary dataflow, in miniature.

  mutable state       Variables live on parameter-server (PS) tasks; the update W <- W - lr * dL/dW is just an
                      AssignSub node placed on the PS -- so users can write their own update rules
  sparse embeddings   a sharded embedding matrix read with Part (split the ids by shard) -> Gather (on each PS shard)
                      -> Stitch (reassemble in order); the gradient touches only the gathered rows (Section 4.2)
  sampled softmax     a vocabulary of 800,000 words: score only the true word + 512 sampled ones (78x less work)
  fault tolerance     Save / Restore are graph operations run periodically; on failure restart from the latest
                      checkpoint (Section 4.3)
  replication         (a) asynchronous: every worker reads, computes, writes on its own -- high throughput, STALE
                      gradients; (b) synchronous: aggregate all n gradients, then update -- fresh gradients, but the
                      step waits for the slowest worker; (c) synchronous with b BACKUP workers: run n + b, use the
                      first n to finish -- stragglers stop mattering (Section 4.4)

This file re-uses paper 082's mini-TensorFlow (adding Part / Gather / Stitch / ScatterSub kernels), and simulates
workers with random step times (a straggler tail) in an event-driven way, to compare the three replication
schemes on throughput, step time and convergence.
"""

import heapq
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "minitf082", Path(__file__).resolve().parent.parent / "082-Abadi-et-al-2015-TensorFlow-Whitepaper" / "minitf.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

# ----------------------------------------------------------------------------------------------- sparse embedding ops

T.KERNELS["Part"] = lambda ids, shards, which: np.asarray(ids[(ids.astype(int) % int(shards)) == int(which)])
T.KERNELS["Gather"] = lambda table, ids: table[(ids.astype(int) // 1)]
T.KERNELS["Stitch"] = None                                             # needs the original order: see below


def _stitch(ids, shards, *parts):
    """Reassemble per-shard gathered rows into the original id order."""
    ids = ids.astype(int)
    out = np.zeros((len(ids), parts[0].shape[1] if len(parts[0]) else 0))
    for s, rows in enumerate(parts):
        pos = np.where(ids % shards == s)[0]
        out[pos] = rows
    return out


T.KERNELS["Stitch"] = lambda ids, shards, *parts: _stitch(ids, int(shards), *parts)


def sharded_embedding(G, ids, n_rows, dim, shards, rng):
    """Build Part -> Gather (per PS shard) -> Stitch. Row r lives on shard r % shards at local index r // shards.
    Returns (output node, list of shard variable nodes, the full table for checking)."""
    table = rng.standard_normal((n_rows, dim))
    shard_vars, gathered = [], []
    k = G.constant(float(shards))
    for s in range(shards):
        var = G.variable(table[s::shards], f"emb_shard{s}")
        var.device = f"ps:{s}"
        local = G.op("LocalIndex", G.op("Part", ids, k, G.constant(float(s))), k)
        g = G.op("Gather", var, local)
        g.device = f"ps:{s}"
        shard_vars.append(var)
        gathered.append(g)
    return G.op("Stitch", ids, k, *gathered), shard_vars, table


T.KERNELS["LocalIndex"] = lambda ids, shards: np.asarray(ids.astype(int) // int(shards), dtype=float)


def sparse_update(G, table_name, rows, grads, lr):
    """ScatterSub: subtract lr * grad only from the touched rows of a shard (what a PS does for sparse updates)."""
    var = G.variables[table_name]
    np.subtract.at(var, rows, lr * grads)
    return len(np.unique(rows))

# ----------------------------------------------------------------------------------------------- sampled softmax

def full_softmax_loss(h, W, y):
    logits = h @ W.T
    logits -= logits.max(1, keepdims=True)
    return float(-(logits[np.arange(len(y)), y] - np.log(np.exp(logits).sum(1))).mean())


def sampled_softmax_grad(h, W, y, n_sampled, rng):
    """Score only the true classes and n_sampled uniformly drawn classes; returns the loss on that subset, the rows of
    W touched and their gradient. (Uniform sampling, so the log-Q correction is a constant and cancels.)"""
    V = W.shape[0]
    sampled = rng.choice(V, n_sampled, replace=False)
    classes = np.unique(np.concatenate([y, sampled]))
    idx = {c: i for i, c in enumerate(classes)}
    Ws = W[classes]
    logits = h @ Ws.T
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    tgt = np.array([idx[c] for c in y])
    loss = float(-np.log(p[np.arange(len(y)), tgt] + 1e-12).mean())
    p[np.arange(len(y)), tgt] -= 1
    return loss, classes, (p.T @ h) / len(y), (p @ Ws) / len(y)

# ----------------------------------------------------------------------------------------------- replication simulator

def worker_times(rng, n, base=1.0, sigma=0.03, p_straggle=0.06, slow=1.3):
    """Step time of each worker: lognormal jitter plus a chance of being a straggler (e.g. a busy machine). The
    defaults were TUNED so that a 50-worker job shows the shape of the paper's Figure 8; the mechanism (order
    statistics + PS traffic), not the exact numbers, is the point."""
    t = base * np.exp(sigma * rng.standard_normal(n))
    return t * np.where(rng.random(n) < p_straggle, slow, 1.0)


def sync_step_time(rng, n, b=0, ps_cost=0.01, **kw):
    """Synchronous step with b backup workers: wait for the n-th fastest of n + b workers; every worker that reports
    before the step closes (n of them plus the discarded backups still in flight) sends traffic to the PS."""
    t = np.sort(worker_times(rng, n + b, **kw))
    return t[n - 1] + ps_cost * (n + b)


def backup_worker_curve(n=50, max_b=6, steps=3000, seed=0, **kw):
    """Figure 8: median step time t(b) and normalised speedup (t(0)/t(b)) * n/(n+b) for b = 0..max_b."""
    rng = np.random.default_rng(seed)
    med = {b: float(np.median([sync_step_time(rng, n, b, **kw) for _ in range(steps)])) for b in range(max_b + 1)}
    return {b: {"median step": med[b], "normalised speedup": med[0] / med[b] * n / (n + b)} for b in med}


def logistic_problem(n=4000, d=20, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    y = (X @ w + 0.5 * rng.standard_normal(n) > 0).astype(float)
    return X, y


def loss_and_grad(w, X, y, idx=None):
    if idx is not None:
        X, y = X[idx], y[idx]
    p = 1 / (1 + np.exp(-(X @ w)))
    loss = float(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean())
    return loss, X.T @ (p - y) / len(y)


def train_replicated(mode, n_workers=20, b=0, lr=0.5, wall=60.0, batch=32, seed=0, **kw):
    """Event-driven data-parallel SGD on a logistic-regression problem.
      async: each worker reads w, computes a gradient taking its random step time, then applies it (stale by however
             many updates happened meanwhile);
      sync:  each step every one of n (+ b backup) workers computes a gradient on the same w; the step ends when the
             n-th finishes, the n gradients are averaged and applied.
    Returns the loss curve against wall-clock time, the number of updates and the mean staleness."""
    rng = np.random.default_rng(seed)
    X, y = logistic_problem(seed=seed)
    w = np.zeros(X.shape[1])
    curve, updates, stale = [(0.0, loss_and_grad(w, X, y)[0])], 0, []
    if mode == "async":
        version = 0
        events = []
        for i in range(n_workers):
            heapq.heappush(events, (worker_times(rng, 1, **kw)[0], i, w.copy(), version))
        while events:
            t, i, w_read, v_read = heapq.heappop(events)
            if t > wall:
                break
            _, g = loss_and_grad(w_read, X, y, rng.integers(0, len(y), batch))
            w -= lr / n_workers * g * 1.0
            version += 1
            updates += 1
            stale.append(version - 1 - v_read)
            if updates % 20 == 0:
                curve.append((t, loss_and_grad(w, X, y)[0]))
            heapq.heappush(events, (t + worker_times(rng, 1, **kw)[0], i, w.copy(), version))
    else:
        t = 0.0
        while t < wall:
            times = worker_times(rng, n_workers + b, **kw)
            used = np.argsort(times)[:n_workers]
            t += np.sort(times)[n_workers - 1]
            g = np.mean([loss_and_grad(w, X, y, rng.integers(0, len(y), batch))[1] for _ in used], axis=0)
            w -= lr * g
            updates += 1
            stale.append(0)
            curve.append((t, loss_and_grad(w, X, y)[0]))
    return {"curve": curve, "updates": updates, "mean staleness": float(np.mean(stale)) if stale else 0.0,
            "final loss": curve[-1][1]}

# ----------------------------------------------------------------------------------------------- checkpoints

def run_with_failures(total_steps=1000, ckpt_every=100, fail_at=(350, 720), step_time=1.0, ckpt_cost=2.0):
    """Training that checkpoints every `ckpt_every` steps (a Save op costing `ckpt_cost`) and, after a failure, restarts
    from the latest checkpoint (Restore). Returns wall-clock time and wasted steps."""
    t, step, wasted, last_ckpt = 0.0, 0, 0, 0
    fails = sorted(fail_at)
    while step < total_steps:
        step += 1
        t += step_time
        if step % ckpt_every == 0:
            last_ckpt = step
            t += ckpt_cost
        if fails and step == fails[0]:
            fails.pop(0)
            wasted += step - last_ckpt
            step = last_ckpt
    return {"wall time": t, "wasted steps": wasted}


REPORTED = {
    "replication (Section 4.4)": "asynchronous, synchronous, synchronous with backup workers; backup workers improve "
                                 "throughput by up to 15%",
    "Figure 8 (50-worker Inception-v3)": "4 backup workers give the shortest median step (1.93 s); 3 give the best "
                                         "normalised speedup (9.5%); a 5th backup slightly hurts (more PS traffic)",
    "synchronous microbenchmark": "median synchronous step ~10% longer than asynchronous with the same workers, much "
                                  "worse above the 90th percentile; null-step time 1.8 ms (1 worker) -> 8.8 ms (100)",
    "Inception-v3 scaling": "throughput rises to 2,300 images/s with 200 workers, with diminishing returns",
    "language model (One Billion Word, 800k vocabulary)": "sampled softmax with 512 classes cuts softmax data transfer "
                                                          "and computation by 78x; more PS tasks raise throughput",
    "single GPU (Table 1)": "TensorFlow faster than Caffe and within 6% of Torch (both use cuDNN); Neon faster on 3 models",
    "embeddings": "sparse embedding layers from Gather / Part / Stitch; some document models have parameters of "
                  "several terabytes",
}
