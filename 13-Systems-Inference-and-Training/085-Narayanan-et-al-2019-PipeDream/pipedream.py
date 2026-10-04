"""PipeDream (Narayanan et al., SOSP 2019 / arXiv 2018): asynchronous pipeline-parallel training.

  partitioning     profile each layer's compute time, weight size and activation size; a DYNAMIC PROGRAM chooses how
                   to cut the layers into stages AND how many machines replicate each stage (data parallelism inside a
                   stage), minimising the time of the slowest stage, counting inter-stage activation transfers and
                   intra-stage weight synchronisation:
                     A(j, m) = min over i < j, m' < m of  max( A(i, m - m'), 2 C_i, T(i+1 -> j, m') )
                     T(i -> j, m) = (1/m) max( sum of compute,  sum of weight-sync time )        (O(N^2 M^2))
  1F1B             in steady state every stage alternates one forward and one backward pass, so no GPU idles;
                   NOAM = ceil(#machines / #machines in the input stage) minibatches are admitted to fill the pipe
  weight versions  the pipeline never flushes, so a minibatch's backward runs after other minibatches have updated the
                   weights. Naive pipelining computes the gradient with a DIFFERENT weight version than the forward pass
                   used -- not a valid gradient of anything. WEIGHT STASHING keeps the version used in the forward pass
                   and reuses it in the backward (stage k is n-k+1 updates stale but consistent). VERTICAL SYNC makes all
                   stages use the same version (equivalent to data-parallel BSP); the paper found it unnecessary.

This file has the DP partitioner, a 1F1B vs GPipe schedule simulator, and a REAL asynchronous pipeline trainer for a
small MLP split into stages, with hand-written backprop so the backward pass can use any weight version -- to compare
plain SGD, naive pipelining, weight stashing and vertical sync.
"""

import math
from functools import lru_cache

import numpy as np

# ----------------------------------------------------------------------------------------------- partitioner

def partition_dp(compute, weights, activations, machines, bandwidth=1.0):
    """PipeDream's dynamic program. compute[l]: time of layer l (forward + backward), weights[l]: its parameter size,
    activations[l]: size of its output. Returns (time per minibatch of the slowest stage, [(first, last, replicas)])."""
    N = len(compute)
    pc, pw = np.concatenate([[0], np.cumsum(compute)]), np.concatenate([[0], np.cumsum(weights)])

    def T(i, j, m):                                                 # layers i..j (inclusive), m replicas
        comp = pc[j + 1] - pc[i]
        sync = 0.0 if m == 1 else 2 * (m - 1) / m * (pw[j + 1] - pw[i]) / bandwidth   # ring all-reduce of weights
        return max(comp, sync) / m

    @lru_cache(None)
    def A(j, m):                                                    # best time for layers 0..j on m machines
        best = (T(0, j, m), ((0, j, m),))                           # one (possibly replicated) stage
        for i in range(j):
            for mp in range(1, m):
                left, stages = A(i, m - mp)
                cand = max(left, 2 * activations[i] / bandwidth, T(i + 1, j, mp))
                if cand < best[0]:
                    best = (cand, stages + ((i + 1, j, mp),))
        return best
    t, stages = A(N - 1, machines)
    return t, list(stages)


def config_string(stages):
    """PipeDream's notation, e.g. '2-1-1' = 3 stages, the first replicated on 2 machines."""
    return "-".join(str(m) for _, _, m in stages)


def noam(stages):
    """NUM_OPT_ACTIVE_MINIBATCHES = ceil(#machines / #machines in the input stage)."""
    return math.ceil(sum(m for _, _, m in stages) / stages[0][2])

# ----------------------------------------------------------------------------------------------- schedules

def simulate_1f1b(stage_time, n_mb, bwd_ratio=2.0):
    """1F1B: stage k starts with (S - k) forwards (warm-up), then alternates backward/forward, then drains. A forward
    of minibatch m on stage k needs stage k-1's forward of m; a backward needs stage k+1's backward of m. Returns the
    makespan and the peak number of minibatches whose activations a stage holds."""
    S = len(stage_time)
    f = np.asarray(stage_time, float)
    b = f * bwd_ratio
    done_f, done_b = {}, {}
    free = np.zeros(S)
    order = []
    for k in range(S):                                              # each stage's op sequence
        warm = min(S - k, n_mb)
        seq = [("F", m) for m in range(warm)]
        nf, nb = warm, 0
        while nb < n_mb:
            seq.append(("B", nb))
            nb += 1
            if nf < n_mb:
                seq.append(("F", nf))
                nf += 1
        order.append(seq)
    ptr = [0] * S
    peak = [0] * S
    while any(ptr[k] < len(order[k]) for k in range(S)):
        progressed = False
        for k in range(S):
            if ptr[k] >= len(order[k]):
                continue
            kind, m = order[k][ptr[k]]
            if kind == "F":
                dep = done_f.get((k - 1, m)) if k else 0.0
            else:
                dep = done_b.get((k + 1, m)) if k < S - 1 else done_f.get((k, m))
            if dep is None:
                continue
            start = max(free[k], dep)
            end = start + (f[k] if kind == "F" else b[k])
            (done_f if kind == "F" else done_b)[(k, m)] = end
            free[k] = end
            ptr[k] += 1
            progressed = True
            held = sum(1 for (kk, mm) in done_f if kk == k and (kk, mm) not in done_b)
            peak[k] = max(peak[k], held)
        if not progressed:
            raise RuntimeError("schedule deadlock")
    return max(free), peak


def simulate_gpipe(stage_time, n_mb, bwd_ratio=2.0):
    """GPipe-style flush: all forwards, then all backwards (no overlap between them). Peak in-flight = n_mb."""
    S = len(stage_time)
    f = np.asarray(stage_time, float)
    b = f * bwd_ratio
    free = np.zeros(S)
    done = {}
    for m in range(n_mb):
        for k in range(S):
            done[(k, m)] = max(free[k], done.get((k - 1, m), 0.0)) + f[k]
            free[k] = done[(k, m)]
    flush = free.max()
    free[:] = flush
    db = {}
    for m in reversed(range(n_mb)):
        for k in reversed(range(S)):
            dep = flush if k == S - 1 else db[(k + 1, m)]
            db[(k, m)] = max(free[k], dep) + b[k]
            free[k] = db[(k, m)]
    return free.max(), [n_mb] * S

# ----------------------------------------------------------------------------------------------- async pipeline SGD

def make_problem(n=2048, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W1, W2 = rng.standard_normal((d, 32)) / 4, rng.standard_normal((32, 1)) / 6
    y = np.tanh(X @ W1) @ W2 + 0.05 * rng.standard_normal((n, 1))
    return X, y


def init_stages(n_stages, d_in=16, width=32, seed=1):
    rng = np.random.default_rng(seed)
    dims = [d_in] + [width] * (n_stages - 1) + [1]
    return [rng.standard_normal((dims[i], dims[i + 1])) / np.sqrt(dims[i]) for i in range(n_stages)]


def forward(Ws, x):
    """Stages: h_{k+1} = tanh(h_k W_k) except the last, which is linear. Returns all stage inputs and the output."""
    hs = [x]
    for k, W in enumerate(Ws):
        z = hs[-1] @ W
        hs.append(z if k == len(Ws) - 1 else np.tanh(z))
    return hs


def backward(Ws_bwd, hs, y):
    """Back-propagate the squared error with the weights Ws_bwd (which may differ from those used in the forward that
    produced hs). Returns one weight gradient per stage."""
    n = len(y)
    delta = (hs[-1] - y) * (2 / n)
    grads = [None] * len(Ws_bwd)
    for k in reversed(range(len(Ws_bwd))):
        grads[k] = hs[k].T @ delta
        if k:
            delta = (delta @ Ws_bwd[k].T) * (1 - hs[k] ** 2)       # through the tanh that produced hs[k]
    return grads


def train_pipelined(mode, n_stages=4, steps=600, lr=0.05, batch=64, seed=0):
    # (diverging runs overflow; those warnings are expected and silenced)
    """Asynchronous pipeline semantics in the steady state of 1F1B with n stages (minibatch t's update to stage k
    happens n - k minibatch-updates after its forward on stage k read the weights):
      'sgd'      ordinary minibatch SGD (no pipeline)
      'naive'    forward on stage k with the version from n - k updates ago, backward with the LATEST weights
      'stash'    weight stashing: forward and backward on stage k both use the version from n - k updates ago
      'vsync'    vertical sync: every stage uses the version from n - 1 updates ago (= data-parallel BSP semantics)
    Returns the loss curve on the full data set."""
    np.seterr(over="ignore", invalid="ignore")
    rng = np.random.default_rng(seed)
    X, y = make_problem(seed=seed)
    Ws = init_stages(n_stages)
    history = [[W.copy() for W in Ws]]                               # history[t] = weights after t updates
    curve = []
    for t in range(steps):
        idx = rng.integers(0, len(X), batch)
        version = lambda delay: history[max(0, len(history) - 1 - delay)]
        if mode == "sgd":
            fwd = bwd = history[-1]
        elif mode == "vsync":
            fwd = bwd = version(n_stages - 1)
        else:
            fwd = [version(n_stages - 1 - k)[k] for k in range(n_stages)]
            bwd = fwd if mode == "stash" else history[-1]
        hs = forward(fwd, X[idx])
        grads = backward(bwd, hs, y[idx])
        Ws = [W - lr * g for W, g in zip(history[-1], grads)]
        history.append([W.copy() for W in Ws])
        if len(history) > n_stages + 2:
            history.pop(0)
        if t % 20 == 0 or t == steps - 1:
            curve.append(float(((forward(Ws, X)[-1] - y) ** 2).mean()))
    return curve


REPORTED = {
    "headline": "up to 5x faster time-to-accuracy than data-parallel training (BSP)",
    "Table 1 (speedup over BSP, communication reduction)": "VGG16: 4 machines (A) config 2-1-1 2.13x, 90%; 8 (A) 7-1 "
        "2.99x, 95%; 16 (A) 9-5-1-1 3.00x, 91%; 8 (B) 7-1 5.12x, 95%. Inception-v3: 8 (A) data parallel (8) 1.00x; "
        "8 (B) 7-1 1.45x, 47%. S2VT: 4 (A) 2-1-1 3.01x, 95%",
    "scheduling": "1F1B in steady state; NOAM = ceil(#machines / #machines in the input stage) minibatches in flight; "
                  "round-robin across replicas of a stage",
    "weight versions": "naive pipelining does not reach data-parallel accuracy; weight stashing is critical; vertical sync "
                       "had negligible impact and is off by default",
    "partitioning": "dynamic programming over layers and replication factors from a short profiling run, O(N^2 M^2)",
}
