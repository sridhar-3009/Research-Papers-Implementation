"""GPipe (Huang et al. 2019): pipeline parallelism with micro-batches.

  partition      cut a sequence of L layers into K consecutive cells, one per accelerator (balanced by a cost estimate)
  micro-batches  split each mini-batch of N examples into M micro-batches; stage k works on micro-batch m while stage
                 k+1 works on micro-batch m-1, so all accelerators are busy most of the time
  synchronous    gradients of all M micro-batches are ACCUMULATED and applied once per mini-batch -- the update is
                 mathematically the same as training on the whole mini-batch on one device (no staleness)
  bubble         at the start and end of every mini-batch some stages are idle: idle fraction (K - 1) / (M + K - 1),
                 negligible when M >= 4K
  re-materialise each stage keeps only its INPUT activations (the partition boundary) for every micro-batch and
                 recomputes its internal activations in the backward pass: peak activation memory
                 O(N + (L/K) (N/M)) instead of O(N L)

This file has an event simulator of the GPipe schedule (forwards for all micro-batches, then backwards in reverse),
a greedy layer partitioner, and a REAL pipelined training step in PyTorch over K stages (run sequentially on one CPU,
following the GPipe order) with optional re-materialisation, which counts the activation floats it keeps and checks
the gradients against ordinary full-batch back-propagation.
"""

import math

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------- schedule simulation

def simulate(stage_fwd, M, backward_ratio=2.0, remat=True):
    """GPipe schedule: forward of micro-batch m on stage k starts when stage k is free and stage k-1 finished m; after
    ALL forwards, backwards run in reverse stage order (micro-batches M..1). With re-materialisation each backward also
    repeats the stage's forward. Returns total time, the busy time per stage and the timeline [(stage, kind, m, start,
    end)]."""
    K = len(stage_fwd)
    f = np.asarray(stage_fwd, float)
    b = f * backward_ratio + (f if remat else 0)
    free = np.zeros(K)
    done_f = np.zeros((K, M))
    timeline = []
    for m in range(M):
        for k in range(K):
            start = max(free[k], done_f[k - 1, m] if k else 0.0)
            done_f[k, m] = start + f[k]
            free[k] = done_f[k, m]
            timeline.append((k, "F", m, start, done_f[k, m]))
    flush = free.max()
    free[:] = flush
    done_b = np.zeros((K, M))
    for m in reversed(range(M)):
        for k in reversed(range(K)):
            start = max(free[k], done_b[k + 1, m] if k < K - 1 else flush)
            done_b[k, m] = start + b[k]
            free[k] = done_b[k, m]
            timeline.append((k, "B", m, start, done_b[k, m]))
    total = free.max()
    busy = M * (f + b)
    return total, busy, timeline


def bubble_fraction(K, M):
    """Idle fraction of a perfectly balanced pipeline: (K - 1) / (M + K - 1)."""
    return (K - 1) / (M + K - 1)


def pipeline_time(layer_costs, K, M, remat=True):
    """Time for ONE mini-batch: the layers are partitioned into K stages; each of the M micro-batches costs 1/M of a
    stage's full-batch forward time."""
    stages = partition(layer_costs, K)
    fwd = [sum(layer_costs[i] for i in st) / M for st in stages]
    return simulate(fwd, M, remat=remat)[0]


def normalised_throughput(layer_costs, K, M, remat=True):
    """Throughput relative to K = 2, M = 1 (how the paper normalises Table 2)."""
    return pipeline_time(layer_costs, 2, 1, remat) / pipeline_time(layer_costs, K, M, remat)


def partition(costs, K):
    """Balanced partition of consecutive layers: cut where the cumulative cost crosses j/K of the total (j = 1..K-1),
    keeping at least one layer per stage."""
    cum = np.cumsum(costs)
    total = cum[-1]
    cuts, prev = [], 0
    for j in range(1, K):
        c = int(np.argmin(np.abs(cum - j * total / K))) + 1
        c = min(max(c, prev + 1), len(costs) - (K - j))
        cuts.append(c)
        prev = c
    bounds = [0] + cuts + [len(costs)]
    return [list(range(bounds[i], bounds[i + 1])) for i in range(K)]


# ----------------------------------------------------------------------------------------------- real pipeline

def make_model(L=12, d=64, seed=0):
    torch.manual_seed(seed)
    layers = []
    for _ in range(L):
        layers += [nn.Linear(d, d), nn.Tanh()]
    return nn.Sequential(*layers, nn.Linear(d, 1))


def split_stages(model, K):
    mods = list(model)
    per = math.ceil(len(mods) / K)
    return [nn.Sequential(*mods[i:i + per]) for i in range(0, len(mods), per)]


def gpipe_step(stages, x, y, M, remat=True):
    """One mini-batch, GPipe order: all micro-batch forwards stage by stage, then backwards in reverse; gradients
    accumulate in the parameters (no optimizer step). Returns the loss and the number of activation floats kept for
    the backward pass at the peak (end of the forward phase)."""
    loss_fn = nn.MSELoss(reduction="sum")
    micro_x, micro_y = x.chunk(M), y.chunk(M)
    K = len(stages)
    boundary = [[None] * M for _ in range(K + 1)]                 # boundary[k][m]: input of stage k for micro-batch m
    graphs = [[None] * M for _ in range(K)]                        # kept autograd outputs (no remat)
    kept = 0
    for m in range(M):                                             # forward phase
        boundary[0][m] = micro_x[m]
        for k, st in enumerate(stages):
            inp = boundary[k][m].detach().requires_grad_(k > 0)
            if remat:
                with torch.no_grad():
                    out = st(inp)
                kept += inp.numel()                                # only the partition-boundary input is stored
            else:
                out = st(inp)
                graphs[k][m] = (inp, out)
                kept += inp.numel() + sum(p.numel() for p in _activations(st, inp))
            boundary[k + 1][m] = out
    total = 0.0
    for m in reversed(range(M)):                                   # backward phase
        grad = None
        for k in reversed(range(K)):
            if remat:                                              # re-materialise the stage's forward
                inp = boundary[k][m].detach().requires_grad_(k > 0)
                out = stages[k](inp)
            else:
                inp, out = graphs[k][m]
            if k == K - 1:
                loss = loss_fn(out, micro_y[m]) / len(x)
                total += float(loss.detach())
                loss.backward()
            else:
                out.backward(grad)
            grad = inp.grad if k > 0 else None
    return total, kept


def _activations(stage, inp):
    """The intermediate activations a stage's autograd graph keeps (one per module output)."""
    acts, h = [], inp
    with torch.no_grad():
        for mod in stage:
            h = mod(h)
            acts.append(h)
    return acts


def reference_grads(model, x, y):
    model.zero_grad()
    loss = nn.MSELoss(reduction="sum")(model(x), y) / len(x)
    loss.backward()
    return float(loss.detach()), [p.grad.clone() for p in model.parameters()]


REPORTED = {
    "bubble": "idle fraction O((K - 1) / (M + K - 1)); negligible when M >= 4K",
    "memory": "with re-materialisation peak activation memory O(N + (L/K)(N/M)) instead of O(N L)",
    "Table 1 (max model size)": "AmoebaNet on 8 GB TPUv2: 82M params on one accelerator without GPipe, 318M with "
                                "re-materialisation, 1.8B on 8 accelerators (25x); Transformer on 16 GB TPUv3: 2.7x larger "
                                "on one core, 83.9B params on 128 partitions (298x)",
    "Table 2 (normalised throughput, K = 2 / 4 / 8)": "AmoebaNet M=1: 1 / 1.13 / 1.38, M=4: 1.07 / 1.26 / 1.72, M=32: "
                                                     "1.21 / 1.84 / 3.48; Transformer M=1: 1 / 1.07 / 1.3, M=4: 1.7 / 3.2 / "
                                                     "4.8, M=32: 1.8 / 3.4 / 6.3",
    "applications": "AmoebaNet-B (18, 512), 557M params, 84.4% top-1 on ImageNet-2012; a 128-layer 6B-parameter "
                    "multilingual Transformer for 103 languages that beats bilingual baselines",
    "synchronous": "gradients accumulated over the M micro-batches and applied once -- consistent regardless of K",
}
