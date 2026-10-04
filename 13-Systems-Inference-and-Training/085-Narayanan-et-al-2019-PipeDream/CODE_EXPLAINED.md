# The code, explained simply

How the code in this folder implements PipeDream (Narayanan et al. 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `pipedream.py` | the DP partitioner with replication, configuration strings and NOAM, 1F1B and flushing schedule simulators, and a pipelined MLP trainer with naive / stashing / vertical-sync weight semantics (manual backprop) |
| `experiments.py` | partitioner sweep, schedule sweep, weight-semantics study, the PyTorch 1F1B API |
| `demo.py` | partitioner table, schedules, weight-version table (~6 seconds) |
| `test_pipedream.py` | 3 quick tests (~0.1 seconds) |

**Run it** (from `13-Systems-Inference-and-Training/085-Narayanan-et-al-2019-PipeDream`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~6 seconds
python3 experiments.py --quick
```

---

## 2. `pipedream.py`

### Partitioner
- **`partition_dp(compute, weights, activations, machines, bandwidth)`:**
  - prefix sums give the compute and weights of any layer range;
  - **T(i, j, m)** = max(compute, ring all-reduce time 2(m−1)/m · weights / bandwidth) / m;
  - **A(j, m)** (memoised): either one stage with all m replicas, or the best split at i with m′ machines for the last stage, taking the max of the left part's time, the boundary activation transfer 2·act[i]/bandwidth, and the last stage's T;
  - returns the time and a list of (first layer, last layer, replicas).
- **`config_string`** gives e.g. "7-1"; **`noam`** gives ⌈total machines / input-stage machines⌉.

### Schedules
- **`simulate_1f1b(stage_time, n_mb)`:**
  - each stage's operation list is (S − k) warm-up forwards, then alternating backward/forward, then the remaining backwards;
  - a forward waits for the previous stage's forward of that minibatch, a backward for the next stage's backward (or its own forward at the last stage);
  - returns the makespan and each stage's peak number of minibatches forwarded but not yet backwarded.
- **`simulate_gpipe`:** all forwards, a flush, then all backwards in reverse.

### Weight semantics
| Name | What it does |
|---|---|
| `make_problem` | a teacher MLP generates regression data |
| `init_stages` | one weight matrix per stage |
| `forward(Ws, x)` | tanh stages with a linear output; returns every stage's input and the output |
| `backward(Ws_bwd, hs, y)` | back-propagates the squared error using **any** weights, with the forward activations from possibly different weights |

- **`train_pipelined(mode, n_stages, lr)`:** keeps a short history of weight versions.
  - Stage k's forward reads the version from S−1−k updates ago (0-indexed k).
  - **"naive"** back-propagates with the latest weights;
  - **"stash"** back-propagates with the same versions as the forward;
  - **"vsync"** uses the version from S−1 updates ago everywhere;
  - **"sgd"** uses the latest weights everywhere.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | random profiles × machines × bandwidth: chosen configuration, time vs data parallelism, DP runtime |
| `e2` | 1F1B vs flush-every-S for stages × minibatches × imbalance: makespan and peak in flight |
| `e3` | 4 semantics × stages {2 … 16} × lr × 10 seeds: median final loss and divergence counts |
| `e4` | checks `torch.distributed.pipelining` exposes `Schedule1F1B` and `ScheduleGPipe` |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_partitioner_finds_hybrid_layout` | "7-1" with NOAM 2 on a slow network; it covers all layers and machines; pure data parallel ("8") on a fast network |
| `test_1f1b_schedule_and_memory` | peak in-flight [4, 3, 2, 1]; faster than flushing every 4; equal to one flush window |
| `test_manual_backprop_and_weight_semantics` | manual backprop matches finite differences; all semantics reduce the loss at a small lr; SGD is best |

---

## 5. Try it yourself

1. Add a memory limit per machine to the DP (stashed weights × NOAM). Which configurations become infeasible?
2. Implement PipeDream-2BW (two weight buffers, gradient accumulation) and compare its staleness.
3. Make the naive mode use weights from 2 × (S − k) updates ago. Does it finally break?
4. Give stages unequal compute in `simulate_1f1b`. How idle do the fast stages become?
5. Profile a real PyTorch model's layers (time, parameters, activation size) and feed them to `partition_dp`.
