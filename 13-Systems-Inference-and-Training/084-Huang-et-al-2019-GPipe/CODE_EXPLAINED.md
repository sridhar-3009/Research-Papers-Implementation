# The code, explained simply

How the code in this folder implements GPipe (Huang et al. 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `gpipe.py` | schedule simulator, bubble formula, pipeline time and normalised throughput, balanced partitioner, a real PyTorch pipelined step with optional re-materialisation, reference gradients |
| `experiments.py` | throughput grid, optimal vs heuristic partition, micro-batch batch-norm statistics, real checkpoint memory, the PyTorch pipelining API |
| `demo.py` | ASCII schedule and bubble table, the real pipeline's exact gradients and memory, Table 2 analogue (instant) |
| `test_gpipe.py` | 3 quick tests (~0.5 seconds) |

**Run it** (from `13-Systems-Inference-and-Training/084-Huang-et-al-2019-GPipe`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `gpipe.py`

### Simulation
- **`simulate(stage_fwd, M, backward_ratio, remat)`:**
  - **Forward phase:** stage k starts micro-batch m when it is free and stage k − 1 has finished m.
  - **Flush:** all stages wait for the last forward.
  - **Backward phase:** micro-batches M−1 … 0, stage k after stage k + 1 has finished that micro-batch. A backward costs `backward_ratio` × forward, plus one forward if re-materialising.
  - Returns the total time, busy time per stage, and the timeline.
- **`bubble_fraction(K, M)`:** (K − 1)/(M + K − 1).
- **`partition(costs, K)`:** cuts where the cumulative cost is closest to j/K of the total, with at least one layer per stage.
- **`pipeline_time(costs, K, M)`:** stage forward time = (sum of the stage's layer costs)/M per micro-batch, then simulate.
- **`normalised_throughput`:** time(K = 2, M = 1)/time(K, M), the same mini-batch either way.

### The real pipeline
- **`make_model`:** L × (Linear + Tanh) then Linear → 1. **`split_stages`:** consecutive slices into K `nn.Sequential` stages.
- **`gpipe_step(stages, x, y, M, remat)`:**
  - **Forward phase:** for each micro-batch, pass through all stages, storing `boundary[k][m]`, the input of stage k.
    - With remat, the forward runs under `no_grad` and only the boundary inputs are counted as kept.
    - Without remat, the autograd graph (input and output) is kept, and the stage's internal activations are counted too.
  - **Backward phase:** micro-batches in reverse. For each stage, from last to first:
    - recompute the forward (remat), or reuse the graph;
    - call `backward` with the gradient from the next stage, or on the micro-batch loss (sum/N) at the last stage;
    - pass the input's `.grad` to the previous stage.
  - Gradients accumulate in the shared parameters.
- **`reference_grads`:** ordinary full-batch loss and gradients.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | speed-up vs 1 device and idle fraction over K × M × backward ratio × remat, for balanced and heavy-tailed layers |
| `e2` | the slowest-stage load of our partition vs the optimal (`optimal_partition`, min-max DP) vs the ideal total/K |
| `e3` | mean absolute difference between per-micro-batch and full-batch normalisation as M grows |
| `e4` | `checkpoint_sequential` vs plain autograd peak CUDA memory |
| `e5` | checks `torch.distributed.pipelining.ScheduleGPipe` is importable (a real run needs torchrun) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_schedule_bubble_matches_formula` | simulated idle fraction = (K − 1)/(M + K − 1) for several (K, M) |
| `test_partition_and_throughput` | even partitions; more stages are faster at M = 32; M = 1 gives no speedup; a layer holding half the work caps the speedup |
| `test_pipeline_gradients_equal_full_batch_and_remat_saves_memory` | pipeline gradients = full-batch gradients with and without remat; remat keeps < 1/3 of the activations |

---

## 5. Try it yourself

1. Implement the 1F1B schedule (interleave one forward and one backward). Does the bubble change? Does memory?
2. Partition AmoebaNet-like costs with `optimal_partition` and recompute Table 2.
3. Add a BatchNorm layer to the real model. How far do the gradients now differ from the full batch as M grows?
4. Make the backward pass 3× the forward. Does M ≥ 4K still make the bubble small?
5. Add communication time between stages in `simulate`. When does it matter?
