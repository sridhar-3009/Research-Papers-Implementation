# The code, explained simply

How the code in this folder implements Parallelized SGD (Zinkevich et al. 2010).
Read [EXPLAINED.md](EXPLAINED.md) first.

The PDF in this folder was downloaded from the NeurIPS 2010 proceedings.

---

## 1. The files

| File | What it is |
|---|---|
| `psgd.py` | hashed sparse data, train/test split, contraction check, losses (Huber, squared, logistic), objective, SimuParallelSGD (k machines vectorised), full-batch minimiser, η*, stationary-distribution sampler |
| `experiments.py` | machines × λ × loss × η grid, theory sweeps, comparison with other parallel schemes, real RCV1 |
| `demo.py` | the algorithm, contraction, stationary distribution, the paper's experiment (~19 seconds) |
| `test_psgd.py` | 3 quick tests (~0.1 seconds) |

**Run it** (from `13-Systems-Inference-and-Training/081-Zinkevich-et-al-2010-Parallelized-SGD`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~19 seconds
python3 experiments.py --quick
```

---

## 2. `psgd.py`

### Data
- **`make_data(m, d, active, noise, seed)`:**
  - each row has `active` random positions set to 1 (feature hashing), then is normalised to unit length;
  - y = sign(w_true·x + noise);
  - `split` cuts one draw into train and test.

### Losses and the objective
- **`dloss(kind, p, y)`:**
  - squared: p − y;
  - Huber: clip(p − y, ±1);
  - logistic: −y/(1 + e^(y·p)).
- **`objective(w, X, y, λ, kind)`:** (λ/2)‖w‖² + mean loss, for one w or a stack of them.

### The algorithm
- **`simu_parallel_sgd(X, y, k, η, λ, kind, snapshots)`:**
  - a random permutation is reshaped into k rows, one per machine with T = m/k examples;
  - each machine shuffles its own order;
  - at step t, **all k machines update simultaneously**: row i of `W` is machine i's vector, with g = L′(w_i·x)·x + λw_i and W −= ηg;
  - returns the average, all machine vectors, and the average at chosen step counts.

### Theory helpers
| Name | What it does |
|---|---|
| `eta_star(X, λ)` | 1/(max‖x‖·1 + λ) (Lemma 3 with c* = 1) |
| `coupled_distance` | two chains given identical examples from different starts; returns their distance per step |
| `stationary_stats(X, y, η, λ, runs, steps)` | many independent chains sampling with replacement; the mean and spread of their final vectors |
| `full_batch_minimiser` | gradient descent on the full objective, to get min c |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | k ∈ {1 … 100} × λ × loss × η; relative objective and test RMSE after 250 / 1,000 / 2,000 examples per machine |
| `e2` | spread and suboptimality for (η, T) = (1, 2k) … (0.125, 16k); averages of 1 / 4 / 16 / 64 chains |
| `e3` | averaging exact shard solutions vs synchronous mini-batch SGD vs SimuParallelSGD: suboptimality and communication rounds |
| `e4` | RCV1 (CCAT vs rest), sparse logistic SGD per machine, test error for k machines |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_losses_and_contraction` | loss derivatives; unit rows; η ≤ η*; the coupled distance shrinks by at least (1 − ηλ) at every step |
| `test_simu_parallel_sgd_partition_and_average` | the shapes of the machine stack; the output is their mean |
| `test_averaging_cuts_variance` | averaging 10 chains cuts the spread by more than 5× and lowers the objective |

---

## 5. Try it yourself

1. Average every 100 steps instead of once (local SGD). How much does the result improve per communication round?
2. Use η = 1.5 > η*. Does the contraction (and the method) break?
3. Give each machine a non-random, biased shard (sorted by label). Does averaging still work?
4. Replace the final average with the median of each coordinate. Better or worse?
5. Compare one machine's averaged iterates (Polyak averaging) with averaging across machines.
