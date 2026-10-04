# The code, explained simply

How the code in this folder implements ZeRO (Rajbhandari et al. 2020).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `zero.py` | memory and communication formulas, max model size, a flat-parameter MLP with manual gradients, a simulated cluster with counted collectives, Adam, and a four-mode mixed-precision data-parallel trainer |
| `experiments.py` | memory grid, cluster scaling, a step-time model, a real FSDP recipe |
| `demo.py` | 16 bytes per parameter, Figure 1, Table 1, max model sizes, the simulated job, ZeRO-R arithmetic (instant) |
| `test_zero.py` | 3 quick tests (~0.1 seconds) |

**Run it** (from `13-Systems-Inference-and-Training/086-Rajbhandari-et-al-2020-ZeRO`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `zero.py`

### Formulas
| Name | What it does |
|---|---|
| `model_state_bytes(psi, nd, stage, K=12)` | (2 + 2 + K)Ψ; 4Ψ + KΨ/N_d; 2Ψ + (2 + K)Ψ/N_d; (2 + 2 + K)Ψ/N_d |
| `comm_volume(stage)` | 2Ψ, or 3Ψ for stage 3 |
| `max_params(gpu_bytes, nd, stage)` | how many parameters fit |

### The model
- **`init_params`** returns all MLP weights as one flat float32 vector plus their shapes. **`unflatten`** cuts it back.
- **`loss_and_grad`** computes the forward pass and manual backward pass of a tanh MLP with squared error, returning a flat gradient.

### The cluster
- **`Cluster(nd)`:**
  - `shards(n)` splits indices into N_d contiguous shards;
  - `reduce_scatter` sums the ranks' vectors and gives rank r shard r, adding (N_d − 1)/N_d · n to every rank's sent count;
  - `all_gather` concatenates the pieces, with the same count;
  - `all_reduce` is both.
- **`adam_update`** updates master/m/v in place, with bias correction.

### `train(stage, nd, steps)`
1. **Setup:** rank r's fp32 master/m/v are the full vectors (stage 0) or only shard r (stages 1–3). The fp16 parameters are full on every rank (stages 0–2), or only the shard (stage 3).
2. **Each step:**
   - stage 3 first all-gathers the fp16 parameters for the forward pass;
   - every rank computes the gradient on its own mini-batch, cast to fp16 and divided by N_d;
   - stage 3 counts a second all-gather, for the backward pass.
3. **Update:**
   - **stage 0:** all-reduce, then every rank updates everything;
   - **stages 1–3:** reduce-scatter, each rank updates its shard;
   - stages 1–2 then all-gather the new fp16 parameters, while stage 3 keeps them sharded.
4. **Returns:** the final weights, persistent model-state bytes per rank (by formula, from the actual shard sizes), the peak held during a step, the elements sent per step in units of Ψ, and the losses.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | model size × N_d × K × stage → GB per GPU, and whether it fits on 16/32/80 GB |
| `e2` | the simulated cluster for N_d = 2 … 32: bytes per parameter vs formula, sent volume, identical weights |
| `e3` | step time = compute + stage communication / bandwidth for N_d and bandwidth |
| `e4` | a recipe: FSDP FULL_SHARD / SHARD_GRAD_OP vs DDP under torchrun |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_memory_formulas_reproduce_figure1_and_table1` | 120 / 31.4 / 16.6 / 1.9 GB; 7,000 GB and 15.6 GB for 1T; 1.5× communication |
| `test_collectives_and_gradient` | all-reduce correctness and sent volume 2(N_d−1)/N_d · n; the manual gradient matches a finite difference |
| `test_all_stages_identical_weights_and_less_memory` | bit-identical weights across stages; memory strictly decreasing; stage-3 communication exactly 1.5× DP's |

---

## 5. Try it yourself

1. Make stage 3 gather parameters layer by layer and free them, and measure the true transient peak.
2. Use plain SGD with momentum (K = 4). How much smaller are the savings?
3. Add CPU offload of the optimiser shard (ZeRO-Offload), and count PCIe traffic as a second channel.
4. Simulate bucketed gradient reduce-scatter during the backward pass and measure peak gradient memory for P_os+g.
5. Combine with 2-way model parallelism: shard each layer's weights first, then apply ZeRO across the data-parallel ranks.
