# The code, explained simply

How the code in this folder implements the analysis of Pope et al. (2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `inference.py` | TPU v4 constants, PaLM configs, KV bytes, max context (Table 1), communication formulas, roofline step time, a virtual chip `Mesh` with counted collectives, and the FFN in 1D WS / 2D WS / weight-gathered layouts |
| `experiments.py` | max-context grid, simulated communication grid, Pareto frontiers, real prefill/decode measurements |
| `demo.py` | KV-cache arithmetic and Table 1, chip simulation, roofline vs Table 2, latency vs cost (~7 seconds) |
| `test_inference.py` | 3 quick tests (~0.1 seconds) |

**Run it** (from `13-Systems-Inference-and-Training/080-Pope-et-al-2022-Efficiently-Scaling-Transformer-Inference`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~7 seconds
python3 experiments.py --quick
```

---

## 2. `inference.py`

### Analytical model
| Name | What it does |
|---|---|
| `kv_bytes_per_token(layers, heads, d_head, kv_heads)` | 2 × layers × kv_heads × d_head × 2 bytes |
| `max_context(variant, batch, chips, fraction)` | per-chip KV budget = fraction × 32 GiB. Multihead: ⌈H/chips⌉ heads per chip (d_head 128). Baseline multiquery: the full single-head cache per chip. Optimized: batch/chips sequences per chip. |
| `comm_time(layout, tokens, E, F, chips)` | 2·BLE/bw, 8·BLE/(√n·bw), or 4E√(BLF)/(√n·bw), in bf16 bytes |
| `step_time(model, chips, batch, ctx, phase, layout, weight_bytes)` | compute = 2N·tokens/(chips·275T); memory = (weights/chips + batch-sharded KV/chips)/1.2 TB/s (KV only for decode); comm = 2 × layers × comm_time; total = max(compute, memory) + comm; MFU = compute/total |

### Virtual chips
- **`Mesh(n)`:** `all_gather(shards, group)` adds (k − 1) × shard bytes to every member; `reduce_scatter(partials, group)` sums and adds (k − 1)/k × partial bytes. These are ring algorithms.
- **`ffn_1d_ws`:** inputs sharded over E are all-gathered; each chip computes gelu(x W_in[:, F_c]) W_out[F_c, :]; the partial outputs are reduce-scattered.
- **`ffn_2d_ws(x, W_in, W_out, mesh, X, YZ)`:** chip (i, j) holds W_in[E_i, F_j] and W_out[F_j, E_i].
  - All-gather x[:, E_i] over j.
  - Partial h over i, reduce-scattered over i, then gelu.
  - Gather h over i (counted).
  - Partial outputs over j, reduce-scattered over j.
- **`ffn_weight_gathered`:** each chip all-gathers the full weights ((n − 1)/n of their bytes) and processes its own slice of the batch.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | max context for 3 variants × chips × KV budget × batch |
| `e2` | bytes per chip for 1D, every 2D split and weight-gathered, × chips × tokens; the best layout per cell |
| `e3` | Pareto frontier (latency vs chip-ms per token) over chips × batch × int8/bf16 × layout, for prefill and decode of three PaLM sizes, keeping only configurations that fit |
| `e4` | GPT-2 vs tiny StarCoder (multiquery): prefill time, decode time per token, and KV bytes per token vs batch |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_table1_max_context` | within 2% of the paper's 1,320 / 165 / 43,000, and the 3 TB KV-cache claim |
| `test_partitioned_ffn_exact_and_comm_scaling` | all three layouts equal the reference FFN; 1D sends exactly 2·(n−1)/n·BLE; 2D sends less at 64 chips than at 16 |
| `test_roofline_regimes` | big-batch prefill is compute-bound, batch-1 decode memory-bound; prefill MFU > decode MFU; int8 speeds up decode |

---

## 5. Try it yourself

1. Add grouped-query attention (8 KV heads) to `max_context`. Where does it land between multihead and multiquery?
2. Overlap communication with compute (total = max(compute, memory, comm)). How close does the roofline get to Table 2?
3. Use a 4K or 32K context in `step_time` for decode. When does KV loading overtake weight loading?
4. Try a 1D split with more chips than d_ff/128. What breaks?
5. Run E3 and find the cheapest configuration with decode latency under 30 ms/token.
