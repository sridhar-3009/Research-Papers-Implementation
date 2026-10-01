# The code, explained simply

How the code in this folder implements multi-query attention (Shazeer 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `mqa.py` | the paper's einsum functions (batched and incremental, multi-head and multi-query), the cost model, parameter matching, and a decoder-only LM with grouped-query attention, a KV cache and an optional local window |
| `experiments.py` | E1–E3: parameter-matched variants on PTB, Table 2 speeds, decoding time vs batch and length (heavy, not run here) |
| `demo.py` | the KV-cache sizes, the memory/compute ratios, the FFN widths, CPU decoding speed, quality on a copy task (~9 seconds) |
| `test_mqa.py` | 9 quick tests (~1 second) |

**Run it** (from `06-Transformers/035-Shazeer-2019-Multi-Query-Attention`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~9 seconds
python3 experiments.py --quick   # ~30-60 minutes
```

---

## 2. `mqa.py`

### The paper's functions (Sections 2.3–3), written with `torch.einsum` exactly as printed
| Function | What it does |
|---|---|
| `multihead_attention_batched(X, M, mask, P_q, P_k, P_v, P_o)` | P_q, P_k: [h, d, k]; P_v, P_o: [h, d, v] |
| `multiquery_attention_batched(...)` | the same, but P_k: [d, k] and P_v: [d, v] (no heads) |
| `multihead_self_attention_incremental(x, prev_K, prev_V, ...)` | one step; the cache is [b, h, m, k] |
| `multiquery_self_attention_incremental(...)` | one step; the cache is [b, m, k] |

### The cost model
| Function | What it computes |
|---|---|
| `incremental_ratio(n, d, h, b, kind)` | n/d + 1/b (MHA) or 1/d + n/(dh) + 1/b (MQA) |
| `kv_cache_numbers(b, n, h, k, layers, kv_heads)` | 2·layers·b·n·kv_heads·k |
| `attention_params(d, h, k, v, kv_heads)` | the parameters of P_q, P_k, P_v, P_o |
| `matching_ffn_width(d, h, k, d_ff, n_attention_layers, n_ffn_layers)` | the widened FFN (gives 5440 and 9088) |

### The model
- **`GroupedAttention(d, h, kv_heads, window)`:**
  - h query heads and g = kv_heads key/value heads;
  - `_attend` reshapes the queries into g groups of h/g, so each group reads its own shared K and V;
  - `forward(x, cache)` appends the new keys and values to the cache, builds the causal (and optional window) mask, and returns the output and the new cache.
- **`Block`:** a pre-LN block with attention and an FFN.
- **`DecoderLM(vocab, d, h, kv_heads, d_ff, layers, max_len, window)`:**
  - learned positions and tied embeddings;
  - `forward(x, caches, start)`;
  - `generate(prompt, steps, incremental)`: greedy decoding, with the KV cache (one new position per step) or without it (recompute everything);
  - `cache_numbers(b, n)`.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Tables 1/3: multi-head, multi-query, grouped, and the paper's "simpler alternatives" (h, d_k) = (1, k), (2, k/2), (4, k/4), (8, k/8), each with a widened FFN so the parameter counts match; PTB dev perplexity |
| `e2` | Table 2: training µs/token and incremental decoding µs/token (batch 128, 128 tokens), full and local (window 32) |
| `e3` | decoding µs/token for batch 1/16/128 × length 32/128/256 |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_multi_head_einsum_equals_a_loop_over_heads` | the batched einsum |
| `test_multi_query_is_multi_head_with_all_key_and_value_heads_tied` | the central identity |
| `test_incremental_steps_equal_the_batched_causal_computation` | the KV-cache code; the MQA cache has one head |
| `test_widened_feed_forward_matches_the_paper` | 5440 and 9088 exactly; savings of 2dk(h − 1) |
| `test_grouped_attention_covers_mha_and_mqa` | g = 4, 2, 1 against the paper's einsum |
| `test_kv_cache_generation_equals_recomputation_and_is_h_times_smaller_for_mqa` | caching is exact; the cache ratio is h |
| `test_local_attention_window` | the window sees exactly w positions |
| `test_memory_to_compute_ratio` | the formulas |
| `test_multi_query_model_still_learns` | an MQA model learns a copy task |

---

## 5. Try it yourself

1. Time `generate` with batch 1 vs 256 for multi-head and multi-query. Where does multi-query help most?
2. Change `kv_heads` to 2 and 4 (grouped-query). How do speed and the copy-task loss change?
3. Set `window=8` and test the copy task with a gap of 12. What breaks, and why?
4. Count the KV-cache bytes of a LLaMA-2-70B-like model (80 layers, 64 query heads, 8 KV heads, k = 128) for a 4096-token context.
