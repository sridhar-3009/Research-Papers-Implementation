# The code, explained simply

How the code in this folder implements LLaMA (Touvron et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `llama.py` | RMSNorm, SwiGLU (with the 256-rounded hidden size), RoPE, memory-efficient causal attention (online softmax, future blocks skipped), the LLaMA model, the Table 2 parameter formula, training-time and carbon arithmetic, Tables 1/2/3/9/15, digit splitting, byte fallback, the LR schedule, normalised answer scoring |
| `experiments.py` | architecture ablation with seeds, over-training curves, digit splitting on arithmetic, attention time/memory, the inference-budget analysis (heavy, not run here) |
| `demo.py` | each architecture change by hand, efficient attention, sizes/days/carbon, the inference-budget argument (with paper 052's law), a GPT-2-vs-LLaMA race on this repository's text, the paper's numbers (~17 seconds) |
| `test_llama.py` | 9 quick tests (~0.6 seconds) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/056-Touvron-et-al-2023-LLaMA`):
```
python3 -m pytest -q             # ~0.6 seconds
python3 demo.py                  # ~17 seconds
python3 experiments.py --quick
```

---

## 2. `llama.py`

### Layers
| Name | What it is |
|---|---|
| `RMSNorm(d)` | x · rsqrt(mean(x²) + ε) · g |
| `ffn_hidden(d)` | int(8d/3) rounded up to a multiple of 256 |
| `SwiGLU(d, hidden)` | w2(silu(w1 x) · w3 x), no biases |
| `rope_frequencies(head_dim, length)` | cos and sin of m·θᵢ, θᵢ = 10000^(−2i/head_dim) |
| `apply_rope(x, cos, sin)` | rotates the pairs (x₀, x₁), (x₂, x₃), … by their angles |
| `memory_efficient_causal_attention(q, k, v, block)` | blockwise with a running max / normaliser / accumulator; never builds T × T; loops only over key blocks ≤ the query block |
| `Attention` | bias-free q, k, v, o projections; RoPE on q and k; standard or memory-efficient kernel |
| `Block` | x + attn(RMSNorm(x)); x + SwiGLU(RMSNorm(x)) |
| `LLaMA(vocab, d, layers, heads, ctx, hidden, efficient)` | embedding, blocks, final RMSNorm, **untied** output layer; `loss(ids)` is next-token cross-entropy |

### Counting and bookkeeping
- **`param_count(d, layers, vocab)`:** 2Vd + layers·(4d² + 3d·h + 2d) + d.
- **`TABLE_2`:** the sizes, learning rates and token counts.
- **`training_days(tokens)`:** tokens / (380 × 2048) / 86400.
- **`carbon(gpu_hours)`:** MWh = GPU-h × 0.4 kW × 1.1; tCO₂ = MWh × 0.385. `TABLE_15_GPU_HOURS` holds the paper's GPU-hours.
- **Data tables:** `TABLE_1` (data mix), `COMMON_SENSE` (Table 3), `MMLU` (Tables 9–10).

### Small helpers
- **`split_digits(text)`:** every digit becomes its own piece.
- **`byte_fallback(text, vocab)`:** unknown characters become <0xNN> tokens.
- **`lr_schedule(step, peak)`:** linear warm-up over 2000 steps, then cosine decay to 10% of the peak.
- **`normalised_choice_score`:** log P(answer | context) − log P(answer | "Answer:").

---

## 3. `experiments.py`

**Corpus:** OpenWebText if `datasets` is available, else Gutenberg, else this repository's markdown. Tokens come from paper 049's `ByteBPE`.

**Pieces:**
- **`Variant(V, d, layers, heads, ctx, rms, swiglu, rope)`:** a decoder in which each of the three changes can be switched on independently. Off means LayerNorm, a GELU MLP with hidden 4d, and learned positions.
- **`train`:** AdamW (0.9, 0.95), weight decay 0.1, clipping, LLaMA's warm-up + cosine-to-10% schedule. It optionally records a validation curve against tokens seen.

| Function | Reproduces |
|---|---|
| `e1` | baseline, + RMSNorm, + SwiGLU, + RoPE, all three; 3 seeds each |
| `e2` | 4 sizes trained to 500 tokens/param; loss at 20 tokens/param vs at the end; fit E + B/D^β |
| `e3` | the addition corpus with BPE that merges digits vs digit splitting; exact-match accuracy |
| `e4` | time (and CUDA peak memory) of naive vs blockwise attention for T = 512 … 8192 |
| `e5` | from E2's curves: the cheapest (N, D) to train vs the cheapest including 10⁹ … 10¹³ served tokens |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_rmsnorm` | (3, −4, 0, 0) → x/2.5; unit RMS output |
| `test_swiglu_hidden_size_and_parameter_parity` | 11008/13824/17920/22016; ⅔·4d gives the same size as a 4d MLP; the formula |
| `test_rope_is_a_rotation_and_depends_only_on_relative_position` | norm preserved; equal offsets give equal scores; no change at position 0 |
| `test_memory_efficient_attention_matches_standard` | equal to SDPA (T = 150, blocks of 32); equal model outputs |
| `test_model_is_causal_and_count_formula_matches` | the built model's count = the formula; future tokens don't change earlier logits |
| `test_table_2_parameter_counts` | within 0.6% of 6.7/13.0/32.5/65.2B; 128-dim heads |
| `test_training_time_and_carbon` | ≈ 21 days; Table 15's tCO₂ values; data mix sums to 1 |
| `test_tokenizer_rules_and_schedule` | digit splitting, byte fallback, warm-up, 10% floor |
| `test_13b_beats_gpt3_on_most_common_sense_tasks` | 5 of 7 |

---

## 5. Try it yourself

1. In `demo.py` section 7, give the GPT-2 model an untied output layer, or the LLaMA model a tied one. How much of the gap remains?
2. Run the race with 3 seeds and 1000 steps. Is the gap stable?
3. Compute θᵢ for head_dim = 128. After how many positions does the slowest pair complete one full turn (2π)?
4. Change `block` in `memory_efficient_causal_attention` to 16 and 512. Is the output identical? How many key blocks are computed for T = 1024?
5. Use `param_count` to design a "LLaMA-1B": choose d and layers with 128-dim heads to land within 5% of 1.0B parameters.
