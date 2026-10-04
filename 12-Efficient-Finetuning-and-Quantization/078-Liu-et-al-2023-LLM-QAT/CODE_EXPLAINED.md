# The code, explained simply

How the code in this folder implements LLM-QAT (Liu et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses the LLaMA architecture of paper 056 through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `llmqat.py` | the toy language, whole-sequence LM pre-training, accuracy, the STE fake quantizer, `QLinear` with per-channel W / per-token A / per-token KV, model quantization, generation (top-1 / sample / hybrid), diversity, QAT with logits or hard-label distillation |
| `experiments.py` | W-A-KV grid, data sources, losses and temperature, MinMax vs clipping, data amount, real OPT-125M data-free QAT |
| `demo.py` | STE example, generation modes, PTQ grid, QAT at 3-8-3 and 2-8-8 (~27 seconds) |
| `test_llmqat.py` | 3 quick tests (~1 second) |

**Run it** (from `12-Efficient-Finetuning-and-Quantization/078-Liu-et-al-2023-LLM-QAT`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~27 seconds
python3 experiments.py --quick
```

---

## 2. `llmqat.py`

### The toy language
| Name | What it does |
|---|---|
| `make_seq`, `real_batch` | `<BOS>(15) task(10–12) x1..x6 SEP(14) y1..y6`, 15 tokens |
| `new_model` | 056's LLaMA (d = 64, 2 layers, 4 heads, context 16) |
| `lm_loss`, `pretrain` | next-token loss on **every** position, so the model can generate whole sequences |
| `accuracy`, `eval_loss` | answer exact match on a fixed set; LM loss |

### Quantization
- **`fake_quant(x, bits, dim)`:** α = max\|x\| along `dim`/(2^(bits−1)−1); round, clamp, multiply. It returns `x + (q − x).detach()`, which is the STE: q in the forward pass, identity gradient in the backward pass.
- **`QLinear`:** keeps the original weight Parameter.
  - forward = linear(fake_quant(x, a_bits, −1), fake_quant(W, w_bits, 1));
  - for wk and wv, the output is also quantized per token at kv_bits.
- **`quantize(model, w, a, kv)`:** a deep copy with every block linear wrapped. Embeddings and the head stay in full precision.

### Data and training
| Name | What it does |
|---|---|
| `generate(teacher, n, mode, k_det)` | autoregressive from `<BOS>`: argmax always, sample always, or argmax for the first `k_det` tokens and then sample |
| `diversity` | share of distinct sequences, and share that are valid task instances |
| `train_qat(student, teacher, data, loss)` | AdamW + cosine schedule. `logits`: −Σ p_T log p_S at every position. `hard`: cross-entropy with the data's next tokens. |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | 7 W-A-KV settings, PTQ vs data-free QAT, 3 seeds |
| `e2` | real, copy-only real, top-1, sampled, hybrid k ∈ {1, 2, 3, 5}; per-task accuracy |
| `e3` | sampling temperature × hard vs logits; logits + hidden-state MSE (`train_hidden`) |
| `e4` | MinMax vs percentile clipping (99.9%, 99%) by swapping `fake_quant` |
| `e5` | 300 … 10k generated sequences × 100 … 1000 steps at 2-8-8 |
| `e6` | OPT-125M: hybrid generation of 2k × 128 tokens, W4-A8 with 4-bit K/V projections, logits distillation, WikiText-2 perplexity vs PTQ |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_fake_quant_values_and_ste` | the worked example's values; the gradient is all ones (STE); per-output-channel weight scales |
| `test_quantized_model_and_generation_shapes` | only wk and wv quantize their outputs; sequence layout; generation starts at `<BOS>`; top-1 generation is deterministic; real data counts as valid |
| `test_qat_updates_through_quantized_forward` | QAT changes the student's weights (the STE lets gradients through) and leaves the teacher untouched |

---

## 5. Try it yourself

1. Make the hybrid scheme deterministic only from token 3 onward (after the task token). Does it match sampling now?
2. Distil at temperature 2 (soften both distributions). Do soft labels start to beat hard ones?
3. Quantize the embeddings and output head too. Which hurts most?
4. Train QAT on copy-only real data (the WikiText analogue) and evaluate on sort. How badly does it over-fit?
5. Lower KV bits to 2 while keeping W4. Can QAT still recover?
