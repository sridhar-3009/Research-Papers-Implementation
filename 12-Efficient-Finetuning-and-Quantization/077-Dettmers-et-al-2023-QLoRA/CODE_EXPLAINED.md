# The code, explained simply

How the code in this folder implements QLoRA (Dettmers et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses paper 073's LoRA layer and tiny LLaMA through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `qlora.py` | NF-k, Int-k and FP4/FP3 value sets; block-wise absmax quantize/dequantize; double quantization; bits per parameter; model quantization; LoRA on all layers; memory budget |
| `experiments.py` | data types × block size × weight distribution, whole-model grid, QLoRA placement / rank / base type, adapter recovery, real bitsandbytes + peft QLoRA |
| `demo.py` | NF4 construction, Gaussian comparison, double quantization, k-bit models, QLoRA table, 65B memory (~18 seconds) |
| `test_qlora.py` | 3 quick tests (~0.5 seconds) |

**Run it** (from `12-Efficient-Finetuning-and-Quantization/077-Dettmers-et-al-2023-QLoRA`):
```
python3 -m pytest -q             # ~0.5 seconds
python3 demo.py                  # ~18 seconds
python3 experiments.py --quick
```

---

## 2. `qlora.py`

### Data types
- **`nf4_values(offset, bits)`:**
  - takes N(0, 1) quantiles at evenly spaced probabilities from `offset` = 0.9677 down to 0.5: 2^(k−1) on the positive side and 2^(k−1) − 1 on the negative side;
  - adds a 0, sorts, and divides by the max.
  - With 4 bits it equals the paper's Appendix E, which a test checks.
- **`int4_values(bits)`:** −(2^(k−1)−1) … 2^(k−1)−1, scaled to [−1, 1].
- **`fp4_values(exp_bits, man_bits, bias)`:** every sign × (1 + m/2^man) × 2^(e−bias) (subnormal when e = 0), normalised. This gives E2M1, E3M0 and E2M0.
- **`DATA_TYPES` / `DATA_TYPES_3BIT`:** the 4-bit and 3-bit sets.

### Quantizing
| Name | What it does |
|---|---|
| `quantize_blockwise(W, values, blocksize)` | reshape into blocks; c = absmax per block; code = argmin \|W/c − value\| |
| `dequantize_blockwise`, `fake_quant` | back to real numbers |
| `double_quantize(c)` | subtract the mean, then 8-bit symmetric absmax in blocks of 256 (a uniform 8-bit grid instead of the paper's FP8: same bit count), and add the mean back |
| `bits_per_parameter` | 4 + 32/64, or 4 + 8/64 + 32/(64·256) with DQ |

### QLoRA model
| Name | What it does |
|---|---|
| `quantize_model(model, values, blocksize, dq)` | replaces every block linear's weight with its de-quantized k-bit version (the frozen base) |
| `add_lora_everywhere(model, r, attention_only)` | 073's `add_lora` on wq/wk/wv/wo, plus LoRA on w1/w2/w3; or only wq/wv |
| `memory_budget_gb` | weights (16-bit or NF4 + DQ) + trained parameters × (weight + gradient + Adam) bytes + activations |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | relative MSE and code entropy for every data type, blocks 16–256, on Gaussian, Laplace and real tiny-LLaMA weights |
| `e2` | every data type × block size × DQ on whole models, 3 seeds |
| `e3` | base data type × LoRA placement (q,v / attention / all) × r, 3 seeds |
| `e4` | 3-bit base, then LoRA trained on the original tasks: is the lost accuracy recovered? |
| `e5` | `BitsAndBytesConfig(load_in_4bit, nf4, double quant, bf16 compute)` + `peft` LoRA on all linear layers vs 16-bit LoRA on Alpaca: loss, peak memory, step time (needs CUDA) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_nf4_matches_appendix_and_beats_int4_on_gaussians` | NF4 equals Appendix E (1e-6), has 16 values including an exact 0, and has lower error than Int4 and E2M1 on Gaussian weights |
| `test_blockwise_and_double_quantization` | block shapes and constants; 4.5 vs 4.127 bits per parameter; DQ error < 0.01 |
| `test_qlora_base_is_frozen_and_quantized` | only the LoRA A/B parameters train (28 tensors: 2 blocks × 7 linears × 2); each weight block lies on ≤ 16 levels |

---

## 5. Try it yourself

1. Quantize Laplace-distributed weights. Is NF4 still best, or would a Laplace-quantile type win?
2. Change the block size to 256. Which data type suffers most?
3. Use NF2 (2 bits). Can LoRA r = 32 on all layers still recover the task?
4. Remove double quantization and compare the 65B memory budget.
5. Put LoRA only on the FFN layers. How does that compare with attention only?
