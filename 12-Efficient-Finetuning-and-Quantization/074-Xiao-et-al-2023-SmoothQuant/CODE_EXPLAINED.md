# The code, explained simply

How the code in this folder implements SmoothQuant (Xiao et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses paper 073's `lora.py` (the tiny LLaMA of 056, the digit tasks and pre-training) through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `smoothquant.py` | quantizer, effective levels, `QuantLinear`, quantized model copies (incl. static calibration), activation statistics, outlier injection, smoothing, layer error, task accuracy, calibration data |
| `experiments.py` | granularity × bits, α × bits × outlier strength, calibration size, where to smooth, real OPT-1.3B perplexity |
| `demo.py` | quantizer example, outlier model, W8A8 table, exactness of smoothing, α sweep (~15 seconds) |
| `test_smoothquant.py` | 3 quick tests (~0.5 seconds) |

**Run it** (from `12-Efficient-Finetuning-and-Quantization/074-Xiao-et-al-2023-SmoothQuant`):
```
python3 -m pytest -q             # ~0.5 seconds
python3 demo.py                  # ~15 seconds
python3 experiments.py --quick
```

---

## 2. `smoothquant.py`

### Quantizing
| Name | What it does |
|---|---|
| `quantize(x, bits, dim, scale)` | symmetric fake quantization: Δ = max\|x\|/(2^(bits−1)−1) over the whole tensor (`dim=None`), per row (`-1`) or per column (`0`), or a given static `scale`; round, clamp, multiply back |
| `effective_levels(X)` | 2^bits · max\|X_j\|/max\|X\| for each channel |
| `QuantLinear` | quantizes the weight once (per-tensor or per-output-channel) and the input on each call: per-tensor / per-token / per-channel dynamic, static with a calibrated scale, or the LLM.int8()-style split (channels whose max exceeds `outlier` stay unquantized) |
| `quantized_copy(model, calib, **kw)` | wraps all 7 linear layers of every block; `a_gran="static"` first records each layer's max \|input\| on `calib` |

### Calibration and smoothing
| Name | What it does |
|---|---|
| `record_input_absmax` | forward hooks record max \|input\| per channel (or per tensor) |
| `inject_outliers(model, rng)` | multiplies 3 RMSNorm gain entries by 150 in both norms of every block, divides the matching columns of the reading layers (Wq, Wk, Wv or W1, W3) by √150, then re-trains 300 steps |
| `smooth(model, calib, alpha)` | for each norm → linears group: s = max\|X\|^α / (max over the group's weights of max\|W_j\|)^(1−α); divides the RMSNorm gain by s and multiplies the weight columns by s |
| `layer_error`, `task_accuracy`, `calibration_batch` | evaluation helpers |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | W8A8 / W6A6 / W4A8 × weight per-tensor or per-channel × 4 activation modes × plain or smoothed, 3 seeds |
| `e2` | α ∈ {0, 0.1, …, 1} × bits {8, 6, 5} × outlier factor {30, 150, 500} |
| `e3` | 4 … 512 calibration sequences |
| `e4` | smoothing only attention inputs, only FFN inputs, or both |
| `e5` | OPT-1.3B: LayerNorm-folded smoothing (gain and bias), static W8A8 fake quantization of all decoder linears, WikiText-2 perplexity, outlier ratio per layer |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_quantizer_and_levels` | rounding error ≤ Δ/2; static scale = plain rounding; effective levels; per-row scales keep small rows |
| `test_smoothing_is_exact_and_flattens_activations` | smoothed and original models give the same outputs; the max/min channel ratio shrinks more than 10× |
| `test_quant_linear_modes` | every activation mode gives the right shape and a close output; no quantization is exact |

---

## 5. Try it yourself

1. Use a stronger outlier factor (500) in `inject_outliers`. Does the best α move toward 0.75, as for GLM-130B?
2. Quantize weights per-output-channel instead of per-tensor. Which α is best now?
3. Calibrate on only "copy" sequences and test on all three tasks. Do static scales still work?
4. Smooth only `ffn_norm`. Which layers still break?
5. Push to W4A4. Is any α enough?
