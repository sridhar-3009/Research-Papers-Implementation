# The code, explained simply

How the code in this folder implements AWQ (Lin et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses paper 074 (the outlier model) and paper 075 (grid, RTN, GPTQ, layer-input capture) through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `awq.py` | layer groups with their fold targets, weight quantizer, mixed-precision baseline, fixed-s scaling, the AWQ scale search, clipping search, whole-model AWQ, RTN model, outlier model, calibration batches |
| `experiments.py` | Table 1 / Table 2 sweeps, calibration robustness, AWQ + GPTQ and act-order, grid size, real OPT-1.3B |
| `demo.py` | error example, outlier model, Tables 1–3 analogues (~16 seconds) |
| `test_awq.py` | 3 quick tests (~0.7 seconds) |

**Run it** (from `12-Efficient-Finetuning-and-Quantization/076-Lin-et-al-2023-AWQ`):
```
python3 -m pytest -q             # ~0.7 seconds
python3 demo.py                  # ~16 seconds
python3 experiments.py --quick
```

---

## 2. `awq.py`

### Groups and fold targets
`GROUPS` lists each set of linear layers that share one input, and where 1/s can be folded:

| Layers | Fold 1/s into |
|---|---|
| Wq, Wk, Wv | the attention RMSNorm gain |
| Wo | the **output rows of Wv**. Attention output is a weighted sum of V vectors, so scaling V's channels scales Wo's inputs. |
| W1, W3 | the FFN RMSNorm gain |
| W2 | the **output rows of W3**, since W2's input is silu(W1x) ⊙ (W3x), which is linear in W3x |

### Baselines
| Function | What it does |
|---|---|
| `wq(W, bits, group)` | 075's RTN with groups |
| `quantize_keep_fp(..., frac, how)` | quantize, then copy back the original columns of the top-`frac` channels chosen by mean \|activation\|, weight-column norm, or at random |
| `scale_salient(..., s)` | multiply the salient columns by s, quantize, divide by s; also counts how many group ranges (Δ) changed |

### AWQ
- **`awq_scale(W_list, X, bits, group, grid)`:** for α = 0, 1/grid, …, 1:
  - s = mean\|X\|^α, normalised so that √(max·min) = 1;
  - error = Σ over the group's layers of ‖(X/s) Q(W·s)ᵀ − X Wᵀ‖²;
  - returns the best s, its α, and the error.
- **`clip_search`:** for each column group, tries range-shrink ratios 1.0 … 0.7 and keeps, per output row, the one with the lowest output error.
- **`awq_model`:** block by block and group by group, in forward order:
  - capture inputs from the current (partly quantized) model;
  - search s;
  - fold 1/s into the norm gain or the previous layer's rows;
  - multiply the weights by s (optionally clip), then quantize.
- **`rtn_model`, `outlier_model`, `task_batch`:** helpers.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | 1–10% of channels kept by activation / weight / random × INT3/4 × groups 16/32, 3 seeds |
| `e2` | s ∈ {1 … 16} |
| `e3` | AWQ vs GPTQ calibrated on all tasks vs copy only, and on 3 / 12 / 126 sequences, for the outlier and plain models |
| `e4` | RTN, AWQ, GPTQ, GPTQ act-order (`gptq_act_order`), AWQ + GPTQ (`awq_then_gptq`) at INT2/3/4 |
| `e5` | grid size 5 … 50 and the α chosen per group |
| `e6` | OPT-1.3B: scales folded into LayerNorms and v_proj; fc2 not scaled (ReLU in between); RTN vs AWQ at INT3/INT4-g128; WikiText-2 perplexity |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_scaling_salient_weight_reduces_its_error` | scaling the big-activation weight by 4 cuts the output error more than 3× |
| `test_awq_search_prefers_activation_aware_scaling` | on a layer with two big, small-weight channels the search picks α > 0, beats plain rounding, scales those channels most, and is exact before quantization |
| `test_clipping_never_increases_error` | clipping search with ratio 1.0 included never makes the output error worse |

---

## 5. Try it yourself

1. Use the plain model (no outliers). Does AWQ still beat RTN at INT3, and which α does it choose?
2. Replace s_X = mean\|x\| with max\|x\|. Better or worse?
3. Search α separately for each layer in a group (Wq, Wk and Wv each). Does it help?
4. Calibrate on only 3 sequences. How much does AWQ change compared with GPTQ?
5. Combine AWQ with GPTQ (E4) at INT2. Does the combination beat both?
