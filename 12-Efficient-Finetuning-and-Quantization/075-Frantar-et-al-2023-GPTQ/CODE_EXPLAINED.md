# The code, explained simply

How the code in this folder implements GPTQ (Frantar et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses paper 073's tiny LLaMA and digit tasks through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `gptq.py` | min-max grid, quantize, RTN (with groups), damped Hessian, layer error, greedy OBQ, literal column-by-column GPTQ, Algorithm 1, layer-input capture, sequential whole-model quantization, evaluation |
| `experiments.py` | layer sweeps incl. act-order, dampening and calibration size, model width × bits × groups, runtime scaling, real OPT GPTQ |
| `demo.py` | grid example, a hand-worked compensation step, layer comparison, exactness check, speed, whole-model table, memory (~12 seconds) |
| `test_gptq.py` | 3 quick tests (~0.6 seconds) |

**Run it** (from `12-Efficient-Finetuning-and-Quantization/075-Frantar-et-al-2023-GPTQ`):
```
python3 -m pytest -q             # ~0.6 seconds
python3 demo.py                  # ~12 seconds
python3 experiments.py --quick
```

---

## 2. `gptq.py`

### Grid
| Name | What it does |
|---|---|
| `grid(w, bits)` | per row: scale = (max − min)/(2^b − 1), with the range always including 0; zero = round(−min/scale) |
| `quant(w, scale, zero, bits)` | round, clamp to [0, 2^b − 1], map back |
| `rtn(W, bits, groupsize)` | one grid per row, or per group of columns |
| `hessian(X, damp)` | 2XXᵀ + damp·mean(diag)·I |
| `layer_error` | ‖(W − Q)X‖² |

### OBQ and GPTQ
- **`obq`:** for each row, repeatedly pick the remaining weight with the smallest (quant(w) − w)²/[H⁻¹]_qq, quantize it, update the rest (Eq. 2), and shrink H⁻¹ (Eq. 3). Done in float64.
- **`gptq_naive`:** the same equations in a fixed left-to-right order for all rows, with one H⁻¹ update per column.
- **`gptq(W, H, bits, blocksize, groupsize)`:** Algorithm 1.
  - The upper Cholesky factor of H⁻¹ supplies every row needed.
  - Inside a block: quantize column j, compute E[:, j] = (W_j − Q_j)/U_jj, and update the block's later columns.
  - After the block: W[:, after] −= E · U[block, after].
  - With grouping, the grid is recomputed from the current (compensated) weights at each group start.

### Whole model
- **`GROUPS`:** the linear layers in forward order. Wq, Wk and Wv share one input; W1 and W3 share another.
- **`layer_inputs`:** a forward hook captures one layer's input on the calibration batch. The hook must **return None**; returning the tensor would replace the layer's output (a bug we hit).
- **`quantize_model`:** for each block and group: capture inputs from the current, partly quantized model; build H; quantize each weight with RTN or GPTQ.
- **`model_loss`, `task_accuracy`, `calibration_batch`:** evaluation helpers.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | RTN / OBQ / GPTQ / act-order × bits × input correlation × width, 5 seeds |
| `e2` | dampening 0 … 10% (0 may break Cholesky); 8 … 512 calibration sequences |
| `e3` | models of width 32/64/128 × {4, 3, 2} bits × groups {8, 16, 32}, RTN vs GPTQ |
| `e4` | runtime of OBQ vs GPTQ for d = 32 … 512 |
| `e5` | OPT-125M/350M, Hessians accumulated over 128 × 2,048-token calibration segments, 4/3-bit, per row and g128, WikiText-2 perplexity |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_grid_and_rtn` | the scale/zero of the worked example; outputs lie on the grid; RTN error ≤ scale/2 |
| `test_gptq_equals_literal_update_and_beats_rtn` | Algorithm 1 with blocks of 1 or 5 equals the literal update; GPTQ error < 0.8 × RTN error |
| `test_obq_two_weight_example_and_order` | the hand example's compensation lowers w₂; fixed order is within 1.5× of greedy OBQ |

---

## 5. Try it yourself

1. Set dampening to 0 in `hessian` on a layer with nearly collinear inputs. When does the Cholesky step fail?
2. Quantize columns in order of decreasing diag(H) ("act-order"). Does 2-bit improve?
3. Use uncorrelated inputs (X = random). How much does GPTQ gain over RTN now, and why?
4. Calibrate on only "copy" sequences and evaluate on "sort". Does GPTQ overfit its calibration data?
5. Quantize the output head too. Which layer is most sensitive?
