# The code, explained simply

How the code in this folder implements Glorot & Bengio (2010).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is | Paper part |
|---|---|---|
| `glorot.py` | Both initializations, 3 activations, the deep net, the measurement tools, the Shapeset generator | Sections 2–4 |
| `experiments.py` | Reproduces Figures 2–3, 5–8, 11–12 and Table 1 | everything |
| `demo.py` | A guided tour (~1 minute) | Sections 3–4 |
| `test_glorot.py` | 9 quick tests | Eqs. 1, 12, 15–17; Figs. 6–8 |

**Run it** (from `02-Training-Deep-Networks/007-Glorot-Bengio-2010-Difficulty-Training-Deep-FF`):
```
python3 -m pytest -q                 # 9 tests, a few seconds
python3 demo.py                      # about 1 minute
python3 experiments.py --quick       # a few minutes, tiny version
python3 experiments.py               # FULL: 30-40 minutes of heavy CPU. Only on a strong machine.
```
Needs `torch`, `torchvision` (for MNIST, downloaded into the shared `data/` folder) and `matplotlib`.

**Why PyTorch here?** The network is 5 layers of 1,000 units, which is too slow in plain NumPy. PyTorch only computes the gradients. The **initializations, activations and every measurement** are written out by hand in `glorot.py`.

---

## 2. `glorot.py`

### Initializations
```python
standard_init:   a = 1/√n_in                  → U[-a, a]   (Eq. 1),  n·Var = 1/3
normalized_init: a = √6/√(n_in + n_out)       → U[-a, a]   (Eq. 16), Var = 2/(n_in+n_out)
```
`torch.rand(...) * 2 - 1` gives U[−1, 1]; multiplying by `a` gives U[−a, a].

### `DeepNet`
```python
DeepNet(n_in=1024, n_out=9, hidden=1000, depth=5, act="tanh", init="normalized")
```
- The weights `W[i]` have shape (fan_in, fan_out), and the biases start at **0** (Section 2.3).
- `forward(x, keep=True)` also returns every layer's **pre-activation s** and **activation z**, using the paper's notation from Section 4:
  ```
  s^i = z^i W^i + b^i,     z^{i+1} = f(s^i)
  ```
  With `keep=True`, `s.retain_grad()` makes PyTorch **keep** ∂Cost/∂s^i after `backward()`. That's the back-propagated gradient the paper plots in Figure 7.

### Measurement tools
| Function | Measures | Figure |
|---|---|---|
| `activation_stats` | per layer: mean, std, 98th percentile of \|z\| | 2, 3, 10 |
| `gradient_stats` | per layer: std of z, of ∂Cost/∂s, of ∂Cost/∂W | 6, 7, 8 |
| `jacobian_singular_values` | the average singular value of J = W · diag(f′(s)) | Eq. 17 |

**The Jacobian, simply:** z^{i+1} = f(z^i W + b), so a small change dz in the input gives a change dz · W · diag(f′(s)) in the output. The singular values of `W * f′(s)` say how much each direction gets stretched. The code computes them with `torch.linalg.svdvals` for 20 examples and averages.

### `shapeset(n)`: the Shapeset-3×2 generator
1. Choose a class: one shape (3 classes) or an unordered pair (6 classes), so **9 classes** in all.
2. For each shape, pick a random centre, size and rotation.
   - **Triangle** and **parallelogram**: a convex polygon. A pixel is inside if it's on the inner side of **every edge** (a cross-product sign test).
   - **Ellipse**: (u/a)² + (v/b)² ≤ 1 in the rotated frame.
3. Resample the second shape until it covers **at most 50%** of the first (as the paper requires).
4. Give each shape a random grey level.

---

## 3. `experiments.py`

| Function | What it does |
|---|---|
| `e1_initialization` | Figs. 6–8 and Eq. 17 for tanh, softsign, sigmoid × standard/normalized |
| `e2_cost_surface` | Fig. 5: cross-entropy vs quadratic cost over a grid of two weights; counts "plateau" points |
| `e3_e4` | trains all 5 settings of Table 1 on Shapeset (online) and MNIST; learning rate from {0.01, 0.03, 0.1} chosen on validation; records activation stats during training (Figs. 2–3) |
| `report`, `figures` | write `results.md` and `figures/` |

- `--quick` makes everything 10× smaller.
- `--report-only` rebuilds the report from a saved `results.json`.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_standard_init_variance` | n·Var[W] = 1/3 (Eq. 15) |
| `test_normalized_init_variance` | Var[W] = 2/(n_in+n_out) (Eqs. 12, 16) |
| `test_standard_init_shrinks_...` | activations fade going up and gradients fade going down (Figs. 6–7, top) |
| `test_normalized_init_keeps_gradients_level` | both stay level (Figs. 6–7, bottom) |
| `test_weight_gradients_level_even_with_standard_init` | Fig. 8's surprise |
| `test_jacobian_singular_values_match_the_paper` | ≈ 0.5 and ≈ 0.8 (Section 4.2.2) |
| `test_network_shapes_and_biases` | 5 × 1000 layers, zero biases |
| `test_shapeset_...` (×2) | 9 balanced classes; two-object images have more ink |

---

## 5. Try it yourself

1. Add **He initialization** (Var = 2/n_in, designed for ReLU) and a ReLU activation. Measure the Jacobian.
2. Make the net 10 layers deep. How small do the standard-init gradients get at layer 1?
3. Use `activation_stats` inside a sigmoid training loop and plot the top layer's mean. Do you see Figure 2?
4. Try normalized init with **sigmoid**. Does it fix the top-layer saturation?
