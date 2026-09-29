# The code, explained simply

How the code in this folder implements Srivastava et al. (2014).
Read [EXPLAINED.md](EXPLAINED.md) first, and Paper 009's code for the basics.

---

## 1. The files

| File | What it is |
|---|---|
| `dropout.py` | ReLU dropout net (Bernoulli or Gaussian), max-norm, a trainer with L1/L2/max-norm, Monte-Carlo averaging, the autoencoder, dropout linear regression |
| `experiments.py` | Table 9, Figures 7–11, Table 10, Section 9.1 |
| `demo.py` | A light tour (seconds) |
| `test_dropout.py` | 7 quick tests |

**Run it** (from `02-Training-Deep-Networks/010-Srivastava-et-al-2014-Dropout`):
```
python3 -m pytest -q             # a few seconds
python3 demo.py                  # a few seconds
python3 experiments.py --quick   # ~5 min
python3 experiments.py           # FULL: ~1 hour on a GPU. Strong machine only.
```

---

## 2. `dropout.py`

### `Net`
```python
Net([784, 1024, 1024, 2048, 10], p_input=0.8, p_hidden=0.5, act="relu", noise="bernoulli")
```
In `forward(x, train)`, for each layer's input `h` with keep probability `p`:

| | training | testing |
|---|---|---|
| `noise="bernoulli"` | `h * (rand < p)` | `h * p` (the same as W_test = pW) |
| `noise="gaussian"` | `h * (1 + √((1−p)/p) · randn)` | unchanged (E[r] = 1) |

`return_hidden=True` also returns every hidden layer's activations (for sparsity plots).

### `max_norm_(net, c)`
The **L2 norm** version (Section 5.1): each hidden unit's incoming weight column is scaled by min(1, c/‖w‖). (Paper 009 capped the *squared* length at 15, which is about c = 3.9.)

### `train(...)`
Standard SGD with momentum and cross-entropy, plus optional:
- `l2` → + λ·Σw²
- `l1` → + λ·Σ|w|
- `max_norm` → a projection after every step
- `lr_decay` → learning rate × decay each epoch

`eval_fn` lets experiments record anything after every epoch.

### `monte_carlo_error(net, X, y, k)`
Run k thinned networks (with `train=True`), average the **softmax probabilities**, take the argmax (Section 7.5).

### `Autoencoder`
784 → 256 ReLU → 784 sigmoid, with optional dropout on the hidden layer (Sections 7.1–7.2).

### Linear regression (Section 9.1)
- `dropout_linear_regression_closed_form(X, y, p)`: sets the gradient of the marginalized objective to zero:
  ```
  (p² XᵀX + p(1−p) diag(XᵀX)) w = p Xᵀy
  ```
- `dropout_linear_regression_sgd(...)`: minimizes the **same objective by brute force**, sampling a random row and a random mask each step. If the paper's formula is right, the two answers match.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1_regularizers` | Table 9: none / L2 / L2+L1 / max-norm / dropout+L2 / dropout+max-norm; settings chosen on validation |
| `e2_retention` | Figure 9: p ∈ {0.1 … 1.0} with n fixed and with p·n fixed |
| `e3_dataset_size` | Figure 10: 100 … 50,000 images, with and without dropout (same number of updates) |
| `e4_monte_carlo` | Figure 11: k = 1 … 120 vs weight scaling |
| `e5_autoencoder` | Figures 7–8: features and activation sparsity |
| `e6_gaussian` | Table 10: Bernoulli vs Gaussian, 3 seeds |
| `e7_linear_regression` | Section 9.1 |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_bernoulli_test_time_scaling_...` | W_test = pW matches the average over dropout masks |
| `test_gaussian_dropout_has_mean_one_...` | Section 10: mean 1, variance (1−p)/p, no test scaling |
| `test_max_norm_projects_onto_the_ball` | Section 5.1 |
| `test_dropout_linear_regression_is_ridge_regression` | Section 9.1 (zero gradient; beats OLS on the dropout loss) |
| `test_smaller_p_means_stronger_regularization...` | the (1−p)/p constant |
| `test_small_dropout_net_trains_...` | end to end, plus Monte-Carlo averaging |
| `test_autoencoder_shapes` | the model |

---

## 5. Try it yourself

1. Implement **inverted dropout** (scale by 1/p during training, nothing at test). Show it's equivalent.
2. Add **dropout RBMs** (Section 8): CD-1 with dropped hidden units.
3. Use `return_hidden=True` to measure how sparse each layer becomes, with and without dropout.
4. Try σ values other than √((1−p)/p) for Gaussian dropout.
