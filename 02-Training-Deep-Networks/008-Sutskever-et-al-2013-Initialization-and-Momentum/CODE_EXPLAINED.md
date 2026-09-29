# The code, explained simply

How the code in this folder implements Sutskever et al. (2013).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is | Paper part |
|---|---|---|
| `momentum.py` | CM and NAG by hand, the Eq. 5 schedule, sparse and echo-state initialization, the autoencoder, the RNN, the addition problem | Sections 2–4 |
| `experiments.py` | Tables 1, 2, 3, 5 and appendix Figure 2 | everything |
| `demo.py` | A tiny tour (runs in ~1 second) | Sections 2–4 |
| `test_momentum.py` | 11 quick tests | Eqs. 1–5, Theorem 2.1, Tables 3–4 |

**Run it** (from `02-Training-Deep-Networks/008-Sutskever-et-al-2013-Initialization-and-Momentum`):
```
python3 -m pytest -q             # < 1 s
python3 demo.py                  # ~1 s
python3 experiments.py --quick   # a few minutes
python3 experiments.py           # FULL: ~1-1.5 hours of GPU/CPU. Strong machine only.
```

---

## 2. `momentum.py`

### `Momentum`: Eqs. (1)–(4) by hand
```python
opt = Momentum(model.parameters(), lr=0.01, mu=0.9, nesterov=True)
loss = opt.step(lambda: loss_function())
```
Inside `step`:
1. **NAG only:** move every parameter to θ + µv (the look-ahead point).
2. Compute the loss and its gradient **there**.
3. **NAG only:** move back to θ.
4. `v = µv − ε·grad` (Eq. 1 or 3), then `θ = θ + v` (Eq. 2 or 4).

It takes a **function** that computes the loss (not a loss value), because NAG must evaluate the gradient at a *different* point from the current θ.

### `mu_schedule(t, mu_max, period=250)`: Eq. (5)
Returns `min(1 − 2^(−1 − log₂(t//period + 1)), mu_max)`. `period` is 250 in the paper. `experiments.py` shrinks it for short runs, so µ still reaches µ_max.

### `quadratic_run`: CM/NAG on q(x) = ½xᵀAx + bᵀx
Pure NumPy, no autograd. It's used to check Theorem 2.1 exactly and to draw appendix Figure 2.

### Initializations
- `sparse_init_(W, 15, scale)`: for each column (unit), pick 15 random rows and fill them with N(0,1)·scale. Everything else is 0.
- `esn_init(...)`: a sparse recurrent matrix, then divided by its current spectral radius and multiplied by 1.1, plus small input and output weights (Table 4).

### Models and data
- `Autoencoder`: 784-1000-500-250-**30**-250-500-1000-784, sigmoid layers, a linear 30-unit code, sigmoid output. It's trained with cross-entropy, and `autoencoder_losses` also returns the **squared error** that Table 1 reports.
- `RNN`: 100 tanh units; `forward` runs over time and returns the last output.
- `addition_problem(batch, T)`: random values plus two markers (one in the first 10%, one before T/2). The target is the sum of the two marked values. Inputs and target are **centred** (Section 4.1).

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1_quadratic` | appendix Fig. 2, plus the NAG-divergence finding |
| `e2_e3_e4` | Table 1 (SGD, NAG and CM × µ_max ∈ {0.9, 0.99, 0.995}), Table 2 (lower µ at the end), Table 3 (sparse-init scale) |
| `e5_rnn` | Table 5, addition problem, µ₀ ∈ {0, 0.9, 0.98} × {NAG, CM} |

**Scale:**
- Autoencoder: 15,000 updates (the paper: 750,000).
- RNN: 4,000 updates at T = 50 (the paper: 50,000 at T = 80).

The docstring explains these choices. Expect higher absolute errors than the paper; look at the **pattern**.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_momentum_class_matches_the_equations` (×2) | the torch optimizer = Eqs. 1–4 exactly |
| `test_theorem_2_1_...` | NAG = CM with µ(1 − λε), exactly |
| `test_nag_oscillates_less_than_cm...` | Section 2.1 / Fig. 2 |
| `test_nag_can_diverge_where_cm_converges` | our finding (λε > 1) |
| `test_cm_and_nag_agree_for_tiny_learning_rates` | "equivalent when ε is small" |
| `test_momentum_schedule` | Eq. 5 values, the cap, never decreasing |
| `test_sparse_initialization_has_15_inputs_per_unit` | Section 3.1 |
| `test_esn_init_spectral_radius` | Table 4 |
| `test_autoencoder_shape` | the architecture |
| `test_addition_problem_targets` | the data is right |

---

## 5. Try it yourself

1. In `demo.py`, find the largest learning rate for which NAG still converges on the valley. Compare with 1/λ_max = 0.01 and 2/λ_max = 0.02.
2. Change the RNN spectral radius to 0.5 and to 3.0. What happens to the hidden state over 50 steps?
3. Implement the "lower µ at the end" trick in your own training loop.
4. Replace sparse init with Xavier init (Paper 007) in the autoencoder. Which trains faster?
