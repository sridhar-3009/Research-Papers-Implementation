# The code, explained simply

How the code in this folder implements "Efficient BackProp".
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is | Paper part |
|---|---|---|
| `efficient_backprop.py` | The network, every trick, and the Hessian tools | Sections 3–9 |
| `experiments.py` | Tests 7 claims; writes `results.md` and `figures/` | Sections 4.1, 4.3–4.6, 5, 8, 9 |
| `demo.py` | A short tour | everything |
| `test_efficient_backprop.py` | 11 tests | everything |

**Run it** (from `02-Training-Deep-Networks/006-LeCun-et-al-1998-Efficient-BackProp`):
```
python3 demo.py                           # ~5 s
python3 experiments.py                    # ~1-2 min (can be longer if the machine is busy)
python3 -m pytest -q                      # < 1 s
```
Needs `numpy` and `scikit-learn` (for the digits data); the figures need `matplotlib`.

**Suggested reading order:**
1. `lecun_tanh`
2. `InputTransform`
3. `MLP.gradient`
4. `MLP.diag_hessian`
5. `hessian_vector` and `power_method`

---

## 2. `efficient_backprop.py`

### Sigmoids
- `lecun_tanh(x) = 1.7159·tanh(2x/3)`, plus its derivative.
- `logistic`, for comparison.

### `InputTransform`: Figure 3's three steps
```python
tf = InputTransform(decorrelate=False).fit(X_train)   # learn mean and scale from TRAINING data
X_train_n, X_test_n = tf(X_train), tf(X_test)         # apply the SAME transform to both
```
- `fit` stores the mean and, for each input, 1/std.
- With `decorrelate=True`, it rotates onto the principal axes first (PCA, the eigenvectors of the covariance) and scales by 1/√eigenvalue. That's **whitening**.
- **Always fit on training data only**, then reuse that transform on test data.

### `MLP`
- `sizes=[64, 30, 10]`, `sigmoid="lecun_tanh"`, `init="lecun"`.
- **LeCun init:** a uniform distribution on [−r, r] has std r/√3, so `r = √(3/m)` gives std m^(−1/2) (Eq. 16).
- `forward` → `Y` (weighted sums) and `X` (states) for every layer (Eqs. 2–3).
- `gradient` → backprop, Eqs. (7)–(9), averaged over the given patterns.
- `get`/`set` → all weights as one flat vector, which the Hessian tools need.

### `MLP.diag_hessian`: Eqs. (54)–(56)
It's backprop with squares:
```python
d2E_dY = d2E_dX * f'(Y)**2                 # (54)
hW     = (X_prev**2).T @ d2E_dY            # (55)
d2E_dX = d2E_dY @ (W**2).T                 # (56)
```
It starts from ∂²E/∂o² = 1 (for ½‖D − o‖²). The result is the **Gauss–Newton** diagonal: the paper drops f'' terms, so all values are ≥ 0. A test checks it against (∂o/∂w)² computed by finite differences.

### Training
- `train_sgd`: one update **per example**, shuffled each epoch (Eq. 11, Section 4.2). An optional `per_weight_lr` gives each weight its own rate.
- `train_batch`: one update per pass, with the full gradient (Eq. 10).

### `stochastic_diag_lm_rates`: Eq. (61)
`eta / (curvature + mu)` for every weight. More curvature means a smaller step.

### Hessian tools (Sections 7.5, 9.2)
- `hessian_vector(net, X, D, v)`: (∇E(w + αv) − ∇E(w)) / α (Eq. 59).
- `power_method`: repeat v ← Hv/‖v‖ (Eq. 60), so ‖Hv‖ → λ_max.
- `online_eigenvalue`: the one-example-at-a-time running average (Eq. 64). It returns the whole history, which is what Figure 24 plots.

### The linear example (Section 5.2)
- `two_gaussians`: 100 points around (−0.4, −0.8) and (0.4, 0.8). std 0.19 gives the paper's covariance eigenvalues (0.84 and 0.036 in theory).
- `lms_hessian`: Eq. (29), the input covariance **including the bias** as an always-1 input.
- `lms_batch`: batch LMS, returning the MSE per epoch.

---

## 3. `experiments.py`

| Function | Claim tested |
|---|---|
| `e1_linear` | η below/above 2/λ_max; finds that the bias makes λ_max = 1.0 (limit 2.0, not 2.38) |
| `e2_mean_shift` | shifting inputs by +3 → condition number 33 → 2,096 |
| `e3_redundant` | stochastic vs batch on 10 copies of 100 digits |
| `e4_recipe` | remove one trick at a time; best learning rate per row; 3 seeds |
| `e5_predicted_rate` | Figures 24–25: power method, on-line estimate, error vs (η / predicted) for batch and SGD |
| `e6_sdlm` | stochastic diagonal LM vs SGD at 3 learning rates |
| `e7_hessian` | full Hessian (760 weights) by finite differences: spectrum and per-layer curvature |

**Fairness:** every configuration gets its **own best learning rate** from a grid. Otherwise you'd just be comparing learning rates.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_recommended_sigmoid_properties` | f(±1) = ±1; curvature peaks at 1 |
| `test_input_transform_...` | mean 0, std 1; whitening gives the identity covariance |
| `test_lecun_init_gives_unit_variance_sums` | Eq. (16) |
| `test_gradient_matches_finite_differences` | Eqs. (7)–(9) |
| `test_diag_hessian_is_the_gauss_newton_diagonal` | Eqs. (54)–(56) |
| `test_power_method_finds_largest_eigenvalue` | Eqs. (59)–(60) |
| `test_lms_hessian_is_input_covariance` | Eq. (29) and the paper's 0.84 / 0.036 |
| `test_learning_rate_limit_is_two_over_lambda_max` | Eq. (38), and the 2.1-diverges finding |
| `test_non_centred_inputs_blow_up_the_condition_number` | Section 5.3 |
| `test_stochastic_beats_batch_on_redundant_data` | Section 4.1 |
| `test_sdlm_rates_are_smaller_where_curvature_is_larger` | Eq. (61) |

---

## 5. Try it yourself

1. Add **momentum** to `train_sgd`. Does it help more in batch mode or in stochastic mode, as the paper wonders?
2. Try `InputTransform(decorrelate=True)` on the digits. Faster than plain scaling?
3. Make the net deeper (64-30-30-30-10) and repeat E4. Do "tiny initial weights" and "targets at the asymptotes" hurt now?
4. Use `power_method` every epoch and set η = 0.1/λ_max. Is this a good automatic learning rate for SGD?
5. Add the linear "twisting term" (1.7159 tanh(2x/3) + 0.01x). Does it help when weights start too big?
