# The code, explained simply

How the code in this folder implements Chapters 3–4 of Sutskever's thesis (2013).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `rtrbm.py` | the RTRBM (NumPy): inference, sampling, CD-k, exact enumeration for tiny models, BPTT gradient |
| `hf_rnn.py` | a tanh RNN with the thesis's sparse init, and HF with structural damping (PyTorch) |
| `tasks.py` | the 8 pathological problems of Section 4.4, the success criterion, bouncing-balls videos |
| `experiments.py` | Figure 4.1, SGD vs HF, RTRBM on bouncing balls (heavy, not run here) |
| `demo.py` | A light tour (about a second) |
| `test_thesis.py` | 16 quick tests (a couple of seconds) |

**Run it** (from `05-Words-and-Sequences/023-Sutskever-2013-PhD-Thesis-Training-RNNs`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~1 second
python3 experiments.py --quick   # ~30-60 minutes
```

---

## 2. `rtrbm.py`

**Parameters:**
- `W` (H × V): the RBM weights;
- `Wp` (W′, H × H): hidden-to-hidden;
- `bv`, `bh`: the biases;
- `r0`: the initial state.

**Methods:**
- `infer(vs)`: r₀ … r_T, with r_t = sigmoid(W v_t + b_h + W′ r_{t−1}).
- `hidden_biases(rs)`: b_t = b_h + W′ r_{t−1}, the bias of the RBM that models frame t.
- `cd_grad(v, b, k)`: CD-k for one RBM.
  - Start from p(h | v); alternate v ← sample, h ← sample.
  - Return ⟨h vᵀ⟩_data − ⟨h vᵀ⟩_k and the bias differences.
- `exact_grad(v, b)` → `rbm_grad_exact`: enumerates all 2^V visible states to compute the model expectations exactly (tiny models only).
- **`gradient(vs, k, exact)`:**
  1. Forward: `infer`, then each step's RBM gradient. W and b_v are accumulated directly; d log P_t / d b_t is stored.
  2. Backward over t = T … 0 with `dr` = dL/dr_t:
     ```python
     dr  = Wp.T @ g_b[t]                                       # r_t is used in the bias of the next RBM
     pre = r_{t+1} (1 - r_{t+1}) * dr_next                     # and in r_{t+1}'s sigmoid
     dr += Wp.T @ pre
     grad Wp += outer(g_b[t], r_t) + outer(pre, r_t);  grad bh += g_b[t] + pre;  grad W += outer(pre, v_{t+1})
     ```
     These are Eqs. 3.23–3.25 written as a loop.
- `log_prob_exact`: the sum of exact RBM log-probabilities (for the tests).
- `sample(T)` (Algorithm 3) and `predict_next` (next-frame prediction for the squared error).

## 3. `hf_rnn.py`

- `RNN(n_in, H, n_out)`: x_t = W_hv v_t + W_hh h_{t−1} + b_h, h_t = tanh(x_t), o_t = W_oh h_t + b_o. **`forward` returns both o and x,** which structural damping needs.
  - Sparse init: 15 nonzero weights per unit; variance 1 for W_hv, 1/15 for W_hh, W_oh and the biases.
- `loss_fn(o, target, mask, kind)`: "mse" for continuous targets, "ce" for symbols, averaged over the targeted steps.
- **`HFStructural.curvature_product(data, v)`:**
  ```python
  (o, x), (Jo, Jx) = jvp(model, params, v)               # one forward-mode pass gives BOTH J_o v and J_x v
  Ho = H_L(o) Jo                                          # MSE: identity; CE: diag(p) - p p^T
  Hx = lambda * mu * (1 - tanh(x)^2) * Jx / N             # structural damping term
  return vjp(model, params)((Ho, Hx))                     # one reverse pass applies both J^T
  ```
- **`step(grad_data, curv_data, structural)`:**
  1. Gradient on the big batch.
  2. CG on (G_f + λμG_S + λI)·d = −g, warm-started from 0.95 × the previous d.
     - q(x) is tracked for free from the residual.
     - Martens' stopping rule: stop when the relative improvement over the last k = max(10, 0.1·i) iterations is below k·5·10⁻⁴.
  3. Update. Then ρ = actual / predicted change; λ ×3/2 if ρ < 1/4, ×2/3 if ρ > 3/4. A worsening step is undone.

## 4. `tasks.py`

| Function | Section 4.4 |
|---|---|
| `addition`, `multiplication`, `xor` (via `marked_pairs`) | 4.4.1: random lengths are right-aligned so the target is always at the last step |
| `temporal_order`, `temporal_order_3bit` (via `_symbols`) | 4.4.2–4.4.3: symbols 1, 2 are special; 3–6 are noise |
| `random_permutation` | 4.4.4 |
| `memorization(T, n_bits, n_values)` | 4.4.5 (5-bit) and the 20-bit variant (10 symbols from 5 values) |
| `error_rate(model, data)` | the < 1% success criterion |
| `bouncing_balls(T, res, n_balls)` | the video data: Gaussian blobs moving at constant speed, bouncing off walls and each other |

`PROBLEMS` maps names to generators.

---

## 5. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 4.1: for each problem and T, HF with and without structural damping; HF iterations until < 1% test error |
| `e2` | the addition problem with SGD + momentum + clipping, for comparison |
| `e3` | RTRBM (400 hidden, CD-10 → CD-25, momentum 0.9, decaying learning rate) vs W′ = 0 on bouncing balls; next-frame squared error |

---

## 6. The tests

| Test | Proves |
|---|---|
| `test_exact_rbm_probabilities_sum_to_one` | the enumeration is right |
| `test_exact_rbm_gradient_matches_finite_differences` | exact RBM gradients |
| `test_inference_is_the_deterministic_recursion` | Algorithm 4 |
| `test_bptt_with_exact_rbm_gradients_is_the_exact_gradient` | **Section 3.10's claim**, against autograd |
| `test_cd_gradient_is_a_noisy_estimate_of_the_exact_one` | CD vs exact |
| `test_sampling_shapes` | Algorithm 3 |
| `test_addition_problem_matches_section_4_4_1` | target (u_I + u_J)/2 |
| `test_xor_targets` | XOR |
| `test_temporal_order_classes` | 4 and 8 classes, 2 and 3 special symbols |
| `test_random_permutation_first_equals_last` | the predictable target |
| `test_memorization_reproduces_the_bits` | bits, blanks, the trigger at T + 5 |
| `test_bouncing_balls` | the video generator |
| `test_sparse_init_15_connections` | Section 4.4's initialization |
| `test_curvature_product_matches_explicit_matrices` | **(G_f + λμG_S)·v is exact** |
| `test_hf_learns_a_short_memorization_task` | HF learns |
| `test_every_problem_generates` | all 8 problems run end to end |

---

## 7. Try it yourself
1. Run `e1` on 5-bit memorization at T = 100 with and without structural damping. Do you see the thesis's "essential for T > 50"?
2. Give the RTRBM a **dynamic visible bias** too (b_v + W″ r_{t−1}) and compare next-frame errors.
3. Replace HF in `e1` with Chapter 7's recipe (SGD with a high momentum schedule, Paper 008), using the same initialization.
4. Train an LSTM (Paper 021) on the same problems and compare the computation needed.
