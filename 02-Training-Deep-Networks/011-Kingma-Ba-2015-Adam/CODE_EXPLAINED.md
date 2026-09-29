# The code, explained simply

How the code in this folder implements Kingma & Ba (2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `optimizers.py` | Adam, AdaMax, AdaGrad, RMSProp, AdaDelta, SGD Nesterov and temporal averaging, all written from scratch |
| `experiments.py` | Figures 1–4 of the paper (heavy, not run here) |
| `demo.py` | A light tour (under a second) |
| `test_optimizers.py` | 17 quick tests (about a second) |

**Run it** (from `02-Training-Deep-Networks/011-Kingma-Ba-2015-Adam`):
```
python3 -m pytest -q                      # ~1 second
python3 demo.py                           # < 1 second
python3 experiments.py fig1a --quick      # a few minutes
python3 experiments.py fig3               # HOURS. Strong machine only.
```

---

## 2. `optimizers.py`

### One interface for every optimizer
```python
opt = Adam(alpha=0.001)
params = opt.step(params, grads)          # lists in, updated list out
```
- `params` and `grads` are **lists of arrays**, one per weight matrix or bias.
- The optimizer keeps its own state (m, v, …) as a list with the same shapes, created on the first call.
- Only `+ − * / **` and `abs` are used, so it works on **NumPy arrays and on torch tensors** alike. The one exception is AdaMax's element-wise max, which is handled by `_max`.
- `_zeros_like(x)` is just `x * 0`: a zero array of the same shape and type, for both libraries.

### `Adam`: Algorithm 1, line by line
```python
self.t += 1                                                     # t <- t + 1
self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g       # m_t
self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g   # v_t
m_hat = self.m[i] / (1 - self.beta1 ** self.t)                  # bias correction
v_hat = self.v[i] / (1 - self.beta2 ** self.t)
out.append(p - alpha * m_hat / (v_hat ** 0.5 + self.eps))       # the update
```
Two switches for the experiments:
- `bias_correction=False` skips the two divisions: the "RMSProp with momentum" of Figure 4.
- `decay_sqrt_t=True` uses α/√t: Figure 1 and the theorem.

### `AdaMax`: Algorithm 2
`u = max(β2·u, |g|)` replaces v. Only m needs bias correction, applied through the step size `alpha / (1 - beta1**t)`. Its `eps` is only a divide-by-zero guard (1e-12).

### `AdaGrad`, `RMSProp`, `AdaDelta`, `SGDNesterov`
The baselines from the paper's figures:

| Class | State | Update |
|---|---|---|
| `AdaGrad` | s = Σg² | p − α·g/√s |
| `RMSProp` | r = running mean of g², optional momentum buffer | p − α·g/√r |
| `AdaDelta` | running means of g² and of Δx² | Δx = −√(E[Δx²]+ε)/√(E[g²]+ε)·g, **no learning rate** |
| `SGDNesterov` | velocity v | v ← µv − lr·g; p ← p + µv − lr·g (Paper 008's Nesterov, rewritten to use the gradient at p) |

### `TemporalAverage` (Section 7.2)
Call `avg.update(params)` after every optimizer step. It returns the bias-corrected moving average of the parameters; evaluate the model with those.

---

## 3. `experiments.py`

One part per figure, run one at a time: `python3 experiments.py <part>`.

| Part | Reproduces | Model |
|---|---|---|
| `fig1a` | Figure 1 left | MNIST logistic regression + L2 |
| `fig1b` | Figure 1 right | 20 Newsgroups bag-of-words (**stand-in for IMDB**), 50% input dropout |
| `fig2` | Figure 2 | MNIST 784-1000-1000-10 ReLU + dropout |
| `fig3` | Figure 3 | CIFAR-10 CNN c64-c64-c128-1000, with and without dropout |
| `fig4` | Figure 4 | MNIST VAE, a grid over β1, β2, α, bias correction on/off |

How it works:
- **`fit(params, loss_fn, batches, opt, epochs, eval_fn)`**: the training loop.
  - `torch.autograd.grad` computes the gradients **only**.
  - The update itself is done by **our** optimizer (`opt.step`), and the result is copied into the parameters.
  - It stops early if the loss becomes NaN, as unstable settings in Figure 4 can.
- **`best_over_grid`**: tries a few learning rates from the **same initialization** and keeps the one with the lowest final training cost, like the paper's grid search (but smaller).
- The models are written **as plain functions of a parameter list** (`forward(P, x, ...)`), not as `nn.Module`s, so the optimizers see raw tensors.
- In the CNN, each 5×5 conv uses padding 2 and each 3×3 max-pool has stride 2. So 32×32 → 16 → 8 → 4, and the flat vector has 128·4·4 = 2048 numbers.
- The VAE uses the reparameterization trick `z = µ + σ·noise` (Kingma & Welling 2014; a later paper in this repo). Loss = reconstruction + KL.
- Output: `results_<part>.json` and `figures/<part>.png`. Figure 4's plot shows the final loss vs log10 α, solid with bias correction and dashed without.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_adam_matches_pytorch` | our Algorithm 1 = `torch.optim.Adam`, over 50 steps |
| `test_first_step_has_size_alpha` | step 1 = α·sign(g) |
| `test_steps_are_bounded_by_alpha` | the Section 2.1 bound |
| `test_invariant_to_gradient_scale` | ×1000 gradients → same path |
| `test_bias_correction_removes_the_zero_start_bias` | v̂_t = g² exactly |
| `test_without_bias_correction_early_steps_explode` | Section 6.4 |
| `test_adagrad_is_a_limit_of_adam` | Section 5 |
| `test_adamax_u_is_the_limit_of_the_lp_norm` | Eqs. 8–12 |
| `test_adamax_steps_never_exceed_alpha` | Section 7.1, with Cauchy gradients |
| `test_every_optimizer_descends` (×6) | every optimizer works |
| `test_optimizers_work_on_torch_tensors` | the shared interface |
| `test_temporal_average_is_unbiased_for_a_constant` | Section 7.2 |

---

## 5. Try it yourself

1. Implement **AMSGrad** (Reddi et al. 2018): keep `v_max = max(v_max, v)` and divide by √v_max. Build their counterexample: f_t(x) = 1010x once every 3 steps and −10x otherwise, with x ∈ [−1, 1]. Plain Adam goes the wrong way.
2. Implement **AdamW** (Loshchilov & Hutter 2019): add weight decay directly to the update (`p − α·λ·p`) instead of to the gradient. Why is that different for Adam but not for SGD?
3. In the demo, plot Adam's path on the badly scaled quadratic for β1 = 0 and β1 = 0.9.
4. Log |m̂|/√v̂ (the "SNR") during training in `fig2`. Does it shrink as the loss flattens?
