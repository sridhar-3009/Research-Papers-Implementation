# The code, explained simply

How the code in this folder implements Ioffe & Szegedy (2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `batchnorm.py` | BN forward and backward by hand (NumPy), Algorithm 2, conv BN, the Section 2 example, a hand-written torch BN layer, the Section 4.1 MLP |
| `experiments.py` | Figure 1, plus small versions of the ImageNet findings (heavy, not run here) |
| `demo.py` | A light tour (under a second) |
| `test_batchnorm.py` | 13 quick tests (about a second) |

**Run it** (from `02-Training-Deep-Networks/012-Ioffe-Szegedy-2015-Batch-Normalization`):
```
python3 -m pytest -q                  # ~1 second
python3 demo.py                       # < 1 second
python3 experiments.py --quick        # a few minutes
python3 experiments.py                # FULL: 30-60 min on a CPU. Strong machine only.
```

---

## 2. `batchnorm.py`: the NumPy part

**Shapes:** `x` is `(m, d)`, with one row per example and one column per feature. Every operation uses `axis=0`, the batch direction, so each feature is handled separately.

### `bn_forward(x, gamma, beta, eps)`
Algorithm 1, four lines:
```python
mu = x.mean(axis=0)
var = ((x - mu) ** 2).mean(axis=0)          # divides by m (biased), as in the paper
x_hat = (x - mu) / np.sqrt(var + eps)
y = gamma * x_hat + beta
```
It also returns a `cache` of everything the backward pass needs.

### `bn_backward(dy, cache)`
The paper's six chain-rule equations, **one line each**, in the same order: `dx_hat`, `dvar`, `dmu`, `dx`, `dgamma`, `dbeta`.
- Why is there a `dmu` term inside `dvar`'s line? Because σ² depends on µ. The paper's equation includes that term; it happens to sum to zero, since Σ(x − µ) = 0.

### `bn_backward_compact(dy, cache)`
The same gradient in one line: `γ/σ · (dy − mean(dy) − x̂·mean(dy·x̂))`. The tests check it agrees with the long form.

### `population_stats(batches)` and `fuse_for_inference(...)`
Algorithm 2:
- average the batch means, and the batch variances × m/(m − 1);
- turn BN into `a·x + c`.

### `bn_conv_forward` / `bn_conv_backward`
Section 3.2. It reshapes `(N, C, H, W)` into `(N·H·W, C)`, so every pixel of every image counts as one "example" for its channel. Then it reuses the 2-D functions and reshapes back.

### `normalize_outside_gradient(u, target, inside)`
Section 2's example as a tiny loop:
- `inside=False`: the gradient on b pretends E[x] is a constant.
- `inside=True`: it includes ∂E[x]/∂b = 1, so ∂x̂/∂b = 1 − 1 = 0.

---

## 3. `batchnorm.py`: the PyTorch part

### `BatchNorm(num_features)`
Our own BN layer. We only write the forward pass; autograd derives the backward, which the tests show is the same as ours.
- **Training mode:** batch mean and variance. Statistics are over dims `(0,)` for 2-D input, or `(0, 2, 3)` for conv input (Section 3.2).
- **It also updates running averages** with `lerp_`: running ← running + momentum·(new − running). The variance is stored unbiased (× m/(m − 1)).
- **Eval mode:** use the running statistics. `net.eval()` / `net.train()` switch modes, which is standard PyTorch.

### `MLP(sizes, act, bn, init_std)`
- `bn=False`: z = W·u + b, then sigmoid.
- `bn=True`: z = BN(W·u) with **no bias** (β replaces it), then sigmoid.
- The output layer never has BN.
- `forward(x, return_preact=True)` also returns every sigmoid's input, which is what Figure 1(b, c) plots.

---

## 4. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1_figure1` | Figure 1: accuracy curves + percentiles (15/50/85) of unit 0's sigmoid input in the last hidden layer, every 500 steps |
| `e2_learning_rates` | Figures 2–3 in miniature: lr ×1, ×5, ×30, with and without BN, for ReLU and sigmoid; "steps to reach plain-x1's best" |
| `e3_deep_sigmoid` | 10 hidden sigmoid layers: without BN it barely learns, with BN it trains |

- `train(...)`: plain SGD, random minibatches of 60, evaluating on the test set every 500 steps (in eval mode, so it uses the running statistics). It stops if the loss becomes NaN and records `diverged_at`.
- Output: `results.json`, `results.md`, `figures/*.png`. `--report-only` rebuilds the report without training.

---

## 5. The tests

| Test | Proves |
|---|---|
| `test_output_has_zero_mean_unit_variance_per_feature` | Algorithm 1 |
| `test_gamma_beta_can_recover_the_identity` | Section 3's argument for γ, β |
| `test_backward_matches_finite_differences` | our backward is correct |
| `test_backward_matches_torch_and_compact_form` | = PyTorch, = the simplified form |
| `test_gradient_has_no_mean_and_no_component_along_x_hat` | what the backward pass does |
| `test_scaling_the_weights_changes_nothing_but_the_weight_gradient` | Section 3.3 |
| `test_population_variance_is_unbiased` | Algorithm 2's m/(m − 1) |
| `test_inference_is_a_single_linear_map` | Algorithm 2, line 11 |
| `test_conv_bn_normalizes_each_feature_map_...` | Section 3.2 (and the gradient = PyTorch) |
| `test_normalizing_outside_the_gradient_makes_b_drift_forever` | Section 2 |
| `test_torch_layer_train_and_eval_modes` | our layer's two modes |
| `test_train_mode_output_depends_on_the_other_examples` | Section 3.4 (regularizing noise) |
| `test_mlp_forward_shapes` | the model, no bias before BN |

---

## 6. Try it yourself

1. Train `MLP(bn=True)` with **batch size 2**, then 4, then 60. What happens to test accuracy, and why? (Hint: noisy statistics, and the train/test mismatch.)
2. Implement **Layer Normalization** (Ba et al. 2016): normalize over the features of **one** example (axis=1) instead of over the batch. It needs no running statistics. Why is it better for RNNs and Transformers?
3. Put BN **after** the sigmoid instead of before it. Does it still help?
4. Fold a trained BN layer into the previous `nn.Linear` using `fuse_for_inference`, and check that the outputs are identical.
