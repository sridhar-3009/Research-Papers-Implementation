# The code, explained simply

How the code in this folder implements Szegedy et al. (2013).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `adversarial.py` | the L-BFGS attack and the search over c, the distortion measure, noise baselines, the paper's weight decay, unit vs random-direction analysis, operator norms (FC, conv via Fourier, conv via power iteration), the Table 1 models |
| `experiments.py` | Tables 1–4, Figures 1–2, Table 5-style bounds, adversarial training on MNIST (heavy, not run here) |
| `demo.py` | A light tour with no dataset (about a second) |
| `test_adversarial.py` | 12 quick tests (about a second) |

**Run it** (from `04-Robustness-and-Generalization/017-Szegedy-et-al-2013-Intriguing-Properties`):
```
python3 -m pytest -q                # ~1 second
python3 demo.py                     # ~1 second
python3 experiments.py --quick      # 100 attacked examples per model: ~20-40 minutes
python3 experiments.py              # 1,000 per model: many hours
```

---

## 2. `adversarial.py`

### `lbfgs_attack(model, x, target, c)`
The penalized problem of Section 4.1, solved with SciPy's **L-BFGS-B**, which supports box bounds natively:
```python
objective(z) = c * ||z - x||^2 + cross_entropy(model(z), target)      # z = x + r
bounds: 0 <= z_i <= 1
```
- PyTorch computes the gradient. `f_and_grad` returns (value, gradient) as NumPy arrays, which is what SciPy wants.
- We use ‖r‖² rather than |r|, because L-BFGS needs a smooth objective.

### `minimal_adversarial(model, x, target)`
The line search over c:
1. Try c = 1.
2. If it succeeds, multiply c by 10 until it fails. If it fails, divide by 10 until it succeeds.
3. **Bisect in log-space** between the last success and the first failure, 8 times.
4. Return the successful point with the largest c, which has the smallest r.

### `distortion(x, x_adv)`
`sqrt(mean((x_adv - x)^2))`, the paper's "Av. min. distortion" measure.

### Baselines
- `gaussian_distort(x, std)`: noise + clipping, the paper's control rows.
- `amplify(x, x_adv, 0.1)`: rescales the perturbation (same direction) to per-pixel stddev 0.1, for Table 4's lower half.
- `decay_penalty(model, lambdas)`: λ·Σw²/k per Linear layer, k = out_features.

### Section 3 tools
`top_activating(features, direction, k)` returns the indices of the k inputs with the largest ⟨φ(x), direction⟩. `unit_direction(n, i)` gives e_i; `random_direction(n)` gives a random unit vector.

### Section 4.3 tools
| Function | How |
|---|---|
| `fc_operator_norm(W)` | largest singular value (`matrix_norm(ord=2)`) |
| `conv_operator_norm_fft(weight, n)` | zero-pad each kernel to n×n, `fft2`, rearrange to one D×C matrix per frequency, take the max of `svdvals`. That's Eq. (1) for stride 1, with circular boundaries. |
| `conv_operator_norm_power(weight, n, stride)` | **power iteration**: x ← AᵀAx / ‖·‖, where A = circular convolution and Aᵀ comes from `autograd.grad(y, x, grad_outputs=y)`. Works for any stride. |
| `conv_as_matrix(weight, n, stride)` | brute force: apply the conv to every basis image to build the full matrix (tests only, small n) |
| `lipschitz_upper_bound(norms)` | the product of the layer norms |

### Models
- `fc_net(hidden)`: FC10 (no hidden layer), FC100-100-10, FC200-200-10, FC123-456-10, with sigmoid.
- `AE400`: encoder 784 → 400 sigmoid, decoder back to 784, classifier 400 → 10.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1_e2` | trains the 6 Table 1 models; attacks `--n-attack` training images (a random wrong target each); reports errors and the average minimal distortion; then feeds every model's adversarial set (and two Gaussian-noise sets) to every model (Table 2) |
| `e3` | P1/P2 split, three models, attacks on test images, plus amplified versions (Tables 3–4) |
| `e4` | top-8 test images for 4 hidden units and 4 random directions of FC100-100-10 (Figures 1–2) |
| `e5` | operator norms of the trained nets; `--alexnet` downloads torchvision's AlexNet and bounds its 5 convs + 3 FC layers (Table 5) |
| `e6` | adversarial training with a refreshed pool (input level only), vs weight decay alone |

`train_ae400` trains the autoencoder with a KL sparsity penalty, then freezes it and trains only the softmax, matching "this layer was not fine-tuned".

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_attack_reaches_the_target_class_inside_the_box` | the attack works and respects [0, 1] |
| `test_minimal_adversarial_finds_the_closest_point_over_the_boundary` | ‖r‖ ≈ 0.2, the exact minimum for a linear boundary |
| `test_bigger_c_means_smaller_perturbation` | the role of c |
| `test_distortion_measure_is_the_perturbation_stddev` | the paper's metric |
| `test_gaussian_noise_and_amplify` | the baselines |
| `test_paper_weight_decay_divides_by_layer_width` | λ·Σw²/k |
| `test_top_activating_for_unit_and_random_direction` | Section 3 tool |
| `test_fc_norm_is_largest_singular_value_and_bounds_a_relu_layer` | L_k ≤ ‖W‖ |
| `test_conv_norm_fourier_formula_equals_brute_force` | Eq. (1) is exact |
| `test_conv_norm_with_stride` | power iteration with stride 2 |
| `test_lipschitz_bound_of_a_whole_network_holds` | L = Π L_k |
| `test_table_1_models_shapes` | the models |

---

## 5. Try it yourself

1. Implement **FGSM** (Goodfellow et al. 2015): x + ε·sign(∇ₓ loss), one gradient step. Compare its distortion and speed with the L-BFGS attack.
2. Attack a **CNN** (e.g. LeNet-5 from Paper 013) and compare the distortion with the fully connected nets.
3. Rescale each layer's weights so that ‖W‖ ≤ 1 (**spectral normalization**, Miyato et al. 2018). Does the needed distortion grow?
4. Repeat E4 with a trained CNN's last hidden layer, and decide for yourself whether random directions look as "meaningful" as units.
