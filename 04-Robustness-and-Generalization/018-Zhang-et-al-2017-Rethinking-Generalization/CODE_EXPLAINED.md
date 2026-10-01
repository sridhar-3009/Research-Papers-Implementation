# The code, explained simply

How the code in this folder implements Zhang et al. (2017).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `randomization.py` | the randomizations, CIFAR-10 preprocessing, the three model families (exact Table 1 sizes), Rademacher fit, Theorem 1's network, the minimum-norm solution and SGD |
| `experiments.py` | Figure 1a–c, Table 1, Section 5's MNIST kernel fit (heavy, not run here) |
| `demo.py` | A light tour (about a second) |
| `test_randomization.py` | 12 quick tests (about a second) |

**Run it** (from `04-Robustness-and-Generalization/018-Zhang-et-al-2017-Rethinking-Generalization`):
```
python3 -m pytest -q                 # ~1 second
python3 demo.py                      # ~1 second
python3 experiments.py --quick       # 5,000 images, ~30-60 minutes
python3 experiments.py --only e2     # full CIFAR-10: many hours
```

---

## 2. `randomization.py`

### The randomizations (Section 2.1)
| Function | How |
|---|---|
| `corrupt_labels(y, p)` | a coin with probability p per example; hit examples get `randint(0, 10)` |
| `shuffle_pixels(X, perm=None)` | `X.flatten(2)[:, :, perm]`, one permutation of the H·W positions for all images. Returns `perm` so the test set gets the same one. |
| `random_pixels(X)` | `argsort` of random numbers gives one permutation **per image**, applied with `gather`. The same permutation is used for all 3 colour channels of an image. |
| `gaussian_images(X)` | `mean + std·randn`, with the per-channel mean and std of the dataset |

### Preprocessing (Appendix A)
- `center_crop(X, 28)`: 32×32 → 28×28.
- `per_image_whitening(X)`: TensorFlow's version, (x − mean)/max(std, 1/√N) for each image.

### Models (Figure 3, Appendix A)
- `ConvModule`: conv (with bias, "SAME" padding) → BatchNorm **without** learnable parameters → ReLU. This choice reproduces the paper's count exactly.
- `InceptionModule(c, ch1, ch3)`: a 1×1 and a 3×3 conv module, concatenated.
- `DownsampleModule(c, ch3)`: a stride-2 3×3 conv module and a stride-2 3×3 max-pool, concatenated. The channels add up: ch3 + c.
- `SmallInception(bn)`: the sequence from Figure 3. The channel count is tracked in the loop; it ends 7×7 → mean → FC 10.
- `SmallAlexNet`: (conv 5×5 64 → max-pool 3/2 → LRN) × 2 → FC 384 → FC 192 → 10.
- `MLP(hidden)`: 2352 → 512 (× hidden) → 10.

### `rademacher_fit(fit_and_predict, X)`
Draws random signs σ, trains the given learner on them, and averages (1/n)·Σσ_i·sign(h(x_i)). This replaces the "sup over the class" in Eq. 1 with "whatever training finds", which is exactly the paper's randomization-test view.

### `finite_sample_network(X, y)` / `finite_sample_predict` (Theorem 1)
```python
z = X @ a                                  # 1-D projections (random a)
zs = sort(z)
b = [zs[0] - 1, midpoints between consecutive zs]       # b_1 < z_1 < b_2 < z_2 < ...
A = relu(zs[:, None] - b[None, :])         # lower-triangular, positive diagonal
w = solve(A, y sorted the same way)
predict: relu(X @ a - b) @ w
```
Everything is in float64, so the fit is exact to about 1e-8.

### Section 5 helpers
- `min_norm_interpolant(X, y)`: α = (XXᵀ)⁻¹y, w = Xᵀα.
- `sgd_least_squares(X, y)`: per-example SGD on ½(w·x_i − y_i)² from w = 0. The step size is below 1/max‖x_i‖², so it converges.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 1a: Inception on 5 versions of the data; loss curves, and epochs until 100% training accuracy |
| `e2` | Figures 1b–c: corruption p = 0 … 1 for Inception, AlexNet, MLP; relative time to fit, and test error |
| `e3` | Table 1: every model × (random crop on/off) × (weight decay on/off), plus random labels |
| `e4` | Section 5: exact Gaussian-kernel interpolation of `--kernel-n` MNIST images, no regularization; plus linear least squares |

- `fit()` follows Appendix A: SGD with momentum 0.9; learning rate 0.1 (Inception) or 0.01 (others), × 0.95 per epoch; batch 128. It stops at 99.9% training accuracy.
- With augmentation (`augment_from`), each epoch takes **new** random 28×28 crops of the 32×32 images.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_label_corruption_rate` | fraction changed = 0.9·p |
| `test_shuffled_pixels_use_one_permutation_for_all_images` | "shuffled pixels" |
| `test_random_pixels_permute_each_image_differently` | "random pixels" |
| `test_gaussian_images_match_the_dataset_statistics` | "gaussian" |
| `test_preprocessing` | crop + per-image whitening |
| `test_parameter_counts_of_table_1` | 1,649,402 / 1,387,786 / 1,735,178 / 1,209,866 |
| `test_models_run_on_28x28_inputs` | shapes, and the 7×7 before global pooling |
| `test_two_layer_relu_net_with_2n_plus_d_weights_fits_any_labels` | Theorem 1 |
| `test_finite_sample_net_fits_random_class_labels_too` | memorizing class labels |
| `test_overparameterized_learner_has_rademacher_complexity_one` | 1.0 when d > n, low when d ≪ n |
| `test_sgd_from_zero_converges_to_the_minimum_norm_solution` | Section 5 |
| `test_hessian_of_linear_least_squares_does_not_depend_on_w` | "the curvature of all optimal solutions is the same" |

---

## 5. Try it yourself

1. In `e2`, also record the epoch at which the **test** accuracy peaks for p = 0.4. Does early stopping beat training to zero error?
2. Run `finite_sample_network` on 1,000 MNIST images with **random** labels, then evaluate it on new images. What test accuracy does a perfect memorizer get?
3. Plot test error against model width for an MLP on 1,000 MNIST images with 20% label noise. Do you see **double descent** around width ≈ number of examples?
4. In demo part 4, start SGD from a random w₀ instead of 0. Is the result still the minimum-norm fit? (Hint: SGD keeps w₀'s component outside the span of the data.)
