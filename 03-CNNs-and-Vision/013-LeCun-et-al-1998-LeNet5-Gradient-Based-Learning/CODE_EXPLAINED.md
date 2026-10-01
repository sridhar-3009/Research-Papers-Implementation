# The code, explained simply

How the code in this folder implements LeCun et al. (1998).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `lenet5.py` | LeNet-5 exactly as in the paper, a loop-based convolution, both losses, the input format, distortions, stochastic diagonal Levenberg-Marquardt |
| `experiments.py` | Figures 5, 6, 9, distortions, shift robustness, MSE vs MAP (heavy, not run here) |
| `demo.py` | A light tour, with no training (about a second) |
| `test_lenet5.py` | 17 quick tests (about a second) |

**Run it** (from `03-CNNs-and-Vision/013-LeCun-et-al-1998-LeNet5-Gradient-Based-Learning`):
```
python3 -m pytest -q                        # ~1 second
python3 demo.py                             # ~1 second
python3 experiments.py --quick              # a few minutes
python3 experiments.py --only e1            # the paper's Figure 5: HOURS at batch size 1
```

---

## 2. `lenet5.py`

### `squash(a)`
`1.7159 * tanh(2a/3)`, Eq. 6. It's used after every layer up to F6.

### `conv2d_naive(x, w, b)`: what a convolution is
Three loops:
```python
for o in range(C_out):                 # each feature map...
    for i in range(H):
        for j in range(W):             # ...at each position...
            out[o, i, j] = np.sum(x[:, i:i+k, j:j+k] * w[o]) + b[o]   # ...the SAME kernel w[o]
```
- `x[:, i:i+k, j:j+k]` is the local receptive field.
- `w[o]` doesn't depend on (i, j): those are the shared weights.

The model itself uses PyTorch's fast `conv2d`, which the tests show gives the same result.

### `C3_TABLE` and `c3_mask()`
Table I as a list: `C3_TABLE[o]` is the tuple of S2 maps that C3 map `o` reads. `c3_mask()` turns it into a 16×6 matrix of 0s and 1s.

### `rbf_prototypes()`
The ten 7×12 bitmaps are typed out as strings (`#` = ink = +1, `.` = −1) and flattened to a (10, 84) tensor. They're our own drawings in the paper's style; the paper's exact bitmaps are only shown as a picture (Figure 3).

### Layers

| Class / layer | How it works |
|---|---|
| `nn.Conv2d(1, 6, 5)` = **C1** | normal convolution |
| `Subsample(maps)` = **S2, S4** | `avg_pool2d(x, 2) * 4` = the sum of each 2×2 block, then `coef * sum + bias` per map, then squash. **2 parameters per map.** |
| `SparseConv()` = **C3** | a full 16×6×5×5 kernel **multiplied by the mask**. A masked weight is always multiplied by 0, so it has no effect and gets zero gradient. `n_params()` counts only the real ones: 1,516. |
| `nn.Conv2d(16, 120, 5)` = **C5** | on a 5×5 input this gives 1×1 outputs |
| `nn.Linear(120, 84)` = **F6** | |
| `prototypes` | an `nn.Parameter` with `requires_grad=False` (fixed), unless `learn_prototypes=True` (for the collapse experiment) |

`_uniform_fan_in(t, F)` is Appendix A's initialization, uniform in ±2.4/F. For C3, each map gets its own fan-in (75, 100 or 150), because the maps read different numbers of S2 maps.

### `LeNet5`
- `features(x)`: everything up to F6's 84 outputs.
- `forward(x)`: Eq. 7, the squared distance to each of the 10 prototypes. `h[:, None, :] - self.prototypes[None]` broadcasts to (N, 10, 84); summing over the last axis gives (N, 10).
- `predict(x)`: `argmin` (the smallest penalty), not argmax.
- `n_trainable()`: counts parameters the way the paper does: 60,000.

### Losses
- `mse_loss(y, labels)`: the mean penalty of the correct class (Eq. 8). `gather` picks one column per row.
- `map_loss(y, labels, j)`: Eq. 9. `logsumexp` computes log(e^{−j} + Σ e^{−y_i}) without overflow. The "rubbish class" is a column with value −j.

### Data
- `prepare(uint8 images)`: scale to [0, 1], pad 28 → 32, then map to −0.1 … 1.175.
- `distort(x)`: random affine transforms (scale, squeeze, horizontal shear, shift) using `affine_grid` + `grid_sample`. It works in "ink" units, so pixels coming from outside the image become background.

### Appendix C: SDLM
- `gauss_newton_diag(model, x)`: estimates h_kk.
  - **Idea:** the MSE loss's Hessian with respect to F6's output is 2I, so the Gauss-Newton matrix is Jᵀ(2I)J.
  - For a random v with covariance 2I, **(Jᵀv)² equals the diagonal on average**. Jᵀv is just one backward pass of `(out * v).sum()`.
  - We do this for each example separately and average. (Squaring a batch-summed gradient would add wrong cross terms.)
- `sdlm_step(model, loss, h, eta, mu)`: `p -= eta / (mu + h) * grad`, a separate learning rate per parameter.
- `paper_eta(epoch)`: the schedule 5e-4, 2e-4, 1e-4, 5e-5, 1e-5.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 5: 20 passes, SDLM, h re-estimated on 500 samples per pass. It saves `lenet5.pt`. |
| `e2` | Figure 6: 15k / 30k / 60k training images |
| `e3` | Distortions: each example shown is distorted with probability 0.9 (= 540k of 600k) |
| `e4` | Figure 9 baselines: linear, MLP 300, MLP 300-100 (SGD + momentum + cross-entropy, not the paper's exact recipe), 3-NN |
| `e5` | Test digits shifted by 0–4 pixels: LeNet-5 vs MLP |
| `e6` | Learned RBF centers: MSE vs MAP. It reports the distance between the two closest centers (≈ 0 means collapse). |

`train_lenet` follows the paper: re-estimate h, set η from the schedule, visit every example once per pass, and measure the training error **on the fly**. That's why Figure 5's training curve looks higher than the test curve early on.

`--batch 1` (the default) is the paper's per-example SGD. `--batch 32` is much faster.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_naive_convolution_matches_torch` | our loops = a real convolution |
| `test_convolution_is_shift_equivariant` | shift in → shift out |
| `test_layer_shapes` | 6@28, 6@14, 16@10, 16@5, 120@1, 84, 10 |
| `test_parameter_counts_match_section_II_B` | every layer's count, **60,000 total** |
| `test_connection_counts_match_section_II_B` | 122,304 / 5,880 / 151,600 / 2,000 |
| `test_c3_table_structure` | 6×3, 9×4, 1×6 |
| `test_c3_disconnected_weights_never_matter` | the mask works (zero gradient, no effect) |
| `test_subsampling_is_sum_times_coefficient_plus_bias` | S2 / S4 |
| `test_squashing_function_appendix_A` | f(±1) = ±1, asymptote 1.7159, max curvature at 1 |
| `test_prototypes_are_distinct_plus_minus_one_bitmaps` | the 84-d codes |
| `test_rbf_output_is_squared_distance_and_argmin_predicts` | Eq. 7 |
| `test_mse_collapses_with_learned_prototypes_but_map_does_not` | Section II.C |
| `test_map_loss_prefers_separated_penalties` | the competitive term |
| `test_input_format` | 32×32, −0.1 / 1.175 |
| `test_distortion_with_zero_strength_is_identity` | the distortion code |
| `test_gauss_newton_diag_on_a_linear_layer` | h_kk = 2u² exactly (in expectation) |
| `test_sdlm_step_reduces_the_loss_and_schedule` | Appendix C, Section III.B |

---

## 5. Try it yourself

1. Replace `Subsample` with **max pooling** and the RBF output with a plain `Linear(84, 10)` + cross-entropy (the "modern LeNet"). Compare with `e1`.
2. Connect C3 **fully** (mask of all ones). How many parameters does that add? Does accuracy change?
3. Visualize C1's six 5×5 kernels after training. Do they look like edge detectors?
4. Use `e5`'s shifted test set with a network trained **with** distortions. Does the robustness improve?
5. Feed a 32×64 image (two digits side by side) into the trained network with F6 and the output applied at every position of C5's now 1×9 map. That's the space-displacement idea of Section VII.
