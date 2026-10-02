# The code, explained simply

How the code in this folder implements PixelCNN++ (Salimans et al. 2017).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `pixelcnnpp.py` | the discretized logistic mixture (log-probability, sampling, RGB coupling), the softmax and dequantized ablation likelihoods, shifted convolutions and deconvolutions, the gated ResNet, the full two-stream U-net PixelCNN++ (with switches for short-cuts, downsampling and class conditioning), sampling |
| `experiments.py` | Table 1, class-conditional, small receptive fields, the four ablations, a 12×12 MNIST short-cut study (heavy, not run here) |
| `demo.py` | edge mass, mixture vs softmax, dequantization, causality vs blind spot, output sizes (~1.5 seconds) |
| `test_pixelcnnpp.py` | 8 quick tests (~0.6 seconds) |

**Run it** (from `07-Generative-Models/045-Salimans-et-al-2017-PixelCNN-plus-plus`):
```
python3 -m pytest -q             # ~0.6 seconds
python3 demo.py                  # ~1.5 seconds
python3 experiments.py --quick
```

---

## 2. `pixelcnnpp.py`

### Pixel values
Like the released code, pixel values are mapped to [−1, 1]:
- integer k sits at 2k/255 − 1 (`to_pm1`);
- one bin is ±1/255 wide (`HALF_BIN`).

### Likelihood
- **`_log_bin_prob(x, μ, log s)`:**
  - log[σ((x + ½bin − μ)/s) − σ((x − ½bin − μ)/s)];
  - at x = 0 it uses log σ(upper); at x = 255 it uses log(1 − σ(lower));
  - when the bin mass underflows, it falls back to the density × bin width.
- **`n_params`, `split_params`:** the network output per pixel is [K logits | means (C×K) | log-scales (C×K) | coupling coefficients (C(C−1)/2 × K, tanh)].
- **`discretized_mix_logistic_log_prob(x, l, K)`:**
  - shifts green's mean by α·r and blue's by β·r + γ·g;
  - adds the channels' log-probabilities **within** each component (they share the component);
  - takes a logsumexp over components, weighted by π.
- **`sample_discretized_mix_logistic`:**
  1. pick one component per pixel;
  2. sample r, then g (mean shifted by r), then b, using logistic noise μ + s·log(u/(1−u));
  3. snap to the 256 levels.

### Ablation likelihoods
| Function | What it computes |
|---|---|
| `softmax_log_prob` | the 256-way softmax likelihood |
| `dequantized_log_density` | the continuous mixture density of x + u (per 8-bit unit), whose average is a lower bound |
| `bits_per_dim` | −log p / (D · ln 2) |

### Network
- **Shifted convolutions:**
  - `DownShiftedConv` (2×3) and `DownRightShiftedConv` (2×2) pad so the output sees only rows above, or rows above plus the left;
  - the matching deconvolutions crop to stay causal;
  - `down_shift` and `right_shift` move a feature map by one row or column.
- **`GatedResnet(nf, conv, skip, dropout, cond)`:** concat-ELU → conv → (+ 1×1 conv of the skip input) → concat-ELU → dropout → conv → (+ class bias) → x + a·σ(b).
- **`PixelCNNpp(C, nr_resnet, nf, K, dropout, shortcuts, downsample, n_classes)`:**
  - **initial streams:** u from a 2×3 down-shifted conv; ul from a 1×3 down-shifted conv plus a 2×1 down-right-shifted conv. A channel of ones marks real pixels against padding.
  - **down pass:** 3 levels × nr_resnet gated layers, with stride-2 convolutions between levels. Every output is kept on a stack.
  - **up pass:** pops the stack. With `shortcuts=True`, each layer receives the matching down-pass output as its skip input.
  - **output:** a 1×1 conv to the mixture parameters.
  - **`log_prob`** sums the per-pixel log-probabilities; **`sample`** generates pixel by pixel.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 1 (unconditional CIFAR-10) |
| `e2` | class-conditional CIFAR-10 |
| `e3` | small-receptive-field models (`downsample=False`, 1–2 layers per block; the field is measured by autograd) |
| `e4` | ablations: `SoftmaxPixelCNN` (1536 outputs per pixel), dequantized training (evaluated with its bound), no short-cuts, no dropout |
| `e5` | short-cuts on 12×12 MNIST over seeds |

**Training:** Adam with learning rate 10⁻³ and decay 0.999995 per step (as in the released code).

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_single_channel_mixture_sums_to_one_and_matches_eq2` | normalisation; Eq. 2 directly |
| `test_edges_keep_the_tails` | P(0) = σ(upper edge); P(0) > P(1) > P(2) when the mean sits below 0 |
| `test_rgb_coupling_and_normalisation_per_channel` | Σ_b p(r, g, b) equals the explicit mixture of P(r)·P(g\|r) |
| `test_sampling_matches_the_probabilities` | the empirical histogram ≈ the model's probabilities |
| `test_stable_log_prob_for_tiny_scales` | a finite, very negative log-probability |
| `test_network_is_causal_with_no_blind_spot` | the output at (4, 4) sees exactly the pixels above and to the left |
| `test_variants_shapes_and_class_conditioning` | all variants run; the class bias changes the output; 100 outputs for K = 10 |
| `test_softmax_and_dequantized_ablations` | the softmax likelihood; the dequantized bound ≤ the discrete log-probability and close to it; 8 bits for uniform |

---

## 5. Try it yourself

1. In demo section 2, raise the training set to 5000 values. When does the softmax catch up?
2. Set K = 1 in `fit_mixture` for MNIST. How much worse is it than K = 5?
3. Use `PixelCNNpp(..., downsample=False)` with `nr_resnet=1` and measure its receptive field with `receptive_field_size` from `experiments.py`.
4. Sample a few 8×8 images from an untrained model with `model.sample(4, 8, 8)`. What do they look like?
