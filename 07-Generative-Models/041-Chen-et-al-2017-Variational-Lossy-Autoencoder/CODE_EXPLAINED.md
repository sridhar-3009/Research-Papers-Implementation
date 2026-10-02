# The code, explained simply

How the code in this folder implements the Variational Lossy Autoencoder (Chen et al. 2017).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `vlae.py` | the window-limited 1-D decoder, exact A×B-window and stacked-3×3 PixelCNN decoders (optional grayscale context), the receptive-field probe, the AF prior, the VLAE model, bits-back code lengths, the toy data with its exact likelihood |
| `experiments.py` | lossy compression (KL bits + decompressions), Table 1, Tables 2–4 with the unconditional baseline, Figure 3 on CIFAR-10, the full toy sweep (heavy, not run here) |
| `demo.py` | information preference, bits-back accounting, receptive fields, AF = IAF (~11 seconds) |
| `test_vlae.py` | 9 quick tests (~1 second) |

`vlae.py` reuses `MADE`, `IAFPosterior` and the discretized logistic likelihood from paper 040, loaded by file path.

**Run it** (from `07-Generative-Models/041-Chen-et-al-2017-Variational-Lossy-Autoencoder`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~11 seconds
python3 experiments.py --quick
```

---

## 2. `vlae.py`

### Decoders
- **`WindowAR1d(length, window, z_dim)`:** a 1-D autoregressive Bernoulli decoder.
  - Each position sees `x_{i−w..i−1}`, taken with `unfold` on the left-padded sequence, plus z and a position embedding.
  - `window=0` is factorized; `window ≥ length−1` is fully autoregressive.
- **`window_mask(A, B)`, `MaskedConv2d`:**
  - the kernel is (B+1) × A;
  - the input is padded so the kernel's last row is the pixel's row and its column `(A−1)//2` is the pixel's column;
  - the mask keeps the B rows above and the (A−1)/2 pixels to the left.
- **`pixelcnn_mask(3, include_centre)`:** the standard masks "A" (without the centre) and "B" (with the centre), in the same layout.
- **`LocalPixelCNN(C, size, z_dim, channels, window, layers, gray_context, likelihood)`:**

  | setting | structure |
  |---|---|
  | `window=(A,B)` | one exact-window masked conv, then 1×1 convs |
  | `window=None` | 6 stacked 3×3 masked convs |

  - z enters every layer as an upsampled 7×7 feature map.
  - **Likelihood:** Bernoulli for binary data, discretized logistic for 8-bit images.
  - **`gray_context=True`:** the masked conv sees only the mean of the channels.
  - **`sample`:** pixel-by-pixel ancestral sampling.
- **`receptive_field(decoder, ...)`:** backprops from one output pixel and marks which inputs got a gradient.

### Prior and model
- **`AFPrior(D, T)`:**
  - **`inverse(z)`** whitens z → ε with log|det dε/dz| = −Σ log σ (one MADE pass per step, flips between steps);
  - **`log_prob`** is log N(ε) plus that log-det;
  - **`sample`** runs the sequential forward flow;
  - **init:** σ = 1 and μ = 0, so the prior starts as N(0, I).
- **`VLAE(encoder, enc_dim, z_dim, decoder, prior)`:**
  - **`terms`** returns (log p(x|z), log p(z), log q(z|x)) for n samples;
  - **`elbo`**, **`kl`** (nats in the code) and **`log_likelihood`** (importance sampling) are built from them.
- **`code_lengths`:** naive, bits-back and the refund (Eqs. 5–7).
- **Toy:** `toy_data`, `toy_true_log_prob` (exact, by summing over the global bit) and `train_toy`.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | static MNIST: VAE vs VLAE; KL in bits; decompression and sample figures |
| `e2` | Table 1: IAF VAE (040's IAF posterior), AF VAE, VLAE |
| `e3` | Tables 2–4: VLAE vs "Unconditional Decoder" on dynamic MNIST, OMNIGLOT and Caltech-101 (each downloaded from its original source) |
| `e4` | Figure 3: CIFAR-10 windows 4×2 / 5×3 / 7×4 / 7×4 grayscale; bits/dim, code KL, decompressions |
| `e5` | the toy over windows {0, 1, 2, 4, 8, 15} × 5 seeds × 5000 steps |

- **Training:** Adamax, gradient clipping.
- **Free bits** are applied to the **total** KL (λ × number of latents). The AF prior's KL doesn't split per dimension.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_window_receptive_field_is_exactly_a_by_b` | 4×2, 5×3, 7×4 windows match the definition exactly |
| `test_stacked_pixelcnn_is_causal_and_local` | the 6-layer stack never sees itself or the future, and can't see 7+ rows up |
| `test_grayscale_context_ignores_colour_of_the_window` | changing colour at constant gray changes nothing |
| `test_window_ar1d_sees_only_its_window` | windows 0 / 1 / 3 / 16 |
| `test_af_prior_density_and_inverse` | log-det = autograd's; the density integrates to 1 |
| `test_af_prior_equals_iaf_posterior_bound` | Eqs. 12–14 numerically |
| `test_bits_back_code_length_is_minus_elbo_and_refund_is_entropy` | the code-length bookkeeping |
| `test_vlae_shapes_bound_and_kl` | importance-sampled log p(x) ≥ bound; shapes |
| `test_discretized_logistic_reused_from_040` | the reuse works; the mask size is right |

---

## 5. Try it yourself

1. In the demo, train the full-window toy for 3000 steps. How close does its KL get to 0?
2. Add free bits (λ = 0.2 per latent) to the full-window toy. Does z now carry information? Is the bound worse?
3. Make the toy's local structure longer-range (copy x_{i−3} instead of x_{i−1}). Which window sizes now still need z?
4. Swap `prior="normal"` for `prior="af"` in `train_toy`. Does the bound improve?
