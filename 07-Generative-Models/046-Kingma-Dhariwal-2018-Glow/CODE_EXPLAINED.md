# The code, explained simply

How the code in this folder implements Glow (Kingma & Dhariwal 2018).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `glow.py` | squeeze, actnorm, the invertible 1×1 convolution (plain and LU), fixed permutations, additive and affine coupling with a zero-initialised last conv, a flow step, the split prior, the multi-scale Glow (encode, log_prob, decode with temperature), dequantization, bits/dim |
| `experiments.py` | Figure 3 permutations, Table 2 (CIFAR-10), 5-bit CIFAR-10, a temperature grid, CelebA interpolation and attributes, LU timing (heavy, not run here) |
| `demo.py` | the exact change of variables, actnorm, the 1×1 conv and LU, a small trained flow, temperature, latent arithmetic (~4 seconds) |
| `test_glow.py` | 7 quick tests (~0.8 seconds) |

**Run it** (from `07-Generative-Models/046-Kingma-Dhariwal-2018-Glow`):
```
python3 -m pytest -q             # ~0.8 seconds
python3 demo.py                  # ~4 seconds
python3 experiments.py --quick
```

---

## 2. `glow.py`

### The layer interface
Every layer has:
- **`forward(x)`**, which returns (y, log-det per example);
- **`reverse(y)`**, which returns x.

### Layers
| Class | Forward | Log-det |
|---|---|---|
| `squeeze2d` / `unsqueeze2d` | 2×2 space ↔ 4 channels | 0 (no parameters) |
| `ActNorm` | y = x·e^{log s} + b; the first call sets b and log s from the batch statistics (an `initialised` buffer) | h·w·Σ log s |
| `InvConv1x1(C, lu)` | a 1×1 conv with W | plain: h·w·slogdet(W); LU: h·w·Σ log\|s\| |
| `Permute(C, mode)` | reverse or random channel order | 0 |
| `Coupling(C, hidden, additive)` | y_a = s·x_a + t from NN(x_b) | Σ log s |

**Details:**
- **`InvConv1x1` initialisation:** W comes from the QR decomposition of a random matrix (a rotation). The LU version stores P (fixed), sign(s) (fixed), the strictly-lower L, the strictly-upper U and log|s|.
- **`Coupling`:**
  - NN = conv3×3 → ReLU → conv1×1 → ReLU → `ZeroConv`;
  - `ZeroConv` is zero-initialised, with a learned output scale exp(3·logs), as in the released code;
  - affine: s = sigmoid(raw + 2) (the released code's choice);
  - additive: s = 1.
- **`FlowStep(C, hidden, perm, additive)`** = actnorm → `perm` ('conv' / 'lu' / 'reverse' / 'shuffle') → coupling.
- **`SplitPrior`:** a `ZeroConv` maps the kept half to (mean, log-std) for the half that leaves.

### The model
**`Glow(C, K, L, hidden, perm, additive, squeeze)`:**
- **`encode(x)`** returns (the list of z per level, the total log-det, the total log p(z)). Each split adds its learned-prior log-probability; the top z uses N(0, I).
- **`log_prob(x)`** = log p(z) + log-det (Eq. 6).
- **`decode(zs)`** inverts exactly.
- **`decode(n=…, shape=…, temperature=T)`** samples, multiplying every Gaussian's std by T.
- **`squeeze=False`** gives a flat flow for vector data (H = W = 1). The 1×1 conv is then just an invertible linear map.

### Data helpers
- **`dequantize(x_uint8, n_bits)`:**
  - reduces the bit depth;
  - scales to [−½, ½) and adds U(0, 1/2^bits);
  - returns the constant c = M log 2^bits.
- **`bits_per_dim`** = (−log p + c)/(M ln 2).

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 3: {additive, affine} × {reverse, shuffle, conv} × 3 seeds on CIFAR-10 |
| `e2` | Table 2: CIFAR-10, saving the model |
| `e3` | Table 3: 5-bit CIFAR-10 |
| `e4` | Figure 8: a temperature grid |
| `e5` | Figures 5–6 on CelebA 64×64, 5-bit: interpolation, and six attribute directions computed from labels after training |
| `e6` | plain vs LU: bits/dim and seconds per epoch |

**Training:** Adam with learning rate 10⁻³, gradient clipping, and actnorm initialised on a first batch.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_squeeze_is_an_invertible_reshape` | the shapes, the block layout, the exact inverse |
| `test_actnorm_data_dependent_init_and_logdet` | mean 0 / std 1 after init; log-det = autograd's; the inverse |
| `test_invertible_1x1_conv_plain_and_lu` | rotation init has log-det 0; log-det = autograd's; the inverse (both forms); permutation inverse |
| `test_coupling_starts_as_near_identity_and_is_invertible` | the additive coupling starts as the exact identity; the affine one starts with s = sigmoid(2); log-det = autograd's; the inverse |
| `test_full_glow_change_of_variables_and_inverse` | the whole 2-level model: Σ log-dets = log\|det\| of the full Jacobian; decode(encode(x)) = x |
| `test_temperature_and_learned_split_prior_sampling` | T = 0 gives identical samples; T = 0.5 is narrower than T = 1 |
| `test_dequantization_and_bits` | ranges and constants; the uniform density gives 8 bits/dim |

---

## 5. Try it yourself

1. In demo section 4, set `additive=True`. Is the NLL worse? Does T now give exactly p(x)^(1/T²)?
2. Compare `perm="reverse"` and `perm="conv"` on the 6-D moons over 5 seeds and 1000 steps. Does a gap appear?
3. Encode two test moons points and decode the straight line between their z's. Does the path stay on the data?
4. Use `perm="lu"` with C = 512 inside a flow step and time 100 forward passes against `perm="conv"`.
