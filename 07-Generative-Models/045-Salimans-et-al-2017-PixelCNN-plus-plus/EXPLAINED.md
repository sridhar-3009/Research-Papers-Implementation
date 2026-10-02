# PixelCNN++, explained simply

**Paper:** Tim Salimans, Andrej Karpathy, Xi Chen & Diederik P. Kingma, *PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications*, ICLR 2017.

**In one sentence:** a PixelCNN generates an image one pixel at a time, each pixel predicted from the ones above and to its left. PixelCNN++ makes five practical changes:
1. a smarter probability for each pixel value (a "discretized logistic mixture" instead of 256 separate probabilities);
2. predicting whole pixels;
3. downsampling;
4. long skip connections;
5. dropout.

Together they set the CIFAR-10 likelihood record: **2.92 bits per sub-pixel**.

---

## 1. Background: autoregressive image models

### 1.1 The chain rule
Any joint distribution can be written as a product of conditionals:

```
p(x) = p(x₁) · p(x₂ | x₁) · p(x₃ | x₁, x₂) · …  = Π_i p(x_i | x_<i)
```

- For an image, order the pixels row by row (raster order).
- Each pixel's distribution depends on all pixels **above it** and **to its left**.
- This is exact, not an approximation, so the model has a **tractable likelihood**: log p(x) is just a sum of per-pixel log-probabilities.

### 1.2 PixelCNN (van den Oord et al. 2016)
- A convolutional network computes every conditional p(x_i | x_<i) **at once** during training.
- **Masked convolutions** stop each output from seeing the current or future pixels.
- **Generation** is still sequential: sample pixel 1, feed it back, sample pixel 2, and so on.

### 1.3 Measuring quality: bits per sub-pixel
- bits/dim = −log₂ p(x) / (number of sub-pixels). A CIFAR-10 image has 32·32·3 = 3072 sub-pixels.
- A uniform guess over 256 values costs log₂ 256 = **8 bits**.
- PixelCNN++ gets **2.92**: it compresses each colour value to under 3 bits on average.

---

## 2. Modification 1: discretized logistic mixture likelihood (Section 2.1)

### 2.1 The old way: a 256-way softmax
The original PixelCNN outputs 256 probabilities for every sub-pixel. Problems:
- **Memory:** 256 numbers × 3 channels × every pixel.
- **The model doesn't know that 127 and 128 are close.** Each value is a separate category, so the network must *learn* that neighbouring intensities behave alike.
- **Sparse gradients:** each training example only pushes up one of the 256 outputs.
- A value never seen in training gets ~0 probability.

### 2.2 The new way
Imagine a hidden **continuous** intensity ν drawn from a mixture of K logistic distributions, which is then **rounded** to the nearest integer:

```
ν ~ Σ_i π_i · logistic(μ_i, s_i)                                                     (Eq. 1)
P(x) = Σ_i π_i [ σ((x + 0.5 − μ_i)/s_i) − σ((x − 0.5 − μ_i)/s_i) ]                    (Eq. 2)
```

- **Why the logistic?** Its CDF is the sigmoid σ. So the probability of the rounding bin [x − 0.5, x + 0.5] is just a difference of two sigmoids. There is no integral to compute.
- **Edge cases:**
  - for x = 0, replace x − 0.5 by −∞, so the bin gets **all** the mass below 0.5;
  - for x = 255, replace x + 0.5 by +∞.
  - This naturally puts extra probability on 0 and 255, which real images need: saturated pixels are common. The paper's Figure 1 shows a CIFAR-10 spike at 255.
- **Few parameters:** about 5–10 components × (weight, mean, scale) per channel, instead of 256 logits.

### 2.3 Worked example (one component)
- μ = 100, s = 5, and we ask for P(x = 102):
  - upper: σ((102.5 − 100)/5) = σ(0.5) = 0.6225;
  - lower: σ((101.5 − 100)/5) = σ(0.3) = 0.5744;
  - **P(102) = 0.0481**.
- For comparison, P(100) = σ(0.1) − σ(−0.1) = 0.5250 − 0.4750 = **0.0500**, the peak.
- **Edge:** P(0) = σ((0.5 − 100)/5) − 0 = σ(−19.9) ≈ 2·10⁻⁹.
- The 256 values' probabilities always sum to exactly 1. Our test checks this.

### 2.4 Numerical stability
- When s is tiny and x is far from μ, the two sigmoids are both 0 or both 1, and the difference underflows.
- The released code then uses the **density** at the bin centre × the bin width instead. Our `_log_bin_prob` does the same, and a test checks it stays finite.

---

## 3. Modification 2: conditioning on whole pixels (Section 2.2)

**The original:** predict R, then G given R, then B given R and G, each as a separate step inside the network. This needs complicated channel-wise masks.

**PixelCNN++:** the network sees only **whole pixels** up and to the left, and outputs a joint distribution for all 3 channels of the current pixel:

```
p(r, g, b | C) = P(r | μ_r, s_r) · P(g | μ_g + α·r, s_g) · P(b | μ_b + β·r + γ·g, s_b)        (Eq. 3)
```

- **One** mixture component is chosen per pixel and shared by all 3 channels.
- Green's mean shifts linearly with red; blue's mean shifts with red and green.
- α, β, γ are outputs of the network (passed through tanh).
- **A typo in the paper:** Eq. 3 prints "γ·b". It must be γ·g, since b is what is being predicted. The released code uses g.
- **Why it's enough:** colour channels within a pixel are strongly but **simply** (nearly linearly) related. A mixture of Gaussians with linear coupling would be exactly a mixture of full-covariance 3-D Gaussians, and logistics are very close to Gaussians.

**Outputs per pixel** with K = 10:
- 10 × (1 weight + 3 means + 3 log-scales + 3 coupling coefficients) = **100**;
- the paper's softmax version needs **1536**.

---

## 4. Modifications 3–4: downsampling and short-cut connections (Sections 2.3–2.4)

### 4.1 Two-stream causal convolutions
**Two feature streams:**

| stream | what it sees | kernel |
|---|---|---|
| **u** (down) | everything **above** the current row | 2×3 "down-shifted" convolutions |
| **ul** (down-right) | everything above **and** to the left | 2×2 "down-right-shifted" convolutions |

- **Shifting:** the input is padded and the output shifted by one row (or one column), so a pixel never sees itself.
- **No blind spot:** stacked plain masked 3×3 convolutions miss a wedge of pixels in the upper right. The two-stream design sees **all** earlier pixels. Our demo prints both receptive fields.

### 4.2 Downsampling instead of dilation
- To model long-range structure the network needs a big receptive field.
- PixelCNN++ uses **stride-2 convolutions**:
  - resolutions 32 → 16 → 8, then back up with transposed convolutions;
  - this is cheaper than dilated convolutions (each downsampling makes the maps 4× smaller).

### 4.3 Long short-cuts (U-net style)
- Downsampling throws away detail. To recover it, **every layer of the downward pass feeds directly into the matching layer of the upward pass**.
- In Figure 2, layer k connects to layer K − k.
- Without them, **the model fails to train** (Figure 7).
- **Our observation** (one development run on 12×12 MNIST, 300 steps):
  - **2.31 bits/dim with** short-cuts;
  - **2.76 bits/dim without**.

### 4.4 Gated ResNet layers
Each layer computes:

```
c = conv(concat_elu(x)) [+ conv1x1(concat_elu(skip))]
c = conv(dropout(concat_elu(c)))
x + a · sigmoid(b),  where (a, b) = split(c)
```

- **`concat_elu(x)`** = ELU([x, −x]). It keeps both signs of information.
- **The gate a·σ(b)** is borrowed from Gated PixelCNN.

**The full CIFAR-10 model:**
- 6 blocks of 5 gated layers;
- 160–192 filters (the paper says 192);
- our code gives **53.6M** parameters with 160 filters.

---

## 5. Modification 5: dropout (Section 2.5)

- PixelCNNs are powerful enough to **overfit badly**.
- Dropout (rate 0.5) on the residual path, after the first convolution, fixes this.
- **Without dropout (Section 3.4.4):**
  - train **< 2.0** bits but test **> 6.0** bits;
  - samples from the overfitted model also look **worse**, not better (Figure 8).

---

## 6. Results

### Table 1: CIFAR-10 (bits per sub-pixel; lower is better)

| Model | bits |
|---|---|
| Deep Diffusion | 5.40 |
| NICE | 4.48 |
| DRAW | 4.13 |
| Deep GMMs | 4.00 |
| Conv DRAW | 3.58 |
| Real NVP | 3.49 |
| PixelCNN | 3.14 |
| VAE with IAF (paper 040) | 3.11 |
| Gated PixelCNN | 3.03 |
| PixelRNN | 3.00 |
| **PixelCNN++** | **2.92** |

The class-conditional version gets 2.94: conditioning made overfitting harder to avoid.

### Table 2: small receptive fields (no downsampling, few layers)

| Model | bits |
|---|---|
| field 11×5, plain | 3.11 |
| field 11×5, NIN | 3.09 |
| field 11×5, autoregressive channel | 3.07 |
| field 15×8, plain | 3.07 |
| field 15×8, NIN | 3.04 |
| field 15×8, autoregressive channel | 3.03 |

- **A surprise:** even an 11×5 field beats the original PixelCNN (3.14), which had a blind spot.
- But samples lack global structure (Figure 5). Likelihood rewards local statistics most.
- This connects to paper 041: a small-window decoder models local texture.

### Ablations (Section 3.4)
| change | result |
|---|---|
| softmax instead of mixture | trains more slowly, each epoch takes longer |
| dequantize and use a continuous mixture | **3.11** bits (a bound) vs 2.92 |
| no short-cuts | fails to train |
| no dropout | train < 2.0, test > 6.0 |

**Why dequantization loses:**
- Adding uniform noise u to each integer and fitting a continuous density gives only a **lower bound**: E_u[log p_c(x + u)] ≤ log ∫_bin p_c = log P_discrete(x). This is Jensen's inequality.
- The discretized model is scored on exactly the probability of the integer, so it pays nothing extra.

---

## 7. Why it works (the intuition)

1. **Respect what pixels are:** ordered integers from a smooth underlying intensity. A logistic mixture builds in "nearby values are similar", so the network learns faster and generalises to unseen values.
2. **Keep simple things simple:** the dependencies between R, G and B in one pixel are nearly linear, so they don't need a deep network.
3. **Multi-scale plus shortcuts:** coarse resolutions see far, and shortcuts restore the fine detail that downsampling loses.
4. **Regularise a model that can memorise:** dropout turns a 6-bit overfitter into a 2.92-bit generaliser.

---

## 8. What our code found

All numbers come from `demo.py` (about 1.5 s) and `test_pixelcnnpp.py` (8 tests, about 0.6 s).

1. **The likelihood is right:**
   - the 256 bin probabilities sum to 1, and match Eq. 2 exactly;
   - the edge bins collect the tails;
   - the RGB coupling matches an explicit component-wise computation;
   - sampling reproduces the probabilities (total variation < 0.04);
   - the stable path stays finite at tiny scales.
2. **Figure 1 on MNIST:**
   - P(0) = **0.808**, a huge edge spike that the edge bin captures exactly (fitted 0.808);
   - a 5-component mixture is within **0.006 bits** of the empirical entropy (1.974 vs 1.968 bits/pixel);
   - **honest note:** MNIST peaks at 254, not 255 (the paper's 255 spike is a CIFAR-10 feature), and the fitted mixture overestimates P(255) (0.0095 vs 0.0064).
3. **Mixture vs softmax** (300 training values, 134 of the 256 values never seen):

   | model | parameters | train bits | test bits |
   |---|---|---|---|
   | logistic mixture | 15 | 7.101 | **7.146** |
   | 256-way softmax | 256 | **6.617** | 8.696 |
   | true distribution | | | 7.127 |

   The softmax memorises the training values; the mixture generalises.
4. **Dequantization:** with the same parameters, the discrete likelihood is 7.1463 bits and the dequantized bound 7.1544 bits. That is a small, always-positive gap, as Jensen predicts. The paper's trained-model gap was much larger (3.11 vs 2.92).
5. **Causality:**
   - a PixelCNN++ output at (4, 4) depends on **exactly** the 36 earlier pixels;
   - the 4-layer stacked masked PixelCNN misses the upper-right wedge (the blind spot);
   - the class-conditional bias changes the outputs.
6. **Short-cuts** (one development run, 12×12 MNIST, 300 steps): 2.31 bits/dim with them vs 2.76 without. `experiments.py --only e5` repeats this over seeds.

**Not run (too heavy):**
- E1: Table 1 on CIFAR-10;
- E2: class-conditional;
- E3: small receptive fields (plain only; NIN and autoregressive-channel variants are not implemented);
- E4: the softmax / dequantized / no-shortcut / no-dropout ablations.

---

## 9. Check yourself

1. A logistic with μ = 50 and s = 2. What is P(x = 50)?
   <details><summary>Answer</summary>σ(0.25) − σ(−0.25) = 0.5622 − 0.4378 = 0.1244.</details>
2. Why does P(255) get extra mass, and why is that good?
   <details><summary>Answer</summary>Its bin extends to +∞, so it collects the whole upper tail. Real images have many saturated (255) sub-pixels, more than a smooth density would put in that single value.</details>
3. How many numbers per pixel does a K = 5 RGB mixture need?
   <details><summary>Answer</summary>5 × (1 + 3 + 3 + 3) = 50.</details>
4. Why is dequantized training only a lower bound on the discrete likelihood?
   <details><summary>Answer</summary>By Jensen's inequality, E_u[log p(x + u)] ≤ log E_u[p(x + u)], and E_u[p(x + u)] (times the bin width) is the discrete bin probability.</details>
5. What goes wrong without the long short-cut connections?
   <details><summary>Answer</summary>Downsampling discards fine detail that the upward pass can't recover, so the model fails to train (Figure 7).</details>
6. Why might a model with an 11×5 receptive field still get a good likelihood?
   <details><summary>Answer</summary>Most of an image's bits are in local statistics, which a small window predicts well. Global structure costs comparatively few bits, so samples can look incoherent while the likelihood stays good.</details>
