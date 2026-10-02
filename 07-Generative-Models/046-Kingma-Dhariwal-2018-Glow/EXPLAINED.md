# Glow, explained simply

**Paper:** Diederik P. Kingma & Prafulla Dhariwal, *Glow: Generative Flow with Invertible 1×1 Convolutions*, NeurIPS 2018.

**In one sentence:** build an image model out of **invertible** layers, so an image can be turned into a Gaussian latent code and back **exactly**. Use the change-of-variables formula to get the **exact likelihood**. Glow's new pieces:
- **actnorm**;
- a learned **invertible 1×1 convolution** instead of a fixed channel shuffle;
- a simplified architecture.

Together they made flows good enough to generate realistic 256×256 faces.

---

## 1. The big idea: normalizing flows

### 1.1 Three families of likelihood-based models (Section 1)
| family | example | strength | weakness |
|---|---|---|---|
| autoregressive | PixelCNN (paper 045) | exact likelihood | slow, sequential sampling |
| VAEs | paper 039 | fast sampling | only a lower bound; approximate inference |
| **flows** | NICE, RealNVP, Glow | exact likelihood, exact inference, parallel sampling | every layer must be invertible, with a cheap determinant |

### 1.2 The generative story (Eqs. 3–5)

```
z ~ N(0, I),      x = g(z),      z = f(x) = g⁻¹(x)
f = f₁ ∘ f₂ ∘ … ∘ f_K:      x ↔ h₁ ↔ h₂ ↔ … ↔ z
```

g is invertible (a **bijection**), so:
- **inference is exact:** z = f(x), with no encoder approximation;
- **sampling is one parallel pass:** x = g(z).

### 1.3 Change of variables (Eqs. 6–8)

```
log p(x) = log p(z) + log |det(dz/dx)| = log p(z) + Σ_i log |det(dh_i / dh_{i−1})|
```

**Intuition:** when f squeezes a region of x-space into a small region of z-space, the density in x-space must be *higher* there to keep the total probability 1. The determinant measures that squeezing factor.

**1-D worked example:**
- z ~ N(0, 1) and x = 2z + 1, so f(x) = (x − 1)/2 and dz/dx = ½.
- At x = 2: z = 0.5.
- log p(x) = log N(0.5; 0, 1) + log ½ = −1.0439 − 0.6931 = **−1.7370**.
- Check: x ~ N(1, 4), and log N(2; 1, 4) = −½ log(8π) − ⅛ = −1.6121 − 0.125 = **−1.7371** ✓.

**The catch:**
- For a general D×D Jacobian, the determinant costs O(D³). A 256×256×3 image has D = 196,608.
- So every layer is designed so that its Jacobian is **triangular** (or block-diagonal). Then log|det| is just the sum of the logs of the diagonal (Eq. 8).

**Our demo:** a small Glow on a 4×4 image. The sum of per-layer log-dets is **2.526668**, and the log|det| of the full 16×16 Jacobian from autograd is **2.526669**.

---

## 2. One step of flow (Figure 2a, Table 1)

```
actnorm  →  invertible 1×1 conv  →  affine coupling
```

All three act on a tensor of shape h × w × c.

### 2.1 Actnorm (Section 3.1)

```
y_{i,j} = s ⊙ x_{i,j} + b          reverse: x = (y − b)/s          log-det: h · w · Σ log|s|
```

- A per-channel scale and bias, like batch norm, but **initialised from data**: on the first minibatch, s and b are set so every channel's output has mean 0 and variance 1. After that they are ordinary trainable parameters.
- **Why not batch norm?** Its noise grows as the batch shrinks. High-resolution Glow trains with **batch size 1 per GPU**.
- **Why h·w in the log-det?** The same scale is applied at every one of the h·w pixel positions, so the Jacobian is diagonal with each s_c repeated h·w times.
- **Our demo:** channels with means (2.88, −1, 0) and stds (4.99, 0.2, 1) come out with mean 0 and std 1.

### 2.2 Invertible 1×1 convolution (Section 3.2), the paper's main idea
- Coupling layers (below) change only **half** the channels at a time. Between couplings, the channels must be **mixed** so every channel eventually gets transformed.
- **RealNVP:** reverse the channel order (a fixed permutation).
- **Glow:** a **learned** c×c matrix W, applied at every pixel (a 1×1 convolution).

```
y_{i,j} = W x_{i,j}          reverse: x = W⁻¹ y          log-det: h · w · log|det W|       (Eq. 9)
```

- **It generalises permutations:** a permutation matrix *is* a valid W, with |det| = 1.
- **Initialisation:** a random **rotation** (orthogonal matrix), so log|det W| = 0 at the start.

**LU trick (Eqs. 10–11):** computing det W costs O(c³). Parameterise instead:

```
W = P · L · (U + diag(s))        log|det W| = Σ log|s|
```

| factor | what it is | trained? |
|---|---|---|
| P | a permutation | fixed |
| L | lower-triangular with ones on the diagonal | yes |
| U | upper-triangular with zeros on the diagonal | yes |
| s | a vector | yes |

- The determinant of a triangular matrix is the product of its diagonal; P and L contribute ±1 and 1.
- **Our demo:** for c = 512, slogdet takes about 0.56 ms against about 0.001 ms for Σ log|s|.

### 2.3 Affine coupling (Section 3.3, from NICE/RealNVP)

```
x_a, x_b = split(x)                 (half the channels each)
(log s, t) = NN(x_b)
y_a = s ⊙ x_a + t,     y_b = x_b
log-det = Σ log|s|
```

**Why it's invertible and cheap:**
- y_b = x_b, so the reverse can recompute NN(y_b) = NN(x_b) and solve x_a = (y_a − t)/s.
- NN can be **any** network: it never needs inverting.
- The Jacobian is block-triangular: ∂y_b/∂x_a = 0 and ∂y_a/∂x_a = diag(s). So log|det| = Σ log|s|.

**Worked example** (2 channels, one pixel):
- x = (x_a, x_b) = (3, 1), and NN(1) gives log s = 0.5, t = −1.
- y_a = e^{0.5}·3 − 1 = 1.6487·3 − 1 = 3.946, and y_b = 1.
- Reverse: (3.946 + 1)/e^{0.5} = 4.946/1.6487 = 3 ✓.
- log-det = 0.5.

**Details:**
- **Additive coupling:** s = 1 (log-det 0), the NICE version.
- **Zero initialisation:** the last convolution of NN starts at zero, so every coupling starts as (almost) the identity. This helps very deep flows train.
- **The scale in the released code:** s = sigmoid(raw + 2) rather than exp(log s). That keeps 0 < s < 1, starting near 0.88, for stability. We follow the code and note the difference.
- **NN() in the paper:** 3×3 conv → ReLU → 1×1 conv (512 channels) → ReLU → 3×3 zero-initialised conv.

---

## 3. The multi-scale architecture (Figure 2b, from RealNVP)

```
for each of L levels:
    squeeze: (h, w, c) → (h/2, w/2, 4c)       each 2×2 block of pixels becomes 4 channels
    K steps of flow
    split: half the channels leave as z_l (except at the last level)
```

- **Squeezing** lets 1×1 operations mix spatial neighbours.
- **Splitting early** saves compute, and gives a hierarchy:
  - early z's hold fine detail;
  - later z's hold coarse structure.
- **Learned prior:** each split-off z_l gets a learned prior N(μ, σ) computed from the remaining half (a zero-initialised conv, as in the released code).
- **The paper's CIFAR-10 model:** L = 3, K = 32.

---

## 4. Continuous data from discrete pixels (Eq. 2)

- Pixels are integers, but a flow models densities.
- **The fix:** add uniform noise, x̃ = x + u with u ~ U(0, a), where a is one bin width. Then minimise

  ```
  −log p(x̃) + c,       c = −M · log a      (M = the number of dimensions)
  ```

  This is the code length in nats.
- **Example:** 8-bit pixels scaled to [−½, ½] have a = 1/256, so c = M·log 256.
  - A flat density p = 1 on the unit cube gives −0 + M log 256 nats = **8 bits/dim**, the uniform baseline.
- **5-bit images:** a = 1/32. The paper uses 5-bit for faces: slightly less colour fidelity, better visual quality.

---

## 5. Temperature (Section 6)

- Sample z ~ N(0, T²I) instead of N(0, I).
- Lower T stays nearer the typical, high-density region: less diverse but cleaner samples.
- **Exactness:**
  - in z-space this samples ∝ p(z)^{1/T²};
  - for **additive** couplings (volume-preserving) the same holds in x-space;
  - for affine couplings it is an approximation.
- The paper picks **T = 0.7** for faces.

**Our demo:**

| T | spread in the moon plane | mean log p(sample) |
|---|---|---|
| 0 | 0 (the single point z = 0) | 7.43 |
| 0.5 | 0.38 | 6.72 |
| 0.7 | 0.50 | 6.01 |
| 1.0 | 0.66 | 4.48 |

---

## 6. Results

### Table 2: bits per dimension (lower is better)

| Model | CIFAR-10 | ImageNet 32×32 | ImageNet 64×64 | LSUN bedroom | LSUN tower | LSUN church |
|---|---|---|---|---|---|---|
| RealNVP | 3.49 | 4.28 | 3.98 | 2.72 | 2.81 | 3.08 |
| **Glow** | **3.35** | **4.09** | **3.81** | **2.38** | **2.46** | **2.67** |

**5-bit versions (Table 3):**

| dataset | bits/dim |
|---|---|
| CIFAR-10 | 1.67 |
| ImageNet 32 | 1.99 |
| ImageNet 64 | 1.76 |
| CelebA-HQ 256 | 1.03 |

### Figure 3: permutation choice (CIFAR-10, K = 32, L = 3, 3 seeds)
- **1×1 conv beats a fixed shuffle, which beats reversal**, for both additive and affine couplings. The 1×1 conv also converges faster.
- Affine couplings converge faster than additive ones.
- The cost is about 7% more wall-clock time.

### Qualitative results (CelebA-HQ 256×256, K = 32, L = 6)
- **Samples:** realistic faces from a likelihood model. A 256×256 image takes about 130 ms on a 1080 Ti.
- **Interpolation:** encode two real faces, linearly interpolate the z's, and decode. Nearly every intermediate image is a realistic face (Figure 5).
- **Attribute manipulation (Figure 6):**
  - take the average z of faces *with* an attribute (smiling, blond, young, male, …) minus the average *without*;
  - adding that direction to any face's z changes that attribute;
  - the labels are used only after training, so the method needs very little supervision.

---

## 7. Why it works (the intuition)

1. **Invertibility gives everything at once:** exact likelihood (no bound), exact encoding (no approximate encoder), and parallel sampling (no pixel-by-pixel loop).
2. **Triangular Jacobians make it affordable.** Coupling and LU-decomposed 1×1 convs have log-dets that are sums, never determinants of huge matrices.
3. **Learned mixing beats fixed shuffling.** A coupling only transforms half the channels, so *how* the halves are chosen matters. A learned rotation-like W picks good mixtures.
4. **Actnorm and zero-init make deep flows trainable** at batch size 1: every layer starts normalised and near the identity.

---

## 8. What our code found

All numbers come from `demo.py` (about 4 s) and `test_glow.py` (7 tests, about 0.8 s).

1. **The change of variables is exact:**
   - for every layer type (actnorm, plain and LU 1×1 conv, additive and affine coupling), the cheap log-det matches the autograd Jacobian's log|det|;
   - so does the whole multi-scale model (2.526668 vs 2.526669);
   - the inverse reconstructs to about 5·10⁻⁷.
2. **Actnorm:** the first minibatch sets mean 0 and std 1 per channel.
3. **The 1×1 conv:**
   - a channel reversal is the special case W = permutation;
   - the rotation init gives log|det| ≈ 0;
   - the LU log-det is much faster at c = 512.
4. **A small flow on "two moons" hidden in 6-D:**
   - the test NLL falls from −1.82 to −4.55 nats per point in 150 steps;
   - the reconstruction error is 6·10⁻⁷.
5. **Temperature:** lower T means less spread and higher model log-density (the table in Section 5).
6. **Latent arithmetic:** moving 5 points from moon 0 along z̄₁ − z̄₀ shifts their mean from (0.39, 0.77) to (1.36, 0.04), toward moon 1's centre (1, −0.14).
7. **Honest note on permutations:** in one quick test on the same toy (150 steps, one seed), reversal, shuffle and 1×1 conv differed by under 0.1 nats (shuffle ≈ conv, slightly better than reversal). That is too small and noisy to confirm Figure 3. `experiments.py --only e1` runs the real CIFAR-10 comparison.

**Not run (too heavy):**
- E1: Figure 3 on CIFAR-10;
- E2: Table 2's CIFAR-10 number;
- E3: 5-bit CIFAR-10;
- E4: temperature grid;
- E5: CelebA interpolation and attribute manipulation at 64×64;
- E6: LU vs plain timing.

---

## 9. Check yourself

1. x = 3z with z ~ N(0, 1). What is log p(x = 3)?
   <details><summary>Answer</summary>z = 1, and log N(1) = −1.419. Then log|dz/dx| = log(1/3) = −1.099, so log p(x) = −2.518.</details>
2. Why is the coupling layer's log-determinant just Σ log|s|?
   <details><summary>Answer</summary>y_b = x_b and y_a = s ⊙ x_a + t(x_b): the Jacobian is block-triangular, with an identity block and diag(s) on the diagonal.</details>
3. An actnorm layer on a 16×16×8 tensor with all scales s = 2. What is its log-det?
   <details><summary>Answer</summary>16 · 16 · 8 · log 2 = 2048 · 0.693 = 1419.6.</details>
4. Why initialise W as a rotation?
   <details><summary>Answer</summary>A rotation is invertible, well conditioned, and has log|det| = 0, so the layer starts volume-preserving and mixes channels evenly.</details>
5. What does the LU decomposition buy, and what is fixed in it?
   <details><summary>Answer</summary>log|det W| becomes Σ log|s|: O(c) instead of O(c³). The permutation P stays fixed; L, U and s are learned.</details>
6. 8-bit data, M = 3072, and a model with −log p(x̃) = 5000 nats. What is the bits/dim?
   <details><summary>Answer</summary>(5000 + 3072 · ln 256)/(3072 · ln 2) = (5000 + 17035)/2129.3 = 10.35 bits/dim. That is worse than uniform (8), so the model is very bad.</details>
