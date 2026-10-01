# He et al. (2016), explained from scratch

**Paper:** *Deep Residual Learning for Image Recognition* ("ResNet")
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
**Published at:** CVPR 2016 (arXiv:1512.03385); best paper award

Read Paper 015 (VGG) and Paper 012 (Batch Norm) first. This guide explains *why* adding x back makes deep nets trainable, with the gradient math of a residual stack, the cost of a bottleneck worked by hand, and our own gradient measurements.

---

## 0. The whole idea in one line

> **Instead of asking a few layers to learn a function H(x), let them learn the difference F(x) = H(x) − x, and add x back through a "shortcut". Extra layers can then always fall back to doing nothing, so networks can be 100+ layers deep and keep improving.**

Every modern deep network, including every Transformer layer (`x + Attention(x)`, `x + MLP(x)`), is built on this one line.

---

## 1. The problem: degradation (Section 1, Figure 1)

- **Initialization was no longer the obstacle.** VGG showed depth helps (Paper 015), and with BN (Paper 012) and good init (Papers 007, 015), very deep nets **do start training**.
- **The surprise:** a deeper plain net is **worse, even on the training set**. On CIFAR-10, a 56-layer plain net has **higher training error** than a 20-layer one. On ImageNet, 34 plain layers are worse than 18.
- **It's not overfitting.** Overfitting means low *training* error and high *test* error. Here the training error itself is higher.

### The construction argument
- Take the trained 20-layer net and append 36 layers that compute the **identity**. This 56-layer net has **exactly** the same error.
- So a 56-layer solution at least as good as the 20-layer one **exists**, but SGD doesn't find it.
- **Conclusion:** the problem is **optimization**. Stacks of non-linear layers find it hard to learn even the identity.

---

## 2. Residual learning (Section 3.1)

Let H(x) be what a few layers should compute. Instead, make them learn the **residual**:
```
F(x) := H(x) − x,          output = F(x) + x
```

**Why this is easier:**
- If the identity is (nearly) optimal, the layers just push **F → 0**: shrink the weights, which weight decay already encourages. A plain stack would have to *build* x → x through convolutions and ReLUs, a delicate thing to learn.
- If the best H is **close to** the identity (a small correction), learning the small correction is easier than learning everything from scratch. The paper calls this "preconditioning".

**The evidence (Figure 7):** learned residual functions have **small responses**, smaller than plain layers' and smaller still in deeper ResNets. Each block makes a small tweak.

---

## 3. Why shortcuts fix the gradients: the math

### 3.1 One block's Jacobian
- **The block:** y = x + F(x), so the Jacobian (how a small change in x moves y) is:
  ```
  ∂y/∂x = I + ∂F/∂x
  ```
- **The extra I is the key.** Even if ∂F/∂x is tiny, the gradient passes through at full strength.
- **Our test:** with F's output set to 0, ∂y/∂x is exactly 1 everywhere.

### 3.2 A whole stack, unrolled
- **For blocks l, l+1, …, L−1 with identity shortcuts** (and ignoring the ReLU after the addition, as the follow-up paper He et al. 2016b does):
  ```
  x_L = x_l + Σ_{i=l}^{L−1} F(x_i)                  (the deep output = the shallow input + a SUM of residuals)
  ∂Loss/∂x_l = ∂Loss/∂x_L · ( 1 + ∂/∂x_l Σ_{i=l}^{L−1} F(x_i) )
  ```
- **The gradient reaching layer l** always contains the term ∂Loss/∂x_L · 1, a **direct copy of the top gradient**, no matter how deep the network is. The second term would have to be exactly −1 to cancel it, which is very unlikely.
- **A plain net is different:** x_L = f_{L−1}(…f_l(x_l)), so ∂x_L/∂x_l is a **product** of L − l Jacobians. Products of many matrices tend to shrink or blow up exponentially (Paper 007's variance argument).
- **Residual nets replace a product with (roughly) a sum.**

### 3.3 Many short paths
- **Expanding the product:** expanding Π(I + J_i) over n blocks gives **2ⁿ paths**, each passing through some subset of the blocks.
- **Path lengths are spread out:** they follow a binomial distribution centred on n/2. With 54 blocks (ResNet-110), the most common length is 27 blocks, yet only 11% of paths have exactly that length.
- **Veit et al. (2016)** showed a ResNet behaves like an **ensemble of these paths**. Long paths pass through many shrinking Jacobians, so most of the gradient comes from relatively **short** paths (about 10–34 blocks in ResNet-110).

---

## 4. The building block (Section 3.2, Figure 2)

```
y = F(x, {W_i}) + x                 (Eq. 1)      F = W₂ · relu(BN(W₁ · x))   (BN after each conv)
output = relu(y)
```
- **Free:** the shortcut adds **no parameters and no computation** beyond one addition. So plain nets and ResNets can be compared **fairly**, with the same depth, width, parameters and FLOPs.
- **When the shapes differ** (more channels or stride 2): **y = F(x) + W_s·x** (Eq. 2), where W_s is a 1×1 conv.
- **F needs at least 2 layers.** With one layer, y = W₁x + x = (W₁ + I)x is just another linear layer and showed no advantage.

### 4.1 The three shortcut options (Table 3)
| Option | Shortcut when the shape changes | Elsewhere | ResNet-34 parameters | top-1 |
|---|---|---|---|---|
| plain | none | none | 21.8M | 28.54% |
| **A** | identity + **zero-padded** extra channels (and stride 2) | identity | 21.62M | 25.03% |
| **B** | **1×1 projection** | identity | 21.80M | 24.52% |
| **C** | projection | projection | 22.72M | 24.19% |

- **A, B and C are all far better than plain, and close to each other.** So projections are not what fixes degradation; **the identity path is**.
- **B** is the standard choice. C's small gain comes from its extra parameters.

---

## 5. The networks (Section 3.3, Table 1)

**The plain baseline (VGG-inspired):**
- mostly 3×3 convs;
- the same size keeps the same width, and **halving the size doubles the width**;
- downsampling is done by **stride-2 convs**, not pooling;
- the end is **global average pooling** + one 1000-way FC (no 100M-parameter FC layers as in VGG).

**ResNet** = the same, plus a shortcut around every pair of 3×3 convs.

| | 18 | 34 | 50 | 101 | 152 |
|---|---|---|---|---|---|
| block | basic (3×3, 3×3) | basic | **bottleneck** | bottleneck | bottleneck |
| blocks per stage | 2-2-2-2 | 3-4-6-3 | 3-4-6-3 | 3-4-23-3 | 3-8-36-3 |
| multiply-adds | 1.8 G | 3.6 G | 3.8 G | 7.6 G | 11.3 G |

### 5.1 The bottleneck, costed by hand (Figure 5, right)
**The bottleneck block:** 1×1 (256 → 64), then 3×3 (64 → 64), then 1×1 (64 → 256). Its multiply-adds **per pixel**:
```
256·64 + 9·64·64 + 64·256 = 16,384 + 36,864 + 16,384 = 69,632
```

| Compare with | per pixel |
|---|---|
| a **basic** block at 64 channels: 2·9·64·64 | 73,728: about the same cost, but the bottleneck has a **256-wide** input and output |
| a basic block at 256 channels: 2·9·256·256 | 1,179,648: **17× more** |

- **Why the shortcut must be an identity here:** a 1×1 projection 256 → 256 would cost 65,536 per pixel, **nearly doubling** the block.
- **Payoff:** ResNet-152 (11.3 G) is **cheaper than VGG-16** (15.3 G), while 8× deeper.

---

## 6. Training (Section 3.4)

**ImageNet:**
- **Data:** the shorter side random in [256, 480] (scale augmentation, Paper 015), then a random 224 crop, flip, mean subtraction and AlexNet's colour augmentation.
- **BN after every conv, before the ReLU.** **He initialization**: Var(W) = 2/fan-in, which makes Paper 015's per-layer variance factor n·Var(W)·½ exactly 1.
- **No dropout.**
- **SGD:** batch 256, **lr 0.1**, ÷10 on plateaus, up to 600k iterations; weight decay 1e-4, momentum 0.9.
- **Testing:** 10-crop for comparisons; fully convolutional multi-scale for the best results.

**CIFAR-10 (Section 4.2):**
- **Architecture:** a 3×3 conv with 16 filters, then 3 stages of n basic blocks (2n layers each) at 32×32 / 16×16 / 8×8 with 16 / 32 / 64 filters, then global average pooling + a 10-way FC.
  - That's 1 + 6n + 1 = **6n + 2** layers: n = 3 gives 20, n = 9 gives 56, n = 18 gives 110, and n = 200 gives 1202.
  - Option A shortcuts.
- **Hand count for ResNet-20:**
  - the conv and FC weights total 268,346;
  - adding BN's γ, β makes **0.27M** ✔.
- **Training:** batch 128, lr 0.1, ÷10 at 32k and 48k iterations, stop at 64k; weight decay 1e-4. Augmentation: pad 4, random 32×32 crop, flip.
- **ResNet-110:** lr 0.1 is "slightly too large to start converging", so it **warms up** at 0.01 until training error < 80% (~400 iterations). This is an early form of the learning-rate warm-up now used for every Transformer.

---

## 7. Results

### 7.1 ImageNet: degradation fixed (Table 2, Figure 4)
| | plain | ResNet |
|---|---|---|
| 18 layers | 27.94% | 27.88% |
| 34 layers | **28.54%** (worse than 18) | **25.03%** (better than 18) |

- **Plain-34 has higher training error throughout.** The authors checked that its gradients have healthy norms and **guessed** the plain nets have "exponentially low convergence rates".
- **At 18 layers** both work, but the ResNet converges faster.

### 7.2 Going deeper (Tables 3–5)
| Model | top-1 (10-crop) | top-5 |
|---|---|---|
| VGG-16 | 28.07% | 9.33% |
| ResNet-34 B | 24.52% | 7.46% |
| ResNet-50 | 22.85% | 6.71% |
| ResNet-101 | 21.75% | 6.05% |
| **ResNet-152** | **21.43%** | **5.71%** |

- **Single-model ResNet-152** (multi-scale): **4.49% top-5**, better than all previous *ensembles*.
- **An ensemble of 6:** **3.57% top-5** on the test set, **1st place at ILSVRC 2015**.
- **Beyond classification:** it also won 1st in ImageNet detection and localization and in COCO detection and segmentation. Swapping VGG-16 → ResNet-101 in Faster R-CNN alone gave **+28% relative** on COCO.

### 7.3 CIFAR-10 (Table 6, Figure 6)
| Network | Parameters | Test error |
|---|---|---|
| ResNet-20 | 0.27M | 8.75% |
| ResNet-32 | 0.46M | 7.51% |
| ResNet-44 | 0.66M | 7.17% |
| ResNet-56 | 0.85M | 6.97% |
| **ResNet-110** | 1.7M | **6.43%** |
| ResNet-1202 | 19.4M | 7.93% |

- **Plain nets get worse with depth** (plain-110 has > 60% error); **ResNets get better**.
- **ResNet-1202** trains fine (training error < 0.1%) but tests worse than ResNet-110. That's **overfitting**: 19.4M parameters on 50k images, with no dropout or maxout. Depth is no longer an optimization problem, only a regularization one.

---

## 8. Why it matters

- **Shortcuts are everywhere:** Transformers, U-Nets, diffusion models and LLMs all use x + f(x).
- **Pre-activation ResNets** (He et al. 2016b) move BN and ReLU **before** the convs, so the shortcut is a pure identity end to end (section 3.2's math becomes exact). 1001-layer nets then train even better.
- **The ensemble view** (Veit et al. 2016): many paths, with the gradient carried mostly by the shorter ones (section 3.3).

---

## 9. What our code found

**Scale note:** at your request, nothing was trained on this laptop. `experiments.py` reproduces the CIFAR-10 results:
- E1: plain vs ResNet at 20–110 layers (Figure 6, Table 6);
- E2: layer responses (Figure 7);
- E3: options A/B/C;
- E4: ResNet-1202 (optional).

**Checked (tests and demo, a few seconds):**
- **Table 1's compute, measured:** ResNet-18/34/50/101/152 = 1.81 / 3.66 / 3.86 / 7.57 / 11.28 G multiply-adds (paper: 1.8 / 3.6 / 3.8 / 7.6 / 11.3).
  - This needs the paper's original bottleneck, with **stride 2 in the first 1×1** conv. torchvision's "v1.5" puts it in the 3×3, which costs 4.1 G for ResNet-50.
  - Measured on "meta" tensors, using no memory.
- **Parameters (option B):** 11.69 / 21.80 / 25.56 / 44.55 / 60.19M, identical to torchvision.
- **Table 6:** 0.27 / 0.46 / 0.66 / 0.85 / 1.7 / 19.4M parameters for 20 / 32 / 44 / 56 / 110 / 1202 layers, **exactly the paper's**.
- **No extra parameters:** plain and residual (option A) nets have identical counts; A < B < C.
- **The construction argument, run for real:** a 32-layer ResNet holding a 20-layer ResNet's weights, with its 6 extra blocks' residuals zeroed, gives **exactly** the same outputs.
- **The shortcut carries the gradient:** with F = 0, ∂y/∂x = 1 exactly.
- **The gradient at the first layer, at initialization (with BN):**

  | Layers | plain | residual |
  |---|---|---|
  | 20 | 2.1 | 0.8 |
  | 56 | 1,089 | 3.9 |
  | 110 | **4,723,074** | 9.0 |

  - **The plain net's gradient doesn't vanish; it explodes** (×2.2 million from 20 to 110 layers). The residual net's grows only ×11.
  - The paper only reported "healthy norms" and guessed at slow convergence. **Yang et al. (2019) later proved that BN causes exactly this explosion in deep plain nets**, and that shortcuts prevent it.
  - Our measurement is at initialization only; the paper looked during training.

---

## 10. Check yourself

1. What is degradation? Why is it not overfitting?
2. Explain the construction argument. What does it say about SGD?
3. Why is learning F = 0 easier than learning H = identity?
4. Show that ∂y/∂x = I + ∂F/∂x for one block. Unroll x_L = x_l + Σ F(x_i) and explain the "1 +" in the gradient.
5. Why does a plain net's gradient involve a *product* of Jacobians, and why is that dangerous?
6. Cost a bottleneck block per pixel. Why must its shortcut be an identity?
7. Derive "6n + 2" for the CIFAR ResNets. What n gives 110 layers?
8. What do options A, B and C do, and what did Table 3 conclude?
9. Why did ResNet-1202 test worse than ResNet-110?
