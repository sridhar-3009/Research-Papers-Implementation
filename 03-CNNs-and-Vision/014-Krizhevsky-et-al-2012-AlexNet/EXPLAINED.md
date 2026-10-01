# Krizhevsky, Sutskever & Hinton (2012), explained from scratch

**Paper:** *ImageNet Classification with Deep Convolutional Neural Networks* ("AlexNet")
**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (University of Toronto)
**Published at:** NIPS 2012

Read Paper 013 (LeNet-5) and Papers 009–010 (dropout) first. AlexNet is LeNet's idea made **much bigger**, with a few new tricks, trained on GPUs and on a dataset 20× larger than MNIST. This guide derives every shape and parameter count, computes where the work happens, and works each trick through with numbers.

---

## 0. The whole idea in one line

> **A big, deep convolutional network (60 million parameters), made trainable by ReLUs and GPUs and kept from overfitting by data augmentation and dropout, beat the best hand-engineered vision systems on ImageNet by a huge margin. This result started the deep-learning era.**

---

## 1. Why this paper mattered (Section 1)

- **The data:** ImageNet has 15 million labelled images in 22,000 categories. The ILSVRC competition uses **1.2 million** training images in **1,000** classes.
- **The model:** that much data needs high capacity, plus **prior knowledge** (the convolutional structure of Paper 013) so the capacity isn't wasted.
- **The hardware:** training took 5–6 days on two GTX 580 GPUs (3 GB each).
- **The result:** **15.3% top-5 error** at ILSVRC-2012, against **26.2%** for the second-best entry. A gap that size had never been seen, and it convinced the vision community almost overnight.

**Top-1 / top-5 error:** an image counts as wrong if the true label is not the model's single best guess (top-1), or not among its 5 best guesses (top-5). Top-5 forgives reasonable confusions among similar classes, such as two breeds of terrier.

---

## 2. The data (Section 2)

- **Resizing:** images come in all sizes. Each is rescaled so its **shorter side is 256**, then the **central 256×256** is cut out.
- **The only preprocessing:** subtract the **mean pixel value** (per position and channel) over the training set. The network learns from raw RGB; no hand-made features.

---

## 3. The architecture (Section 3)

Sections 3.1–3.4 are ordered by importance.

### 3.1 ReLU: f(x) = max(0, x), the biggest win
- **The problem with tanh:** its slope is f′(x) = 1 − tanh²x ≈ 4e^(−2|x|) for large |x|. It dies **exponentially fast**:
  - at x = 0.5 it's 0.786;
  - at x = 2 it's 0.071;
  - at x = 5 it's **0.00018** (our demo).

  A unit with a large input passes almost no gradient back. This is **saturation** (Paper 007).
- **ReLU's slope is exactly 1** for every positive input, so gradients pass unchanged. It's also cheaper to compute.
- **Figure 1:** a 4-layer CNN on CIFAR-10 with ReLUs reached 25% training error **six times faster** than the same net with tanh.
- **The authors:** without this speed-up they couldn't have experimented with nets this large.
- **The trade-off:** a ReLU whose input is always negative passes **no** gradient and can "die". The positive bias init (section 5) reduces this.

### 3.2 Training on two GPUs (grouped convolution)
- **Why split:** the net didn't fit in 3 GB, so **half of each layer's kernels live on each GPU**.
- **Where the GPUs communicate:** only at certain layers.
  - **conv3 and the FC layers read all maps.**
  - **conv2, conv4 and conv5 only read the maps on their own GPU.** That's why their kernels have depth 48 or 192 (half of 96 or 384) instead of the full depth.
- **The gain:** −1.7% / −1.2% top-1/top-5, compared with a one-GPU net that has half the kernels.
- **A side effect (Figure 3):** GPU 1's kernels became mostly **colour-blind** (edges, textures), and GPU 2's mostly **colour-specific**, on every run.
- **The modern name:** a **grouped convolution** (`groups=2`).

### 3.3 Local response normalization (LRN)
```
b^i_{x,y} = a^i_{x,y} / ( k + α Σ_{j = i−n/2}^{i+n/2} (a^j_{x,y})² )^β        k = 2, n = 5, α = 10⁻⁴, β = 0.75
```
Each map's activity at (x, y) is divided by a measure of how active the **n neighbouring maps** are at the **same position**: "lateral inhibition", as in real neurons.

**Worked numbers (our demo):**
- **All maps at 10:** the sum over 5 maps is 500, so the denominator is (2 + 10⁻⁴·500)^0.75 = 2.05^0.75 = 1.713, giving **10/1.713 = 5.84**.
- **One loud map at 300:**
  - its own sum is 300² + 4·10² = 90,400, so the denominator is (2 + 9.04)^0.75 = 6.06, giving **300/6.06 = 49.5**;
  - each neighbour within 2 maps also has 90,000 in its sum, which squashes it to **1.65**;
  - maps further away keep 5.88.
- **The loud map "inhibits" its neighbours.**

**Results:**
- −1.4% / −1.2% top-1/top-5;
- on CIFAR-10, a 4-layer CNN went from 13% to 11% error.

**Later:** VGG (Paper 015) found LRN didn't help, and Batch Norm (Paper 012) replaced it.

### 3.4 Overlapping pooling
- **The change:** max-pool **3×3 windows every 2 pixels**, so the windows overlap, instead of 2×2 every 2.
- **Same output size:** (55 − 3)/2 + 1 = **27**, and (55 − 2)/2 + 1 → 27 too.
- **The gain:** −0.4% / −0.3%, and the models were slightly harder to overfit.

### 3.5 Shapes: the output-size formula
For an n×n input, a k×k kernel, padding p and stride s:
```
output = ⌊ (n + 2p − k) / s ⌋ + 1
```
- **conv1:** n = 224, k = 11, s = 4, p = 0 gives (224 − 11)/4 + 1 = **54.25**, not an integer!
  - The paper's 55 needs either padding p = 2, giving (228 − 11)/4 + 1 = 55.25 → 55, or a 227×227 crop, giving (227 − 11)/4 + 1 = **55** exactly.
  - Implementations do one or the other.
- **pool1:** (55 − 3)/2 + 1 = 27.
- **conv2:** 5×5, pad 2: (27 + 4 − 5)/1 + 1 = 27, then pool → 13.
- **conv3–5:** 3×3, pad 1, keep 13, then pool → **6**.

### 3.6 The network (Figure 2) with every parameter count derived
| Layer | Kernels | Output | Parameters = kernels × (k·k·depth) + biases |
|---|---|---|---|
| conv1 (+ReLU, LRN, pool) | 96 @ 11×11×3, stride 4 | 96×55×55 → 27×27 | 96·363 + 96 = **34,944** |
| conv2 (+ReLU, LRN, pool) | 256 @ 5×5×**48** (per GPU) | 256×27×27 → 13×13 | 256·1200 + 256 = **307,456** |
| conv3 (+ReLU) | 384 @ 3×3×**256** (both GPUs) | 384×13×13 | 384·2304 + 384 = **885,120** |
| conv4 (+ReLU) | 384 @ 3×3×**192** (per GPU) | 384×13×13 | 384·1728 + 384 = **663,936** |
| conv5 (+ReLU, pool) | 256 @ 3×3×192 (per GPU) | 256×13×13 → 6×6 | 256·1728 + 256 = **442,624** |
| fc6 (+ReLU, dropout) | | 4096 | (256·6·6 = 9,216)·4096 + 4096 = **37,752,832** |
| fc7 (+ReLU, dropout) | | 4096 | 4096·4096 + 4096 = **16,781,312** |
| fc8 → softmax | | 1000 | 4096·1000 + 1000 = **4,097,000** |
| | | | **60,965,224 ≈ 61M** |

### 3.7 Where the parameters are vs where the work is
**Multiply-adds per image** (output positions × kernels × kernel volume):

| layer | M multiply-adds |
|---|---|
| conv1 | 55·55·96·363 = 105 |
| conv2 | 27·27·256·1200 = 224 |
| conv3 | 150 |
| conv4 | 112 |
| conv5 | 75 |
| fc6 | 9216·4096 = 38 |
| fc7 | 17 |
| fc8 | 4 |
| **total** | **≈ 724M**, of which the **conv layers do 92%** |

- **Parameters:** **96% are in the FC layers**, but the FC layers do only **8%** of the computation. A conv weight is reused at every position (Paper 013's sharing), while an FC weight is used once.
- **That's why dropout goes in fc6 and fc7:** that's where the overfitting capacity is.

---

## 4. Fighting overfitting (Section 4)

60M parameters and 1.2M images means the net overfits badly without help.

### 4.1 Data augmentation (computed on the CPU while the GPU trains, so it's free)
**1. Random crops and mirrors:**
- **Training:** a random 224×224 patch of the 256×256 image (32 × 32 positions), randomly flipped left–right (×2), gives **2,048 variants** of every image. The variants are highly correlated, but they teach the net that position and left–right orientation don't change the label.
- **Testing:** average the softmax outputs over **10 patches** (4 corners + centre, plus their mirrors). This gives 37.5% / 17.0% instead of 39.0% / 18.3% with one central crop. Averaging reduces the variance of the prediction, like a small ensemble.

**2. PCA colour augmentation:**
- **Step 1:** compute the 3×3 covariance of all RGB pixel values in the training set. Find its eigenvectors p₁, p₂, p₃ (the main directions in which colours vary across natural images) and eigenvalues λ₁, λ₂, λ₃ (how much they vary).
- **Step 2:** for each training image, draw α_i ~ N(0, 0.1²) **once**, and add the same RGB shift to **every pixel**:
  ```
  [p₁ p₂ p₃] · [α₁λ₁, α₂λ₂, α₃λ₃]ᵀ
  ```
- **Why it doesn't change the label:** changing the colour and brightness of the **illumination** doesn't change what the object is.
- **The gain:** −1% top-1.
- **Our demo:** eigenvalues 0.030, 0.053, 0.084, and one image got the shift (−0.0005, +0.0015, +0.0020): small, and the same for every pixel.

### 4.2 Dropout (Papers 009–010)
- **During training:** in fc6 and fc7, each unit is zeroed with probability 0.5.
- **At test time:** all units are kept and their outputs **× 0.5** (Paper 009's mean network).
- **The cost:** without dropout the net overfits substantially; with it, training needs **about twice** as many iterations.

---

## 5. Training (Section 5)

- SGD with **batch 128, momentum 0.9, weight decay 0.0005**:
  ```
  v ← 0.9·v − 0.0005·ε·w − ε·⟨∂L/∂w⟩_batch
  w ← w + v
  ```
  - The −0.0005·ε·w term is **weight decay**: it pulls every weight slightly toward 0 each step (Paper 005, section 5.3).
  - The paper found it **lowers the training error too**, so it isn't only a regularizer.
- **The loss:** the average −log P(correct label), i.e. softmax + cross-entropy (Paper 007, section 4.1).
- **Init:**
  - weights ~ N(0, 0.01²);
  - **biases = 1** in conv2, conv4, conv5 and the hidden FC layers, so the ReLUs start with positive inputs and pass gradients from the first step;
  - biases = 0 elsewhere.
- **Learning rate:** start at 0.01 and **÷10 when validation error stops improving**. That happened 3 times.
- **Length:** about 90 epochs, 5–6 days on two GPUs.

---

## 6. Results (Section 6)

**ILSVRC-2010 (Table 1):**

| Model | Top-1 | Top-5 |
|---|---|---|
| sparse coding (2010 winner) | 47.1% | 28.2% |
| SIFT + Fisher vectors | 45.7% | 25.7% |
| **CNN** | **37.5%** | **17.0%** |

**ILSVRC-2012 (Table 2), top-5:**

| Model | Val | Test |
|---|---|---|
| SIFT + FVs (second-best entry) | — | 26.2% |
| 1 CNN | 18.2% | — |
| 5 CNNs averaged | 16.4% | 16.4% |
| 1 CNN + extra conv layer, pretrained on ImageNet Fall 2011 | 16.6% | — |
| **7 CNNs** | 15.4% | **15.3%** |

**Ensembles:** averaging the softmax outputs of independently trained nets helps because their errors are partly independent (18.2% → 16.4% with 5).

**What the network learned (Section 6.1):**
- **conv1's kernels (Figure 3)** are oriented edges at several frequencies and coloured blobs, much like V1 cells in the visual cortex, learned purely from data.
- **Figure 4 (left):** even the mistakes are reasonable.
- **Figure 4 (right):** images whose **4096-d fc7 vectors** are close are semantically similar even when their pixels differ greatly. **fc7 is a general-purpose image representation.** This is the basis of transfer learning and image retrieval.
- **Depth matters:** removing any middle conv layer costs ~2% top-1.

---

## 7. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` tests each design choice on CIFAR-10 with a small 4-layer CNN, as the paper did for Figure 1 and LRN:
  - E1: ReLU vs tanh;
  - E2: LRN;
  - E3: overlapping pooling;
  - E4: augmentation;
  - E5: dropout.
- E6 trains the full AlexNet on any ImageNet-style folder, such as Imagenette.
- The paper's exact CIFAR net isn't given, so ours is a similar 3-conv + 1-FC network.

**Checked (tests and demo, a few seconds):**
- **Parameters:** 60,965,224, with **96% in the FC layers**; every per-layer count matches the formula above.
- **Neurons:** conv2–conv5 match Figure 2 exactly (186,624; 64,896; 64,896; 43,264). conv1 has 96·55·55 = **290,400**; the paper's 253,440 can't come from 96 maps of 55×55.
- **The two-GPU split:** changing "GPU 2"'s input maps doesn't change "GPU 1"'s conv4 outputs.
- **Init:** biases of 1 exactly where the paper says; weight std 0.01.
- **ReLU vs tanh slopes:** tanh 0.786 / 0.071 / 0.00018 at inputs 0.5 / 2 / 5; ReLU 1 throughout.
- **LRN:**
  - our formula matches PyTorch's `local_response_norm`, once you account for PyTorch dividing α by n;
  - the "loud map" example gives 49.53 for the loud map, 1.65 for its neighbours and 5.88 for maps further away.
- **Overlapping 3×3/2 and plain 2×2/2 pooling** both give 27×27 from 55×55.
- **Ten-crop:** corners, centre and mirrors are correct; 32·32·2 = 2,048.
- **PCA colour:** the eigen-decomposition reconstructs the RGB covariance, and each image gets **one** RGB shift for all its pixels.
- **Paper-style dropout:** keeps ~50% during training and gives exactly 0.5·x at test.
- **The update rule** equals `torch.optim.SGD(momentum=0.9, weight_decay=0.0005)`.
- **"÷10 on plateau"** behaves as described.

---

## 8. Check yourself

1. Compute tanh′(3) and ReLU′(3). What does that mean for gradients?
2. Use the output-size formula to show why 224 doesn't give 55 for conv1, and how 227 or padding fixes it.
3. Derive conv2's 307,456 parameters. Why is the kernel depth 48, not 96?
4. Why do the FC layers hold 96% of the parameters but do only 8% of the computation?
5. Compute LRN for 5 maps all at 20 (k = 2, α = 10⁻⁴, β = 0.75). (Sum 2,000 → (2.2)^0.75 = 1.806 → 11.07.)
6. Explain both augmentations. Why does PCA colour noise not change the label?
7. Why average the predictions over 10 crops? Over 7 networks?
8. What does Figure 4 (right) say about fc7, and why is that useful beyond ImageNet?
