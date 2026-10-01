# Krizhevsky, Sutskever & Hinton (2012), explained simply

**Paper:** *ImageNet Classification with Deep Convolutional Neural Networks* ("AlexNet")
**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (University of Toronto)
**Published at:** NIPS 2012

Read Paper 013 (LeNet-5) and Papers 009–010 (dropout) first. AlexNet is LeNet's idea made **much bigger**, with a few new tricks, trained on GPUs and on a dataset 20 times larger than MNIST.

---

## The big idea in one line

> **A big, deep convolutional network (60 million parameters), made trainable by ReLUs and GPUs and kept from overfitting by data augmentation and dropout, beat the best hand-engineered computer-vision systems on ImageNet by a huge margin. This result started the deep learning era.**

---

## 1. Why this paper mattered (Section 1)

- **The data had arrived.** ImageNet has 15 million labeled images in 22,000 categories. The ILSVRC competition uses 1.2 million training images in 1,000 classes.
- **A model for that much data** needs large capacity, but also **prior knowledge** (the convolutional structure of Paper 013).
- **GPUs** made training feasible: the whole network trained in 5–6 days on two GTX 580s (3 GB each).
- **The result:** 15.3% top-5 error at ILSVRC-2012, against **26.2%** for the second-best entry. A gap that size was unheard of, and it convinced the computer-vision community.

**Top-5 error** means the correct label is not among the model's 5 most likely guesses.

---

## 2. The data (Section 2)

- Images come in all sizes. Each is rescaled so the **shorter side is 256**, then the **center 256×256** is cropped.
- The only preprocessing is **subtracting the mean pixel value** of the training set. The network learns from raw RGB.

---

## 3. The architecture (Section 3)

Sections 3.1–3.4 are ordered by importance, most important first.

### 3.1 ReLU: f(x) = max(0, x) (the biggest win)
- tanh and sigmoid **saturate**: for large inputs, their slope is ≈ 0, so gradients vanish (Paper 007).
- ReLU's slope is 1 for any positive input.
- **Figure 1:** a 4-layer CNN with ReLUs reached 25% training error on CIFAR-10 **six times faster** than the same net with tanh.
- Without this speed-up, the authors say they couldn't have experimented with such large networks.

### 3.2 Training on two GPUs
- The net didn't fit in 3 GB, so **half of the kernels sit on each GPU**.
- The GPUs **only communicate in certain layers**:
  - conv3 and the fully connected layers see everything;
  - conv2, conv4 and conv5 only see maps on their own GPU.
- This split lowered top-1/top-5 error by 1.7% / 1.2% compared with a one-GPU net with half as many kernels.
- **A side effect (Figure 3):** the kernels on GPU 1 became mostly **colour-blind** (edges, textures), while the ones on GPU 2 became mostly **colour-specific**. This happened on every run.
- Today we'd call this a **grouped convolution** (`groups=2`).

### 3.3 Local response normalization (LRN)
```
b^i = a^i / (k + α Σ_{j = i−n/2 … i+n/2} (a^j)²)^β      k = 2, n = 5, α = 10⁻⁴, β = 0.75
```
- It divides each activity by the activity of the **neighbouring kernel maps at the same position**. This is "lateral inhibition", as in real neurons.
- It's applied after ReLU in conv1 and conv2.
- It gave −1.4% / −1.2% top-1/top-5 error, and on CIFAR-10 a 4-layer CNN went from **13% to 11%** error.
- **Later:** VGG (Paper 015) found LRN didn't help, and Batch Normalization (Paper 012) replaced it.

### 3.4 Overlapping pooling
- Pooling windows of **3×3 every 2 pixels** (they overlap), instead of the usual 2×2 every 2. Both give the same output size.
- It gave −0.4% / −0.3% top-1/top-5, and the models found it slightly harder to overfit.

### 3.5 The whole network (Figure 2)

| Layer | Kernels | Output | Parameters |
|---|---|---|---|
| input | | 3 × 224 × 224 | |
| conv1 + ReLU + LRN + max-pool | 96 @ 11×11×3, stride 4 | 96 × 55 × 55 → 27 × 27 | 34,944 |
| conv2 + ReLU + LRN + max-pool | 256 @ 5×5×48 (per GPU) | 256 × 27 × 27 → 13 × 13 | 307,456 |
| conv3 + ReLU | 384 @ 3×3×256 (both GPUs) | 384 × 13 × 13 | 885,120 |
| conv4 + ReLU | 384 @ 3×3×192 (per GPU) | 384 × 13 × 13 | 663,936 |
| conv5 + ReLU + max-pool | 256 @ 3×3×192 (per GPU) | 256 × 13 × 13 → 6 × 6 | 442,624 |
| fc6 + ReLU + dropout | | 4096 | 37,752,832 |
| fc7 + ReLU + dropout | | 4096 | 16,781,312 |
| fc8 → softmax | | 1000 | 4,097,000 |
| | | | **≈ 61 million** |

- The training objective is the average **log-probability of the correct label** (softmax + cross-entropy).
- **96% of the parameters are in the three fully connected layers**, but most of the computation is in the conv layers.
- **Two small inconsistencies in the paper:**
  - With a 224×224 input, an 11×11 kernel at stride 4 gives (224 − 11)/4 + 1 = **54.25**, not 55. Implementations either pad by 2 or use 227×227 crops.
  - Figure 2's caption lists 253,440 neurons for conv1, but 96 × 55 × 55 = **290,400**. The other layers' counts (186,624; 64,896; 64,896; 43,264) match exactly.

---

## 4. Fighting overfitting (Section 4)

60 million parameters with 1.2 million images overfits badly. The paper uses two fixes.

### 4.1 Data augmentation (computed on the CPU while the GPU trains, so effectively free)
1. **Random crops and mirror images:**
   - **Training:** random 224×224 patches of the 256×256 images, flipped left-right at random. That's 32 × 32 × 2 = **2048** variations of each image.
   - **Testing:** 10 patches (4 corners + the center, plus their mirror images), averaging the softmax outputs. Without this averaging, the error is 39.0% / 18.3% instead of 37.5% / 17.0%.
2. **PCA colour augmentation:**
   - Compute the principal components of all RGB pixel values in the training set.
   - For each image, add `[p1 p2 p3]·[α1λ1, α2λ2, α3λ3]ᵀ` to **every pixel**, with α_i ~ N(0, 0.1²) drawn once per image.
   - This mimics changes in the **colour and intensity of the lighting**, which don't change what the object is.
   - It gave −1% top-1.

### 4.2 Dropout (Papers 009–010)
- In fc6 and fc7, each unit is set to 0 with probability 0.5 during training. At test time all units are kept and their outputs are **multiplied by 0.5**.
- Without dropout the net overfit substantially. **With it, training takes about twice as many iterations.**

---

## 5. How it was trained (Section 5)

- SGD, **batch size 128, momentum 0.9, weight decay 0.0005**:
  ```
  v ← 0.9·v − 0.0005·ε·w − ε·⟨∂L/∂w⟩_batch
  w ← w + v
  ```
  - The paper found that this small weight decay **lowers the training error too**. It isn't only a regularizer.
- **Initialization:**
  - weights ~ N(0, 0.01²);
  - biases = 1 in conv2, conv4, conv5 and the hidden fc layers (so the ReLUs start with positive inputs and learn early);
  - biases = 0 elsewhere.
- **Learning rate:** start at 0.01 and **divide by 10 whenever the validation error stops improving**. That happened 3 times.
- **About 90 epochs**, taking 5–6 days on two GPUs.

---

## 6. Results (Section 6)

**ILSVRC-2010 (Table 1)**, where test labels are available:

| Model | Top-1 | Top-5 |
|---|---|---|
| Sparse coding (the 2010 winner) | 47.1% | 28.2% |
| SIFT + Fisher vectors (best published) | 45.7% | 25.7% |
| **CNN** | **37.5%** | **17.0%** |

**ILSVRC-2012 (Table 2)**, top-5:

| Model | Val | Test |
|---|---|---|
| SIFT + FVs (second-best entry) | — | 26.2% |
| 1 CNN | 18.2% | — |
| 5 CNNs averaged | 16.4% | 16.4% |
| 1 CNN with an extra conv layer, pre-trained on all of ImageNet Fall 2011 (15M images, 22K classes) | 16.6% | — |
| **7 CNNs (5 + 2 pre-trained)** | 15.4% | **15.3%** |

**What the network learned (Section 6.1):**
- **conv1's kernels (Figure 3):** oriented edges at many frequencies and coloured blobs. That's much like the V1 cells of the visual cortex, learned from data.
- **Figure 4, left:** even the mistakes are reasonable (e.g. confusing kinds of cats).
- **Figure 4, right:** images whose **4096-d fc7 vectors** are close are semantically similar, even when their pixels are very different (dogs in different poses). **The last hidden layer is a good general-purpose image representation.** This idea powers transfer learning and image retrieval.
- **Depth matters:** removing any single middle conv layer made top-1 about 2% worse.

---

## 7. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` tests each design choice on CIFAR-10 with a small 4-layer CNN, as the paper did for Figure 1 and the LRN claim:
  - E1: ReLU vs tanh;
  - E2: LRN;
  - E3: overlapping pooling;
  - E4: augmentation;
  - E5: dropout.
- E6 trains the full AlexNet on any ImageNet-style folder, such as Imagenette.
- The exact CIFAR net from the paper isn't given (it points to cuda-convnet files), so ours is a similar 3-conv + 1-fc network.

**Checked (tests and demo, a few seconds):**
- **Parameters:** 60,965,224 ("60 million"), with **96% in the fc layers**. Each layer's count matches the formula.
- **Neurons:** conv2–conv5 match Figure 2 exactly: 186,624; 64,896; 64,896; 43,264. conv1 has 290,400; the paper's 253,440 can't come from 96 maps of 55×55.
- **The two-GPU split:** changing "GPU 2"'s input maps doesn't change "GPU 1"'s conv4 outputs.
- **Initialization:** biases of 1 exactly where the paper says; weight std = 0.01.
- **ReLU vs tanh:** at input 5, tanh's slope is 0.00018 while ReLU's is still 1.
- **LRN:**
  - our hand-written formula matches PyTorch's `local_response_norm`, once you account for PyTorch dividing α by n;
  - one loud map (300) cuts its 4 neighbours from 5.8 to 1.65 and leaves maps further away alone.
- **Overlapping 3×3/2 and plain 2×2/2 pooling** both give 27×27 from 55×55.
- **Ten-crop:** the corners, center and mirror images are where they should be. 32 × 32 × 2 = 2048.
- **PCA colour:** the eigen-decomposition reconstructs the RGB covariance, and each image gets **one** RGB shift, the same for every pixel.
- **Paper dropout:** keeps ~50% during training and gives exactly 0.5·x at test.
- **Update rule:** the paper's formula gives the same weights as `torch.optim.SGD(momentum=0.9, weight_decay=0.0005)`.
- **"Divide by 10 on plateau"** behaves as described.

---

## 8. Check yourself

1. Why do ReLUs make training faster than tanh? What is "saturation"?
2. Which layers of AlexNet hold most of the parameters, and which do most of the computation? Why does that decide where dropout goes?
3. How does the two-GPU split show up in the kernel shapes (5×5×48, 3×3×192)?
4. What does LRN do, and why is it called "brightness normalization"?
5. Explain both kinds of data augmentation. Why does PCA colour noise not change the label?
6. Why average predictions over 10 crops at test time?
7. What does Figure 4 (right) tell you about the fc7 features, and why is that useful beyond ImageNet?
