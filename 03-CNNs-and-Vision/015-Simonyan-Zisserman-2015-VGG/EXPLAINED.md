# Simonyan & Zisserman (2015), explained simply

**Paper:** *Very Deep Convolutional Networks for Large-Scale Image Recognition* ("VGG")
**Authors:** Karen Simonyan, Andrew Zisserman (Visual Geometry Group, University of Oxford)
**Published at:** ICLR 2015 (arXiv:1409.1556)

Read Paper 014 (AlexNet) first. VGG keeps AlexNet's recipe and changes one thing on purpose: **depth**.

---

## The big idea in one line

> **Use only tiny 3×3 filters everywhere, and stack many of them. A stack of 3×3 layers sees as much as a big filter, with fewer weights and more non-linearities. Going from 11 to 16–19 layers clearly improves accuracy.**

---

## 1. The question (Section 1)

- After AlexNet, people improved ConvNets in many ways at once: smaller first-layer filters, dense multi-scale evaluation, and more.
- This paper isolates **one** factor, depth. Every other design choice is fixed, and the network is made deeper step by step.
- That's possible because the filters are tiny: a 3×3 layer adds few parameters, so you can afford many of them.

---

## 2. The design (Section 2.1)

- **Input:** 224×224 RGB, with the mean RGB value subtracted. No other preprocessing.
- **Conv layers:** always **3×3**, the smallest size that still captures left/right, up/down and center. **Stride 1, padding 1**, so the size stays the same.
- **Pooling:** 5 max-pools of 2×2 with stride 2. Each one halves the size: 224 → 112 → 56 → 28 → 14 → 7.
- **Width:** starts at 64 channels and **doubles after each pool**, up to 512.
- **The head:** FC-4096, FC-4096, FC-1000 + softmax, the same in every network.
- **ReLU everywhere.** No LRN, except in one test network.

---

## 3. The six networks (Table 1) and their sizes (Table 2)

| Net | Weight layers | What's new | Parameters |
|---|---|---|---|
| **A** | 11 (8 conv + 3 FC) | the base | 133M |
| **A-LRN** | 11 | A + AlexNet's LRN after the first layer | 133M |
| **B** | 13 | two 3×3 convs in the first two blocks | 133M |
| **C** | 16 | B + one **1×1** conv in each of the last 3 blocks | 134M |
| **D** ("VGG-16") | 16 | B + one more **3×3** in each of the last 3 blocks | 138M |
| **E** ("VGG-19") | 19 | four 3×3 convs in each of the last 3 blocks | 144M |

**Adding 8 conv layers (A → E) adds only 11M parameters.** About 103M of D's 138M parameters are in the first FC layer (7·7·512 → 4096).

---

## 4. Why 3×3? (Section 2.3)

**Receptive field:** a stack of two 3×3 layers (no pooling between them) sees a **5×5** window of its input, and three see **7×7**.

So why not use one 7×7 layer? Three 3×3 layers are better in two ways:
1. **More non-linearity:** three ReLUs instead of one, which makes the function more discriminative.
2. **Fewer parameters:** with C channels in and out,
   - three 3×3 layers: 3·(3²C²) = **27C²**;
   - one 7×7 layer: 7²C² = **49C²**, which is 81% more.

   You can see this as a **regularization**: the 7×7 filter is forced to decompose into 3×3 pieces with non-linearities between them.

**1×1 convolutions (config C):** a 1×1 conv is a linear map across channels at each pixel. Followed by a ReLU, it adds non-linearity **without changing the receptive field**. This idea comes from "Network in Network" and is used heavily by GoogLeNet and ResNet.

---

## 5. Training (Section 3.1)

- **Same as AlexNet:** SGD, **batch 256**, momentum 0.9, weight decay 5·10⁻⁴, dropout 0.5 in the first two FC layers. The learning rate starts at 10⁻² and is divided by 10 when validation accuracy stops improving (3 times). They stopped at **370K iterations (74 epochs)**, fewer than AlexNet's 90.
  - The authors' guess for why it needed fewer epochs: the implicit regularization of depth and small filters, plus pre-initialization.
- **Initialization was a real problem.** "Bad initialisation can stall learning due to the instability of gradient in deep nets."
  1. First train **net A**, which is shallow enough for random initialization.
  2. Then start the deeper nets with **A's first four conv layers and its three FC layers**. The layers in between are random.
  - Random weights: "normal distribution with zero mean and 10⁻² **variance**", i.e. std 0.1. (Many people read it as std 0.01. See section 9.)
  - Added after submission: **Glorot initialization** (Paper 007) works without this pre-training step.
- **Augmentation:** random 224×224 crops, flips, and AlexNet's RGB colour shift.
- **Training scale S** (the shorter side of the rescaled image the crop is taken from):
  - **Single-scale:** S = 256 or S = 384. The 384 net starts from the 256 net, with a lower learning rate of 10⁻³.
  - **Multi-scale ("scale jittering"):** each image gets its own random **S ∈ [256, 512]**, so objects appear at many sizes. This is augmentation by scale.

---

## 6. Testing (Section 3.2)

- **Dense evaluation:**
  1. Rescale the test image so its shorter side is Q (Q may differ from S).
  2. **Turn the FC layers into convolutions:** FC-4096 on 7×7×512 becomes a 7×7 conv, and the last two become 1×1 convs.
  3. Run this **fully convolutional** network over the **whole image**. You get a **map of class scores**.
  4. Average the map, and average with the horizontally flipped image.

  There's no need to compute many crops one by one.
- **Multi-crop:** 5×5 grid × 2 flips = 50 crops per scale, 150 over 3 scales. It's slightly better, and **complementary** to dense evaluation, because the two treat borders differently: a crop is padded with zeros, while dense evaluation sees the real neighbouring pixels.
- **Training cost:** 4 Titan Black GPUs with data parallelism (3.75× faster than 1 GPU), **2–3 weeks per network**.

---

## 7. Results on ILSVRC-2012 (Section 4)

### Table 3: one test scale (validation error, %)
| Net | train S | test Q | top-1 | top-5 |
|---|---|---|---|---|
| A | 256 | 256 | 29.6 | 10.4 |
| A-LRN | 256 | 256 | 29.7 | 10.5 |
| B | 256 | 256 | 28.7 | 9.9 |
| C | 256 / 384 / [256;512] | 256 / 384 / 384 | 28.1 / 28.1 / 27.3 | 9.4 / 9.3 / 8.8 |
| D | 256 / 384 / [256;512] | 256 / 384 / 384 | 27.0 / 26.8 / **25.6** | 8.8 / 8.7 / **8.1** |
| E | 256 / 384 / [256;512] | 256 / 384 / 384 | 27.3 / 26.9 / **25.5** | 9.0 / 8.7 / **8.0** |

What it shows:
1. **LRN doesn't help** (A-LRN ≈ A), so it's dropped. It only costs memory and time.
2. **Deeper is better**, from 11 layers (A) to 16–19 (D, E). It **saturates at 19** on this dataset.
3. **C < D**: the 1×1 layers help (C beats B), but 3×3 layers that **capture spatial context** help more.
4. **Small filters beat big ones:** B with each pair of 3×3 replaced by one 5×5 was **7% worse in top-1**.
5. **Scale jittering in training** ([256; 512]) clearly beats a fixed S, even when testing at one scale.

### More test scales, crops, ensembles (Tables 4–7)
- Testing at 3 scales: D and E reach **24.8% / 7.5%** top-1/top-5.
- **Dense + multi-crop:** E reaches 24.4% / **7.1%** (Table 5).
- **2 nets (D + E):** 23.7% / **6.8%** val, and **6.8% top-5 test**.
- **The ILSVRC-2014 submission** (7 nets): 7.3% test, **2nd place** in classification and **1st in localization** (Appendix A).

| Method (Table 7) | top-5 test error |
|---|---|
| AlexNet (5 nets, 2012) | 16.4% |
| OverFeat (7 nets, 2013) | 13.6% |
| Zeiler & Fergus (6 nets, 2013) | 14.8% |
| GoogLeNet (1 net) | 7.9% |
| GoogLeNet (7 nets, the 2014 winner) | **6.7%** |
| **VGG (1 net)** | **7.0%**: best single network |
| **VGG (2 nets)** | **6.8%** |

---

## 8. Transfer (Appendix B): VGG features are general

- Take the trained D or E, remove the last layer, and use the **4096-d output of the penultimate layer** as an image descriptor:
  - computed densely, averaged over positions and flips, across several scales;
  - L2-normalized;
  - fed to a **linear SVM**, without fine-tuning.
- **Results:**
  - **VOC-2007 89.3% / VOC-2012 89.0% mAP** (D or E alone; 89.7 / 89.3 for D + E), more than **6% above** the previous best;
  - **Caltech-101 92.7%, Caltech-256 86.2%** (D + E).
- This is why VGG-16 became the **default feature extractor** for years: for detection, segmentation, style transfer and perceptual losses.

---

## 9. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` tests each claim on CIFAR-10 with VGG networks shrunk in width (channels / 4) but with the **paper's depths and layer patterns**:
  - E1: depth A → E;
  - E2: 3×3 pairs vs one 5×5;
  - E3: initialization of the 19-layer net;
  - E4: scale jittering;
  - E5: dense vs multi-crop.

**Checked (tests and demo, about a second):**
- **Table 2, exactly:** A 132.9M, B 133.0M, C 133.6M, D 138.4M (138,357,544), E 143.7M (143,667,240). They round to the paper's 133 / 133 / 134 / 138 / 144. The full-size nets were built on PyTorch's "meta" device (shapes only, no memory), so this cost nothing.
- **Depths of Table 1:** 11, 11, 13, 16, 16, 19 weight layers; 5 pools; widths 64 → 512.
- **Two 3×3 convs really see a 5×5 window:** the gradient of one output reaches exactly rows and columns 3–7 of the input.
- **27C² vs 49C²:** the ratio is 1.81 ("81% more").
- **The FC → conv conversion:** on the training size, the fully convolutional net gives **exactly** the original outputs. On bigger images it gives a score map (e.g. a 160×160 image → 4×4 map).
- **Initialization**, measured on E's 16 conv layers at full width:

  | Init | Activation std after 16 layers |
  |---|---|
  | std 0.1 (as written) | **explodes to 10⁷** |
  | std 0.01 | **vanishes to 10⁻⁹** |
  | Glorot | still shrinks ~1000× (0.17 → 0.00012) |

  Glorot's shrinking happens because ReLU zeroes half its inputs, and Glorot's formula was made for tanh. **He et al. (2015) fixed it with an extra factor of 2.** This is exactly why the paper needed to pre-train net A first.
- **`init_from_A` needs an interpretation:** A's first 4 layers are 3→64, 64→128, 128→256, 256→256, but E's are 3→64, 64→64, 64→128, 128→128, so copying by position doesn't fit. We copy each A layer into the **next deep layer with the same shape** (E's layers 1, 3, 5, 6). The paper doesn't say exactly how.
- Scale jittering picks a new S for each image, and multi-crop gives 50 crops per scale.

---

## 10. Check yourself

1. What is the receptive field of three stacked 3×3 layers? Why prefer them to one 7×7?
2. Where are most of VGG-16's 138M parameters? Why does adding conv layers add so few?
3. What does a 1×1 convolution do? Why was C worse than D?
4. Why did the authors train net A first? What do the activation numbers in section 9 tell you?
5. What is scale jittering, and why does it help?
6. How do you turn an FC layer into a convolution? Why does that let you classify an image of any size?
7. Why are dense and multi-crop evaluation complementary?
