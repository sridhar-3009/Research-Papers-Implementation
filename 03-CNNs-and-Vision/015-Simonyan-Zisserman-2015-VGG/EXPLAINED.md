# Simonyan & Zisserman (2015), explained from scratch

**Paper:** *Very Deep Convolutional Networks for Large-Scale Image Recognition* ("VGG")
**Authors:** Karen Simonyan, Andrew Zisserman (Visual Geometry Group, University of Oxford)
**Published at:** ICLR 2015 (arXiv:1409.1556)

Read Paper 014 (AlexNet) first. VGG keeps AlexNet's recipe and deliberately changes one thing: **depth**. This guide:
- derives the receptive-field and parameter formulas;
- counts VGG-16's 138M parameters by hand;
- explains, with variance arithmetic, why deep VGG nets were hard to initialize.

---

## 0. The whole idea in one line

> **Use only tiny 3×3 filters everywhere, and stack many of them. A stack of 3×3 layers sees as much as one big filter, with fewer weights and more non-linearities. Going from 11 to 16–19 layers clearly improves accuracy.**

---

## 1. The question (Section 1)

- **After AlexNet,** people improved ConvNets in many ways at once (smaller first-layer filters, multi-scale testing, …), so it was unclear what mattered.
- **This paper isolates one factor: depth.** Everything else is fixed, and the network grows deeper step by step.
- **Why that's affordable:** tiny filters. A 3×3 layer adds few parameters, so you can stack many.

---

## 2. The design (Section 2.1)

- **Input:** 224×224 RGB, minus the mean RGB. No other preprocessing.
- **Conv layers:** always **3×3**, the smallest size that captures left/right, up/down and centre. **Stride 1, padding 1**, so the spatial size is unchanged: (n + 2·1 − 3)/1 + 1 = n.
- **Pooling:** five 2×2/2 max-pools, each halving the size: 224 → 112 → 56 → 28 → 14 → **7**.
- **Width:** 64 channels, **doubling after each pool** up to 512 (64, 128, 256, 512, 512).
  - **Why double:** halving the resolution cuts the work per channel by 4×, and doubling the channels roughly balances the cost of each block.
- **The head:** FC-4096, FC-4096, FC-1000 + softmax, identical in every configuration.
- **ReLU everywhere;** no LRN (except in one test net).

---

## 3. The six networks (Table 1) and their sizes (Table 2)

| Net | Weight layers | What's new | Parameters |
|---|---|---|---|
| **A** | 11 (8 conv + 3 FC) | base | 133M |
| **A-LRN** | 11 | A + AlexNet's LRN after layer 1 | 133M |
| **B** | 13 | 2 convs in the first two blocks | 133M |
| **C** | 16 | B + one **1×1** conv in each of the last 3 blocks | 134M |
| **D** ("VGG-16") | 16 | B + one more **3×3** in each of the last 3 blocks | 138M |
| **E** ("VGG-19") | 19 | four 3×3 convs in each of the last 3 blocks | 144M |

### 3.1 Counting VGG-16's parameters by hand
**A k×k conv with C_in → C_out channels has k²·C_in·C_out weights + C_out biases.**

| Part | Count |
|---|---|
| 13 conv layers, e.g. the first: 9·3·64 + 64 = 1,792; a 512→512 layer: 9·512·512 + 512 = 2,359,808 | **14,714,688** in total |
| FC6: the 7·7·512 = 25,088 inputs → 4096 | 25,088·4,096 + 4,096 = **102,764,544** |
| FC7: 4096 → 4096 | 16,781,312 |
| FC8: 4096 → 1000 | 4,097,000 |
| **Total** | **138,357,544** ✔ (our model counts exactly this) |

- **FC6 alone holds 74%** of the network. **Adding 8 conv layers (A → E) adds only 11M.**

### 3.2 Where the computation goes
- **Multiply-adds** (output positions × C_out × 9 × C_in): VGG-16 needs **15.3 billion** for its convs and only 0.12 billion for its FC layers. That's **≈ 15.5 G** per image.
- **Compared with AlexNet** (≈ 0.72 G, Paper 014), that's **21× more compute**, for ~2.3× the parameters. Depth at full resolution is expensive.
- This is why later designs (GoogLeNet, ResNet's bottlenecks, Paper 016) work hard to cut computation.

---

## 4. Why 3×3? (Section 2.3)

### 4.1 Receptive fields of stacked layers
- **The formula:** for stride-1 layers with kernels k₁, k₂, …, one output sees an input window of size
  ```
  r = 1 + Σ_l (k_l − 1)
  ```
  (With strides s between layers, each (k_l − 1) is multiplied by the product of the earlier strides.)
- **For 3×3 stacks:**
  - two 3×3 layers: r = 1 + 2 + 2 = **5**;
  - three: **7**.
- **Our test** confirms it: the gradient of one output of a two-layer stack reaches exactly a 5×5 patch of the input (rows and columns 3–7).

### 4.2 Three 3×3 layers vs one 7×7 layer (same receptive field)
**1. More non-linearity:** three ReLUs instead of one, so a more expressive, more discriminative function.

**2. Fewer parameters.** With C channels in and out:
```
three 3×3:  3 · (3² C²) = 27 C²
one 7×7:    7² C²       = 49 C²          → 49/27 = 1.81, i.e. 81% more
```
- **Our demo** (C = 512):

  | stack | weights |
  |---|---|
  | three 3×3 | 7.08M |
  | one 7×7 | 12.85M |
  | two 3×3 | 4.72M |
  | one 5×5 | 6.55M |

- **The paper's interpretation:** this is a **regularization**. The 7×7 filter is forced to factor into 3×3 pieces with non-linearities between them.

### 4.3 1×1 convolutions (config C)
- **What it is:** a 1×1 conv is a per-pixel linear map across channels, out_c = Σ_c′ W[c, c′]·in_c′.
- **With a ReLU after it,** it adds non-linearity **without enlarging the receptive field**.
- **Origin and influence:** this idea comes from "Network in Network", and GoogLeNet and ResNet use it to cheaply change the channel count.
- **In VGG, C < D:** extra depth that also captures **spatial context** (3×3) beats depth that doesn't (1×1).

---

## 5. Training (Section 3.1)

- **AlexNet's recipe:**
  - SGD, **batch 256**, momentum 0.9, weight decay 5·10⁻⁴;
  - dropout 0.5 in FC6–7;
  - learning rate 10⁻², ÷10 on plateaus (3 times);
  - stop at **370K iterations (74 epochs)**, fewer than AlexNet's 90 (the authors credit the implicit regularization of depth and small filters, plus pre-initialization).
- **Augmentation:** random 224×224 crops, flips, and AlexNet's RGB colour shift.

### 5.1 Initialization was a real problem, and the variance arithmetic explains it
- **The setup:** take a 3×3 conv layer with C_in input channels and ReLU after it. Its fan-in is n = 9·C_in.
- **The recursion:** following Paper 007's argument, with ReLU zeroing half of its inputs (keeping half of the second moment), each layer multiplies the activation variance by about
  ```
  factor = n · Var(W) · ½
  ```
- **The paper says weights ~ "normal with zero mean and 10⁻² variance",** i.e. std **0.1**. For C_in = 64: factor = 576 · 0.01 · ½ = **2.9** per layer, so it **explodes**. Read as std 0.01 instead: factor = 576 · 0.0001 · ½ = **0.029** per layer, so it **vanishes**.
- **Our measurement** on E's 16 conv layers at full width:

  | Init | activation std after 16 layers |
  |---|---|
  | std 0.1 ("10⁻² variance", as written) | **explodes to 10⁷** |
  | std 0.01 | **vanishes to 10⁻⁹** |
  | Glorot (Paper 007) | still shrinks ~1000× (0.17 → 0.00012) |

- **Glorot shrinks** because it sets Var(W) ≈ 1/n (made for tanh), so the factor is ≈ ½ per layer. The variance halves each layer, and over 16 layers the std shrinks by about (1/√2)¹⁶ ≈ 1/256, or more with the real layer shapes.
- **He et al. (2015)** fixed this with Var(W) = **2/n**, which makes the factor exactly 1.
- **What VGG actually did:**
  1. Train the shallow **net A** from random init.
  2. Start the deeper nets with **A's first four conv layers and three FC layers**, and the layers in between random.
  3. After submission they noted that Glorot init alone also works.

### 5.2 Training scale S: "scale jittering"
- **What S is:** the shorter side of the rescaled image that the 224 crop is taken from.
- **Single scale:** S = 256 or 384. The 384 net starts from the 256 net, with lr 10⁻³.
- **Multi-scale:** each image gets its own random **S ∈ [256, 512]**, so objects appear at many sizes. This is augmentation by **scale**, teaching size invariance.

---

## 6. Testing (Section 3.2)

### 6.1 Dense evaluation: turn the FC layers into convolutions
- **The conversion:**
  - FC6 reads a 7×7×512 block, so it is exactly a **7×7 conv** with 4096 output channels;
  - FC7 and FC8 become **1×1 convs**.
- **On a 224 image,** the converted net gives exactly the original outputs (our test checks this).
- **On a bigger image of side Q,** the last conv map is Q/32 wide. The 7×7 "FC6 conv" slides over it and produces a **(Q/32 − 6) × (Q/32 − 6) map of class scores**. Q = 384 gives 12 − 6 = 6, a 6×6 map.
- **Average the map**, and average with the flipped image. That's one pass instead of many crops.
- **Our small demo** (trained on 64×64, so FC6 is a 2×2 conv on a 2×2 map): images of 64, 96×128 and 160 give score maps of 1×1, 2×3 and 4×4.

### 6.2 Multi-crop
- **The protocol:** a 5×5 grid × 2 flips = 50 crops per scale, 150 over 3 scales.
- **It is complementary to dense evaluation** because the borders are treated differently: crops are zero-padded, while dense evaluation sees the real neighbouring pixels.

### 6.3 Hardware
4 Titan Black GPUs with data parallelism (3.75× faster than 1 GPU), **2–3 weeks per network**.

---

## 7. Results on ILSVRC-2012 (Section 4)

### Table 3: single test scale (validation error, %)
| Net | train S | test Q | top-1 | top-5 |
|---|---|---|---|---|
| A | 256 | 256 | 29.6 | 10.4 |
| A-LRN | 256 | 256 | 29.7 | 10.5 |
| B | 256 | 256 | 28.7 | 9.9 |
| C | 256 / 384 / [256;512] | 256 / 384 / 384 | 28.1 / 28.1 / 27.3 | 9.4 / 9.3 / 8.8 |
| D | 256 / 384 / [256;512] | 256 / 384 / 384 | 27.0 / 26.8 / **25.6** | 8.8 / 8.7 / **8.1** |
| E | 256 / 384 / [256;512] | 256 / 384 / 384 | 27.3 / 26.9 / **25.5** | 9.0 / 8.7 / **8.0** |

1. **LRN doesn't help** (A-LRN ≈ A), so it's dropped.
2. **Deeper is better** from 11 to 16–19 layers, and it **saturates at 19** on this data.
3. **C < D:** 3×3 layers (spatial context) beat 1×1 layers.
4. **Small filters beat big ones:** B with each pair of 3×3s replaced by one 5×5 was **7% worse top-1**.
5. **Scale jittering in training** clearly beats a fixed S.

### Tables 4–7: more scales, crops, ensembles
- **D and E at 3 test scales:** 24.8% / 7.5% top-1/top-5.
- **Dense + multi-crop:** E reaches 24.4% / **7.1%**.
- **D + E:** **6.8% top-5 test**.
- **ILSVRC-2014:** **2nd in classification** (7.3% test with 7 nets) and **1st in localization**.

| Method (Table 7) | top-5 test |
|---|---|
| AlexNet (5 nets, 2012) | 16.4% |
| OverFeat (7 nets, 2013) | 13.6% |
| Zeiler & Fergus (6 nets, 2013) | 14.8% |
| GoogLeNet (1 net) | 7.9% |
| GoogLeNet (7 nets, winner) | **6.7%** |
| **VGG (1 net)** | **7.0%**: the best single network |
| **VGG (2 nets)** | **6.8%** |

---

## 8. Transfer (Appendix B): VGG features are general

- **The method:**
  - remove the last layer;
  - use the **4096-d penultimate output** as an image descriptor (dense, averaged over positions and flips, several scales);
  - L2-normalize it;
  - train a **linear SVM**, with no fine-tuning.
- **Results:**
  - **VOC-2007 89.3% / VOC-2012 89.0% mAP** (89.7 / 89.3 for D + E), more than **6% above** the previous best;
  - **Caltech-101 92.7%, Caltech-256 86.2%**.
- **Legacy:** VGG-16 became **the** default feature extractor for detection, segmentation, style transfer and "perceptual losses" for years.

---

## 9. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` tests each claim on CIFAR-10 with VGG nets shrunk in width (channels / 4), but with the **paper's depths and patterns**:
  - E1: depth A → E;
  - E2: 3×3 pairs vs one 5×5;
  - E3: initialization of the 19-layer net;
  - E4: scale jittering;
  - E5: dense vs multi-crop.

**Checked (tests and demo, about a second):**
- **Table 2, exactly:** A 132.9M, B 133.0M, C 133.6M, D **138,357,544**, E **143,667,240**, built on PyTorch's "meta" device (shapes only, no memory).
- **Table 1's depths:** 11, 11, 13, 16, 16, 19; 5 pools; widths 64 → 512.
- **Receptive field:** two 3×3 convs reach exactly a 5×5 input patch.
- **27C² vs 49C²:** ratio 1.81.
- **FC → conv:** exact on the training size; a score map on bigger images.
- **Initialization:** 10⁷ / 10⁻⁹ / ~1000× shrink for std 0.1 / std 0.01 / Glorot, which shows why the paper pre-trained net A.
- **`init_from_A` needs an interpretation:**
  - A's first 4 layers are 3→64, 64→128, 128→256, 256→256, but E's are 3→64, 64→64, 64→128, 128→128, so copying by position doesn't fit;
  - we copy each A layer into the **next deep layer with the same shape** (E's layers 1, 3, 5, 6);
  - the paper doesn't say exactly how.
- **Scale jittering** picks a new S per image; multi-crop gives 50 crops per scale.

---

## 10. Check yourself

1. Use r = 1 + Σ(k − 1) to find the receptive field of four stacked 3×3 layers. (9.)
2. Compare the weights of three 3×3 layers vs one 7×7 layer (C channels). Where does 81% come from?
3. Derive FC6's 102.8M parameters. What fraction of VGG-16 is that?
4. Why does VGG-16 need ~21× AlexNet's compute with only ~2.3× its parameters?
5. With fan-in n = 9·64 and ReLU, what per-layer variance factor do std 0.1, std 0.01 and He init give?
6. Why was C worse than D?
7. Turn FC6 into a convolution. What size score map does a 384×384 image give?
8. Why are dense and multi-crop evaluation complementary?
