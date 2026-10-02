# An Image is Worth 16x16 Words (ViT), explained simply

**Paper:** Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit & Houlsby, *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.

**In one sentence:** cut an image into small square patches, treat each patch like a word, and feed the sequence to an ordinary Transformer encoder (the BERT one from paper 037). With enough training data this beats the best CNNs while costing less compute.

---

## 1. The big idea

### 1.1 The problem
- From 2012 (AlexNet, 014) to 2020, image recognition meant **convolutional networks**.
- Transformers (034) had taken over language, but images seemed different:
  - a 224×224 image has 50,176 pixels;
  - self-attention compares every pair of tokens, so 50,176 tokens means 50,176² ≈ 2.5 **billion** pairs per layer per head. Far too many.
- Earlier attempts to use attention on images kept convolutions or used special local attention patterns that ran slowly on real hardware.

### 1.2 The trick
- Don't use pixels as tokens. Use **patches**:
  - cut the 224×224 image into 16×16 squares;
  - that gives (224/16)² = 14² = **196 patches**;
  - flatten each patch into a vector (16·16·3 = 768 numbers) and map it linearly to the model width D.
- Now the image is a sentence of 196 "words", and attention compares 196² ≈ 38k pairs. Easy.
- Everything after that is **unchanged** from a standard Transformer encoder. That is the point of the paper: as little image-specific design as possible.

### 1.3 The surprise
- Trained on ImageNet alone (1.3M images), ViT is **worse** than a ResNet of similar size.
- Trained on 14M (ImageNet-21k) or 300M images (JFT-300M), it **wins**, and needs 2–4× less compute.
- The paper's slogan: **"large scale training trumps inductive bias."**

---

## 2. What is "inductive bias", and why does a CNN have more?

An **inductive bias** is an assumption built into a model before it sees any data.

A convolution assumes two things about images:
1. **Locality:** a pixel mostly relates to its neighbours (a 3×3 filter only looks at a 3×3 window).
2. **Translation equivariance:** the same filter slides everywhere, so a cat's ear detector works at every position. If the image shifts, the feature map shifts the same way.

ViT assumes almost nothing:
- only the patch-cutting step and the MLP (applied to each patch separately) are local and translation-equivariant;
- attention is **global** from layer 1: any patch can look at any other;
- position embeddings start **random**. The model is never told that patch 15 sits under patch 1. It must learn the 2-D layout from data.

**Why this matters:**
- With little data, assumptions are free knowledge, so the CNN wins.
- With huge data, the assumptions become a cage. ViT can learn better patterns (for example long-range ones in layer 1), so it wins.

Our demo shows the small-data side: on a tiny task of finding which cell contains a bright square, a small CNN reaches **100%** while a ViT trained the same way reaches **40.8%**.

---

## 3. The model, step by step (Eqs. 1–4)

### 3.1 Symbols

| Symbol | Meaning | ViT-B/16 value |
|---|---|---|
| H, W | image height, width (pixels) | 224, 224 |
| C | colour channels | 3 |
| P | patch side (pixels) | 16 |
| N | number of patches = HW/P² | 196 |
| x_p^i | the i-th patch, flattened | length P²C = 768 |
| D | model width | 768 |
| E | patch embedding matrix | (P²C) × D = 768 × 768 |
| x_class | the learnable [class] token | length D |
| E_pos | position embeddings, one per token | (N+1) × D |
| L | number of layers | 12 |
| z_ℓ | all N+1 token vectors after layer ℓ | (N+1) × D |

### 3.2 Cutting patches

Take a tiny 4×4 grey image (C = 1) with P = 2:

```
 1  2 | 3  4
 5  6 | 7  8
------+------
 9 10 |11 12
13 14 |15 16
```

- N = (4·4)/2² = 4 patches, in row-major order (left to right, top to bottom):
  - x_p^1 = [1, 2, 5, 6]
  - x_p^2 = [3, 4, 7, 8]
  - x_p^3 = [9, 10, 13, 14]
  - x_p^4 = [11, 12, 15, 16]
- With colour, each pixel contributes 3 numbers, so a 16×16 colour patch becomes 16·16·3 = 768 numbers.

The resolution decides the sequence length:

| image | P = 16 | P = 14 |
|---|---|---|
| 224 | 196 | 256 |
| 384 | 576 | 729 (H/14 uses 518 → 37² = 1369) |
| 512 | 1024 | |

Halving P quadruples N, and attention cost grows with N². That is why the "/14" and "/16" in names like ViT-H/14 matter: smaller patches mean more accuracy and more compute.

### 3.3 Eq. 1: the input sequence

```
z_0 = [ x_class ; x_p^1 E ; x_p^2 E ; … ; x_p^N E ] + E_pos
```

Read it piece by piece:
1. **x_p^i E:** each patch (a row vector of length 768) times a 768 × D matrix gives a D-vector. This is literally a linear layer applied to every patch. (It is the same as a convolution with a 16×16 kernel and stride 16.)
2. **x_class:** one extra learnable vector placed in front, exactly like BERT's [CLS] (037). It owns no patch; it gathers information through attention, and its final state is used to classify.
3. **+ E_pos:** add a learnable vector to each of the N+1 positions, otherwise attention can't tell where a patch came from (attention is permutation-invariant: shuffle the inputs and the outputs shuffle the same way).

**Worked example** (D = 2, the 2×2-patch image above, scaled down):
- Say E = [[0.1, 0], [0, 0.1], [0.1, 0], [0, 0.1]] (4 × 2).
- x_p^1 E = [1·0.1 + 5·0.1, 2·0.1 + 6·0.1] = [0.6, 0.8].
- x_p^4 E = [11·0.1 + 15·0.1, 12·0.1 + 16·0.1] = [2.6, 2.8].
- With x_class = [0, 0] and E_pos rows [0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.2, 0.2]:
  - x_p^2 E = [3·0.1 + 7·0.1, 4·0.1 + 8·0.1] = [1.0, 1.2].
  - z_0 = [[0, 0], [0.7, 0.8], [1.0, 1.3], …] (each patch row plus its position row).

### 3.4 Eqs. 2–3: the encoder layer (pre-LN)

```
z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}          (Eq. 2)
z_ℓ  = MLP(LN(z'_ℓ))   + z'_ℓ              (Eq. 3)
```

- **LN** = LayerNorm: for each token, subtract its mean and divide by its standard deviation, then scale and shift with learned γ, β.
  - Example: [1, 3] has mean 2 and std 1, so it becomes [−1, +1].
- **MSA** = multi-head self-attention (paper 034):
  - for each head: Q = hW_Q, K = hW_K, V = hW_V;
  - weights A = softmax(QKᵀ/√d_h) (each row sums to 1);
  - output = AV;
  - concatenate the heads and project.
- **MLP** = Linear(D → 4D) → GELU → Linear(4D → D).
  - GELU(x) = x·Φ(x), where Φ is the normal CDF. GELU(1) ≈ 0.84 and GELU(−1) ≈ −0.16: a smooth ReLU.
- **"+ z"** = residual connections.

**Pre-LN vs post-LN:**
- The original Transformer (034) and BERT (037) normalize **after** adding the residual: z = LN(z + Sublayer(z)).
- ViT normalizes **before**: z = z + Sublayer(LN(z)).
- With pre-LN, the residual path is a clean sum from input to output: z_L = z_0 + Σ (sublayer outputs). Gradients flow straight through, so deep models train stably without a careful warm-up.

### 3.5 Eq. 4: the output

```
y = LN(z_L^0)
```

- Take only token 0 (the [class] token) from the last layer, normalize it, and feed it to the classification head.
- **Pre-training head:** an MLP with one hidden layer (tanh).
- **Fine-tuning head:** a single D × K linear layer, **initialised to zero**.

**Why zero?**
- At the start of fine-tuning, the logits are all 0, so softmax gives a uniform 1/K for every class and the loss is ln K.
- The new head doesn't throw random noise into the pretrained body.
  - The gradient reaching the body is W_headᵀ (p − y). With W_head = 0, that is **exactly zero** on the first step.
  - So step 1 only trains the head (its gradient (p − y)·featureᵀ is sensible). The body starts changing once the head has learned something useful, never because of random head weights.
- Our test checks that a fresh fine-tuning model outputs exactly 0 logits.

### 3.6 Model sizes (Table 1)

| Model | Layers | D | MLP | Heads | Params (paper) | Params (our code) |
|---|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M | 86.6M (B/16) |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M | 304.3M (L/16) |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M | 632.0M (H/14) |

**Counting ViT-B/16 by hand** (D = 768):
- **patch embedding:** 768·768 + 768 = 590,592;
- **[class] + positions:** 768 + 197·768 = 152,064;
- **one layer:**
  - attention: 4·(768² + 768) = 2,362,368;
  - MLP: 768·3072 + 3072 + 3072·768 + 768 = 4,722,432;
  - two LNs: 2·2·768 = 3,072;
  - total ≈ 7.09M;
- **12 layers:** ≈ 85.05M;
- **final LN + 1000-class head:** 1,536 + 769,000;
- **total ≈ 86.6M.** ✓

Our L/16 count is 304.3M against the paper's 307M. The paper's figure likely includes the pre-training MLP head or a bigger class count (JFT has 18k classes; an 18k head alone is 1024 · 18k ≈ 18M parameters). This is a bookkeeping difference, not a model difference.

---

## 4. Position embeddings: 1-D is enough

- ViT uses **plain learnable 1-D** position embeddings: one vector per index 0…N.
- The paper also tried:
  - none at all;
  - 2-D (half the vector encodes the row, half the column);
  - relative positions.
- **Appendix D.4:** with no positions it is clearly worse (61.4% vs about 64% in Table 8, an ImageNet 5-shot linear probe on ViT-B/16). Among 1-D, 2-D and relative there is **no significant difference**.
- **Why?** The model works at patch level. A 14×14 grid has only 196 positions, so learning their layout from scratch is easy.
- **Figure 7 (centre)** shows what is learned: the similarity of each position's embedding to all others forms a little picture of the grid. Nearby patches are similar, and the same row or column is similar. The model rediscovered 2-D geometry on its own.

### Cosine similarity, step by step
cos(a, b) = (a·b) / (|a| |b|).
- Example: a = [1, 0], b = [1, 1] gives cos = 1 / (1 · √2) = 0.707.
- In Figure 7, a cosine near 1 means "these two positions are encoded alike".

---

## 5. Fine-tuning at higher resolution (Section 3.2)

- Fine-tuning at a **higher** resolution than pre-training helps (an idea from Touvron et al. 2019): for example, pre-train at 224, fine-tune at 384.
- Keep the patch size P, so the grid grows: 224/16 = 14 → 384/16 = 24. Now N = 576 instead of 196.
- The weights E, the attention and the MLP don't care how many tokens there are. **Only E_pos has the wrong size** (197 rows, but 577 are needed).
- **Fix:**
  1. reshape the 196 patch position vectors into a 14×14×D grid;
  2. **2-D interpolate** it to 24×24×D (bicubic);
  3. flatten it back.
- The paper notes this is one of only two places where the 2-D structure of images is put in by hand (the other is cutting patches).

**Linear interpolation, 1-D example:**
- Positions 0 and 1 have embeddings [0, 0] and [2, 4].
- A new grid twice as fine needs a value at 0.5: [1, 2], the average.
- Bicubic does the same with a smooth cubic curve through 4 neighbours instead of a line through 2.

Our `resize_positions` does exactly this, and a test checks that resizing to the same grid changes nothing, that the [class] position is kept, and that the model runs on the bigger image.

---

## 6. The hybrid architecture

- Instead of raw pixel patches, take the **feature map of a CNN** (a ResNet stage) and treat each 1×1 cell of it as a patch.
- The sequence is then the CNN's grid (for example 14×14 after a ResNet's stride-16 stages).
- **Figure 5:** hybrids beat pure ViT at **small** compute budgets. The gap vanishes for bigger models.
  - The CNN's inductive bias helps when the Transformer is small, and stops mattering once it is big enough to learn its own features.

Our `CNNStem` is a tiny stand-in (stride-2 convs), not a ResNet.

---

## 7. Training (Section 4.1)

**Pre-training:**
- Adam (β₁ = 0.9, β₂ = 0.999);
- batch 4096;
- **weight decay 0.1** (high; the paper found it useful for transfer);
- linear warm-up of 10k steps, then linear (or cosine) decay;
- resolution 224.
- Adam beat SGD for ResNets too in this setting.

**Fine-tuning:**
- SGD with momentum 0.9;
- batch 512;
- no weight decay;
- gradient clipping at norm 1;
- resolution 384, or 512 for L/16 and 518 for H/14 in Table 2.

**Datasets:**

| name | images | classes |
|---|---|---|
| ImageNet (ILSVRC-2012) | 1.3M | 1k |
| ImageNet-21k | 14M | 21k |
| JFT-300M | 303M | 18k |

The downstream test sets were de-duplicated against the pre-training data.

---

## 8. Results

### 8.1 Table 2 (pre-trained on JFT-300M, unless noted)

| Model | ImageNet | ReaL | CIFAR-10 | CIFAR-100 | Pets | Flowers | VTAB (19 tasks) | TPUv3-core-days |
|---|---|---|---|---|---|---|---|---|
| ViT-H/14 | **88.55** | **90.72** | **99.50** | **94.55** | **97.56** | 99.68 | **77.63** | 2.5k |
| ViT-L/16 | 87.76 | 90.54 | 99.42 | 93.90 | 97.32 | **99.74** | 76.28 | 0.68k |
| ViT-L/16 (ImageNet-21k) | 85.30 | 88.62 | 99.15 | 93.25 | 94.67 | 99.61 | 72.72 | 0.23k |
| BiT-L (ResNet152x4) | 87.54 | 90.54 | 99.37 | 93.51 | 96.62 | 99.63 | 76.29 | 9.9k |
| Noisy Student (EfficientNet-L2) | 88.4 | 90.55 | – | – | – | – | – | 12.3k |

**How to read it:**
- ViT-H/14 slightly beats the best CNNs **and** costs **4–5× less** compute (2.5k vs 9.9k/12.3k core-days).
- ViT-L/16 matches BiT-L at **1/14** of the compute.
- Even pre-trained on the public ImageNet-21k, ViT-L/16 gets 85.3% for 0.23k core-days ("about 30 days on 8 cores").

A "TPUv3-core-day" is one TPU core running for one day. 2.5k core-days is about 10 days on a 256-core pod.

### 8.2 Figure 3: pre-training dataset size
- **Pre-trained on ImageNet only:** ViT-L is **worse** than ViT-B, and both are worse than BiT ResNets.
- **On ImageNet-21k:** they're about equal.
- **On JFT-300M:** ViT-L/H are the best, and bigger ViTs keep helping.
- Even with dropout, weight decay and label smoothing tuned, small-data ViT can't catch up.

### 8.3 Figure 4: random subsets of JFT (9M, 30M, 90M, 300M)
- All models trained identically, with no extra regularization, so this tests the models themselves.
- **At 9M:** ResNets are better (ViT overfits more).
- **At 90M+:** ViT is better.
- This is the clearest evidence for the inductive-bias story.

### 8.4 Figure 5: scaling study (performance vs pre-training compute)
- At equal compute, ViT beats ResNets, using **about 2–4× less compute** for the same accuracy.
- Hybrids help at small sizes; the gap disappears for large models.
- ViT shows no sign of saturating in the tested range.

### 8.5 Self-supervision (Section 4.6)
- **Masked patch prediction:** corrupt 50% of the patch embeddings and predict each corrupted patch's **3-bit mean colour**.
  - 3 bits per channel means 8 levels per channel, so 8³ = **512 classes**.
- The corruption copies BERT's 80/10/10 rule:
  - 80% are replaced by a learnable [mask] embedding;
  - 10% are replaced by a random other patch;
  - 10% are kept.
- **ViT-B/16 on ImageNet:**

  | training | ImageNet accuracy |
  |---|---|
  | self-supervised pre-training | 79.9% |
  | from scratch | about 2% lower |
  | supervised pre-training | about 4% higher |

**Mean-colour target, worked out:**
- A patch whose average colour is (R, G, B) = (0.95, 0.10, 0.50).
- Quantise each channel to 8 levels with floor(value · 8): R = 7, G = 0, B = 4.
- Class = (R·8 + G)·8 + B = (56 + 0)·8 + 4 = **452**.
- A pure red patch, (1, 0, 0): R is clamped to 7, so the class is 7·64 = **448** (our test).
- Pure white is 7·64 + 7·8 + 7 = **511**, and black is **0**.

---

## 9. Inspecting what ViT learned (Section 4.5, Figure 7)

### 9.1 Patch-embedding filters
- **Principal component analysis (PCA)** of the rows of E (each row is how one embedding dimension responds to the 16×16×3 patch).
- The top components look like smooth **basis functions**: blobs, gradients and stripes at low frequencies. These resemble the first-layer filters of CNNs.

**PCA in one line:**
- Centre the data, compute the SVD W = U S Vᵀ, and the rows of Vᵀ are directions sorted by how much variance (S²) they explain.
- Our `embedding_filters_pca` does exactly that.

### 9.2 Position-embedding similarity
- This is the "grid picture" from Section 4.
- In larger grids there are visible sinusoid-like patterns.

### 9.3 Mean attention distance (the "receptive field" of attention)
For a head in a layer:

```
distance = average over query patches i of  Σ_j A_ij · ‖pos_i − pos_j‖  (pixels)
```

- **Worked example:** a query patch at (0, 0) gives 0.5 weight to itself (distance 0) and 0.5 to a patch 32 px away. Its distance is 0.5·0 + 0.5·32 = **16 px**.
- **The paper's finding:**
  - in **layer 1**, some heads already attend across nearly the whole image (distance about 100+ px on 224 images), while others stay very local;
  - as depth grows, every head becomes global.
  - CNNs can't do this in layer 1: a 3×3 conv sees 3 pixels.
  - Local heads are rarer in hybrids, which suggests they play the role of the early conv layers.
- **Figure 6:** attention rollout (multiplying the attention maps through layers) highlights the semantically relevant object.

---

## 10. Why it works (the intuition)

1. **Patches make attention affordable.** N² with N = 196 is cheap; with pixels it's impossible.
2. **A linear patch embedding is a strided convolution.** The very first step is a "stem" that can learn edge and colour detectors (Section 9.1).
3. **Global attention from layer 1** lets the model combine far-apart evidence early. A CNN needs many layers to grow its receptive field.
4. **Few assumptions + lots of data = better features.** The Transformer is a very general function family. Data teaches it locality where locality helps (local heads) and long range where that helps.
5. **Transformers scale well on accelerators.** Big dense matrix multiplies make it about 2–4× more compute-efficient than ResNets at the same accuracy.

---

## 11. What our code found

All numbers come from `demo.py` (about 6 s on a laptop) and `test_vit.py` (8 tests, about 1 s).

1. **Sequence lengths:** 224/16 → 196 patches; 384/16 → 576; 518/14 → 1369 (Section 2).
2. **Sizes:** B/16 = 86.6M, L/16 = 304.3M, H/14 = 632.0M, against the paper's 86 / 307 / 632. B and H match; L differs by about 1% (head bookkeeping, Section 3.6).
3. **Inductive bias:** on "which of 4 cells contains the bright square" (tiny images, a few hundred updates), the CNN reaches **100%** and the ViT **40.8%**.
   - This is a toy echo of Figures 3–4: with little data and training, convolution's built-in locality wins easily.
4. **Position similarity after short training:**

   | pair of positions | mean cosine |
   |---|---|
   | same row | +0.027 |
   | same column | +0.079 |
   | neither | +0.093 |

   - **No grid structure yet.** The paper's Figure 7 picture comes from long training on huge data; our 500 updates on tiny images are nowhere near that.
   - We report this honestly instead of claiming the effect.
5. **Attention distance:** the heads average about 8–9.6 px, against **8.0 px** for perfectly uniform attention on our grid.
   - So the heads are still **nearly uniform**: no local or global specialisation has formed after so little training.
   - The tool works (tested); the trained-at-scale pattern needs `experiments.py --only e5`.
6. **Masked-patch targets:** pure red, a dark patch, white and black become classes **[448, 32, 511, 0]**.
   - The tests also check that corruption picks about 50% of the patches, split 80/10/10.
7. **Zero-initialised head:** a fresh fine-tuning head gives exactly 0 logits, so the first loss is exactly ln K.
8. **Position interpolation:** resizing to the same grid is a no-op; resizing to a larger grid keeps the [class] slot, gives (new_grid² + 1) positions, and the model runs at the new resolution.

**Not run (too heavy for the laptop):** `experiments.py` reproduces the paper's claims at CIFAR-10 scale:
- **E1:** ViT vs ResNet on 5–100% of the data;
- **E2:** patch size;
- **E3:** hybrid;
- **E4:** higher-resolution fine-tuning;
- **E5:** Figure 7;
- **E6:** masked patch prediction.

**What to expect:** on CIFAR-10 from scratch, a ResNet should beat a small ViT at every fraction, because 50k images is far below the paper's crossover. That is the paper's prediction, not a contradiction of it.

---

## 12. Check yourself

1. A 384×384 image with P = 16. How many tokens enter the Transformer (count the [class] token)?
   <details><summary>Answer</summary>(384/16)² + 1 = 24² + 1 = 577.</details>
2. Why does fine-tuning at a higher resolution only need new **position** embeddings, not a new E or new attention weights?
   <details><summary>Answer</summary>E acts on one patch at a time (its size is fixed by P), and attention/MLP work for any number of tokens. Only E_pos has one row per position, so its row count must change.</details>
3. What is the 3-bit mean-colour class of a patch with mean (0.3, 0.6, 0.9)?
   <details><summary>Answer</summary>floor(0.3·8) = 2, floor(0.6·8) = 4, floor(0.9·8) = 7, so (2·8 + 4)·8 + 7 = 167.</details>
4. Why does ViT lose to ResNets when trained on ImageNet alone but win on JFT-300M?
   <details><summary>Answer</summary>CNNs build in locality and translation equivariance, which is valuable knowledge when data is scarce. ViT must learn these from data; with 300M images it can, and its flexibility (for example global attention early on) then lets it learn better features.</details>
5. Show that the patch embedding is a convolution.
   <details><summary>Answer</summary>Each output token is a weighted sum of the pixels in one P×P patch, with the same weights for every patch, and patches don't overlap. That is a conv with kernel P and stride P, with D output channels.</details>
6. Why is the fine-tuning head zero-initialised, and what is the loss at step 0 with K = 10 classes?
   <details><summary>Answer</summary>So the new random head doesn't push noise gradients into the pretrained body. Zero logits give a uniform softmax, so the loss is ln 10 ≈ 2.303.</details>
7. A query's attention: 0.25 to itself, 0.75 to a patch at a pixel distance of 48. What is its mean attention distance?
   <details><summary>Answer</summary>0.25·0 + 0.75·48 = 36 px.</details>
