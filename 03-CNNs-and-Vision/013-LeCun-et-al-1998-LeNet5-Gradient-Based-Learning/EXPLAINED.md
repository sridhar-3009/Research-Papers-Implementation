# LeCun et al. (1998), explained from scratch

**Paper:** *Gradient-Based Learning Applied to Document Recognition*
**Authors:** Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
**Published in:** Proceedings of the IEEE, 86(11), November 1998, pp. 2278–2324
**What to read:** Sections I–III (pages 1–14) and Appendices A and C. Sections IV–IX (graph transformer networks, check reading) are about whole document systems; skim them.

Read Paper 006 first: this paper uses its tricks (scaled tanh, fan-in init, stochastic diagonal Levenberg–Marquardt). This guide computes a convolution by hand, derives every parameter count in LeNet-5, and works the loss functions with numbers.

---

## 0. The whole idea in one line

> **Don't hand-design image features. Build two facts about images into the network itself: "nearby pixels matter together" and "a feature is the same feature wherever it appears". Do it with local receptive fields, shared weights and subsampling, then learn everything end to end with backprop. That's the convolutional network.**

---

## 1. Why not a plain fully connected network? (Section II, page 5)

The traditional pipeline was a hand-crafted feature extractor followed by a trainable classifier. Learning directly from pixels with a fully connected (FC) net has three problems:
1. **Too many weights.**
   - A 32×32 image into just 100 hidden units needs 102,400 weights.
   - Mapping 32×32 to C1's 6×28×28 units fully would need **4,821,600** weights (our demo), where LeNet uses **156**.
   - More weights need more data.
2. **No built-in shift invariance:** a digit moved by 2 pixels activates completely different input weights. An FC net must learn its "7 detector" separately at every position.
3. **The topology is ignored:** permute the pixels in any fixed order and an FC net trains just as well. But images have strong local 2-D structure that this throws away.

---

## 2. The three ideas (Section II.A)

### 2.1 Local receptive fields
- Each unit looks only at a small window (5×5) of the layer below.
- **Early units** can only detect local **elementary features**: oriented edges, end-points, corners.
- **Later layers** combine those into bigger features.
- This mirrors Hubel & Wiesel's orientation-selective cells in the cat's visual cortex.

### 2.2 Shared weights: the operation is a convolution
- **All units of one feature map use the same weights,** each at a different position.
- **The resulting operation** (ignoring the bias and squashing) is a 2-D convolution: strictly a cross-correlation, but the name stuck:
  ```
  out[i, j] = Σ_{a=0..k−1} Σ_{b=0..k−1}  w[a, b] · in[i + a, j + b]  + bias
  ```
- **Output size** for an n×n input and a k×k kernel (no padding, stride 1): **(n − k + 1) × (n − k + 1)**. For example, 32 − 5 + 1 = **28** for C1.

**Worked example** (a 4×4 input, a 2×2 kernel with w = [[1, 0], [0, −1]], which detects "brighter top-left than bottom-right"):
```
input            kernel        output (3×3)
1 2 0 1          1  0           0 −1 −1
0 1 3 1          0 −1          −1  1  3
2 1 0 0                         2  0 −2
1 0 1 2
```
The top-left output is 1·1 + 2·0 + 0·0 + 1·(−1) = **0**. Then slide the kernel one step and repeat.

**The shift property (equivariance):**
- **The claim:** if the input moves by (dx, dy), the feature map moves by (dx, dy) and is otherwise unchanged.
- **The proof:** the same w is applied at every position, so moving the input just changes which position computes each value. Our test checks this for a shift of (1, 2).

### 2.3 Subsampling (pooling)
- **Exact position doesn't matter:** once a feature is detected, only its rough position relative to other features does. "End-point at top-left, corner at top-right, end-point at bottom" is a 7, wherever exactly those features are.
- **A subsampling layer** sums 2×2 blocks, then applies one trainable coefficient and one bias per map, then squashes:
  ```
  out[i, j] = f( c · Σ_{2×2 block} in  +  b )
  ```
  - It **halves the resolution** and makes the output less sensitive to small shifts and distortions.
  - With a small c, the map is in the linear range of f and just blurs. With a large c, it acts like a "noisy OR/AND".
- **The "bi-pyramid":** convolution and subsampling alternate. At each step, the resolution goes **down** and the number of feature maps goes **up**. That trades "where" for "what".

---

## 3. LeNet-5, layer by layer, with every count derived (Section II.B, Figure 2)

| Layer | What | Output | Parameters | Connections |
|---|---|---|---|---|
| input | 32×32 image (digit ≤ 20×20 in the centre) | 1@32×32 | | |
| **C1** | conv 5×5 | 6@28×28 | 156 | 122,304 |
| **S2** | 2×2 subsample | 6@14×14 | 12 | 5,880 |
| **C3** | conv 5×5, **partial connections** (Table I) | 16@10×10 | 1,516 | 151,600 |
| **S4** | 2×2 subsample | 16@5×5 | 32 | 2,000 |
| **C5** | conv 5×5 (fully covers the 5×5 maps) | 120@1×1 | 48,120 | 48,120 |
| **F6** | fully connected | 84 | 10,164 | 10,164 |
| output | 10 Euclidean RBF units | 10 | fixed | 840 |
| | | | **60,000** | **340,908** |

### 3.1 The arithmetic
**C1:**
- each map has 5×5 = 25 weights + 1 bias = 26 parameters;
- 6 maps give **156** parameters;
- every one of the 28×28 = 784 output positions uses all 156, so the connections are 784·156 = **122,304**;
- that's **784 connections per parameter**, the payoff of sharing.

**S2:**
- 1 coefficient + 1 bias per map, × 6 = **12**;
- connections: 14·14 positions × 6 maps × (4 inputs + 1 bias) = **5,880**.

**C3**, with Table I's partial connections (which S2 maps each C3 map reads):
```
6 maps read 3 S2 maps:  6 × (3·25 + 1) =   456
6 maps read 4 S2 maps:  6 × (4·25 + 1) =   606
3 maps read 4 S2 maps:  3 × (4·25 + 1) =   303
1 map  reads all 6:     1 × (6·25 + 1) =   151
                                  total = 1,516 ✔
```
Connections: 10·10 = 100 positions × 1,516 = **151,600**.

**S4:** 16 × 2 = **32**; connections 5·5·16·5 = **2,000**.

**C5:** 120 maps × (16·25 + 1) = 120 × 401 = **48,120**. The output is 1×1, so the connections equal the parameters.

**F6:** 84 × (120 + 1) = **10,164**.

**Total:** 156 + 12 + 1,516 + 32 + 48,120 + 10,164 = **60,000** ✔ (our model counts exactly this).

### 3.2 Receptive fields: how much of the image each layer "sees"
| Unit | Sees (pixels of the input) | Why |
|---|---|---|
| C1 | 5×5 | its kernel |
| S2 | 6×6 | 2 adjacent C1 windows, offset 1 |
| C3 | 14×14 | 5 S2 units, each step = 2 pixels: 6 + 4·2 |
| S4 | 16×16 | 2 adjacent C3 units, each step = 2 pixels |
| C5 | **32×32** | 5 S4 units, each step = 4 pixels: 16 + 4·4. **The whole image** |

**Why the input is 32×32 when digits are ≤ 20×20:** it lets the **centres** of the top detectors' receptive fields cover the digit's edges, so stroke end-points near the border are seen as well as central features.

### 3.3 Other details
- **The pixel scale:** background = −0.1 and ink = 1.175, so the mean ≈ 0 and the variance ≈ 1 (Paper 006).
- **Why C3 uses partial connections (Table I):**
  - it keeps the connection count manageable;
  - it **breaks symmetry**: maps that read different inputs are forced to learn different features.
- **C5 is "convolutional",** not FC: on a larger input it would output a larger map. That is exactly how Section VII slides the network over whole words.
- **Squashing** (Eq. 6): f(a) = 1.7159·tanh(2a/3), with f(±1) = ±1 (Paper 006, section 3.4).
- **Init** (Appendix A): uniform in [−2.4/F, 2.4/F], with F = fan-in. Its variance is (2.4/F)²/3 = 1.92/F², which keeps the weighted sums inside f's useful range.

---

## 4. The output layer: Euclidean RBF units (Eq. 7)

```
y_i = Σ_{j=1..84} (x_j − w_ij)²          the squared distance from F6's state x to prototype w_i
```
**The smallest y_i wins.**

### 4.1 Why "distance" is a log-likelihood
- **The connection:** a Gaussian centred at w_i has density ∝ exp(−‖x − w_i‖²/2σ²), so its negative log-likelihood is ‖x − w_i‖²/2σ² + const.
- **So y_i is (up to scale) "how unlikely x is under class i's Gaussian".** A penalty: lower is better.

### 4.2 The fixed prototypes
- **They are not learned:** they are ±1 patterns forming a **7×12 bitmap of each character**. That's why F6 has 84 = 7·12 units (Figure 3; our demo prints all ten).
- **Why bitmaps rather than "1-of-10" codes:**
  - confusable characters (O/o/0, l/1/I) get **similar** codes, which helps a later language model;
  - 1-of-N codes behave badly with many classes (sigmoid outputs must be "off" almost all the time);
  - RBF units respond only in a bounded region, so they reject non-characters better.
- **The ±1 targets** sit where f has maximum curvature, keeping F6 away from saturation.

---

## 5. The loss function (Section II.C), with numbers

### 5.1 MSE / maximum likelihood (Eq. 8)
```
E(W) = (1/P) Σ_p y_{D_p}(Z^p, W)            push the correct class's penalty down
```
**Two problems:**
1. **Collapse:** if the prototypes could learn, they could all become equal, and F6 could output that constant for every input. Every penalty is then 0 and E = 0, a "perfect" loss for a network that **ignores its input**. (That's why the prototypes are fixed.)
2. **No competition:** nothing pushes the wrong classes' penalties up.

### 5.2 MAP / discriminative loss (Eq. 9)
```
E(W) = (1/P) Σ_p [ y_{D_p} + log( e^{−j} + Σ_i e^{−y_i} ) ]
```
- **What the terms do:**
  - **The second term** is a "soft minimum" over all penalties. Lowering it means pushing the **wrong** classes' penalties **up**.
  - **e^{−j}** is a constant "rubbish class". It stops already-large penalties from being pushed up forever, since their e^{−y_i} is already negligible next to e^{−j}.
- **Equivalently:** E = −log [ e^{−y_D} / (e^{−j} + Σ_i e^{−y_i}) ], the negative log of a softmax over "−penalties" with one extra rubbish class. That is today's cross-entropy loss.

**Worked numbers (j = 1, 10 classes; our demo):**
- **Collapsed network** (all y_i = 0):
  - MSE = **0** ("perfect"!);
  - MAP = 0 + log(e^{−1} + 10·e⁰) = log(10.368) = **2.339**.
- **Good network** (correct penalty 1, the others very large):
  - MSE = 1;
  - MAP = 1 + log(e^{−1} + e^{−1} + ≈0) = 1 + log(0.736) = **0.693**.
- **So MSE prefers the useless collapsed net, and MAP correctly prefers the good one.**

### 5.3 Backprop with shared weights
- **The rule:** compute each **connection's** gradient as if it had its own weight, then **add up** the gradients of all connections that share a parameter.
- **Why:** a parameter used in 784 places affects the loss through all 784 paths, and the chain rule sums them. This is the same rule as backprop-through-time (Paper 004). Autograd does it automatically.

---

## 6. Training (Section III.B and Appendix C)

- **20 passes** over 60,000 images, updating **after every example** (batch size 1).
- **Stochastic diagonal Levenberg–Marquardt** (Paper 006, section 8.1): each parameter k gets its own rate
  ```
  ε_k = η / (µ + h_kk)            h_kk ≈ the diagonal Hessian (Gauss–Newton), µ = 0.02
  ```
  - h_kk is re-estimated on **500 examples** before each pass.
  - **Flat directions** (small h_kk) get bigger steps; **steep** ones get smaller.
  - The paper's effective rates were ~7×10⁻⁵ to 0.016.
- **The global schedule η:** 0.0005 (passes 1–2), 0.0002 (3–5), 0.0001 (6–8), 0.00005 (9–12), 0.00001 after that.

---

## 7. Results (Section III)

### 7.1 MNIST: this paper created it (III.A)
- **The source:** a mix of NIST SD-3 (Census employees, clean) and SD-1 (students, messier), so training and test sets come from the **same distribution**.
- **Preprocessing:** each digit is size-normalized to 20×20 and centred by its centre of mass in 28×28.
- **Size:** 60,000 training and 10,000 test images.

### 7.2 LeNet-5 (Figures 5–6)
- **Test error 0.95%,** stable after ~10 passes; training error 0.35% after 19.
- **No overfitting seen.** The authors' explanation: a relatively large learning rate keeps the weights moving and favours **broad minima**, which generalize better.
- **More data helps** (15k → 30k → 60k, Figure 6).
- **Distortions:**
  - 540,000 extra, artificially distorted images (random translation, scaling, squeezing, shear) bring the error to **0.8%**;
  - they teach invariances that the architecture doesn't build in exactly.

### 7.3 Comparison (Figure 9; test error on MNIST)
| Method | Error |
|---|---|
| linear classifier | 12.0% |
| pairwise linear | 7.6% |
| K-NN, Euclidean | 5.0% |
| 1000 RBF + linear | 3.6% |
| 40 PCA + quadratic | 3.3% |
| MLP 784-300-10 | 4.7% |
| MLP 784-300-100-10 | 3.05% |
| tangent distance (16×16) | 1.1% |
| SVM poly 4 | 1.1% |
| LeNet-1 (~2,600 parameters) | 1.7% |
| LeNet-4 | 1.1% |
| **LeNet-5** | **0.95%** |
| LeNet-5 + distortions | **0.8%** |
| virtual SVM poly 9 + distortions | 0.8% |
| boosted LeNet-4 + distortions | **0.7%** |

- **LeNet-5 is also robust** (strong noise and distortion) and needs **far less memory and computation** than SVMs or K-NN.
- **A curiosity (C.5):** the big MLPs overfit less than expected. The weights first **shrink** (the origin is attractive), so the sigmoids act nearly linearly and the net starts as a low-capacity model. Capacity grows as the weights grow: an accidental "structural risk minimization".

---

## 8. The rest of the paper (Sections IV–IX), briefly

- **Graph Transformer Networks:** train a whole pipeline (segmentation + recognition + language model) end to end, with gradients flowing through graphs of hypotheses.
- **Space-displacement networks:** slide the convolutional net over a **whole word** and read every character without segmenting first. This is possible because C5 is convolutional.
- **Deployment:** the check-reading system read **several million checks per day**.

---

## 9. What our code found

**Scale note:** at your request, nothing was trained on this laptop. `experiments.py` reproduces Figures 5, 6 and 9, the distortion result, a shift-robustness test and the MSE-vs-MAP collapse (hours at batch size 1 on CPU).

**Checked (tests and demo, about a second):**
- **The exact paper counts:**
  - 156 / 12 / 1,516 / 32 / 48,120 / 10,164 parameters, **60,000 in total**;
  - connections 122,304 / 5,880 / 151,600 / 2,000.

  Our C3 table gives exactly 1,516, which confirms it matches Table I (the scan of that table is unreadable).
- **Convolution as plain loops = PyTorch's `conv2d`.**
- **The shift property:** shifting the input by (1, 2) moves the feature map by (1, 2).
- **C3's missing connections really are absent:** changing a disconnected weight by +100 doesn't change the output, and its gradient is 0.
- **Subsampling** = f(c·(2×2 sum) + b).
- **Squashing:** f(±1) = ±1, the asymptote is 1.7159, and |f″| peaks at |a| ≈ 1.
- **Loss functions:** a collapsed network gets MSE 0 ("perfect") vs MAP 2.339; a good network gets MSE 1.000 vs MAP 0.693.
- **The Gauss–Newton diagonal:** for a linear layer our estimate equals the exact 2u_j².
- **SDLM** lowers the loss; its per-parameter rates span ~2×10⁻⁴ to 0.025 (paper: 7×10⁻⁵ to 0.016).
- **One small finding:** 5×10⁻⁴, the paper's first-pass rate, overshoots when **one** batch of random noise is repeated. The paper's rate assumes real digits visited once per pass.

---

## 10. Check yourself

1. Convolve the 4×4 input above with the kernel [[0, 1], [1, 0]]. What is the top-left output? (2 + 0 = 2.)
2. What output size does a 5×5 kernel give on a 14×14 map? (10×10: C3.)
3. Derive C1's 156 parameters and 122,304 connections. Derive C3's 1,516 from Table I.
4. Why does C5 see the whole 32×32 image? Trace the receptive field through the layers.
5. Why is an RBF unit's output a negative log-likelihood? Why is the smallest one the prediction?
6. Compute MSE and MAP (j = 1) for a collapsed network. Why does MAP fix the collapse?
7. Why are gradients of shared weights **added**?
8. Why did 540,000 distorted copies help, even though they come from the same 60,000 digits?
