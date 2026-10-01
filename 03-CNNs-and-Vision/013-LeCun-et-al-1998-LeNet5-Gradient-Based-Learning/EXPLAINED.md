# LeCun et al. (1998), explained simply

**Paper:** *Gradient-Based Learning Applied to Document Recognition*
**Authors:** Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
**Published in:** Proceedings of the IEEE, 86(11), November 1998, pp. 2278–2324
**What to read:** Sections I–III (pages 1–14) and Appendices A and C. Sections IV–IX (graph transformer networks, check reading) are about whole document systems; skim them.

Read Paper 006 (Efficient BackProp) first: this paper uses its tricks (scaled tanh, fan-in init, stochastic diagonal Levenberg-Marquardt).

---

## The big idea in one line

> **Don't hand-design features for images. Build the knowledge "nearby pixels matter together, and a feature is the same feature wherever it appears" into the network itself, using local receptive fields, shared weights and subsampling. Then learn everything end to end with backprop. That's the convolutional network.**

---

## 1. Why not a plain fully connected network? (Section II, page 5)

The traditional pipeline was hand-crafted feature extractor → trainable classifier. Learning directly from pixels with a fully connected network has three problems:

1. **Too many weights.** A 32×32 image with 100 hidden units already needs 100,000 weights. More weights need more data.
2. **No built-in invariance.** A digit shifted by 2 pixels looks completely different to a fully connected net. It must learn the same detector separately at every position.
3. **The topology is ignored**: shuffle the pixels in a fixed order and a fully connected net trains just as well. But images have strong **local 2-D structure**.

---

## 2. The three ideas of a convolutional network (Section II.A)

### 2.1 Local receptive fields
Each unit looks only at a small window (5×5) of the layer below. Early units detect **elementary features** (oriented edges, end-points, corners), and later layers combine them into higher-level features. (Hubel & Wiesel found such local, orientation-selective cells in the cat's visual cortex.)

### 2.2 Shared weights (feature maps)
All units in one **feature map** use the **same weights** at different positions. So:
- a useful edge detector in one corner is automatically applied everywhere;
- the layer's operation is exactly a **convolution** (+ bias + squashing);
- **Shift property:** shift the input and the feature map shifts by the same amount. Otherwise it's unchanged.
- **Far fewer parameters.** C1 has 122,304 connections but only 156 parameters.

A convolutional layer has **several feature maps**, so several features are extracted at every location.

### 2.3 Subsampling (pooling)
Once a feature is detected, its **exact position doesn't matter**; only its rough position relative to other features does. "Horizontal end-point at top left, corner at top right, vertical end-point at bottom" is a 7, wherever exactly those features sit.
- A **subsampling layer** averages 2×2 blocks, which halves the resolution and reduces sensitivity to shifts and distortions.
- In LeNet-5 each map has one trainable coefficient and one bias:
  - with a small coefficient, the layer just blurs;
  - with a large one, it acts like a "noisy OR" or "noisy AND".

**The "bi-pyramid":** the layers alternate convolution and subsampling. At each step the **spatial resolution goes down** and the **number of feature maps goes up**.

---

## 3. LeNet-5, layer by layer (Section II.B, Figure 2)

| Layer | What | Output | Parameters | Connections |
|---|---|---|---|---|
| input | 32×32 image (digit ≤ 20×20 in the center) | 1@32×32 | | |
| **C1** | conv 5×5 | 6@28×28 | 156 | 122,304 |
| **S2** | 2×2 subsample | 6@14×14 | 12 | 5,880 |
| **C3** | conv 5×5, **partial connections** (Table I) | 16@10×10 | 1,516 | 151,600 |
| **S4** | 2×2 subsample | 16@5×5 | 32 | 2,000 |
| **C5** | conv 5×5 (acts like a full connection here) | 120@1×1 | 48,120 | 48,120 |
| **F6** | full | 84 | 10,164 | 10,164 |
| output | 10 Euclidean RBF units | 10 | fixed | 840 |
| | | | **60,000** | **340,908** |

Details that matter:
- **The input is 32×32, bigger than the digit.** That way, features at the digit's edge (stroke end-points) can still sit in the center of the top-level detectors' receptive fields.
- **Pixel values:** background = −0.1, ink = 1.175, so the mean is ≈ 0 and the variance ≈ 1 (Paper 006's advice).
- **C3 doesn't connect to all six S2 maps (Table I).**
  - The first 6 C3 maps read 3 neighboring S2 maps.
  - The next 6 read 4 neighboring maps.
  - The next 3 read 4 non-neighboring maps.
  - The last one reads all 6.

  Why? It keeps the connections manageable, and it **breaks symmetry**: maps with different inputs are forced to learn different features.
- **C5 is called a convolution**, not a full layer, because on a bigger input it would produce a bigger map. It's the same weights slid over a larger image, which Section VII uses to read whole words.
- **Squashing function** (Eq. 6): f(a) = 1.7159·tanh(2a/3), so f(±1) = ±1 (Appendix A).
- **Weight initialization** (Appendix A): uniform in [−2.4/F, 2.4/F], where F is the unit's fan-in. That keeps the weighted sums in the squashing function's useful range.

### The output layer: Euclidean RBF units (Eq. 7)
```
y_i = Σ_j (x_j − w_ij)²      the squared distance from F6's state x to prototype w_i
```
- **The smallest y_i wins.** y_i is a **penalty**: roughly the negative log-likelihood of a Gaussian centered at w_i.
- **The prototypes are fixed, not learned.** They are ±1 values forming a **7×12 bitmap of each character** (hence F6 has 84 = 7×12 units; Figure 3 shows the full ASCII set).
- Why bitmaps instead of "1 of 10" codes?
  - Confusable characters (O, o, 0; l, 1, I) get **similar codes**, which is useful for a later language model.
  - "1 of N" codes behave badly with many classes, because sigmoid outputs must be "off" almost all the time.
  - RBF units are active only in a **bounded region**, so they're better at rejecting non-characters.
- **±1 targets** are where the sigmoid has **maximum curvature**. That keeps F6 away from saturation.

---

## 4. The loss function (Section II.C)

**MSE / maximum likelihood (Eq. 8):**
```
E(W) = (1/P) Σ_p y_{D_p}(Z^p, W)        just push down the correct class's penalty
```
It has two problems:
1. **Collapse:** if the RBF centers could learn, everything could become equal. F6 outputs a constant, every penalty is 0, and the loss is perfect, **while the network ignores the input**. (That's why the centers are fixed.)
2. **No competition** between classes.

**MAP / discriminative criterion (Eq. 9):**
```
E(W) = (1/P) Σ_p [ y_{D_p} + log( e^{−j} + Σ_i e^{−y_i} ) ]
```
- The second term **pulls up the wrong classes' penalties**.
- e^{−j} is a "rubbish class" that stops already-large penalties being pushed up forever.
- It prevents collapse even when the centers learn.

**Backprop with shared weights:** compute each connection's gradient as if the weights weren't shared, then **add up** the gradients of all connections that share a parameter. (Autograd does exactly this.)

---

## 5. Training (Section III.B and Appendix C)

- **20 passes** over the 60,000 training images, updating **after every example**.
- **Stochastic diagonal Levenberg-Marquardt** (from Paper 006): each parameter k has its own learning rate
  ```
  ε_k = η / (µ + h_kk)
  ```
  where h_kk is an estimate of the diagonal of the Hessian (Gauss-Newton approximation), µ = 0.02.
  - h_kk is re-estimated on **500 examples** before each pass, then kept fixed for that pass.
  - The effective rates came out between ~7×10⁻⁵ and 0.016.
- **Global schedule η:** 0.0005 (passes 1–2), 0.0002 (next 3), 0.0001 (next 3), 0.00005 (next 4), 0.00001 after that.

---

## 6. Results (Section III)

### MNIST (III.A)
The paper **created MNIST**:
- It mixes NIST's SD-3 (Census employees, clean) and SD-1 (students, messier), so training and test sets are drawn from the same distribution.
- Each digit is size-normalized to 20×20 and centered by its center of mass in 28×28.
- 60,000 training images, 10,000 test images.

### LeNet-5 (Figures 5–6)
- **Test error 0.95%**, stable after ~10 passes. Training error reaches 0.35% after 19 passes.
- **No overfitting** was seen. The authors' explanation: a relatively large learning rate keeps the weights moving, which favors **broad minima**, and broad minima generalize better.
- **More data helps** (Figure 6: 15k → 30k → 60k).
- **With distortions:** 540,000 extra artificially distorted images (random translation, scaling, squeezing, horizontal shear) cut the test error to **0.8%**.

### Comparison (Figure 9, Section III.C), test error on MNIST
| Method | Error |
|---|---|
| Linear classifier | 12.0% |
| Pairwise linear | 7.6% |
| K-NN, Euclidean | 5.0% |
| 1000 RBF + linear | 3.6% |
| 40 PCA + quadratic | 3.3% |
| MLP 784-300-10 | 4.7% |
| MLP 784-300-100-10 | 3.05% |
| Tangent distance (16×16) | 1.1% |
| SVM poly 4 | 1.1% |
| LeNet-1 (only ~2,600 parameters) | 1.7% |
| LeNet-4 | 1.1% |
| **LeNet-5** | **0.95%** |
| LeNet-5 + distortions | **0.8%** |
| Virtual SVM poly 9 + distortions | 0.8% |
| Boosted LeNet-4 + distortions | **0.7%** |

The paper also shows LeNet-5 is **robust**: it still recognizes digits under strong noise and distortion, and needs **far less memory and computation** than SVMs or K-NN (Section III.C).

**A curiosity (C.5):** the big MLPs don't overfit as badly as expected. The authors suggest the weights **shrink first** (the origin is attractive), so the sigmoids act nearly linearly and the network starts as a low-capacity model. As the weights grow, capacity increases. It's an accidental version of Vapnik's "structural risk minimization".

---

## 7. The rest of the paper (Sections IV–IX), in brief

- **Multi-module systems and Graph Transformer Networks (GTNs):** train a whole pipeline (segmentation + recognition + language model) end to end, with gradients flowing through graphs of hypotheses.
- **Space-displacement neural networks:** run the convolutional network over a **whole word** and read all characters at once, without segmenting first (possible because C5 is convolutional).
- **Deployed:** the check-reading system built on this was in commercial use, reading **several million checks per day** (the abstract's claim).

---

## 8. What our code found

**Scale note:** at your request, nothing was trained on this laptop. `experiments.py` reproduces Figures 5, 6 and 9, the distortion result, a shift-robustness test and the MSE-vs-MAP collapse (hours at batch size 1 on a CPU).

**Checked (tests and demo, about a second):**
- **Our LeNet-5 has exactly the paper's numbers:**
  - 156 / 12 / 1,516 / 32 / 48,120 / 10,164 parameters per layer, **60,000 in total**;
  - the connection counts 122,304 / 5,880 / 151,600 / 2,000.

  The C3 table we used gives exactly 1,516, which confirms it matches Table I (the scan of that table was unreadable).
- **Convolution** written as plain loops matches PyTorch's `conv2d`.
- **The shift property:** shifting the input by (1, 2) moves the feature map by (1, 2).
- **C3's missing connections really are absent:** changing a disconnected weight by +100 doesn't change the output, and its gradient is 0.
- **Subsampling:** the result is squash(coefficient × (sum of the 2×2 block) + bias).
- **The squashing function:** f(±1) = ±1, the asymptote is 1.7159, and |f″| is largest at |a| ≈ 1. That's why ±1 targets keep F6 in its "most non-linear" range.
- **Loss functions:** a collapsed network gets MSE = 0, i.e. "perfect", while MAP gives 2.34. MSE can't tell a good separation from a bad one, but MAP can.
- **Gauss-Newton diagonal:** for a linear layer our estimate equals the exact 2u_j².
- **SDLM** lowers the loss. Its per-parameter learning rates range from about 2×10⁻⁴ to 0.025 (paper: 7×10⁻⁵ to 0.016).
- One small finding: 5 × 10⁻⁴, the paper's first-pass rate, overshoots when **one** batch of random noise is repeated. The paper's rate is meant for real digits, visited once each per pass.

---

## 9. Check yourself

1. What three ideas make a network "convolutional"? What does each buy you?
2. Why does C1 have 122,304 connections but only 156 parameters?
3. Why is the input 32×32 when the digits are at most 20×20?
4. Why doesn't every C3 map connect to every S2 map?
5. What does an RBF output unit compute, and why is the smallest one the prediction?
6. Why can MSE collapse if the RBF centers are learned? How does the MAP loss prevent it?
7. Why would a fully connected network need to learn a "7 detector" at every position separately?
8. Why did 540,000 distorted images help, even though they're just transformations of the same 60,000?
