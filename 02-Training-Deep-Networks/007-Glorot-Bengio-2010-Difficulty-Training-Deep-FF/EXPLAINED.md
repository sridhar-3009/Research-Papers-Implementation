# Glorot & Bengio (2010), explained simply

**Paper:** *Understanding the difficulty of training deep feedforward neural networks*
**Authors:** Xavier Glorot, Yoshua Bengio (Université de Montréal)
**Published in:** AISTATS 2010, JMLR W&CP volume 9, pp. 249–256
**Page numbers** below are the proceedings pages (249–256).

---

## The big idea in one line

> **Deep networks trained badly from random weights because the signal shrinks (or grows) a little at every layer, and after 5 layers almost nothing is left. Pick the starting weights so that each layer passes the signal on at the same size, in both directions, and deep networks train fine.**

That starting rule is **Xavier (Glorot) initialization**:
```
W ~ Uniform[ −√6/√(n_in + n_out),  +√6/√(n_in + n_out) ]
```
It's still the default in many libraries.

---

## How it connects to the earlier papers

- **Paper 004** (backprop): our 5-layer family-tree network **stalled** from small initial weights: the vanishing gradient.
- **Paper 006** (Efficient BackProp): recommended weights with std = 1/√fan-in, and a zero-centred sigmoid.
- **This paper** measures exactly **what goes wrong in each layer** of a deep net, and fixes it by also taking fan-**out** into account.

---

## 1. The question (page 249)

- Since 2006, deep networks worked if you first **pre-trained** them layer by layer without labels.
- Plain backprop from **random** weights did badly. **Why?**
- The approach: **watch** the activations and gradients of every layer during training, and see what goes wrong.

---

## 2. The experimental setup (pages 250–251)

### Data
| Data set | What | Size |
|---|---|---|
| **Shapeset-3×2** (new) | 32×32 images with 1 or 2 shapes (triangle, parallelogram, ellipse), random size, rotation, position and grey level. **9 classes** = which shapes are present | infinite: new images generated on the fly |
| MNIST | handwritten digits, 28×28 | 50k train, 10k validation, 10k test |
| CIFAR-10 | small colour photos, 10 classes | 50k/10k |
| Small-ImageNet | 37×37 grey images, 10 classes | 90k/10k/10k |

### Network
- **1 to 5 hidden layers, 1,000 units each**, softmax output.
- **Cost:** negative log-likelihood, −log P(y|x) (cross-entropy).
- **SGD on minibatches of 10.** The learning rate is picked on the validation set.
- **Three activations:**

| Name | Formula | Range | Note |
|---|---|---|---|
| sigmoid | 1/(1+e⁻ˣ) | (0, 1) | **not centred at 0** |
| tanh | tanh(x) | (−1, 1) | centred |
| **softsign** | x / (1 + \|x\|) | (−1, 1) | centred; approaches ±1 **slowly** (polynomially, not exponentially) |

### The "standard" initialization, Eq. (1)
```
W ~ U[ −1/√n,  1/√n ]         n = size of the previous layer
```
The biases start at 0.

---

## 3. What happens during training (pages 251–252)

### 3.1 Sigmoid: the top layer dies (Figure 2)
- Right after training starts, the **top hidden layer's** sigmoid outputs are pushed to **0**, their saturated floor. The other layers stay above 0.5.
- **Why?** At first, the lower layers compute random, useless features. The output layer learns fastest by relying on its **biases** and turning off the useless top-layer signal, so it pushes h → 0.
  - For **tanh**, 0 is the **middle** (the steepest point), so that's harmless.
  - For **sigmoid**, 0 is the **flat floor**: the slope is ~0, so no gradient flows back, and the lower layers can't learn.
- With 4 layers the net **slowly escapes** (around epoch 100). With 5 layers it **never** escapes.
- **Lesson:** sigmoid is a bad choice for deep nets initialized with small random weights.

### 3.2 Tanh: saturation climbs up the layers (Figure 3)
With the standard init, **layer 1 saturates first** (its values go to ±1), then layer 2, then layer 3, and so on. The authors say: *"Why this is happening remains to be understood."*

### 3.3 Softsign: gentler (Figures 3–4)
- All layers saturate **a little, and together**.
- At the end, most softsign activations sit at the **"knees"** (±0.6 to ±0.8): non-linear, but not flat.
- Tanh ends up with many values at the flat extremes (±1).

---

## 4. Gradients (pages 252–254): the heart of the paper

### 4.1 Cross-entropy beats squared error (Figure 5)
For the same network, the **quadratic** cost has **more plateaus** (flat regions) than the **cross-entropy** cost. Use cross-entropy for classification.

### 4.2 The variance argument (Eqs. 2–15)
Assume we're in the **linear** part of the activation (f′(0) = 1), and the weights are independent with variance Var[W]. Then:
- **Going up** (activations):
  ```
  Var[z^i] = Var[x] · Π (n · Var[W])        (5)
  ```
  Each layer **multiplies the variance by n·Var[W]**.
- **Going down** (back-propagated gradients):
  ```
  Var[∂Cost/∂s^i] = Var[∂Cost/∂s^d] · Π (n_{next} · Var[W])     (6)
  ```
  Each layer multiplies it by **n_out·Var[W]**.
- **To keep signals the same size** in both directions:
  ```
  n_in · Var[W] = 1        (forward, Eq. 10)
  n_out · Var[W] = 1       (backward, Eq. 11)
  ```
- Both can only hold if n_in = n_out, so use the **compromise**:
  ```
  Var[W] = 2 / (n_in + n_out)        (12)
  ```

### 4.3 Why the standard init fails (Eq. 15)
- U[−1/√n, 1/√n] has variance 1/(3n), so **n·Var[W] = 1/3**.
- Every layer shrinks the variance to about **one third**. After 5 layers, **(1/3)⁵ ≈ 0.4%** is left.
- Activations fade going up, and gradients fade going down.

### 4.4 The fix: normalized initialization (Eq. 16)
```
W ~ U[ −√6/√(n_in + n_out),  +√6/√(n_in + n_out) ]
```
A uniform distribution on [−a, a] has variance a²/3. With a = √6/√(n_in+n_out), that gives exactly Var[W] = 2/(n_in+n_out), which is Eq. 12.

### 4.5 A surprise (Eq. 14, Figure 8)
- Even with the standard init, the **weight** gradients are about the **same size** in every layer, although the **back-propagated** gradients shrink.
- **Why?** Each weight gradient = (activation below) × (gradient above). Going down, the activation grows while the gradient shrinks, and the two effects cancel.

### 4.6 The Jacobian's singular values (Eq. 17)
- J_i = ∂z^{i+1}/∂z^i describes how one layer maps to the next. Its average singular value is the average factor by which **a small change gets stretched or squeezed** per layer.
- The paper measured **≈ 0.8 with normalized init** and **≈ 0.5 with standard init**.
- Close to 1 is good, because signals pass through without fading.

---

## 5. Results (pages 254–256, Table 1)

Test error (%) with **5 hidden layers**. "N" means normalized init:

| | Shapeset | MNIST | CIFAR-10 | Small-ImageNet |
|---|---|---|---|---|
| Softsign | 16.27 | 1.64 | 55.78 | 69.14 |
| Softsign N | 16.06 | 1.72 | 53.8 | 68.13 |
| Tanh | 27.15 | 1.76 | 55.9 | 70.58 |
| **Tanh N** | **15.60** | **1.64** | **52.92** | **68.57** |
| Sigmoid | 82.61 | 2.21 | 57.28 | 70.66 |

- **Sigmoid is worst everywhere.** On Shapeset it barely learns (82.6%; chance is 88.9%).
- **Normalized init helps tanh the most:** 27.15% → 15.60% on Shapeset.
- **Softsign is less sensitive** to the initialization.

**Conclusions (page 256):**
1. **Monitor** activations and gradients layer by layer. It's a powerful debugging tool.
2. **Avoid sigmoid** with small random initial weights in deep nets.
3. **Keep each layer's Jacobian near 1**, so activations and gradients both flow well. That closes much of the gap to unsupervised pre-training.

---

## 6. What our code found

**Scale note:** at your request, the full experiments were **not run** on this laptop. `experiments.py` reproduces everything when run on a machine that can handle it (about 30–40 minutes on a fast CPU). The paper's Table 1 above is the reference to compare against.

**What was measured before the long run was stopped** (all at initialization, 5 tanh layers of 1,000 units, 300 Shapeset images, which is quick to compute):

| | standard init (Eq. 1) | normalized init (Eq. 16) |
|---|---|---|
| activation std, layer 1 → 5 | 0.16 → 0.09 → 0.05 → 0.03 → **0.018** (fades) | 0.26 → 0.24 → 0.23 → 0.22 → **0.21** (steady) |
| back-prop gradient, layer 1 vs layer 5 | **10× smaller** at layer 1 | about the same |
| weight gradients across layers | about the same (the "surprise") | about the same |
| **average Jacobian singular value** | **0.49** (paper: 0.5) | **0.80** (paper: 0.8) |

**Partial training results before stopping** (MNIST, 5 epochs instead of the paper's 5 million updates, best learning rate):
- Softsign, standard init: **3.54%** test error.
- Softsign, normalized init: **2.97%**.

Both are higher than the paper's 1.6–1.7%, because of the much shorter training.

**Our Shapeset:** built from the paper's description (the original generator code isn't used). After 30,000 updates, softsign nets were still at about 74% error, far from the paper's 16%. That's consistent with the paper needing millions of updates for this hard task.

---

## 7. Check yourself

1. What happens to the top hidden layer of a sigmoid network at the start of training, and why is that bad?
2. With the standard init, what is n·Var[W]? What fraction of the variance survives 5 layers?
3. Derive Var[W] = 2/(n_in + n_out) from the forward and backward conditions.
4. Why is a = √6/√(n_in+n_out) the right bound for a *uniform* distribution?
5. Why can weight gradients stay level while back-propagated gradients shrink?
6. What does an average Jacobian singular value of 0.5 vs 0.8 mean for a 5-layer network?
