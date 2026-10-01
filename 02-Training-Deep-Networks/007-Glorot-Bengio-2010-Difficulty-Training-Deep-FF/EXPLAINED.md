# Glorot & Bengio (2010), explained from scratch

**Paper:** *Understanding the difficulty of training deep feedforward neural networks*
**Authors:** Xavier Glorot, Yoshua Bengio (Université de Montréal)
**Published in:** AISTATS 2010, JMLR W&CP volume 9, pp. 249–256
**Page numbers** below are the proceedings pages (249–256).

Read Papers 004 (backprop) and 006 (Efficient BackProp) first. This guide derives every variance formula step by step, with numbers for a 1,000-unit layer.

---

## 0. The whole idea in one line

> **A deep network trained from random weights fails because the signal shrinks (or grows) a little at every layer, so after 5 layers almost nothing is left, forwards and backwards. Choose the starting weights so each layer passes the signal on at the same size in both directions, and deep nets train fine.**

That rule is **Xavier (Glorot) initialization**:
```
W ~ Uniform[ −√6/√(n_in + n_out),  +√6/√(n_in + n_out) ]
```
It is still the default for tanh-like networks in most libraries.

---

## 1. The puzzle (page 249)

- **What worked:** since 2006, deep networks trained well if first **pre-trained** one layer at a time without labels (Hinton's deep belief nets, stacked autoencoders).
- **What didn't:** plain backprop from **random** weights did badly.
- **Why?** Instead of guessing, the authors **watch** every layer's activations and gradients during training.
- **The background problem** (from Paper 004, section 11): every layer multiplies the backward signal by roughly (slope of the activation) × (weights). Many such factors multiplied together either **vanish** or **explode**.

---

## 2. Setup (pages 250–251)

### 2.1 Data
| Dataset | What | Size |
|---|---|---|
| **Shapeset-3×2** (new) | 32×32 images with 1 or 2 shapes (triangle, parallelogram, ellipse) of random size, angle, position and grey level; **9 classes** = which shapes are present | infinite (generated on the fly) |
| MNIST | handwritten digits | 50k / 10k / 10k |
| CIFAR-10 | small colour photos, 10 classes | 50k / 10k |
| Small-ImageNet | 37×37 grey images, 10 classes | 90k / 10k / 10k |

### 2.2 Network
- **Architecture:** 1–5 hidden layers of **1,000 units**, with a softmax output.
- **Cost:** −log P(correct class | x) (cross-entropy).
- **Training:** SGD with minibatches of 10, and the learning rate chosen on validation data.

### 2.3 The three activations
| Name | f(x) | f′(x) | Range | Note |
|---|---|---|---|---|
| sigmoid | 1/(1+e⁻ˣ) | f(1 − f) ≤ 0.25 | (0, 1) | **not centred at 0**; f′(0) = 0.25 |
| tanh | tanh x | 1 − f² ≤ 1 | (−1, 1) | centred; f′(0) = 1 |
| **softsign** | x/(1 + \|x\|) | 1/(1 + \|x\|)² | (−1, 1) | centred; approaches ±1 **slowly** |

**What "slowly" means:**
- **tanh:** its slope 1 − tanh²x ≈ 4e^(−2|x|) decays **exponentially**. At x = 5 it's ~0.0002, so tanh goes flat fast.
- **softsign:** its slope 1/(1+|x|)² decays only **polynomially**. At x = 5 it's 1/36 ≈ 0.028, more than **100× larger**.
- So a saturated softsign unit can still learn.

### 2.4 The "standard" initialization (Eq. 1)
```
W ~ U[ −1/√n, 1/√n ]       n = number of inputs to the layer;  biases = 0
```

---

## 3. What happens during training (pages 251–252)

### 3.1 Sigmoid: the top layer dies (Figure 2)
- **Observation:** soon after training starts, the **top** hidden layer's sigmoid outputs are driven to **0**, the sigmoid's flat floor. The lower layers stay near 0.5.
- **Why:**
  - At the start, the lower layers compute random, useless features.
  - The quickest way for the output layer to reduce the error is to rely on its own **biases** (predicting the class frequencies) and switch off the noisy top-layer input. It does that by pushing h → 0.
  - For **tanh**, 0 is the middle of the curve, where the slope is largest: harmless.
  - For **sigmoid**, 0 is the floor, where f′ ≈ 0. No gradient flows down, so the lower layers can't improve, and nothing gives the top layer a reason to come back.
- **The outcome:** with 4 layers it slowly escapes (around epoch 100); with 5 it **never** does.
- **Our demo** (sigmoid net on Shapeset): the top layer's mean activation goes **0.50 → 0.15 → 0.11 → 0.09 → 0.07** over 400 updates. For tanh it stays ≈ 0.00 (the healthy middle).

### 3.2 Tanh: saturation climbs the layers (Figure 3)
With the standard init, layer 1 saturates first (its values go to ±1), then layer 2, then 3, and so on. *"Why this is happening remains to be understood."*

### 3.3 Softsign: gentler (Figures 3–4)
- All layers saturate **a little, together**.
- At the end, most values sit at the **knees** (±0.6 to ±0.8): curved but not flat.
- Tanh ends with many values at the flat extremes (±1).

---

## 4. Gradients: the heart of the paper (pages 252–254)

### 4.1 Cross-entropy beats squared error (Figure 5)
**Why, in one derivation:**
- Softmax output p, true class y, and pre-softmax score s.
- **Cross-entropy** C = −log p_y:
  ```
  ∂C/∂s_k = p_k − 1[k = y]
  ```
  That is never small when the network is wrong. If p_y = 0.01, the gradient on s_y is −0.99.
- **Squared error with sigmoid outputs**, C = ½(o − t)² with o = σ(s):
  ```
  ∂C/∂s = (o − t) · o(1 − o)
  ```
  When the output is confidently wrong (o ≈ 0 but t = 1), the factor o(1 − o) ≈ 0. The worse the mistake, the smaller the push.
- That's why the quadratic cost shows **more plateaus**. **Use cross-entropy for classification.**

### 4.2 The variance argument (Eqs. 2–15)
**The toolkit: the variance of a product.**
- If W and x are **independent** with **mean 0**: Var[W x] = E[W²x²] − (E[Wx])² = E[W²]E[x²] − 0 = **Var[W] · Var[x]**.
- **The variance of a sum** of independent terms is the sum of their variances.

**The setup:**
- Layer i computes s^i = W^i z^i + b, then z^{i+1} = f(s^i).
- **Assume:**
  - the units are in the **linear regime** at init (f′(0) = 1 for tanh/softsign, so z ≈ s);
  - weights and inputs are independent, with mean 0;
  - n_i is the number of units in layer i.

**Forward (activations going up):**
- One unit's input is a sum of n_i terms W·z:
  ```
  Var[z^{i+1}] = n_i · Var[W^i] · Var[z^i]
  ```
- So across the whole network:
  ```
  Var[z^i] = Var[x] · Π_{k<i} (n_k · Var[W^k])                         (5)
  ```
- **Each layer multiplies the activation variance by n_in · Var[W].**

**Backward (gradients going down):**
- By backprop (Paper 004, Eq. 7), ∂C/∂s^i_j = f′ · Σ_l W^{i+1}_{lj} ∂C/∂s^{i+1}_l. That's a sum of n_{i+1} terms (the units this one feeds):
  ```
  Var[∂C/∂s^i] = n_{i+1} · Var[W^{i+1}] · Var[∂C/∂s^{i+1}]
  ```
  ```
  Var[∂C/∂s^i] = Var[∂C/∂s^d] · Π_{k≥i} (n_{k+1} · Var[W^k])          (6)
  ```
- **Each layer multiplies the gradient variance by n_out · Var[W].**

**Keeping both steady:**
```
forward:   n_in  · Var[W] = 1      (10)
backward:  n_out · Var[W] = 1      (11)
```
Both hold exactly only if n_in = n_out. The **compromise** is 1 divided by the *average* of the two fans:
```
Var[W] = 2 / (n_in + n_out)        (12)
```

### 4.3 Why the standard init fails (Eq. 15)
**The variance of a uniform distribution U[−a, a]:**
```
Var = (1/2a) ∫_{−a}^{a} w² dw = (1/2a) · (2a³/3) = a²/3
```
So U[−1/√n, 1/√n] has variance 1/(3n), and:
```
n · Var[W] = 1/3
```
- **Every layer keeps only 1/3 of the variance.** After 5 layers, (1/3)⁵ = **0.0041**: 0.4% is left. Activations fade going up, and gradients fade going down.
- In standard-deviation terms that's a factor √(1/3) ≈ 0.58 per layer. Our demo measured activation spreads 0.164 → 0.094 → 0.054 → 0.031 → 0.018, a ratio of ≈ 0.57 per layer ✔.

### 4.4 The fix: normalized ("Xavier") initialization (Eq. 16)
- We need a uniform distribution with a²/3 = 2/(n_in + n_out), so a² = 6/(n_in + n_out):
  ```
  W ~ U[ −√6/√(n_in + n_out),  +√6/√(n_in + n_out) ]
  ```
- **Numbers for a 1000 → 1000 layer:**
  - a = √6/√2000 = **0.0548**, and Var[W] = 0.001;
  - so n·Var[W] = **1.0**, compared with 1/3 for the standard init's a = 1/√1000 = 0.0316.
- **Our demo:** the activation spread stays ≈ 0.27 → 0.21 across 5 layers, and the backward gradients stay within a factor of 1.3 of each other (vs 10× for the standard init).

### 4.5 The surprise: weight gradients stay level anyway (Eq. 14, Figure 8)
- **The observation:** even with the standard init, the **weight** gradients are about the same size in every layer, although the back-propagated gradients shrink.
- **Why:**
  - **∂C/∂W^i = (gradient arriving at layer i) × (activation entering it).**
  - Going **down** the network, the gradient factor shrinks by (1/3) per layer (Eq. 6), while the activation factor **grows** by 3 per layer (Eq. 5 read the other way).
  - Their product stays constant: Var[∂C/∂W^i] ∝ [n·Var W]^{d−i} · [n·Var W]^{i−1}, which is the same for every i.
- **So the problem isn't simply "lower layers get smaller weight gradients".** The problem is that the signals themselves fade, which leaves the network in a poorly conditioned state.

### 4.6 The Jacobian's singular values (Eq. 17)
- **What J measures:** the Jacobian J_i = ∂z^{i+1}/∂z^i says how a small change in one layer becomes a change in the next. Its **singular values** are the stretch factors in different directions.
- **The ideal:** an average near **1** means small changes pass through at the same size, both forward and (via Jᵀ) backward.
- **The measurements:**
  - paper: ≈ **0.5** (standard init) vs ≈ **0.8** (normalized);
  - **our measurements: 0.49 vs 0.80** ✔.
- **Over 5 layers:** 0.5⁵ ≈ 0.03 vs 0.8⁵ ≈ 0.33. That's a 10× difference in how much signal survives.

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

- **Sigmoid is worst everywhere.** On Shapeset it barely learns (82.6%, where chance is 8/9 = 88.9%).
- **Normalized init helps tanh the most** (27.15 → 15.60 on Shapeset).
- **Softsign is less sensitive** to the init, thanks to its slow tails.

**Conclusions (page 256):**
1. **Monitor** activations and gradients layer by layer. It's a powerful debugging tool.
2. **Avoid sigmoids** with small random init in deep nets.
3. **Keep each layer's Jacobian near 1** so signals flow both ways. This closes much of the gap to pre-training.

**What came next:**
- He et al. (2015) redid this argument for **ReLU**. ReLU zeros half its inputs, which halves the variance, so it needs Var[W] = **2/n_in** ("He init").
- Batch normalization (Paper 012) enforces unit variance *during* training, instead of only at the start.

---

## 6. What our code found

**Scale note:** at your request, the full experiments were **not run** on this laptop. `experiments.py` reproduces Table 1 and Figures 2–8 on a machine that can handle it (≈ 30–40 minutes on a fast CPU).

**Measured at initialization** (5 tanh layers × 1,000 units, 300 Shapeset images):

| | standard init (Eq. 1) | normalized init (Eq. 16) |
|---|---|---|
| activation std, layers 1 → 5 | 0.164 → 0.094 → 0.054 → 0.031 → **0.018** (fades) | 0.268 → 0.248 → 0.232 → 0.222 → **0.213** (steady) |
| back-prop gradient, layer 1 vs 5 | **~10× smaller** at layer 1 | about equal |
| weight gradients across layers | about equal (the "surprise") | about equal |
| **average Jacobian singular value** | **0.49** (paper: 0.5) | **0.80** (paper: 0.8) |

**Sigmoid's top layer (Figure 2):** its mean activation falls 0.50 → 0.07 in 400 updates, toward the flat floor. Tanh's stays at 0.00 (the steep middle).

**Short training runs** (MNIST, 5 epochs instead of millions of updates, best learning rate):
- softsign, standard init: **3.54%** test error;
- softsign, normalized init: **2.97%**.

Both are above the paper's 1.6–1.7% because training was much shorter.

**Our Shapeset** is built from the paper's description (the original generator isn't used). After 30,000 updates, softsign nets were still at ≈ 74% error vs the paper's 16%, consistent with the paper needing millions of updates for this hard task.

---

## 7. Check yourself

1. Show that Var[Wx] = Var[W]·Var[x] for independent, zero-mean W and x.
2. Derive the forward rule Var[z^{i+1}] = n_in Var[W] Var[z^i]. What is the backward rule, and why does it use n_out?
3. Derive Var = a²/3 for U[−a, a]. With a = 1/√n, what is n·Var[W]? What survives 5 layers?
4. Derive the Xavier bound a = √6/√(n_in + n_out). Compute it for 784 → 1000.
5. Why does cross-entropy give a large gradient when the network is confidently wrong, while squared error with a sigmoid doesn't?
6. Why does the sigmoid's top layer get stuck at 0 while tanh's doesn't?
7. Explain why the weight gradients can stay level while the back-propagated gradients shrink.
8. What does an average Jacobian singular value of 0.5 vs 0.8 mean after 5 layers?
