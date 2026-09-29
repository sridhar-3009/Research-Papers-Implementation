# Ioffe & Szegedy (2015), explained simply

**Paper:** *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*
**Authors:** Sergey Ioffe, Christian Szegedy (Google)
**Published at:** ICML 2015 (arXiv:1502.03167)

Read Paper 006 (Efficient BackProp: normalize the inputs) and Paper 007 (Glorot: activations saturate) first. BN applies Paper 006's input normalization **inside** the network, at every layer, all the time.

---

## The big idea in one line

> **Before each nonlinearity, normalize every feature to mean 0 and variance 1 using the current mini-batch's statistics, then let the network rescale it with two learned numbers (γ, β). Do this inside the network, so gradients flow through the normalization. Training becomes much faster and far less sensitive to the learning rate and initialization.**

---

## 1. The problem: "internal covariate shift" (Sections 1–2)

- Paper 006 showed that training is faster when inputs have **mean 0 and variance 1** (and are decorrelated).
- But each layer's input is the output of the layers below it. **Every update to the lower layers changes the distribution the upper layers see.** The paper calls this **internal covariate shift**.
- Consequences:
  - small learning rates are needed;
  - initialization must be careful;
  - saturating nonlinearities (sigmoid) get stuck. Inputs drift into the flat part, where g′(x) ≈ 0 and gradients vanish (Paper 007).
- The goal: keep each layer's input distribution **fixed** while training.

> **Note (later research):** Santurkar et al. (2018, "How Does Batch Normalization Help Optimization?") showed that BN helps even when covariate shift is deliberately added back. The main effect seems to be a **smoother loss landscape**, which is what allows larger steps. The method works; the paper's *explanation* of why is debated.

---

## 2. Why the normalization must be inside the gradient (Section 2)

A tempting shortcut: every few steps, compute the mean activation and subtract it, **outside** backprop.

The paper's example: x = u + b, and normalize x̂ = x − E[x].
- The gradient step ignores that E[x] depends on b, so it updates b ← b + Δb.
- But then u + (b + Δb) − E[u + (b + Δb)] = u + b − E[u + b]: **the output doesn't change**.
- So the loss stays the same while **b grows forever**. With scaling added too, the paper saw models blow up.

**Fix:** make the normalization part of the model, so backprop sees it. Our demo shows this: b drifts to 41 in 200 steps with a flat loss, while the correct gradient is exactly 0.

---

## 3. Two simplifications (Section 3)

Full whitening, Cov[x]^(−1/2)·(x − E[x]), is expensive and hard to differentiate. So BN:

1. **Normalizes each feature separately:** mean 0, variance 1, no decorrelation. (Paper 006 showed this already helps.)
2. **Uses mini-batch statistics** instead of the whole training set. Now the statistics are a function of the batch, and gradients flow through them.

Normalizing can reduce what a layer can represent (e.g. it confines sigmoid inputs to the near-linear middle). So BN adds **γ and β** per feature, which can **undo** the normalization: with γ = √Var[x] and β = E[x], y = x. The network decides how much normalization it wants.

---

## 4. Algorithm 1: the BN transform

For one feature over a mini-batch B = {x_1 … x_m}:
```
µ_B   = (1/m) Σ x_i                  mini-batch mean
σ²_B  = (1/m) Σ (x_i − µ_B)²         mini-batch variance
x̂_i   = (x_i − µ_B) / √(σ²_B + ε)    normalize (ε for numerical stability)
y_i   = γ·x̂_i + β                    scale and shift (learned)
```

### The backward pass (page 4)
```
∂ℓ/∂x̂_i  = ∂ℓ/∂y_i · γ
∂ℓ/∂σ²_B = Σ_i ∂ℓ/∂x̂_i · (x_i − µ_B) · (−½)(σ²_B + ε)^(−3/2)
∂ℓ/∂µ_B  = Σ_i ∂ℓ/∂x̂_i · (−1/√(σ²_B + ε))  +  ∂ℓ/∂σ²_B · Σ_i −2(x_i − µ_B)/m
∂ℓ/∂x_i  = ∂ℓ/∂x̂_i · 1/√(σ²_B + ε)  +  ∂ℓ/∂σ²_B · 2(x_i − µ_B)/m  +  ∂ℓ/∂µ_B · 1/m
∂ℓ/∂γ    = Σ_i ∂ℓ/∂y_i · x̂_i
∂ℓ/∂β    = Σ_i ∂ℓ/∂y_i
```
Simplified (not in the paper, but it's the same thing; see `bn_backward_compact`):
```
∂ℓ/∂x = (γ/σ) · ( dy − mean(dy) − x̂ · mean(dy · x̂) )
```
**Meaning:** the gradient reaching x has its **mean removed** and its **component along x̂ removed**. The layer below can't usefully shift or rescale the whole batch, because BN would undo it, so the gradient doesn't even try. This is the formal version of Section 2's lesson.

---

## 5. Algorithm 2: training and inference (Section 3.1)

- **Training:** use mini-batch statistics. Each example's output depends on the **other examples in its batch**.
- **Inference:** we want a deterministic output that depends only on the input. Use the **population** statistics:
  ```
  E[x]   = E_B[µ_B]                  average of the batch means
  Var[x] = m/(m−1) · E_B[σ²_B]        unbiased (σ²_B divides by m, not m − 1)
  ```
  In practice, people use **moving averages** during training (the paper mentions this too). Our torch layer does that.
- At test time, BN is just a **fixed linear map**:
  ```
  y = (γ/√(Var[x]+ε))·x + (β − γ·E[x]/√(Var[x]+ε))
  ```
  It can be folded into the previous layer's weights, so it costs nothing at inference.

---

## 6. Where to put it, and the convolutional version (Section 3.2)

- Put BN **right before the nonlinearity**: z = g(BN(W·u)).
  - Not on the layer input u: u is the output of a nonlinearity, so its shape keeps changing, and fixing two moments doesn't fix it.
  - W·u is a sum of many terms, so it's "more Gaussian", and normalizing it gives a stable distribution.
- **Drop the bias b:** BN subtracts the mean, which cancels b. β takes its place.
- **Convolutions:** one γ and β **per feature map**, with statistics over the batch **and all spatial locations** (effective batch size m·p·q). That keeps the "same operation everywhere" property of convolution.

---

## 7. Why it allows higher learning rates (Section 3.3)

- Normalizing stops small parameter changes from amplifying through many layers.
- **Scale invariance:** BN((aW)u) = BN(Wu). So:
  - ∂BN((aW)u)/∂u = ∂BN(Wu)/∂u: the Jacobian doesn't depend on the weight scale.
  - ∂BN((aW)u)/∂(aW) = (1/a)·∂BN(Wu)/∂W: **bigger weights get smaller gradients**. If a large learning rate makes the weights grow, their updates automatically shrink. This is self-stabilizing.
- **Conjecture:** BN pushes layer Jacobians toward singular values ≈ 1 (JJᵀ ≈ I, the ideal of Paper 007 and Saxe et al.). Gradients then neither explode nor vanish.

---

## 8. BN as a regularizer (Section 3.4)

In training, an example's output depends on which other examples are in its batch. That's **noise**, like dropout. The paper found dropout could be **removed or reduced** in BN networks.

---

## 9. Experiments (Section 4)

### 4.1 MNIST (Figure 1)
- The network: 784 → 3×100 sigmoid → 10, small Gaussian init, 50,000 steps, 60 examples per batch.
- **(a)** The BN network reaches higher test accuracy, faster.
- **(b, c)** They plot the 15th/50th/85th percentiles of one last-layer sigmoid input over training.
  - **Without BN:** the mean and spread drift a lot.
  - **With BN:** they're stable.

### 4.2 ImageNet (Inception, Figures 2–3)
"Accelerating" a BN network meant:
- raising the learning rate (×5 or ×30);
- removing dropout and local response normalization;
- reducing L2 weight decay 5×;
- decaying the learning rate 6× faster;
- shuffling more thoroughly;
- using fewer photometric distortions.

| Model | Steps to reach 72.2% | Best accuracy |
|---|---|---|
| Inception (lr 0.0015) | 31.0M | 72.2% |
| BN-Baseline (just add BN) | 13.3M | 72.7% |
| **BN-x5** | **2.1M (14× fewer)** | 73.0% |
| BN-x30 | 2.7M | **74.8%** |
| BN-x5-Sigmoid | n/a | 69.8% |

- The same ×5 learning rate made the original Inception's parameters reach "machine infinity".
- Inception with sigmoid never beat chance (0.1%); **with BN it reaches 69.8%.**

### 4.2.3 Ensemble (Figure 4)
Six BN-Inception networks together reach **4.9% top-5 validation error** (4.82% on the test set), better than the best published result and than estimated human accuracy.

---

## 10. What our code found

**Scale note:** at your request, the MNIST experiments were **not run** on this laptop. `experiments.py` reproduces Figure 1 (E1), a small version of the learning-rate findings of Figures 2–3 (E2), and the "sigmoid only trains with BN" finding on a 10-layer net (E3). That's about 30–60 minutes on a CPU. The paper gives no learning rate or init scale for Figure 1; we chose lr 0.5 and std 0.1.

**Checked (tests and demo, about a second):**
- Output has mean 0 and variance 1 per feature, and **γ = √Var, β = E recovers the identity**.
- **Our hand-written backward** (the paper's equations, one per line) matches finite differences, `torch.nn.functional.batch_norm`, and the compact form.
- The gradient reaching x **sums to 0** and is **orthogonal to x̂**.
- **Scale invariance:** multiplying W by 7 leaves the output and ∂ℓ/∂u unchanged, and divides ∂ℓ/∂W by 7.
- **Section 2:** without the E[x] term in the gradient, b drifts linearly (to 41 after 200 steps) with a constant loss. With it, the gradient on b is exactly 0.
- **Algorithm 2's m/(m−1):** with batches of 4, the plain average of batch variances is 3.0 for a true variance of 4. The corrected estimate gives 4.0.
- Inference BN is a single linear map a·x + c.
- Conv BN normalizes each feature map over (N, H, W), and its gradient matches PyTorch.
- In training mode an example's output **changes when its batch-mates change** (the regularizing noise). In eval mode a single example gives the same output as in a batch.
- A small detail in the demo: a feature with variance 5·10⁻⁵ comes out with std 0.91, not 1, because ε = 10⁻⁵ is added to it.

---

## 11. Check yourself

1. Write Algorithm 1. Why are γ and β needed?
2. Explain the b-drift example in Section 2. What does it teach about where normalization must live?
3. Why does BN use different statistics at training and test time? Why m/(m − 1)?
4. Why is the bias b useless before BN?
5. For conv layers, which values are averaged together, and why?
6. Show that BN((aW)u) = BN(Wu). What does that do to the weight gradient, and why does it allow higher learning rates?
7. Why does BN act like a regularizer? What goes wrong with a batch size of 1?
