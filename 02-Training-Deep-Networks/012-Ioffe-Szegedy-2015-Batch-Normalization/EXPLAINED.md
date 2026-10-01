# Ioffe & Szegedy (2015), explained from scratch

**Paper:** *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*
**Authors:** Sergey Ioffe, Christian Szegedy (Google)
**Published at:** ICML 2015 (arXiv:1502.03167)

Read Paper 006 (normalize the inputs) and Paper 007 (activations saturate) first. BN applies Paper 006's input normalization **inside** the network, at every layer, all the time. This guide works one tiny batch through BN forward and backward by hand, and derives every claim.

---

## 0. The whole idea in one line

> **Before each nonlinearity, normalize every feature to mean 0 and variance 1 using the current mini-batch's statistics, then let the network rescale it with two learned numbers (γ, β). Because the normalization is part of the network, gradients flow through it. Training becomes much faster and far less sensitive to the learning rate and initialization.**

---

## 1. The problem: "internal covariate shift" (Sections 1–2)

- **Paper 006:** training is fastest when inputs have mean 0, variance 1 (and are decorrelated), because that keeps the error surface's curvature well balanced.
- **But each layer's input is the output of the layers below,** and every update to those layers changes the distribution the next layer sees. The paper calls this **internal covariate shift**.
- **The consequences:**
  - small learning rates are needed;
  - initialization must be careful;
  - saturating nonlinearities get stuck: inputs drift into the flat part of a sigmoid, where g′ ≈ 0 (Paper 007).
- **The goal:** keep each layer's input distribution fixed during training.

> **Later research note:** Santurkar et al. (2018) showed BN still helps when covariate shift is deliberately re-injected. Its main effect seems to be a **smoother loss landscape** (smaller, more predictable gradient changes), which allows larger steps. The method works; the paper's *explanation* is debated.

---

## 2. Why normalization must be inside the gradient (Section 2)

**A tempting shortcut:** every few steps, measure the mean activation and subtract it, **outside** backprop.

**The paper's counterexample:**
- The layer is x = u + b, normalized as x̂ = x − E[x].
- The gradient step ignores that E[x] depends on b, so it updates b ← b + Δb.
- But:
  ```
  (u + b + Δb) − E[u + b + Δb] = u + b − E[u + b]
  ```
  So **the output doesn't change at all**.
- The loss stays the same while b keeps getting the same push, so **b grows forever**. With scaling included, the paper saw models blow up.

**The fix:** make normalization part of the model, so backprop "knows" that shifting b does nothing. The correct gradient on b is then exactly 0.

**Our demo:**
- ignoring E[x], b reaches 10.25 → 20.50 → **41.00** after 50/100/200 steps, with the loss stuck at 3.083;
- including it, b stays at **0.00**.

---

## 3. Two simplifications (Section 3)

Full **whitening**, Cov[x]^(−1/2)(x − E[x]), is expensive (a matrix inverse square root) and awkward to backprop through. So BN:
1. **Normalizes each feature separately** to mean 0 and variance 1, with no decorrelation. Paper 006 showed this already helps a lot.
2. **Uses mini-batch statistics** instead of the whole dataset. The statistics are then a differentiable function of the batch.

### Why add γ and β?
- **Normalizing can reduce what a layer can represent.** For example, it pins sigmoid inputs to the near-linear middle.
- **So BN adds a learnable scale γ and shift β per feature.** They can even undo the normalization: with γ = √Var[x] and β = E[x], y = x.
- **The network chooses how much normalization it wants.** The crucial difference from before: the mean and scale are now set by **two direct parameters**, not by a fragile combination of all the weights below.

---

## 4. Algorithm 1: the BN transform

For one feature over a mini-batch B = {x_1 … x_m}:
```
µ_B   = (1/m) Σ x_i                   mini-batch mean
σ²_B  = (1/m) Σ (x_i − µ_B)²          mini-batch variance
x̂_i   = (x_i − µ_B) / √(σ²_B + ε)     normalize (ε avoids dividing by 0)
y_i   = γ x̂_i + β                     scale and shift (learned)
```

### 4.1 Worked example (a batch of 4, one feature)
**Forward:**
```
x = (1, 2, 3, 6)
µ = 12/4 = 3
σ² = [(−2)² + (−1)² + 0² + 3²] / 4 = 14/4 = 3.5      σ = 1.871
x̂ = (−2, −1, 0, 3)/1.871 = (−1.069, −0.535, 0, 1.604)    → mean 0, variance 1 ✔
with γ = 2, β = 1:  y = (−1.138, −0.069, 1, 4.207)
```

**ε matters for tiny variances:**
- In our demo, a feature with variance 5·10⁻⁵ comes out with std √(5·10⁻⁵/(5·10⁻⁵ + 10⁻⁵)) = **0.91**, not 1.
- ε = 10⁻⁵ is not negligible next to such a small variance.

---

## 5. The backward pass (page 4), and a cleaner form

**The paper's chain rule, one line per intermediate:**
```
∂ℓ/∂x̂_i  = ∂ℓ/∂y_i · γ
∂ℓ/∂σ²_B = Σ_i ∂ℓ/∂x̂_i · (x_i − µ_B) · (−½)(σ²_B + ε)^(−3/2)
∂ℓ/∂µ_B  = Σ_i ∂ℓ/∂x̂_i · (−1/√(σ²_B + ε))  +  ∂ℓ/∂σ²_B · Σ_i −2(x_i − µ_B)/m
∂ℓ/∂x_i  = ∂ℓ/∂x̂_i /√(σ²_B + ε)  +  ∂ℓ/∂σ²_B · 2(x_i − µ_B)/m  +  ∂ℓ/∂µ_B /m
∂ℓ/∂γ    = Σ_i ∂ℓ/∂y_i · x̂_i
∂ℓ/∂β    = Σ_i ∂ℓ/∂y_i
```
- **Why x_i appears in three terms:** x_i affects the loss **three ways**: directly through its own x̂_i, through the batch mean µ_B, and through the batch variance σ²_B.
- **The chain rule adds up all three paths.**

### 5.1 The compact form (same math; `bn_backward_compact` in our code)
Substituting and simplifying (σ = √(σ²_B + ε), dy = ∂ℓ/∂y):
```
∂ℓ/∂x = (γ/σ) · ( dy − mean(dy) − x̂ · mean(dy · x̂) )
```
**What it says:**
- **The gradient reaching x has its batch mean removed** (Σ ∂ℓ/∂x = 0).
- **It also has its component along x̂ removed** (Σ ∂ℓ/∂x · x̂ = 0).
- **In words:** the layer below **cannot usefully shift or rescale the whole batch**, because BN would undo it, so the gradient doesn't even try. This is the formal version of section 2's lesson.

**Worked numbers** (continuing the example with γ = 1 and dy = (0.5, −1, 0.2, 0.3)):
```
mean(dy) = 0.0      mean(dy·x̂) = [0.5(−1.069) + (−1)(−0.535) + 0 + 0.3(1.604)]/4 = 0.120
∂ℓ/∂x = (1/1.871)·(dy − 0 − 0.120·x̂) = (0.336, −0.500, 0.107, 0.057)
check: sum = 0.000 ✔      Σ ∂ℓ/∂x·x̂ = 0.000 ✔
```

---

## 6. Algorithm 2: training vs inference (Section 3.1)

- **Training:** use **mini-batch** statistics. Each example's output depends on the other examples in its batch.
- **Inference:** we want a deterministic output that depends only on the input. Use **population** statistics:
  ```
  E[x]   = E_B[µ_B]                      average of the batch means
  Var[x] = m/(m − 1) · E_B[σ²_B]          unbiased variance
  ```

### 6.1 Why m/(m − 1)?
- **The bias:** σ²_B measures deviations from the **batch's own mean**, which sits *closer* to the batch's points than the true mean does. So it underestimates. Precisely:
  ```
  E[σ²_B] = ((m − 1)/m) · Var[x]
  ```
- **The intuition:** one of the m "degrees of freedom" was used up estimating the mean.
- **Multiplying by m/(m − 1) removes the bias.**
- **Our checks:**
  - **Batches of 4:** the plain average of batch variances is 3.0 for a true variance of 4 (factor 3/4), and the corrected value is 4.0.
  - **Batches of 10 (demo):** 3.55 (factor 9/10), and the corrected value is 3.945 ≈ 4.
- **In practice:** people track **moving averages** during training (our torch layer does this).

### 6.2 At inference, BN is a fixed linear map
```
y = [γ/√(Var[x] + ε)] · x + [β − γ E[x]/√(Var[x] + ε)] = a·x + c
```
It can be **folded** into the previous layer's weights (W ← aW, b ← ab + c), so it costs nothing at inference. Our demo learns y = 0.7552x − 2.0744.

---

## 7. Where to put it; convolutions (Section 3.2)

- **Before the nonlinearity:** z = g(BN(W u)).
  - **Not on u:** u is the output of the previous nonlinearity, so its shape keeps changing, and fixing two moments wouldn't stabilize it.
  - **W u is a sum of many terms,** which makes it "more Gaussian". Mean and variance then describe it well.
- **Drop the bias b:** BN subtracts the mean, which cancels any constant b. β takes its job.
- **Convolutions:**
  - **One γ and β per feature map** (channel).
  - **Statistics over the batch and all spatial positions:** an effective batch of m·p·q values for a p×q map.
  - **Why:** this keeps convolution's "the same operation everywhere in the image" property. Each location of a feature map is normalized identically.

---

## 8. Why higher learning rates become safe (Section 3.3)

### 8.1 Scale invariance, proved
For a scalar a > 0, the mean and standard deviation of (aW)u are a·µ and a·σ, so:
```
BN((aW)u) = (aWu − aµ)/(aσ) = (Wu − µ)/σ = BN(Wu)
```
Differentiating both sides:
```
∂BN((aW)u)/∂u    = ∂BN(Wu)/∂u                 the Jacobian w.r.t. the input doesn't depend on the weight scale
∂BN((aW)u)/∂(aW) = (1/a) · ∂BN(Wu)/∂W         bigger weights get SMALLER gradients
```
- **Why this is self-stabilizing:** if a large learning rate inflates the weights, their gradients shrink in proportion, so the updates calm down by themselves.
- **Our demo:** W×10 gives an identical output. Our test: W×7 gives identical outputs and ∂ℓ/∂u, and ∂ℓ/∂W divided by 7.

### 8.2 A conjecture
BN may push layer Jacobians toward singular values ≈ 1 (JJᵀ ≈ I), the ideal of Paper 007, so gradients neither explode nor vanish.

---

## 9. BN as a regularizer (Section 3.4)

- **The source of noise:** during training, an example's output depends on which random batch-mates it got, because they shift µ_B and σ_B.
- **That is noise injected into the hidden units,** like dropout. The paper found dropout could be reduced or removed.
- **The flip side:** with a batch size of 1, σ²_B = 0 and x̂ = 0. The layer outputs β regardless of the input. **BN needs reasonably large batches.** (Later fixes: Layer Norm, used in Transformers; Group Norm.)

---

## 10. Experiments (Section 4)

### 10.1 MNIST (Figure 1)
- **The network:** 784 → 3×100 sigmoid → 10, small Gaussian init, 50,000 steps, batch 60.
- **(a)** BN reaches higher test accuracy, faster.
- **(b, c)** The 15th/50th/85th percentiles of one last-layer sigmoid input drift a lot without BN, and stay stable with it.

### 10.2 ImageNet with Inception (Figures 2–3)
**"Accelerating" a BN net:**
- raise the learning rate (×5 or ×30);
- remove dropout and local response normalization;
- reduce L2 5×;
- decay the learning rate 6× faster;
- shuffle more thoroughly;
- use fewer distortions.

| Model | Steps to 72.2% | Best accuracy |
|---|---|---|
| Inception (lr 0.0015) | 31.0M | 72.2% |
| BN-Baseline (just add BN) | 13.3M | 72.7% |
| **BN-x5** | **2.1M (14× fewer)** | 73.0% |
| BN-x30 | 2.7M | **74.8%** |
| BN-x5-Sigmoid | n/a | 69.8% |

- **Without BN,** the same ×5 rate drove Inception's parameters to "machine infinity".
- **Sigmoid Inception never beat chance (0.1%) without BN; with BN it reaches 69.8%.**
- **An ensemble of 6 BN-Inceptions** reached **4.9% top-5 validation error** (4.82% on test), better than the best published result and than estimated human accuracy.

---

## 11. What our code found

**Scale note:** at your request, the MNIST experiments were **not run** on this laptop. `experiments.py` reproduces:
- Figure 1 (E1);
- a small version of Figures 2–3's learning-rate findings (E2);
- "sigmoid only trains with BN" on a 10-layer net (E3).

That's about 30–60 minutes on a CPU. The paper gives no learning rate or init scale for Figure 1; we chose lr 0.5 and std 0.1.

**Checked (tests and demo, about a second):**
- **Algorithm 1:** mean 0 and variance 1 per feature; γ = √Var, β = E recovers the identity.
- **Our hand-written backward** (one paper equation per line) matches finite differences, `torch.nn.functional.batch_norm`, and the compact form.
- **The gradient reaching x** sums to 0 and is orthogonal to x̂ (the worked example above too).
- **Scale invariance:** W×7 leaves the output and ∂ℓ/∂u unchanged, and divides ∂ℓ/∂W by 7.
- **Section 2's drift:** b goes to 41 after 200 steps with a constant loss; with the correct gradient it stays at 0.
- **Algorithm 2's m/(m − 1):** see section 6.1.
- **Inference BN** is one linear map a·x + c.
- **Conv BN** normalizes each channel over (N, H, W), and its gradient matches PyTorch.
- **Batch dependence:** in training mode an example's output changes with its batch-mates (the regularizing noise). In eval mode a single example gives the same output as in a batch.

---

## 12. Check yourself

1. Normalize x = (2, 4, 6, 8) by hand (ε = 0). (µ = 5, σ² = 5, x̂ = (−1.342, −0.447, 0.447, 1.342).)
2. Why are γ and β needed? How can they undo the normalization?
3. Explain the b-drift example. What does it teach?
4. From the compact backward formula, show that Σ_i ∂ℓ/∂x_i = 0. What does that mean physically?
5. Why is E[σ²_B] = ((m − 1)/m)·Var[x]? What is the fix?
6. Why is a bias before BN useless?
7. Prove BN((aW)u) = BN(Wu). What does this do to ∂ℓ/∂W, and why does that allow higher learning rates?
8. Why is BN with a batch size of 1 useless in training mode?
