# Szegedy et al. (2013), explained from scratch

**Paper:** *Intriguing Properties of Neural Networks*
**Authors:** Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, Rob Fergus
**Published at:** ICLR 2014 (arXiv:1312.6199)

Read Papers 013–014 (CNNs) first. This short paper found the strange side of networks that work very well. This guide explains the geometry behind it:
- distances to decision boundaries;
- why high dimensions make tiny changes powerful;
- why random noise is harmless while chosen noise is not;
- how to bound how much a network can amplify a change.

---

## 0. The whole idea in one line

> **(1) Single neurons are not special: random directions in a layer's activation space are just as "meaningful". (2) Networks that generalize well can be fooled by tiny, invisible, carefully chosen changes to the input, called "adversarial examples", and those changes often fool OTHER networks too.**

---

## 1. Property 1: units vs directions (Section 3)

- **The old method of interpretation:** to see what hidden unit i "means", find the images x′ that maximize its activation ⟨φ(x), e_i⟩, where φ(x) is the layer's activation vector and e_i is the i-th basis vector. If those images share a feature, the unit "detects" it.
- **The paper's control experiment:** do the same with a **random direction** v, i.e. maximize ⟨φ(x), v⟩.
- **The finding (Figures 1–4):** random directions give images that are **just as semantically coherent**, on MNIST and on AlexNet. For example, a random direction picked out "white dogs".
- **Conclusion:**
  - Meaning lives in the **space**, not in individual coordinates.
  - **Any rotation of a layer's units** would compute the same function if the next layer's weights were counter-rotated. So the network has no reason to align concepts with single units.
  - This casts doubt on "one unit = one concept". It resembles word vectors, where *directions* encode relations (Paper 019).

---

## 2. Property 2: blind spots (Section 4)

**The assumption ("smoothness prior"):** if x is classified correctly, any x + r with tiny ‖r‖ should be too.

**The finding:** **false** for deep nets. For **every** image tried, a nearly invisible r changed the prediction to **any chosen class**.

### 2.1 The geometry: distance to a decision boundary
- **The formula:** for a linear classifier with score s(x) = w·x + b, the boundary is the flat surface s = 0. The **shortest** move that reaches it goes straight along w:
  ```
  distance = |w·x + b| / ‖w‖₂,      r* = −(w·x + b) · w / ‖w‖₂²
  ```
- **Our demo:**
  - the boundary is x₀ = 0.5 (w = (1, 0), b = −0.5), and x = (0.3, 0.5);
  - the distance is |0.3 − 0.5|/1 = **0.2**, straight across;
  - our attack finds ‖r‖ = **0.2003**, without moving along x₁.
- **For a deep net** the boundary is curved and the attack has to *search* (next).

### 2.2 The attack (Section 4.1)
**What we'd like to solve:** for an image x and a target label l ≠ f(x),
```
minimize ‖r‖₂    subject to    f(x + r) = l,    x + r ∈ [0, 1]^m
```
"f(x + r) = l" is a hard, non-smooth constraint. **The penalty version:**
```
minimize  c·|r| + loss_f(x + r, l)    subject to  x + r ∈ [0, 1]^m
```
- The **loss** term (cross-entropy toward class l) pulls x + r into class l.
- The **c·|r|** term keeps r small.
- **How c trades the two off:**
  - **big c:** small r, but it may not reach class l;
  - **small c:** it reaches l easily, but with a bigger r.

  Our test confirms that a bigger c gives a smaller r.
- **The search:** do a **line search over c**, and keep the **largest c that still succeeds**.
  - The paper says "minimum c", but that would give the *largest* r.
  - For convex losses this penalty method finds the exact answer; for neural nets it's an approximation.
- **The solver:** **box-constrained L-BFGS**, a quasi-Newton method (Paper 005, section 8.4) that builds a curvature estimate from recent gradients. "Box-constrained" means it keeps every pixel inside [0, 1].

### 2.3 What they found (Section 4.2)
1. **Always possible:** for MNIST nets, AlexNet and the 1-billion-parameter "QuocNet", every sample had a nearby adversarial example. For AlexNet the average distortion was 0.0065 (Figure 5). Every image became an "ostrich".
2. **Cross-model generalization:** examples made for one network are often misclassified by networks with **different architectures, hyperparameters or initializations**.
3. **Cross-training-set generalization:** they even fool networks trained on a **disjoint half** of the data.

### 2.4 Table 1: MNIST models and the distortion needed
| Model | Train err | Test err | Min. distortion |
|---|---|---|---|
| FC10(λ=10⁻⁴): softmax only | 6.7% | 7.4% | 0.062 |
| FC10(λ=10⁻²) | 10% | 9.4% | 0.1 |
| FC10(λ=1) | 21.2% | 20% | 0.14 |
| FC100-100-10 (sigmoid) | 0% | 1.64% | 0.058 |
| FC200-200-10 | 0% | 1.54% | 0.065 |
| AE400-10 (autoencoder + softmax) | 0.57% | 1.9% | 0.086 |

- **Distortion** = √(Σ(x′ − x)²/n), the per-pixel RMS change (pixels in [0, 1]).
- **Even a plain linear softmax classifier is vulnerable.** Strong weight decay (λ = 1) raises the needed distortion, but the model gets much worse. Section 2.1 says why: shrinking w reduces |w·x + b| as well as ‖w‖.

### 2.5 Table 2: transfer between models (the error caused)
| Examples made for ↓ / fed to → | FC10(10⁻⁴) | FC100-100-10 | AE400-10 |
|---|---|---|---|
| FC10(10⁻⁴) | 100% | 2% | 2.7% |
| FC10(10⁻²) | 87.1% | 35.9% | 9.8% |
| FC10(1) | 71.9% | 48.1% | 34.4% |
| FC100-100-10 | 28.9% | 100% | 2% |
| FC200-200-10 | 38.2% | 20.3% | 2.7% |
| Gaussian noise, stddev 0.1 | 5.0% | 0% | 0.8% |
| Gaussian noise, stddev 0.3 | 15.6% | 5% | 3.1% |

**Random noise of equal or larger size does far less damage.** Adversarial perturbations are special directions, not just "noise".

### 2.6 Tables 3–4: disjoint training sets
- **The setup:** FC100-100-10 and FC123-456-10 were trained on half P1, and FC100-100-10′ on half P2 (30k images each).
- **The transfer:** examples made for FC100-100-10 fool FC123-456-10 26.2% of the time, and FC100-100-10′ 5.9%. Amplified to stddev 0.1, that becomes **98%** and **43%**.
- **For comparison:** Gaussian noise of stddev 0.1 causes only 2.6–2.8% error.

### 2.7 Adversarial training
- **The method:** a 100-100-10 net was trained with a **constantly refreshed pool** of adversarial examples (made for every layer) mixed into the training data.
- **The result:** **< 1.2% test error**, vs 1.6% with weight decay and ~1.3% with dropout. This is the first **adversarial training**.

---

## 3. Why high dimensions make this easy (the linear argument)

This was developed by Goodfellow, Shlens & Szegedy (2015); our demo makes it concrete.

### 3.1 Chosen changes add up
- **The setup:** take a linear score s = w·x with n inputs, and change **every** input by ±ε in the direction of sign(w_i).
- **The score moves by:**
  ```
  Δs = Σ_i ε |w_i| = ε ‖w‖₁ ≈ ε · n · (average |w_i|)          grows like n
  ```
- **The typical size of a score**, with random-signed inputs, is ~ √n · (size of w_i): **it grows only like √n**.
- **So a fixed per-pixel ε matters ~√n times more in high dimensions:**

  | inputs n | score shift from ε = 0.01 | typical score |
  |---|---|---|
  | 10 | 0.07 | 0.83 |
  | 784 (MNIST) | 6.4 | 6.2 |
  | 150,528 (ImageNet) | **1,201** | 33 |

  At ImageNet size, an invisible 0.01 change per pixel moves the score **36× more** than its typical size.
- **The FGSM attack** is this rule applied to a network's local linearization: x + ε·sign(∇ₓ loss).

### 3.2 Random changes mostly cancel
- **The setup:** random noise r with the same length ‖r‖ points in a random direction.
- **Its component along the one direction that matters** (the boundary's normal) is only about **‖r‖/√n**. In high dimensions, a random vector is nearly orthogonal to any fixed direction.
- **So random noise of a given size barely moves the score**, while the attack puts **all** of ‖r‖ into the one direction that counts.
- **Our demo:**
  - a chosen perturbation of stddev 0.045 flipped the label;
  - Gaussian noise of the **same** stddev flipped it **0 times in 200**.

---

## 4. A spectral bound on instability (Section 4.3)

### 4.1 Lipschitz constants
- **The definition:** a function g is **L-Lipschitz** if ‖g(x) − g(x′)‖ ≤ L‖x − x′‖: it can stretch distances by at most L.
- **Composition multiplies:**
  ```
  ‖φ(x) − φ(x + r)‖ ≤ (L₁ · L₂ · … · L_K) · ‖r‖              for φ = φ_K ∘ … ∘ φ₁
  ```
  **Proof:** apply the definition one layer at a time.

### 4.2 One layer's L: the operator norm
- **ReLU and max-pooling never stretch distances:** |max(0, a) − max(0, b)| ≤ |a − b|. So a layer max(0, Wx + b) has L ≤ **‖W‖**.
- **‖W‖ is the operator norm:** the largest factor by which W can stretch a vector, which equals W's **largest singular value**.
- **Computing it:**
  - **exactly with an SVD,** or
  - **by power iteration:** repeat v ← WᵀWv/‖WᵀWv‖; then ‖Wv‖ → ‖W‖. This is the same idea as Paper 006's λ_max.

### 4.3 Convolutions via the Fourier transform (Eq. 1)
- **The problem:** a convolution is a huge, structured matrix (every output pixel × every input pixel).
- **The key fact:** the 2-D discrete Fourier transform **diagonalizes** convolutions. In frequency space, a convolution becomes a separate small multiplication at each frequency ξ:
  ```
  output(ξ) = A(ξ) · input(ξ)          A(ξ) = the C_out × C_in matrix of the kernels' DFTs at frequency ξ
  ```
- **Parseval's theorem** says the DFT preserves lengths (it is a rotation, up to scale). So:
  ```
  ‖W_conv‖ = max over frequencies ξ of ‖A(ξ)‖
  ```
  The norm is found from many tiny SVDs instead of one giant one.
- **For AlexNet** the paper reports 2.75, 10, 7, 7.5, 11, 3.12, 4, 4. Every layer can amplify, starting from the first, and the product bound is enormous.
- **The limits:** this is only an **upper bound**. Large bounds don't *prove* instability, and the real worst case is usually much smaller. The paper suggests penalizing these norms; that later became spectral normalization and Lipschitz-constrained networks.

---

## 5. Why it matters

- **It started adversarial machine learning:** attacks, defences, certified robustness.
- **Goodfellow et al. (2015):** the linear explanation (section 3) and **FGSM**.
- **Madry et al. (2018):** adversarial training as robust optimization (PGD).
- **Transferability enables black-box attacks:** attack your own copy of a model, then use the examples on the real one.

---

## 6. What our code found

**Scale note:** at your request, nothing was trained on this laptop. `experiments.py` reproduces:
- Tables 1–2 (E1);
- Tables 3–4, cross training set (E3);
- Figures 1–2, units vs random directions (E4);
- spectral bounds for the trained nets, plus torchvision's AlexNet with `--alexnet` (E5);
- adversarial training (E6).

**Checked (tests and demo, about a second):**
- **The attack finds the true minimum on a linear boundary:** ‖r‖ = 0.2003 vs the exact 0.2, with no wasted movement along x₁, and always inside [0, 1].
- **A bigger c gives a smaller r,** hence "the largest c that succeeds".
- **The Fourier operator norm of a conv layer is exact:** it matches the largest singular value of the full convolution matrix (built by brute force) and power iteration, which also handles stride 2.
- **The Lipschitz bound holds** for random inputs and perturbations, for single ReLU layers and whole nets.
- **The linear argument:** the table in section 3.1.
- **Random vs chosen:** 0 flips in 200 Gaussian tries at the attack's own size.

**Two small issues in the paper's text:**
- It says "minimum c"; the logic needs the largest c that still succeeds.
- Its amplification formula x + 0.1·(x′ − x)/‖x′ − x‖₂ would give a much **smaller** change than "stddev 0.1". We follow the stated intent: rescale the perturbation to a per-pixel stddev of 0.1.

---

## 7. Check yourself

1. How did the paper test whether single units are special? Why does a rotation of the units argue against "one unit = one concept"?
2. Compute the distance from x = (0.3, 0.5) to the line x₀ = 0.5. What is the minimal r?
3. Write the penalty problem. Why does a larger c give a smaller r? Which c do you keep?
4. Show that a per-pixel change of ε moves w·x by ε‖w‖₁. Why does that matter more as n grows?
5. Why does random noise of the same size rarely flip the label? (Hint: the projection onto one direction is ~‖r‖/√n.)
6. Why is a ReLU layer's Lipschitz constant at most ‖W‖? How do you get a conv layer's ‖W‖ cheaply?
7. Why is the product of the layer norms only an *upper* bound?
8. What is cross-model transfer, and why is it dangerous?
