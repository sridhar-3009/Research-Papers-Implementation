# Zhang et al. (2017), explained from scratch

**Paper:** *Understanding Deep Learning Requires Rethinking Generalization*
**Authors:** Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
**Published at:** ICLR 2017 (best paper award; arXiv:1611.03530)

Read Papers 010 (dropout) and 012 (BatchNorm) first. This guide:
- explains the learning-theory tools the paper knocks down (Rademacher complexity, stability);
- proves the paper's memorization theorem on a 3-point example;
- shows with plain geometry why SGD finds the minimum-norm solution.

---

## 0. The whole idea in one line

> **The same networks that generalize well on real data can perfectly memorize random labels, and even pure noise images, just as easily. So whatever makes them generalize is not "limited capacity" and not explicit regularization, and classical learning theory can't explain it.**

---

## 1. The puzzle (Section 1)

- **Generalization gap** = (test error) − (training error). We want it small.
- **Deep nets** often have **more parameters than training examples** (Table 1: 24–35 parameters per CIFAR-10 image), yet their gap is small.
- **Classical theory** bounds the gap with one of:
  - a measure of **model complexity** (VC dimension, Rademacher complexity);
  - **stability** of the algorithm;
  - **regularization** (weight decay, dropout, early stopping).
- **The question:** does any of these explain deep nets?

---

## 2. The randomization test (Section 2)

Train standard architectures on CIFAR-10 (and ImageNet) after deliberately destroying the data:

| Setting | What is changed |
|---|---|
| **true labels** | nothing |
| **partially corrupted labels** | each label replaced by a uniformly random class with probability p |
| **random labels** | all labels random (and fixed across epochs) |
| **shuffled pixels** | one fixed pixel permutation applied to **every** image |
| **random pixels** | a **different** permutation for each image |
| **gaussian** | images replaced by Gaussian noise with matching mean and variance |

**Corruption arithmetic:** a random class is the original one 1/10 of the time, so with probability p the label is resampled, and a fraction **0.9p** actually changes. Our code checks this.

### 2.1 The central finding: deep neural networks easily fit random labels
- **Zero training error** in every setting, with the **same** hyperparameters and no tuning.
- **Optimization stays easy:** random labels take only a small constant factor longer (Figure 1b).
- **Noise images are fitted even faster than random labels** (Figure 1a). The paper's guess: noise images are more spread out than natural images of one class, so they are easier to separate.
- **Test error rises smoothly with corruption** up to 90%, which is chance for 10 classes (Figure 1c). The network fits the real signal **and** memorizes the noise.
- **ImageNet:** Inception V3 reaches **95.2% top-1 training accuracy on random labels** among 1000 classes, and about 90% even with dropout and weight decay.

### 2.2 Why this breaks complexity bounds (Section 2.2)
**Rademacher complexity:** how well can a class H of models fit **random ±1 labels** σ_i?
```
R̂_n(H) = E_σ [ sup_{h∈H} (1/n) Σ_i σ_i h(x_i) ]          between 0 (can't fit) and 1 (fits anything)
```
A classic theorem (for losses in [0, 1]) says that with probability ≥ 1 − δ, for every h ∈ H:
```
test error ≤ training error + 2·R̂_n(H) + O( √( log(1/δ) / n ) )
```
- **If the model can fit any labels,** R̂_n ≈ 1, and the bound says "test error ≤ training error + 2". That's **vacuous**: errors are at most 1 anyway.
- **VC dimension** has the same problem.
- **Our demo** (a linear model on 100 points):

  | features | Rademacher fit |
  |---|---|
  | 10 | 0.25 |
  | 50 | 0.69 |
  | 100 | **1.00** |
  | 400 | **1.00** |

  Once parameters ≥ data points, it fits every sign pattern.

**Uniform stability:**
- It asks how much the trained model changes if one training example is replaced. It's a property of the **algorithm** alone, independent of the labels.
- But the **same** algorithm generalizes on true labels and fails (90% error) on random ones, so stability can't tell them apart.

**The punchline:** randomizing labels is only a **data** transformation. Model, size, optimizer and hyperparameters all stay the same, yet the gap goes from small to 90%. **Any explanation must depend on the data.**

---

## 3. The role of regularization (Section 3)

### Table 1: CIFAR-10 (train / test accuracy, %)
| Model | Parameters | crop | weight decay | Train | Test |
|---|---|---|---|---|---|
| Inception | 1,649,402 | yes | yes | 100.0 | **89.05** |
| | | yes | no | 100.0 | 89.31 |
| | | no | yes | 100.0 | 86.03 |
| | | **no** | **no** | 100.0 | **85.75** |
| | | (random labels) | | 100.0 | 9.78 |
| Inception w/o BatchNorm | 1,649,402 | no | yes / no | 100.0 | 83.00 / 82.00 |
| AlexNet | 1,387,786 | yes | yes | 99.90 | 81.22 |
| | | no | no | 100.0 | 76.07 |
| MLP 3×512 | 1,735,178 | no | yes / no | 100.0 | 53.35 / 52.39 |
| MLP 1×512 | 1,209,866 | no | yes / no | 99.80 / 100.0 | 50.39 / 50.51 |

- **Regularization helps a little,** but with **everything off, the models still generalize well** (Inception: 85.75%).
- **ImageNet:** turning everything off costs about 18% top-1 (59.80% remains), still far above chance (0.1%). Augmentation alone gives 72.95%.
- **Verdict:** explicit regularization "may improve generalization performance, but is **neither necessary nor by itself sufficient**". It's a tuning knob.
- **Implicit regularizers:**
  - **Early stopping** could help on ImageNet, but not on CIFAR-10.
  - **BatchNorm** adds 3–4% on CIFAR-10.
  - Neither is *the* reason.

---

## 4. Finite-sample expressivity (Section 4)

**Theorem 1:** a **two-layer ReLU network with 2n + d weights** can represent **any** function on **any** n points in d dimensions.

### 4.1 The construction (Appendix C)
1. **Project:** pick a direction a so that all z_i = ⟨a, x_i⟩ are different. A random a works with probability 1.
2. **Sort and interleave thresholds:** b₁ < z₁ < b₂ < z₂ < … < b_n < z_n.
3. **The network:** c(x) = Σ_j w_j · max(⟨a, x⟩ − b_j, 0), one ReLU "kink" per threshold.
4. **On the data,** the outputs are A·w with A_ij = max(z_i − b_j, 0).
   - Since b_j < z_i exactly when j ≤ i, A is **lower-triangular** with a **positive diagonal** (z_i − b_i > 0).
   - A triangular matrix with a non-zero diagonal is invertible (its determinant is the product of the diagonal), so **w = A⁻¹y fits any y exactly**.
5. **Count:** a has d numbers, b has n, w has n, so **2n + d**.

### 4.2 Worked example: 3 points on a line (d = 1, a = 1)
- **Data:** x = 1, 2, 3 with labels y = (5, −1, 4). Put the thresholds at b = 0.5, 1.5, 2.5.
- **The matrix A_ij = max(x_i − b_j, 0):**
  ```
  x=1: [0.5, 0,   0  ]
  x=2: [1.5, 0.5, 0  ]
  x=3: [2.5, 1.5, 0.5]
  ```
- **Solve top to bottom:**
  - 0.5w₁ = 5, so w₁ = 10;
  - 1.5·10 + 0.5w₂ = −1, so w₂ = −32;
  - 2.5·10 + 1.5·(−32) + 0.5w₃ = 4, so w₃ = 54.
- **Check x = 2:** 10·1.5 − 32·0.5 = 15 − 16 = −1 ✔.
- **So any labels can be fitted, including nonsense ones.**

**Our demo:** a network with exactly 2n + d = **1,020** weights fits 500 points in 20-d to an error of ~10⁻⁸, for smooth labels (sin x₁) and random labels alike.

**The lesson:** capacity is **cheap**. As soon as the parameters slightly exceed n, even a depth-2 net can memorize. (The paper also gives depth-k versions with width O(n/k).)

---

## 5. Implicit regularization: what linear models teach (Section 5)

With d ≥ n features, Xw = y has **infinitely many** exact solutions. Which does SGD find, and does it generalize?

### 5.1 Curvature can't tell them apart
- **The Hessian:** for linear models, (1/n)·Xᵀ·diag(β)·X **doesn't depend on w**.
- **So** every exact solution sits in an equally "sharp" or "flat" valley. Flatness can't be what selects one.

### 5.2 SGD picks the minimum-norm solution: a geometric proof
**Step 1: SGD never leaves the span of the data.**
- Each update is w ← w − η·e_t·x_{i_t}, which adds a multiple of a data point.
- Starting from w = 0, w is always a combination of data points: **w = Xᵀα** (the **row space** of X).

**Step 2: inside that space, the solution is unique.**
- Xw = y with w = Xᵀα gives **XXᵀα = y** (Eq. 3, the kernel form), so w_min = Xᵀ(XXᵀ)⁻¹y.

**Step 3: it's the shortest (Pythagoras).**
- Any other solution is w = w_min + v with Xv = 0. That means v is **orthogonal** to every data point, hence to the row space and to w_min.
- So:
  ```
  ‖w‖² = ‖w_min‖² + ‖v‖²  ≥  ‖w_min‖²
  ```

**Our demo** (30 equations, 100 unknowns):

| solution | training error | ‖w‖ |
|---|---|---|
| SGD from 0 | 2·10⁻¹⁵ | **0.503** |
| Xᵀ(XXᵀ)⁻¹y | 8·10⁻¹⁶ | **0.503** (the same vector) |
| another exact fit | 5·10⁻¹⁴ | **16.7** |

### 5.3 It works surprisingly well, but norm isn't the whole story
- **Results:**
  - Exact kernel interpolation with **no regularization** reaches **1.2% test error on MNIST** (0.6% with Gabor wavelet preprocessing).
  - On CIFAR-10: 46% with a Gaussian kernel, 17% with random-CNN features (15% with ℓ₂).
- **But the norm doesn't predict generalization:** on MNIST the minimum norm is ~220 on raw pixels and ~390 with wavelets, yet the wavelet model makes **half** the errors.
- **So "SGD finds small-norm solutions"** is a piece of the story, not all of it.

---

## 6. Why it matters

- **It overturned the textbook explanation** ("they generalize because they're limited or regularized"). That launched research on:
  - the **implicit bias of SGD**;
  - **flat minima**;
  - **margin- and norm-based bounds**;
  - **double descent** (Belkin et al. 2019): test error can *improve* past the point where the model interpolates the training data;
  - **benign overfitting** (Bartlett et al. 2020).
- **For practice:** 100% training accuracy does **not** by itself mean overfitting. Memorization and generalization coexist.

---

## 7. What our code found

**Scale note:** at your request, nothing was trained on this laptop. `experiments.py` reproduces:
- Figure 1a, all the randomizations (E1);
- Figures 1b–1c, label corruption (E2);
- Table 1, the regularization on/off grid (E3);
- Section 5's kernel interpolation on MNIST (E4).

**Checked (tests and demo, about a second):**
- **Table 1's parameter counts exactly:** Inception **1,649,402**, AlexNet **1,387,786**, MLP 3×512 **1,735,178**, MLP 1×512 **1,209,866**.
  - Small Inception matches only if the **convs have biases and BatchNorm has no learnable parameters**. That also explains why "Inception w/o BatchNorm" has the **same** count.
  - Small AlexNet matches with 64 channels per conv (TensorFlow's CIFAR-10 model); the paper doesn't state it.
- **Theorem 1, run for real:** 1,020 weights fit 500 points (d = 20) with any labels to ~10⁻⁸.
- **Rademacher fit:** 0.25 / 0.69 / 1.00 / 1.00 for 10 / 50 / 100 / 400 features.
- **SGD from zero = the minimum-norm solution** (section 5.2's table). The linear Hessian is the same at every w.
- **Randomization tools:**
  - corruption changes a fraction 0.9p of the labels;
  - shuffled pixels reuse one permutation;
  - random pixels use a new one per image (channels move together);
  - Gaussian images match the per-channel mean and std.

**One issue in the paper:**
- Solving Eq. 3 with the **linear** kernel XXᵀ can't fit raw MNIST exactly. There are only d = 784 features for n = 60,000 points, so XXᵀ has rank ≤ 784 and is not invertible.
- The 1.2% result must use a nonlinear kernel (as their CIFAR-10 numbers do). Our E4 uses a Gaussian kernel.

---

## 8. Check yourself

1. Describe the randomization test. Why is it such a clean experiment?
2. Write the Rademacher bound. Why does R̂_n ≈ 1 make it useless?
3. Why can't uniform stability explain the difference between true and random labels?
4. With label corruption probability p and 10 classes, what fraction of labels actually change?
5. Prove Theorem 1: why is A invertible? Redo the 3-point example with y = (1, 1, 1).
6. Why does SGD from w = 0 stay in the row space? Use Pythagoras to show it gives the minimum-norm solution.
7. Why is the Hessian of a linear model useless for choosing between solutions?
8. If the minimum norm doesn't predict generalization (220 vs 390 on MNIST), what else might?
