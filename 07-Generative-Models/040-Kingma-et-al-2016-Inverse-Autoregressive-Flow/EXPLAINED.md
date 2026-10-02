# Inverse Autoregressive Flow (IAF), explained simply

**Paper:** Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever & Max Welling, *Improved Variational Inference with Inverse Autoregressive Flow*, NIPS 2016.

**In one sentence:** a VAE's encoder normally outputs a simple "blob" (a diagonal Gaussian) as its guess of which codes explain an image. IAF bends that blob into almost any shape with a few cheap, invertible steps. The bound gets tighter and the model gets better, while sampling stays one parallel pass.

---

## 1. The big idea

### 1.1 The problem left by the VAE (paper 039)
- The VAE's bound is

  ```
  L(x) = log p(x) − KL( q(z|x) ‖ p(z|x) )
  ```

- Whatever q can't match of the true posterior p(z|x) is **lost**, and it is lost twice:
  1. the bound is looser than log p(x);
  2. the decoder is pushed toward models whose posteriors happen to look like q (diagonal Gaussians). That limits the generator.
- A **diagonal** Gaussian q(z|x) = N(μ, diag σ²) can't represent:
  - **correlations** ("if z₁ is high, z₂ is probably low");
  - **curved** shapes (bananas);
  - **several modes**.

### 1.2 Normalizing flows (the general fix, Rezende & Mohamed 2015)
- Start with a simple sample z₀ ~ N(μ, σ²), then push it through invertible functions:

  ```
  z_t = f_t(z_{t−1}),   t = 1…T
  ```

- If you know each step's **Jacobian determinant**, you still know the density of the final z_T (Section 2).
- The catch: for a general function in D dimensions, the determinant costs O(D³).
- The earlier "planar flow" uses a 1-unit bottleneck per step (cheap), but it needs very long chains in high dimensions.

### 1.3 IAF's idea
- Use an **autoregressive** network: one whose i-th output depends only on inputs 1…i−1.
- Then each step's Jacobian is **triangular**, so its determinant is just the product of the diagonal. That is O(D).
- The step is a single **parallel** pass, so it scales to thousands of latent dimensions (the paper uses CIFAR-10 latent feature maps).

---

## 2. Change of variables: how densities move (Eq. 5)

### 2.1 In one dimension
- Let z ~ q₀ and y = f(z), with f increasing.
- Probability mass is conserved: q₁(y)·|dy| = q₀(z)·|dz|. So

  ```
  q₁(y) = q₀(z) / |f'(z)|      ⇒      log q₁(y) = log q₀(z) − log |f'(z)|
  ```

- **Worked example:** z ~ N(0, 1) and y = 3z (stretch by 3).
  - At z = 0.5, y = 1.5, and q₀(0.5) = 0.3521.
  - q₁(1.5) = 0.3521/3 = 0.1174.
  - Check: N(1.5; 0, 9) = (1/√(18π)) e^{−2.25/18} = 0.1330 · 0.8825 = **0.1174** ✓.
  - Stretching spreads the mass thinner, so the density divides by the stretch factor.

### 2.2 In D dimensions
|f'| becomes |det J|, where J = ∂f/∂z is the D × D Jacobian matrix:

```
log q(z_T) = log q(z_0) − Σ_t log |det (∂z_t / ∂z_{t−1})|        (Eq. 5)
```

A determinant is the factor by which f scales tiny volumes.

---

## 3. Autoregressive transformations (Section 3)

### 3.1 The autoregressive Gaussian, and its two directions
- An autoregressive model generates y one coordinate at a time:

  ```
  y_i = μ_i(y_{1:i−1}) + σ_i(y_{1:i−1}) · ε_i
  ```

- **Sampling** is sequential: you need y₁ before y₂, and so on. That is D passes.
- The **inverse** (recover ε from y) is **parallel**, because every μ_i(y_{<i}) and σ_i(y_{<i}) can be computed at once from the known y:

  ```
  ε = (y − μ(y)) / σ(y)                                           (Eq. 7)
  ```

### 3.2 Its Jacobian is triangular
- ε_i depends on y_i (through the division) and on y_{<i} (through μ_i, σ_i), but **not** on y_{>i}.
- So ∂ε_i/∂y_j = 0 for j > i: the Jacobian is lower-triangular.
- Its diagonal is ∂ε_i/∂y_i = 1/σ_i.
- The determinant of a triangular matrix is the product of its diagonal:

  ```
  log |det ∂ε/∂y| = − Σ_i log σ_i                                 (Eq. 8)
  ```

**Worked 2-D example:**
- Let μ₁ = 0, σ₁ = 1, μ₂ = 0.8·y₁, σ₂ = 0.6.
- Then ε₁ = y₁ and ε₂ = (y₂ − 0.8 y₁)/0.6.
- J = [[1, 0], [−0.8/0.6, 1/0.6]] = [[1, 0], [−1.333, 1.667]].
- det = 1 · 1.667 − 0 = 1.667 = 1/(σ₁σ₂) ✓.
- The off-diagonal −1.333 doesn't matter, which is why μ and σ can be **any** neural network of the earlier coordinates.

### 3.3 The key insight
**Use the fast direction for what VAEs need.** A VAE needs to:
1. **sample** z from q;
2. know **log q(z)** at that sample.

Normalizing flows built from the inverse transformation do both in one parallel pass. That is "inverse autoregressive".

---

## 4. The IAF step (Section 4, Algorithm 1)

### 4.1 The chain
1. The encoder outputs μ₀, σ₀ and an extra **context** vector h.
2. Start: z₀ = μ₀ + σ₀ ⊙ ε, with ε ~ N(0, I).
3. For t = 1…T, run an autoregressive network on (z_{t−1}, h) to get m_t, s_t, then

   ```
   σ_t = sigmoid(s_t)                                              (Eq. 13)
   z_t = σ_t ⊙ z_{t−1} + (1 − σ_t) ⊙ m_t                            (Eq. 14)
   ```

4. Log density (Eq. 11):

   ```
   log q(z_T|x) = − Σ_i [ ½ ε_i² + ½ log 2π + Σ_{t=0}^{T} log σ_{t,i} ]
   ```

### 4.2 Why this update?
- It is the same as Eq. 10, z_t = μ_t + σ_t ⊙ z_{t−1}, with μ_t = (1−σ_t) m_t. So Eq. 11 still applies:
  - the Jacobian is triangular, because m_t, σ_t only see z_{t−1, <i};
  - its diagonal is σ_t.
- It looks like an **LSTM gate**: σ is a "forget gate" mixing the old z with a new proposal m.
- With σ ∈ (0, 1), it is **numerically stable**: it can never blow z up.
- **Forget-gate bias:** start with s_t ≈ +1 or +2, so σ ≈ 0.73–0.88 and each step barely moves z at first. Training then starts near the plain diagonal VAE.
  - Our code adds a bias of 1.5, so σ = sigmoid(1.5) = 0.82, and zero-initialises the output layer.

### 4.3 Algorithm 1, traced on numbers (D = 2, T = 1)

| step | values |
|---|---|
| encoder | μ₀ = (0, 1), σ₀ = (1, 0.5); noise ε = (0.4, −1.0) |
| start | z₀ = (0 + 0.4, 1 − 0.5) = (0.4, 0.5) |
| starting log q | −[½(0.16 + 1.0) + log 2π + log 1 + log 0.5] = −[0.58 + 1.8379 − 0.6931] = **−1.7248** |
| autoregressive net | m = (0, 2·z₀₁) = (0, 0.8) and s = (2, 0), so σ = (0.881, 0.5) (σ₁ and m₁ see nothing; σ₂ and m₂ see only z₀₁) |
| new z | z₁ = (0.881·0.4 + 0.119·0, 0.5·0.5 + 0.5·0.8) = (0.352, 0.65) |
| update log q | log q −= log 0.881 + log 0.5 = −0.1266 − 0.6931 = −0.8197, so **log q(z₁) = −0.9051** |

The density went **up**: the step squeezed space (σ < 1), so the same mass sits in a smaller volume.

### 4.4 Reversing the order between steps
- Within a step, z₁ is never changed by anything (it has no predecessors).
- Flipping the order between steps lets every coordinate depend on every other after two steps.
- A permutation has |det| = 1, so Eq. 11 is unchanged.

### 4.5 The autoregressive network: MADE (Germain et al. 2015)
A normal MLP with **masked** weights:
- give each input a degree 1…D and each hidden unit a degree 1…D−1;
- a hidden unit may connect to an input only if its degree ≥ the input's degree;
- output i may connect to a hidden unit only if i > that unit's degree.

So every path from input j to output i has j < i (**strictly** autoregressive).

Our demo prints the result for D = 4: output 1 sees nothing, output 2 sees z₁, output 3 sees z₁ and z₂, and output 4 sees z₁ to z₃.

### 4.6 The simplest IAF: linear (Appendix A)
- With no hidden layer, an IAF step is z = L·y with L **unit lower-triangular** (ones on the diagonal), produced by the encoder.
- det L = 1, so log q(z) = log q(y).
- The covariance becomes L diag(σ²) Lᵀ, and **every** covariance matrix can be written that way (the LDLᵀ / Cholesky decomposition).
- So one linear IAF step upgrades a diagonal Gaussian to a full-covariance Gaussian.

---

## 5. How much does a diagonal Gaussian lose? (exact math)

**The setup:**
- The true posterior is N(0, Σ) with Σ = [[1, ρ], [ρ, 1]].
- Fit a diagonal q by minimising KL(q ‖ p) (which is what maximising the ELBO does).

**Step 1: the optimum.**
- For this direction of KL, the optimal diagonal variances are 1/Λ_ii, where Λ = Σ⁻¹ is the precision matrix.
- Here Λ_ii = 1/(1−ρ²).

**Step 2: the KL of two zero-mean Gaussians.**

```
KL = ½ [ tr(Λ Σ_q) − D + log det Σ − log det Σ_q ]
```

**Step 3: plug in.**
- tr(Λ Σ_q) = Σ_i Λ_ii · (1/Λ_ii) = D, so the first two terms cancel.
- KL = ½ [ log(1−ρ²) − 2 log(1−ρ²) ] = **−½ log(1 − ρ²)**.

| ρ | gap (nats) | what our demo's diagonal q reached | linear IAF |
|---|---|---|---|
| 0.5 | 0.144 | 0.143 | 0.001 |
| 0.9 | 0.830 | 0.833 | 0.002 |
| 0.99 | 1.959 | 1.974 | 0.004 |

- The diagonal q gets stuck exactly at the theoretical gap. One linear IAF step removes it.
- Note that the gap is **per pair of correlated latents**. A real posterior over 32 or 1024 latents has many correlations, so the losses add up.

---

## 6. Free bits (Appendix C.8, Eq. 15)

**The problem:**
- Early in training the decoder is weak, so the easiest way to raise the bound is q(z|x) = p(z) (KL = 0, codes carry no information).
- Encoder gradients there are noisy, so training can get stuck in this "posterior collapse".

**The fix:** give each group j of latents λ "free" nats:

```
L̃_λ = E[log p(x|z)] − Σ_j max( λ, E_batch[ KL(q(z_j|x) ‖ p(z_j)) ] )
```

- Below λ nats, a group's KL costs nothing extra (the gradient of max(λ, ·) is 0), so the model is never pushed to switch it off.
- Above λ, the objective is the normal bound.
- **Example:** λ = 0.5, and two groups with average KL 0.2 and 1.5. The penalty is max(0.5, 0.2) + max(0.5, 1.5) = 0.5 + 1.5 = **2.0**.
  - The first group feels no pressure to shrink further; it is free to grow up to 0.5 nats.
  - This is exactly what our test checks.
- **Paper:** λ ∈ [0.125, 2] improved CIFAR-10 by more than 0.1 bits/dim, and Figure 7 shows more stochastic layers being used.

---

## 7. Discretized logistic likelihood (Appendix C.5)

- Pixels are integers 0…255. Model each with a **logistic** distribution (location μ, scale s) and give each integer the mass of its bin:

  ```
  P(x) = CDF((x + 1/256 − μ)/s) − CDF((x − μ)/s),     CDF = sigmoid
  ```

  (The paper scales pixels to [0, 1], so a bin is 1/256 wide.)
- **Why logistic?** Its CDF is the sigmoid, so the bin probability is a cheap closed form.
- **Edge bins:** our code gives the lowest bin everything below it and the highest bin everything above, so the 256 probabilities sum to exactly 1 (tested). This is the usual implementation detail.
- **Bits per dimension** = −log₂ p(x) / (3 · 32 · 32).
  - A uniform model over 256 values costs log₂ 256 = **8 bits/dim**. That is Table 2's first row.

---

## 8. Results from the paper

### 8.1 Table 1: MNIST (dynamically binarized), conv VAE with 32 latents

| Model | VLB | log p(x) ≈ (128 importance samples) |
|---|---|---|
| Convolutional VAE + HVI | −83.49 | −81.94 |
| DLGM 2hl + IWAE | | −82.90 |
| LVAE | | −81.74 |
| DRAW + VGP | −79.88 | |
| **Diagonal covariance** | −84.08 (±0.10) | −81.08 (±0.08) |
| **IAF (depth 2, width 320)** | −82.02 (±0.08) | −79.77 (±0.06) |
| **IAF (depth 2, width 1920)** | −81.17 (±0.08) | −79.30 (±0.08) |
| **IAF (depth 4, width 1920)** | −80.93 (±0.09) | −79.17 (±0.08) |
| **IAF (depth 8, width 1920)** | **−80.80** (±0.07) | **−79.10** (±0.07) |

**How to read it:**
- More expressive posteriors improve **both** columns.
- The gap VLB − log p(x) shrinks from 3.0 nats (diagonal) to 1.7 nats (deep IAF): the bound is tighter.
- The *log-likelihood itself* also improves by 2 nats: a better q lets the decoder become a better generator.

### 8.2 Table 2: CIFAR-10, bits/dim (lower is better)

| Method | bits/dim |
|---|---|
| Uniform | 8.00 |
| Multivariate Gaussian | 4.70 |
| NICE | 4.48 |
| Deep GMMs | 4.00 |
| Real NVP | 3.49 |
| PixelRNN | 3.00 |
| Gated PixelCNN | 3.03 |
| Deep Diffusion | 5.40 |
| Convolutional DRAW | 3.58 |
| **ResNet VAE with IAF** | **3.11** |

- This was the best latent-variable model of its time, close to PixelRNN/PixelCNN.
- **Sampling speed:** 0.05 s/image vs **52 s/image** for a naive PixelCNN (about 1000× faster), because a VAE generates all pixels in one pass.

### 8.3 Table 4: CIFAR-10 ablation (bits/dim; ResNet depth 4 / 8 / 12)

| Posterior | 4 | 8 | 12 |
|---|---|---|---|
| Bottom-up, factorized Gaussians | 3.71 | 3.55 | 3.44 |
| Bottom-up, IAF linear, 1 step | 3.68 | 3.55 | 3.41 |
| Bottom-up, IAF 1 hidden layer, 1 step | 3.61 | 3.49 | 3.38 |
| Bidirectional, factorized Gaussians | 3.74 | 3.60 | 3.46 |
| Bidirectional, IAF linear, 1 step | 3.67 | 3.52 | 3.40 |
| Bidirectional, IAF 1 hidden layer, 1 step | 3.56 | 3.42 | 3.28 |
| Bidirectional, IAF 2 hidden layers, 1 step | 3.54 | 3.39 | 3.27 |
| Bidirectional, IAF 1 hidden layer, 2 steps | **3.53** | **3.36** | **3.26** |

- Nonlinear IAF beats linear IAF, which beats diagonal.
- Bidirectional inference helps only together with IAF.
- **Table 3:** a fixed σ = 1 ("location-only") is almost as good as a learned scale (for example 3.45 vs 3.42).

### 8.4 Appendix D: the same thing as an autoregressive prior
- A factorized prior + an IAF posterior equals an **autoregressive prior** + a factorized posterior, just viewed in the "whitened" coordinates.
- Making q more flexible and making p(z) more flexible are two sides of one coin.

---

## 9. Why it works (the intuition)

1. **Tight bound ⇒ good learning signal.** The decoder is trained through the bound. If q can't match the posterior, the decoder is penalised for having posteriors q can't match. A flexible q removes that penalty.
2. **Triangular Jacobians make densities free.** log q needs only Σ log σ, which IAF computes anyway.
3. **Parallel in the right direction.** Sampling and scoring the sample (all a VAE needs) are one pass per step, with no loop over dimensions. Only scoring an *arbitrary* z would need the slow inverse, and VAEs never need that.
4. **Stable by construction.** The gated update with σ ∈ (0, 1) and a positive bias starts near the identity and never explodes.

---

## 10. What our code found

All numbers come from `demo.py` (about 5 s) and `test_iaf.py` (9 tests, about 1.3 s).

1. **The autoregressive property is exact:**
   - MADE Jacobians are strictly lower-triangular (tested for 0, 1 and 2 hidden layers, with a context input);
   - every IAF step's autograd log-determinant equals Σ log σ;
   - the location-only step has log det = 0.
2. **The density is correct:**
   - Algorithm 1's log q integrates to **1** (within 3%, checked by importance sampling in 2-D);
   - it matches the log q obtained by inverting the step and applying Eq. 5 by hand;
   - the inverse is exact to 5·10⁻⁷ but needs D sequential passes. For D = 128 that is 13.2 ms vs 0.14 ms for the forward pass.
3. **Diagonal vs linear IAF** on a correlated Gaussian: the diagonal q stops at the theory gap −½ log(1−ρ²) (0.833 vs 0.830 at ρ = 0.9), while linear IAF reaches about 0 (table in Section 5).
4. **A curved ("banana") posterior**, KL after 500 steps:

   | posterior | KL (nats) | parameters |
   |---|---|---|
   | diagonal | 0.887 | 4 |
   | linear IAF | 0.901 | 12 |
   | planar flow, K = 8 | 0.756 | 44 |
   | IAF, T = 1 | 0.080 | 232 |
   | IAF, T = 2 | 0.027 | 460 |
   | IAF, T = 4 | **0.021** | 916 |

   - Gaussians can't bend.
   - **Fairness note:** the planar flow here has far fewer parameters than the IAF steps, so this isn't an equal-budget comparison. It only shows that a bottleneck flow needs many more steps.
5. **Figure 1's toy** (a VAE with a 2-D latent and 4 datapoints):

   | posterior | bound | log p(x) | gap |
   |---|---|---|---|
   | diagonal | −2.435 | −1.683 | **0.752** nats |
   | IAF | −2.060 | −1.629 | **0.431** nats |

   - The best possible log p(x) is −1.386.
   - The more flexible posterior gives a tighter bound and a slightly better model, the same pattern as Table 1.
6. **Free bits:** the penalty and its zero gradient below λ behave as in Eq. 15. The discretized logistic sums to 1 over 256 bins.

**Not run (too heavy):** `experiments.py` covers Table 1 (MNIST, 5 posteriors × seeds), Table 3, a single-layer CIFAR-10 version of Tables 2/4 with free bits, Figure 7's free-bits study, Figure 1, and synthesis speed. The paper's 20-layer bidirectional ResNet VAE isn't reproduced.

---

## 11. Check yourself

1. z ~ N(0, 1) and y = 0.5 z. What is log q(y) at y = 0.25?
   <details><summary>Answer</summary>z = 0.5, and log N(0.5; 0, 1) = −0.919 − 0.125 = −1.044. Subtract log 0.5 = −0.693, so log q(y) = −1.044 + 0.693 = −0.351. Squeezing raises the density.</details>
2. Why is the determinant of an IAF step just Π σ_i?
   <details><summary>Answer</summary>Output i depends on z_i only through the σ_i · z_i term (m_i and σ_i see only z_{<i}), so the Jacobian is triangular with σ_i on the diagonal, and a triangular determinant is the product of the diagonal.</details>
3. Why is IAF fast for VAEs but slow as a density model for arbitrary data?
   <details><summary>Answer</summary>Going from noise to z (sampling) is one parallel pass, and log q comes out for free. Computing the density of a given z requires inverting, which is sequential over D coordinates.</details>
4. A diagonal Gaussian approximates N(0, [[1, 0.8], [0.8, 1]]). What is the minimal KL?
   <details><summary>Answer</summary>−½ log(1 − 0.64) = −½ log 0.36 = 0.511 nats.</details>
5. What does the forget-gate bias do at the start of training?
   <details><summary>Answer</summary>s ≈ +1.5 makes σ ≈ 0.82, so z_t ≈ 0.82 z_{t−1} + 0.18 m: each step changes z only a little. The model starts close to the diagonal VAE and learns to use the flow gradually.</details>
6. With free bits λ = 1 and group KLs (0.3, 0.9, 2.5), what is the KL penalty, and which groups get gradient?
   <details><summary>Answer</summary>1 + 1 + 2.5 = 4.5. Only the third group (above λ) gets gradient pushing its KL down.</details>
7. Why is reversing the variable order between steps allowed without changing log q?
   <details><summary>Answer</summary>A permutation is volume-preserving (|det| = 1), so it adds log 1 = 0 to the sum in Eq. 5.</details>
