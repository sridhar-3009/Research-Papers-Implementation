# Auto-Encoding Variational Bayes (the VAE), explained simply

**Paper:** Diederik P. Kingma & Max Welling, *Auto-Encoding Variational Bayes*, ICLR 2014.

**In one sentence:** to learn a model that can *generate* data from hidden "codes", train two networks together:
- an **encoder** that guesses which codes could have produced an image;
- a **decoder** that turns codes back into images.

Both maximise one number, the **evidence lower bound (ELBO)**. A small trick, writing a random sample as `z = μ + σ·ε`, lets ordinary backpropagation do it.

---

## 1. The big idea

### 1.1 What we want
- We want a **generative model**: a machine that can produce new, realistic images (new handwritten digits, new faces).
- The story we assume:
  1. pick a hidden **code** z (a few numbers, like "slant", "thickness", "which digit") from a simple distribution p(z);
  2. turn the code into an image x with a neural network: p(x|z).
- To **train** it, we want the model to give our real images high probability. That probability is

  ```
  p(x) = ∫ p(z) p(x|z) dz        (average over every possible code)
  ```

### 1.2 The problem
- That integral is **intractable**: z has, say, 20 dimensions, and p(x|z) is a neural network. There is no formula, and averaging over random z's is hopeless, because almost every random code produces an image that looks nothing like *this* x.
- For the same reason, the **posterior** p(z|x) = p(x|z)p(z)/p(x) ("which codes explain this image?") is intractable too.

### 1.3 The solution, in three moves
1. **Approximate the posterior** with a network q(z|x), the **encoder** (the paper's "recognition model"). Given x, it outputs a Gaussian over z: a mean μ(x) and a spread σ(x).
2. **Optimise a lower bound** on log p(x) that only needs samples from q. This is the ELBO.
3. **Make sampling differentiable** with the reparameterization trick: z = μ + σ·ε with ε ~ N(0, 1). Gradients now flow through μ and σ into the encoder.

The result looks like an **auto-encoder**:
- x → encoder → z → decoder → x̂;
- plus a penalty that keeps the codes close to N(0, I).

That is the **variational auto-encoder (VAE)**.

---

## 2. Background you need

### 2.1 Probability densities and logs
- A **Gaussian** N(z; μ, σ²) has density (1/√(2πσ²)) · exp(−(z−μ)²/(2σ²)).
- Its log is

  ```
  log N(z; μ, σ²) = −½ [ log(2π) + log σ² + (z−μ)²/σ² ]
  ```

  - Example: z = 1, μ = 0, σ² = 1 gives −½(1.8379 + 0 + 1) = **−1.419**.
- We work with **logs** because probabilities of images are tiny products (784 pixels). Logs turn them into sums.

### 2.2 Expectation
- E_q[f(z)] = "average of f(z) when z is drawn from q".
- **Monte Carlo** estimate: draw z₁…z_L from q and average f(z_l).

### 2.3 KL divergence
KL(q‖p) = E_q[log q(z) − log p(z)].
- It measures how different q is from p. It is **≥ 0**, and **= 0 only if q = p**.
- **Why ≥ 0 (Jensen):**
  - −KL = E_q[log(p/q)] ≤ log E_q[p/q] = log ∫ p = log 1 = 0.
  - The inequality is Jensen's, because log is concave (the log of an average ≥ the average of the logs).

### 2.4 Bernoulli likelihood (binary pixels)
- If the decoder says pixel i is on with probability y_i, then

  ```
  log p(x|z) = Σ_i [ x_i log y_i + (1−x_i) log(1−y_i) ]
  ```

  This is minus the binary cross-entropy (App. C.1).
- **Example:** pixels x = (1, 0) and y = (0.9, 0.2) give log 0.9 + log 0.8 = −0.105 − 0.223 = **−0.328 nats**.

---

## 3. The ELBO, derived step by step (Eqs. 1–3)

### 3.1 The identity (Eq. 1)

```
log p(x) = KL( q(z|x) ‖ p(z|x) )  +  L(x)
```

**Derivation**, for any q:
1. log p(x) doesn't depend on z, so log p(x) = E_q[log p(x)].
2. By Bayes, p(x) = p(x, z)/p(z|x). So log p(x) = E_q[ log p(x,z) − log p(z|x) ].
3. Add and subtract log q(z|x):

   log p(x) = E_q[ log p(x,z) − log q(z|x) ] + E_q[ log q(z|x) − log p(z|x) ].

4. The second term is KL(q ‖ posterior). The first is **L(x)**, the ELBO (Eq. 2):

   ```
   L(x) = E_q[ log p(x, z) − log q(z|x) ]                     (Eq. 2)
   ```

**What this tells us:**
- KL ≥ 0, so **L(x) ≤ log p(x)**: it is a *lower bound* on the "evidence" log p(x).
- log p(x) is fixed for given decoder weights. So pushing L **up** with respect to the encoder pushes the KL **down**: the encoder becomes a better approximation of the true posterior.
- Pushing L up with respect to the decoder raises (a bound on) the likelihood of the data: the generator improves.
- **One objective trains both networks.** That is the key difference from wake-sleep (Section 7).

### 3.2 The auto-encoder form (Eq. 3)
Split log p(x, z) = log p(x|z) + log p(z):

```
L(x) = E_q[ log p(x|z) ]  −  KL( q(z|x) ‖ p(z) )                   (Eq. 3)
        reconstruction        regularizer
```

- **Reconstruction:** encode x into codes, decode them, and check how likely the original x is. That is an auto-encoder's reconstruction error, but averaged over *noisy* codes.
- **Regularizer:** keep each image's code distribution close to the prior N(0, I).
  - Without it, the encoder could give every image a far-away, razor-thin code (memorising).
  - With it, codes overlap and fill the prior, so a *random* z ~ N(0, I) decodes to a sensible image. That is why we can sample new images.

### 3.3 Worked example on a model where everything is exact (our demo, section 1)

**The model:**
- p(z) = N(0, 1) and p(x|z) = N(2z, 0.25).
- Then p(x) = N(0, 2² + 0.25) = N(0, 4.25).
- The true posterior for x = 1.3 is N(0.612, 0.0588), from Gaussian algebra:
  - posterior variance v = 1/(1 + w²/s²) = 1/(1 + 16) = 0.0588;
  - posterior mean = v · w · x / s² = 0.0588 · 2 · 1.3 / 0.25 = 0.612.

**Results:**

| q(z) | bound L | KL(q ‖ posterior) | L + KL |
|---|---|---|---|
| N(0.612, 0.0588), the posterior | −1.8412 | 0.0000 | −1.8412 |
| N(0.500, 0.0588) | −1.9474 | 0.1062 | −1.8412 |
| N(0.612, 0.5) | −4.5212 | 2.6800 | −1.8412 |
| N(0, 1), the prior | −11.6058 | 9.7646 | −1.8412 |

- log p(x) = −1.8412.
- Every row adds up to exactly log p(x). That is Eq. 1.
- The better q matches the posterior, the tighter the bound.

---

## 4. The Gaussian KL in closed form (Appendix B, Eq. 10)

- Encoder: q(z|x) = N(μ, diag σ²). Prior: p(z) = N(0, I). J = the number of latent dimensions.
- Everything splits per dimension, so do one dimension:

**Cross term:**
- E_q[log p(z)] = E_q[−½ log 2π − ½ z²] = −½ log 2π − ½ (μ² + σ²).
- This uses E[z²] = μ² + σ² (mean² + variance).

**Entropy term:**
- E_q[log q(z)] = E_q[−½ log 2π − ½ log σ² − (z−μ)²/(2σ²)] = −½ log 2π − ½ log σ² − ½.
- This uses E[(z−μ)²] = σ².

**Subtract:**

```
−KL = E_q[log p] − E_q[log q] = ½ (1 + log σ² − μ² − σ²)
```

**Sum over dimensions:**

```
−KL(q ‖ p) = ½ Σ_j ( 1 + log σ_j² − μ_j² − σ_j² )                (App. B)
```

**Checks:**
- μ = 0, σ = 1 gives ½(1 + 0 − 0 − 1) = **0**. q equals the prior, so there is no penalty. ✓
- μ = 2, σ = 1 gives ½(1 + 0 − 4 − 1) = −2, so **KL = 2 nats**. Moving the mean costs μ²/2.
- μ = 0, σ = 0.1 gives ½(1 + log 0.01 − 0 − 0.01) = ½(1 − 4.605 − 0.01) = −1.81, so **KL = 1.81**. Shrinking the spread is penalised too (the −log σ² part), which stops codes from becoming points.

Our test compares this formula with PyTorch's own KL on random inputs.

---

## 5. The reparameterization trick (Section 2.4)

### 5.1 The problem: you can't backprop through a coin flip
- We need the gradient of E_{q_φ}[f(z)] with respect to the encoder parameters φ.
- The randomness depends on φ: change φ and the *distribution* you sample from changes.
- "Sample z, then differentiate f(z)" ignores that dependence.

### 5.2 The old way: the score-function estimator (Section 2.2)
- It uses ∇_φ q = q · ∇_φ log q:

  ```
  ∇_φ E_q[f(z)] = E_q[ f(z) ∇_φ log q(z) ]
  ```

- It is unbiased, but it multiplies f(z), which can be big and isn't centred, by a random score.
- **Very high variance.** The paper calls it "impractical for our purposes".

### 5.3 The trick
- Write the sample as a deterministic function of the parameters plus *parameter-free* noise:

  ```
  z = g_φ(ε, x) = μ + σ ⊙ ε,     ε ~ N(0, I)                         (Eq. 4)
  ```

- Now E_q[f(z)] = E_ε[f(μ + σε)].
- The distribution being averaged over (ε's) doesn't depend on φ, so the gradient moves inside:

  ```
  ∇_μ E[f] = E_ε[ f'(μ + σε) ],      ∇_σ E[f] = E_ε[ f'(μ + σε) · ε ]
  ```

- The gradient uses the **slope of f**, which is much more informative than f's value times a random score.

**Worked example** (our demo, section 2): f(z) = z², μ = 1, σ = 1.

| | score function | reparameterized |
|---|---|---|
| true d/dμ | 2μ = 2 | 2μ = 2 |
| per-sample gradient | z²·(z−μ)/σ² | 2z |
| mean | 1.991 | 1.990 |
| variance | **30.35** | **3.99** |

- The reparameterized variance is Var(2z) = 4σ² = 4 exactly. ✓
- Both estimators are unbiased, but the reparameterized one has **8× less variance** for μ and **12×** for log σ.
- Real networks have thousands of parameters and messier f's, and there the gap is much larger.

### 5.4 Which distributions can be reparameterized?
The paper lists three recipes:
1. **Tractable inverse CDF:** z = F⁻¹(u) with u ~ Uniform(0, 1). Works for the exponential, Cauchy, logistic, Gumbel and others.
2. **Location–scale families:** z = location + scale · ε. Works for the Gaussian, Laplace, Student-t and others.
3. **Compositions:** log-normal = exp(Gaussian); Gamma = a sum of exponentials; and so on.

Discrete z (coin flips, categories) has no such trick. That case needed later ideas (Gumbel-softmax, REINFORCE with baselines).

---

## 6. The SGVB estimators and the AEVB algorithm (Section 2.3, Algorithm 1)

### 6.1 Estimator A (generic, Eq. 6)

```
L_A(x) = (1/L) Σ_l [ log p(x, z_l) − log q(z_l|x) ],   z_l = μ + σ ⊙ ε_l
```

Everything is sampled.

### 6.2 Estimator B (Eq. 7 and Eq. 10 for the Gaussian VAE)

```
L_B(x) = ½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)  +  (1/L) Σ_l log p(x | z_l)
```

- The KL part is computed **exactly** (Section 4), and only the reconstruction is sampled.
- Fewer random parts means lower variance.
- **Our demo** (one random small VAE, 4000 repeats, one sample each):

  | estimator | mean | standard deviation |
  |---|---|---|
  | A | −10.616 | 3.187 |
  | B | −10.610 | **0.749** |

  The means match; B is about 4× less noisy.

### 6.3 Minibatches (Eq. 8)
- The full-data bound is estimated by (N/M) Σ over a minibatch of M points.
- The paper uses **M = 100** and **L = 1**: one noise sample per image is enough when the batch is big, because the batch averages the noise away.

### 6.4 Algorithm 1 (AEVB)

```
repeat
    X_M ← random minibatch of M = 100 datapoints
    ε   ← random noise, one per datapoint
    g   ← ∇_{θ,φ} L̃(θ, φ; X_M, ε)       (backprop through z = μ + σε)
    θ, φ ← update with g  (SGD / Adagrad)
until converged
```

That is all. It is an auto-encoder with noise in the middle and a KL penalty.

### 6.5 The networks (Appendix C)

**Encoder** (Gaussian MLP):

```
h = tanh(W₃x + b₃),   μ = W₄h + b₄,   log σ² = W₅h + b₅
```

**Decoder:**
- **Bernoulli** (MNIST):

  ```
  y = sigmoid(W₂ tanh(W₁z + b₁) + b₂),   log p(x|z) = Σ x log y + (1−x) log(1−y)
  ```

- **Gaussian** (Frey Face): the same as the encoder, with the means squashed into (0, 1) by a sigmoid.

**Why log σ² rather than σ?** It can be any real number, and σ = exp(½ log σ²) is always positive.

**Size (MNIST, 500 hidden units, J = 20 latents):**
- encoder: 784·500 + 500 + 2·(500·20 + 20) = 412,540;
- decoder: 20·500 + 500 + 500·784 + 784 = 403,284;
- total **815,824** (checked in our tests).

---

## 7. The baselines

### 7.1 Wake-sleep (Hinton et al. 1995)
It uses the same two networks, trained with two different objectives:
- **Wake** (trains the decoder): encode a real x, sample z ~ q(z|x), and raise log p(x, z).
- **Sleep** (trains the encoder): "dream" z ~ p(z) and x' ~ p(x|z), then raise log q(z|x'), so the encoder learns to invert the decoder's *dreams*.

**Problem:** the two losses don't add up to one objective (no bound is being optimised). The sleep phase also trains the encoder on fake data, not real data.

**Advantage:** it works with discrete latents.

### 7.2 Monte Carlo EM (Appendix E)
- No encoder.
- For each datapoint, sample z from the true posterior with **Hamiltonian (Hybrid) Monte Carlo (HMC)**, which uses ∇_z log p(x, z), then update the decoder on those samples.
- It is accurate but slow (an inner sampling loop per datapoint), and not online, so it can't handle all of MNIST.

### 7.3 HMC in one paragraph
To sample from a density π(z):
1. give z a random momentum p ~ N(0, I);
2. simulate a frictionless ball rolling on the surface −log π(z) using **leapfrog** steps (half-step momentum, full-step position, half-step momentum);
3. accept the end point with probability min(1, e^{−ΔH}), where H = −log π(z) + ½|p|² is the total energy.

- Exact simulation conserves H, so the acceptance rate is near 1.
- The method can travel far without random-walk behaviour.
- Our test checks that HMC on the exact model recovers the posterior mean and variance.

### 7.4 Measuring log p(x) itself (Appendix D)
The bound isn't the likelihood. To compare methods fairly, the paper estimates log p(x) with a harmonic-mean-style trick:
1. take HMC samples z_l from p(z|x);
2. fit a simple density q(z) to them;
3. use fresh posterior samples in

   ```
   1/p(x) ≈ (1/L) Σ_l  q(z_l) / ( p(z_l) p(x|z_l) )
   ```

**Why it works:**
- E_{p(z|x)}[ q(z)/p(x, z) ] = ∫ p(z|x) q(z)/p(x,z) dz = ∫ q(z)/p(x) dz = 1/p(x).
- This uses p(z|x)/p(x,z) = 1/p(x).

It is reliable only in low dimensions (the paper used Nz = 3). Our test shows it recovering the exact log p(x) of the linear-Gaussian model to within 0.05 nats.

---

## 8. Results from the paper

### 8.1 Figure 2: lower bound vs training samples seen
- **Setup:** MNIST with Nz = 3, 5, 10, 20, 200 (500 hidden units) and Frey Face with Nz = 2, 5, 10, 20 (200 hidden units).
- **Findings:**
  - **AEVB converged considerably faster and reached a better bound in all experiments.**
  - **More latents did not overfit.** Even Nz = 200 has no train/test gap, "explained by the regularizing effect of the lower bound".
  - Training cost was 20–40 minutes per million samples on a 2013 CPU.

### 8.2 Figure 3: estimated marginal likelihood (Nz = 3, 100 hidden units)

| | N_train = 1000 | N_train = 50000 |
|---|---|---|
| AEVB | best | best, and fastest |
| MCEM | competitive (slowly) | can't run on full data (not online) |
| Wake-sleep | behind | behind |

### 8.3 Figures 4–5
- **Figure 4:** with 2-D latents, decoding a grid of z values (spaced by the Gaussian inverse CDF) shows a smooth "map":
  - MNIST digits morph into each other;
  - Frey faces change pose and expression continuously.
- **Figure 5:** random samples from models with Nz = 2, 5, 10, 20 are shown side by side (no numbers, only pictures).

---

## 9. Why it works (the intuition)

1. **Amortised inference.** One encoder network does inference for every datapoint in a single forward pass, instead of running an optimisation or a Markov chain per image. That is the "auto-encoding" in the title.
2. **One objective for both networks.** The ELBO is a true lower bound, so every gradient step on it either improves the model or tightens the bound.
3. **Low-variance gradients.** The reparameterization uses the derivative of the decoder's log-likelihood, not just its value, so a single sample per image works.
4. **The KL acts as a regularizer with no knob to tune.** Each latent dimension must "pay" in KL for the information it carries. Unneeded dimensions can stay at the prior for free, so extra latents don't cause overfitting.
5. **Noise in the code makes the decoder smooth.** Nearby codes must decode to similar images, which gives the smooth manifolds of Figure 4.

---

## 10. What our code found

All numbers come from `demo.py` (about 2.5 s) and `test_vae.py` (9 tests, about 1 s).

1. **Eq. 1 holds exactly** on the linear-Gaussian model. Bound + KL = −1.8412 = log p(x) for four different q's, and the bound is tight only at the true posterior (table in Section 3.3).
2. **Reparameterization vs score function**, for the gradient of E[z²]:
   - both unbiased (means 1.990 / 1.991 against the true 2);
   - variance **3.99 vs 30.35** for μ and **11.9 vs 143.2** for log σ;
   - that is 8× and 12× less noise.
3. **Estimator B vs A:** the same mean (−10.610 vs −10.616), with standard deviation **0.749 vs 3.187**.
4. **AEVB vs wake-sleep**:
   - setup: 5000 binarized MNIST digits, Nz = 2, 200 hidden units, Adagrad at 0.1, 8 epochs each;

   | | train bound | test bound | importance-sampled log p(x), test |
   |---|---|---|---|
   | AEVB | −187.8 | −186.8 | −178.7 |
   | wake-sleep | −201.6 | −198.2 | −189.1 |

   - AEVB is ahead by about 11–14 nats after the same number of updates, the direction of Figure 2.
   - Its 2-D codes group by digit **without labels**: class centres sit 2.18 apart, while points lie 1.09 from their own centre.
5. **More latents**:

   | | train bound | test bound | train − test gap |
   |---|---|---|---|
   | Nz = 2 | −187.9 | −186.7 | −1.2 nats |
   | Nz = 20 | −166.0 | −168.9 | 2.9 nats |

   - Ten times more latents improve the bound by about 20 nats with only a tiny train/test gap: no overfitting, as the paper says.
   - **Honest caveat:** the KL per dimension is very uneven (4.65 down to 0.13 nats), but **none of the 20 dimensions switched off completely** on this small slice. Fully "inactive" latent units are reported for long training on full data (for example Burda et al. 2016). We did not reproduce that here.
6. **HMC and Appendix D:**
   - HMC recovered the exact posterior mean and variance of the linear-Gaussian model;
   - the Appendix D estimator matched the exact log p(x) within 0.05 nats;
   - Monte Carlo estimators A and B agree to within 0.05 nats;
   - the importance-sampled log-likelihood sits above the bound, as it must.

**Not run (too heavy):**
- `experiments.py` covers all of Figures 2–5 and an estimator-variance study (E6).
- The paper trains up to 10⁸ samples per curve.

---

## 11. Check yourself

1. Why is L(x) ≤ log p(x)? When is it equal?
   <details><summary>Answer</summary>log p(x) − L(x) = KL(q(z|x) ‖ p(z|x)) ≥ 0. Equality holds exactly when q(z|x) equals the true posterior.</details>
2. Compute KL(N(μ = 1, σ² = 1) ‖ N(0, 1)).
   <details><summary>Answer</summary>−½(1 + log 1 − 1 − 1) = −½(−1) = 0.5 nats.</details>
3. With z = μ + σε, what is ∂z/∂μ and ∂z/∂σ for a given ε = −0.7?
   <details><summary>Answer</summary>∂z/∂μ = 1 and ∂z/∂σ = ε = −0.7.</details>
4. Why can't we just backprop through `z = torch.normal(mu, sigma)`?
   <details><summary>Answer</summary>The sampling operation has no derivative with respect to the distribution's parameters. The randomness depends on μ, σ. Reparameterizing moves the randomness into ε, which doesn't depend on them, so z becomes a differentiable function.</details>
5. If the KL term were removed from the VAE objective, what would the encoder learn to do, and why couldn't we sample new images?
   <details><summary>Answer</summary>It would shrink σ toward 0 and spread the μ's anywhere (a plain auto-encoder). Codes wouldn't fill N(0, I), so a random z from the prior would land where the decoder was never trained, giving garbage.</details>
6. A decoder outputs y = (0.8, 0.1, 0.5) for the binary pixels x = (1, 0, 1). What is log p(x|z)?
   <details><summary>Answer</summary>log 0.8 + log 0.9 + log 0.5 = −0.223 − 0.105 − 0.693 = −1.021 nats.</details>
7. Why does wake-sleep not optimise a bound on log p(x)?
   <details><summary>Answer</summary>The sleep phase minimises KL(p(z|x) ‖ q(z|x)) on dreamed data (the reverse direction, on the wrong data), while the wake phase uses the ELBO's decoder part. The two updates come from different objectives, so there is no single quantity guaranteed to improve.</details>
