# Generative Adversarial Nets (GANs), explained simply

**Paper:** Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville & Yoshua Bengio, *Generative Adversarial Nets*, NIPS 2014.

**In one sentence:** train two networks against each other:
- a **generator** G that turns random noise into fake data;
- a **discriminator** D that tries to tell fake from real.

G improves by fooling D. In the ideal limit, G's fakes become indistinguishable from real data, and D can only guess 50/50.

---

## 1. The big idea

### 1.1 Why a new approach?
In 2014, generative models (Boltzmann machines, deep belief nets) needed:
- **Markov chains** (slow, mixing problems);
- or **intractable normalising constants**.

VAEs (paper 039) were brand new and need a likelihood p(x|z) plus a variational bound.

GANs need **neither a likelihood nor a Markov chain**: only a network that maps noise to samples, trained by backpropagation.

### 1.2 The counterfeiter and the police
- **G**, the counterfeiter, prints fake money: x = G(z), with z random noise (uniform or Gaussian).
- **D**, the police, looks at a bill and outputs D(x) = the probability it is **real**.
- D is trained to be right on both real and fake bills.
- G is trained to make D wrong.

Each improvement on one side forces the other to improve. If both had unlimited capacity, the game ends when the fakes are perfect.

---

## 2. The objective (Eq. 1)

```
min_G max_D V(D, G) = E_{x~p_data}[ log D(x) ] + E_{z~p_z}[ log(1 − D(G(z))) ]
```

**Symbols:**

| symbol | meaning |
|---|---|
| p_data | the real data distribution |
| p_z | the noise distribution (here uniform on [−1, 1]) |
| G(z) | a generated sample |
| p_g | the distribution of G(z) when z ~ p_z (G defines it **implicitly**: we can sample but never evaluate it) |
| D(x) ∈ (0, 1) | the discriminator's belief that x is real |

**Reading the objective:**
- **For D**, this is just the log-likelihood of a binary classifier (label 1 for real, 0 for fake). It is the usual cross-entropy, with the sign flipped.
- **G** only appears in the second term. G wants D(G(z)) → 1, which makes log(1 − D(G(z))) → −∞.

**Tiny example:** D says 0.8 on a real image and 0.3 on a fake.
- The contribution to V is log 0.8 + log(1 − 0.3) = −0.223 − 0.357 = **−0.580**.
- **D wants V higher.** A perfect D (1 on real, 0 on fake) gets 0.
- **G wants V lower.** Fooling D drives log(1 − D) toward −∞.

---

## 3. The theory, step by step

### 3.1 Proposition 1: the best discriminator for a fixed G (Eq. 2)

**Step 1.** Rewrite V as one integral over x. This uses E_z[f(G(z))] = E_{x~p_g}[f(x)]:

```
V = ∫ [ p_data(x) log D(x) + p_g(x) log(1 − D(x)) ] dx          (Eq. 3)
```

**Step 2.** Maximise pointwise: for each x separately, pick the y = D(x) that maximises a·log y + b·log(1 − y), with a = p_data(x) and b = p_g(x).
- Set the derivative to zero: a/y − b/(1 − y) = 0, so a(1 − y) = b·y, so **y = a/(a + b)**.

```
D*(x) = p_data(x) / (p_data(x) + p_g(x))
```

**Numbers:** at some x, p_data = 0.3 and p_g = 0.1, so D* = 0.3/0.4 = **0.75**. "Three times more likely to be real than fake", so 75%.

**Our test** checks this on a fine grid of y and on random D's: none beats D*.

### 3.2 Theorem 1: what G is really minimising

**Step 1.** Plug D* back in:

```
C(G) = max_D V = E_{p_data}[ log p_data/(p_data + p_g) ] + E_{p_g}[ log p_g/(p_data + p_g) ]
```

**Step 2.** Write p_data + p_g = 2 · m, where m = (p_data + p_g)/2 is the average distribution:

```
log p_data/(p_data + p_g) = log p_data/m − log 2      (and the same for p_g)
```

**Step 3.** So

```
C(G) = −log 4 + KL(p_data ‖ m) + KL(p_g ‖ m) = −log 4 + 2 · JSD(p_data ‖ p_g)     (Eqs. 5–6)
```

**JSD** is the Jensen–Shannon divergence:
- it is symmetric;
- 0 ≤ JSD ≤ log 2;
- it is 0 only when the two distributions are equal.

**Conclusion:**
- C(G) ≥ −log 4 = −1.386, with equality **if and only if p_g = p_data**.
- At that point D* = ½ everywhere.

**Our demo** (data N(0, 1), generator N(μ, 1)):

| p_g | D*(0) | C(G) | −log 4 + 2 JSD |
|---|---|---|---|
| N(3, 1) | 0.989 | −0.3328 | −0.3327 |
| N(1, 1) | 0.622 | −1.1635 | −1.1634 |
| N(0.3, 1) | 0.511 | −1.3641 | −1.3640 |
| N(0, 1) | 0.500 | **−1.3864** | −1.3863 |

The last digit differs only because of the integration grid.

### 3.3 Proposition 2: convergence (in function space)
- Suppose D is trained to its optimum at every step, and G takes small steps on p_g directly (not on network weights).
- Then sup_D V is **convex in p_g**, so gradient descent reaches the unique optimum p_g = p_data.
- **Caveat:** real G's are networks with parameters θ_g, which isn't convex. The paper admits the guarantee doesn't carry over: "the excellent performance of multilayer perceptrons in practice suggests that they are a reasonable model to use despite their lack of theoretical guarantees."

---

## 4. Algorithm 1 (training in practice)

```
for each iteration:
    repeat k times:
        sample m noise vectors z and m real examples x
        update D by ASCENDING  (1/m) Σ [ log D(x_i) + log(1 − D(G(z_i))) ]
    sample m noise vectors z
    update G by DESCENDING (1/m) Σ log(1 − D(G(z_i)))
```

- The paper uses **k = 1** and SGD with momentum.
- D isn't trained to optimality each time. That would be too expensive and would overfit; it just stays "near" its optimum while G moves slowly.

### 4.1 The saturation problem and the non-saturating fix (Section 3)
- **Early in training**, G is bad, so D rejects fakes confidently: D(G(z)) ≈ 0.
- Write D = sigmoid(l), where l is the logit. Then:

  | loss | gradient with respect to l |
  |---|---|
  | minimax: log(1 − D) | **−D ≈ 0** (vanishes!) |
  | non-saturating: −log D | **−(1 − D) ≈ −1** (strong) |

- So the paper suggests training G to **maximise log D(G(z))** instead. It has the same fixed point but much stronger early gradients.
- **Numbers** (our test): at D = 0.01, the gradients are −0.01 vs −0.99, a **99×** difference.
- **Our demo** (G starts far away, D pre-trained): the generator's gradient norm is **6.4·10⁻⁴** vs **2.8**, about 4,400× larger.

### 4.2 The "Helvetica scenario" (mode collapse, Section 6)
- If G is trained too much without updating D, G finds the single x that the current D likes most and maps **every z** there. Many noise values map to one output.
- That is why D must be kept "synchronized" with G.
- **Our demo:**
  - balanced training on a ring of 8 Gaussians covers 7 of 8 modes;
  - then 100 generator-only steps against the frozen D collapse it to **1 mode** (1859 of 2000 samples in one spot).

---

## 5. The networks (Section 5)

**Generator:**
- noise only at the **bottom** layer;
- ReLU hidden layers and **sigmoid** outputs (pixels in [0, 1]);
- the released MNIST model uses 100 → 1200 → 1200 → 784.

**Discriminator:**
- **maxout** units (each unit outputs the maximum of k linear pieces);
- **dropout**: it regularises D, which otherwise overfits to the finite training set.

**A maxout example:** pieces (2.0, −1.0, 0.5) give output 2.0. It is a learned convex piecewise-linear activation.

---

## 6. Evaluation: Parzen windows (Table 1)

GANs give no p_g(x). To get *some* likelihood number, the paper:
1. draws many samples s_j from G;
2. puts a Gaussian bump of width σ on each, giving the density p̂(x) = (1/n) Σ_j N(x; s_j, σ²I);
3. picks σ to maximise the likelihood of a **validation** set;
4. reports the mean log p̂ on the **test** set.

```
log p̂(x) = log Σ_j exp(−‖x − s_j‖² / 2σ²) − log n − d·log(σ√(2π))
```

**Worked example (2-D):**
- samples at (0, 0) and (2, 0); x = (1, 0); σ = 0.7.
- Both squared distances are 1, so each bump gives (1/(2π·0.49)) e^{−1/0.98} = 0.3248 · 0.3604 = 0.1171.
- The average is 0.1171, and log 0.1171 = **−2.145** (our test checks this).

**Table 1** (log-likelihood; higher is better):

| Model | MNIST | TFD |
|---|---|---|
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1.6 | 2110 ± 50 |
| Deep GSN | 214 ± 1.1 | 1890 ± 29 |
| **Adversarial nets** | **225 ± 2** | 2057 ± 26 |

The values are positive because these are *densities* of real-valued pixels (they can exceed 1).

**The paper's own warning:** the estimator "has somewhat high variance and does not perform well in high dimensional spaces". Later work showed Parzen scores can be badly misleading.

**Our demo shows how:**
- on 2-D data, a too-narrow generator (std 0.3 instead of 1) scores **−2.820**, about the same as perfect samples (−2.842; exact −2.814);
- the reason: cross-validation widens σ to 1.0, which covers up the missing spread.

---

## 7. Advantages and disadvantages (Section 6, Table 2)

**Advantages:**
- no Markov chains, for training or for sampling;
- no inference during learning;
- only backprop;
- any differentiable G;
- G never sees data directly, only gradients through D;
- it can represent very **sharp**, even degenerate, distributions. Markov-chain methods need blur to mix.

**Disadvantages:**
- no explicit p_g(x), so likelihood is hard to evaluate;
- D and G must be kept in balance (the Helvetica scenario);
- training is a **game**, not a minimisation. Simultaneous gradient steps can oscillate instead of converging.
  - Our 1-D demo shows the spread swinging between 0.56 and 0.31 around the target 0.5.

---

## 8. Why it works (the intuition)

1. **D is a learned loss function.** Instead of hand-designing a likelihood or a pixel distance (which produce blur), G is judged by a network that learns *what makes data look real*.
2. **D's gradient points G toward the data.** ∇_x D(x) says how to change a sample to look more real. G follows it (Figure 1c).
3. **The optimal game has the right answer.** For the best D, G is minimising the JSD to the data, which is zero only at p_g = p_data.
4. **Sampling is one forward pass.** There is no chain to mix, and samples are independent.

---

## 9. What our code found

All numbers come from `demo.py` (about 5 s) and `test_gan.py` (7 tests, about 2.4 s).

1. **Theory, exactly:**
   - D* = a/(a + b) maximises a log y + b log(1 − y), and no random D beats it;
   - C(G) = −log 4 + 2 JSD on 20 random discrete distributions;
   - the minimum −log 4 is reached only at p_g = p_data, where D* = ½;
   - disjoint distributions give JSD = log 2.
2. **Figure 1 in 1-D:**
   - G moves from mean −3 onto N(3, 0.5²) (mean 2.97–2.94);
   - D(x) on the data drifts to about 0.44–0.57;
   - **honest caveat:** the spread oscillates (0.56 → 0.31) instead of settling. Also, we used Adam (β₁ = 0.5) where the paper used momentum SGD.
3. **Saturation:** with D(G(z)) ≈ 0.0002, the minimax generator gradient is 6.4·10⁻⁴ against 2.8 for the non-saturating loss.
4. **The Helvetica scenario:** coverage goes from 7/8 modes to 1/8 after 100 generator-only steps.
5. **Parzen windows:**
   - correct on a hand-computed case;
   - σ by cross-validation gives −2.842 against the exact −2.814;
   - but a wrong (too-narrow) generator scores −2.820, just as well.

**Not run (too heavy):**
- E1: MNIST with the paper's maxout/dropout nets and the Parzen score (paper: 225 ± 2);
- E2: saturating vs non-saturating losses;
- E3: k = 1 vs 5;
- E4: CIFAR-10 samples from FC and convolutional models;
- E5: nearest-neighbour and interpolation figures.
- TFD is skipped because it isn't freely available.

---

## 10. Check yourself

1. p_data(x) = 0.2 and p_g(x) = 0.6. What does the optimal D output?
   <details><summary>Answer</summary>0.2/(0.2 + 0.6) = 0.25.</details>
2. What is C(G) when p_g = p_data, and why?
   <details><summary>Answer</summary>D* = ½ everywhere, so C = log ½ + log ½ = −log 4 ≈ −1.386.</details>
3. Two distributions with no overlap at all. What are JSD and C(G)?
   <details><summary>Answer</summary>JSD = log 2 (the maximum), so C(G) = −log 4 + 2 log 2 = 0. D can be perfect.</details>
4. D(G(z)) = 0.05. Compare the gradients (with respect to the logit) of log(1 − D) and −log D.
   <details><summary>Answer</summary>−0.05 vs −0.95: a 19× stronger signal for the non-saturating loss.</details>
5. Why does the generator collapse if D is frozen?
   <details><summary>Answer</summary>G simply maximises a fixed function D(G(z)). The best way is to send every z to the single point D rates highest, so nothing rewards diversity unless D adapts and starts rejecting that point.</details>
6. Why can a Parzen-window score be misleading?
   <details><summary>Answer</summary>It measures the samples smoothed by a tuned σ. A tuned blur can hide missing diversity, and in high dimensions the estimate is very noisy and favours blur.</details>
