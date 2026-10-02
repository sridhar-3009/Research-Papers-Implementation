# InfoGAN, explained simply

**Paper:** Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever & Pieter Abbeel, *InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets*, NIPS 2016.

**In one sentence:** give a GAN's generator a few extra input "knobs" c (for example a 10-way switch and two dials), and reward it when a helper network Q can **read the knob settings back** from the generated image. The generator learns to make each knob control one clear, meaningful property (digit type, slant, width), **without any labels**.

---

## 1. The big idea

### 1.1 The problem
- A normal GAN (paper 042) takes a noise vector z and outputs an image.
- Nothing tells it to use z in an **organised** way.
- Changing z[17] might change the digit's slant *and* its thickness *and* its identity at once. The representation is **entangled**.
- We'd like **disentangled** factors, where each input dimension controls one human-meaningful property.

### 1.2 The fix
1. Split the generator's input into:
   - **z**: incompressible noise (unstructured details);
   - **c** = (c₁, …, c_L): a structured **latent code**, for example c₁ ~ Uniform{0…9} and c₂, c₃ ~ Uniform(−1, 1).
2. Require that **c can be recovered from the generated image**: the **mutual information** I(c; G(z, c)) should be high.

If the generator ignored c, then G(z, c) would carry no information about c, and I = 0. Maximising I forces G to make c **visible** in its output. The easiest visible way to use a 10-way switch on MNIST is to use it for the 10 digit types.

---

## 2. Background: mutual information

### 2.1 Entropy
H(X) = −Σ P(x) log P(x) measures uncertainty.
- A uniform 10-way choice: H = log 10 = **2.303 nats**.
- A fair coin: H = log 2 = 0.693 nats.

### 2.2 Mutual information (Eq. 2)

```
I(X; Y) = H(X) − H(X | Y)
```

This is how much knowing Y reduces your uncertainty about X.
- If X and Y are **independent**: I = 0.
- If Y reveals X **exactly**: H(X|Y) = 0, so I = H(X), the maximum.

**Worked example:** c is a fair coin, and x = c with probability 0.9 (otherwise flipped).
- H(c) = 0.693.
- Seeing x leaves c with a 0.9/0.1 split: H(c|x) = −0.9 ln 0.9 − 0.1 ln 0.1 = 0.095 + 0.230 = 0.325.
- So I = 0.693 − 0.325 = **0.368 nats**.

### 2.3 The InfoGAN objective (Eq. 3)

```
min_G max_D  V_I(D, G) = V(D, G) − λ · I(c; G(z, c))
```

- V is the ordinary GAN game.
- The extra term **rewards G** (G minimises, so −λI pushes I up) for keeping c recoverable.

---

## 3. Making I computable: the variational bound (Section 5)

### 3.1 The problem
- I(c; x) = H(c) − H(c|x) needs the **posterior** P(c|x): which code produced this image?
- That is intractable, because G is a neural network.

### 3.2 The trick: a helper Q(c|x) (Eq. 4)
Introduce a network Q(c|x) that *guesses* the code from the image. For any Q:

```
I(c; x) = H(c) + E_x E_{c~P(c|x)}[ log P(c|x) ]
        = H(c) + E_x[ KL(P(·|x) ‖ Q(·|x)) ] + E_x E_{c~P(c|x)}[ log Q(c|x) ]
        ≥ H(c) + E_x E_{c~P(c|x)}[ log Q(c|x) ]                 (the KL is ≥ 0)
```

**Line by line:**
- **Line 1:** −H(c|x) = E[log P(c|x)].
- **Line 2:** add and subtract log Q. Then E_{P}[log P − log Q] = KL(P ‖ Q).
- **Line 3:** drop the non-negative KL.

This is the same move as the VAE's ELBO (paper 039), applied to a mutual information instead of a likelihood.

### 3.3 Removing the posterior sample (Lemma 5.1)
- The bound still samples c from P(c|x).
- **Lemma 5.1** says: drawing x then c|x gives the same joint as drawing c then x|c. So we can draw **c from its prior, x = G(z, c)**, and score log Q(c|x) for *that same* c:

```
L_I(G, Q) = E_{c~P(c), x~G(z,c)}[ log Q(c|x) ] + H(c)  ≤  I(c; G(z, c))      (Eq. 5)
```

- **What this means in code:** generate an image from a known code, ask Q to predict the code, and use Q's log-likelihood of the true code. That is just the **cross-entropy** for categorical codes, or the **Gaussian log-likelihood** for continuous ones.

**When is the bound tight?**
- When Q equals the true posterior.
- For discrete codes, when L_I reaches H(c): then x reveals c perfectly.

**Our demo** (an exact discrete channel; true I = 0.3190):

| Q | L_I |
|---|---|
| Q = true posterior | **0.3190** |
| Q = 50% blurred toward uniform | 0.2087 |
| Q = uniform | 0.0000 |

### 3.4 The final game (Eq. 6)

```
min_{G,Q} max_D  V(D, G) − λ · L_I(G, Q)
```

- **Q** is trained to *read* codes (it tightens the bound).
- **G** is trained to *write* codes readably (it raises I).
- **D** plays the usual GAN game.
- **Appendix B** calls it a **"sleep-sleep" algorithm**: like wake-sleep's sleep phase (train the recognition network on dreamed samples), but the generator is *also* updated on dreams, to make them decodable.

---

## 4. Implementation details (Section 6, Appendix C)

**Q is cheap:**
- it **shares all convolutional layers with D** and adds one small fully connected head;
- this adds negligible compute.

**Code types:**

| code | Q's output | L_I term |
|---|---|---|
| categorical | softmax | −cross-entropy + log K |
| continuous | a factored **Gaussian** (mean, and std = exp(output)) | log N(c; μ, σ²) + H(Uniform(−1, 1)) = … + log 2 |

**Setting λ:**
- λ = 1 works for discrete codes.
- Continuous codes use smaller λ (0.1 here, 0.05–10 for chairs), because the Gaussian log-density can be large. Its differential entropy is on another scale.

**MNIST networks (Table 1):**
- **G:** 74 inputs (62 noise + 10 one-hot + 2 continuous) → FC 1024 → FC 7×7×128 → upconv 64 → upconv 1.
- **D/Q:** conv 64 → conv 128 → FC 1024 (leaky ReLU 0.1, batch norm). D's head is 1 logit; Q's head is FC 128 → 10 + 2 + 2 outputs.

**Training:** Adam, learning rate 2·10⁻⁴ for D/Q and 10⁻³ for G.

### 4.1 A worked L_I computation
- Batch of 2 with a 4-way code: true codes (1, 3).
- Q's softmax probabilities for the true codes: 0.8 and 0.5.
- E[log Q(c|x)] = (ln 0.8 + ln 0.5)/2 = (−0.223 − 0.693)/2 = −0.458.
- L_I = −0.458 + ln 4 = −0.458 + 1.386 = **0.928 nats**, out of a maximum of 1.386.

---

## 5. Results from the paper

### 5.1 Figure 1: is MI actually maximised?
- **MNIST** with c ~ Cat(10):
  - InfoGAN's L_I climbs to **H(c) ≈ 2.30** within a few hundred iterations, so the bound is tight;
  - a regular GAN with the same (separately trained) Q stays near **0**: nothing forces G to use c.
- L_I "always converges faster than normal GAN objectives", so InfoGAN comes nearly free.

### 5.2 Figure 2: MNIST disentanglement
- **c₁ (10-way)** captures **digit type**. Used as a classifier (matching each category to a digit) it gets **5% error**, with no labels at all.
- **c₂** captures **rotation** (slant).
- **c₃** captures **width**.
- Pushing c₂ and c₃ to ±2 (they were trained on ±1) still gives sensible, natural digits. The generator adjusts stroke details rather than just stretching pixels.
- A regular GAN's c₁ has **no clear meaning** (Figure 2b).

### 5.3 Other datasets
| dataset | codes | what was learned |
|---|---|---|
| **3D faces** | 5 continuous | azimuth (pose), elevation, lighting and **wide vs narrow face**, matching DC-IGN, which needed supervision; wide/narrow was never labelled anywhere |
| **3D chairs** | 4 categorical (20-way) + 1 continuous | rotation, and width interpolating between chair types |
| **SVHN** | 4 ten-way + 4 continuous | lighting, plate context |
| **CelebA** | 10 ten-way | azimuth, glasses, hairstyle, emotion |

---

## 6. Why it works (the intuition)

1. **Information demands visibility.** To make c readable from x, G must let c change x in a *consistent, recognisable* way.
2. **The data shapes what's easiest.** A 10-way code on MNIST is most easily made readable by mapping it to the 10 digit clusters: the data already has 10 natural, well-separated modes. A continuous code is most easily made readable by a smooth, global change (slant, width).
3. **The GAN term keeps it realistic.** G can't just stamp a code pattern onto images, because D would reject them. So the codes must be expressed **through realistic variations** of the data.
4. **The KL gap shrinks automatically.** Q is a strong network sharing D's features, so the bound stays tight and maximising L_I really maximises I.

---

## 7. What our code found

All numbers come from `demo.py` (about 7.5 s) and `test_infogan.py` (8 tests, about 2 s).

1. **The bound on an exact channel:**
   - L_I = I exactly when Q is the posterior (0.3190), smaller for worse Q's (0.2087), and 0 for a clueless Q;
   - Lemma 5.1 holds numerically;
   - a code that is fully revealed gives I = H(c).
2. **Figure 1 in miniature:**
   - **data:** 4 clusters, each a short segment; codes: a 4-way c₁ and one continuous c₂;
   - **categorical L_I:**

     | iteration | 50 | 300 | 1200 | 2400 |
     |---|---|---|---|---|
     | InfoGAN | 0.829 | 1.364 | 1.384 | **1.384** |
     | regular GAN | −0.003 | 0.004 | 0.012 | −0.002 |

     The maximum is H(c) = ln 4 = 1.386.
3. **Disentangling without labels**, after 2400 steps:

   | | c₁ → cluster accuracy | \|corr(c₂, position along segment)\| | points on a cluster |
   |---|---|---|---|
   | InfoGAN | **1.000** | **0.829** | 63% |
   | regular GAN | 0.255 (chance is 0.25) | 0.111 | 97% |

   - **Honest note:** InfoGAN's sample quality fluctuated during training: 63% of points on a cluster at this step, against 97% for the regular GAN.
   - The claim we verify is about the **codes**, not about which model makes better samples.
4. **Exaggerated codes:** moving c₂ from −2 to +2 (trained range ±1) keeps moving the point consistently along its segment's direction (y from −3.33 to −0.67), past the training range.

**Not run (too heavy):**
- E1: MNIST Figure 1 curves;
- E2: Figure 2 grids and c₁ as a classifier (paper: 5% error);
- E3: SVHN;
- E4: CelebA;
- E5: a λ sweep for continuous codes.
- 3D faces and chairs need rendered datasets and are skipped.

---

## 8. Check yourself

1. Why does a normal GAN with extra inputs c not automatically learn meaningful codes?
   <details><summary>Answer</summary>Nothing in the GAN loss requires G to use c at all, or to use it in a consistent way. G can ignore it (P(x|c) = P(x)) or entangle it with z.</details>
2. A 4-way code. Q gives the true code probabilities 0.6 and 0.9 on two samples. What is L_I?
   <details><summary>Answer</summary>(ln 0.6 + ln 0.9)/2 + ln 4 = (−0.511 − 0.105)/2 + 1.386 = −0.308 + 1.386 = 1.078 nats.</details>
3. Why is L_I ≤ I(c; x)?
   <details><summary>Answer</summary>I = H(c) + E[log P(c|x)] = L_I + E[KL(P(c|x) ‖ Q(c|x))], and the KL is ≥ 0.</details>
4. What does Lemma 5.1 let us avoid?
   <details><summary>Answer</summary>Sampling c from the intractable posterior P(c|x). We sample c from its prior, generate x = G(z, c), and evaluate log Q(c|x) on that same pair.</details>
5. Why is λ smaller for continuous codes?
   <details><summary>Answer</summary>The Gaussian log-density (a differential entropy) can be large and unbounded, unlike categorical log-probabilities (≤ 0). A smaller λ keeps the information term on the same scale as the GAN loss.</details>
6. In what sense is InfoGAN a "sleep-sleep" algorithm?
   <details><summary>Answer</summary>Q is trained on generated ("dreamed") samples, as in wake-sleep's sleep phase. But G is also updated using those dreams (to make codes decodable), so both updates happen in the "sleep" phase.</details>
