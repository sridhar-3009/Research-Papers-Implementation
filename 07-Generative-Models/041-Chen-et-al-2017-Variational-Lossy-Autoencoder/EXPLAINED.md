# Variational Lossy Autoencoder (VLAE), explained simply

**Paper:** Xi Chen, Diederik P. Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever & Pieter Abbeel, *Variational Lossy Autoencoder*, ICLR 2017.

**In one sentence:** a VAE's code z only stores what its decoder **can't** model on its own. So if you give the decoder a small *local* view of the image (a little autoregressive window), z is forced to keep only the *global* picture. This gives a deliberately **lossy** code, and with a learned autoregressive prior it also gives excellent likelihoods.

---

## 1. The puzzle this paper solves

**What people saw:**
- Researchers tried to combine VAEs with powerful **autoregressive** decoders, which predict each pixel or word from the previous ones (RNNs, PixelCNNs).
- The result was always the same: the decoder **ignored z completely**. The KL term went to 0, and q(z|x) = p(z) for every x.
- This was usually called an "optimization problem", to be fixed with tricks like KL annealing.

**This paper's answer:**
- It isn't (only) an optimization problem. Even with **perfect optimization**, ignoring z is the *right* answer whenever the decoder can model the data alone.
- They explain why with **bits-back coding**, then turn the "bug" into a feature: choose *what* the decoder can see, and you choose *what* z must store.

---

## 2. Background: code lengths and probabilities

- **Shannon:** an event of probability p can be sent in −log₂ p bits (or −ln p nats). Good models give short codes.
- The best possible average code length for data from p_data is its **entropy**: H(data) = E[−log p_data(x)].
- **Example:**
  - a fair coin needs 1 bit = ln 2 = **0.693 nats** per flip;
  - a coin with P(heads) = 0.9 needs H = −0.9 log₂ 0.9 − 0.1 log₂ 0.1 = 0.137 + 0.332 = **0.469 bits**.
- **Training a model by maximum likelihood is the same as minimising code length:** −log p_model(x) is the length of x's code under the model.

---

## 3. Bits-back coding: the VAE as a compressor (Section 2.2)

### 3.1 The naive two-part code (Eq. 5)
To send an image x with a VAE:
1. pick a code z ~ q(z|x) and send it using the prior: −log p(z) nats;
2. send x given z: −log p(x|z) nats.

```
C_naive = E_{z~q}[ −log p(z) − log p(x|z) ]
```

This wastes space: we picked z **randomly** from q, and that random choice is itself information the receiver gets for nothing useful.

### 3.2 Getting bits back (Eqs. 6–7)
- The trick (Hinton & van Camp 1993): use that randomness to carry *other* data.
  - The sender "samples" z by **decoding some other message** with q(z|x) as the code book.
  - The receiver, after recovering x, runs the same encoder q(z|x) and recovers that other message.
- Those bits come back. On average that is the entropy of q, H(q) = E_q[−log q(z|x)] nats.

```
C_bitsback = C_naive − H(q) = E_q[ log q(z|x) − log p(z) − log p(x|z) ] = −L(x)
```

**The VAE's negative ELBO is literally a code length.** Maximising the bound means building a better compressor.

**Our demo** (the trained window-1 toy model, nats per 16-bit sequence):

| | nats |
|---|---|
| naive code | 10.586 |
| refund H(q) | −5.117 |
| bits-back code (= −ELBO) | **5.469** |
| Shannon limit H(data) | 5.400 |

### 3.3 The extra cost of an imperfect posterior (Eqs. 8–11)
From the VAE identity log p(x) = L(x) + KL(q(z|x) ‖ p(z|x)):

```
C_bitsback = E[ −log p(x) + KL(q(z|x) ‖ p(z|x)) ]
           ≥ H(data) + E[ KL(q(z|x) ‖ p(z|x)) ]
```

**Two penalties:**
1. a model that doesn't match the data (−log p(x) ≥ the entropy, on average);
2. an encoder that doesn't match the true posterior: the **KL(q ‖ true posterior)** term.

The second one is the **inefficiency of using z at all**. No real q matches the true posterior exactly, so sending information through z always costs a bit extra.

---

## 4. The information preference property

**The argument:**
1. Some information about x can be modelled **directly by the decoder** p(x|z), without z. Example: an autoregressive decoder predicts pixel i from pixels 1…i−1.
2. Routing that information through z instead costs the extra KL(q ‖ posterior) from Section 3.3.
3. So the optimum puts it in the decoder, and **only information the decoder can't model alone goes into z**.

**Consequences:**
- **Fully autoregressive decoder** (sees all x_{<i}): it can model *any* distribution, so z gets **nothing**. This is the "ignored code" everyone observed, and it is optimal behaviour, not a failure.
- **Factorized decoder** (p(x|z) = Π_i p(x_i|z), sees no other pixels): every correlation between pixels must go through z. This is why ordinary VAEs reconstruct well.
- **Local window decoder** (sees a small neighbourhood): local correlations are modelled by the decoder, and **only long-range (global) structure goes into z**.

### 4.1 Our toy, worked out
**The data:** 16-bit sequences generated like this:
1. flip a fair coin g, the **global** bit;
2. bit 1 is 1 with probability 0.85 if g = 1, else 0.15;
3. each later bit **copies** the previous bit with probability 0.6 (**local** structure); otherwise it is a fresh coin with the same g-dependent bias.

**Exact entropy:** p(x) = ½ Σ_{g=0,1} Π_i p(x_i | x_{i−1}, g), where each factor is

```
p(x_i | x_{i−1}, g) = 0.6·[x_i = x_{i−1}] + 0.4·P(x_i | g)
```

Averaged over samples, the entropy is **H = 5.400 nats per sequence**.

**What z must carry**, for each decoder:

| decoder | what it can model itself | what's left for z | our KL (nats in z) |
|---|---|---|---|
| factorized (sees nothing) | nothing | g **and** the copying pattern | **2.415** |
| window 1 (sees x_{i−1}) | the copying | the global bit g (at most ln 2 = 0.693) | **0.571** |
| full (sees all x_{<i}) | everything (it can infer g from the history) | nothing | **0.270**, still falling |

- The window-1 decoder's z carries about one bit: exactly the global information.
- Its bound is within 0.07 nats of the true entropy.
- The full decoder's KL is still shrinking after our short run of 400 steps. At the optimum it is 0. We report what we measured.

---

## 5. A lossy code by design (Section 3.1)

**The recipe:** use a decoder

```
p_local(x|z) = Π_i p(x_i | z, x_WindowAround(i))
```

that sees only a small window of earlier pixels.
- **Local statistics** (stroke thickness, texture, exact pixel noise) are predictable from the window, so the decoder handles them.
- **Global structure** (which digit, overall shape, object layout) can't be seen through a small window, so it must go into z.

**"Decompression"** = encode x, then sample a new image from p(x|z):
- the global shape comes back;
- the exact local details are **re-invented** (Figure 1a: different binary masks, slightly different strokes).

**The paper's numbers** on statically binarized MNIST:

| model | bits in the code per image |
|---|---|
| VAE (factorized decoder) | 37.3 |
| VLAE | 13.3 nats = **19.2 bits** |

The VLAE stores about half as much: a lossier, more "semantic" code.

### 5.1 Window notation (Section 4.3)
A window "A×B" means pixel i sees:
- the A-wide, B-tall block of pixels directly **above** it;
- the (A−1)/2 pixels to its **left** in its own row.

Our demo draws them (o = the pixel, X = visible):

```
window 4x2            window 7x4
.....XXXX...          ...XXXXXXX..
.....XXXX...          ...XXXXXXX..
.....Xo.....          ...XXXXXXX..
                      ...XXXXXXX..
                      ...XXXo.....
```

- On CIFAR-10 (Figure 3), a **bigger** window lets the decoder model more, so the code keeps less:
  - **4×2:** detailed shapes;
  - **7×4:** only rough shapes.
- Colour is very predictable locally, so it is often dropped from z.
- **Grayscale context:** the decoder sees only the *grayscale* of the window, so colour can no longer be modelled locally. z must store it, and the decompressions keep their colours (Figure 3d).

**The paper's MNIST decoder** is a 6-layer PixelCNN with 3×3 masked convolutions:
- its receptive field grows to a triangle of earlier pixels;
- it has the famous **blind spot** on the upper right, which our demo shows (68 visible pixels).

---

## 6. The autoregressive-flow (AF) prior (Section 3.2)

### 6.1 Definition
Instead of p(z) = N(0, I), let the prior be an autoregressive flow of Gaussian noise ε ~ u = N(0, I):

```
z_i = ε_i · σ_i(z_{<i}) + μ_i(z_{<i})
```

- **Density** (one parallel pass, because z is known):

  ```
  ε_i = (z_i − μ_i(z_{<i})) / σ_i(z_{<i})
  log p(z) = log u(ε) − Σ_i log σ_i
  ```

- **Sampling** is sequential (z₁ first, then z₂, …), but it is only needed to *generate* new images, never during training.

### 6.2 AF prior = IAF posterior (Eqs. 12–14)
Write ε = f⁻¹(z) and plug log p(z) = log u(ε) + log|det dε/dz| into the bound:

```
L = E_q[ log p(x|z) + log u(ε) + log|det dε/dz| − log q(z|x) ]
  = E_q[ log p(x|f(ε)) + log u(ε) − ( log q(z|x) − log|det dε/dz| ) ]
                                     └──── an IAF posterior over ε ────┘
```

**The same number, read two ways:**
- **AF prior over z** with a diagonal posterior;
- **IAF posterior over ε** (paper 040), with a decoder that first applies f.

**The difference:** in the AF reading, the *generator* p(x|f(ε)) is **deeper** (it includes the flow), and the training cost is identical. So AF should be at least as good.

**Table 1 agrees:** AF VAE scores 79.30 nats against 79.88 for the equivalent IAF VAE.

**Our demo** computes both readings on a random flow: **−19.633381 = −19.633381**.

### 6.3 A worked example (1-D, T = 1)
- μ = 0.5 and σ = 2 (constants, since there is no z_{<1} in one dimension).
- For z = 1.5: ε = (1.5 − 0.5)/2 = 0.5.
- log p(z) = log N(0.5; 0, 1) − log 2 = −1.0439 − 0.6931 = **−1.7370**.
- Check: z ~ N(0.5, 4), so log N(1.5; 0.5, 4) = −½ log(2π·4) − 1/8 = −1.6121 − 0.125 = **−1.7371** ✓.

---

## 7. Results from the paper

### Table 1: statically binarized MNIST (test NLL, nats; lower is better)

| Model | NLL |
|---|---|
| Normalizing flows | 85.10 |
| DRAW | < 80.97 |
| Discrete VAE | 81.01 |
| PixelRNN | 79.20 |
| IAF VAE | 79.88 |
| AF VAE | 79.30 |
| **VLAE** | **79.03** |

- Swapping the IAF posterior for an equivalent AF prior helps (0.58 nats).
- Adding a PixelCNN decoder helps more.

### Tables 2–4: one VLAE, hyperparameters tuned only on static MNIST

| dataset | previous best | "Unconditional decoder" (the PixelCNN alone) | **VLAE** |
|---|---|---|---|
| dynamically binarized MNIST | 79.10 (IAF VAE) | 87.55 | **78.53** |
| OMNIGLOT | < 91.00 (Conv DRAW) | 95.02 | 90.98 (89.83 when tuned) |
| Caltech-101 Silhouettes | 88.48 (SpARN) | 89.26 | **77.36** |

The small PixelCNN alone is much worse. The combination (z for global structure, the PixelCNN for local detail) beats both parts.

### Table 5: CIFAR-10 (bits/dim)

| Model | bits/dim |
|---|---|
| PixelCNN | 3.14 |
| Gated PixelCNN | 3.03 |
| PixelRNN | 3.00 |
| PixelCNN++ | 2.92 |
| ResNet VAE with IAF (paper 040) | 3.11 |
| ResNet VLAE | 3.04 |
| **DenseNet VLAE** | **2.95** |

This was the best latent-variable model at the time.

---

## 8. Why it works (the intuition)

1. **A VAE is a compressor**, and z is an expensive channel: every nat in z costs an extra KL(q ‖ posterior). Information takes the cheapest path.
2. **The decoder's view decides the cheapest path.** What it can see locally, it models; what it can't, z must carry. So the **architecture** (the window) decides **what the representation contains**.
3. **Two experts:**
   - z, with a flexible AF prior, models global structure compactly;
   - a small PixelCNN models local texture, which is what autoregressive models are best at.
4. **The AF prior is free expressiveness:** the same training cost as an IAF posterior, with a deeper generative path.

---

## 9. What our code found

All numbers come from `demo.py` (about 11 s) and `test_vlae.py` (9 tests, about 1 s).

1. **Information preference, measured** (Section 4.1 table):
   - z carries 2.415 nats with a factorized decoder, 0.571 with a window-1 decoder (about the global bit) and 0.270 with a full decoder (falling toward 0);
   - the window-1 bound is within 0.07 nats of the true entropy, 5.400.
   - **Honest caveat:** only 400 training steps, so the full decoder hasn't reached KL = 0. `experiments.py --only e5` runs the sweep longer, over windows 0–15 and several seeds.
2. **Bits-back accounting:** naive code 10.586 − refund 5.117 = bits-back code 5.469 = −ELBO (Shannon limit 5.400).
3. **Exact receptive fields:**
   - the A×B windows (4×2, 5×3, 7×4) match the paper's definition pixel for pixel (tested with autograd);
   - the 6-layer 3×3 PixelCNN is causal and local, with a blind spot;
   - the grayscale-context decoder ignores colour changes that keep the gray value.
4. **AF prior:**
   - its density integrates to 1 (by importance sampling);
   - its log-determinant matches autograd;
   - the AF-prior bound equals the IAF-posterior bound exactly (Eqs. 12–14);
   - density evaluation is one parallel pass, while sampling is sequential.

**Not run (too heavy):**
- E1: lossy KL bits, VAE vs VLAE (paper 37.3 vs 19.2) with decompression figures;
- E2: Table 1;
- E3: Tables 2–4 with the unconditional baseline;
- E4: Figure 3's CIFAR-10 windows, including grayscale;
- E5: the full toy sweep.

---

## 10. Check yourself

1. Why does a VAE with a fully autoregressive decoder ignore z, even with perfect optimization?
   <details><summary>Answer</summary>The decoder can represent p_data(x) without z. Sending any information through z costs an extra KL(q ‖ true posterior) > 0 in the bits-back code. So the optimum sets q(z|x) = p(z) and models everything in the decoder.</details>
2. Naive code 12.0 nats, H(q) = 4.5 nats. What is the bits-back code length, and what is the ELBO?
   <details><summary>Answer</summary>12.0 − 4.5 = 7.5 nats, so the ELBO is −7.5.</details>
3. With a window 3×1 decoder on MNIST, will z store stroke thickness? Will it store the digit identity?
   <details><summary>Answer</summary>Thickness: mostly not, since it is visible locally (the decoder sees the stroke above and to the left). Identity: yes, since no 3×1 window can tell a 3 from an 8, so that global information must come from z.</details>
4. How can you force z to store colour on CIFAR-10?
   <details><summary>Answer</summary>Let the decoder see only the grayscale of its window (Figure 3d). Colour then can't be predicted locally, so it must be encoded in z.</details>
5. Show that an AF prior and an IAF posterior give the same bound.
   <details><summary>Answer</summary>Use log p(z) = log u(ε) + log|det dε/dz| with ε = f⁻¹(z), and move the log-det next to log q(z|x). That gives log q(ε|x) for the IAF-transformed posterior, so the terms are identical (Section 6.2).</details>
6. 1-D AF prior with μ = −1, σ = 0.5. What is log p(z) at z = 0?
   <details><summary>Answer</summary>ε = (0 − (−1))/0.5 = 2, and log N(2; 0, 1) = −0.919 − 2 = −2.919. Subtract log 0.5: −2.919 + 0.693 = −2.226.</details>
