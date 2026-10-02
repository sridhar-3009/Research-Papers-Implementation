# DALL·E: Zero-Shot Text-to-Image Generation, explained simply

**Paper:** Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen & Ilya Sutskever, *Zero-Shot Text-to-Image Generation*, ICML 2021.

**In one sentence:** turn every image into a short sequence of "visual words" with a discrete VAE, glue the caption's words in front, and train one giant GPT-style transformer to predict the whole sequence. To make a picture from a caption, give it the caption and let it write the image words.

---

## 1. The big idea

### 1.1 The goal
- **Input:** a sentence like "an armchair in the shape of an avocado".
- **Output:** an image of it, including combinations no one ever photographed.
- **Zero-shot** means: no training on the evaluation dataset (MS-COCO) and no task-specific tricks. Just one general model trained on 250 million image–caption pairs from the internet.

### 1.2 Why not just use pixels?
A 256×256 colour image is **196,608 numbers**. As a sequence for a transformer:
1. **Memory:** attention over 196k positions is far too expensive.
2. **Wasted capacity:** likelihood models spend most of their effort on tiny high-frequency details (PixelCNN++ paper 045, Section 3.3), not on the shapes that make an object recognisable.

### 1.3 The two-stage answer
| stage | what it learns | sizes |
|---|---|---|
| **1. dVAE** (discrete VAE) | compress each 256×256 image into a **32×32 grid of tokens**, each one of K = 8192 codebook entries | 192× fewer positions |
| **2. Transformer** | model the joint sequence [≤ 256 caption BPE tokens ; 1024 image tokens] autoregressively | 12 billion parameters |

The whole thing is one big **evidence lower bound** (Eq. 1):

```
ln p(x, y) ≥ E_{z ~ q(z|x)} [ ln p_θ(x | y, z)  −  β · KL( q(y, z | x) ‖ p_ψ(y, z) ) ]
```

| symbol | meaning |
|---|---|
| x | the image |
| y | the caption |
| z | the 32×32 image tokens |
| q_φ | the dVAE encoder |
| p_θ | the dVAE decoder |
| p_ψ | the transformer |

- **Stage 1** maximises this with p_ψ fixed to a **uniform** prior.
- **Stage 2** freezes the dVAE and maximises it with respect to the transformer. The KL term then becomes the transformer's cross-entropy on the token sequence.
- The bound only holds for β = 1. The paper uses **β = 6.6** in stage 1 (see Section 3.4).

---

## 2. Background you need

| idea | where it comes from |
|---|---|
| VAEs and the ELBO | paper 039 |
| autoregressive transformers, cross-entropy on next-token prediction | papers 034–036 |
| sparse attention patterns | paper 036 (Sparse Transformer) |
| BPE text tokens | papers 033–034 |

The new ingredients are a **discrete** latent (tokens instead of real numbers) and **scale**.

---

## 3. Stage 1: the discrete VAE (Section 2.1, Appendix A)

### 3.1 The architecture
**Encoder:** ResNet, with
- a 7×7 first conv;
- max-pooling to downsample by 8;
- a final **1×1 conv** that outputs 8192 logits at each of the 32×32 positions.

So q(z|x) is a **categorical distribution per position**.

**Decoder:**
- takes the 32×32 grid of (one-hot or relaxed) codes;
- a **1×1 conv** in, nearest-neighbour upsampling, a 1×1 conv out;
- outputs 6 maps: μ and ln b for each colour channel.

**Small details that mattered:**
- 1×1 convs around the bottleneck, so the relaxation generalises to the real discrete codes;
- the residual branches are multiplied by a small constant for stable initialisation.

### 3.2 The problem: you can't backprop through "pick a token"
- The reparameterization trick (paper 039) needs continuous noise.
- Choosing one of 8192 categories is discrete, so the gradient is zero almost everywhere.

### 3.3 The fix: the Gumbel-softmax relaxation
**The Gumbel-max trick:** sampling k ~ Categorical(softmax(ℓ)) is the same as

```
k = argmax_i (ℓ_i + g_i),   g_i = −log(−log U_i),  U_i ~ Uniform(0, 1)
```

**The relaxation:** replace argmax with a softmax at temperature τ:

```
y_i = exp((ℓ_i + g_i)/τ) / Σ_j exp((ℓ_j + g_j)/τ)
```

- Large τ gives a smooth blend; τ → 0 gives exactly one-hot.
- The decoder receives y, a soft one-hot vector, so gradients flow back into the logits.
- **Annealing:** τ goes from 1 to 1/16 over the first 150k updates, on a cosine schedule. A linear schedule often diverged.

**Worked example:** logits ℓ = (1, 0, −1), Gumbel draws g = (0.2, 1.5, −0.3).
- Sums: (1.2, 1.5, −1.3).
- At τ = 1: y = softmax(1.2, 1.5, −1.3) = (0.41, 0.56, 0.03).
- At τ = 1/16: y ≈ (0.01, 0.99, 0.00), essentially the one-hot of the argmax.

**Our test** checks that the cold samples follow softmax(ℓ): token frequencies match (0.665, 0.245, 0.090) within 0.02.

### 3.4 The KL to a uniform prior, and β
With the uniform prior over K codes:

```
KL(q ‖ Uniform) = Σ_k q_k log q_k + log K        (per position)
```

- 0 if q is uniform; log K = 9.01 nats if q is one-hot (K = 8192).
- The paper divides the loss by 256·256·3, so the KL weight per pixel value is effectively β/192.

**β = 6.6, larger than the "correct" 1, works better.** In the paper's words, larger β "promotes better codebook usage and ultimately leads to a smaller reconstruction error". Their guess: with small β, the relaxation noise early in training makes the optimiser shrink codebook usage, which then never recovers.

**Our demo** (16 codes, 5 colours):

| β | codebook perplexity | outcome |
|---|---|---|
| 0 | 1.0–1.65 | collapse to about one code |
| 6.6 | 4.4–5.0 | several codes alive, close to the 5 colours |

### 3.5 The logit-Laplace likelihood (Appendix A.3)
**The problem:** pixels live in a bounded range, but Gaussian (ℓ2) or Laplace (ℓ1) reconstruction losses put probability on impossible values (below 0, above 255).

**The fix:** model the pixel as **sigmoid of a Laplace variable**:

```
f(x | μ, b) = 1 / (2b · x(1 − x)) · exp(−|logit(x) − μ| / b),    x ∈ (0, 1)        (Eq. 2)
```

**Where it comes from (change of variables, as in paper 046):**
- If L ~ Laplace(μ, b) and x = sigmoid(L), then L = logit(x) and dL/dx = 1/(x(1 − x)).
- So f(x) = Laplace(logit(x); μ, b) · 1/(x(1 − x)).

**Pixel mapping (Eq. 3):** pixels are mapped into (0.1, 0.9) first: ϕ(x) = 0.8·x/255 + 0.1.
- This keeps 1/(x(1 − x)) finite.
- Example: pixel 0 → 0.1, 255 → 0.9, 127.5 → 0.5.

**Reconstructing an image:** x̂ = ϕ⁻¹(sigmoid(μ)).

**Our demo:**
- a Laplace centred at 0.95 (scale 0.1) puts **30%** of its mass outside [0, 1];
- the logit-Laplace integrates to exactly 1 inside.

---

## 4. Stage 2: the transformer prior (Section 2.2, Appendix B)

### 4.1 One stream of tokens

```
[ t₁ t₂ … t_len  pad … pad | z₁ z₂ … z₁₀₂₄ ]
   up to 256 caption tokens    32×32 image tokens (raster order)
```

- The caption is lowercased and BPE-encoded (vocabulary 16,384, at most 256 tokens).
- **Image tokens** come from the dVAE encoder by **argmax** (no Gumbel noise).
- **Padding:** if the caption is shorter than 256 tokens, each unused position gets its **own learned padding token**.
  - Compared with simply masking padding out, this gave a *higher* validation loss but *better* behaviour on out-of-distribution captions.
- **Image positions** get a row embedding plus a column embedding (Figure 10).

### 4.2 Attention masks
- **Text → text:** ordinary causal.
- **Image → text:** every image token sees the **whole caption**.
- **Image → image:** sparse patterns, in the spirit of paper 036:

| mask | an image token attends to | used in |
|---|---|---|
| **row** | the previous W tokens in raster order, and itself (so up to the same column of the previous row) | most layers |
| **column** | the same column in all earlier rows | every layer with (i − 2) mod 4 = 0 |
| **convolutional** | a causal 11×11 neighbourhood | the last layer only |

- The schedule is "row, column, row, row, row, column, …", with conv last.
- **Example** (4×4 grid, token 9 = row 2, column 1):
  - row mask: tokens 5–9;
  - column mask: tokens 1, 5, 9;
  - 3×3 conv mask: tokens 4, 5, 6, 8, 9.

  Our test checks exactly these.

**The size:** 64 layers, 62 heads of size 64, d_model = 3968. Using ~12·d² per layer (attention 4d², MLP 8d²):

```
12 × 3968² × 64 = 12.09 billion parameters
```

### 4.3 The loss
- Cross-entropy for next-token prediction.
- **Weighted:** text × 1/8 + image × 7/8. Each is averaged over its own token count, because the image is what matters.
- The paper reserved 606k images for validation and saw **no overfitting**.

---

## 5. Making 12 billion parameters trainable (Sections 2.4–2.5, Appendices D–E)

### 5.1 16-bit underflow and per-resblock gradient scaling
- To save memory, activations and gradients are stored in 16-bit floats. fp16 has only **5 exponent bits**: the smallest positive value is about 6·10⁻⁸ and the largest is 65,504.
- **The problem:** gradient magnitudes shrink steadily from early to late resblocks. They spanned too many orders of magnitude for any single "loss scale" to keep all of them representable.
- **The fix (Figure 4):** a **separate gradient scale per resblock**. Each block's incoming gradient is scaled up before its backward pass and scaled down afterwards; Inf and NaN values are filtered out.
- **Our demo:** with gradient sizes 10¹ … 10⁻¹¹ across 5 blocks, a global scale of 2¹⁰ (the largest that avoids overflow in block 1) loses **99.6%** of block 5's gradient to underflow. Per-block scales lose **0%**.

### 5.2 PowerSGD gradient compression
- Averaging gradients across machines was the bottleneck.
- **PowerSGD** (Vogels et al. 2019) sends a **rank-r approximation** of each gradient matrix M:

  ```
  P = orthonormalise(M Q),   send P (n×r) and Q' = Mᵀ P (m×r),   M̂ = P Q'ᵀ
  ```

- **Error feedback:** the part not sent (M − M̂) is **added to the next step's gradient**, so nothing is permanently lost.
  - Summed over steps, the total sent = total true gradient − (only the final residual). Our test checks this.
- **Compression rate:**

  ```
  1 − 5r / (8 d_model)
  ```

  **Table 1:**

  | d_model | r | rate |
  |---|---|---|
  | 1920 | 512 | 83% |
  | 2688 | 640 | 85% |
  | 3968 | 896 | 86% |

  Our test reproduces all three.
- **A surprise:** keeping Q **fixed** (a random Gaussian, never updated) worked as well as the original warm-started Q.

### 5.3 Parameter sharding
- The 12B model needs 24 GB in fp16, more than a 16 GB V100.
- So each GPU on a machine holds 1/8 of the parameters, and they are fetched (all-gather) just in time for each resblock.

---

## 6. Generating images (Section 2.6)

1. Feed the caption.
2. Sample the 1024 image tokens one by one.
3. Decode them with the dVAE.

**Reranking:** draw **N = 512** samples per caption and keep the best ones under a **contrastive model** (CLIP), which scores how well an image matches the caption.
- Figure 9(c): quality improves with N, with diminishing returns after about 32.
- **Our demo** (oracle scorer, a held-out caption the model gets right 31% of the time):

  | best of N | agreement with the caption |
  |---|---|
  | 1 | 83% |
  | 4 | 94% |
  | 16 | 100% |

---

## 7. Results

### MS-COCO, zero-shot (Section 3.1)
- **Human evaluation vs DF-GAN** (best-of-five vote):

  | question | DALL·E chosen |
  |---|---|
  | which matches the caption better? | **93.3%** |
  | which is more realistic? | **90.0%** |

- **FID:** within 2 points of the best prior model, **without ever training on COCO captions**.
- **FID with a slight blur** (radius 1, to set aside high-frequency detail lost by the dVAE): the best by about 6 points.
- **Data overlap:** about 21% of COCO validation images appear in the training set (but not their captions). Removing them didn't change the results.

### CUB birds
- Much worse: a nearly 40-point FID gap to the best specialised model.
- A zero-shot general model is weaker on narrow, specialised distributions. Fine-tuning is left to future work.

### Qualitative findings (Section 3.3)
- **Combining unusual concepts:** "a tapir made of accordion" (Figure 2a).
- **Rendering text in images.**
- **Variable binding**, though **inconsistent**: "a baby hedgehog in a christmas sweater walking a dog". Sometimes both animals get sweaters.
- **Zero-shot image-to-image translation:** given "the exact same cat on the top as a sketch at the bottom" plus the top half of a cat photo's tokens, it draws a sketch of the cat below.

---

## 8. Why it works (the intuition)

1. **Discrete tokens make images look like text.** Once an image is a sentence of visual words, everything that made GPT work (next-token prediction, scale, attention) applies.
2. **The dVAE spends bits on what matters.** 8× downsampling throws away fine texture, so the transformer's capacity goes to shapes and layout, the part people care about.
3. **One stream means one model for everything.** Captions condition images through ordinary attention. Image completion and image-to-image translation fall out for free: just provide part of the stream.
4. **Scale gives generalisation:** 250M pairs and 12B parameters let it combine concepts it never saw together, at least some of the time.

---

## 9. What our code found

All numbers come from `demo.py` (about 21 s) and `test_dalle.py` (9 tests, about 0.5 s).

1. **The building blocks are exact:**
   - the logit-Laplace integrates to 1 and equals the sigmoid-of-Laplace density;
   - ϕ maps 0 / 127.5 / 255 to 0.1 / 0.5 / 0.9;
   - KL to uniform is 0 for uniform q and log K for one-hot q;
   - cold Gumbel-softmax samples follow softmax(ℓ);
   - the masks match Figure 11 (row 5–9, column 1/5/9, conv 4/5/6/8/9 for token 9 of a 4×4 grid);
   - the transformer never sees its target, and ignores caption tokens past the length (padding);
   - the loss is 1/8·text + 7/8·image.
2. **Paper arithmetic:**
   - 12.3B parameters from d = 3968 and 64 layers (≈ 12 · d² · 64 plus embeddings);
   - 192× compression;
   - PowerSGD rates of 83 / 85 / 86%, matching Table 1.
3. **dVAE and β:** β = 0 collapsed the codebook (perplexity 1.0–1.65), while β = 6.6 kept 4.4–5.0 codes in use, matching footnote 4.
   - **Honest note:** reconstruction quality varied a lot between seeds in these 250-step runs. A separate 3-seed check gave pixel errors of 57.8, 55.7 and **24.5** at β = 6.6: only one seed learned clean colours.
4. **Zero-shot composition** (4 held-out colour × position captions, 3 seeds, 300 steps):

   | | seed 0 | seed 1 | seed 2 |
   |---|---|---|---|
   | seen captions | 99% | 100% | 100% |
   | held-out captions | 50% | 45% | 57% |

   - Which new combinations work differs by seed. This is the paper's "performs inconsistently", at toy scale.
   - **Image completion:** given the first image row of a held-out caption, all 3 models complete it 100% correctly (from scratch: 100% / 90% / 40%).
5. **Reranking** with an oracle scorer: 83% → 94% → 100% as N goes 1 → 4 → 16.
6. **Scale tricks:**
   - per-resblock scales recover the 99.6% of a tiny-gradient block lost under one global fp16 scale;
   - PowerSGD with error feedback loses only the final residual.

**Not run (too heavy):** `experiments.py` uses a synthetic captioned-shapes world:
- E1: dVAE β ablation;
- E2: zero-shot attribute composition;
- E3: sparse vs dense attention;
- E4: loss weights;
- E5: learned vs fixed padding on out-of-distribution captions;
- E6: reranking with a small trained CLIP;
- E7: PowerSGD rank vs loss gap.

---

## 10. Check yourself

1. Why does the dVAE reduce the transformer's context by 192×?
   <details><summary>Answer</summary>256·256·3 = 196,608 values become 32·32 = 1024 tokens, and 196,608/1024 = 192.</details>
2. Logits (2, 0) and Gumbel draws (−0.5, 1.0). What is the hard sample, and the relaxed sample at τ = 1?
   <details><summary>Answer</summary>The sums are (1.5, 1.0), so the hard sample is index 0. Relaxed: softmax(1.5, 1.0) = (0.62, 0.38).</details>
3. What is KL(q ‖ Uniform) for K = 4 and q = (0.7, 0.1, 0.1, 0.1)?
   <details><summary>Answer</summary>Σ q log q + log 4 = (0.7 ln 0.7 + 3 · 0.1 ln 0.1) + 1.386 = (−0.250 − 0.691) + 1.386 = 0.446 nats.</details>
4. Why use the logit-Laplace instead of an ℓ1 loss?
   <details><summary>Answer</summary>ℓ1 corresponds to a Laplace distribution on the whole real line, which wastes probability on impossible pixel values. The logit-Laplace lives exactly on (0, 1).</details>
5. With W = 32 columns, how many image tokens does the row mask let a token see?
   <details><summary>Answer</summary>Itself plus the previous 32 in raster order: 33. That reaches the same column of the previous row.</details>
6. What is PowerSGD's compression rate for d_model = 1024 and r = 128?
   <details><summary>Answer</summary>1 − 5·128/(8·1024) = 1 − 640/8192 = 92.2%.</details>
7. Why does a per-resblock gradient scale help, when one global loss scale doesn't?
   <details><summary>Answer</summary>Gradients in different blocks differ by many orders of magnitude, and fp16 has a narrow exponent range. One scale big enough for the tiniest block overflows the largest. One scale per block puts each block's gradient in range.</details>
