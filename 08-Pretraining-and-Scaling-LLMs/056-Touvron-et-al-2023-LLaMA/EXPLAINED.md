# LLaMA: Open and Efficient Foundation Language Models, explained simply

**Paper:** Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave & Guillaume Lample (Meta AI), *LLaMA: Open and Efficient Foundation Language Models*, 2023.

**In one sentence:** train relatively **small** language models (7B–65B parameters) on **far more text** than "compute-optimal" rules suggest (1–1.4 trillion tokens), using **only public data** and three architecture tweaks (RMSNorm, SwiGLU, rotary embeddings). The result: **LLaMA-13B beats GPT-3 (175B)** on most benchmarks and runs on a single GPU, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B.

---

## 1. The big idea: optimise for inference, not training

### 1.1 What Chinchilla said
- Chinchilla (paper 052): for a fixed **training** budget, use about **20 tokens per parameter**.
- **Example:** for the compute of a 10B model on 200B tokens, that's about the best you can do.

### 1.2 What it ignores
- You train a model **once**, but you **run** it for millions of users.
- Every generated token costs about **2N FLOPs** (N = parameters). A smaller model is cheaper every single time it is used.
- **LLaMA's goal:** the best model for a given **inference** budget.
  - Take a smaller model and train it **longer**, past the compute-optimal point.
  - Training costs more, but the model keeps improving and is cheaper forever after.
- **The paper's observation:** "the performance of a 7B model continues to improve even after 1T tokens", that is **~150 tokens per parameter**.

### 1.3 Worked example with Chinchilla's fitted law (our demo, section 6)
- **The budget:** 10B × 200B tokens costs C = 6 × 10¹⁰ × 2 × 10¹¹ = 1.2 × 10²² FLOPs.
- Under the fitted law L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28, the compute-optimal model at this budget is **5.6B on 357B tokens**, with loss 2.126.
- **A model half that size** (2.8B) reaches the **same loss** with 862B tokens. That is 308 tokens per parameter and **1.21×** the training compute.
- **But it saves** 2 × 2.8B = 5.6 × 10⁹ FLOPs on **every** generated token. The extra 2.5 × 10²¹ training FLOPs are repaid after **4.5 × 10¹¹ generated tokens**, a volume a popular model serves quickly.

---

## 2. The data (Section 2.1, Table 1): only public sources

| Source | Share | Epochs (at 1.4T tokens) | Disk |
|---|---|---|---|
| CommonCrawl (CCNet) | 67.0% | 1.10 | 3.3 TB |
| C4 | 15.0% | 1.06 | 783 GB |
| GitHub | 4.5% | 0.64 | 328 GB |
| Wikipedia (20 languages) | 4.5% | 2.45 | 83 GB |
| Gutenberg + Books3 | 4.5% | 2.23 | 85 GB |
| ArXiv | 2.5% | 1.06 | 92 GB |
| StackExchange | 2.0% | 1.03 | 78 GB |

**Total:** about **1.4 trillion tokens**. Most data is seen once; Wikipedia and books about **twice**.

**How each source was cleaned:**
- **CommonCrawl:**
  - de-duplicated by line;
  - language-identified;
  - low-quality text removed with an n-gram language model;
  - plus a classifier that keeps pages that "look like" pages cited by Wikipedia.
- **GitHub:** permissive licences only (Apache, BSD, MIT); filtered by line length and alphanumeric ratio; exact-duplicate files removed.
- **Books:** near-duplicate books (> 90% overlap) removed.
- **ArXiv:** everything before the first section, and the bibliography, removed.
- **StackExchange:** answers sorted by score.

**Arithmetic check:**
- Wikipedia gets 4.5% × 1.4T = 63B tokens of training.
- At 2.45 epochs, that means about 25.7B unique Wikipedia tokens, repeated.

### 2.1 The tokenizer
- **BPE** (paper 034's idea) via SentencePiece, with a **32,000-token** vocabulary, plus two rules:
  - **Numbers are split into single digits:** "2023" → "2", "0", "2", "3". Every number is then spelled the same way, which helps arithmetic. Otherwise "2023" might be one token and "2024" two.
  - **Byte fallback:** a character not in the vocabulary is encoded as its UTF-8 **bytes**. For example, "€" becomes <0xE2> <0x82> <0xAC>. Nothing is ever "unknown".

---

## 3. The architecture (Section 2.2): GPT plus three upgrades

### 3.1 Pre-normalisation with RMSNorm [from GPT-3]
**Pre-norm:** normalise the **input** of each sub-layer (x + f(norm(x))), as in GPT-2/3 (paper 049). It trains more stably than the original post-norm.

**RMSNorm** (Zhang & Sennrich 2019) is a cheaper LayerNorm:
```
LayerNorm(x) = (x − mean(x)) / std(x) · g + b
RMSNorm(x)   = x / sqrt(mean(x²) + ε) · g
```
It skips subtracting the mean and has no bias.

**Example:** x = (1, 2, 3, 6).
- mean(x²) = (1 + 4 + 9 + 36)/4 = 12.5, so the RMS is √12.5 = 3.536.
- **RMSNorm:** (0.283, 0.566, 0.849, 1.697).
- **LayerNorm:** mean 3, standard deviation 1.871, giving (−1.069, −0.535, 0, 1.604).

**Why it's enough:** the important part of normalisation is controlling the **scale** of the activations. Re-centring adds little, and RMSNorm is faster.

### 3.2 SwiGLU feed-forward [from PaLM]
**The standard Transformer MLP:** W₂ · GELU(W₁ x), with a hidden size of 4d.

**SwiGLU** (Shazeer 2020) uses a **gate**:
```
FFN(x) = W₂ · ( SiLU(W₁ x) ⊙ (W₃ x) )        where SiLU(z) = z · sigmoid(z)
```
- ⊙ multiplies element by element.
- One branch, W₃x, carries information. The other, SiLU(W₁x), decides **how much of each hidden unit to let through**.
- **SiLU values:** SiLU(2) = 2 × 0.881 = 1.762; SiLU(0) = 0; SiLU(−2) = −2 × 0.119 = −0.238. It is smooth and lets small negative values through.

**Size bookkeeping:**
- There are three matrices instead of two, so the hidden size shrinks to **⅔ · 4d**: 3 · d · (8d/3) = 8d² parameters, the same as the usual 2 · d · 4d.
- The released code rounds the hidden size up to a multiple of 256: d = 4096 gives 10,922.7, rounded to **11,008**.

**Why:** across many experiments (Shazeer; PaLM), gated units give a lower loss at the same parameter count.

### 3.3 Rotary position embeddings (RoPE) [from GPT-Neo]
**The old way:** GPT adds a learned position vector to each token at the input.

**RoPE** (Su et al. 2021) removes that. Instead, inside **every attention layer**, it **rotates** the query and key vectors by an angle proportional to the position:
- Split the head's dimensions into pairs (x₁, x₂), (x₃, x₄), …
- Rotate pair i at position m by the angle m·θᵢ, where θᵢ = 10000^(−2i/d_head):
```
(x₁, x₂) → (x₁ cos mθ − x₂ sin mθ,  x₁ sin mθ + x₂ cos mθ)
```
- **Why this is clever:** rotations preserve lengths, and the dot product of two rotated vectors depends only on the **difference** of their angles. So the attention score between position m and position n depends only on **m − n**, the relative distance:
```
⟨R(mθ) q, R(nθ) k⟩ = ⟨q, R((n − m)θ) k⟩
```
- **2-D example:** q = k = (1, 0) and θ = 0.5.
  - Positions (m, n) = (3, 1): the angles are 1.5 and 0.5, so the score is cos(1.0) = 0.540.
  - Positions (10, 8): the angles are 5.0 and 4.0, so the score is cos(1.0) = 0.540 again.
- **Different speeds:** with head dimension 128, θ₀ = 1 radian per position (fast rotation, for fine local order) down to θ₆₃ ≈ 0.0001 (slow, for long distances).

### 3.4 Sizes (Table 2)
| Model | Parameters | d | Heads | Layers | Peak learning rate | Tokens |
|---|---|---|---|---|---|---|
| 7B | 6.7B | 4096 | 32 | 32 | 3.0e−4 | 1.0T |
| 13B | 13.0B | 5120 | 40 | 40 | 3.0e−4 | 1.0T |
| 33B | 32.5B | 6656 | 52 | 60 | 1.5e−4 | 1.4T |
| 65B | 65.2B | 8192 | 64 | 80 | 1.5e−4 | 1.4T |

- **Shared settings:** every model uses 128-dimensional heads, a 2048-token context, a 32k vocabulary, and **untied** input and output embeddings.
- **Parameter count:** 2·V·d (input + output embeddings) + layers × (4d² attention + 3·d·h SwiGLU + 2d norm gains) + d.
  - 7B: 2 × 32000 × 4096 + 32 × (4 × 4096² + 3 × 4096 × 11008 + 8192) + 4096 = **6.74B** ✓.

### 3.5 Optimisation (Section 2.3)
- AdamW (β₁ = 0.9, β₂ = 0.95), weight decay 0.1, gradient clipping 1.0;
- **2000 warm-up steps**, then cosine decay to **10% of the peak**;
- batch **4M tokens** (2048 sequences of 2048 tokens).

---

## 4. Making training fast (Section 2.4)

**Memory-efficient causal attention** (xformers; FlashAttention-style):
- **Never store the T × T attention matrix:** process the keys in blocks and keep a **running softmax**.
- **Skip the future:** with a causal mask, about half the scores are masked anyway, so whole future blocks are never computed.

**The online softmax** combines blocks exactly. For one query, the scores are [1, 2] in block 1 and [3] in block 2.
- **Block 1:** running max m = 2. Normaliser l = e^(1−2) + e^(2−2) = 1.368. Accumulator = e^(−1)·v₁ + 1·v₂.
- **Block 2:** the new max is 3, so rescale the old values by e^(2−3) = 0.368. Then l = 1.368 × 0.368 + 1 = 1.503.
- **The final weights** are (e^(−2), e^(−1), 1)/1.503 = (0.090, 0.245, 0.665), exactly softmax([1, 2, 3]).

**Activation checkpointing, done by hand:** they save expensive activations (the outputs of linear layers) and recompute only the cheap ones, using a hand-written backward pass, model and sequence parallelism, and overlapped communication.

**Speed:**
- **Throughput:** about **380 tokens per second per GPU** on 2048 A100-80GB GPUs.
- **Time for 65B:** 1.4 × 10¹² / (380 × 2048) = 1.8 × 10⁶ s ≈ **21 days**.
- **Utilisation:** 6ND = 6 × 65.3B × 1.4T = 5.5 × 10²³ FLOPs. Against the A100's 312 TFLOP/s peak, that is about **48%**, very good for this scale.

---

## 5. Results

### 5.1 Common-sense reasoning, zero-shot (Table 3)
| Model | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|---|
| GPT-3 175B | 60.5 | 81.0 | 78.9 | 70.2 | 68.8 | 51.4 | 57.6 |
| LLaMA-7B | 76.5 | 79.8 | 76.1 | 70.1 | 72.8 | 47.6 | 57.2 |
| **LLaMA-13B** | **78.1** | 80.1 | **79.2** | **73.0** | **74.8** | **52.7** | 56.4 |
| LLaMA-33B | 83.1 | 82.3 | 82.8 | 76.0 | 80.0 | 57.8 | 58.6 |
| LLaMA-65B | 85.3 | 82.8 | 84.2 | 77.0 | 78.9 | 56.0 | 60.2 |

- **LLaMA-13B beats GPT-3 on 5 of these 7**, with 13× fewer parameters.
- **Scoring:** for multiple-choice questions, the answer with the highest likelihood is chosen (paper 050).
  - For some datasets, the likelihood is normalised by the number of characters.
  - For others, it is normalised by the answer's likelihood given only "Answer:" as context: P(completion | context) / P(completion | "Answer:").

### 5.2 MMLU, 5-shot (Tables 9–10)
| Model | MMLU |
|---|---|
| GPT-3 175B | 43.9 |
| Gopher 280B | 60.0 |
| Chinchilla 70B | 67.5 |
| PaLM 540B | 69.3 |
| LLaMA-7B / 13B / 33B / 65B | 35.1 / 46.9 / 57.8 / **63.4** |
| LLaMA-I 65B (brief instruction tuning) | **68.9** |

- **LLaMA-65B is behind Chinchilla and PaLM here.** The authors suspect it is because LLaMA saw fewer books and academic papers (177 GB vs up to 2 TB of books for the others).
- **A short instruction fine-tune** (LLaMA-I) raises the score to 68.9.

### 5.3 Other findings
- **Code:**
  - LLaMA-13B and larger beat LaMDA-137B on HumanEval and MBPP;
  - LLaMA-65B beats PaLM-62B;
  - none of them was trained specifically on code (paper 054's metric, pass@k).
- **Maths:** on GSM8k, LLaMA-65B beats Minerva-62B, even though Minerva was trained on mathematical data.
- **Steady improvement:** performance on most benchmarks rises steadily during training (Figure 2). WinoGrande is less correlated with training perplexity.

### 5.4 Bias, toxicity, truthfulness (Section 5)
- **Toxicity (RealToxicityPrompts):** it increases with model size.
- **Bias:**
  - CrowS-Pairs: LLaMA-65B is slightly less biased than GPT-3 and OPT on average, but strongly biased on religion;
  - WinoGender: it shows gender bias in co-reference resolution.
- **Truthfulness (TruthfulQA):** better than GPT-3, but still often wrong. The authors note it is likely to "hallucinate".

### 5.5 Carbon (Table 15)
**The formula:** MWh = GPU-hours × 400 W × 1.1 (PUE, the data-centre overhead); tCO₂eq = MWh × 0.385 (the US average grid).

| Model | GPU-hours | MWh | tCO₂eq |
|---|---|---|---|
| LLaMA-7B | 82,432 | 36 | 14 |
| LLaMA-65B | 1,022,362 | 449 | 173 |
| OPT-175B | 809,472 | 356 | 137 |
| BLOOM-175B | 1,082,880 | 475 | 183 |

**Example (65B):** 1,022,362 × 0.4 kW × 1.1 = 449,839 kWh = 449.8 MWh, and × 0.385 = **173 t**.

**The whole project:** 2048 GPUs for ~5 months, about 2,638 MWh and **1,015 tCO₂eq**. One reason to release the models is that others need not repeat this.

---

## 6. Why it matters
- **Open weights.** The models were released to researchers. This kicked off the open-model ecosystem: Alpaca, Vicuna, Llama 2 and 3, Mistral, and many fine-tunes.
- **"Train small models longer" became standard practice.** Llama 3 8B was trained on 15T tokens, about 1,900 tokens per parameter.
- **RMSNorm + SwiGLU + RoPE became the default recipe** for decoder LLMs.

---

## 7. What our code found
Everything below runs on a laptop in ~17 seconds. No large model is trained.

**Exact checks (tests):**
- **RMSNorm** = x / RMS (unit RMS output, mean not removed).
- **SwiGLU hidden sizes** are exactly **11008 / 13824 / 17920 / 22016** for d = 4096 / 5120 / 6656 / 8192, and with hidden = ⅔·4d it has the same parameter count as a 4d MLP.
- **RoPE** preserves norms, and the score depends only on the offset. In the demo, positions (5, 2), (50, 47) and (120, 117) all give +1.1783.
- **Memory-efficient attention** equals standard causal attention: max difference 6 × 10⁻⁷ at T = 1024. It stores at most 128 × 128 scores per head (1.6% of 1024²) and computes 36 of 64 key blocks.
- **The model** is causal, and the parameter formula equals the built model.
- **Table 2:** **6.74 / 13.02 / 32.53 / 65.29B** vs the paper's 6.7 / 13.0 / 32.5 / 65.2B. The paper truncates; 65.29 → "65.2".
- **The 65B run:** **20.8 days** at 380 tokens/s/GPU on 2048 GPUs (paper: ~21), at about **48%** utilisation of peak FLOPs.
- **Table 15's carbon numbers** reproduce exactly: 14 / 23 / 90 / 173 t; OPT 137; BLOOM 183.
- **The tokenizer rules:** digit splitting and byte fallback ("€" → 3 bytes).
- **The LR schedule** ends at 10% of the peak.

**The inference-budget argument** under paper 052's law: a half-size model matches the compute-optimal loss with 1.21× the training compute, and repays it after 4.5 × 10¹¹ generated tokens. LLaMA-7B on 1T tokens is predicted to reach an even lower loss (2.052 vs 2.126) at 3.5× that compute.
- **Honest note:** under the fitted Approach-3 law, "10B on 200B" is not exactly optimal. The law prefers 5.6B on 357B (about 64 tokens per parameter), the inconsistency between Chinchilla's approaches seen in paper 052.

**A tiny race** (byte-level, ~650k parameters, 3 layers, 250 steps on this repository's own text):

| Model | Validation loss (nats/byte) |
|---|---|
| GPT-2 block (LayerNorm, GELU 4d, learned positions, tied) | 2.799 |
| LLaMA block (RMSNorm, SwiGLU, RoPE, untied) | **2.091** |

- **The LLaMA block wins clearly here.**
- **Honest caveat:** one seed, a tiny budget, and several differences at once (including tied vs untied output embeddings). A gap this large probably reflects **faster early optimisation** as much as final quality.
- `experiments.py` E1 ablates each change separately, with 3 seeds and longer training.

**`experiments.py` (written, not run on this laptop):**
- **E1:** one-at-a-time ablation of RMSNorm / SwiGLU / RoPE;
- **E2:** small models trained to 500 tokens per parameter (does the loss keep improving?);
- **E3:** digit splitting vs BPE-merged digits on addition;
- **E4:** time and memory of naive vs blockwise attention up to 8192 tokens;
- **E5:** the cheapest (N, D) for training only vs training plus inference.

---

## 8. Check yourself

1. Why would you train a 7B model on 1T tokens when Chinchilla says ~140B is compute-optimal?
<details><summary>Answer</summary>Compute-optimal minimises training cost for a given loss. But inference costs 2N FLOPs per token, forever. A smaller model trained longer reaches the same (or better) loss and is cheaper on every request, so the extra training is repaid once enough tokens are served.</details>

2. Compute RMSNorm of x = (3, −4) (with g = 1).
<details><summary>Answer</summary>mean(x²) = (9 + 16)/2 = 12.5, so RMS = 3.536 and the output is (0.849, −1.131). The mean (−0.5) was not subtracted.</details>

3. Why is SwiGLU's hidden size ⅔ · 4d rather than 4d?
<details><summary>Answer</summary>SwiGLU has three weight matrices (W₁, W₃ of size d × h and W₂ of size h × d), so 3dh parameters. Setting h = 8d/3 makes 3dh = 8d², the same as a standard two-matrix MLP with hidden 4d. That keeps the comparison at equal parameters.</details>

4. With RoPE, why does the attention score depend only on the distance between tokens?
<details><summary>Answer</summary>The query at position m is rotated by mθ and the key at n by nθ. The dot product of two rotated vectors equals the dot product of the original query with the key rotated by (n − m)θ. Only the difference of the angles survives.</details>

5. Where does RoPE inject position information, compared with GPT-2's learned position embeddings?
<details><summary>Answer</summary>GPT-2 adds a position vector once, at the input. RoPE rotates the queries and keys inside every attention layer, so position affects only the attention scores (relative positions), not the values or the residual stream directly.</details>

6. Explain how memory-efficient attention avoids storing the T × T matrix but still gets the exact softmax.
<details><summary>Answer</summary>It processes the keys in blocks and keeps, for each query, the running maximum m, the running normaliser l = Σ exp(s − m) and the running weighted sum of values. When a new block raises the maximum, the old l and sum are rescaled by exp(m_old − m_new). At the end, sum / l is exactly softmax(scores) · V. Future blocks are skipped entirely because the causal mask zeros them.</details>

7. Check the "21 days" claim.
<details><summary>Answer</summary>380 tokens/s/GPU × 2048 GPUs = 778,240 tokens/s. 1.4 × 10¹² / 778,240 ≈ 1.80 × 10⁶ s ≈ 20.8 days.</details>

8. Why split numbers into single digits?
<details><summary>Answer</summary>With BPE, numbers get split inconsistently ("2023" might be one token, "2024" two, "12345" some odd chunks), which makes arithmetic patterns hard to learn. With one token per digit, every number has a consistent, place-value-like representation.</details>

9. LLaMA-65B scores 63.4 on MMLU, below Chinchilla (67.5). What reason do the authors give?
<details><summary>Answer</summary>LLaMA used a relatively small amount of books and academic papers (ArXiv, Gutenberg, Books3: 177 GB), while Gopher, Chinchilla and PaLM used up to 2 TB of books. MMLU rewards that kind of knowledge.</details>

10. In our tiny race, the LLaMA block beat the GPT-2 block by 0.7 nats/byte. Why shouldn't you quote that as "LLaMA's architecture is 25% better"?
<details><summary>Answer</summary>One seed, 250 steps and ~650k parameters: tiny-budget results mostly measure how fast each model starts learning, not its final quality. Several things changed at once (norm, MLP, positions, tied vs untied output), so you can't attribute the gap to any one of them. A proper comparison needs ablations, multiple seeds and longer training (experiments.py E1).</details>
