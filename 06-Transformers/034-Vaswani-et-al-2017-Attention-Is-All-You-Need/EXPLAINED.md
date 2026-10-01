# Vaswani et al. (2017), explained from scratch

**Paper:** *Attention Is All You Need*
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research)
**Published at:** NIPS 2017 (arXiv:1706.03762)

Read Papers 027 (Seq2Seq), 028 (attention) and 033 (ByteNet) first. This paper introduces the **Transformer**, the architecture behind BERT, GPT, LLaMA, ViT and almost every large model today. This guide:
- computes attention by hand;
- derives why it is scaled by √d_k;
- proves the "relative position" property of the sinusoidal encoding;
- counts the parameters of the base model;
- explains every training trick with numbers.

---

## 0. The whole idea in one line

> **Drop recurrence and convolution entirely. Every layer lets every position look directly at every other position through attention (several "heads" at once), followed by a small per-position network. Add position information explicitly. The result trains in parallel, connects any two words in one step, and beat every previous translation model at a fraction of the cost.**

---

## 1. Why get rid of RNNs? (Sections 1–2, 4)

- **RNNs are sequential:** h_t needs h_{t−1}, so the n steps of a sentence can't be computed in parallel. That hurts most on long sequences, where memory limits batching.
- **Convolutional models** (ByteNet, Paper 033; ConvS2S) are parallel, but connecting two distant positions takes O(log n) layers (dilated) or O(n/k) layers (plain).
- **Attention connects any two positions in one step.** Until now it was always used *with* an RNN (Paper 028). This paper uses **only** attention.

### Table 1: per-layer cost, sequential steps and maximum path length
| Layer type | Cost per layer | Sequential operations | Maximum path length |
|---|---|---|---|
| **self-attention** | O(n²·d) | **O(1)** | **O(1)** |
| recurrent | O(n·d²) | O(n) | O(n) |
| convolutional | O(k·n·d²) | O(1) | O(log_k n) |
| restricted self-attention (window r) | O(r·n·d) | O(1) | O(n/r) |

**Where the costs come from:**
- **Self-attention:** every one of the n positions computes a dot product with every other position, each of size d. That's n² dot products × d = **n²·d**.
- **RNN:** each of the n steps multiplies a d-vector by a d×d matrix, so **n·d²**.

**Self-attention is cheaper when n < d.** Our demo, with d = 512:
- at n = 50: 1.28M (attention) vs 13.1M (RNN);
- at n = 500 they're about equal;
- at n = 5000 attention is 10× *more* expensive.

That quadratic cost later motivated Paper 036 (sparse attention).

---

## 2. Attention, precisely (Section 3.2)

### 2.1 Queries, keys and values
- **The analogy:** attention is a **soft dictionary lookup**.
  - Each position emits a **query** q ("what am I looking for?"), a **key** k ("what do I contain?") and a **value** v ("what do I hand over if chosen?").
  - A query is compared with every key, and the result is a **weighted average of the values**, weighted by how well each key matches.
- **Scaled dot-product attention (Eq. 1)**, for all queries at once (stacked as matrix rows):
  ```
  Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
  ```
  1. Q Kᵀ: entry (i, j) is the dot product of query i with key j (the match score).
  2. Divide by √d_k (next section).
  3. softmax along each row: weights that are positive and sum to 1.
  4. Multiply by V: each output row is a weighted average of the value rows.

### 2.2 By hand (our demo)
**Setup:**
```
Q = [[1, 0], [0, 1]]          K = [[1, 0], [0, 1], [1, 1]]          V = [[10, 0], [0, 10], [5, 5]]
```
**Step 1, scores:** Q Kᵀ/√2 = [[0.707, 0, 0.707], [0, 0.707, 0.707]].

**Step 2, softmax of each row:** e^0.707 = 2.028 and e^0 = 1, so row 1 becomes (2.028, 1, 2.028)/5.056 = **(0.401, 0.198, 0.401)**.

**Step 3, output for query 1:**
```
0.401·(10, 0) + 0.198·(0, 10) + 0.401·(5, 5) = (6.02, 3.98)
```
**Read it:** query 1 resembles keys 1 and 3, so it mostly reads values 1 and 3.

### 2.3 Why divide by √d_k? (footnote 4)
- **The variance:** if the q and k components are independent with mean 0 and variance 1, then q·k = Σ_{i=1}^{d_k} q_i k_i is a sum of d_k terms, each with variance 1. So
  ```
  Var(q·k) = d_k              std = √d_k
  ```
- **The problem:** for d_k = 512 the scores have std ≈ 23. The softmax of numbers that far apart is almost one-hot, and its gradients (p(1 − p)) almost vanish.
- **The fix:** dividing by √d_k brings the variance back to **1**.
- **Our numbers** (the largest softmax weight among 6 random keys):

  | d_k | unscaled | scaled |
  |---|---|---|
  | 4 | 0.57 | 0.37 |
  | 512 | **1.000** (saturated) | 0.33 |

- **Our test** checks Var(q·k) ≈ 512 and Var after scaling ≈ 1, over 20,000 samples.
- **Why not additive attention?** Additive attention (Paper 028) is comparable in quality, but dot-product attention is **one matrix multiply**, which is far faster on GPUs.

### 2.4 Masking
- **How:** set forbidden scores to −∞ before the softmax. e^{−∞} = 0, so those positions get **exactly zero weight**.
- **Used for:**
  - **padding** (never attend to pad tokens);
  - **the decoder's causal mask** (position i may only attend to positions ≤ i).

### 2.5 Multi-head attention (Section 3.2.2)
```
MultiHead(Q, K, V) = Concat(head₁, …, head_h) W^O,    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```
- **Why several heads:** a single attention computes **one** weighted average per position, which blurs different kinds of information together ("averaging inhibits this").
- **With h heads,** each head projects into its own smaller subspace (d_k = d_v = d_model/h = 64 for h = 8) and can attend to **different positions for different reasons**: one head tracks the previous word, another the subject of the verb, and so on.
- **The cost:** h heads of size 64 cost about the same as one head of size 512.
- **Parameters per attention layer:** W^Q, W^K, W^V, W^O are each d_model × d_model, so **4·d_model² (+ biases)**. For d_model = 512 that's 1.05M.

### 2.6 The three uses of attention in the model (Section 3.2.3)
1. **Encoder self-attention:** Q, K and V all come from the previous encoder layer. Every source word can look at every source word.
2. **Decoder self-attention, masked:** every target position looks at earlier target positions only.
3. **Encoder–decoder attention:** queries come from the decoder, and keys and values from the **encoder output**. This is exactly Paper 028's attention, in dot-product form.

---

## 3. The rest of the layer

### 3.1 Position-wise feed-forward network (Eq. 2)
```
FFN(x) = max(0, x W₁ + b₁) W₂ + b₂          d_model = 512 → d_ff = 2048 → 512
```
- **Per position:** applied to each position **separately and identically**, like two 1×1 convolutions.
- **Its job:** attention **mixes information across positions**; the FFN **processes it within each position**.
- **Parameters:** 2·512·2048 + 2048 + 512 ≈ **2.1M**, twice the attention layer.

### 3.2 Residuals and layer normalization (Section 3.1)
- **Every sub-layer is wrapped as:**
  ```
  output = LayerNorm(x + Dropout(Sublayer(x)))
  ```
  - **The residual connection** (Paper 016) lets gradients flow through 12+ layers.
  - **Layer norm** (Paper 033, section 2.3) keeps each position's vector at a stable scale.
- **"Post-LN":** the norm comes after the addition. Later work moved it **before** the sub-layer ("pre-LN": x + Sublayer(LN(x))), which trains more stably without warm-up. Most LLMs use pre-LN.

### 3.3 Stacks (Figure 1)
- **The encoder:** N = 6 layers of [self-attention, FFN].
- **The decoder:** N = 6 layers of [masked self-attention, encoder–decoder attention, FFN].
- **Everything is d_model = 512 wide,** so the residual additions line up.

### 3.4 Embeddings and the output layer (Section 3.4)
- **Weight tying:** **one** matrix is shared by the source embedding, the target embedding and the pre-softmax projection. Source and target share one BPE vocabulary of ~37k units, so a word piece means the same thing everywhere, and the model saves ~37M parameters.
- **The embeddings are multiplied by √d_model** (≈ 22.6):
  - the shared matrix is initialized small, because it also serves as the output layer;
  - the positional encodings have entries of size ~1;
  - scaling keeps the token signal from being drowned out by the position signal.

---

## 4. Positional encoding (Section 3.5)

### 4.1 Why it's needed
- **Self-attention is order-blind:** permuting the input positions just permutes the outputs (Paper 032).
- **So position must be injected.** The Transformer **adds** a position vector to each token embedding at the bottom of both stacks.

### 4.2 The sinusoids
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- **Each pair of dimensions (2i, 2i+1)** is a point going round a circle as pos increases.
- **The speeds differ:** angular frequency ω_i = 1/10000^(2i/d), so the wavelengths range from **2π** (fast, i = 0) to **10000·2π** (slow).
- **Like a clock:** the fast dimensions tell nearby positions apart, and the slow ones place a token coarsely in a long sequence. It's like seconds, minutes and hours hands.

### 4.3 The relative-position property, proved
For one pair of dimensions, the position is (sin ωp, cos ωp). Shifting by k:
```
sin(ω(p+k)) =  sin(ωp)cos(ωk) + cos(ωp)sin(ωk)
cos(ω(p+k)) =  cos(ωp)cos(ωk) − sin(ωp)sin(ωk)
⇒  [sin ω(p+k); cos ω(p+k)] = [[cos ωk, sin ωk], [−sin ωk, cos ωk]] · [sin ωp; cos ωp]
```
- **What it says:** PE(pos + k) = **M_k · PE(pos)**, where M_k is a block-diagonal **rotation** that depends only on k, not on pos.
- **The consequence:** "look k positions back" is one fixed linear map, the same everywhere. Also, PE(p)·PE(p + k) = Σ_i cos(ω_i k) depends only on k.
- **Our demo:** PE(p)·PE(p + 5) = **23.504** at p = 0, 37 and 150 alike.
- **Our test** checks M_k exactly.
- **Learned vs fixed:** learned position embeddings worked equally well (Table 3, row E). The sinusoids were kept because they *might* extrapolate to longer sequences. Later models mostly use RoPE (rotary embeddings, Paper 056), which builds this rotation directly into Q and K.

---

## 5. Training (Section 5)

### 5.1 Data and batching
- **WMT14 En–De:** 4.5M pairs with a **shared BPE vocabulary of ~37k** units.
- **WMT14 En–Fr:** 36M pairs, 32k word-pieces.
- **Batches** of about **25,000 source + 25,000 target tokens**, grouped by length.

**Byte-pair encoding (Sennrich et al. 2016; our `learn_bpe`):**
- start from characters, and repeatedly merge the **most frequent adjacent pair** into a new symbol;
- frequent words become single units, and rare words split into known pieces;
- so there are **no unknown words** and the vocabulary stays small.

*Example:* in {low ×5, lower ×2, newest ×6, widest ×3}, the pairs "es", "st" and "t</w>" each occur 9 times, so one of them merges first. An unseen word like "lowest" still splits into known pieces.

### 5.2 Hardware
- **Base:** 8 P100 GPUs, 0.4 s/step, **100k steps (12 hours)**.
- **Big:** 1.0 s/step, **300k steps (3.5 days)**.

### 5.3 Optimizer and the warm-up schedule (Eq. 3)
- **Adam** with β₁ = 0.9, **β₂ = 0.98** (shorter second-moment memory than the default 0.999, Paper 011), ε = 10⁻⁹.
- **The schedule:**
  ```
  lrate = d_model^(−0.5) · min(step^(−0.5), step · warmup^(−1.5)),      warmup = 4000
  ```
  - **For step < 4000,** the second term is smaller: lr grows **linearly**.
  - **For step > 4000,** the first term is smaller: lr decays like **1/√step**.
  - **The peak**, at step 4000: 512^(−0.5)·4000^(−0.5) = **0.000699** (our demo). At step 16,000 it has halved (0.000349).
- **Why warm up:**
  - early on, Adam's second-moment estimates are poor and post-LN Transformers have large gradients near the output;
  - big steps then can wreck training;
  - warm-up starts gently (pre-LN models need far less).

### 5.4 Regularization
- **Residual dropout 0.1:**
  - on each sub-layer's output before it's added to the residual;
  - on the embeddings + positions sum.
- **Label smoothing ε = 0.1:** the training target becomes
  ```
  q = (1 − ε)·one_hot + ε·uniform over K classes
  ```
  - **The model never sees a 100%-certain target,** so it can't push its logits to infinity. That improves calibration of the decoding distribution, accuracy and BLEU, although **perplexity gets worse** (the model is deliberately less sure).
  - **The best achievable loss** is no longer 0 but the **entropy of q**. For K = 12: q = (0.908, 0.008, …), so H(q) = **0.526**.
  - **Our toy model's training loss ended at 0.537,** right at that floor.

---

## 6. Results (Section 6)

### 6.1 Table 2: translation (BLEU on newstest2014)
| Model | EN–DE | EN–FR | Training FLOPs (EN–DE) |
|---|---|---|---|
| ByteNet | 23.75 | | |
| GNMT + RL | 24.6 | 39.92 | 2.3·10¹⁹ |
| ConvS2S | 25.16 | 40.46 | 9.6·10¹⁸ |
| MoE | 26.03 | 40.56 | 2.0·10¹⁹ |
| GNMT + RL ensemble | 26.30 | 41.16 | 1.8·10²⁰ |
| ConvS2S ensemble | 26.36 | 41.29 | 7.7·10¹⁹ |
| **Transformer (base)** | **27.3** | 38.1 | **3.3·10¹⁸** |
| **Transformer (big)** | **28.4** | **41.8** | 2.3·10¹⁹ |

- **The big model beats every previous model, ensembles included, by over 2 BLEU,** with far less compute.
- **Inference:** beam 4, length penalty α = 0.6, max length = input + 50.
- **Checkpoint averaging:** the last 5 (base) or 20 (big) checkpoints, written 10 minutes apart, are averaged into one model.
- **A detail:** the text of Section 6.1 says the big En–Fr model reaches **41.0**, while the abstract and Table 2 say **41.8**. The arXiv versions differ.

**Length penalty** (from GNMT):
- beam search ranks finished candidates by log p(Y)/lp(Y), with lp(Y) = ((5 + |Y|)/6)^α;
- this counters the short-output bias of summing log-probabilities (Paper 027, section 5.3).

### 6.2 Table 3: model variations (dev set, newstest2013)
| Row | Change | BLEU (base 25.8) |
|---|---|---|
| (A) | 1 head (d_k = 512) | 24.9 (−0.9) |
| (A) | 16 heads (d_k = 32) | 25.8 |
| (A) | 32 heads (d_k = 16) | 25.4 |
| (B) | d_k = 16 (keys only smaller) | 25.1 |
| (C) | 2 layers | 23.7 |
| (C) | d_model = 1024 | 26.0 |
| (C) | d_ff = 4096 | 26.2 |
| (D) | no dropout | 24.6 |
| (E) | learned positions | 25.7 |
| **big** | d_model 1024, d_ff 4096, 16 heads, dropout 0.3, 300k steps | **26.4** |

**The lessons:**
- **heads help, but not too many;**
- **small keys hurt:** "determining compatibility is not easy";
- **bigger is better;**
- **dropout matters;**
- **positions can be learned or fixed.**

### 6.3 Table 4: constituency parsing (WSJ section 23)
| Parser | Training | F1 |
|---|---|---|
| Vinyals & Kaiser (Paper 030) | WSJ only | 88.3 |
| Petrov et al. (BerkeleyParser) | WSJ only | 90.4 |
| **Transformer (4 layers)** | WSJ only | **91.3** |
| Vinyals & Kaiser | semi-supervised | 92.1 |
| **Transformer** | semi-supervised | **92.7** |

- **The same architecture,** with almost no tuning, beats the RNN seq2seq parser.
- **It beats the BerkeleyParser using only 40k sentences**, which RNN seq2seq models couldn't do.

---

## 7. Why it matters

- **Almost every modern large model is a Transformer:**
  - BERT (the encoder only, Paper 037);
  - GPT (the decoder only, Papers 048–050);
  - T5 (both);
  - ViT (images, Paper 038);
  - Whisper (audio), CLIP, …
- **Two properties made scaling possible:**
  - **parallel training over positions**, which suits GPUs;
  - **short paths**, which make long-range dependencies learnable.
- **Its weaknesses shaped later work:**
  - quadratic cost in n (Papers 036, 079);
  - slow autoregressive decoding (Paper 035 and KV caching, Paper 080);
  - post-LN instability (pre-LN).

---

## 8. What our code found

**Scale note:**
- At your request, nothing heavy was run on this laptop.
- `experiments.py` reproduces, on Multi30k with our own BPE:
  - E1: Table 2's base model, with checkpoint averaging;
  - E2: Table 3's variation rows;
  - E3: training speed vs Paper 028's RNNsearch;
  - E4: attention-head plots;
  - E5: Table 4's parsing on NLTK's WSJ sample.

**Checked (tests and demo, ~7 seconds):**
- **Eq. 1** matches a hand computation; masked positions get exactly 0 weight.
- **Footnote 4:** Var(q·k) ≈ 512 for d_k = 512, and ≈ 1 after scaling (20,000 samples).
- **Multi-head:** with h = 1 it equals projected attention.
- **Sinusoidal positions:** the formula is exact, PE(pos + k) = M_k·PE(pos) with a rotation matrix, and the dot products depend only on the offset.
- **Masks:**
  - **causal:** changing target token 4 changes no prediction before position 4;
  - **padding:** appending pad tokens leaves the real tokens' encodings unchanged.
- **Model sizes:**
  - **base: 63.1M parameters** (the paper says 65M), with a shared 37k vocabulary and tied embeddings;
  - **big: 214.3M** (the paper: 213M).

  The small gap depends on details the paper doesn't give, such as the exact vocabulary size and which biases exist.
- **The Noam schedule** peaks at exactly 512^−½·4000^−½ and halves by step 16,000.
- **Label smoothing:** the loss matches the formula, and its minimum equals the entropy of the smoothed target.
- **Checkpoint averaging, the length penalty and BPE** behave as described.
- **A tiny Transformer** (2 layers, d_model 64) learns to reverse sequences:
  - **50/50 correct** with beam 4;
  - training loss 0.537 vs the label-smoothing floor 0.526;
  - **one decoder–encoder head shows a clean anti-diagonal.**

**An observation:**
- On the toy task, the paper's warm-up schedule scaled to d_model 64 (peak ≈ 0.009 with warm-up 200) got only **42%** of sequences exactly right after 400 steps, while a constant lr of 10⁻³ got **100%**.
- The schedule is tuned for 100k-step runs; tiny models and short runs prefer smaller, constant rates.

---

## 9. Check yourself

1. Compute Attention for one query q = (1, 1) over keys (1, 0) and (0, 1) with values 2 and 4 (d_k = 2). (Equal scores 0.707, so the weights are ½ and ½ and the output is 3.)
2. Show Var(q·k) = d_k for independent unit-variance components. What goes wrong without scaling?
3. Why use 8 heads of size 64 instead of 1 head of size 512? What does Table 3 (A) show?
4. Count the parameters of one encoder layer (d_model 512, d_ff 2048). (≈ 1.05M + 2.10M + LN ≈ 3.15M.)
5. Prove PE(pos + k) = M_k·PE(pos) for one pair of dimensions.
6. List the three kinds of attention in the model and where Q, K, V come from in each.
7. Compute the learning rate at steps 1000, 4000 and 16000.
8. Why does label smoothing hurt perplexity but help BLEU? What is the best loss for K classes and ε?
9. Why is self-attention cheaper than an RNN for short sentences, but not for long documents?
