# Child, Gray, Radford & Sutskever (2019), explained from scratch

**Paper:** *Generating Long Sequences with Sparse Transformers*
**Authors:** Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever (OpenAI)
**Published:** arXiv:1904.10509, 2019

Read Paper 034 (the Transformer) first, and Paper 033 (ByteNet) for dilation. This guide:
- explains why full attention is too expensive for long sequences;
- explains how "factorized" attention patterns cut the cost from n² to n√n while still connecting every pair of positions in two steps;
- covers the training changes (pre-norm residuals, depth-scaled init, recomputation) that let the network be 128 layers deep.

---

## 0. The whole idea in one line

> **Instead of every position attending to every earlier position (n² pairs), split attention into two heads that each look at only about √n positions: for example "the last √n positions" and "every √n-th position". Any position can still reach any earlier one in two hops. That makes attention over 10,000–1,000,000-long sequences (images as pixels, audio, raw text bytes) affordable.**

---

## 1. The problem: attention is quadratic (Sections 1, 3)

- **The task:** autoregressive generation models p(x) = Π_i p(x_i | x₁ … x_{i−1}) (Eq. 1). For images, audio and bytes, the sequences are **long**:
  - a CIFAR-10 image is 32·32·3 = **3,072** bytes;
  - a 64×64 ImageNet image is 12,288;
  - 5 seconds of 12 kHz audio is **65,536**.
- **The cost:** full self-attention computes a weight for every pair (j ≤ i): n(n+1)/2 pairs per head per layer.
  - At n = 16,384 that's 134 million pairs, and the attention matrices must be stored for backprop.
- **Earlier long-sequence models** (WaveNet, ByteNet: Paper 033) used dilated convolutions, which have fixed connectivity. Attention can **choose** what to look at; the question is how to keep that flexibility at an affordable cost.

---

## 2. What full attention learns (Section 4.1, Figure 2)

A 128-layer dense Transformer trained on CIFAR-10 shows:
- **early layers:** **local**, convolution-like patterns;
- **layers 19–20:** attention split into **row** and **column** attention (it factorized itself);
- **some layers:** **global, data-dependent** patterns;
- **layers 64–128:** **very sparse** attention.

Most layers use only a small part of the n² matrix. That suggests imposing sparsity up front.

---

## 3. Factorized self-attention (Section 4.2)

### 3.1 The formalism
- **The notation:** a self-attention layer has a connectivity pattern S = {S₁, …, S_n}, where S_i is the set of positions output i attends to:
  ```
  Attend(X, S) = ( a(x_i, S_i) )_i,       a(x_i, S_i) = softmax( (W_q x_i) K_{S_i}ᵀ / √d ) V_{S_i}       (Eqs. 2–4)
  ```
  K_{S_i} and V_{S_i} stack the keys and values of the positions in S_i.
- **Full causal attention:** S_i = {j : j ≤ i}.
- **Factorized attention** uses p heads with **subsets** A^(m)_i ⊂ {j ≤ i}, each of size ∝ n^{1/p}.

### 3.2 Two requirements
1. **Small:** |A^(m)_i| ∝ √n (for p = 2), so the total work is O(n√n).
2. **Valid:**
   - for every j ≤ i there must be a **path** j → a → i, with j ∈ A^(1)_a and a ∈ A^(2)_i;
   - so information from any earlier position reaches i within **2 attention steps** (p + 1 counting the direct case).
   - That keeps the Transformer's short-path advantage (Paper 034, Table 1).

### 3.3 The strided pattern (stride l ≈ √n)
```
A^(1)_i = {j : i − l < j ≤ i}                    the previous l positions ("local")
A^(2)_i = {j ≤ i : (i − j) mod l = 0}            every l-th earlier position ("same column")
```
- **The picture:** lay the sequence out in rows of width l, like an image. Head 1 sees the recent row segment, and head 2 sees the **same column** in all rows above (Figure 3b; our demo draws it).
- **Why it's valid:** to get from j to i, first hop to the position a = i − l·⌊(i − j)/l⌋. It lies within l after j, so j is in a's local window. And a is in i's column, so i can see a.
- **Our check:** every j ≤ i is reachable in two steps, in **either** order of the heads.

### 3.4 The fixed pattern
```
A^(1)_i = {j ≤ i : ⌊j/l⌋ = ⌊i/l⌋}                the current block
A^(2)_i = {j ≤ i : j mod l ∈ the last c of the block}      'summary' cells of every block
```
- **The idea:** the last c cells of each block act as **summaries** that every later position can read. In the paper's example (stride 128, c = 8), all later positions read positions 120–128, 248–256, and so on.
- **Order matters:** the fixed pattern is valid **only in one order**. Information first spreads **within a block** (head 1) into the summary cells, then **across blocks** (head 2). Our check confirms that block-then-summary covers everything, while summary-then-block doesn't.
- **The c trade-off:** c = 1 forces whole blocks through one cell and "limits the expressivity significantly". c ∈ {8, 16, 32} worked, at c× the cost.
- **Our addition:** in the first block, positions before the first summary cell have **no** summary cells to read, so their second-head set would be empty and the softmax undefined (our first run gave NaN). We also let every position attend to **itself** in that head.

### 3.5 Which pattern when?
- **Strided** suits data with **periodic structure** aligned with the stride (images: the stride is one row; music).
- **For text,** "the network can fail to properly route information with the strided pattern", because position i − l has no special meaning in text. **Fixed** works better there (Table 2: enwik8 strided 1.13 vs fixed 0.99).

### 3.6 The cost
- **Per position:** about l + n/l pairs; at l = √n that's 2√n, so **O(n√n)** in total.
- **Our counts:**

  | n | dense pairs | strided pairs | saving |
  |---|---|---|---|
  | 1,024 | 524,800 | 48,144 | 10.9× |
  | 4,096 | 8.39M | 389,152 | 21.6× |
  | 16,384 | 134M | ~4.19M | ~32× |

  The saving grows like √n/2.

---

## 4. Putting the heads together (Section 5.1)

**Three ways:**
- **Interleave (Eq. 6):** block r uses pattern A^(r mod p). The heads alternate across layers.
- **Merged (Eq. 7):** one head attends to the **union** of the patterns. That's slightly more work, but "a constant factor".
- **Multi-head (Eq. 8):** split the n_h heads among the patterns, computed in parallel and concatenated.

The enwik8 model used merged heads.

---

## 5. Training very deep networks (Section 5.2)

### 5.1 Pre-activation residual blocks
```
H₀ = embed(X, W_e)
H_k = H_{k−1} + resblock(H_{k−1})                                    (Eq. 10)
a(H) = dropout(attention(norm(H)))                                   (Eq. 12)
b(H) = dropout(ff(norm(H + a(H))))                                   (Eq. 13)
resblock(H) = a(H) + b(H)                                            (Eq. 14)
y = softmax(norm(H_N) W_out)                                         (Eq. 11)
```
- **What changed from Paper 034:** the layer norm is applied **before** each sub-layer ("pre-LN", as in pre-activation ResNets), and the residual stream itself is never normalized.
- **Why it helps depth:** H_N = H₀ + the sum of all a's and b's, so **every block gets gradient directly from the output** (the same argument as Paper 016's unrolled sum). This is why 128-layer models trained without auxiliary losses.
- **This is now the default in GPT-2/3 and most LLMs.**

### 5.2 Initialization that doesn't grow with depth
- **The problem:** H_N sums 2N block outputs. If each has variance σ², the stream's variance grows like 2N·σ².
- **The fix:** scale the initial W₂ (the second FFN matrix) and W_p (the attention output) by **1/√(2N)**. The sum then keeps roughly the same scale for any depth.
- **Other inits:**
  - weights N(0, 0.125/√fan-in);
  - embeddings N(0, 0.125/√d);
  - **the output logits start at 0,** so the first prediction is exactly uniform (our test checks this).

### 5.3 GELU
- **The activation:** ff(x) = W₂ f(W₁x + b₁) + b₂, where f is GELU, approximated as **x·σ(1.702x)** (our test: within 0.03 of exact GELU).
- **Width:** the hidden layer is 4× wide (or 2×, "half-size").

### 5.4 Positions for different data (Section 5.3)
- **The method:** learned embeddings are added for **each coordinate** of the position (Eq. 15).
  - **images:** (row, column, channel);
  - **text and audio:** the (row, column) of the position in a matrix whose width is the stride, i.e. the "attention embedding".
- **Why:** the model is told where it sits relative to the attention pattern.

### 5.5 Recomputation (gradient checkpointing, Section 5.4)
- **The idea:** store only each block's **input**, and recompute the attention and feed-forward internals during the backward pass.
- **Memory:** it no longer grows with the stored n² attention matrices of every layer. "Using recomputation alone", dense attention trained on length 16,384 with hundreds of layers.
- **The cost:** about one extra forward pass.
- **Our test:** checkpointed and normal training give **identical gradients**.

### 5.6 Kernels and precision (Sections 5.5–5.6)
- **Block-sparse GPU kernels:**
  - **local windows** are computed directly on blocks;
  - **strided attention** is "computed by transposing the matrix and computing a local window";
  - the upper triangle is never computed.
- **Our CPU versions of both ideas:** block-local attention (reshape the sequence into blocks) and strided column attention (reshape into an (n/l) × l grid, transpose, attend down columns). Both equal the masked computation exactly, without ever forming the n × n matrix.
- **Mixed precision:** fp16 activations with dynamic loss scaling. Queries and keys are cast to fp32 when sampling, to avoid overflow.

---

## 6. Results (Section 7)

**Training:** Adam, 5,000-step linear warm-up, then cosine decay; gradient clipping 1.0; weight decay 0.01.

### Table 1: density modelling (bits per byte / per dimension)
| Data | Previous best | Sparse Transformer |
|---|---|---|
| CIFAR-10 | 2.85 (PixelSNAIL) | **2.80** (59M, strided, 128 layers) |
| enwik8 | 0.99 (Transformer-XL 277M); 1.03 at similar size | **0.99** (95M, fixed, 30 layers, context 12,288) |
| ImageNet 64×64 | 3.52 (SPN 150M) | **3.44** (152M, strided, 48 layers) |
| classical music, 5 s at 12 kHz | — | 1.97 (152M) |

### Table 2: sparse vs dense
| Data | Pattern | bits | time/iter |
|---|---|---|---|
| enwik8 (context 12,288) | dense | 1.00 | 1.31 |
| | **fixed** | **0.99** | 0.55 |
| | strided | 1.13 | **0.35** |
| CIFAR-10 (3,072) | dense | 2.82 | 0.54 |
| | fixed | 2.85 | 0.47 |
| | **strided** | **2.80** | **0.38** |

- **Sparse is faster and, surprisingly, sometimes better than dense.**
- **The authors' explanation:** either a useful inductive bias, or an optimization problem with full attention.

### Tables 3–4: long contexts
- **Table 3 (more context at test time keeps helping on enwik8):**

  | minimum context | bits per byte |
  |---|---|
  | 6,144 | 0.9952 |
  | 12,160 | 0.9908 |

  The model really uses long-range information.
- **Table 4 (audio, the largest model that fits in 16 GB):**

  | sequence length | parameters | bits per byte |
  |---|---|---|
  | 65k | 152M | 1.97 |
  | 262k | 25M | 2.17 |
  | **1,048,576** | 3M | 2.99 |

  4× the length costs about 8× the capacity. **Self-attention over a million timesteps** is possible in principle.

---

## 7. Why it matters

- **The architecture behind GPT-3** (Paper 050). GPT-3 "alternates dense and locally banded sparse attention patterns ... similar to the Sparse Transformer".
- **It opened the "efficient attention" line:** Longformer, BigBird (local + global + random patterns), and later FlashAttention (Paper 079), which keeps exact dense attention but computes it blockwise.
- **Pre-LN residuals and 1/√(2N) init** became standard in GPT-2 and onward.

---

## 8. What our code found

**Scale note:**
- At your request, nothing heavy was run on this laptop.
- `experiments.py` reproduces:
  - Table 2 on enwik8 (E1) and CIFAR-10 (E2);
  - Table 3's context study (E3);
  - time and memory vs length with the blocked kernels (E4).

**Checked (tests and demo, ~5 seconds):**
- **The patterns match the definitions** on hand examples and never look at the future.
- **Validity:**
  - strided connects every earlier position in two steps, in either head order;
  - **fixed only in the order "block, then summary"**, which is why the heads' order matters.
- **The cost:** strided needs 10.9×, 21.6× and ~32× fewer pairs than dense at n = 1,024, 4,096 and 16,384.
- **Efficient kernels:** block-local and strided-column attention equal the masked computations exactly, and run faster (0.7 vs 3.1 ms at n = 1,024 on CPU).
- **Head modes:** interleave / merged / multi-head give the right masks.
- **The architecture:**
  - the residual block equals Eqs. 12–14;
  - the logits start uniform;
  - W₂'s init is scaled by 1/√(2N);
  - GELU's approximation is within 0.03;
  - **recomputation gives identical gradients**.
- **The pattern–data match, shown on a periodic task** (every row of 8 repeats the row above; 2 layers, 200 updates):

  | pattern | loss |
  |---|---|
  | dense | **0.002** |
  | strided | **0.002** |
  | fixed (c = 2) | 1.075 |

  The guessing level is 2.08. **Strided reads "same column, one row up" directly; fixed must squeeze a row through 2 summary cells.** This mirrors the paper's finding in reverse (strided won on images, fixed on text).

**Honest notes:**
- **Masks vs kernels:** our model applies the patterns as **masks** on a dense score matrix, which is exact but saves no time. The speed-up needs blocked kernels, which we implement and test separately.
- **The fixed pattern's first block:** we add self-attention to its second head (section 3.4).
- **Position embeddings for images:** with a stride of one image row, our 2-D "attention embedding" (row, position in row) carries the same information as the paper's (row, column, channel) data embedding.

---

## 9. Check yourself

1. How many query–key pairs does dense causal attention compute at n = 4,096? And strided with l = 64? (8,390,656 vs ~2·4096·64 ≈ 524k; the exact count is 389,152.)
2. Write A^(1) and A^(2) for the strided pattern. Show the two-hop path from j to i.
3. Why is the fixed pattern valid only in the order "block, then summary"?
4. When should you use strided vs fixed? Why did strided fail on text?
5. Write Eqs. 12–14. Why does pre-LN help very deep networks?
6. Why scale W₂ and W_p by 1/√(2N)?
7. What does recomputation trade, and why does it help attention so much?
8. How do you compute strided attention without the n × n matrix?
