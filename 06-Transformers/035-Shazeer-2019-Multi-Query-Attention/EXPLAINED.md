# Shazeer (2019), explained from scratch

**Paper:** *Fast Transformer Decoding: One Write-Head is All You Need*
**Author:** Noam Shazeer (Google)
**Published:** arXiv:1911.02150, 2019

Read Paper 034 (the Transformer) first. This short paper is about **inference speed**, and its idea (share keys and values across heads) is used in PaLM, Falcon, LLaMA-2/3 (as "grouped-query attention"), Gemini and most modern LLMs. This guide:
- explains why generating text is limited by **memory traffic**, not arithmetic, with the paper's own cost model;
- explains how sharing one key/value head fixes it;
- derives the paper's "widen the FFN" numbers exactly.

---

## 0. The whole idea in one line

> **When a Transformer generates text one token at a time, each step must re-read the stored keys and values of every earlier token, for every head. Let all heads share ONE set of keys and values (keep separate queries). The stored "KV cache" shrinks h-fold, decoding gets 10× faster, and quality barely changes.**

---

## 1. Background: attention in einsum form (Section 2)

**The notation.** The paper writes attention with `einsum`, which names each tensor's dimensions:
- b = batch, n = query positions, m = memory positions;
- d = model width, h = heads, k = key size, v = value size.

**Multi-head attention (MHA):**
```
Q = einsum("bnd,hdk->bhnk", X, P_q)          each head projects the queries
K = einsum("bmd,hdk->bhmk", M, P_k)          each head has its OWN keys
V = einsum("bmd,hdv->bhmv", M, P_v)          … and its own values
logits = einsum("bhnk,bhmk->bhnm", Q, K)
O = einsum("bhnm,bhmv->bhnv", softmax(logits + mask), V)
Y = einsum("bhnv,hdv->bnd", O, P_o)
```
- **Reading einsum:** "bnd,hdk->bhnk" means multiply and sum over the index d, which appears in the inputs but not in the output.
- **The 1/√k scale** of Paper 034 is omitted here; it can be folded into P_q.

---

## 2. Why incremental decoding is slow (Sections 2.3–2.4)

### 2.1 Training is fine
- **All n positions are known in training,** so the whole sequence is processed in parallel (batched).
- **The cost (with k = v = d/h and m = n ≤ d):**
  - arithmetic Θ(b·n·d²);
  - memory accessed Θ(b·n·d + b·h·n² + d²).
- **Their ratio is O(1/k + 1/(bn)),** small, so the hardware stays busy computing.

### 2.2 Generation re-reads the whole cache every step
- **One position at a time:** when generating, token t+1 depends on token t, so positions are processed **one at a time**.
- **The KV cache:** each step computes one new key and value per head and appends them. Then it must **read all the stored keys and values**, b·h·m·k numbers for K plus the same for V, to compute attention for the single new query.
- **Over n steps** (Section 2.4.1):
  ```
  arithmetic: Θ(b·n·d²)              memory: Θ(b·n²·d + n·d²)        ratio: Θ(n/d + 1/b)
  ```
- **Why a ratio near 1 is bad:**
  - Modern accelerators perform roughly **100× more arithmetic per second than they can move numbers from memory**. So when the ratio is near 1, they spend most of their time **waiting for memory**.
  - The 1/b term is easy to fix: use a bigger batch.
  - **The n/d term is the hard one.** It comes from re-reading the b·n²·d-sized KV cache. Long sequences make it worse.
- **Our demo's ratios (d = 1024):**

  | n | b | multi-head | multi-query |
  |---|---|---|---|
  | 128 | 1 | 1.125 | 1.017 |
  | 128 | 128 | 0.133 | 0.024 |
  | 1024 | 128 | **1.008** | **0.134** |

---

## 3. Multi-query attention (Section 3)

**Keep h query heads, but give the keys and values no heads dimension:**
```
Q = einsum("bnd,hdk->bhnk", X, P_q)          h heads of queries (unchanged)
K = einsum("bmd,dk->bmk",   M, P_k)          ONE key head    (P_k is d×k, not h×d×k)
V = einsum("bmd,dv->bmv",   M, P_v)          ONE value head
logits = einsum("bhnk,bmk->bhnm", Q, K)      every query head reads the same K …
O = einsum("bhnm,bmv->bhnv", weights, V)     … and the same V
```
"Remove the letter h from the einsum equations where it represents the heads dimension of K, V, P_k, P_v."

**What changes:**
- **The cache shrinks h-fold:** b·m·k numbers instead of b·h·m·k.
- **The memory ratio** becomes
  ```
  Θ(1/d + n/(d·h) + 1/b)
  ```
  The troublesome n/d is **divided by h**.
- **The heads still differ:** each head still has its own **queries**, so it can still look at different positions. Only *what* it can read (the keys and values) is shared.

**MQA is a special case of MHA:**
- multi-query = multi-head with all h key projections tied to one matrix, and likewise for values;
- our test checks that the two einsum codes agree when P_k is repeated across heads.

**A later generalization: grouped-query attention (GQA, Ainslie et al. 2023):**
- use g key/value heads, each shared by h/g query heads;
- g = h is MHA and g = 1 is MQA;
- LLaMA-2-70B uses g = 8 with h = 64.

**Our code implements all three with one class.**

### 3.1 How big is the cache? (the paper's translation model)
- **The setup:** b = 1024 sequences, n = 128, h = 8, k = 128, 6 decoder layers.
- **The cache, counting K and V:**
  ```
  multi-head : 2·6·1024·128·8·128 = 1.61 billion numbers = 3.0 GiB (16-bit)
  multi-query: 2·6·1024·128·1·128 = 0.20 billion numbers = 0.38 GiB
  ```
- **Every one of the 128 decoding steps re-reads this cache.**

---

## 4. Experiments (Section 4)

### 4.1 Setup
- **Translation:** WMT14 En–De. An encoder–decoder Transformer with 6 layers, d_model = 1024, d_ff = 4096, h = 8, d_k = d_v = 128, learned positions, and tied embeddings: **211M parameters**.
  - 100k steps, batches of 128 × 256-token examples, on a 32-core TPUv3.
  - **Every attention layer** (encoder self, decoder self, encoder–decoder) becomes multi-query.
- **Fair comparison:**
  - each attention layer **saves** 2·d·k·(h − 1) parameters in P_k and P_v;
  - **the FFN is widened** to win them back.

  **Derivation (ours, exact):**
  - an encoder–decoder has 6 + 2·6 = 18 attention layers, saving 18 × 2·1024·128·7 = 33.0M;
  - spread over 12 FFNs, each 2·1024 wider per unit of width, that's 33.0M/(12·2048) = 1344 extra units;
  - so **d_ff = 4096 + 1344 = 5440**, exactly the paper's number.
  - The language model (6 layers, d_ff 8192) gives **9088** the same way.
- **The "simpler alternatives":** shrink the K/V cache by using fewer heads or smaller heads instead, at the same total width ((h, d_k) = (1, 128), (2, 64), (4, 32), (8, 16)), again with widened FFNs.
- **"Local" variants:** decoder self-attention sees only the current and the previous 31 positions, which is orthogonal to MQA.
- **Language modelling:** a transformer decoder on the Billion-Word benchmark (192M parameters).

### 4.2 Table 1: WMT14 En–De
| Attention | h | d_k, d_v | d_ff | ln(PPL) dev | BLEU dev | BLEU test (beam 1 / 4) |
|---|---|---|---|---|---|---|
| multi-head | 8 | 128 | 4096 | 1.424 | 26.7 | 27.7 / 28.4 |
| **multi-query** | 8 | 128 | 5440 | 1.439 | 26.5 | 27.5 / **28.5** |
| multi-head local | 8 | 128 | 4096 | 1.427 | 26.6 | 27.5 / 28.3 |
| multi-query local | 8 | 128 | 5440 | 1.437 | 26.5 | 27.6 / 28.2 |
| multi-head | 1 | 128 | 6784 | 1.518 | 25.8 | |
| multi-head | 2 | 64 | 6784 | 1.480 | 26.2 | 26.8 / 27.9 |
| multi-head | 4 | 32 | 6784 | 1.488 | 26.1 | |
| multi-head | 8 | 16 | 6784 | 1.513 | 25.8 | |

- **MQA is slightly worse on the dev set** (ln PPL 1.439 vs 1.424; BLEU 26.5 vs 26.7) but **far better than the alternatives** that shrink the cache the naive way.
- **On test with beam 4, MQA even scored highest (28.5).**

### 4.3 Table 2: speed (TPUv2 microseconds per output token, sequence length 128)
| Attention | Training | Inference (enc + dec) | Beam-4 (enc + dec) |
|---|---|---|---|
| multi-head | 13.2 | 1.7 + **46** | 2.0 + 203 |
| **multi-query** | 13.0 | 1.5 + **3.8** | 1.6 + 32 |
| multi-head local | 13.2 | 1.7 + 23 | 1.9 + 47 |
| multi-query local | 13.0 | 1.5 + 3.3 | 1.6 + 16 |

- **Training speed is unchanged** (it's arithmetic-bound).
- **Decoding is 12× faster** (46 → 3.8 µs/token), and 6× faster with beam search.

### 4.4 Table 3: Billion-Word language model (dev perplexity)
| Attention | h | d_k | d_ff | dev PPL |
|---|---|---|---|---|
| multi-head | 8 | 128 | 8192 | **29.9** |
| multi-query | 8 | 128 | 9088 | 30.2 |
| multi-head | 1 | 128 | 9984 | 31.2 |
| multi-head | 2 | 64 | 9984 | 31.1 |
| multi-head | 4 | 32 | 9984 | 31.0 |
| multi-head | 8 | 16 | 9984 | 30.9 |

The same picture: MQA costs a little; the naive alternatives cost more.

---

## 5. Why it matters

- **Today's LLMs are served under memory-bandwidth limits.** The KV cache is often the largest object in GPU memory, and it grows with batch size × context length.
- **MQA and GQA** are standard in PaLM, Falcon, LLaMA-2/3, Mistral and Gemini.
- **The same analysis** motivates the KV-cache work of Paper 080 (scaling Transformer inference), paged attention, and cache quantization.

---

## 6. What our code found

**Scale note:**
- At your request, nothing heavy was run on this laptop.
- `experiments.py` reproduces, on PTB words:
  - E1: Table 3/1's parameter-matched variants;
  - E2: Table 2's training and decoding speed, with and without a 32-position window;
  - E3: decoding time vs batch and length.

**Checked (tests and demo, ~9 seconds):**
- **The paper's einsum code, implemented literally:**
  - batched multi-head = a loop over heads;
  - **multi-query = multi-head with tied key/value heads**;
  - the incremental (cached) versions equal the batched causal computation step by step;
  - the multi-query cache has **one** head.
- **Grouped attention** with g = 4, 2, 1 equals the paper's multi-head einsum with the right repeated K/V projections.
- **KV-cache generation equals full recomputation** token for token, and the multi-head cache is exactly h× the multi-query one.
- **The local window:** a position attends to exactly the current and previous (w − 1) positions.
- **The FFN widths: 5440 and 9088, exactly the paper's.**
- **Decoding speed on this CPU** (64 sequences, 120 tokens, d = 256, 4 layers):

  | attention | ms per step | cache size |
  |---|---|---|
  | multi-head | 6.0 | 16.8M numbers |
  | grouped (2 K/V heads) | 2.8 | 4.2M |
  | **multi-query** | **2.2** | **2.1M** |
  | multi-head **without** a KV cache | 47.5 | — |

  Caching gives 8×, and multi-query another 2.8×.
- **Quality:** parameter-matched multi-head and multi-query models both learn a copy task (loss 0.0009 vs 0.0013 after 300 updates).

---

## 7. Check yourself

1. Why can training process all positions in parallel while generation can't?
2. Write the size of the KV cache for b sequences, n tokens, L layers, h heads, key size k. What does multi-query change?
3. Derive the memory/compute ratio Θ(n/d + 1/b) for multi-head incremental decoding. Which term does a big batch fix?
4. Show that multi-query attention is multi-head attention with tied key and value projections.
5. Derive the widened FFN size 5440 for the translation model.
6. Why is multi-query better than simply using 1 head, or tiny heads, at the same cache size?
7. What is grouped-query attention, and how does it interpolate between MHA and MQA?
