# FlashAttention, explained simply

**Paper:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford, University at Buffalo), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022.

**In one sentence:** standard attention writes the huge N×N score matrix to the GPU's slow main memory and reads it back several times. FlashAttention computes **exactly the same result** tile by tile inside the fast on-chip memory, never storing the N×N matrix. It is up to 3× faster and up to 20× more memory-efficient, which enables much longer contexts.

---

## 1. Why attention is slow: memory, not math

### The GPU memory hierarchy (A100)

| Memory | Size | Bandwidth |
|---|---|---|
| HBM (main GPU memory) | 40–80 GB | 1.5–2.0 TB/s |
| SRAM (on-chip) | 192 KB per streaming multiprocessor × 108 | ~19 TB/s, **10× faster** |

- **Many operations are memory-bound:** their time is spent moving data between HBM and the chip, not computing. Softmax, masking, dropout and element-wise operations all fall into this category.

### Standard attention (Algorithm 0)
```
S = Q Kᵀ        (N×N, written to HBM)
P = softmax(S)  (read S, write N×N P)
O = P V         (read P, V; write O)
```
- **The cost:** Θ(Nd + N²) HBM reads and writes, and **O(N²) memory**.
- **For GPT-2 medium** (Figure 2): 40.3 GB of HBM traffic and 41.7 ms. The FLOPs (66.6 GFLOPs) are not the problem.

---

## 2. The trick: tiling plus the online softmax
- **The obstacle:** softmax needs a whole row of scores (max and sum) before normalising. That seems to require the whole N×N matrix.
- **It doesn't:** keep two running numbers per row, the max m and the sum of exponentials ℓ. When a new block x⁽²⁾ arrives after x⁽¹⁾:
```
m = max(m₁, m₂)
ℓ = e^(m₁−m) ℓ₁ + e^(m₂−m) ℓ₂
```
- **The outputs can be rescaled too.** A running output O built with the old statistics is corrected by e^(m_old − m_new) · ℓ_old / ℓ_new when new blocks arrive.

### Worked example (from our demo)
- **Setup:** x = [1, 3, 2, 5] in blocks of 2.
- **Block [1, 3]:** m = 3, ℓ = e^(1−3) + e^(3−3) = 0.1353 + 1 = 1.1353.
- **Block [2, 5]:** block max 5, block sum e^(2−5) + 1 = 1.0498.
  - New m = 5, and ℓ = e^(3−5)·1.1353 + e^(5−5)·1.0498 = 0.1536 + 1.0498 = **1.2034**.
- **Result:** softmax = e^(x−5)/1.2034 = [0.0152, 0.1125, 0.0414, 0.831], exactly torch's softmax.

### Algorithm 1 (forward)
- **Block sizes:** B_c = ⌈M/4d⌉ key/value rows and B_r = min(B_c, d) query rows, so the working set fits in SRAM of size M.
- **The loops:**
  - **outer loop** over K, V blocks: load K_j, V_j into SRAM;
  - **inner loop** over Q blocks: load Q_i, O_i, m_i, ℓ_i; compute S_ij = Q_i K_jᵀ **on chip**;
  - update m and ℓ, rescale and accumulate O_i, then write O_i, m_i, ℓ_i back.
- **Everything is fused into one GPU kernel.** S and P exist only block by block in SRAM.

### Backward pass with recomputation
- **What is saved:** only O and the row statistics (m, ℓ), not P.
- **What happens:** the backward pass **recomputes** each S_ij and P_ij block from Q, K and (m, ℓ). It uses D_i = rowsum(dO_i ∘ O_i) for the softmax Jacobian.
- **The trade:** more FLOPs (75.2 vs 66.6 GFLOPs), but far fewer HBM accesses (4.4 vs 40.3 GB) and **7.3 vs 41.7 ms**.

---

## 3. IO complexity (Theorem 2)
- **Standard attention:** Θ(Nd + N²) HBM accesses.
- **FlashAttention:** Θ(N²d²/M).
- **Why:** each K, V block is read once, and Q and O are re-read once per K/V block, i.e. N/B_c = Θ(Nd/M) times. So the total is Θ(Nd · Nd/M).
- **How big the saving is:** with d = 64–128 and M ≈ 100 KB, d²/M is much less than 1, so many times fewer accesses.
- **Proposition 3:** no exact attention algorithm can asymptotically beat this for all SRAM sizes.

### Block-sparse FlashAttention
- **Skip zero blocks of a block mask:** Θ(Nd + N²d²s/M) accesses, where s is the fraction of non-zero blocks (Proposition 4).
- **Speed:** 2–4× faster than dense FlashAttention, and it scales to sequence length 64K.

---

## 4. Results
- **Training speed:**
  - BERT-large (sequence 512): **15% faster** than the MLPerf 1.1 record;
  - GPT-2 (sequence 1K): **3× faster** than HuggingFace and Megatron;
  - Long Range Arena (1K–4K): **2.4×** faster.
- **Longer context, better models:**
  - GPT-2 with a 4K context trains faster than Megatron with 1K, and reaches **0.7 lower perplexity**;
  - long-document classification improves by 6.4 points;
  - the **first better-than-chance Transformer on Path-X** (sequence 16K, 61.4%) and on Path-256 (64K, 63.1%, block-sparse).
- **Memory:** linear in N, **up to 20× less** than exact baselines.

---

## 5. Why it matters
- **FlashAttention (and FlashAttention-2/3) is now the default attention kernel** in PyTorch (`scaled_dot_product_attention`), vLLM and most LLM training stacks. Long-context models (32K–1M tokens) depend on it.
- **It popularised "IO-aware" algorithm design:** count memory traffic, not just FLOPs. That is also the theme of the next papers on inference systems.

---

## 6. What our code found
Our "HBM" is a counter of elements moved between the big arrays and the working blocks, so the IO complexity is measured exactly on a CPU.

- **Exact:** the forward pass matches standard attention (3e-7). The backward pass, with recomputed scores and only O, m, ℓ saved, matches autograd (3e-7).
- **N = 512, d = 32, M = 8192:**
  - forward traffic: standard 1,114,112 elements vs **442,368**;
  - backward traffic: 2,228,224 vs **635,392**.

**Traffic vs N and SRAM size M** (d = 32):

| N | M = 2,048 | M = 8,192 | M = 32,768 |
|---|---|---|---|
| 256 | 0.7× better | 2.5× better | 7.0× better |
| 512 | 0.7× | 2.5× | 8.2× |
| 1024 | 0.7× | 2.5× | **9.1×** |

- **The scaling matches Theorem 2:** doubling N multiplies FlashAttention's traffic by ~3.5 (≈ N²), and 4× more SRAM divides it by ~3.6 (≈ 1/M).
- **With a tiny SRAM** (M = 2,048, blocks of 16 × 32) FlashAttention is **worse**: d²/M must be small, which real GPUs provide.

**Memory:** at N = 65,536 and d = 64, standard attention stores S and P (32 GiB in fp32); FlashAttention stores O, m and ℓ (16.5 MiB).

**Block-sparse:** keeping 52.5% / 31.2% / 15.6% of blocks gives 54.4% / 33.9% / 18.9% of the dense traffic, proportional as Proposition 4 says.

**Honest timing note:** on a CPU, our Python-loop FlashAttention (8.3 ms) is slower than one PyTorch call (0.6 ms) at N = 1,024. The speed-up needs a fused GPU kernel. What we reproduce is the memory traffic and the footprint.

**`experiments.py`:**
- **E1:** fits the exponents of N, d and M;
- **E2:** fwd + bwd traffic and recomputation FLOPs;
- **E3:** block size;
- **E4:** local + global block-sparse patterns vs a masked dense reference;
- **E5:** real GPU timing and memory of PyTorch's flash vs math SDPA backends.
- None were run here.

---

## 7. Check yourself

1. Why is standard attention memory-bound rather than compute-bound?
<details><summary>Answer</summary>It reads and writes the N×N matrices S and P to HBM, and the softmax does very little arithmetic per element it moves. The time goes into memory traffic, which is ~10× slower than on-chip SRAM.</details>

2. Combine block A with (m₁ = 2, ℓ₁ = 1.5) and block B with (m₂ = 4, ℓ₂ = 1.2).
<details><summary>Answer</summary>m = 4; ℓ = e^(2−4)·1.5 + e^(0)·1.2 = 0.203 + 1.2 = 1.403.</details>

3. Why does FlashAttention do more FLOPs but run faster?
<details><summary>Answer</summary>The backward pass recomputes S and P from Q, K and the saved statistics instead of reading a stored N×N P from HBM. Extra arithmetic is cheap; avoiding HBM traffic saves much more time (40.3 → 4.4 GB, 41.7 → 7.3 ms).</details>

4. Is FlashAttention an approximation?
<details><summary>Answer</summary>No. It computes exactly the same output and gradients (up to floating-point rounding); only the order and placement of the computation change.</details>

5. Derive the Θ(N²d²/M) HBM accesses roughly.
<details><summary>Answer</summary>K and V are read once (Nd). For each of the N/B_c ≈ 4Nd/M key blocks, all of Q and O (Nd each) are read and written, so Nd · Nd/M = N²d²/M dominates.</details>

6. When would FlashAttention's tiling NOT reduce memory traffic?
<details><summary>Answer</summary>When SRAM is too small relative to d² (d²/M not ≪ 1): the blocks are tiny and Q/O are re-read too many times. Our toy with M = 2,048 and d = 32 shows 0.7× (worse).</details>

7. What does FlashAttention save for the backward pass?
<details><summary>Answer</summary>The output O and per-row softmax statistics (m, ℓ), which is O(N) extra, instead of the N×N attention matrix.</details>

8. How does block-sparse FlashAttention change the IO cost?
<details><summary>Answer</summary>Zero blocks of the mask are skipped, so the N²d²/M term is multiplied by the fraction s of non-zero blocks: Θ(Nd + N²d²s/M).</details>

9. Name two capabilities FlashAttention enabled.
<details><summary>Answer</summary>Longer context for the same compute (GPT-2 at 4K context with 0.7 lower perplexity), and the first better-than-chance Transformers on Path-X (16K) and Path-256 (64K).</details>

10. Why didn't our CPU implementation run faster than standard attention?
<details><summary>Answer</summary>Its benefit is fewer slow-memory transfers in a fused GPU kernel. On a CPU, with big caches, a single optimised matmul is fast, and our Python loops add overhead. We measured the traffic reduction instead.</details>
