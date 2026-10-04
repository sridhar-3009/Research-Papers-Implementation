# Efficiently Scaling Transformer Inference, explained simply

**Paper:** Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean (Google), *Efficiently Scaling Transformer Inference*, MLSys 2023 (arXiv 2022).

**In one sentence:** serving a 540B-parameter model means splitting it over many chips. This paper builds a simple cost model (compute, memory traffic, chip-to-chip communication) and uses it to choose **how to partition** the model for low latency or high throughput, plus a **multiquery-attention layout** that makes long contexts fit. The result: PaLM 540B generates a token every **29 ms**, and processes prompts at **76% of peak FLOPS**.

---

## 1. The vocabulary of inference

### Two phases
- **Prefill:** process the whole prompt at once. Many tokens are computed in parallel, so it is **compute-bound**.
- **Decode:** generate one token per sequence per step, sequentially. Each step must **reload every weight and the whole KV cache** for only `batch` new tokens, so it is **memory-bound**.

### KV cache
The keys and values of all previous tokens are stored so they aren't recomputed. Its size is:
```
2 (K and V) × layers × kv_heads × d_head × bytes   per token
```
- **It is unique per sequence**, so it grows with batch × context.
- **Example (multihead 540B variant, 48 heads × d_head 128, bf16):** batch 512 × context 2048 needs **≈ 3 TB**, three times the weights.

### Costs per forward pass
| Cost | Formula |
|---|---|
| Compute time | 2·N·tokens / (chips × peak FLOPS); each parameter does one multiply-add per token |
| Memory time | (weight bytes + KV bytes) / (chips × HBM bandwidth) |
| Communication time | depends on how the matrices are split (below) |

### MFU (model FLOPS utilisation)
- **Definition:** achieved throughput ÷ peak throughput.
- **Low latency usually means small batches,** which means low MFU and a high cost per token. This is the core trade-off.

**Hardware: TPU v4** has 275 TFLOPS (bf16), 32 GiB of HBM at 1,200 GB/s, and a 270 GB/s interconnect in a 3D torus.

---

## 2. Partitioning the feed-forward layer
The feed-forward layer is W_in (E×F) then W_out (F×E), with F ≈ 4E.

| Layout | How it splits | Communication per layer | Scales with chips |
|---|---|---|---|
| **1D weight-stationary** (Megatron) | each chip holds a slice of F; activations are all-gathered in and reduce-scattered out | 2·BLE / bandwidth | **no**: constant |
| **2D weight-stationary** | split over both E (X ways) and F (YZ ways); alternate the aggregation axis between the two matmuls | 8·BLE / (√chips · bandwidth), with X = ½√n, YZ = 2√n | yes, ∝ 1/√chips |
| **Weight-gathered** (X / XY / XYZ) | keep **activations** in place (split over the batch) and all-gather the **weights** instead | 4E·√(BLF) / (√chips · bandwidth) | grows only as √tokens |

- **The choice depends on batch size.** Moving weights costs the same at any batch size, while moving activations grows with it. So 2D weight-stationary is best for small batches (decode), and weight-gathered for large ones (big-batch prefill), as in Figure 3.
- **Notation:** B = batch, L = sequence length, so BL = tokens in the batch.

---

## 3. Multiquery attention needs the right layout
- **Multiquery attention** (as in PaLM) uses many query heads but **one** shared K/V head, so the KV cache is **n_heads×** smaller.
- **The trap:** if attention is partitioned over **heads** (as usual), that single K/V head must be **replicated on every chip**. The savings vanish, and it even does worse than multihead.
- **The fix:** partition the attention (and the KV cache) over the **batch**. Each chip then holds the caches of batch/chips sequences: n_chips× less KV memory and memory time per chip, at the cost of an all-to-all to re-shard activations.

**Table 1, maximum context for PaLM 540B on 64 chips** (30% of HBM for the KV cache):

| Variant | Batch 128 | Batch 512 |
|---|---|---|
| Multihead (d_head 128) | 1,320 | 330 |
| Baseline multiquery (head-sharded) | 660 | 165 |
| **Optimized multiquery (batch-sharded)** | **43,000** | **10,700** |

That is up to **32–64× longer** contexts.

### Worked example (baseline multiquery, batch 128)
- **KV per token:** 2 × 118 layers × 1 head × 256 × 2 bytes = 120,832 bytes.
- **Each chip has** 0.3 × 32 GiB = 10.3 GB.
- **Batch × context** = 10.3e9 / 120,832 ≈ 85,300, so context ≈ 85,300 / 128 = **666** (paper: 660).
- **Batch-sharded,** each chip stores only 128/64 = 2 sequences, so context ≈ 64 × 666 ≈ **42,600** (paper: 43,000).

---

## 4. Other techniques
- **int8 weight quantization** halves weight-loading time. It helps **decode latency** most: 28.5 vs 36.9 ms/token at batch 64 on PaLM 540B.
- **Parallel attention / feed-forward layers** (a PaLM feature) reduce communication.
- **Low-level optimisations:** looped collective einsums (overlapping communication with compute) and reordered operations.
- **Mixing batch sizes:** batch-1 prefill (lowest latency) with batch-64 decode, which costs little extra latency and gives much better MFU.

---

## 5. Results (PaLM 540B, 64 TPU v4 chips, context 2,048; Table 2)

| Scenario | Batch | Weights | Layout | MFU | Latency |
|---|---|---|---|---|---|
| Low-latency prefill | 1 | int8 | 2D WS | 43% | 0.29 s (2,048 tokens) |
| Low-latency decode | 64 | int8 | 2D WS | 14% | 1.82 s (64 tokens, ~28 ms/token) |
| High-throughput prefill | 512 | bf16 | weight-gathered | **76%** | 85.2 s |
| High-throughput decode | 512 | bf16 | 2D WS | 33% | 6.0 s |

- **An interactive chatbot turn** (read 64 new tokens, use 1,920 cached tokens, write 64 tokens) takes **1.9 s**.
- **Offline processing** runs at 73% FLOPS efficiency.
- **Comparison:** these are better latency–MFU trade-offs than FasterTransformer on comparable models.

---

## 6. Why it matters
- **This is the playbook of modern LLM serving:**
  - prefill vs decode as different regimes;
  - KV-cache memory as the long-context bottleneck;
  - MQA/GQA with the right sharding;
  - weight quantization for decode latency;
  - batching for cost.
- **It led to:** grouped-query attention (GQA, Llama 2/3) and serving systems like vLLM, TensorRT-LLM and disaggregated prefill/decode.

---

## 7. What our code found
- **Table 1 reproduced** from our per-chip KV model:

| Variant | Batch 128 (ours / paper) | Batch 512 (ours / paper) |
|---|---|---|
| Multihead | 1,333 / 1,320 | 333 / 330 |
| Baseline multiquery | 666 / 660 | 167 / 165 |
| Optimized multiquery | 42,654 / 43,000 | 10,663 / 10,700 |

  The multihead KV cache at batch 512 and context 2,048 is **3.0 TB vs 1.08 TB** of bf16 weights: the paper's "3 TB, 3× the model".

**A feed-forward layer split over virtual chips** (exact to 1e-14). Bytes sent per chip, in units of BLE:

| Chips | 1D WS | 2D WS | Formula 8/√n |
|---|---|---|---|
| 16 | 1.88 | 1.38 | 2.00 |
| 64 | 1.97 | 0.84 | 1.00 |
| 256 | 1.99 | **0.46** | 0.50 |

- **1D stays near 2 × BLE whatever the chip count; 2D shrinks like 1/√chips**, matching the formulas.
- **Weight-gathered** sends a fixed 16.5 MB per chip. It beats 2D WS only at **65,536 tokens per batch** (226 MB for 2D WS), which is the Figure 3 crossover.

**Roofline model of PaLM 540B vs Table 2:**

| Scenario | Ours | Paper |
|---|---|---|
| Prefill, batch 1 | 0.19 s | 0.29 s |
| Decode, batch 64 | 0.60 s | 1.82 s |
| Prefill, batch 512 | 68.9 s | 85.2 s |
| Decode, batch 512 | 3.07 s | 6.0 s |

- **The model gets the trends right:** prefill is compute-bound with high MFU; decode is memory-bound with low MFU at small batch; int8 helps decode (16.3 → 9.3 ms per step).
- **It is a lower bound,** 1.2–3× faster than the real system, because it ignores many overheads.

**Latency vs cost:** more chips lower the latency per token but raise chip-ms per token. Larger decode batches cut the cost per token by ~50× (batch 1 → 64) for only slightly higher latency.

**`experiments.py`:**
- **E1:** max context over batch, chips and budget;
- **E2:** a simulated comm grid with all 2D splits and the winner per regime;
- **E3:** Pareto frontiers for PaLM 8B / 62B / 540B;
- **E4:** real prefill vs decode timing and KV bytes for GPT-2 (multihead) vs a multiquery model.
- None were run here.

---

## 8. Check yourself

1. Why is decoding memory-bound while prefill is compute-bound?
<details><summary>Answer</summary>Each decode step reloads all weights (and the KV cache) from HBM to produce just one token per sequence: little arithmetic per byte. Prefill uses the same weight load for thousands of prompt tokens, so arithmetic dominates.</details>

2. Compute the KV-cache bytes per token for 32 layers, 32 heads, d_head 128, bf16, multihead.
<details><summary>Answer</summary>2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5 MB per token (×4,096 tokens × 64 sequences ≈ 137 GB).</details>

3. Why does 1D weight-stationary stop scaling with more chips?
<details><summary>Answer</summary>Every chip must all-gather the full activations (BLE) and reduce-scatter its partial outputs: about 2·BLE per chip regardless of chip count. Compute and memory time shrink, but communication doesn't.</details>

4. What does 2D weight-stationary change?
<details><summary>Answer</summary>Weights are split along both E and F, and the aggregation axis alternates between the two matmuls, so each chip only exchanges activation slices of size ~BLE/√n. Communication falls as 1/√chips.</details>

5. When is a weight-gathered layout better?
<details><summary>Answer</summary>When the batch has very many tokens (large-batch prefill): moving activations would cost more than moving the fixed-size weights. Activations stay put (split over batch) and weights are all-gathered.</details>

6. Why can multiquery attention fail to save memory?
<details><summary>Answer</summary>If attention is sharded over heads, the single shared K/V head must be replicated on every chip, so each chip still holds the whole cache. Sharding the KV cache over the batch restores the n_heads (and n_chips) savings.</details>

7. Reproduce the baseline multiquery maximum context at batch 512.
<details><summary>Answer</summary>Per-chip budget 0.3 × 32 GiB ≈ 10.3 GB; per token 2 × 118 × 256 × 2 = 120,832 bytes; 10.3e9/120,832 ≈ 85,300 tokens over a batch of 512 ≈ 167 (paper: 165).</details>

8. Why does int8 weight quantization help decode more than prefill?
<details><summary>Answer</summary>Decode time is dominated by loading weights from HBM, which int8 halves. Prefill is compute-bound, and the matmuls still run in bf16, so it barely changes.</details>

9. What is MFU, and why was it only 14% in the low-latency decode setting?
<details><summary>Answer</summary>Model FLOPS utilisation = useful FLOPs per second ÷ peak. With batch 64, each step does little compute but waits on weight and KV loads (memory-bound), so the chips are mostly idle.</details>

10. Why pair batch-1 prefill with batch-64 decode?
<details><summary>Answer</summary>Batch-1 prefill gives the lowest time to process a prompt. Decode is memory-bound, so batching 64 sequences costs almost no extra latency per step but multiplies throughput, giving much better MFU and cost.</details>
