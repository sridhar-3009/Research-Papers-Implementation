# GPTQ, explained simply

**Paper:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (IST Austria, ETH Zurich, Neural Magic), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023.

**In one sentence:** quantize an LLM's **weights** to 3–4 bits after training. Go one layer at a time and one column at a time; every time a weight is rounded, **adjust the not-yet-rounded weights** to cancel the error, using the layer's input statistics (the Hessian). A 175B model takes ~4 GPU hours, and accuracy stays near FP16, where plain rounding falls apart.

---

## 1. Why weight-only, and why it's hard
- **Generating text is memory-bound.** Each new token reads every weight once, so storing weights in 3–4 bits cuts memory **and** speeds up generation (3.25× on A100, 4.5× on A6000 with custom kernels).
- **The simplest method is round-to-nearest (RTN):** put each weight on the nearest grid level. It works at 8 bits but **collapses at 3 bits**. OPT-175B goes from 8.34 perplexity to 10.54 with 4-bit RTN, and to 7.3e3 with 3-bit RTN.
- **Second-order methods** like OBQ (Optimal Brain Quantization) are accurate but take hours for a 100M-parameter model. They are far too slow for 175B.

---

## 2. The quantization grid
GPTQ uses a per-row **asymmetric min-max grid** with 2^b levels:
```
scale = (max(w) − min(w)) / (2^b − 1),   zero = round(−min(w)/scale)
quant(w) = scale · (clamp(round(w/scale) + zero, 0, 2^b − 1) − zero)
```
**Worked example:** w = [0.9, −0.4, 0.1, 0.35] at 2 bits.
- scale = 1.3/3 = 0.433 and zero = 1, so the levels are [−0.433, 0, 0.433, 0.867].
- RTN gives [0.867, −0.433, 0, 0.433].

---

## 3. The layer-wise problem and OBQ
- **The goal:** for one linear layer with weights W (d_row × d_col) and calibration inputs X (d_col × n), find quantized Q that keeps the **outputs** close:
```
argmin_Q ‖W X − Q X‖²
```
- **It splits into rows:** each row of W is a separate least-squares problem, with Hessian **H = 2 X Xᵀ**. The Hessian is the same for every row, because it depends only on the inputs.
- **OBQ** quantizes one weight at a time and updates the remaining ones:
```
quantize w_q  →  error e = (w_q − quant(w_q)) / [H_F⁻¹]_qq
δ_F = − e · (H_F⁻¹)_{:,q}                          (Eq. 2: the remaining weights F absorb the error)
H_F⁻¹ ← H_F⁻¹ − (H_F⁻¹)_{:,q}(H_F⁻¹)_{q,:} / [H_F⁻¹]_qq   (Eq. 3: remove q from the inverse)
```
  It picks the weight with the smallest added error next (greedy order). The cost is O(d_row · d_col³).

### Worked example (our demo, Section 2)
- **Setup:** two inputs that are almost equal (x₁ ≈ x₂), weights w = [0.3, 0.3], and grid {0, 0.5}.
- **RTN:** gives [0.5, 0.5], so w₁x₁ + w₂x₂ becomes ~1.0 instead of 0.6. Layer error **0.640**.
- **OBQ step:**
  - quantize w₁ → 0.5 (error −0.2);
  - Eq. 2 moves w₂ from 0.3 to **0.107**, which rounds to 0;
  - the output is ~0.5 instead of 0.6, a layer error of **0.051**.
- **Lesson:** because the inputs are correlated, the second weight can cancel the first weight's error.

---

## 4. GPTQ's three changes
1. **Arbitrary order.**
   - Quantizing in greedy order helps little on large layers.
   - So GPTQ quantizes **all rows in the same fixed column order**. Then H_F⁻¹ is the same for every row and is updated **once per column**.
   - The cost drops to O(max(d_row·d_col², d_col³)), a factor of min(d_row, d_col) faster.
2. **Lazy batch updates.**
   - Rounding column j depends only on updates to column j, so it is wasteful to update the whole W after every column (memory-bound).
   - Instead, process **B = 128 columns** at a time, collect their errors E, and update all later columns **once per block**: W[:, later] −= E · H⁻¹[block, later].
3. **Cholesky reformulation.**
   - Repeated application of Eq. 3 accumulates numerical error at scale. H_F⁻¹ can become indefinite and wreck a layer.
   - Every row of H_F⁻¹ that is ever needed equals (up to scaling) a row of the **upper Cholesky factor** of H⁻¹. So compute it once, stably.
   - Add **1% dampening:** λ = 1% of mean(diag H) on the diagonal.

**Algorithm 1 (whole method):**
```
H⁻¹ ← Cholesky((2XXᵀ + λI)⁻¹)ᵀ
for each block of B columns:
    for each column j in the block:
        Q[:, j] = quant(W[:, j])
        E[:, j] = (W[:, j] − Q[:, j]) / H⁻¹[j, j]
        W[:, j:block_end] −= E[:, j] · H⁻¹[j, j:block_end]
    W[:, block_end:] −= E · H⁻¹[block, block_end:]
```
- **Running it on a whole model:** layers are processed **sequentially**. Each layer's X comes from passing the calibration data (128 random 2,048-token C4 segments) through the **already-quantized** earlier layers.
- **Grouping:** a separate grid per group of 128 or 1,024 columns improves accuracy further for a tiny extra cost.

---

## 5. Results

**OPT perplexity on WikiText-2 (Table 3):**

| Bits | Method | 125M | 1.3B | 13B | 66B | 175B |
|---|---|---|---|---|---|---|
| 16 | FP16 | 27.65 | 14.63 | 10.13 | 9.34 | 8.34 |
| 4 | RTN | 37.28 | 48.17 | 11.32 | 110 | 10.54 |
| 4 | **GPTQ** | 31.12 | 15.47 | 10.31 | 9.55 | **8.37** |
| 3 | RTN | 1.3e3 | 1.3e4 | 3.4e3 | 6.1e3 | 7.3e3 |
| 3 | **GPTQ** | 53.85 | 20.97 | 11.61 | 14.16 | **8.68** |

- **At 175B:** 4-bit GPTQ loses only 0.03 perplexity, while 4-bit RTN is worse than the full-precision 13B model. At 3 bits RTN collapses completely; GPTQ stays usable.
- **Larger models quantize more easily.**
- **The 175B models** (OPT and BLOOM) quantize in **~4 GPU hours**, and at 4 bits stay within ≤ 0.25 perplexity of FP16.
- **At 3 bits, OPT-175B fits on a single 80 GB A100.**
- **With grouping,** 2-bit and even ternary weights become usable.

---

## 6. Why it matters
- **GPTQ made 3–4-bit LLMs practical.** It drives much local-LLM tooling (AutoGPTQ, ExLlama, "GPTQ" model downloads).
- **"Quantize column by column and update the rest with the inverse Hessian"** is the core of many later methods.
- **Compared with AWQ** (paper 076): GPTQ compensates errors explicitly; AWQ protects important channels by scaling. **QLoRA** (077) fine-tunes on top of 4-bit weights.

---

## 7. What our code found

**One layer** (32×48 with strongly correlated inputs), layer error ‖WX − QX‖²:

| Bits | RTN | OBQ (greedy) | GPTQ (fixed order) | GPTQ / RTN |
|---|---|---|---|---|
| 4 | 11,805 | 5,773 | 5,632 | 0.48 |
| 3 | 50,546 | 25,786 | 28,622 | 0.57 |
| 2 | 274,662 | 142,133 | 158,674 | 0.58 |

- **Error compensation roughly halves the error.**
- **A fixed order is nearly as good as greedy order** (the "arbitrary order insight"); at 4 bits it was even slightly better.
- **Algorithm 1 is exact:** with Cholesky and lazy blocks it gives **identical** results to the literal column-by-column update (max difference 0.0).

**Speed:** on a 64×128 layer, OBQ takes ~330 ms and GPTQ ~2 ms (~150× here). The gap grows with layer size.

**A whole tiny LLaMA:** paper 073's model, all 14 block linear layers quantized in forward order with 126 calibration sequences. Accuracy (loss):

| | RTN | GPTQ |
|---|---|---|
| Full precision | 98.6% (0.0049) | |
| 4-bit per row | 98.7% (0.0039) | 98.6% (0.0051) |
| 3-bit per row | 97.3% (0.0152) | **98.1% (0.0080)** |
| **2-bit per row** | **75.5% (0.1450)** | **97.7% (0.0100)** |
| 2-bit, groups of 16 | 90.8% (0.0641) | 98.3% (0.0080) |

- **At 4 bits even RTN is fine** on this small model.
- **At 2 bits RTN collapses** (loss ×30) while GPTQ holds.
- **Grouping helps RTN most.** This is the paper's pattern, shifted to lower bit widths because our model is tiny and easy.

**Memory:** 175B parameters take 326 GiB at FP16, 81 GiB at 4 bits and 61 GiB at 3 bits.

**`experiments.py`:**
- **E1:** layer-level sweeps, including act-order;
- **E2:** dampening and calibration size;
- **E3:** width × bits × groups for whole models;
- **E4:** runtime scaling;
- **E5:** real GPTQ on OPT-125M/350M with WikiText-2 perplexity.
- None were run here.

---

## 8. Check yourself

1. Why is the Hessian the same for every row of W?
<details><summary>Answer</summary>The error of row i is (w_i − q_i)ᵀ X Xᵀ (w_i − q_i). Its Hessian 2XXᵀ depends only on the layer inputs, not on the row's weights.</details>

2. Quantize w = [1.0, 0.0, 0.25] at 2 bits on the min-max grid.
<details><summary>Answer</summary>scale = 1/3, zero = 0; levels [0, 0.333, 0.667, 1.0]; result [1.0, 0.0, 0.333].</details>

3. What does Eq. 2 do intuitively?
<details><summary>Answer</summary>It spreads the rounding error of w_q over the remaining weights, in proportion to how correlated their inputs are with input q (via H⁻¹), so the layer output changes as little as possible.</details>

4. Why does a fixed column order make GPTQ so much faster than OBQ?
<details><summary>Answer</summary>All rows share the same set of remaining weights F, so H_F⁻¹ is updated once per column (d_col times) instead of once per weight (d_row·d_col times).</details>

5. What problem do lazy batch updates solve?
<details><summary>Answer</summary>Updating the whole remaining matrix after every column does few FLOPs per memory access, so the GPU waits on memory. Batching 128 columns and doing one big matrix-multiply update per block uses compute efficiently. The result is identical.</details>

6. Why the Cholesky reformulation and 1% dampening?
<details><summary>Answer</summary>Repeated rank-1 updates of H⁻¹ accumulate rounding error and can make it indefinite at scale, which wrecks layers. The needed rows of H_F⁻¹ are rows of the Cholesky factor of H⁻¹, computable once and stably; dampening keeps H well conditioned.</details>

7. Why quantize layers sequentially, feeding calibration data through already-quantized layers?
<details><summary>Answer</summary>So each layer is optimised for the inputs it will actually receive in the quantized model, partly correcting the errors of earlier layers.</details>

8. Why is weight-only quantization useful even though compute stays in FP16?
<details><summary>Answer</summary>Generation with small batches is memory-bound: time is spent reading weights. Reading 3–4-bit weights and de-quantizing on the fly gives large speed-ups (3.25–4.5×) and lets huge models fit on fewer GPUs.</details>

9. How much memory does a 175B model need at 3 bits?
<details><summary>Answer</summary>175e9 × 3/8 bytes ≈ 65.6 GB ≈ 61 GiB, plus a little for scales, so it fits on one 80 GB A100.</details>

10. In our toy, why did 4-bit RTN already work while 2-bit RTN failed?
<details><summary>Answer</summary>At 4 bits the 16 levels are fine enough for a small, well-behaved model. At 2 bits (4 levels) rounding errors are large and add up across layers. GPTQ's compensation keeps the layer outputs close, so accuracy stays at 97.7%.</details>
