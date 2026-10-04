# SmoothQuant, explained simply

**Paper:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han (MIT, NVIDIA), *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML 2023.

**In one sentence:** LLM **activations** are hard to quantize to 8 bits because a few channels are ~100× larger than the rest, while **weights** are easy. SmoothQuant divides each activation channel by a factor s_j and multiplies the matching weight row by s_j. This is mathematically identical, but it **moves the difficulty from activations to weights**, so both quantize well with plain INT8 (W8A8), with no retraining.

---

## 1. Background: integer quantization

### The quantizer (Eq. 1)
Store a tensor X as 8-bit integers:
```
Δ = max|X| / (2^(N−1) − 1) = max|X| / 127   (N = 8, symmetric)
X_int = round(X / Δ),       X ≈ Δ · X_int
```
**Worked example:** X = [0.3, −1.2, 0.05, 2.0].
- Δ = 2.0/127 = 0.0157, and X/Δ rounds to [19, −76, 3, 127].
- Back-conversion gives [0.299, −1.197, 0.047, 2.0], which is very close.
- **Add one outlier, 200:** now Δ = 1.575 and the others become [0, −1.575, 0, 1.575]. **The small values are destroyed.**

### Why bother?
- **Memory:** INT8 halves memory vs FP16 (a 175B model needs at least 350 GB in FP16).
- **Speed:** INT8 matrix multiplies (GEMMs) roughly double throughput.
- **Both inputs must be INT8:** to use INT8 GEMM, the **weights and the activations** must be INT8 (W8A8).

### Granularity
- **Per-tensor:** one Δ for the whole matrix.
- **Per-token:** one Δ per row of X.
- **Per-channel:** one Δ per column of X, or per output row of W.
- **The constraint:** for X (T×Cᵢ) · W (Cᵢ×Cₒ) on INT8 hardware, the scales must factor out of the sum over Cᵢ:
```
Y ≈ diag(Δ_X) · (X_int W_int) · diag(Δ_W)
```
  So **per-token** activation scales and **per-output-channel** weight scales are fine. **Per-input-channel activation scales are not**: they sit inside the sum.

### Static vs dynamic
- **Static:** Δ is computed once from calibration data.
- **Dynamic:** Δ is computed at run time from the actual tensor (slower).

---

## 2. Three observations (Section 3)
1. **Weights are easy to quantize:** their distribution is flat and uniform. INT8 weights alone barely hurt.
2. **Activations have outliers.** From about 6.7B parameters, a few channels are **~100× larger** than the rest.
   - With per-tensor Δ, channel j only uses 2⁸ · mⱼ/m of the levels, where mⱼ is its max and m the global max.
   - For normal channels that is **2–3 levels**.
3. **Outliers sit in fixed channels.**
   - The same channels are large in **every token**, and the variance *within* a channel is small.
   - So **per-channel** activation scales would fix it, and per-token scales barely help.
   - Table 1, OPT-175B: per-tensor 32.3%, per-token 31.7%, per-channel **71.4%** (FP16 71.6%).
   - But per-channel activation scales don't fit INT8 GEMM.

---

## 3. SmoothQuant
**Scale activation channels down and weights up, by the same factor:**
```
Y = X W = (X · diag(s)⁻¹) · (diag(s) · W) = X̂ Ŵ
```
- **Exact in floating point.**
- **Free at run time:** X usually comes from a LayerNorm, so diag(s)⁻¹ is **folded into the LayerNorm's gain and bias**, and diag(s) into W, offline.

**How to choose s (Eq. 4):**
```
s_j = max|X_j|^α / max|W_j|^(1−α)
```
- **α = 1:** s_j = max|X_j|. All activation channels become equally sized, but all the difficulty moves into the weights.
- **α = 0:** s_j = 1/max|W_j|. All the difficulty stays in the activations.
- **α = 0.5:** after smoothing, activation and weight maxima in channel j are equal (both √(max|X_j|·max|W_j|)). The difficulty is **shared evenly**.
- **Defaults:** α = 0.5 for OPT and BLOOM; 0.75 for GLM-130B, whose activations have ~30% outliers.

**Worked example:** a channel with max|X_j| = 100 and max|W_j| = 0.01 at α = 0.5:
- s_j = √100 / √0.01 = 10/0.1 = 100;
- the activation max becomes 100/100 = **1**, and the weight max becomes 0.01·100 = **1**.

**Calibration:** 512 random sentences from the Pile, used once for both the smoothing factors and the static scales.

**Efficiency levels (Table 2):**

| Level | Weights | Activations |
|---|---|---|
| O1 | per-tensor | per-token dynamic |
| O2 | per-tensor | per-tensor dynamic |
| O3 | per-tensor | per-tensor **static** (fastest) |

All linear layers and attention BMMs run in INT8; Softmax and LayerNorm stay in FP16.

---

## 4. Results

**OPT-175B (Table 3):** average of 7 zero-shot tasks, plus WikiText perplexity.

| Method | Average accuracy | WikiText ppl |
|---|---|---|
| FP16 | 66.9% | 10.99 |
| W8A8 naive | 35.5% | 93,080 |
| ZeroQuant | 35.8% | 84,648 |
| LLM.int8() (outliers in FP16) | 66.7% | 11.10 |
| Outlier Suppression | 36.0% | 96,151 |
| **SmoothQuant O1 / O2 / O3** | **66.5 / 66.4 / 66.8%** | 11.11 / 11.14 / **11.17** |

- **It works on other large models too:** BLOOM-176B and GLM-130B (Table 4).
- **Efficiency:**
  - up to **1.56× speedup** and **2× memory reduction** (FasterTransformer);
  - 1.51× speedup and 1.96× memory saving in PyTorch;
  - a **530B** model served on a single 8-GPU node.
- **LLM.int8() is accurate but slow:** its mixed-precision decomposition is hard to run efficiently. SmoothQuant keeps **everything in INT8**.

---

## 5. Why it matters
- **SmoothQuant made W8A8 inference of 100B+ models practical and training-free.** It is integrated into TensorRT-LLM, FasterTransformer and other serving stacks.
- **"Migrate difficulty with an equivalent rescaling"** became a standard trick:
  - AWQ (paper 076) rescales weights by activation statistics;
  - later rotation methods (QuaRot, SpinQuant) generalise per-channel scaling to orthogonal transforms.

---

## 6. What our code found

### Setup
- **Model:** paper 073's tiny LLaMA, pre-trained on copy / reverse / sort.
- **Making outliers the way they arise in real LLMs:**
  - three RMSNorm gain entries ×150;
  - the matching weight columns ÷ only √150;
  - a brief re-training.
- **Result:** task accuracy 98.9%. Block 0's Wq input has channel maxima with median 1.53, but channels 3, 17 and 41 reach **396, 245 and 244** (~259×). They are large in 92% of tokens. A normal channel keeps only **~1 of 256** INT8 levels under one per-tensor scale.

**W8A8:**

| Method | Task accuracy | Wq output error |
|---|---|---|
| Floating point | 98.9% | – |
| Per-tensor dynamic | 5.9% | 0.406 |
| Per-tensor static | 8.2% | 0.406 |
| Per-token dynamic | 80.0% | 0.186 |
| Per-channel (not INT8-GEMM friendly) | 97.0% | 0.121 |
| LLM.int8()-style (outlier channels in FP) | 96.9% | 0.121 |
| **SmoothQuant α = 0.5 + per-token (O1)** | **99.0%** | 0.012 |
| **SmoothQuant α = 0.5 + per-tensor static (O3)** | **98.9%** | 0.014 |

- **The paper's story at toy scale:**
  - naive W8A8 collapses;
  - per-token scales help only partly;
  - per-channel scales work but can't run on INT8 hardware;
  - SmoothQuant reaches full accuracy with the **cheapest** scheme.
- **Smoothing is exact:** max output difference 1.1e-5. The activation channel range shrinks from 0.76–396 to **0.48–2.93**, and the weight columns grow to 0.29–2.93.

**Migration strength α** (per-tensor static):

| | α = 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| W8A8 | 98.6% | 98.7% | 98.9% | 99.2% | 98.3% |
| W6A6 | 92.9% | 96.5% | **97.0%** | 95.6% | 92.4% |

- **At 8 bits any smoothing is enough** in our toy.
- **At 6 bits the trade-off appears:** α = 0 leaves activations hard, α = 1 makes the weights hard, and α ≈ 0.5 is best, as in the paper's Figure 10.

**`experiments.py`:**
- **E1:** granularity × bits grid;
- **E2:** α × bits × outlier strength;
- **E3:** calibration-set size;
- **E4:** where to smooth;
- **E5:** real OPT-1.3B W8A8 perplexity with and without SmoothQuant.
- None were run here.

---

## 7. Check yourself

1. Quantize x = [0.5, −0.25, 1.0] to INT8 per-tensor.
<details><summary>Answer</summary>Δ = 1.0/127 ≈ 0.00787; x/Δ = [63.5, −31.75, 127] → round to [64, −32, 127] (63.5 rounds to 64) → back ≈ [0.504, −0.252, 1.0].</details>

2. Why can't INT8 kernels use per-input-channel activation scales?
<details><summary>Answer</summary>Y_ik = Σ_j X_ij W_jk. A per-input-channel scale Δ_j sits inside the sum over j, so it can't be pulled out as one factor per output element. Only per-row (token) scales of X and per-column (output channel) scales of W factor out.</details>

3. Why is SmoothQuant mathematically exact?
<details><summary>Answer</summary>(X diag(s)⁻¹)(diag(s) W) = X W, since diag(s)⁻¹ diag(s) = I. Only the split of magnitudes between the two factors changes.</details>

4. Where does the extra division by s go at run time?
<details><summary>Answer</summary>Nowhere extra: it is folded offline into the preceding LayerNorm's gain and bias (or the previous linear layer), and s is folded into the weights.</details>

5. max|X_j| = 64, max|W_j| = 0.25, α = 0.5. Compute s_j and the new maxima.
<details><summary>Answer</summary>s = √64 / √0.25 = 8/0.5 = 16. Activation max 64/16 = 4; weight max 0.25·16 = 4. They are balanced.</details>

6. What goes wrong with α = 1?
<details><summary>Answer</summary>s_j = max|X_j| makes every activation channel equal (easy), but the weights of the old outlier channels are multiplied by ~100, creating weight outliers that hurt weight quantization.</details>

7. Why does per-token quantization barely help on real LLMs?
<details><summary>Answer</summary>The outliers are in fixed channels present in almost every token, so every token's maximum is set by an outlier; per-token scales are nearly as coarse as per-tensor.</details>

8. How does LLM.int8() handle outliers, and what's its downside?
<details><summary>Answer</summary>It keeps the outlier channels in FP16 and multiplies the rest in INT8 (mixed-precision decomposition). It is accurate, but the decomposition is hard to make fast on hardware.</details>

9. What is the difference between O1 and O3?
<details><summary>Answer</summary>O1 uses per-token dynamic activation scales (computed at run time); O3 uses a single per-tensor static scale from calibration (fastest). SmoothQuant keeps accuracy even at O3.</details>

10. In our toy, why did per-tensor W8A8 collapse to ~6–8% while SmoothQuant held 98.9%?
<details><summary>Answer</summary>The outlier channels (up to 396 vs a median of 1.5) set the per-tensor scale, leaving normal channels about 1 quantization level. Smoothing shrank the activation range to 0.48–2.93, so every channel gets many levels; the weights absorbed the factor but stayed easy to quantize.</details>
