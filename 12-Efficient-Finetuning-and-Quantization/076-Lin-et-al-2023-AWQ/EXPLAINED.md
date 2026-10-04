# AWQ: Activation-aware Weight Quantization, explained simply

**Paper:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (MIT, SJTU, Tsinghua, UMass, MIT-IBM Watson AI Lab), *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*, MLSys 2024 (Best Paper).

**In one sentence:** when quantizing LLM weights to 3–4 bits, a tiny fraction of weights matters most: the ones that **multiply large activations**. Protect them by **scaling those weight channels up** (and the activations down) before rounding. This is an exact transformation, it needs only per-channel activation statistics, and it keeps a plain uniform low-bit format that is fast on hardware.

---

## 1. Not all weights are equal (Table 1)
- **The test:** quantize OPT models to INT3 (groups of 128), but keep a few **input channels** (columns of W) in FP16.

| Keep 1% of channels in FP16, chosen by… | OPT-6.7B perplexity |
|---|---|
| none (plain round-to-nearest) | 23.54 |
| **activation magnitude** | **11.39** (FP16: 10.86) |
| weight norm | 22.37 |
| random | 23.54 |

- **The lesson:** "important" weights are found by looking at the **activations**, not at the weights. A channel that carries big inputs amplifies any rounding error in its weights.
- **The catch:** a mixed format (some FP16 columns among INT3) is awkward and slow on hardware.

---

## 2. Scaling instead of keeping FP16 (Section 3.2)
- **The quantizer:** Q(w) = Δ · Round(w/Δ), with Δ = max|w|/2^(N−1) for a group of weights.
- **Scale one weight w by s > 1 and its input x by 1/s.** The product is unchanged, and the error becomes:
```
Err(Q(w·s)·x/s) = Δ' · RoundErr(w·s/Δ') · x / s
```
- **Why it helps:**
  - **RoundErr** is about uniform on [0, 0.5], averaging ~0.25, whatever the scale;
  - scaling **one** element rarely changes the group's maximum, so **Δ' ≈ Δ**;
  - so the error on the salient channel is **divided by ~s**.
- **The limit:** if s is too large, the scaled weight becomes the group's maximum. Δ' grows, and **every other weight** in the group gets a larger error.
- **Table 2 (OPT-6.7B), scaling the 1% salient channels by s:**

| s | 1 | 1.25 | 1.5 | 2 | 4 |
|---|---|---|---|---|---|
| Groups whose Δ changes | 0% | 2.8% | 4.4% | 8.2% | 21.2% |
| Perplexity | 23.54 | 12.87 | 12.48 | **11.92** | 12.36 |

### Worked example (our demo)
- **Setup:** a group of 8 weights with range 1.78, so INT3 gives Δ = 1.78/7 = 0.254. The salient weight 0.07 meets an activation of 50.
- **s = 1:** 0.07 rounds to 0, an error of 0.07 × 50 = 3.5 in the output.
- **s = 4:** 0.28 rounds to 0.254, so the effective weight is 0.254/4 = 0.064 and the error drops to about 0.006 × 50 = 0.3.
- **Δ is unchanged** in both cases.

---

## 3. The AWQ search
- **Pick per-input-channel scales s to minimise the layer's output error:**
```
s* = argmin_s ‖ Q(W·diag(s)) · (diag(s)⁻¹·X) − W X ‖          (Eq. 4)
```
- **The search space is tiny.** Saliency is determined by activation size, so
```
s = s_X^α,   s_X = mean |activation| of each input channel,   α ∈ [0, 1] by grid search (~20 points)    (Eq. 5)
```
  - α = 0: no scaling. α = 1: the most aggressive.
- **Weight clipping:** shrink each group's range a little to minimise output error.
- **Free at run time:** diag(s)⁻¹ is folded into the **previous operator**, e.g. a LayerNorm's gain or the previous linear layer's output rows. The idea is the same as SmoothQuant (074), but used here for **weight-only** quantization.
- **No backprop, no regression:** it only needs the **average magnitude per channel** from a small calibration set, so it **over-fits the calibration data less** than GPTQ (paper 075).
  - It needs ~10× less calibration data (16 vs 192 sequences).
  - With a mismatched calibration domain (PubMed ↔ Enron), AWQ's perplexity rises 0.5–0.6 while GPTQ's rises 2.3–4.9.

---

## 4. Results and TinyChat
- **INT3-g128 on OPT (Table 3):**

| Method | OPT-1.3B | OPT-6.7B | OPT-13B | OPT-30B |
|---|---|---|---|---|
| FP16 | 14.62 | 10.86 | 10.13 | 9.56 |
| RTN | 119.47 | 23.54 | 46.04 | 18.80 |
| 1% FP16 (by activation) | 16.91 | 11.39 | 10.43 | 9.85 |
| s = 2 for the 1% salient channels | 18.63 | 11.92 | 10.80 | 10.32 |
| **AWQ** | **16.32** | **11.39** | 10.56 | **9.77** |

- **Llama and Llama-2, 7B–70B (Table 4):** AWQ beats RTN and GPTQ (with and without reordering) at INT3 and INT4.
- **It works beyond text models:** instruction-tuned, multi-modal and code models, with good generalisation.
- **TinyChat**, the paper's W4A16 inference engine:
  - **3.2–3.3×** faster than Hugging Face FP16 on desktop and mobile GPUs;
  - runs Llama-2-70B on a 64 GB Jetson Orin.
  - **Why weight-only:** generation at batch size 1 is memory-bound, and weight reads dwarf activation reads.

---

## 5. Why it matters
- **AWQ is one of the most-used 4-bit formats** for open LLMs (AutoAWQ; supported in vLLM, TensorRT-LLM, Hugging Face).
- **"Activation-aware" saliency** is a simple, data-light principle. It complements GPTQ's error compensation, and the two can be combined.

---

## 6. What our code found
- **Model:** paper 074's tiny LLaMA with LLM-like outliers. Three input channels have mean |x| of 108, 90 and 73 vs a median of 0.59, and their weight columns are **small** (norm 0.06 vs 0.70). Judged by the weights, they look unimportant.

**Keep 3 of 64 input channels (~5%) in FP** (groups of 16):

| | RTN | By activation | By weight | Random |
|---|---|---|---|---|
| INT3 | 50.6% | **97.4%** | 50.5% | 45.3% |
| INT4 | 26.7% | **99.1%** | 26.4% | 37.4% |

The paper's Table 1 pattern holds: **activation** picks the right channels; weight norm and random do nothing.

**Fixed scale s for the salient channels** (INT4-g16):

| s | Groups whose Δ changed | Accuracy |
|---|---|---|
| 1 | 0% | 26.7% |
| 1.5 | 7.3% | 30.4% |
| 2 | 10.5% | 50.4% |
| 4 | 15.0% | 86.3% |
| 8 | 19.8% | 94.5% |

- **The share of changed Δ** behaves like the paper's (2.8% → 21.2%).
- **Honest difference:** our best s is large (8), not 2, because our salient weights are ~12× smaller than the rest. One fixed s cannot suit every model, which is why AWQ searches.

**Table 3 analogue**, accuracy (loss):

| Method | INT3-g16 | INT4-g16 |
|---|---|---|
| RTN | 50.6% (0.807) | 26.7% (1.118) |
| Keep 3 channels in FP | 97.4% (0.011) | 99.1% (0.004) |
| **AWQ** | 89.4% (0.063) | **98.2%** (0.007) |
| **AWQ + clipping** | **95.5%** (0.022) | 98.6% (0.006) |
| GPTQ (075) | 60.8% (0.431) | 49.3% (0.646) |

- **AWQ nearly matches mixed precision** in a uniform format. The chosen α values range from 0.2 to 0.85 per layer group.
- **Honest note:** GPTQ does badly here. The huge outlier inputs dominate its Hessian, and their tiny weights get rounded with little correlated capacity left to compensate. "Act-order" GPTQ and AWQ + GPTQ are in `experiments.py` E4. The paper also finds AWQ ≥ GPTQ on Llama models.
- **Odd detail:** our INT4 RTN (26.7%) is worse than INT3 RTN (50.6%). Both collapse because the salient small weights round to zero; which wrong answers come out differs.

**`experiments.py`:**
- **E1:** fractions × selection rule × bits × groups;
- **E2:** a fine s sweep;
- **E3:** calibration robustness, AWQ vs GPTQ with copy-only vs all-task calibration and 3–126 sequences;
- **E4:** AWQ + GPTQ and act-order;
- **E5:** grid size and α distribution;
- **E6:** real OPT-1.3B INT3/INT4-g128 perplexity.
- None were run here.

---

## 7. Check yourself

1. Why are weights connected to large-activation channels the "salient" ones?
<details><summary>Answer</summary>The output error from a rounding error δ in weight w_j is δ·x_j. Channels with large x_j amplify any rounding error, so their weights matter most for the layer output.</details>

2. Why doesn't keeping 1% of weights in FP16 solve the problem in practice?
<details><summary>Answer</summary>A mixed-precision layout (a few FP16 columns among INT3/INT4 ones) complicates kernels and memory layout and is hard to make fast. AWQ gets the same benefit with a uniform format.</details>

3. Show that scaling by s reduces a salient weight's error.
<details><summary>Answer</summary>Q(w·s)·(x/s) has error Δ'·RoundErr(ws/Δ')·x/s. RoundErr averages ~0.25 regardless, and Δ' ≈ Δ if the group maximum doesn't change, so the error falls by about a factor s.</details>

4. Why does a very large s hurt?
<details><summary>Answer</summary>The scaled weight becomes the group's maximum, so Δ' grows and every other weight in the group gets a larger quantization step (and error).</details>

5. What is AWQ's search space, and why is it so small?
<details><summary>Answer</summary>s = s_X^α with one α per layer, grid-searched in [0, 1]. Saliency is set by activation magnitude, so scaling as a power of the mean activation captures it with a single number.</details>

6. Where does diag(s)⁻¹ go at inference time?
<details><summary>Answer</summary>It is folded into the previous operator (the LayerNorm gain and bias, or the output rows of the previous linear layer), so it costs nothing extra.</details>

7. Why is AWQ less prone to over-fitting the calibration set than GPTQ?
<details><summary>Answer</summary>It only uses per-channel mean activation magnitudes to pick a scale, with no regression or weight updates fitted to the calibration inputs. GPTQ adjusts weights to reconstruct the calibration outputs, which can fit that distribution too closely.</details>

8. Why does weight-only 4-bit speed up on-device generation?
<details><summary>Answer</summary>At batch size 1, generation is memory-bound: each token reads all weights. 4-bit weights mean ~4× less data to read (W4A16 raises arithmetic intensity about 4×), so TinyChat gets a 3.2–3.3× speed-up.</details>

9. In our toy, why did selecting channels by weight norm fail completely?
<details><summary>Answer</summary>The salient channels had SMALL weights (norm 0.06 vs 0.70), compensating their huge activations, so weight norm points at the wrong channels. Only activation statistics reveal them.</details>

10. How do AWQ and SmoothQuant differ, given both use per-channel scaling?
<details><summary>Answer</summary>SmoothQuant (W8A8) moves difficulty from activations to weights so that BOTH can be INT8. AWQ (weight-only, e.g. W4A16) scales salient weight channels UP so their weights are quantized more finely; activations stay in FP16.</details>
