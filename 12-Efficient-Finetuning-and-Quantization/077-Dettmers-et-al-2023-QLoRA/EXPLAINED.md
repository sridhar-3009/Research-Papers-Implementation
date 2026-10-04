# QLoRA, explained simply

**Paper:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington), *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023.

**In one sentence:** store a pre-trained model's weights in **4 bits** (a new data type, NormalFloat-4), keep them **frozen**, and fine-tune small **LoRA** adapters (paper 073) through them. A 65B-parameter model can then be fine-tuned on **one 48 GB GPU** instead of more than 780 GB of GPUs, with no loss in quality. The resulting chatbot, Guanaco, reached 99.3% of ChatGPT on the Vicuna benchmark after 24 hours of training.

---

## 1. The memory problem
- **Full 16-bit fine-tuning of a 65B model needs, per parameter:**
  - 2 bytes for the weights;
  - 2 bytes for the gradients;
  - 8 bytes for Adam's two FP32 states;
  - **≈ 12 bytes × 65B ≈ 780 GB**, plus activations.
- **LoRA** removes the gradient and optimizer cost of the frozen weights, but the 16-bit weights alone are still 130 GB.
- **QLoRA also shrinks those frozen weights to ~4 bits:** about 33 GB.

---

## 2. Three ingredients

### (1) 4-bit NormalFloat (NF4)
- **Block-wise absmax quantization:** split the weights into blocks of 64; divide each block by its absmax c, so the values lie in [−1, 1]; snap each value to the nearest of 16 code values; store the 4-bit code plus c.
- **Which 16 values?**
  - Pre-trained weights are roughly **normally distributed** (zero mean).
  - **Quantile quantization** makes every code equally likely, which is information-theoretically optimal: 4 bits of information per weight.
  - For a known distribution the quantiles are exact, with no costly estimation.
- **The NF4 values:**
```
q_i = ½ [ Q_N(i/(2^k + 1)) + Q_N((i+1)/(2^k + 1)) ]      (Eq. 4; Q_N = quantile function of N(0, 1))
```
  - Use **8 positive and 7 negative** quantiles, so that **0 is exact** (useful for padding and zeros) and all 16 codes are used.
  - Normalise to [−1, 1]. Appendix E gives:
```
[−1.0, −0.6962, −0.5251, −0.3949, −0.2844, −0.1848, −0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```
- **The levels are denser near 0**, where most weights are, unlike Int4's even spacing.
- **Table 2, mean perplexity of 125M–13B models after 4-bit quantization:**

| Data type | Mean perplexity |
|---|---|
| Int4 | 34.34 |
| FP4 (E2M1) | 31.07 |
| FP4 (E3M0) | 29.48 |
| **NF4 + DQ** | **27.41** |

### (2) Double quantization (DQ)
- **The constants cost space:** one FP32 constant per 64 weights is 32/64 = **0.5 bits per parameter**.
- **DQ quantizes those constants** with 8-bit floats in blocks of 256, after subtracting their mean:
```
8/64 + 32/(64·256) = 0.125 + 0.002 = 0.127 bits per parameter
```
  This saves **0.373 bits per parameter, ~3 GB on a 65B model**.

### (3) Paged optimizers
- Optimizer states live in NVIDIA **unified memory**. During memory spikes (e.g. long sequences with gradient checkpointing) they are paged to CPU RAM automatically, avoiding out-of-memory crashes.

### Putting it together (Eq. 5)
```
Y_BF16 = X_BF16 · doubleDequant(c1_FP32, c2_8bit, W_NF4) + X_BF16 · L1_BF16 · L2_BF16
```
- **Two data types:** a **storage** type (NF4) and a **compute** type (BF16).
- **The forward pass:** weights are de-quantized to BF16 for each matmul.
- **The backward pass:** gradients flow **through** the de-quantized weights to the input (needed for earlier layers), but only the **LoRA** parameters get weight gradients.

---

## 3. Findings
- **QLoRA matches 16-bit:** on GLUE and Super-NaturalInstructions it matches 16-bit full fine-tuning and 16-bit LoRA. The quantization loss is recovered by the adapters.
- **MMLU 5-shot, LLaMA 7–65B (Table 4):**

| | BF16 | FP4 | NF4 + DQ |
|---|---|---|---|
| Mean | 53.0 | 52.2 | **53.1** |

  NF4 matches 16-bit; FP4 is about 1 point behind.
- **LoRA placement matters most.** With LoRA only on query and value (the original LoRA default), QLoRA **did not** match full fine-tuning. **LoRA on all linear layers** is required, and then the rank r matters little.
- **Guanaco** (QLoRA on the OASST1 dataset):
  - **Guanaco-65B reaches 99.3% of ChatGPT** on the Vicuna benchmark (judged by GPT-4) and trains in 24 hours on one GPU;
  - memory footprints: Guanaco-65B 41 GB, 33B 21 GB, 13B 10 GB.
- **Data quality beats quantity:** 9k high-quality OASST1 examples beat 450k FLAN v2 examples for chatbot quality.
- **Benchmarks disagree:** MMLU and chatbot (Elo) rankings do not align.
- **GPT-4 as a judge** largely agrees with human raters on system-level rankings, but has biases.

---

## 4. Why it matters
- **QLoRA put fine-tuning of 30–70B models on single consumer and workstation GPUs.** It triggered a wave of open fine-tunes.
- **It is the default recipe** in Hugging Face PEFT + bitsandbytes (`load_in_4bit`, `nf4`, `double_quant`).
- **NF4 is an influential data type:** "quantize to the distribution's quantiles".

---

## 5. What our code found

**NF4, built from normal quantiles, matches Appendix E to 6e-8.**

**Normally distributed weights**, blocks of 64:

| Data type | Relative MSE | Code entropy (max 4 bits) |
|---|---|---|
| Int4 | 0.0115 | 3.51 |
| FP4 (E2M1) | 0.0112 | 3.82 |
| FP4 (E3M0) | 0.0352 | 3.50 |
| **NF4** | **0.0085** | **3.89** |

- **NF4 is best and uses its codes most evenly**, as in Table 2.
- **The FP4 variants order differently** from the paper: E3M0 is worst on pure Gaussians, while on real models it beat E2M1.

**Double quantization:** constants go from 0.500 to **0.127** bits per parameter (saving ~3.0 GB on 65B). The constants' relative error is 0.0017, and weight MSE is unchanged (0.0085).

**A tiny LLaMA stored in k bits** (loss multiplier vs full precision):

| Data type | Loss multiplier |
|---|---|
| Int4 | ×0.7 |
| FP4 E2M1 | ×0.9 |
| NF4 | ×1.0 |
| FP4 E3M0 | ×2.7 |
| Int3 | **×5.6** |
| FP3 | ×2.1 |
| NF3 | ×2.1 |

At 4 bits this small model barely notices, except with E3M0. At 3 bits Int3 is clearly worst.

**QLoRA fine-tuning** (new task: the SORT prompt sorts descending):

| Method | Stored base | Trainable | Accuracy |
|---|---|---|---|
| Full fine-tuning | 16-bit | 133,440 | 99.2% |
| LoRA r = 8, all linear layers | 16-bit | 23,552 | 99.7% |
| LoRA r = 8, **q, v only** | 16-bit | 4,096 | **94.9%** |
| **QLoRA (NF4 + DQ), all layers** | **4-bit** | 23,552 | **99.3%** |
| QLoRA with an FP4 E3M0 base | 4-bit | 23,552 | 99.6% |

- **QLoRA = 16-bit LoRA = full fine-tuning.**
- **LoRA on all layers matters more than the data type:** q, v only falls short, as the paper found.
- **On this tiny model the adapters recover even the FP4 base.** The paper's MMLU shows FP4 ~1 point behind.

**Memory for 65B** (our rough budget):

| Method | Memory |
|---|---|
| 16-bit full fine-tuning | ~786 GB |
| 16-bit LoRA | ~139 GB |
| **QLoRA** | **~43 GB** |

This matches the paper's > 780 GB vs < 48 GB.

**`experiments.py`:**
- **E1:** data types on Gaussian, Laplace and real weights;
- **E2:** whole-model quantization grid;
- **E3:** base type × LoRA placement × r;
- **E4:** do adapters recover 3-bit losses?
- **E5:** real QLoRA with bitsandbytes + peft on OPT-350M.
- None were run here.

---

## 6. Check yourself

1. Why do quantiles give an "information-theoretically optimal" data type?
<details><summary>Answer</summary>If each code value covers an equal share of the probability mass, every code is used equally often, so the 4 bits carry the maximum possible information (entropy 4 bits) about the weights.</details>

2. Why does NF4 use 8 positive and 7 negative values instead of a symmetric set?
<details><summary>Answer</summary>A symmetric set of 16 quantiles has no exact zero. Using 2^(k−1) quantiles for one side and 2^(k−1)+1 for the other, then merging the shared zero, gives an exact 0 and still uses all 16 codes.</details>

3. How does block-wise absmax quantization work, and why small blocks?
<details><summary>Answer</summary>Each block of 64 weights is divided by its largest |w| so it lies in [−1, 1], then rounded to the nearest code. Small blocks keep one outlier from ruining the precision of many weights, but every block needs a stored constant.</details>

4. Compute the constant overhead with and without double quantization.
<details><summary>Answer</summary>Without: 32 bits / 64 weights = 0.5 bits per parameter. With 8-bit constants and one FP32 second-level constant per 256 first-level ones: 8/64 + 32/(64·256) ≈ 0.127.</details>

5. In QLoRA, which tensors get gradients?
<details><summary>Answer</summary>Only the LoRA matrices. Gradients still pass through the frozen de-quantized weights to compute ∂E/∂X for earlier layers, but no gradient is stored for the 4-bit weights.</details>

6. Estimate the memory of 16-bit full fine-tuning for 65B parameters.
<details><summary>Answer</summary>About 2 (weights) + 2 (grads) + 8 (Adam) = 12 bytes per parameter → 780 GB, plus activations.</details>

7. What LoRA setting did QLoRA find necessary?
<details><summary>Answer</summary>Adapters on all linear layers of every transformer block. Only q and v (the original LoRA default) failed to match full fine-tuning, and then r mattered little.</details>

8. What do paged optimizers solve?
<details><summary>Answer</summary>Memory spikes (e.g. from long sequences) that would cause out-of-memory errors. Optimizer states sit in unified memory and are paged to CPU RAM when the GPU is full, then paged back for the update.</details>

9. What is the difference between QLoRA's storage and compute data types?
<details><summary>Answer</summary>Weights are stored in NF4 (plus quantized constants) to save memory, but every matmul de-quantizes them to BF16 and computes in BF16. Low-bit storage, normal-precision compute.</details>

10. In our toy, why did LoRA on q, v only reach 94.9% while all-layer LoRA reached 99.7%?
<details><summary>Answer</summary>The new behaviour needs changes in several parts of the network (attention output, FFN). Adapters only on q and v can't make all of those changes with the same training, which echoes the paper's finding.</details>
