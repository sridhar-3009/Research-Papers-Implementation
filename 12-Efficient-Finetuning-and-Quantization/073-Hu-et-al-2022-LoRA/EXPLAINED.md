# LoRA: Low-Rank Adaptation, explained simply

**Paper:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.

**In one sentence:** to fine-tune a huge model, **freeze** all its weights and learn, for a few weight matrices, a tiny **low-rank** correction ΔW = B·A. On GPT-3 175B this cuts trainable parameters **10,000×** and GPU memory **3×**, matches or beats full fine-tuning, and adds **no** inference latency, because ΔW can be merged into W.

---

## 1. The problem
- **Full fine-tuning** updates every parameter. For GPT-3 that means:
  - storing a **350 GB** copy per task;
  - Adam needs extra optimizer state for every parameter (about 1.2 TB of VRAM to train).
- **Earlier parameter-efficient methods** each had a catch:
  - **adapters** insert small layers, adding inference **latency**, especially at batch size 1;
  - **prefix/prompt tuning** uses up part of the context window and is hard to optimise.

---

## 2. The idea: the update has low intrinsic rank
- **The hypothesis:** pre-trained models have a low "intrinsic dimension", and the **change** in weights during adaptation is also low-rank.
- **The method:** for a weight W₀ ∈ ℝ^(d×k), write the fine-tuned weight as
```
W = W₀ + ΔW = W₀ + (α/r)·B·A,      B ∈ ℝ^(d×r),  A ∈ ℝ^(r×k),  r ≪ min(d, k)
h = W₀ x + (α/r)·B A x
```
- **Training:** W₀ is **frozen**; only A and B are trained.
- **Initialisation:** A is random Gaussian and **B = 0**, so ΔW = 0 at the start. The model begins exactly as the pre-trained one.
- **Scaling by α/r:** α is fixed (set to the first r tried). With Adam, changing α is like changing the learning rate, and this scaling means you **don't need to re-tune** when you change r.
- **No inference latency:** after training, compute W = W₀ + BA once and serve a normal model. To switch tasks, subtract BA and add B′A′.
- **It generalises full fine-tuning:** with r equal to the full rank, LoRA can express any update.

### Counting parameters (worked example)
- **One d×d matrix** with rank r adds r·(d + d) parameters.
- **GPT-3** has d = 12,288 and 96 layers. Adapting **Wq and Wv** with r = 4:
```
96 layers × 2 matrices × 4 × (12,288 + 12,288) = 18,874,368 ≈ 18.9M parameters
```
  That is about 38 MB in fp16 (the paper quotes 18M and ~35 MB), versus 175B parameters: **~10,000× fewer**.
- **With r = 1** on Wq and Wv: 4.7M, the "4.7M" row of Table 4.
- **Storing 100 tasks:** 350 GB + 100 × 35 MB ≈ 354 GB, instead of 100 × 350 GB = 35 TB.

---

## 3. Results

### GPT-3 175B (Table 4)

| Method | Trainable params | WikiSQL | MNLI-m | SAMSum R1/R2/RL |
|---|---|---|---|---|
| Full fine-tuning | 175,255.8M | 73.8 | 89.5 | 52.0/28.0/44.5 |
| BitFit | 14.2M | 71.3 | 91.0 | 51.3/27.4/43.5 |
| **LoRA** | **4.7M** | 73.4 | **91.7** | **53.8/29.8/45.9** |
| LoRA | 37.7M | **74.0** | 91.6 | 53.4/29.2/45.1 |

- **Smaller models:** LoRA also matches or beats fine-tuning on RoBERTa, DeBERTa and GPT-2.
- **Training cost:** about 25% faster training on GPT-3 (no gradients for frozen weights) and a 3× memory reduction (1.2 TB → 350 GB).

### Which matrices? (Table 5, fixed budget of 18M parameters)

| Adapted | r | WikiSQL | MNLI |
|---|---|---|---|
| Wq | 8 | 70.4 | 91.0 |
| Wk | 8 | 70.0 | 90.8 |
| Wv | 8 | 73.0 | 91.0 |
| Wo | 8 | 73.2 | 91.3 |
| Wq, Wk | 4 | 71.4 | 91.3 |
| **Wq, Wv** | 4 | **73.7** | 91.3 |
| Wq, Wk, Wv, Wo | 2 | **73.7** | **91.7** |

**Spreading the budget over more matrices at a lower rank is better** than putting it all into one.

### How small can r be? (Table 6)
- **For Wq and Wv:** r = 1 already gives 73.4 / 91.3, about as good as r = 64.
- **For Wq alone:** larger r is needed.

### What ΔW looks like (Section 7)
- **Subspace similarity (Figure 3):** take the top singular directions of A learned with r = 8 and with r = 64. The **top few directions overlap strongly**; the rest look like random noise. ΔW really is low-rank.
- **Amplification (Table 7):** project W onto ΔW's top-r singular directions, giving ‖UᵀWVᵀ‖.
  - **The numbers:** 0.32 on ΔW's directions, 21.67 on W's own top directions, and 0.02 on random directions (GPT-3 layer 48, r = 4).
  - **Correlation with W:** ΔW is correlated with W (more than random), but it does **not** repeat W's top directions. It picks features W contains but does not emphasise.
  - **Amplification factor:** ‖ΔW‖/‖UᵀWVᵀ‖ = 6.91 / 0.32 ≈ **21.5**.

---

## 4. Why it matters
- **LoRA is the default way to fine-tune large models on modest hardware.** It is everywhere in open-source LLM work (Hugging Face PEFT, Stable Diffusion fine-tunes).
- **It enabled:**
  - many-task serving: one base model plus swappable small adapters;
  - **QLoRA** (paper 077): LoRA on a 4-bit-quantised base model, so a 65B model fine-tunes on one GPU.
- **"Weight updates are low-rank"** is now a central intuition about fine-tuning.

---

## 5. What our code found

### Setup
- **Model:** a tiny LLaMA (paper 056: d = 64, 2 layers, 133k parameters).
- **Pre-training:** COPY / REVERSE / SORT of 6 digits. After it, sort accuracy is 95.8%.
- **New behaviour to learn:** the **same** SORT prompt should now sort **descending**. The base model scores 0.0%.
- **Fine-tuning:** 300 Adam steps.

### Rank sweep (LoRA on Wq and Wv)

| Method | Trainable params | Descending-sort accuracy |
|---|---|---|
| Full fine-tuning | 133,440 | 99.8% |
| LoRA r = 1 | 512 | 47.1% |
| LoRA r = 2 | 1,024 | 81.0% |
| LoRA r = 4 | 2,048 | 85.9% |
| **LoRA r = 8** | **4,096** | **97.4%** |
| LoRA r = 16 | 8,192 | 97.9% |
| LoRA r = 64 (full rank) | 32,768 | 92.0% |

- **33× fewer parameters** (r = 8) come within a few points of full fine-tuning.
- **Full rank is no better.**
- **r = 1 is not enough in our toy.** For GPT-3, a far richer pre-trained model, it was.

### Same budget, different matrices (2,048 parameters)

| Adapted | Accuracy |
|---|---|
| Wq r8 | 83.0% |
| Wk r8 | 59.0% |
| Wv r8 | 37.5% |
| Wo r8 | 87.8% |
| Wq, Wk r4 | 68.5% |
| Wq, Wv r4 | 85.9% |
| **all four r2** | **97.0%** |

- **As in the paper, spreading the budget over all four matrices wins.**
- **The single-matrix ranking differs:** Wo alone is good here.

### Task switching
- **Full fine-tuning** gets descending 99.8% but ascending **0.0%**: the old skill is overwritten.
- **LoRA adapter on:** descending 97.4%. **Adapter off:** ascending 95.8%, which is the untouched base model.

### What ΔW looks like (last layer)
- **Subspace similarity:** φ(1, 8) = 0.49 between r = 8 and r = 64 adapters (chance is 0.12). The top direction is shared, though less strongly than in GPT-3.
- **Amplification (Wq, r = 4):**
  - ‖UᵀWVᵀ‖ = 1.01 on ΔW's directions, vs 4.71 on W's own top directions and 0.34 on random ones;
  - ‖ΔW‖ = 11.8, an amplification of **~12×**.
  - This is the paper's qualitative pattern: amplify what W has but doesn't emphasise.

**`experiments.py`:**
- **E1:** a rank × weight-type grid with learning-rate tuning;
- **E2:** budget splits;
- **E3:** subspace similarity across different seeds;
- **E4:** amplification for every layer;
- **E5:** the small-data regime;
- **E6:** α/learning-rate coupling;
- **E7:** real GPT-2 + LoRA on SST-2 vs full fine-tuning (accuracy, parameters, memory, speed).
- None were run here.

---

## 6. Check yourself

1. Why is B initialised to zero?
<details><summary>Answer</summary>So that ΔW = BA = 0 at the start: the adapted model begins exactly equal to the pre-trained model, and training moves it gradually.</details>

2. How many trainable parameters does LoRA add for one 1024×1024 matrix at r = 8?
<details><summary>Answer</summary>r·(d + k) = 8·2048 = 16,384 (vs 1,048,576 for the full matrix: 64× fewer).</details>

3. Why does LoRA add no inference latency, unlike adapters?
<details><summary>Answer</summary>BA has the same shape as W₀, so after training you store W = W₀ + BA and run the model normally. Adapters add extra sequential layers that must run at inference.</details>

4. What does the α/r scaling buy?
<details><summary>Answer</summary>The update's size doesn't change much when you change r, so a learning rate tuned for one r keeps working for others (with Adam, α acts like a learning rate).</details>

5. With a fixed budget, why might Wq, Wv at r = 4 beat Wq at r = 8?
<details><summary>Answer</summary>The update has low intrinsic rank, so rank 4 is already enough per matrix, and adapting more matrices gives more places to change behaviour. Table 5 shows it, and our toy agrees (all four matrices at r = 2 was best).</details>

6. How do you switch a deployed LoRA model between tasks?
<details><summary>Answer</summary>Subtract the current BA from W and add the new task's B′A′ (or keep W₀ and apply adapters unmerged). Only the small A, B pairs are stored per task.</details>

7. What does subspace similarity between r = 8 and r = 64 adapters tell us?
<details><summary>Answer</summary>If the top directions learned with r = 8 lie inside those learned with r = 64, the important update directions are few and stable. The rest of the r = 64 directions carry mostly noise, which supports a low intrinsic rank.</details>

8. What did Table 7 show about the relation between ΔW and W?
<details><summary>Answer</summary>ΔW is correlated with W (much more than random) but not along W's top singular directions. It amplifies features already present but not emphasised in W, by a large factor (~21.5 for r = 4 in GPT-3).</details>

9. GPT-3 with r = 4 on Wq, Wv: compute the trainable parameters.
<details><summary>Answer</summary>96 × 2 × 4 × (12,288 + 12,288) = 18,874,368 ≈ 18.9M.</details>

10. In our toy, why did full fine-tuning lose ascending sort while LoRA kept it?
<details><summary>Answer</summary>Full fine-tuning changes the shared weights to serve the new behaviour, overwriting the old one. LoRA leaves W₀ frozen: turning the adapter off gives back the exact pre-trained model.</details>
