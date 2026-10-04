# GPipe, explained simply

**Paper:** Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen (Google), *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism*, NeurIPS 2019.

**In one sentence:** a model too big for one accelerator is cut into **K consecutive pieces** (stages), one per accelerator. Each mini-batch is split into **M micro-batches** that flow through the stages like an assembly line, so all accelerators work at once. Gradients are summed and applied once per mini-batch (fully synchronous), and only stage-boundary activations are stored (**re-materialisation**). This trained a 557M-parameter AmoebaNet (84.4% ImageNet top-1) and a 6B-parameter, 128-layer translation model.

---

## 1. The problem
- **Bigger models are more accurate,** but one accelerator's memory limits model size.
- **Naive model parallelism** (layers 1–10 on GPU 1, 11–20 on GPU 2, …) works, but **only one device is busy at a time**: GPU 2 waits for GPU 1's output, and so on.
- **Data parallelism** doesn't help when the model itself doesn't fit.

---

## 2. GPipe's three ingredients

### (1) Partition into cells
- **The model** is written as a sequence of L layers, which are grouped into **K cells**, one per accelerator.
- **The heuristic:** balance the estimated compute cost per cell.
- **Communication** happens only at the partition boundaries, so fast interconnects are not required.

### (2) Micro-batch pipelining
- **Split the mini-batch** of N examples into M micro-batches.
- **Accelerator k processes micro-batch m while accelerator k + 1 processes micro-batch m − 1,** like an assembly line.
- **Order:** all forward passes run first, then all backward passes in reverse.
- **Synchronous update:** gradients from all micro-batches are **accumulated** and applied **once** at the end of the mini-batch. The update is **identical** to training on the full mini-batch on one device, so there is no staleness. Results don't depend on K, which makes the method easy to use.

### The bubble
- **The idle time:** at the start (filling the pipe) and end (draining it) of each mini-batch, some stages sit idle.
- **For balanced stages** the idle fraction is:
```
bubble = (K − 1) / (M + K − 1)
```

**Worked example:**

| K | M | Bubble |
|---|---|---|
| 4 | 1 | 3/4 = 75% (naive model parallelism) |
| 4 | 4 | 3/7 ≈ 43% |
| 4 | 32 | 3/35 ≈ 8.6% |

- **The paper found the bubble "negligible" when M ≥ 4K,** partly because re-computation in the backward pass can start early.

### (3) Re-materialisation (activation checkpointing)
- **Normal back-propagation keeps every layer's activations:** O(N × L) memory.
- **GPipe stores only each cell's input** (the partition boundary) for every micro-batch. In the backward pass, each cell **recomputes** its forward pass to get its internal activations.
- **The peak activation memory becomes:**
```
O(N + (L/K) · (N/M))
```
  - boundary inputs for all N examples;
  - plus one cell's internal activations for one micro-batch.
- **The cost:** about one extra forward pass, i.e. ~25–33% more compute.

---

## 3. Results

### Maximum model size (Table 1)
- **AmoebaNet** (TPUv2, 8 GB per accelerator):
  - 82M parameters fit on one accelerator without GPipe;
  - **318M** with re-materialisation (activation memory 6.26 GB → 3.46 GB);
  - **1.8B on 8 accelerators**, 25× more.
- **Transformer** (TPUv3, 16 GB per core):
  - 2.7× larger on one core;
  - **83.9B parameters on 128 partitions**, 298× more.
  - It scales linearly because all layers are identical.

### Throughput (Table 2, normalised to K = 2, M = 1)

| | M | K = 2 | K = 4 | K = 8 |
|---|---|---|---|---|
| AmoebaNet | 1 | 1 | 1.13 | 1.38 |
| AmoebaNet | 4 | 1.07 | 1.26 | 1.72 |
| AmoebaNet | 32 | 1.21 | 1.84 | 3.48 |
| Transformer | 1 | 1 | 1.07 | 1.3 |
| Transformer | 4 | 1.7 | 3.2 | 4.8 |
| Transformer | 32 | 1.8 | 3.4 | **6.3** |

- **The Transformer gets 3.5× from 4× more accelerators,** i.e. nearly linear.
- **AmoebaNet is sub-linear** because its layers have **imbalanced** compute and memory, so the slowest stage sets the pace.

### Applications
- **AmoebaNet-B (18, 512):** 557M parameters, 480×480 inputs, 4 partitions, **84.4% top-1** on ImageNet-2012. It transfers well to other datasets.
- **A 128-layer, 6B-parameter multilingual Transformer** for 103 languages (102 → English), trained as **one model**. It beats strong bilingual baselines, with gains largest for low-resource languages. Deeper models help more than wider ones at equal parameter count. Larger batches help BLEU.

---

## 4. Why it matters
- **GPipe (with PipeDream, paper 085) established pipeline parallelism,** one of the three axes of large-model training:
  - **data parallelism** (ZeRO, paper 086);
  - **tensor parallelism** (Megatron);
  - **pipeline parallelism** (GPipe / PipeDream).
- **Its fill–drain schedule and micro-batching are built into DeepSpeed, Megatron-LM and PyTorch's `torch.distributed.pipelining`** (which has a `ScheduleGPipe`).
- **Activation re-materialisation (gradient checkpointing)** became a standard memory tool.

---

## 5. What our code found
- **Schedule** (K = 4, M = 4, backward costing 2 forwards): an ASCII timeline shows the staircase fill and drain, and the measured idle fraction **0.429 equals (K−1)/(M+K−1)**.
- **The bubble shrinks with M:** 75% → 42.9% → 27.3% → 8.6% for M = 1, 4, 8, 32 at K = 4.

**A real micro-batched pipeline** (12-layer MLP, K = 4 stages, M = 8 micro-batches of a 64-example batch, run in GPipe order):
- **Loss 1.212068, exactly the full-batch loss.** The gradients match ordinary back-propagation to **1e-8**, with or without re-materialisation, so GPipe is exactly synchronous.
- **Re-materialisation stored 16,384 activation values instead of 114,752 (7× less),** plus 3,072 values for one stage's activations for one micro-batch while re-computing.

**Normalised throughput** (ours vs paper, M = 32):

| | K = 2 | K = 4 | K = 8 |
|---|---|---|---|
| Uniform layers (Transformer-like) | 1.94 (1.8) | 3.66 (3.4) | **6.56 (6.3)** |
| Heavy-tailed layers (AmoebaNet-like; one layer = 23% of the work) | 1.89 (1.21) | 3.29 (1.84) | 3.89 (3.48) |

- **Balanced models scale nearly linearly once M ≫ K; imbalanced ones are capped by their slowest stage.**
- **Honest notes:**
  - our M = 1 rows stay at 1.0 for every K (one micro-batch can't overlap), while the paper's grow because it enlarged the batch when more partitions freed memory;
  - individual cells differ, since we model compute only.

**`experiments.py`:**
- **E1:** a large K × M × backward-ratio × remat grid;
- **E2:** our partitioner vs the optimal dynamic-programming partition;
- **E3:** how per-micro-batch batch-norm statistics differ from full-batch ones;
- **E4:** real `torch.utils.checkpoint` memory;
- **E5:** a check for PyTorch's `ScheduleGPipe`.
- None were run here.

---

## 6. Check yourself

1. Why is naive model parallelism slow?
<details><summary>Answer</summary>Each layer group depends on the previous group's output, so with one batch only one device works at a time and the others wait. With K devices, utilisation is about 1/K.</details>

2. Compute the bubble fraction for K = 8 and M = 32.
<details><summary>Answer</summary>(8 − 1)/(32 + 8 − 1) = 7/39 ≈ 17.9%.</details>

3. Why does GPipe give exactly the same update as single-device training?
<details><summary>Answer</summary>It accumulates the gradients of all micro-batches computed with the same parameters, and applies them once. The sum of micro-batch gradients equals the full mini-batch gradient. (Batch norm is the exception: its statistics are per micro-batch.)</details>

4. What does re-materialisation store, and what does it cost?
<details><summary>Answer</summary>Only each stage's input activations for every micro-batch. In the backward pass each stage recomputes its forward to rebuild its internal activations, costing roughly one extra forward pass of compute.</details>

5. Write GPipe's peak activation memory with re-materialisation.
<details><summary>Answer</summary>O(N + (L/K)·(N/M)): boundary activations for all N examples, plus the activations of one stage (L/K layers) for one micro-batch (N/M examples).</details>

6. Why did AmoebaNet scale sub-linearly while the Transformer scaled almost linearly?
<details><summary>Answer</summary>The Transformer's layers are identical, so stages are balanced. AmoebaNet's layers differ a lot in compute and memory, so some stages are heavier, and the pipeline runs at the speed of the slowest stage.</details>

7. Why does GPipe need little communication bandwidth?
<details><summary>Answer</summary>Only the activations at partition boundaries (and their gradients) move between accelerators, which is small compared with all-reducing gradients of all parameters as in data parallelism.</details>

8. What is the rule of thumb for M?
<details><summary>Answer</summary>M ≥ 4K micro-batches makes the bubble overhead negligible (at K = 4, M = 16 gives a bubble of 3/19 ≈ 16%, which the earlier start of re-computation partly hides).</details>

9. How large a model did GPipe train, and why did the Transformer scale to 298×?
<details><summary>Answer</summary>Up to 83.9B parameters over 128 partitions. Every Transformer layer has the same size, so each extra partition holds the same number of layers and capacity grows linearly with partitions.</details>

10. In our toy, why does pipelining with M = 1 give no speedup over K = 2?
<details><summary>Answer</summary>With one micro-batch, the stages run strictly one after another (fill then drain), so adding stages adds no overlap. Only splitting into more micro-batches lets stages work simultaneously.</details>
