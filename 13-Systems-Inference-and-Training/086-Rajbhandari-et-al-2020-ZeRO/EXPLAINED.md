# ZeRO, explained simply

**Paper:** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (Microsoft), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, SC 2020.

**In one sentence:** in ordinary data-parallel training every GPU keeps a **full copy** of the parameters, gradients and optimiser state, which is pure redundancy. **ZeRO** (Zero Redundancy Optimizer) **partitions** these "model states" across the data-parallel GPUs instead: each GPU owns 1/N_d of them and fetches the rest only when needed. Memory per GPU drops up to N_d-fold, with the same or 1.5× the communication. This let DeepSpeed train 100B+ parameter models and the 17B Turing-NLG.

---

## 1. Where the memory goes

### Model states with mixed-precision Adam (per parameter)

| Item | Bytes |
|---|---|
| fp16 parameter (for forward/backward) | 2 |
| fp16 gradient | 2 |
| fp32 master copy of the parameter | 4 |
| fp32 Adam momentum | 4 |
| fp32 Adam variance | 4 |
| **Total** | **2 + 2 + K = 16**, with K = 12 for Adam |

- **Example:** a 7.5B-parameter model needs 16 × 7.5e9 = **120 GB**, more than any GPU, and plain data parallelism (DP) **replicates all of it on every GPU**.
- **The limit:** DP runs out of memory beyond about **1.4B parameters** on 32 GB GPUs once activations are counted.

### Residual states
- activations;
- temporary buffers;
- memory fragmentation.

### Why not just model parallelism (Megatron)?
- **It splits each layer across GPUs.** That works within one node, but it is communication-heavy across nodes: a 40B model across two DGX-2 nodes ran at only about 5 TFlops per GPU.
- **It requires rewriting the model.**

---

## 2. ZeRO-DP: three cumulative stages

| Stage | What is partitioned | Memory per GPU | Example (Ψ = 7.5B, N_d = 64) | Communication |
|---|---|---|---|---|
| Plain DP | nothing | (2 + 2 + K)Ψ = 16Ψ | **120 GB** | 2Ψ |
| **P_os** | optimizer states | 2Ψ + 2Ψ + KΨ/N_d | **31.4 GB (4× less)** | 2Ψ (same) |
| **P_os+g** | + gradients | 2Ψ + (2 + K)Ψ/N_d | **16.6 GB (8× less)** | 2Ψ (same) |
| **P_os+g+p** | + parameters | (2 + 2 + K)Ψ/N_d | **1.9 GB (64× less)** | **3Ψ (1.5×)** |

### How each stage runs a step
- **P_os:** each rank computes the full gradient on its data. Gradients are **reduce-scattered**, so rank r receives the summed gradient for its 1/N_d shard. It updates that shard with its own Adam state, and the updated fp16 parameters are **all-gathered**. Total traffic: Ψ + Ψ = 2Ψ, the same as DP's all-reduce (which is itself reduce-scatter + all-gather).
- **P_os+g:** the same, but each rank keeps only its gradient shard. Gradients are reduced in buckets during the backward pass and the rest freed.
- **P_os+g+p:** each rank stores only its parameter shard. Before each layer's forward pass, its parameters are all-gathered from their owners and freed after use; the same happens again in the backward pass. With the gradient reduce-scatter that totals Ψ + Ψ + Ψ = **3Ψ**.

### Worked example: P_os+g on a 7.5B model, N_d = 64
- 2Ψ = 15 GB (fp16 parameters);
- plus 14Ψ/64 = 105 GB/64 = 1.64 GB;
- total **16.6 GB**.

### A trillion parameters (Table 1)
| N_d | P_os | P_os+g | P_os+g+p |
|---|---|---|---|
| 1 | 16,000 GB | 16,000 GB | 16,000 GB |
| 64 | 4,187 GB | 2,218 GB | 250 GB |
| 1024 | 4,011 GB | 2,013 GB | **15.6 GB** |

Only full partitioning fits 1T parameters on 32 GB GPUs: 16 TB / 1024 = 15.6 GB each.

---

## 3. ZeRO-R: the residual states
- **P_a, partitioned activation checkpoints.** With model parallelism, each model-parallel GPU normally stores a **replicated** copy of the activation checkpoints; ZeRO partitions them across the model-parallel ranks instead. Example: a 100B model (batch 32, sequence 1024, model-parallel degree 16) needs about **33 GB** of checkpoints per GPU, which falls to **about 2 GB**. **P_a+cpu** moves them to CPU memory, to nearly 0.
- **C_B, constant-size buffers.** Libraries fuse all gradients into one big buffer for fast all-reduce; for a 3B model a 32-bit fused buffer is 12 GB. ZeRO uses a fixed-size buffer instead.
- **M_D, memory defragmentation.** Activation checkpoints and gradients live long while other tensors are short-lived, which fragments memory. ZeRO copies the long-lived ones into pre-allocated contiguous memory.

---

## 4. Results (ZeRO-100B = P_os+g + ZeRO-R)
- **Over 100B parameters on 400 V100 GPUs at 15 PFlops** (about 38 TFlops per GPU), with **super-linear** scaling: more GPUs leave more memory per GPU for larger batches.
- **That is 8× larger models and 10× higher throughput** than the state of the art (Megatron-LM, which can't scale efficiently beyond ~40B).
- **Up to 13B parameters without any model parallelism,** larger than Megatron GPT 8.3B and T5 11B, so ordinary data-parallel code just works.
- **It trained Turing-NLG (17B),** the largest language model at the time, with a record perplexity.
- **The analysis shows all three stages could train a 1T-parameter model on 1024 GPUs.**

---

## 5. Why it matters
- **ZeRO (DeepSpeed) and its twin, PyTorch FSDP** (Fully Sharded Data Parallel, essentially ZeRO-3), are standard ways to train large models. ZeRO-Offload/Infinity extend it to CPU and NVMe.
- **It changed the trade-off:** data parallelism no longer means replication. Model and pipeline parallelism (GPipe, PipeDream, Megatron) are needed for speed or activation memory, not just to fit the weights.

---

## 6. What our code found
- **Memory formulas reproduce the paper exactly:**
  - Figure 1: **120 / 31.4 / 16.6 / 1.9 GB**;
  - Table 1: e.g. 7,000 / 5,500 / 4,000 GB at N_d = 4 and **15.6 GB** at N_d = 1024 for 1T parameters.
- **Largest model whose model states fit in 32 GB:** plain DP is stuck at **2B** at any N_d. With N_d = 64: P_os 7.6B, P_os+g 14.4B, P_os+g+p **128B**.

**A simulated data-parallel job** (2,177-parameter MLP, mixed-precision Adam with fp16 parameters/gradients and fp32 master/moments, 50 steps):

| N_d = 16 | Bytes per parameter per rank | Sent per rank per step |
|---|---|---|
| Plain DP | 16.00 | 1.88 Ψ |
| P_os | 4.76 | 1.88 Ψ |
| P_os+g | 2.88 | 1.88 Ψ |
| P_os+g+p | 1.01 | **2.81 Ψ (1.5×)** |

- **All four stages end with bit-identical weights.** ZeRO only changes where the states live, not the arithmetic.
- **The numbers match the formulas:**
  - bytes per parameter: 16, 4 + 12/N_d, 2 + 14/N_d, 16/N_d;
  - communication: 2(N_d − 1)/N_d · Ψ, and 3(N_d − 1)/N_d · Ψ for the last stage.
- **Simplification:** our simulation computes each full gradient at once, whereas real ZeRO frees buckets layer by layer.

**ZeRO-R:** our P_a arithmetic gives 33 GB / 16 = 2.1 GB, matching the paper's "about 2 GB".

**`experiments.py`:**
- **E1:** a memory grid over model size × N_d × K × stage, with fits on 16/32/80 GB GPUs;
- **E2:** the simulated cluster for N_d up to 32;
- **E3:** a step-time model of when the 1.5× communication matters;
- **E4:** a recipe for real FSDP vs DDP.
- None were run here.

---

## 7. Check yourself

1. Why does mixed-precision Adam need 16 bytes per parameter?
<details><summary>Answer</summary>fp16 parameter (2) + fp16 gradient (2) + fp32 master parameter, momentum and variance (4 + 4 + 4 = 12) = 16 bytes.</details>

2. Compute P_os memory per GPU for Ψ = 10B parameters and N_d = 100.
<details><summary>Answer</summary>4Ψ + 12Ψ/N_d = 40 GB + 120 GB/100 = 41.2 GB.</details>

3. Why do P_os and P_os+g cost no extra communication?
<details><summary>Answer</summary>DP's all-reduce is a reduce-scatter followed by an all-gather (2Ψ). ZeRO does the reduce-scatter on gradients, updates each shard locally, then all-gathers the updated parameters, which is the same two collectives and the same volume.</details>

4. Why does P_os+g+p cost 1.5× communication?
<details><summary>Answer</summary>Parameters must be all-gathered for the forward pass (Ψ) and again for the backward pass (Ψ), since they were freed, plus the gradient reduce-scatter (Ψ): 3Ψ instead of 2Ψ.</details>

5. What is the memory reduction of P_os+g+p, and what does it enable?
<details><summary>Answer</summary>N_d× (memory = 16Ψ/N_d). With 1024 GPUs a 1T-parameter model's 16 TB of states becomes about 16 GB per GPU.</details>

6. Why does plain DP fail beyond about 1.4B parameters on 32 GB GPUs?
<details><summary>Answer</summary>Every GPU holds all 16Ψ bytes of model states plus activations; 16 × 1.4B ≈ 22 GB leaves little room for activations and buffers on a 32 GB GPU.</details>

7. What does P_a (partitioned activation checkpointing) do?
<details><summary>Answer</summary>With model parallelism, each MP rank would store a full replicated copy of the activation checkpoints. P_a stores 1/N_m of them per rank and gathers when needed, e.g. 33 GB → ~2 GB for the paper's 100B example.</details>

8. Why did ZeRO-100B show super-linear speedup?
<details><summary>Answer</summary>With more GPUs, each holds a smaller share of the model states, freeing memory for larger per-GPU batches, which raises per-GPU efficiency. Throughput grew faster than the GPU count.</details>

9. How does ZeRO relate to model parallelism?
<details><summary>Answer</summary>ZeRO removes the memory reason for model parallelism (up to 13B could train with pure DP), but can be combined with it (ZeRO-100B with MP trained up to 170B) when activations or per-GPU compute need splitting too.</details>

10. In our simulation, why were the final weights bit-identical across stages?
<details><summary>Answer</summary>Every stage computes the same gradients, sums them in the same way, and applies the same Adam update to every parameter. Partitioning only decides which rank stores and updates each piece, not what is computed.</details>
