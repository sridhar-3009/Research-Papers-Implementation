# PipeDream, explained simply

**Paper:** Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, Matei Zaharia (Microsoft Research, Carnegie Mellon, Stanford), *PipeDream: Generalized Pipeline Parallelism for DNN Training* (SOSP 2019). The PDF in this folder is the arXiv 2018 version, *PipeDream: Fast and Efficient Pipeline Parallel DNN Training*.

**In one sentence:** data-parallel training spends most of its time synchronising weights when models are large. PipeDream instead splits the model into **pipeline stages** (optionally replicating some stages), keeps every GPU busy with a **one-forward-one-backward (1F1B)** schedule that never flushes the pipeline, and fixes the resulting weight-version mix-ups with **weight stashing**. It trains up to **5× faster** than data parallelism.

---

## 1. Why not just data parallelism?
- **Data parallelism** (each GPU has a full copy of the model; gradients are all-reduced every step) must communicate **all the weights** every minibatch.
- **For models with many weights** (e.g. VGG-16's huge fully-connected layers) on fast GPUs, communication can take most of the time (up to 85% of total training time in the paper's measurements).
- **Model or pipeline parallelism** only sends **activations** at stage boundaries, which is often far less data. But naive pipelines leave GPUs idle, and asynchronous ones mix weight versions.

---

## 2. The three pieces

### (1) Automatic partitioning (with replication)
- **Profiling first:** a short run measures, for every layer, its compute time, weight size and output-activation size.
- **A dynamic program then chooses:**
  - where to cut the layers into stages;
  - **how many machines replicate each stage** (data parallelism inside a stage);
  - the objective is to minimise the time of the slowest stage.
- **The recurrence**, for layers 1..j on m machines:
```
A(j, m) = min over i < j, m' < m of  max( A(i, m − m'),  2·C_i,  T(i+1 → j, m') )
T(i → j, m) = (1/m) · max( compute of layers i..j,  weight-synchronisation time of layers i..j over m replicas )
```
  - C_i is the time to send layer i's activations to the next stage.
  - The complexity is O(N²M²).
- **Notation:** "2-1-1" means 3 stages, the first replicated on 2 machines. Plain data parallelism on 8 machines is "8".
- **Intuition for VGG-16:** convolution layers have lots of compute but few weights, so it is cheap to replicate them. The fully-connected layers have huge weights but little compute, so put them on one machine and **never all-reduce** them. That gives **"7-1"**.

### (2) 1F1B scheduling
- **Startup:** the input stage admits **NOAM** minibatches, where NOAM = ⌈#machines / #machines in the input stage⌉, to fill the pipeline.
- **Steady state:** each stage alternates one **forward** pass (for a newer minibatch) and one **backward** pass (for an older one). In a balanced pipeline **no GPU is idle**, and learning progresses continuously, with no flush per update as in GPipe (084).
- **Memory:** stage k holds at most (stages − k) minibatches' activations.
- **Replicated stages** receive minibatches round-robin (minibatch ID mod replicas).

### (3) Weight stashing (and vertical sync)
- **The problem:** the pipeline never stops, so between a minibatch's forward and backward pass on a stage, **other minibatches update the weights**.
  - **Naive pipelining** does the forward with one version and the backward with another. The result is not a valid gradient of the loss at any single weight setting, and "naive pipelining does not achieve the same accuracy as data-parallel training".
  - Different stages also see different staleness (the last stage none, the first stage the most).
- **Weight stashing:** keep a copy of the weights each in-flight minibatch used in its forward pass, and **use the same copy in its backward pass**.
  - Within a stage, the gradient is then consistent: stage k's gradient is computed at weights from n − k + 1 updates ago.
  - The memory cost is one weight copy per in-flight minibatch.
```
w(t+1) = w(t) − ν · ∇f( w₁(t−n+1), w₂(t−n+2), …, wₙ(t) )        (weight stashing)
```
- **Vertical sync:** additionally make **all** stages use the version from when the minibatch entered the pipeline:
```
w(t+1) = w(t) − ν · ∇f( w₁(t−n+1), …, wₙ(t−n+1) )
```
  The paper says this is semantically like BSP data parallelism, but found its impact **negligible** and turns it off by default.

---

## 3. Results (arXiv version, Table 1)

| Model | Machines (cluster) | PipeDream config | Speed-up over BSP | Communication reduction |
|---|---|---|---|---|
| VGG16 | 4 (A) | 2-1-1 | 2.13× | 90% |
| VGG16 | 8 (A) | 7-1 | 2.99× | 95% |
| VGG16 | 16 (A) | 9-5-1-1 | 3.00× | 91% |
| VGG16 | 8 (B, faster GPUs) | **7-1** | **5.12×** | 95% |
| Inception-v3 | 8 (A) | 8 (pure data parallel) | 1.00× | 0% |
| Inception-v3 | 8 (B) | 7-1 | 1.45× | 47% |
| S2VT (video captioning) | 4 (A) | 2-1-1 | 3.01× | 95% |

- **Inception-v3 on cluster A:** the partitioner correctly chooses **plain data parallelism**, because communication isn't a bottleneck there.
- **Faster GPUs (cluster B)** make communication relatively more expensive, so pipelining helps more.

---

## 4. Why it matters
- **PipeDream introduced 1F1B scheduling,** now the standard pipeline schedule (Megatron-LM, DeepSpeed, PyTorch's `Schedule1F1B`), usually in a *synchronous* form (PipeDream-Flush / 1F1B with flushes) to avoid weight versions.
- **It introduced automatic hybrid partitioning:** pipeline + data parallelism chosen by a profiler-driven optimiser, an idea continued in Alpa and other auto-parallelisation work.
- **It brought the weight-version / staleness question into pipeline training,** with weight stashing as the answer.

---

## 5. What our code found

**Partitioner** on a VGG-like 12-layer profile (convolutions: most compute, few weights; fully-connected: 48 of 52 weight units, little compute):

| Machines | Network | Config | Speed-up vs data parallel |
|---|---|---|---|
| 4 | slow | 3-1 | 3.02× |
| 8 | slow | **7-1** | 3.65× |
| 16 | slow | 15-1 | 3.92× |
| any | fast | pure data parallel ("4", "8", "16") | 1.00× |

- **On a slow network the DP rediscovers the paper's VGG-16 layout, "7-1".**
- **On a fast network,** data parallelism is already best, as the partitioner chose for Inception-v3 on cluster A.

**Schedules** (4 balanced stages, 16 minibatches):

| Schedule | Time units | Minibatches held per stage |
|---|---|---|
| 1F1B, never flushing | **57** | [4, 3, 2, 1] |
| Flush every 4 (GPipe-style) | 84 | [4, 4, 4, 4] |

**Weight versions** (MLP split into S stages with 1F1B steady-state staleness; median final loss over 5 seeds):

| Stages | lr | Plain SGD | Naive | Weight stashing | Vertical sync |
|---|---|---|---|---|---|
| 4 | 0.1 | 0.0215 | 0.0219 | 0.0219 | 0.0236 |
| 4 | 0.2 | 0.0152 | 0.0166 | 0.0166 | diverged |
| 8 | 0.1 | 0.0207 | 0.0259 | 0.0248 | diverged |
| 8 | 0.2 | 0.0167 | 0.0284 | 0.0413 | diverged |

- **Pipelining costs accuracy at a fixed step budget, more with more stages,** because of staleness.
- **Honest notes:**
  1. **We do not reproduce naive pipelining failing.** On this small smooth MLP, mixing weight versions hurt no more than stashing's staleness; at 8 stages with lr 0.2, naive was even better. Weight stashing's real guarantee is a **valid** gradient, which the paper needed on large CNNs and RNNs.
  2. **Vertical sync diverged at these learning rates,** because it makes every stage S − 1 updates stale. The paper found it unnecessary.

**`experiments.py`:**
- **E1:** the partitioner on random profiles × machines × bandwidth;
- **E2:** schedules with imbalance;
- **E3:** a larger weight-semantics study (10 seeds, up to 16 stages);
- **E4:** a check for PyTorch's `Schedule1F1B`.
- None were run here.

---

## 6. Check yourself

1. Why can data parallelism be slow for VGG-16-like models?
<details><summary>Answer</summary>Every step all-reduces every weight, and VGG-16's fully-connected layers have most of the weights (over 100M), so communication dominates, especially on fast GPUs where compute is quick.</details>

2. What does the configuration "9-5-1-1" mean?
<details><summary>Answer</summary>Four pipeline stages: the first replicated on 9 machines, the second on 5, the last two on one machine each (16 machines in total).</details>

3. Compute NOAM for the configuration 7-1.
<details><summary>Answer</summary>⌈8 / 7⌉ = 2 minibatches admitted to keep the pipeline full.</details>

4. What does 1F1B mean, and what does it achieve?
<details><summary>Answer</summary>One forward, one backward: in steady state each stage alternates a forward pass for a new minibatch and a backward pass for an older one, so balanced stages are never idle and the pipeline never has to drain.</details>

5. Why does naive pipelining produce an invalid gradient?
<details><summary>Answer</summary>The forward pass of a minibatch used one weight version, but by its backward pass other minibatches have updated the weights. The backward combines activations from old weights with new weights, which is not the gradient of the loss at any single set of weights.</details>

6. What does weight stashing store, and what does it guarantee?
<details><summary>Answer</summary>The weight version each in-flight minibatch used in its forward pass on that stage. Reusing it in the backward makes the stage's gradient a correct gradient for those weights (stale, but consistent).</details>

7. How stale are stage 1's weights under weight stashing with n stages?
<details><summary>Answer</summary>Its gradient is computed at weights from about n − 1 updates earlier (w₁(t − n + 1)), while the last stage uses the current weights.</details>

8. What does vertical sync add, and did the paper use it?
<details><summary>Answer</summary>All stages use the same weight version (the one current when the minibatch entered the pipeline), giving BSP-like semantics. The paper found the benefit negligible and leaves it off by default.</details>

9. Why did PipeDream choose plain data parallelism for Inception-v3 on cluster A?
<details><summary>Answer</summary>Inception-v3 has relatively few weights for its compute, and on that cluster weight synchronisation wasn't the bottleneck, so splitting the model wouldn't help. The partitioner's DP found that one replicated stage is fastest.</details>

10. In our toy, why did the partitioner put the last two layers alone on one machine?
<details><summary>Answer</summary>Those layers hold 48 of 52 weight units but little compute. Replicating them would require all-reducing those weights over the slow network, while one machine handles their small compute without any weight synchronisation.</details>
