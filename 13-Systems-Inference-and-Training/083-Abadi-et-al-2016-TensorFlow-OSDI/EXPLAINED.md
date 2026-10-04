# TensorFlow: A System for Large-Scale Machine Learning (OSDI 2016), explained simply

**Paper:** Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, Xiaoqiang Zheng (Google Brain), *TensorFlow: A System for Large-Scale Machine Learning*, OSDI 2016.

**In one sentence:** the earlier "parameter server" systems built shared parameters and their update rules into privileged system code. TensorFlow instead expresses **everything** as one dataflow graph with **mutable state**: the parameters, the update rules, the sharding, the checkpointing and the synchronisation. Researchers can then change any of it without touching the system.

---

## 1. Background: parameter servers
- **The architecture** (DistBelief, Project Adam, Parameter Server by Li et al.):
  - **worker** tasks compute gradients on data shards;
  - **parameter server (PS)** tasks hold the model and apply updates.
- **The update rule** (e.g. W ← W − α·∂L/∂W) is hard-coded on the PS. Trying a new optimiser, sparse layer or consistency model means changing the system's C++ code.

### TensorFlow's answer
- **Variables are graph nodes** holding mutable tensors, placed on PS tasks.
- **Updates are graph nodes too:** Assign, AssignAdd, ScatterSub, and any composition of them.
- **So the PS is just devices running ordinary ops.**
- **Users get, in their own code:**
  - new optimisers (momentum, AdaGrad, Adam);
  - sparse updates;
  - custom synchronisation.

---

## 2. Key mechanisms

### Sparse embedding layers (Figure 3)
- **The scale:** a language model's embedding matrix can have hundreds of millions of rows, and some document models have **terabytes** of parameters. It is sharded over many PS tasks.
- **The lookup is built from primitive ops:**
  1. **Part** (dynamic partition) splits the batch's word ids by shard;
  2. **Gather** on each shard reads only the needed rows;
  3. **Stitch** (dynamic stitch) reassembles the rows in the original order.
- **The gradient is sparse:** it touches only the gathered rows, applied with ScatterSub on each shard.

### Sampled softmax
- **The problem:** the output layer of a language model with an 800,000-word vocabulary is huge.
- **The fix:** during training, score only the true word plus a few **sampled** words. With 512 samples the softmax data transfer and computation fall by **78×**.

### Fault tolerance (Section 4.3)
- **No per-operation fault tolerance:** failures are not that frequent, and per-op tolerance would be costly.
- **Instead, Save and Restore are ops:** a checkpointing subgraph runs periodically, and after a failure the client restores the latest checkpoint.
- **Policies are the user's choice,** e.g. keep the best checkpoint by validation score.
- **Consistency:** checkpoints are not consistent with asynchronous training (and needn't be). With synchronous training, checkpoint after the update step.

### Synchronous replica coordination (Section 4.4, Figure 4)

| Scheme | How it works | Trade-off |
|---|---|---|
| (a) **Asynchronous** | each worker reads the parameters, computes a gradient and applies it independently | high utilisation, but steps use **stale** parameters |
| (b) **Synchronous** | gradients from all n workers are aggregated, then applied once | fresh gradients, but every step waits for the **slowest worker** (straggler) |
| (c) **Synchronous with backup workers** | run n + b workers, take the **first n** gradients, discard the rest | like MapReduce backup tasks, but proactive; stragglers stop mattering, at the cost of extra machines and server traffic |

---

## 3. Results
- **Single machine (Table 1):** faster than Caffe and within 6% of Torch on four CNNs (both use cuDNN). Neon is faster on three, thanks to hand-written assembly kernels.
- **Synchronous microbenchmark:**
  - null-step time grows from 1.8 ms (1 worker) to 8.8 ms (100 workers);
  - with real work, the median synchronous step is about **10% longer** than asynchronous, and much worse above the 90th percentile (stragglers).
- **Inception-v3 training** scales to **2,300 images/s with 200 workers,** with diminishing returns.
- **Backup workers (Figure 8, 50-worker Inception):**
  - **4 backups** give the shortest median step (**1.93 s**);
  - **3 backups** give the best **normalised speed-up**, which discounts the extra machines: **9.5%**;
  - a 5th backup slightly hurts, because the 51st worker's result is likely a non-straggler's and adds parameter-server traffic.
  - Overall, backups improve throughput by up to 15%.
- **Language modelling** (One Billion Word, 800k vocabulary): more PS tasks raise throughput by parallelising the softmax, and sampled softmax gives a large further boost.

---

## 4. Why it matters
- **TensorFlow 1.x's design** (graph + Variables + PS tasks + Save/Restore + SyncReplicasOptimizer) powered production ML at Google and elsewhere for years.
- **The broader lesson outlived the parameter server:** make the system's policies (updates, sharding, synchronisation) ordinary programmable code. Later systems shifted to synchronous all-reduce (e.g. MirroredStrategy, PyTorch DDP), and many follow-ups re-examined sync vs async training.
- **Backup workers and sampled softmax** are standard tools.

---

## 5. What our code found
The code builds on paper 082's mini-TensorFlow.

**Sharded embedding as dataflow:**
- A 100,000 × 16 table split over 4 PS shards. Part → Gather → Stitch for a batch of 32 ids **equals the dense lookup**.
- It reads 32 rows (2 KiB) instead of 6.1 MiB.
- The sparse update on shard 0 touched only its 11 gathered rows.

**Sampled softmax** (vocabulary 20,000):
- It touches at most 640 output rows per step instead of 20,000 (about 31× less; the paper's 78× is for 800,000 words).
- **Honest note:** per step it learned slightly more slowly here (loss 7.68 vs 7.59 after 150 steps), but ran 12× faster (0.10 s vs 1.19 s).

**Checkpoints** (1,000 steps, 2 failures, a save costs 2 steps):

| Interval | Wall time |
|---|---|
| Every 10 | 1,200 |
| Every 50 | **1,060** |
| Every 100 | 1,090 |
| Every 500 | 1,574 |

Saving too often and too rarely both cost time.

**Backup workers** (50 workers; median step and normalised speed-up):

| b | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Median step | 1.828 | 1.805 | 1.783 | 1.602 | **1.599** | 1.602 | 1.607 |
| Normalised speed-up | 1.000 | 0.993 | 0.986 | **1.077** | 1.059 | 1.037 | 1.016 |

- **The paper's shape:** shortest step at b = 4, best normalised value at b = 3, and worse beyond.
- **Honest note:** the straggler model (6% of workers 30% slower, plus a server cost per message) was **tuned** to produce this shape. What's real is the mechanism: order statistics plus server traffic.

**Async vs sync** (20 workers, logistic regression, simulated wall clock; loss at t = 5 / 20 / 40):

| Scheme | Updates | Mean staleness | Loss |
|---|---|---|---|
| Asynchronous | 772 | 18.7 | 0.483 / 0.294 / 0.232 |
| Synchronous | 33 | 0 | 0.523 / 0.316 / 0.249 |
| Synchronous + 3 backups | 39 | 0 | 0.490 / 0.298 / 0.234 |

- **Backup workers recover most of synchronous training's straggler loss.**
- **Honest note:** on this easy convex problem staleness barely hurts, so asynchronous training is slightly ahead. The paper reports encouraging synchronous results at scale.

**`experiments.py`:**
- **E1:** backups × workers × straggler model × server cost;
- **E2:** step-time percentiles, async vs sync;
- **E3:** learning-rate sensitivity;
- **E4:** sampled softmax vocabulary × samples;
- **E5:** real `tf.distribute` throughput.
- None were run here.

---

## 6. Check yourself

1. What does it mean that TensorFlow expresses the parameter server "as dataflow"?
<details><summary>Answer</summary>Parameters are Variable nodes placed on PS tasks, and updates are ordinary ops (Assign, ScatterSub, ...) in the same graph. There is no special server code, so users can write any update rule or sharding in their program.</details>

2. How is a sharded embedding lookup built?
<details><summary>Answer</summary>Part splits the ids by shard; Gather on each shard reads the requested rows; Stitch puts them back in the original order. The gradient flows back only to the gathered rows (a sparse update).</details>

3. Why does sampled softmax help, and by how much in the paper?
<details><summary>Answer</summary>The full softmax over 800,000 words needs every output row each step. Scoring only the true word and 512 samples cuts that softmax data transfer and computation by 78×.</details>

4. What are the drawbacks of asynchronous and of synchronous training?
<details><summary>Answer</summary>Asynchronous: gradients are computed on stale parameters, so each step is less effective. Synchronous: each step waits for the slowest worker, so stragglers set the pace.</details>

5. How do backup workers work?
<details><summary>Answer</summary>Launch n + b workers for each step, aggregate the first n gradients to arrive, and discard the rest. A few slow machines no longer delay the step.</details>

6. Why can too many backup workers hurt?
<details><summary>Answer</summary>Each extra worker costs a machine and sends traffic to the parameter servers. Once enough backups cover the typical number of stragglers, more just add load (the paper's 5th backup slightly increased the step time).</details>

7. What is the "normalised speed-up", and why did the best one need fewer backups than the fastest step?
<details><summary>Answer</summary>Speed-up × n/(n + b): it discounts the gain by the extra resources. The 4th backup still shortened the step slightly, but not enough to pay for another machine, so 3 was the best value.</details>

8. How does TensorFlow handle failures?
<details><summary>Answer</summary>With periodic checkpoints via Save ops (a checkpointing subgraph) and Restore on restart. There is no fine-grained per-op recovery, because failures are infrequent enough.</details>

9. Why aren't TensorFlow's checkpoints necessarily consistent, and when does that matter?
<details><summary>Answer</summary>Checkpointing runs concurrently with training, so it may capture some updates of a step but not others. That is fine for asynchronous training, which is inconsistent anyway; synchronous users checkpoint right after the update step.</details>

10. In our toy, why was asynchronous training slightly ahead despite staleness of ~19 updates?
<details><summary>Answer</summary>Logistic regression is convex and smooth, and each stale gradient still points roughly downhill, so many small slightly-stale updates make good progress. Staleness hurts more with larger learning rates or harder (non-convex, high-curvature) models.</details>
