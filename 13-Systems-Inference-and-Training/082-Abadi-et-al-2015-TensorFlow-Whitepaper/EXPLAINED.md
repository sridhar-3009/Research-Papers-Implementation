# TensorFlow (2015 whitepaper), explained simply

**Paper:** Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, and many others (Google Research), *TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems*, 2015.

**In one sentence:** describe a machine-learning computation as a **dataflow graph** (operations as nodes, tensors flowing along edges). Build the graph first and run it later, on anything from a phone to thousands of GPUs. The system **adds gradient nodes automatically**, **places** nodes on devices, and **inserts Send/Receive nodes** wherever data must cross devices.

---

## 1. The programming model
- **A graph of nodes (operations)** such as MatMul, Add, ReLU, SoftmaxCrossEntropy, Variable, Assign, Queue and Save.
- **Edges carry tensors** (n-dimensional arrays).
- **Control dependencies** are special edges that carry no data and only mean "run this first". They are needed because some nodes have **state** (Variables, queues).
- **Kernels:** each operation has kernels, i.e. implementations for particular devices (CPU, GPU).
- **Sessions:** a client builds the graph, then calls `session.run(fetches, feeds)` many times. Usually the same graph is run thousands of times (training steps).
- **Variables** are persistent mutable tensors that survive across runs. Parameters live there and are updated by Assign / AssignAdd nodes.

**Why a graph?** Because the whole computation is known before it runs, the system can **optimise** it (prune, deduplicate, schedule), **differentiate** it, **place** it on devices and **distribute** it.

---

## 2. How it runs

### Single device
- **The executor counts each node's unfinished inputs.** Nodes with zero pending inputs go to a **ready queue** and run in any order. Finishing a node decrements its consumers' counts.

### Multiple devices: placement (Section 3.2.1)
- **Run a simulated execution of the graph using a cost model:** estimated or measured time of each op on each kind of device, plus data-transfer time.
- **Greedy rule:** for each node, try every **feasible** device (one that has a kernel for the op), and pick the device where the node would **finish earliest**, counting the time to receive inputs from other devices.
- **Users can add constraints,** e.g. "only GPUs" or "colocate with node X". The feasible sets of colocated groups are intersected.

### Cross-device edges: Send / Receive (Section 3.2.2)
- **The rewrite:** every edge x → y with x and y on different devices becomes x → **Send** … **Recv** → y.
- **One Recv per (tensor, device),** shared by all consumers there, so the same tensor is not sent twice to one device.
- **All communication is isolated inside Send/Recv pairs.** Across machines they use TCP or RDMA, and the rest of the runtime never sees networking.

### Distributed execution
- **The same pattern across machines.** The master gives each worker its subgraph, and the Send/Recv pairs synchronise the workers.
- **Failures:** detected via Send/Recv errors and health checks. The whole graph execution restarts from checkpoints of the Variables.

---

## 3. Extensions (Section 4)
1. **Automatic gradients.** `tf.gradients(C, [X1, …])` walks **backward** from C and, for each op on the path, adds that op's registered **gradient function** as new nodes. Partial gradients are summed (AddN). The gradients are just more graph, so placement, partitioning and so on apply to them too.
2. **Partial execution.** `run(fetches, feeds)` executes **only the subgraph needed for the fetches**. Feeding a tensor replaces its producer: special feed and fetch nodes are inserted and the graph is pruned.
3. **Device constraints** (hints for placement).
4. **Control flow:** **Switch** (route a value to one of two outputs; the other output is "dead"), **Merge** (forward whichever input is alive), and **Enter / Leave / NextIteration** for loops. This is inspired by Arvind's dataflow machines, and lets if/while run *inside* the graph, even distributed.
5. **Input operations** that read data files directly, and **queues** for prefetching and batching.
6. **Containers** for long-lived shared state.

---

## 4. Optimisations (Section 5)
- **Common subexpression elimination:** layers of client code often build the same computation twice, so identical (op, inputs, attributes) nodes are merged (Click's algorithm).
- **Scheduling:** ASAP/ALAP analysis decides when Receive nodes should start, so remote data doesn't arrive too early and waste memory.
- **Asynchronous kernels** (e.g. Receive, Enqueue) don't block a thread while waiting.
- **Optimised libraries:** BLAS, cuBLAS, cuDNN and Eigen.
- **Lossy compression** of cross-device transfers: convert float32 to 16 bits by **dropping the low 16 bits of the mantissa** (keeping the sign, the 8-bit exponent and 7 mantissa bits), then pad with zeros on the other side. This halves the bytes at a relative error of at most 2⁻⁷.

---

## 5. Experiences and idioms (Sections 6–7)
- **Porting Inception from DistBelief** (TensorFlow's predecessor): 13.6M parameters, **36,000 operations** in the graph, and 2 billion multiply-adds per 224×224 image.
- **Lessons from the port:**
  - build tools to inspect parameters;
  - start small and scale up;
  - make sure the objective matches with learning turned off;
  - match a single-machine implementation before going distributed;
  - guard against numerical errors.
- **Some LSTM language models have over 15,000 nodes.**
- **Idioms:**
  - **synchronous and asynchronous data parallelism** (many replicas sharing parameters);
  - **model parallelism** (different parts of the model on different devices);
  - **concurrent steps** to pipeline computation.
- **TensorBoard** visualises graphs and summaries; **EEG** profiles performance.

---

## 6. Why it matters
- **TensorFlow became one of the two dominant deep-learning frameworks,** alongside PyTorch.
- **Its graph ideas live on** in TF/XLA, JAX, PyTorch's `torch.compile` / FX graphs, ONNX and model-serving runtimes:
  - automatic differentiation by graph extension;
  - placement;
  - Send/Recv partitioning;
  - in-graph control flow.
- **The 2016 OSDI paper** (083) explains the production design: parameter servers as ordinary graph nodes, mutable state, and fault tolerance.

---

## 7. What our code found
We built a miniature TensorFlow in about 300 lines.

- **Graph and gradients:** a 2-layer MLP graph has 12 nodes. `gradients(cost, [W1, b1, W2, b2])` extends it to 27 nodes, and the gradients match finite differences to 3e-11. Running the AssignSub update nodes for 300 steps lowers the cost from 0.782 to 0.045 (99.5% training accuracy).
- **Partial execution:**

| Request | Nodes executed (of 36) |
|---|---|
| Fetch the cost | 10 |
| Fetch only the hidden layer | 5 |
| Feed the hidden layer, fetch logits | 4 (x, W1, b1 are never touched) |

- **CSE:** four identical "layers" collapse from 11 needed nodes to 5 (6 duplicates merged), with the same result.
- **Placement + partitioning:**
  - the cost model makes MatMul 20× faster on a GPU (other ops 2×), and SoftmaxXent and inputs have no GPU kernel;
  - greedy placement put 6 nodes on gpu:0, 3 on gpu:1 and 3 on the CPU;
  - simulated time was **4.6 ms vs 15.7 ms** all on the CPU;
  - partitioning inserted 5 Send/Recv pairs, and the partitioned graph computes **exactly the same cost** (0.045083).
- **16-bit transfers:** the maximum relative error is 7.7e-3, below 2⁻⁷ = 7.8e-3.
- **Switch/Merge:** "if p: v×10 else v−1" gives 30 for (p = 1, v = 3) and 2 for (p = 0, v = 3), with dead values propagating through the untaken branch.

**`experiments.py`:**
- **E1:** greedy placement vs all-CPU / round-robin / random on random DAGs;
- **E2:** shared vs per-edge Recv;
- **E3:** training with 16-bit cross-device tensors;
- **E4:** ASAP vs ALAP Recv memory on a simple chain;
- **E5:** cross-check against real TensorFlow's `GradientTape` and traced graph.
- None were run here.

---

## 8. Check yourself

1. Why build a graph first and run it later instead of executing operations immediately?
<details><summary>Answer</summary>Knowing the whole computation lets the system optimise it (prune, deduplicate, schedule), add gradient nodes, place nodes on devices and distribute it, then run the same optimised graph many times.</details>

2. What is a control dependency, and why are they needed?
<details><summary>Answer</summary>An edge with no data that forces one node to finish before another starts. Mutable state (Variables, queues) means the order of some operations matters even when no data flows between them.</details>

3. How does TensorFlow compute gradients?
<details><summary>Answer</summary>It walks backward from the cost, and for each op on a path to the requested inputs, adds nodes implementing that op's registered gradient function. Partial gradients from several consumers are summed with AddN. The result is more graph.</details>

4. What does partial execution do with `run(fetches, feeds)`?
<details><summary>Answer</summary>It prunes the graph to the nodes needed to compute the fetches, and treats fed tensors as inputs, cutting off their producers, so only that subgraph runs.</details>

5. Describe the greedy placement rule.
<details><summary>Answer</summary>Simulate execution with a cost model. For each node, consider every device that has a kernel for it, estimate when the node would finish there (including the time to receive inputs from other devices), and choose the earliest-finishing device.</details>

6. Why insert Send/Recv nodes instead of letting ops communicate directly?
<details><summary>Answer</summary>It isolates all communication in two node types, so the rest of the runtime and every kernel can ignore networking. The same mechanism works within a machine (GPU↔CPU) and across machines (TCP/RDMA).</details>

7. Why is there one Recv per (tensor, device), not per consumer?
<details><summary>Answer</summary>So a tensor needed by several nodes on the same device is transferred only once, saving bandwidth and memory.</details>

8. How does the 32→16-bit compression work, and how big is the error?
<details><summary>Answer</summary>It keeps the sign, the full 8-bit exponent and the top 7 mantissa bits (zeroing the low 16 bits), then refills with zeros on the receiving side. The relative error is below 2⁻⁷ ≈ 0.8%, and the bytes are halved.</details>

9. How do Switch and Merge implement an if-else in a dataflow graph?
<details><summary>Answer</summary>Switch sends its data to the true or false output based on a predicate, and the other output carries a "dead" value. Ops with a dead input produce dead outputs, so the untaken branch is effectively skipped, and Merge forwards whichever input is alive.</details>

10. In our toy, why did greedy placement keep the cost node on the CPU?
<details><summary>Answer</summary>SoftmaxXent had no GPU kernel in our feasibility rule, so the CPU was the only feasible device. Its inputs (logits) were received from the GPU through a Send/Recv pair.</details>
