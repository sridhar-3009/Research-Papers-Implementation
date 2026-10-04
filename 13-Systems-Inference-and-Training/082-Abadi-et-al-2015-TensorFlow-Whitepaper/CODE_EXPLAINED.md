# The code, explained simply

How the code in this folder implements a miniature TensorFlow (Abadi et al. 2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `minitf.py` | `Node` / `Graph` (placeholders, constants, variables, AssignSub, control edges), kernels and gradient functions, `gradients`, topological pruning, `Session.run` with partial execution, CSE, greedy placement, Send/Recv partitioning, 16-bit truncation, Switch/Merge with dead values |
| `experiments.py` | placement policies on random DAGs, Recv canonicalisation, 16-bit training, ASAP vs ALAP, real TensorFlow cross-check |
| `demo.py` | training via graph updates, partial execution, CSE, placement + partitioning, compression, control flow (instant) |
| `test_minitf.py` | 3 quick tests |

**Run it** (from `13-Systems-Inference-and-Training/082-Abadi-et-al-2015-TensorFlow-Whitepaper`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `minitf.py`

### Graph building
- **`Node(graph, op, inputs, attrs, name, control)`:** `+ - * @` build Add/Sub/Mul/MatMul nodes.
- **`Graph`:**
  - `op(...)` creates a node;
  - `placeholder`, `constant`, `variable` (stores the value in `graph.variables`);
  - `assign_sub(var, delta)` is a stateful update node.

### Kernels and gradients
- **`KERNELS`:** numpy implementations per op type, including the gradient helper ops ReLUGrad, SumToShape (which undoes broadcasting), SoftmaxXentGrad and Fill.
- **`GRADIENTS`:** for each op, a function (upstream gradient node, forward node) → gradient nodes for its inputs. Mean is handled in `gradients` with an InvSize node.
- **`gradients(C, xs)`:**
  1. find the nodes reachable from xs;
  2. go through the topological order of C's subgraph in reverse;
  3. sum each node's partial gradients (AddN);
  4. call the op's gradient function and route the results to its inputs.

### Running
- **`_topo(targets, stop)`:** the nodes needed for the targets (data and control inputs), not going past `stop` (the fed tensors).
- **`Session.run(fetches, feeds)`:**
  - prune;
  - count each node's pending inputs;
  - start with the ready nodes, execute, decrement consumers, and enqueue those that become ready.
  - It counts executed nodes in `self.executed`.
- **`_execute`:** special cases for Placeholder, Const, Variable, AssignSub, Switch (returns data or `DEAD`), Merge (the first live input), dead-value propagation, Fill, Send/Recv (identity) and Compress16.

### Optimisations
| Name | What it does |
|---|---|
| `cse(graph, targets)` | in topological order, rewire each node's inputs to canonical nodes, and merge nodes with an equal (op, canonical inputs, attributes) key; stateful nodes are never merged |
| `place(targets, devices, cost, feasible, shapes, bandwidth)` | simulated execution tracking when each device is free and when each node finishes; transfer time = output bytes / bandwidth when an input is on another device; each node takes the feasible device with the earliest finish |
| `partition(graph, targets)` | rewrites cross-device inputs through Send (on the producer's device) and Recv (on the consumer's device), with one Recv per (tensor, device) via `recv_for` |
| `truncate_to_16` | `float32` view as `uint32`, `& 0xFFFF0000` |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | random layered DAGs × GPUs × bandwidth: makespan of all-CPU, round-robin, random and greedy placement (`simulate` evaluates a fixed assignment) |
| `e2` | number of transfers and bytes, per-edge Recv vs shared Recv |
| `e3` | MLP training with activations and gradients passed through Compress16 (straight-through gradient), 5 seeds |
| `e4` | peak resident received tensors for ASAP vs ALAP on a chain |
| `e5` | TensorFlow `tf.function` + `GradientTape` vs our gradients; op count of TensorFlow's traced graph |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_gradients_match_finite_differences` | symbolic gradients through MatMul, Add-with-broadcast, ReLU, Square and Mean equal numerical ones |
| `test_partial_execution_and_cse` | feeding an intermediate runs only 2 nodes; CSE merges a duplicate Square and keeps the value |
| `test_placement_partition_switch_merge_and_compression` | placement respects feasibility and cost; 2 Send/Recv pairs; partitioned results are correct; Switch/Merge if-else; 16-bit error ≤ 2⁻⁷ |

---

## 5. Try it yourself

1. Add a `While` loop using Enter / NextIteration / Exit nodes (you will need frames for iteration counts).
2. Make the placement greedy over groups of colocated nodes (e.g. a Variable and its AssignSub).
3. Insert Compress16 automatically on every Send/Recv in `partition`.
4. Add a dead-code pass that removes nodes no fetch can reach.
5. Measure how many nodes `gradients` adds per forward node for a 10-layer MLP.
