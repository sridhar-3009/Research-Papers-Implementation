"""A miniature TensorFlow in ~2 seconds: build a dataflow graph, train it with symbolically added gradient nodes,
partial execution with feeds and fetches, common subexpression elimination, greedy cost-model placement on simulated
devices, Send/Recv partitioning, lossy 16-bit transfers, and Switch/Merge control flow."""

import time

import numpy as np

from minitf import (REPORTED, Graph, Session, _topo, cse, gradients, partition, place, truncate_to_16)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


rng = np.random.default_rng(0)
Xd = rng.standard_normal((200, 4))
yd = (Xd[:, 0] + Xd[:, 1] * Xd[:, 2] > 0).astype(int)
onehot = np.eye(2)[yd]

section("1. Build a graph first, run it later; gradients are MORE graph")
G = Graph()
x, y = G.placeholder("x"), G.placeholder("y")
W1, b1 = G.variable(rng.standard_normal((4, 16)) * 0.5, "W1"), G.variable(np.zeros(16), "b1")
W2, b2 = G.variable(rng.standard_normal((16, 2)) * 0.5, "W2"), G.variable(np.zeros(2), "b2")
h = G.op("ReLU", x @ W1 + b1, name="hidden")
logits = G.op("Add", h @ W2, b2, name="logits")
C = G.op("SoftmaxXent", logits, y, name="cost")
n_forward = len(G.nodes)
grads = gradients(C, [W1, b1, W2, b2])
print(f"  forward graph: {n_forward} nodes; after tf.gradients(cost, [W1, b1, W2, b2]): {len(G.nodes)} nodes")
print("  (gradients walk backward from the cost and add each op's registered gradient function as new nodes)")
lr = G.constant(0.5)
train = [G.assign_sub(v, G.op("Mul", lr, g)) for v, g in zip((W1, b1, W2, b2), grads)]
sess = Session(G)
losses = []
for step in range(300):
    out = sess.run([C] + train, {x: Xd, y: onehot})
    losses.append(float(out[0]))
acc = (sess.run(logits, {x: Xd}).argmax(1) == yd).mean()
print(f"  SGD by running the update nodes: cost {losses[0]:.3f} -> {losses[-1]:.3f}, training accuracy {acc:.1%}")

section("2. Partial execution: only what the fetches need runs; feeds cut the graph")
for label, fetch, feed in (("fetch the cost", C, {x: Xd, y: onehot}), ("fetch only the hidden layer", h, {x: Xd}),
                           ("feed the hidden layer, fetch logits", logits, {h: np.ones((3, 16))})):
    s = Session(G)
    s.run(fetch, feed)
    print(f"    {label:40s} nodes executed: {s.executed:3d} of {len(G.nodes)}")
print("  -> feeding 'hidden' replaces its whole upstream subgraph (x, W1, b1 are never touched)")

section("3. Common subexpression elimination")
G2 = Graph()
a = G2.placeholder("a")
Wc = G2.variable(rng.standard_normal((3, 3)), "Wc")
outs = [G2.op("ReLU", a @ Wc) for _ in range(4)]                  # four 'layers of abstraction' build the same thing
total = G2.op("AddN", *outs)
before = len(_topo([total]))
val_before = Session(G2).run(total, {a: np.ones((2, 3))})
(total,), removed = cse(G2, [total])
val_after = Session(G2).run(total, {a: np.ones((2, 3))})
print(f"  nodes needed before CSE {before}, after {len(_topo([total]))} ({removed} duplicates merged); same result: "
      f"{np.allclose(val_before, val_after)}")

section("4. Placement by simulated execution + Send/Recv partitioning")
shapes = {}                                                          # a profiling run records output shapes
s = Session(G)
vals = s.run([n for n in _topo([C]) if n.op not in ("Placeholder",)], {x: Xd, y: onehot})
for n, v in zip([n for n in _topo([C]) if n.op not in ("Placeholder",)], vals):
    shapes[id(n)] = np.shape(v)
shapes[id(x)], shapes[id(y)] = Xd.shape, onehot.shape
devices = {"cpu:0": "cpu", "gpu:0": "gpu", "gpu:1": "gpu"}


def cost(n, kind):
    work = 1e-6 * (np.prod(shapes.get(id(n), (1,))) + (2000 if n.op == "MatMul" else 0))
    return work / (20 if kind == "gpu" and n.op == "MatMul" else 2 if kind == "gpu" else 1)


feasible = lambda op, kind: not (kind == "gpu" and op in ("SoftmaxXent", "Placeholder"))   # no GPU kernel
targets = [C]
for n in G.nodes:
    n.device = None
all_cpu = place(targets, {"cpu:0": "cpu"}, cost, feasible, shapes)
for n in G.nodes:
    n.device = None
span = place(targets, devices, cost, feasible, shapes, bandwidth=2e8)
counts = {}
for n in _topo(targets):
    counts[n.device] = counts.get(n.device, 0) + 1
print(f"  cost model: MatMul 20x faster on a GPU, other ops 2x; SoftmaxXent and input feeding have no GPU kernel")
print(f"  greedy placement: {counts}; simulated time {span * 1e3:.3f} ms vs {all_cpu * 1e3:.3f} ms all on the CPU")
print(f"  e.g. {', '.join(f'{n.name}->{n.device}' for n in _topo(targets) if n.name in ('hidden', 'logits', 'cost'))}")
c_before = Session(G).run(C, {x: Xd, y: onehot})
pairs = partition(G, targets)
v_part = Session(G).run(C, {x: Xd, y: onehot})
cross = sum(1 for n in _topo(targets) if n.op == "Recv")
print(f"  partitioning inserted {pairs} Send/Recv pairs (one Recv per tensor per device, {cross} Recv nodes); the")
print(f"  partitioned graph computes exactly the same cost: {c_before:.6f} vs {v_part:.6f}")

section("5. Lossy compression of transfers (32 -> 16 bits by dropping mantissa bits)")
t = rng.standard_normal(10000)
tc = truncate_to_16(t)
print(f"  max relative error {np.max(np.abs(tc - t) / np.abs(t)):.2e} (2^-7 = {2 ** -7:.2e}); halves the bytes sent")

section("6. Control flow with Switch / Merge (a dataflow if-else)")
G3 = Graph()
v, p = G3.placeholder("v"), G3.placeholder("p")
true_branch = G3.op("Mul", G3.op("Switch", v, p, branch=True), G3.constant(10.0))
false_branch = G3.op("Sub", G3.op("Switch", v, p, branch=False), G3.constant(1.0))
out = G3.op("Merge", true_branch, false_branch)
s3 = Session(G3)
print(f"  if p: v*10 else v-1  ->  p=1, v=3: {s3.run(out, {v: 3.0, p: 1.0})};  p=0, v=3: {s3.run(out, {v: 3.0, p: 0.0})}")
print("  (the untaken Switch output is 'dead'; dead values propagate through ops and Merge picks the live one)")

section("7. The paper's numbers")
for k, val in REPORTED.items():
    print(f"  {k}: {val}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
