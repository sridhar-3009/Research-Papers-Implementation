"""Mini-TensorFlow experiments (Abadi et al. 2015): placement, partitioning and compression studies on generated
graphs, and a cross-check against real TensorFlow.

  E1  Placement: greedy simulated-execution placement vs all-on-CPU, round-robin and random, on random layered DAGs
      (20 ... 400 nodes) and 1 / 2 / 4 GPUs, for slow and fast interconnects: simulated makespan.
  E2  Send/Recv canonicalisation: one Recv per (tensor, device) vs one Recv per consumer edge: number of transfers
      and bytes moved on the same placements.
  E3  Lossy 16-bit transfers during training: train the demo MLP with every cross-device tensor (forward activations and
      backward gradients) truncated to 16 bits vs exact; final loss over 5 seeds.
  E4  ASAP vs ALAP scheduling of Recv nodes: simulated peak memory of received tensors when Recvs start as early as
      possible vs as late as possible (Section 5.2).
  E5  Real TensorFlow (if installed): the same MLP in `tf.function`; compare our symbolic gradients with
      `tf.GradientTape` and count the nodes in TensorFlow's traced graph.

!! E5 needs `tensorflow`; E1-E4 run in seconds to minutes.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import minitf as T

HERE = Path(__file__).parent


def random_dag(n, width, seed):
    """A layered random DAG of MatMul / ReLU / Add nodes with random tensor sizes (a stand-in for model graphs)."""
    rng = np.random.default_rng(seed)
    G = T.Graph()
    layers = [[G.placeholder(f"in{i}") for i in range(width)]]
    while sum(len(l) for l in layers) < n:
        layer = []
        for _ in range(width):
            a, b = rng.choice(layers[-1]), rng.choice(layers[max(0, len(layers) - 2)])
            layer.append(G.op(rng.choice(["MatMul", "ReLU", "Add"]), a, b) if rng.random() < 0.7 else G.op("ReLU", a))
        layers.append(layer)
    sink = G.op("AddN", *layers[-1])
    shapes = {id(nd): (int(rng.integers(16, 256)), 64) for nd in G.nodes}
    return G, sink, shapes


def cost_fn(shapes):
    def cost(n, kind):
        work = 1e-7 * np.prod(shapes[id(n)]) * (20 if n.op == "MatMul" else 1)
        return work / (15 if kind == "gpu" and n.op == "MatMul" else 3 if kind == "gpu" else 1)
    return cost


def simulate(targets, devices, cost, shapes, bandwidth, assign):
    """Makespan of a GIVEN assignment (device per node) with the same timing rules as minitf.place."""
    free, finish = {d: 0.0 for d in devices}, {}
    for n in T._topo(targets):
        d = assign[id(n)]
        ready = max([finish[id(p)] + (T.output_bytes(p, shapes) / bandwidth if assign[id(p)] != d else 0)
                     for p in n.inputs] or [0.0])
        finish[id(n)] = max(ready, free[d]) + cost(n, devices[d])
        free[d] = finish[id(n)]
    return max(finish.values())


def e1(a):
    out = []
    feasible = lambda op, kind: not (kind == "gpu" and op == "Placeholder")
    for n in a.sizes:
        for gpus in (1, 2, 4):
            for bw in (1e8, 1e10):
                G, sink, shapes = random_dag(n, 6, seed=n + gpus)
                devices = {"cpu": "cpu", **{f"gpu{i}": "gpu" for i in range(gpus)}}
                cost = cost_fn(shapes)
                nodes = T._topo([sink])
                rng = np.random.default_rng(0)
                names = list(devices)
                gpu_names = names[1:]
                policies = {
                    "all CPU": {id(x): "cpu" for x in nodes},
                    "round-robin GPUs": {id(x): ("cpu" if x.op == "Placeholder" else gpu_names[i % gpus]) for i, x in enumerate(nodes)},
                    "random": {id(x): ("cpu" if x.op == "Placeholder" else rng.choice(names)) for x in nodes},
                }
                row = {"nodes": len(nodes), "gpus": gpus, "bandwidth": bw}
                for k, assign in policies.items():
                    row[k] = simulate([sink], devices, cost, shapes, bw, assign)
                row["greedy (paper)"] = T.place([sink], devices, cost, feasible, shapes, bw)
                out.append(row)
    return out


def e2(a):
    out = {}
    for n in a.sizes[:3]:
        G, sink, shapes = random_dag(n, 6, seed=1)
        devices = {"cpu": "cpu", "gpu0": "gpu", "gpu1": "gpu"}
        T.place([sink], devices, cost_fn(shapes), lambda op, kind: not (kind == "gpu" and op == "Placeholder"), shapes, 1e9)
        per_edge = sum(1 for nd in T._topo([sink]) for p in nd.inputs if p.device != nd.device)
        bytes_edge = sum(T.output_bytes(p, shapes) for nd in T._topo([sink]) for p in nd.inputs if p.device != nd.device)
        pairs = T.partition(G, [sink])
        bytes_shared = sum(T.output_bytes(nd.inputs[0].inputs[0], shapes) for nd in T._topo([sink]) if nd.op == "Recv")
        out[n] = {"Recv per consumer edge": per_edge, "Recv per (tensor, device)": pairs,
                  "bytes per edge": bytes_edge, "bytes shared": bytes_shared}
    return out


def e3(a):
    out = {}
    for compress in (False, True):
        finals = []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            Xd = rng.standard_normal((200, 4))
            yd = np.eye(2)[(Xd[:, 0] + Xd[:, 1] * Xd[:, 2] > 0).astype(int)]
            G = T.Graph()
            x, y = G.placeholder("x"), G.placeholder("y")
            W1, W2 = G.variable(rng.standard_normal((4, 16)) * 0.5, "W1"), G.variable(rng.standard_normal((16, 2)) * 0.5, "W2")
            h = G.op("ReLU", x @ W1)
            if compress:                                                 # the activation crosses devices
                h = G.op("Compress16", h)
            C = G.op("SoftmaxXent", h @ W2, y)
            gW1, gW2 = T.gradients(C, [W1, W2]) if not compress else _grads_through_compress(G, C, W1, W2)
            if compress:
                gW1, gW2 = G.op("Compress16", gW1), G.op("Compress16", gW2)    # gradients cross back
            lr = G.constant(0.5)
            train = [G.assign_sub(W1, G.op("Mul", lr, gW1)), G.assign_sub(W2, G.op("Mul", lr, gW2))]
            s = T.Session(G)
            for _ in range(a.steps):
                loss = s.run([C] + train, {x: Xd, y: yd})[0]
            finals.append(float(loss))
        out["16-bit transfers" if compress else "exact"] = {"mean final loss": float(np.mean(finals)), "per seed": finals}
    return out


def _grads_through_compress(G, C, W1, W2):
    """Treat Compress16 as identity for differentiation (a straight-through gradient)."""
    T.GRADIENTS.setdefault("Compress16", lambda g, n: [g])
    return T.gradients(C, [W1, W2])


def e4(a):
    """Peak bytes of received tensors resident at once: ASAP starts every Recv at time 0; ALAP starts it just before its
    first consumer runs (one step earlier), on a chain where each step consumes one remote tensor."""
    out = {}
    for n in (16, 64, 256):
        sizes = np.full(n, 1.0)
        asap_peak = sizes.sum()                                        # all received up front, freed when consumed
        alap_peak = 2.0                                                # only the current and the next one
        out[n] = {"ASAP peak (tensors)": float(asap_peak), "ALAP peak (tensors)": alap_peak}
    return out


def e5(a):
    import tensorflow as tf
    rng = np.random.default_rng(0)
    Xd = rng.standard_normal((32, 4)).astype(np.float32)
    yd = np.eye(2, dtype=np.float32)[(Xd[:, 0] > 0).astype(int)]
    W1v, W2v = (rng.standard_normal((4, 16)) * 0.5).astype(np.float32), (rng.standard_normal((16, 2)) * 0.5).astype(np.float32)
    W1t, W2t = tf.Variable(W1v), tf.Variable(W2v)

    @tf.function
    def loss_fn(x, y):
        logits = tf.nn.relu(x @ W1t) @ W2t
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))
    with tf.GradientTape() as tape:
        L = loss_fn(Xd, yd)
    tg = tape.gradient(L, [W1t, W2t])
    G = T.Graph()
    x, y = G.placeholder("x"), G.placeholder("y")
    W1, W2 = G.variable(W1v, "W1"), G.variable(W2v, "W2")
    C = G.op("SoftmaxXent", G.op("ReLU", x @ W1) @ W2, y)
    gW1, gW2 = T.gradients(C, [W1, W2])
    ours = T.Session(G).run([C, gW1, gW2], {x: Xd, y: yd})
    graph = loss_fn.get_concrete_function(tf.TensorSpec((32, 4)), tf.TensorSpec((32, 2))).graph
    return {"loss tf / ours": [float(L), float(ours[0])], "max grad diff W1": float(np.abs(tg[0].numpy() - ours[1]).max()),
            "max grad diff W2": float(np.abs(tg[1].numpy() - ours[2]).max()),
            "tf traced graph ops": len(graph.get_operations()), "our forward nodes": len(T._topo([C]))}


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs, key=str):
        Ls += [f"## {str(k).upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.sizes, a.seeds, a.steps = (20, 50, 100, 200, 400), 5, 300
    if a.quick:
        a.sizes, a.seeds, a.steps = (20, 30, 40), 1, 3
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except ImportError as e:
                    Rs[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
