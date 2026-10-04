"""A miniature TensorFlow (Abadi et al. 2015 whitepaper): computation as a DATAFLOW GRAPH that is built first and run
later, with the mechanisms of Sections 3-5 of the paper.

  graph            nodes = operations (MatMul, Add, ReLU, Variable, Placeholder, ...), edges = tensors; control
                   dependencies are edges that carry no data, only "run this first"
  Session.run      run(fetches, feeds): PARTIAL EXECUTION -- only the nodes the fetches need are run, and fed
                   tensors replace (cut off) their producers (Section 4.2); nodes are executed in dependency order from
                   a ready queue (Section 3.1)
  gradients        tf.gradients(C, xs) EXTENDS the graph: it walks backward from C and adds, for every op on the
                   path, the op's registered gradient function as new nodes, summing partial gradients (Section 4.1)
  CSE              common subexpression elimination merges nodes with the same op, inputs and attributes (Section 5.1)
  placement        a SIMULATED execution walks the graph and greedily puts each node on the feasible device where it
                   would FINISH SOONEST, using a cost model of op time per device plus data-transfer time (Section 3.2.1)
  Send / Recv      every cross-device edge x -> y becomes x -> Send ... Recv -> y; one Recv per (tensor, device) shared
                   by all its consumers there (Section 3.2.2); transfers can be lossily compressed 32 -> 16 bits by
                   dropping the low mantissa bits (Section 5.5)
  control flow     Switch routes a value to one of two outputs (the other is 'dead'); Merge forwards whichever input is
                   alive -- enough for a dataflow if/else (Section 4.4)
"""

import itertools
from collections import defaultdict, deque

import numpy as np

DEAD = object()                                                      # the 'dead' value of an untaken Switch branch

# ----------------------------------------------------------------------------------------------- graph

class Node:
    _ids = itertools.count()

    def __init__(self, graph, op, inputs=(), attrs=None, name=None, control=()):
        self.graph, self.op, self.inputs = graph, op, list(inputs)
        self.attrs = attrs or {}
        self.control = list(control)                                  # control-dependency predecessors
        self.name = name or f"{op}_{next(Node._ids)}"
        self.device = None
        graph.nodes.append(self)

    def __repr__(self):
        return self.name

    # operator sugar
    def __add__(self, o):
        return self.graph.op("Add", self, o)

    def __sub__(self, o):
        return self.graph.op("Sub", self, o)

    def __mul__(self, o):
        return self.graph.op("Mul", self, o)

    def __matmul__(self, o):
        return self.graph.op("MatMul", self, o)


class Graph:
    def __init__(self):
        self.nodes, self.variables = [], {}

    def op(self, op, *inputs, **attrs):
        name = attrs.pop("name", None)
        control = attrs.pop("control", ())
        return Node(self, op, inputs, attrs, name, control)

    def placeholder(self, name):
        return self.op("Placeholder", name=name)

    def constant(self, value, name=None):
        return self.op("Const", value=np.asarray(value, dtype=float), name=name)

    def variable(self, value, name):
        self.variables[name] = np.asarray(value, dtype=float)
        return self.op("Variable", var=name, name=name)

    def assign_sub(self, var_node, delta, name=None):
        return self.op("AssignSub", var_node, delta, var=var_node.attrs["var"], name=name)

# ----------------------------------------------------------------------------------------------- kernels and gradients

def _softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


KERNELS = {
    "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b, "Mul": lambda a, b: a * b,
    "MatMul": lambda a, b: a @ b, "ReLU": lambda a: np.maximum(a, 0), "Square": lambda a: a * a,
    "Mean": lambda a: np.asarray(a.mean()), "Sum": lambda a: np.asarray(a.sum()),
    "AddN": lambda *xs: sum(xs), "Transpose": lambda a: a.T, "Identity": lambda a: a,
    "ReLUGrad": lambda g, a: g * (a > 0), "Fill": lambda like, v: np.full(np.shape(like), float(v)),
    "SumToShape": lambda g, like: _sum_to_shape(g, np.shape(like)),
    "SoftmaxXent": lambda logits, onehot: np.asarray(-(onehot * np.log(_softmax(logits) + 1e-12)).sum(-1).mean()),
    "SoftmaxXentGrad": lambda g, logits, onehot: g * (_softmax(logits) - onehot) / logits.shape[0],
    "Greater": lambda a, b: a > b,
}


def _sum_to_shape(g, shape):
    """Undo broadcasting: sum g over the axes that were broadcast to produce it."""
    g = np.asarray(g)
    while g.ndim > len(shape):
        g = g.sum(0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(ax, keepdims=True)
    return g


def _grad_add(g, node, sign=1.0):
    a, b = node.inputs
    G = node.graph
    gb = G.op("SumToShape", g, b) if sign > 0 else G.op("Mul", G.op("SumToShape", g, b), G.constant(-1.0))
    return [G.op("SumToShape", g, a), gb]


GRADIENTS = {     # op -> function(grad_node, forward_node) -> list of gradient nodes for its inputs (None = none)
    "Add": lambda g, n: _grad_add(g, n),
    "Sub": lambda g, n: _grad_add(g, n, -1.0),
    "Mul": lambda g, n: [n.graph.op("SumToShape", n.graph.op("Mul", g, n.inputs[1]), n.inputs[0]),
                         n.graph.op("SumToShape", n.graph.op("Mul", g, n.inputs[0]), n.inputs[1])],
    "MatMul": lambda g, n: [n.graph.op("MatMul", g, n.graph.op("Transpose", n.inputs[1])),
                            n.graph.op("MatMul", n.graph.op("Transpose", n.inputs[0]), g)],
    "ReLU": lambda g, n: [n.graph.op("ReLUGrad", g, n.inputs[0])],
    "Square": lambda g, n: [n.graph.op("Mul", n.graph.op("Mul", g, n.inputs[0]), n.graph.constant(2.0))],
    "Sum": lambda g, n: [n.graph.op("Mul", n.graph.op("Fill", n.inputs[0], v=1.0), g)],
    "SoftmaxXent": lambda g, n: [n.graph.op("SoftmaxXentGrad", g, *n.inputs), None],
    "Identity": lambda g, n: [g],
}


def gradients(C, xs):
    """Add gradient nodes for dC/dx for each x in xs to C's graph (reverse traversal, summing partials with AddN)."""
    G = C.graph
    order = _topo([C])
    reach = {id(x) for x in xs}                                      # nodes that depend on some x
    for n in order:
        if id(n) in reach or any(id(i) in reach for i in n.inputs):
            reach.add(id(n))
    partials = defaultdict(list)
    partials[id(C)].append(G.op("Fill", C, v=1.0, name=f"grad_{C.name}"))
    grads = {}
    for n in reversed(order):
        if id(n) not in reach or not partials[id(n)]:
            continue
        g = partials[id(n)][0] if len(partials[id(n)]) == 1 else G.op("AddN", *partials[id(n)])
        grads[id(n)] = g
        if n.op in ("Variable", "Placeholder", "Const"):
            continue
        if n.op == "Mean":                                           # d mean / d a = g / size, for every element
            gin = [G.op("Mul", G.op("Fill", n.inputs[0], v=1.0), G.op("Mul", g, G.op("InvSize", n.inputs[0])))]
        else:
            gin = GRADIENTS[n.op](g, n)
        for inp, gi in zip(n.inputs, gin):
            if gi is not None and id(inp) in reach:
                partials[id(inp)].append(gi)
    return [grads.get(id(x)) for x in xs]


KERNELS["InvSize"] = lambda a: np.asarray(1.0 / np.size(a))

# ----------------------------------------------------------------------------------------------- execution

def _topo(targets, stop=()):
    """All nodes needed for `targets` (data + control inputs), stopping at nodes in `stop`, in dependency order."""
    seen, order, stop_ids = set(), [], {id(s) for s in stop}

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        if id(n) not in stop_ids:
            for p in n.inputs + n.control:
                visit(p)
        order.append(n)
    for t in targets:
        visit(t)
    return order


class Session:
    """run(fetches, feeds): prune to the needed subgraph, then execute nodes as they become ready."""

    def __init__(self, graph):
        self.graph = graph
        self.executed = 0

    def run(self, fetches, feeds=None):
        feeds = feeds or {}
        single = not isinstance(fetches, (list, tuple))
        fetches = [fetches] if single else list(fetches)
        needed = _topo(fetches, stop=list(feeds))                    # partial execution
        pending = {id(n): sum(1 for p in n.inputs + n.control if id(p) not in {id(f) for f in feeds}) for n in needed}
        consumers = defaultdict(list)
        for n in needed:
            for p in n.inputs + n.control:
                consumers[id(p)].append(n)
        values = {id(f): np.asarray(v, dtype=float) for f, v in feeds.items()}
        ready = deque(n for n in needed if pending[id(n)] == 0 and id(n) not in values)
        while ready:
            n = ready.popleft()
            values[id(n)] = self._execute(n, values)
            self.executed += 1
            for c in consumers[id(n)]:
                pending[id(c)] -= 1
                if pending[id(c)] == 0 and id(c) not in values:
                    ready.append(c)
        out = [values[id(f)] for f in fetches]
        return out[0] if single else out

    def _execute(self, n, values):
        G = self.graph
        args = [values[id(i)] for i in n.inputs]
        if n.op == "Placeholder":
            raise KeyError(f"placeholder {n.name} must be fed")
        if n.op == "Const":
            return n.attrs["value"]
        if n.op == "Variable":
            return G.variables[n.attrs["var"]]
        if n.op == "AssignSub":
            G.variables[n.attrs["var"]] = G.variables[n.attrs["var"]] - args[1]
            return G.variables[n.attrs["var"]]
        if n.op == "Switch":                                          # (data, pred) -> output port chosen by attrs
            data, pred = args
            if data is DEAD:
                return DEAD
            return data if bool(pred) == n.attrs["branch"] else DEAD
        if n.op == "Merge":
            alive = [a for a in args if a is not DEAD]
            return alive[0] if alive else DEAD
        if any(a is DEAD for a in args):                              # dead values propagate
            return DEAD
        if n.op == "Fill":
            return KERNELS["Fill"](args[0], n.attrs["v"])
        if n.op in ("Send", "Recv"):
            return args[0]
        if n.op == "Compress16":
            return truncate_to_16(args[0])
        return KERNELS[n.op](*args)

# ----------------------------------------------------------------------------------------------- CSE

def cse(graph, targets):
    """Common subexpression elimination: nodes with equal (op, inputs, attrs) are merged. Returns the new targets
    and how many nodes were removed. Stateful nodes (Variable, AssignSub, Placeholder) are never merged."""
    canon, removed = {}, 0
    rename = {}

    def key(n):
        att = tuple(sorted((k, v.tobytes() if isinstance(v, np.ndarray) else v) for k, v in n.attrs.items()))
        return (n.op, tuple(id(rename.get(id(i), i)) for i in n.inputs), att)
    for n in _topo(targets):
        n.inputs = [rename.get(id(i), i) for i in n.inputs]
        if n.op in ("Variable", "AssignSub", "Placeholder"):
            continue
        k = key(n)
        if k in canon:
            rename[id(n)] = canon[k]
            removed += 1
        else:
            canon[k] = n
    live = [rename.get(id(t), t) for t in targets]
    graph.nodes = [n for n in graph.nodes if id(n) not in rename]
    return live, removed

# ----------------------------------------------------------------------------------------------- placement

def output_bytes(n, shapes):
    return 4 * int(np.prod(shapes.get(id(n), (1,))))


def place(targets, devices, cost, feasible, shapes, bandwidth=1e9):
    """Greedy simulated execution (Section 3.2.1). cost(op, device_kind, n) -> seconds; feasible(op, kind) -> bool;
    shapes: id(node) -> output shape (from a profiling run). Each node goes to the feasible device where it would
    finish earliest, counting the time to receive its inputs from other devices. Returns the simulated makespan."""
    free = {d: 0.0 for d in devices}
    finish = {}
    for n in _topo(targets):
        best = None
        for d, kind in devices.items():
            if not feasible(n.op, kind):
                continue
            ready = 0.0
            for p in n.inputs + n.control:
                t = finish[id(p)]
                if p.device != d:
                    t += output_bytes(p, shapes) / bandwidth
                ready = max(ready, t)
            done = max(ready, free[d]) + cost(n, kind)
            if best is None or done < best[0]:
                best = (done, d)
        n.device = best[1]
        finish[id(n)] = best[0]
        free[best[1]] = best[0]
    return max(finish.values())


def partition(graph, targets):
    """Insert Send/Recv for every edge whose endpoints are on different devices; ONE Recv per (tensor, consumer
    device), shared by all consumers there. Returns (#send/recv pairs, bytes-free count of cross edges)."""
    recv_for = {}
    pairs = 0
    for n in _topo(targets):
        new_inputs = []
        for p in n.inputs:
            if p.device is not None and n.device is not None and p.device != n.device:
                key = (id(p), n.device)
                if key not in recv_for:
                    send = graph.op("Send", p, name=f"send_{p.name}_to_{n.device}")
                    send.device = p.device
                    recv = graph.op("Recv", send, name=f"recv_{p.name}_on_{n.device}")
                    recv.device = n.device
                    recv_for[key] = recv
                    pairs += 1
                new_inputs.append(recv_for[key])
            else:
                new_inputs.append(p)
        n.inputs = new_inputs
    return pairs


def truncate_to_16(x):
    """Section 5.5: keep the sign, exponent and top 7 mantissa bits of a float32 (zero the low 16 bits)."""
    a = np.asarray(x, dtype=np.float32).copy()
    a.view(np.uint32)[...] &= np.uint32(0xFFFF0000)
    return a.astype(float)


REPORTED = {
    "Inception port": "13.6M parameters, 36,000 operations in the TensorFlow graph, 2 billion multiply-adds per "
                      "224x224 image; some LSTM language models have over 15,000 nodes",
    "placement": "simulated execution with a cost model; greedy: put each node on the feasible device where it finishes "
                 "soonest, including the cost of receiving its inputs; user hints and constraints narrow the choice",
    "Send/Recv": "all cross-device communication is isolated in Send/Recv node pairs; one Receive per tensor per device; "
                 "TCP or RDMA across machines",
    "optimizations": "common subexpression elimination (Click's algorithm), ASAP/ALAP scheduling of Receive nodes, "
                     "asynchronous kernels, optimized libraries, lossy 32->16-bit compression of transfers",
    "control flow": "Switch, Merge, Enter, Leave, NextIteration -- a small set of primitives for if/while on a dataflow "
                    "graph (after Arvind's dataflow machines)",
    "DistBelief": "TensorFlow is the successor of DistBelief; dozens of internal clients had already switched",
}
