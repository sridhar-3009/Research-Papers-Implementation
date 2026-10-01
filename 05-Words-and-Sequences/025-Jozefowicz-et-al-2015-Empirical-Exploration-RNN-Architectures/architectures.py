"""Jozefowicz, Zaremba & Sutskever (2015), "An Empirical Exploration of Recurrent Network Architectures".

Hand-written cells (all inputs are first embedded to the hidden size n, so x and h have the same size):
  TanhRNN          h' = tanh(W_xh x + W_hh h + b)
  LSTM             the paper's exact form (Graves 2013 without peepholes):
                       i = tanh(W_xi x + W_hi h + b_i)      the candidate input
                       j = sigm(W_xj x + W_hj h + b_j)      the input gate
                       f = sigm(W_xf x + W_hf h + b_f)      the forget gate
                       o = tanh(W_xo x + W_ho h + b_o)      the output 'gate' (tanh, as written in the paper)
                       c' = c * f + i * j,      h' = tanh(c') * o
                   options: no_forget (LSTM-f: f = 1), no_input (LSTM-i: j = 1), no_output (LSTM-o: o = 1),
                   forget_bias = 1.0 (LSTM-b: 'adding a bias of 1 to the LSTM's forget gate')
  GRU              r = sigm(.), z = sigm(.), h~ = tanh(W_xh x + W_hh (r * h) + b), h' = z * h + (1 - z) * h~
  MUT1, MUT2, MUT3 the three best architectures found by the search (Section 4)

Architecture search (Section 3):
  Graph            a cell as a computation graph of nodes: inputs (h, c, x), Linear(a, .) (a new matrix with a
                   added to its diagonal), activations, element-wise +, *, -
  lstm_graph, gru_graph   the two starting architectures as graphs
  mutate           the six transformations of Section 3.2
  fitness          Eq. (1): min over tasks of (best accuracy / GRU's best accuracy)
  Search           the top-100 list procedure of Section 3.1 (with a pluggable evaluator)
"""

import copy
import random

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hand-written cells. Every cell: forward(x, state) -> (h_out, new_state); state is a tuple.
# ---------------------------------------------------------------------------

class TanhRNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n, self.lin = n, nn.Linear(2 * n, n)

    def init_state(self, N, device=None):
        return (torch.zeros(N, self.n, device=device),)

    def forward(self, x, state):
        h = torch.tanh(self.lin(torch.cat([x, state[0]], -1)))
        return h, (h,)


class LSTM(nn.Module):
    def __init__(self, n, no_forget=False, no_input=False, no_output=False, forget_bias=0.0):
        super().__init__()
        self.n = n
        self.lin = nn.Linear(2 * n, 4 * n)
        self.no_forget, self.no_input, self.no_output = no_forget, no_input, no_output
        self.forget_bias = forget_bias
        with torch.no_grad():
            self.lin.bias[2 * n:3 * n] += forget_bias          # the forget gate's slice

    def init_state(self, N, device=None):
        z = torch.zeros(N, self.n, device=device)
        return (z, z)

    def forward(self, x, state):
        h, c = state
        a_i, a_j, a_f, a_o = self.lin(torch.cat([x, h], -1)).chunk(4, -1)
        i = torch.tanh(a_i)
        j = torch.ones_like(a_j) if self.no_input else torch.sigmoid(a_j)
        f = torch.ones_like(a_f) if self.no_forget else torch.sigmoid(a_f)
        o = torch.ones_like(a_o) if self.no_output else torch.tanh(a_o)
        c = c * f + i * j
        h = torch.tanh(c) * o
        return h, (h, c)


class GRU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.gates = nn.Linear(2 * n, 2 * n)                  # r and z
        self.W_xh, self.W_hh = nn.Linear(n, n), nn.Linear(n, n, bias=False)

    def init_state(self, N, device=None):
        return (torch.zeros(N, self.n, device=device),)

    def forward(self, x, state):
        h = state[0]
        r, z = torch.sigmoid(self.gates(torch.cat([x, h], -1))).chunk(2, -1)
        h_tilde = torch.tanh(self.W_xh(x) + self.W_hh(r * h))
        h = z * h + (1 - z) * h_tilde
        return h, (h,)


class MUT(nn.Module):
    """The three architectures of Section 4:
    MUT1: z = sigm(W_xz x + b_z);           r = sigm(W_xr x + W_hr h + b_r)
          h' = tanh(W_hh (r * h) + tanh(x) + b_h) * z + h * (1 - z)
    MUT2: z = sigm(W_xz x + W_hz h + b_z);  r = sigm(x + W_hr h + b_r)
          h' = tanh(W_hh (r * h) + W_xh x + b_h) * z + h * (1 - z)
    MUT3: z = sigm(W_xz x + W_hz tanh(h) + b_z);  r = sigm(W_xr x + W_hr h + b_r)
          h' = tanh(W_hh (r * h) + W_xh x + b_h) * z + h * (1 - z)"""

    def __init__(self, n, which=1):
        super().__init__()
        self.n, self.which = n, which
        L = lambda bias=True: nn.Linear(n, n, bias=bias)
        self.W_xz, self.W_hz = L(), L(False)
        self.W_xr, self.W_hr = L(), L(False)
        self.W_hh, self.W_xh = L(), L(False)
        self.b_r = nn.Parameter(torch.zeros(n))                 # MUT2's r uses x directly, plus its own bias

    def init_state(self, N, device=None):
        return (torch.zeros(N, self.n, device=device),)

    def forward(self, x, state):
        h = state[0]
        if self.which == 1:
            z = torch.sigmoid(self.W_xz(x))
            r = torch.sigmoid(self.W_xr(x) + self.W_hr(h))
            cand = torch.tanh(self.W_hh(r * h) + torch.tanh(x))
        elif self.which == 2:
            z = torch.sigmoid(self.W_xz(x) + self.W_hz(h))
            r = torch.sigmoid(x + self.W_hr(h) + self.b_r)
            cand = torch.tanh(self.W_hh(r * h) + self.W_xh(x))
        else:
            z = torch.sigmoid(self.W_xz(x) + self.W_hz(torch.tanh(h)))
            r = torch.sigmoid(self.W_xr(x) + self.W_hr(h))
            cand = torch.tanh(self.W_hh(r * h) + self.W_xh(x))
        h = cand * z + h * (1 - z)
        return h, (h,)


CELLS = {"Tanh": TanhRNN, "LSTM": LSTM, "LSTM-f": lambda n: LSTM(n, no_forget=True),
         "LSTM-i": lambda n: LSTM(n, no_input=True), "LSTM-o": lambda n: LSTM(n, no_output=True),
         "LSTM-b": lambda n: LSTM(n, forget_bias=1.0), "GRU": GRU,
         "MUT1": lambda n: MUT(n, 1), "MUT2": lambda n: MUT(n, 2), "MUT3": lambda n: MUT(n, 3)}


class SequenceModel(nn.Module):
    """Embedding (or a linear map for real-valued inputs) -> one or more cells -> output layer."""

    def __init__(self, cell_factory, n_in, n, n_out, layers=1, discrete=True, init_scale=1.0):
        super().__init__()
        self.inp = nn.Embedding(n_in, n) if discrete else nn.Linear(n_in, n)
        self.cells = nn.ModuleList([cell_factory(n) for _ in range(layers)])
        self.out = nn.Linear(n, n_out)
        x = init_scale / n ** 0.5                              # 'U[-x, x], x = scale / sqrt(units)'
        for p in self.parameters():
            nn.init.uniform_(p, -x, x)
        with torch.no_grad():                                   # re-apply LSTM-b's forget bias after the re-init
            for c in self.cells:
                if isinstance(c, LSTM) and c.forget_bias:
                    c.lin.bias[2 * n:3 * n] += c.forget_bias

    def init_state(self, N, device=None):
        return [c.init_state(N, device) for c in self.cells]

    def forward(self, xs, states):
        outs, states = [], list(states)
        for x in self.inp(xs):
            for k, cell in enumerate(self.cells):
                x, states[k] = cell(x, states[k])
            outs.append(self.out(x))
        return torch.stack(outs), states


# ---------------------------------------------------------------------------
# Section 3.2: architectures as graphs, and their mutations
# ---------------------------------------------------------------------------

ACTIVATIONS = ["tanh", "sigmoid", "relu", "lin0", "lin1", "lin0.9", "lin1.1"]
ELEMENTWISE = ["add", "mul", "sub"]


class Graph:
    """nodes: list of dicts {op, inputs}. op in: 'h', 'c' (state inputs), 'x' (the input), an activation,
    'lin<a>' (Linear(a, .): a new matrix + bias with a added to the diagonal), or an element-wise op.
    outputs: node ids for the new (h, c ...). The first output is the cell's visible output h."""

    def __init__(self, nodes, outputs):
        self.nodes, self.outputs = nodes, outputs

    def state_inputs(self):
        return [i for i, nd in enumerate(self.nodes) if nd["op"] in ("h", "c")]

    def ancestors(self, k):
        seen, stack = set(), [k]
        while stack:
            for p in self.nodes[stack.pop()]["inputs"]:
                if p not in seen:
                    seen.add(p); stack.append(p)
        return seen

    def consumers(self, k):
        return [i for i, nd in enumerate(self.nodes) if k in nd["inputs"]] + [-1 for o in self.outputs if o == k]

    def prune(self):
        """Drop nodes that no output depends on (keep the inputs), renumber."""
        keep = set(self.outputs)
        for o in self.outputs:
            keep |= self.ancestors(o)
        keep |= {i for i, nd in enumerate(self.nodes) if nd["op"] in ("h", "c", "x")}
        order = sorted(keep)
        remap = {old: new for new, old in enumerate(order)}
        self.nodes = [{"op": self.nodes[i]["op"], "inputs": [remap[p] for p in self.nodes[i]["inputs"]]} for i in order]
        self.outputs = [remap[o] for o in self.outputs]
        return self

    def is_valid(self):
        for i, nd in enumerate(self.nodes):
            if any(p >= i for p in nd["inputs"]):              # nodes must only read EARLIER nodes (no cycles)
                return False
            n_in = len(nd["inputs"])
            if nd["op"] in ("h", "c", "x") and n_in != 0:
                return False
            if nd["op"] in ACTIVATIONS and n_in != 1:
                return False
            if nd["op"] in ELEMENTWISE and n_in != 2:
                return False
        return len(self.outputs) == len(self.state_inputs())   # outputs must match the state inputs


def _x(g, op, *inputs):
    g.nodes.append({"op": op, "inputs": list(inputs)})
    return len(g.nodes) - 1


def lstm_graph():
    """The paper's LSTM as a graph (forget gate and all)."""
    g = Graph([{"op": "h", "inputs": []}, {"op": "c", "inputs": []}, {"op": "x", "inputs": []}], [])
    h, c, x = 0, 1, 2
    pre = lambda: _x(g, "add", _x(g, "lin0", x), _x(g, "lin0", h))     # W_x x + W_h h (+ biases)
    i = _x(g, "tanh", pre()); j = _x(g, "sigmoid", pre()); f = _x(g, "sigmoid", pre()); o = _x(g, "tanh", pre())
    c2 = _x(g, "add", _x(g, "mul", c, f), _x(g, "mul", i, j))
    h2 = _x(g, "mul", _x(g, "tanh", c2), o)
    g.outputs = [h2, c2]
    return g


def gru_graph():
    """The GRU as a graph: h' = h~ + z * (h - h~), which equals z * h + (1 - z) * h~."""
    g = Graph([{"op": "h", "inputs": []}, {"op": "x", "inputs": []}], [])
    h, x = 0, 1
    r = _x(g, "sigmoid", _x(g, "add", _x(g, "lin0", x), _x(g, "lin0", h)))
    z = _x(g, "sigmoid", _x(g, "add", _x(g, "lin0", x), _x(g, "lin0", h)))
    ht = _x(g, "tanh", _x(g, "add", _x(g, "lin0", x), _x(g, "lin0", _x(g, "mul", r, h))))
    h2 = _x(g, "add", ht, _x(g, "mul", z, _x(g, "sub", h, ht)))
    g.outputs = [h2]
    return g


class GraphCell(nn.Module):
    """Runs a Graph as an RNN cell. Each 'lin<a>' node owns an n x n matrix + bias (a added to the diagonal)."""

    def __init__(self, graph, n, scale=0.1):
        super().__init__()
        self.g, self.n = graph, n
        self.lins = nn.ModuleDict()
        for i, nd in enumerate(graph.nodes):
            if nd["op"].startswith("lin"):
                lin = nn.Linear(n, n)
                with torch.no_grad():
                    lin.weight.uniform_(-scale, scale); lin.bias.zero_()
                    lin.weight += float(nd["op"][3:]) * torch.eye(n)
                self.lins[str(i)] = lin

    def init_state(self, N, device=None):
        return tuple(torch.zeros(N, self.n, device=device) for _ in self.g.state_inputs())

    def forward(self, x, state):
        vals, s = {}, iter(state)
        for i, nd in enumerate(self.g.nodes):
            op, a = nd["op"], [vals[p] for p in nd["inputs"]]
            if op in ("h", "c"):
                vals[i] = next(s)
            elif op == "x":
                vals[i] = x
            elif op.startswith("lin"):
                vals[i] = self.lins[str(i)](a[0])
            elif op == "tanh":
                vals[i] = torch.tanh(a[0])
            elif op == "sigmoid":
                vals[i] = torch.sigmoid(a[0])
            elif op == "relu":
                vals[i] = torch.relu(a[0])
            elif op == "add":
                vals[i] = a[0] + a[1]
            elif op == "mul":
                vals[i] = a[0] * a[1]
            elif op == "sub":
                vals[i] = a[0] - a[1]
        out = tuple(vals[o] for o in self.g.outputs)
        return out[0], out


def mutate(graph, rng=random, max_tries=100):
    """Section 3.2: apply 1-3 random transformations (each anchored at a random 'current node'):
      1. activation -> another activation       2. element-wise op -> another element-wise op
      3. insert an activation between the node and one of its parents
      4. remove a node that has one input and one consumer
      5. replace the node by one of its ancestors
      6. replace the node by (+, *, -) of an ancestor of it and an ancestor of a random node A
    Graphs that come out invalid are rejected and we try again."""
    for _ in range(max_tries):
        g = copy.deepcopy(graph)
        for _ in range(rng.randint(1, 3)):
            k = rng.randrange(len(g.nodes))
            nd = g.nodes[k]
            t = rng.randint(1, 6)
            if t == 1 and nd["op"] in ACTIVATIONS:
                nd["op"] = rng.choice([a for a in ACTIVATIONS if a != nd["op"]])
            elif t == 2 and nd["op"] in ELEMENTWISE:
                nd["op"] = rng.choice([e for e in ELEMENTWISE if e != nd["op"]])
            elif t == 3 and nd["inputs"]:
                j = rng.randrange(len(nd["inputs"]))
                g.nodes.insert(k, {"op": rng.choice(ACTIVATIONS), "inputs": [nd["inputs"][j]]})
                for m in g.nodes[k + 1:]:                       # renumber nodes after the insertion
                    m["inputs"] = [p + 1 if p >= k else p for p in m["inputs"]]
                g.outputs = [o + 1 if o >= k else o for o in g.outputs]
                g.nodes[k + 1]["inputs"][j] = k
            elif t == 4 and len(nd["inputs"]) == 1 and len(g.consumers(k)) == 1:
                parent = nd["inputs"][0]
                for m in g.nodes:
                    m["inputs"] = [parent if p == k else p for p in m["inputs"]]
                g.outputs = [parent if o == k else o for o in g.outputs]
            elif t == 5 and g.ancestors(k):
                anc = rng.choice(sorted(g.ancestors(k)))
                for m in g.nodes[k + 1:]:
                    m["inputs"] = [anc if p == k else p for p in m["inputs"]]
                g.outputs = [anc if o == k else o for o in g.outputs]
            elif t == 6 and nd["op"] not in ("h", "c", "x"):
                A = rng.randrange(len(g.nodes))
                cand_a = [p for p in g.ancestors(k) | {k} if p < k]
                cand_b = [p for p in g.ancestors(A) | {A} if p < k]
                if cand_a and cand_b:
                    g.nodes[k] = {"op": rng.choice(ELEMENTWISE), "inputs": [rng.choice(cand_a), rng.choice(cand_b)]}
        g.prune()
        if g.is_valid():
            return g
    return copy.deepcopy(graph)


# ---------------------------------------------------------------------------
# Section 3.1: the search procedure
# ---------------------------------------------------------------------------

def fitness(best_acc, gru_acc):
    """Eq. (1): min over tasks of (architecture's best accuracy / GRU's best accuracy). The minimum
    makes the search look for an architecture that is good on EVERY task."""
    return min(best_acc[t] / gru_acc[t] for t in gru_acc)


class Search:
    """Keeps the 100 best architectures. evaluate(graph, task, n_settings) -> best accuracy over that many
    random hyperparameter settings. passes_memorization(graph) -> bool (the first filter, >= 95%)."""

    def __init__(self, evaluate, passes_memorization, tasks, gru_acc, rng=random, top=100):
        self.evaluate, self.passes_memo, self.tasks, self.gru_acc = evaluate, passes_memorization, tasks, gru_acc
        self.rng, self.top = rng, top
        self.pool = [{"graph": lstm_graph(), "best": dict(gru_acc), "name": "LSTM"},
                     {"graph": gru_graph(), "best": dict(gru_acc), "name": "GRU"}]
        self.evaluated = 0

    def score(self, entry):
        return fitness(entry["best"], self.gru_acc)

    def step(self):
        if self.rng.random() < 0.5:                             # re-evaluate an existing architecture
            e = self.rng.choice(self.pool)
            for t in self.tasks:
                e["best"][t] = max(e["best"][t], self.evaluate(e["graph"], t, 20))
            return "re-evaluated"
        child = mutate(self.rng.choice(self.pool)["graph"], self.rng)   # or propose a mutation
        self.evaluated += 1
        if not self.passes_memo(child):
            return "failed memorization"
        best = {}
        for t in self.tasks:                                    # one task at a time; stop below 90% of the GRU
            best[t] = self.evaluate(child, t, 20)
            if best[t] < 0.9 * self.gru_acc[t]:
                return f"failed {t}"
        self.pool.append({"graph": child, "best": best, "name": f"arch{self.evaluated}"})
        self.pool.sort(key=self.score, reverse=True)
        self.pool = self.pool[:self.top]
        return "added"
