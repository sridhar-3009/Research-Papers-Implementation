"""The benchmark tasks of Hochreiter & Schmidhuber (1997), Section 5.

  embedded_reber        Experiment 1: the embedded Reber grammar (next-symbol prediction)
  noise_free_2a         Experiment 2a: (y, a1..a_{p-1}, y) vs (x, a1..a_{p-1}, x): remember the first symbol
  adding_problem        Experiment 4: output 0.5 + (X1 + X2) / 4 for the two marked values
  multiplication        Experiment 5: output X1 * X2 for the two marked values
  temporal_order        Experiment 6a: classify a sequence by the ORDER of two far-apart symbols
"""

import numpy as np

# ---------------------------------------------------------------------------
# Experiment 1: embedded Reber grammar
# ---------------------------------------------------------------------------

REBER_SYMBOLS = "BTSXPVE"
# the Reber grammar as a graph: node -> [(symbol, next node)]
REBER_GRAPH = {0: [("T", 1), ("P", 2)], 1: [("S", 1), ("X", 3)], 2: [("T", 2), ("V", 4)],
               3: [("X", 2), ("S", 5)], 4: [("P", 3), ("V", 5)], 5: []}


def reber_string(rng):
    """B, then a random walk through the graph (each choice with probability 0.5), then E."""
    out, node = ["B"], 0
    while REBER_GRAPH[node]:
        sym, node = REBER_GRAPH[node][rng.integers(len(REBER_GRAPH[node]))]
        out.append(sym)
    return out + ["E"]


def reber_next_sets(string):
    """For each position, the set of symbols that may legally come next."""
    sets, node = [], 0
    for i, sym in enumerate(string):
        if i == 0:                                           # after B
            sets.append({"T", "P"})
            continue
        if sym == "E":
            sets.append(set())
            continue
        node = dict(REBER_GRAPH[node])[sym]
        sets.append({s for s, _ in REBER_GRAPH[node]} or {"E"})
    return sets


def embedded_reber(rng):
    """B, (T or P), <a Reber string>, (the SAME T or P), E. Predicting the second-to-last symbol
    requires remembering the second one across the whole inner string (the 'long' dependency).
    Returns (string, inputs one-hot (L-1, 7), targets multi-hot (L-1, 7) of legal next symbols)."""
    branch = "T" if rng.random() < 0.5 else "P"
    inner = reber_string(rng)
    s = ["B", branch] + inner + [branch, "E"]
    nxt = [{"T", "P"}, {"B"}]                                 # after the outer B; after T/P comes the inner B
    nxt += reber_next_sets(inner)[:-1] + [{branch}]          # inside; after the inner E comes the branch symbol
    nxt += [{"E"}]                                           # after the final T/P comes E
    idx = {c: i for i, c in enumerate(REBER_SYMBOLS)}
    X = np.zeros((len(s) - 1, 7))
    Y = np.zeros((len(s) - 1, 7))
    for t in range(len(s) - 1):
        X[t, idx[s[t]]] = 1
        for c in nxt[t]:
            Y[t, idx[c]] = 1
    return s, X, Y


def reber_is_valid(s):
    """Checks a whole embedded Reber string against the grammar."""
    if len(s) < 9 or s[0] != "B" or s[-1] != "E" or s[1] not in "TP" or s[-2] != s[1]:
        return False
    inner = s[2:-2]
    if inner[0] != "B" or inner[-1] != "E":
        return False
    node = 0
    for sym in inner[1:-1]:
        d = dict(REBER_GRAPH[node])
        if sym not in d:
            return False
        node = d[sym]
    return node == 5


# ---------------------------------------------------------------------------
# Experiment 2a
# ---------------------------------------------------------------------------

def noise_free_2a(p, rng):
    """Symbols a_1..a_{p-1}, a_p = x, a_{p+1} = y (one-hot, p+1 dims). The two training sequences are
    (y, a_1, ..., a_{p-1}, y) and (x, a_1, ..., a_{p-1}, x). Predict the next symbol at every step;
    the last prediction needs the first symbol, p steps earlier."""
    first = p if rng.random() < 0.5 else p - 1                 # index of x or y
    seq = [first] + list(range(p - 1)) + [first]
    X = np.eye(p + 1)[seq[:-1]]
    Y = np.eye(p + 1)[seq[1:]]
    return X, Y


# ---------------------------------------------------------------------------
# Experiments 4 and 5
# ---------------------------------------------------------------------------

def _marked_pairs(T, rng, values):
    """Common structure of Experiments 4-5. Length L in [T, T + T/10]. Second components: 1.0 at the
    two marked positions (one among the first 10, one among the first T/2 - 1 still unmarked), -1 at
    the first and the last pair (when not marked), 0 elsewhere. Returns (inputs (L, 2), i1, i2)."""
    L = int(rng.integers(T, T + T // 10 + 1))
    x = np.zeros((L, 2))
    x[:, 0] = values(L)
    i1 = int(rng.integers(0, 10))
    choices = [i for i in range(T // 2 - 1) if i != i1]
    i2 = int(rng.choice(choices))
    x[0, 1], x[-1, 1] = -1.0, -1.0
    x[i1, 1] = x[i2, 1] = 1.0
    return x, i1, i2


def adding_problem(T, rng):
    """Experiment 4: target 0.5 + (X1 + X2) / 4.0, the sum scaled into [0, 1]. Values in [-1, 1].
    'In the rare case where the first pair of the sequence gets marked, we set X1 to zero.'"""
    x, i1, i2 = _marked_pairs(T, rng, lambda L: rng.uniform(-1, 1, L))
    if i1 == 0:
        x[0, 0] = 0.0
    return x, 0.5 + (x[i1, 0] + x[i2, 0]) / 4.0


def multiplication(T, rng):
    """Experiment 5: values in [0, 1], target X1 * X2 ('if the first pair gets marked, X1 = 1.0')."""
    x, i1, i2 = _marked_pairs(T, rng, lambda L: rng.uniform(0, 1, L))
    if i1 == 0:
        x[0, 0] = 1.0
    return x, x[i1, 0] * x[i2, 0]


# ---------------------------------------------------------------------------
# Experiment 6a: temporal order
# ---------------------------------------------------------------------------

ORDER_SYMBOLS = ["E", "B", "a", "b", "c", "d", "X", "Y"]
ORDER_CLASSES = {("X", "X"): 0, ("X", "Y"): 1, ("Y", "X"): 2, ("Y", "Y"): 3}       # Q, R, S, U


def temporal_order(rng):
    """Length 100-110: E, random distractors from {a, b, c, d}, B at the end. X or Y is placed at a
    random t1 in [10, 20] and t2 in [50, 60]. The class (4 classes) is the ORDER: XX, XY, YX, YY."""
    L = int(rng.integers(100, 111))
    seq = ["E"] + list(rng.choice(["a", "b", "c", "d"], L - 2)) + ["B"]
    t1, t2 = int(rng.integers(10, 21)), int(rng.integers(50, 61))
    s1, s2 = rng.choice(["X", "Y"]), rng.choice(["X", "Y"])
    seq[t1], seq[t2] = s1, s2
    X = np.eye(8)[[ORDER_SYMBOLS.index(c) for c in seq]]
    return X, ORDER_CLASSES[(s1, s2)], (t1, t2)
