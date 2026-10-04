import numpy as np

from minitf import Graph, Session, _topo, cse, gradients, partition, place, truncate_to_16


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    G = Graph()
    x = G.placeholder("x")
    W = G.variable(rng.standard_normal((3, 2)), "W")
    b = G.variable(rng.standard_normal(2), "b")
    C = G.op("Mean", G.op("Square", G.op("ReLU", x @ W + b)))
    gW, gb = gradients(C, [W, b])
    s = Session(G)
    X = rng.standard_normal((4, 3))
    aW, ab = s.run([gW, gb], {x: X})
    eps, num = 1e-6, np.zeros((3, 2))
    W0 = G.variables["W"].copy()
    for i in range(3):
        for j in range(2):
            G.variables["W"] = W0.copy(); G.variables["W"][i, j] += eps; cp = s.run(C, {x: X})
            G.variables["W"] = W0.copy(); G.variables["W"][i, j] -= eps; cm = s.run(C, {x: X})
            num[i, j] = (cp - cm) / (2 * eps)
    assert np.allclose(aW, num, atol=1e-6) and ab.shape == (2,)


def test_partial_execution_and_cse():
    G = Graph()
    a, c = G.placeholder("a"), G.constant(2.0)
    mid = G.op("Mul", a, c)
    out = G.op("Add", mid, c)
    s = Session(G)
    assert s.run(out, {mid: 5.0}) == 7.0 and s.executed == 2                      # mid fed: a, Mul never run
    G2 = Graph()
    p = G2.placeholder("p")
    dup = G2.op("AddN", G2.op("Square", p), G2.op("Square", p))
    (dup,), removed = cse(G2, [dup])
    assert removed == 1 and Session(G2).run(dup, {p: 3.0}) == 18.0


def test_placement_partition_switch_merge_and_compression():
    G = Graph()
    a = G.placeholder("a")
    w = G.constant(np.ones((2, 2)))
    m = G.op("MatMul", a, w)
    r = G.op("ReLU", m)
    shapes = {id(n): (2, 2) for n in G.nodes}
    feasible = lambda op, kind: not (kind == "gpu" and op in ("Placeholder", "ReLU"))
    cost = lambda n, kind: 1.0 if kind == "cpu" else 0.1
    place([r], {"cpu": "cpu", "gpu": "gpu"}, cost, feasible, shapes, bandwidth=1e9)
    assert a.device == "cpu" and m.device == "gpu" and r.device == "cpu"
    assert partition(G, [r]) == 2                                                  # a -> gpu, m -> cpu
    assert np.allclose(Session(G).run(r, {a: np.eye(2)}), np.ones((2, 2)))
    G3 = Graph()
    v, pr = G3.placeholder("v"), G3.placeholder("p")
    out = G3.op("Merge", G3.op("Identity", G3.op("Switch", v, pr, branch=True)),
                G3.op("Mul", G3.op("Switch", v, pr, branch=False), G3.constant(-1.0)))
    assert Session(G3).run(out, {v: 4.0, pr: 1.0}) == 4.0 and Session(G3).run(out, {v: 4.0, pr: 0.0}) == -4.0
    t = np.array([1.0, 3.14159, -1e-3])
    assert np.all(np.abs(truncate_to_16(t) - t) <= np.abs(t) * 2 ** -7)
