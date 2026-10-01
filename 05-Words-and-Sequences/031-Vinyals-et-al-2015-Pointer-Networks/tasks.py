"""The three geometric problems of Pointer Networks (Sec. 3), their exact / approximate solvers and their metrics.

Points are uniform in [0, 1]^2. Indices here are 1-based as in the paper (index 0 is reserved for the end token).
"""

import itertools
import math

import numpy as np


# ---------------------------------------------------------------------------------------------------- convex hull
def convex_hull(P):
    """Sec. 3.1 target: hull vertices, starting from the LOWEST index, counter-clockwise, and closing back on the
    start (Figure 2a: 2, 4, 3, 5, 6, 7, 2). Andrew's monotone chain, O(n log n)."""
    pts = sorted(range(len(P)), key=lambda i: (P[i][0], P[i][1]))

    def cross(o, a, b):
        return (P[a][0] - P[o][0]) * (P[b][1] - P[o][1]) - (P[a][1] - P[o][1]) * (P[b][0] - P[o][0])

    lower, upper = [], []
    for i in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    hull = lower[:-1] + upper[:-1]                          # counter-clockwise
    k = hull.index(min(hull))
    hull = hull[k:] + hull[:k]
    return [i + 1 for i in hull] + [hull[0] + 1]


def polygon_area(P, idx):
    """Shoelace area of the polygon through 1-based indices idx (a closing repeat is ignored)."""
    if len(idx) > 1 and idx[0] == idx[-1]:
        idx = idx[:-1]
    if len(idx) < 3:
        return 0.0
    xy = np.array([P[i - 1] for i in idx])
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0


def is_simple_polygon(P, idx):
    """No repeated vertex and no two non-adjacent edges crossing."""
    if len(idx) > 1 and idx[0] == idx[-1]:
        idx = idx[:-1]
    if len(idx) < 3 or len(set(idx)) != len(idx) or min(idx) < 1:
        return False
    k = len(idx)
    E = [(P[idx[i] - 1], P[idx[(i + 1) % k] - 1]) for i in range(k)]
    for a in range(k):
        for b in range(a + 2, k):
            if a == 0 and b == k - 1:
                continue
            if _segments_cross(*E[a], *E[b]):
                return False
    return True


def same_polygon(c1, c2):
    """Sec. 4.2: two outputs are the same if they represent the same polygon (same cyclic order, either start)."""
    a = c1[:-1] if len(c1) > 1 and c1[0] == c1[-1] else c1
    b = c2[:-1] if len(c2) > 1 and c2[0] == c2[-1] else c2
    if len(a) != len(b) or set(a) != set(b):
        return False
    k = b.index(a[0])
    return a == b[k:] + b[:k]


def hull_metrics(Ps, preds):
    """Table 1: accuracy, and area coverage of the true hull (only over simple polygons; 'FAIL' if > 1% are not)."""
    acc = np.mean([same_polygon(p, convex_hull(P)) for P, p in zip(Ps, preds)])
    simple = [(P, p) for P, p in zip(Ps, preds) if is_simple_polygon(P, p)]
    if len(simple) < 0.99 * len(Ps):
        return {"accuracy": 100 * acc, "area": "FAIL"}
    # any simple polygon on the points lies inside the hull, so coverage = its area / the hull's area
    area = np.mean([polygon_area(P, p) / polygon_area(P, convex_hull(P)) for P, p in simple])
    return {"accuracy": 100 * acc, "area": 100 * area}


# ---------------------------------------------------------------------------------------------------- Delaunay
def incenter(P, tri):
    a, b, c = (np.array(P[i - 1]) for i in tri)
    la, lb, lc = np.linalg.norm(b - c), np.linalg.norm(a - c), np.linalg.norm(a - b)
    return (la * a + lb * b + lc * c) / (la + lb + lc)


def delaunay(P):
    """Sec. 3.2 target: triangles as increasing index triples, ordered by their incenters (lexicographically)."""
    from scipy.spatial import Delaunay
    tris = [tuple(sorted(int(i) + 1 for i in s)) for s in Delaunay(np.asarray(P)).simplices]
    return sorted(tris, key=lambda t: tuple(incenter(P, t)))


def delaunay_metrics(Ps, preds):
    """Sec. 4.3: accuracy (the exact set of triangles) and triangle coverage (% of true triangles predicted)."""
    acc, cov = [], []
    for P, pred in zip(Ps, preds):
        true = set(delaunay(P))
        got = {tuple(sorted(t)) for t in pred}
        acc.append(got == true)
        cov.append(len(true & got) / len(true))
    return {"accuracy": 100 * np.mean(acc), "triangle coverage": 100 * np.mean(cov)}


# ---------------------------------------------------------------------------------------------------- TSP
def dist_matrix(P):
    P = np.asarray(P)
    return np.linalg.norm(P[:, None] - P[None], axis=-1)


def tour_length(P, tour):
    """Closed tour length; tour is 1-based and does not repeat the start."""
    D = dist_matrix(P)
    t = [i - 1 for i in tour]
    return float(sum(D[t[k], t[(k + 1) % len(t)]] for k in range(len(t))))


def held_karp(P):
    """Exact TSP by dynamic programming over subsets, O(2^n n^2) (Sec. 3.3, used up to n = 20 in the paper).
    Returns the optimal tour starting at city 1."""
    n = len(P)
    D = dist_matrix(P)
    if n <= 3:
        return list(range(1, n + 1))
    m = n - 1                                               # cities 1..n-1 (0-based); city 0 is the start
    full = 1 << m
    Dm = D[1:, 1:]
    C = np.full((full, m), np.inf)                          # C[S, j]: shortest path 0 -> (all of S) ending at j
    parent = np.zeros((full, m), dtype=np.int64)
    for j in range(m):
        C[1 << j, j] = D[0, j + 1]
    for S in range(1, full):
        members = [j for j in range(m) if S >> j & 1]
        if len(members) < 2:
            continue
        for j in members:
            cost = C[S ^ (1 << j)] + Dm[:, j]               # come to j from any k in S \ {j} (others are inf)
            k = int(np.argmin(cost))
            C[S, j], parent[S, j] = cost[k], k
    S = full - 1
    j = int(np.argmin([C[S, j] + D[j + 1, 0] for j in range(n - 1)]))
    tour = []
    while S:
        tour.append(j + 1)
        S, j = S ^ (1 << j), int(parent[S, j])
    return [1] + [i + 1 for i in reversed(tour)]


def _canonical(tour0):
    """0-based tour -> 1-based, starting at city 1 ('we always start in the first city')."""
    k = tour0.index(0)
    return [i + 1 for i in tour0[k:] + tour0[:k]]


def nearest_neighbour(P):
    D = dist_matrix(P)
    tour, left = [0], set(range(1, len(P)))
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j])
        tour.append(nxt); left.remove(nxt)
    return _canonical(tour)


def two_opt(P, tour):
    """Reverse segments while that shortens the tour."""
    D = dist_matrix(P)
    t = [i - 1 for i in tour]
    n, improved = len(t), True
    while improved:
        improved = False
        for a in range(n - 1):
            for b in range(a + 2, n if a else n - 1):
                i, j, k, l = t[a], t[a + 1], t[b], t[(b + 1) % n]
                if D[i, k] + D[j, l] < D[i, j] + D[k, l] - 1e-12:
                    t[a + 1:b + 1] = reversed(t[a + 1:b + 1])
                    improved = True
    return _canonical(t)


def greedy_edge(P):
    """Repeatedly add the shortest edge that keeps every degree <= 2 and closes no early cycle."""
    n = len(P)
    D = dist_matrix(P)
    edges = sorted((D[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    deg, comp, adj = [0] * n, list(range(n)), [[] for _ in range(n)]

    def find(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]; x = comp[x]
        return x

    added = 0
    for _, i, j in edges:
        if deg[i] < 2 and deg[j] < 2 and find(i) != find(j):
            adj[i].append(j); adj[j].append(i); deg[i] += 1; deg[j] += 1
            comp[find(i)] = find(j); added += 1
            if added == n - 1:
                break
    start = next(i for i in range(n) if deg[i] < 2) if n > 1 else 0
    tour, prev = [start], None
    while len(tour) < n:
        nxt = next(k for k in adj[tour[-1]] if k != prev)
        prev = tour[-1]; tour.append(nxt)
    return _canonical(tour)


def christofides(P):
    import networkx as nx
    from networkx.algorithms.approximation import christofides as chr_
    D = dist_matrix(P)
    G = nx.Graph()
    n = len(P)
    G.add_weighted_edges_from((i, j, D[i, j]) for i in range(n) for j in range(i + 1, n))
    return _canonical(chr_(G)[:-1])


# Stand-ins for the paper's A1 / A2 / A3 (three GitHub solvers, refs [18]-[20]; A3 is Christofides + 2-opt)
A1 = greedy_edge
def A2(P): return two_opt(P, nearest_neighbour(P))
def A3(P): return two_opt(P, christofides(P))


def is_valid_tour(tour, n):
    return sorted(tour) == list(range(1, n + 1))


def brute_force_tsp(P):
    """For testing Held-Karp on tiny n."""
    n = len(P)
    best = min(itertools.permutations(range(2, n + 1)), key=lambda r: tour_length(P, [1, *r]))
    return [1, *best]


def sample_points(n, rng):
    return rng.random((n, 2)).tolist()
