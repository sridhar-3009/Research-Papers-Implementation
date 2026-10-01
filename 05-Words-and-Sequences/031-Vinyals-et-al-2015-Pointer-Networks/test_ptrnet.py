"""Tests for Pointer Networks (Vinyals, Fortunato & Jaitly 2015). A few seconds.

Run with:  python3 -m pytest -q
"""

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay

from ptrnet import END, PtrNet, beam_search, targets
from tasks import (A1, A2, A3, brute_force_tsp, convex_hull, delaunay, delaunay_metrics, held_karp, hull_metrics,
                   is_simple_polygon, is_valid_tour, polygon_area, same_polygon, sample_points, tour_length, two_opt)

rng = np.random.default_rng(0)
torch.manual_seed(0)


def test_convex_hull_target_format():
    for _ in range(20):
        P = sample_points(12, rng)
        h = convex_hull(P)
        assert h[0] == h[-1] == min(h)                                       # lowest index first, closed
        assert set(h) == {int(i) + 1 for i in ConvexHull(np.array(P)).vertices}
        xy = np.array([P[i - 1] for i in h[:-1]])
        signed = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
        assert signed > 0                                                     # counter-clockwise


def test_hull_metrics_polygon_equality_and_simplicity():
    assert same_polygon([2, 4, 3, 5, 2], [3, 5, 2, 4, 3])                    # same cycle, other start
    assert not same_polygon([2, 4, 3, 5, 2], [2, 3, 4, 5, 2])
    sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert is_simple_polygon(sq, [1, 2, 3, 4, 1]) and not is_simple_polygon(sq, [1, 3, 2, 4, 1])
    assert polygon_area(sq, [1, 2, 3, 4, 1]) == 1.0
    Ps = [sample_points(10, rng) for _ in range(5)]
    assert hull_metrics(Ps, [convex_hull(P) for P in Ps]) == {"accuracy": 100.0, "area": 100.0}
    assert hull_metrics(Ps, [[1, 2, 1]] * 5)["area"] == "FAIL"               # degenerate outputs


def test_delaunay_target_is_empty_circle_and_ordered():
    P = sample_points(15, rng)
    tris = delaunay(P)
    A = np.array(P)
    for t in tris:
        assert list(t) == sorted(t)
        a, b, c = A[[i - 1 for i in t]]
        M = np.array([[a[0] - c[0], a[1] - c[1]], [b[0] - c[0], b[1] - c[1]]])
        rhs = 0.5 * np.array([a @ a - c @ c, b @ b - c @ c])
        center = np.linalg.solve(M, rhs)
        r = np.linalg.norm(a - center)
        others = np.delete(A, [i - 1 for i in t], 0)
        assert np.all(np.linalg.norm(others - center, axis=1) > r - 1e-9)   # no point inside the circumcircle
    assert len(tris) == len(Delaunay(A).simplices)
    assert delaunay_metrics([P], [tris])["accuracy"] == 100
    assert abs(delaunay_metrics([P], [tris[:len(tris) // 2]])["triangle coverage"] - 100 * (len(tris) // 2) / len(tris)) < 1e-9


def test_held_karp_is_optimal_and_heuristics_are_valid():
    for n in (4, 6, 7):
        P = sample_points(n, rng)
        opt = held_karp(P)
        assert opt[0] == 1 and is_valid_tour(opt, n)
        assert abs(tour_length(P, opt) - tour_length(P, brute_force_tsp(P))) < 1e-9
        for algo in (A1, A2, A3):
            t = algo(P)
            assert is_valid_tour(t, n) and t[0] == 1
            assert tour_length(P, t) >= tour_length(P, opt) - 1e-9
        assert tour_length(P, A3(P)) <= 1.5 * tour_length(P, opt) + 1e-9   # Christofides' guarantee
        bad = list(range(1, n + 1))
        assert tour_length(P, two_opt(P, bad)) <= tour_length(P, bad) + 1e-12


def test_pointer_dictionary_grows_with_the_input():
    m = PtrNet(16)
    for n in (3, 7, 20):                                                       # one model, any n
        S = m.start(torch.rand(2, n, 2))
        lp, _ = m.step(S)
        assert lp.shape == (2, n + 1) and torch.allclose(lp.exp().sum(-1), torch.ones(2))
    base = PtrNet(16, mode="seq2seq", n_max=10)
    assert base.step(base.start(torch.rand(2, 4, 2)))[0].shape == (2, 11)      # fixed dictionary
    att = PtrNet(16, mode="attention", n_max=10)
    lp, S = att.step(att.start(torch.rand(2, 4, 2)))
    assert lp.shape == (2, 11) and S["feed"].shape == (2, 32)                  # [d ; d'] fed to the next step


def test_log_prob_is_the_chain_rule_with_copied_inputs():
    m = PtrNet(16, embed=False)
    P = torch.rand(1, 5, 2)
    C = [3, 1, 4]
    got = m.log_prob(P, targets([C]))
    S, total = m.start(P), 0
    x_expected = m.go
    for c in C + [END]:
        assert torch.allclose(S["x"][0], x_expected)                           # decoder input = P_{C_{i-1}}
        lp, S = m.step(S)
        total = total + lp[0, c]
        S = m.feed(S, torch.tensor([c]))
        x_expected = P[0, c - 1] if c > 0 else torch.zeros(2)
    assert torch.allclose(got[0], total, atol=1e-6)
    padded = m.log_prob(P.expand(2, -1, -1), targets([C, [1, 2, 3, 4, 5]]))  # -1 padding is ignored
    assert torch.allclose(padded[0], got[0], atol=1e-6)


def test_constrained_beam_search_returns_valid_tours():
    m = PtrNet(16).eval()
    for n in (4, 8):
        t = beam_search(m, torch.rand(n, 2), beam=3, constraint="tour")
        assert is_valid_tour(t, n)


def test_ptr_net_learns_to_sort():
    torch.manual_seed(0)
    m = PtrNet(64, in_dim=1)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(300):
        n = int(torch.randint(5, 11, ()))
        X = torch.rand(64, n, 1)
        loss = -m.log_prob(X, targets((X[..., 0].argsort(1) + 1).tolist())).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
    m.eval()
    hits = 0
    for _ in range(50):
        x = torch.rand(5, 1)
        hits += beam_search(m, x) == (x[:, 0].argsort() + 1).tolist()
    assert hits >= 30
