"""Tests: each checks a definition, example or theorem of the Introduction to
*Perceptrons* (Minsky & Papert, 1969).

Run with:  python3 -m pytest -q
"""

from itertools import product

import numpy as np
import pytest

from minsky import (Perceptron, conjunctive, convexity_perceptron, fooling_figure, four_figures,
                    is_connected, is_line_convex, ladder, locality_identity_holds, mask_predicate,
                    order_of, point_predicate, representable, seesaw_perceptron, tips_right,
                    train_perceptron, window_features)


# ---- Section 0.8: definitions and examples ----

def test_perceptron_is_a_weighted_vote():
    X = np.array([[1, 0, 1]], bool)
    phis = [point_predicate((0, i)) for i in range(3)]
    assert Perceptron(phis, [1, 1, 1], 1.5)(X) == 1          # 2 black points > 1.5
    assert Perceptron(phis, [1, 1, 1], 2.5)(X) == 0


def test_order_and_diameter():
    phi = mask_predicate([(0, 0), (0, 3), (4, 0)])
    assert phi.order == 3
    assert phi.diameter == pytest.approx(5.0)                # (0,3) to (4,0)


def test_seesaw_matches_physics_for_every_arrangement():
    # Figure 0.3: sum (i - 4) phi_i(X) > 0 decides whether the seesaw tips right.
    S = seesaw_perceptron()
    assert S.order == 1
    for bits in product([0, 1], repeat=7):
        X = np.array([bits], bool)
        assert S(X) == tips_right(X)


def test_conjunctive_predicate_as_a_perceptron():
    # Example in Section 0.8: weights -1 and threshold -1 turn "all phi say yes" into a vote.
    phis = [point_predicate((0, i)) for i in range(3)]
    psi = conjunctive(phis)
    for bits in product([0, 1], repeat=3):
        assert psi(np.array([bits], bool)) == int(all(bits))


# ---- Section 0.6: CONVEX has order 3 ----

def test_convexity_perceptron_has_order_3_and_is_correct():
    P = convexity_perceptron((6, 6))
    assert P.order == 3
    rng = np.random.default_rng(0)
    for _ in range(200):
        X = rng.random((6, 6)) < rng.uniform(0.1, 0.9)
        assert P(X) == is_line_convex(X)


def test_convexity_examples():
    P = convexity_perceptron((6, 6))
    square = np.zeros((6, 6), bool); square[1:5, 1:5] = True
    ring = square.copy(); ring[2:4, 2:4] = False
    two_blobs = np.zeros((6, 6), bool); two_blobs[0, 0] = two_blobs[0, 5] = True
    assert P(square) == 1 and P(ring) == 0 and P(two_blobs) == 0


# ---- Theorem 0.6.1: CONNECTED is not conjunctively local of any order ----

@pytest.mark.parametrize("k", [1, 2, 5, 10, 20])
def test_any_k_point_test_is_fooled(k):
    rng = np.random.default_rng(k)
    n = k + 1                                  # more rungs than points
    Y0 = ladder(n)
    assert not is_connected(Y0) and is_connected(ladder(n, range(n)))
    all_points = list(zip(*np.nonzero(np.ones(Y0.shape, bool))))
    for _ in range(100):
        support = [all_points[i] for i in rng.choice(len(all_points), k, replace=False)]
        _, Y2 = fooling_figure(support, n)
        assert is_connected(Y2)                           # Y2 is connected...
        assert all(Y0[p] == Y2[p] for p in support)       # ...but looks like Y0 to the test


# ---- Theorem 0.8: no diameter-limited perceptron computes CONNECTED ----

def test_the_four_figures():
    f = four_figures(30)
    for diagonal in (False, True):
        assert [is_connected(f[k], diagonal) for k in [(0, 0), (0, 1), (1, 0), (1, 1)]] == \
               [False, True, True, False]


@pytest.mark.parametrize("d", [3, 5, 10, 20, 30])
def test_locality_identity_for_every_local_window(d):
    # The proof's key step, checked for every window position and every local predicate.
    assert locality_identity_holds(four_figures(30), d)


def test_identity_fails_once_a_window_sees_both_ends():
    assert not locality_identity_holds(four_figures(30), 40)


def test_identity_holds_for_random_weighted_sums_of_local_predicates():
    # Build random local perceptrons explicitly and check S(X11)-S(X10)-S(X01)+S(X00)=0.
    f = four_figures(30)
    F = window_features(f, 10)
    rng = np.random.default_rng(0)
    for _ in range(100):
        S = F @ rng.normal(size=F.shape[1])          # rows: 00, 01, 10, 11
        assert S[3] - S[2] - S[1] + S[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("d, learnable", [(5, False), (20, False), (30, False), (40, True)])
def test_perceptron_learning_fails_with_local_features(d, learnable):
    f = four_figures(30)
    hist = train_perceptron(window_features(f, d), np.array([0, 1, 1, 0]), epochs=500)
    assert (hist[-1] == 0) == learnable


# ---- Extension: order of parity (Chapter 3 of the book) ----

@pytest.mark.parametrize("n", [2, 3, 4])
def test_parity_has_order_n_and_and_has_order_1(n):
    parity = lambda X: sum(X) % 2 == 1
    assert order_of(n, parity) == n
    assert not representable(n, n - 1, parity)
    assert order_of(n, lambda X: all(X)) == 1
