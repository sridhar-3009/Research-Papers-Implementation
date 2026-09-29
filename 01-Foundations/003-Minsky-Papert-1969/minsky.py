"""Minsky & Papert, *Perceptrons* (1969), Introduction: the definitions and theorems.

A figure X is a set of black points on a grid "retina" (a 2-D boolean array).

A PERCEPTRON (Section 0.8) computes a yes/no predicate psi(X) in two stages:
  Stage I   many simple "partial predicates" phi_1(X), ..., phi_n(X), each 0 or 1,
            each looking at only part of the retina (its "support");
  Stage II  a weighted vote:  psi(X) = 1  if  sum_i alpha_i * phi_i(X) > theta.

The ORDER of a perceptron = the most points any phi looks at.
Its DIAMETER = the widest spread of points any phi looks at.

This module builds perceptrons, the convexity perceptron of order 3 (Section 0.6),
the seesaw of Figure 0.3, and machine-checked versions of Theorem 0.6.1 and
Theorem 0.8, which say that CONNECTEDNESS can't be computed locally.
"""

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from math import gcd

import numpy as np


# ---------------------------------------------------------------------------
# Partial predicates and perceptrons (Sections 0.5 and 0.8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predicate:
    """A partial predicate phi: a yes/no test that looks only at `support` points.

    rule(values) gets the 0/1 values of X at the support points, in order.
    """
    support: tuple            # tuple of (row, col) points
    rule: object              # function: tuple of 0/1 -> 0/1
    name: str = ""

    def __call__(self, X):
        return int(self.rule(tuple(int(X[p]) for p in self.support)))

    @property
    def order(self):
        return len(self.support)

    @property
    def diameter(self):
        pts = np.array(self.support)
        if len(pts) < 2:
            return 0.0
        return float(max(np.hypot(*(a - b)) for a, b in combinations(pts, 2)))


def point_predicate(p):
    """phi_p(X) = 1 if point p is black (Section 0.5, the simplest predicate)."""
    return Predicate((p,), lambda v: v[0], f"[{p} in X]")


def mask_predicate(points):
    """phi_A(X) = 1 if every point of A is black: 'A is a subset of X' (Section 0.5)."""
    return Predicate(tuple(points), lambda v: all(v), f"[{set(points)} subset of X]")


@dataclass
class Perceptron:
    """psi(X) = 1 iff sum(alpha_i * phi_i(X)) > theta   (Section 0.8)."""
    phis: list
    alphas: list
    theta: float

    def __call__(self, X):
        return int(sum(a * phi(X) for a, phi in zip(self.alphas, self.phis)) > self.theta)

    @property
    def order(self):
        return max(phi.order for phi in self.phis)

    @property
    def diameter(self):
        return max(phi.diameter for phi in self.phis)


def conjunctive(phis):
    """A 'conjunctively local' predicate: psi = 1 iff EVERY phi says 1 (Section 0.6).

    As a perceptron (the Example in Section 0.8): use the opposite tests
    (1 - phi) with weight -1 and threshold -1. Then the sum is > -1 exactly
    when none of the opposite tests fires, i.e. when every phi says 1.
    """
    flipped = [Predicate(phi.support, (lambda r: lambda v: 1 - r(v))(phi.rule), "not " + phi.name)
               for phi in phis]
    return Perceptron(flipped, [-1] * len(flipped), -1)


# ---------------------------------------------------------------------------
# The seesaw (Figure 0.3): an order-1 perceptron
# ---------------------------------------------------------------------------

def seesaw_perceptron():
    """Seven pebble positions 1..7 with the pivot at position 4.
    'Tips to the right'  <=>  sum_i (i - 4) * phi_i(X) > 0."""
    phis = [point_predicate((0, i)) for i in range(7)]
    alphas = [(i + 1) - 4 for i in range(7)]
    return Perceptron(phis, alphas, 0)


def tips_right(X):
    """Physics check: total torque of the pebbles about the pivot is positive."""
    return int(sum(((i + 1) - 4) * int(X[0, i]) for i in range(7)) > 0)


# ---------------------------------------------------------------------------
# Convexity: conjunctively local of order 3 (Section 0.6)
# ---------------------------------------------------------------------------

def points_between(p, r):
    """Grid points strictly inside the segment from p to r."""
    (y1, x1), (y2, x2) = p, r
    g = gcd(abs(y2 - y1), abs(x2 - x1))
    dy, dx = (y2 - y1) // g if g else 0, (x2 - x1) // g if g else 0
    return [(y1 + k * dy, x1 + k * dx) for k in range(1, g)]


def convexity_perceptron(shape):
    """The order-3 perceptron for CONVEX on a retina of the given shape.

    Section 0.6: X fails to be convex iff there are three points with q on the
    segment from p to r, p and r in X, and q NOT in X. One partial predicate per
    such triplet checks exactly that; psi = 1 iff none of them fires.
    """
    pts = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    phis = []
    for p, r in combinations(pts, 2):
        for q in points_between(p, r):
            # fires (=1) when the triplet shows X is NOT convex
            phis.append(Predicate((p, q, r), lambda v: v[0] and not v[1] and v[2], f"gap {q} on {p}-{r}"))
    # psi = 1 iff sum(-1 * phi) > -1, i.e. no triplet fires
    return Perceptron(phis, [-1] * len(phis), -1)


def is_line_convex(X):
    """Direct check of the same definition: every grid point between two black
    points is black."""
    black = list(zip(*np.nonzero(X)))
    return int(all(X[q] for p, r in combinations(black, 2) for q in points_between(p, r)))


# ---------------------------------------------------------------------------
# Connectedness
# ---------------------------------------------------------------------------

def is_connected(X, diagonal=False):
    """True if all black points form ONE piece (4-neighbours, or 8 with diagonal).
    The empty figure counts as connected."""
    black = list(zip(*np.nonzero(X)))
    if not black:
        return True
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen, todo = {black[0]}, deque([black[0]])
    while todo:
        y, x = todo.popleft()
        for dy, dx in steps:
            q = (y + dy, x + dx)
            if 0 <= q[0] < X.shape[0] and 0 <= q[1] < X.shape[1] and X[q] and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(black)


# ---- Theorem 0.6.1: CONNECTED is not conjunctively local of ANY order ----

def ladder(n_rungs, filled=()):
    """Two long horizontal rails with n_rungs possible 'middle squares' between them.

    Y0 = ladder(n)             -> no rungs: two separate rails, NOT connected
    Y1 = ladder(n, all rungs)  -> connected
    Y2 = ladder(n, {j})        -> just rung j: connected
    """
    width = 2 * n_rungs + 1
    X = np.zeros((5, width), bool)
    X[0, :] = X[4, :] = True                          # the two rails
    for j in filled:
        X[1:4, 2 * j + 1] = True                      # middle square j joins them
    return X


def fooling_figure(support, n_rungs):
    """The heart of the proof of Theorem 0.6.1.

    Suppose a partial predicate looks only at the points in `support` (at most k of
    them) and says 0 on Y0 (as one of them must, since Y0 is disconnected). With more
    rungs than points, some rung j contains none of those points. Add just that rung:
    the new figure Y2 is connected, yet the predicate sees exactly what it saw on Y0,
    so it still says 0 and wrongly rejects a connected figure.
    Returns (j, Y2).
    """
    touched = {(p[1] - 1) // 2 for p in support if 1 <= p[0] <= 3 and p[1] % 2 == 1}
    for j in range(n_rungs):
        if j not in touched:
            return j, ladder(n_rungs, {j})
    raise ValueError("support touches every rung: use more rungs than points")


# ---- Theorem 0.8: no DIAMETER-LIMITED perceptron computes CONNECTED ----

def four_figures(length=30):
    """The four figures X00, X01, X10, X11 of Theorem 0.8.

    Four horizontal strands (rows 0, 2, 4, 6). Each END is closed off in one of
    two ways:
      state 0: strands 1-2 joined, and 3-4 joined        (two small U-turns)
      state 1: strands 1-4 joined outside, 2-3 inside    (nested U-turns)
    The resulting pieces:
      X00: loops {1,2} and {3,4}   -> 2 pieces, NOT connected
      X11: loops {1,4} and {2,3}   -> 2 pieces, NOT connected
      X01, X10: 1 -> 2 -> 3 -> 4 -> 1, one long loop -> connected
    The two ends are `length` columns apart, much wider than any local predicate.
    """
    figs = {}
    for a in (0, 1):
        for b in (0, 1):
            X = np.zeros((7, length + 6), bool)
            X[[2, 4], 2:length + 4] = True                     # inner strands
            X[[0, 6], 2:length + 4] = True                     # outer strands
            for state, (inner_col, outer_col, stub) in [(a, (2, 0, range(0, 2))),
                                                        (b, (length + 3, length + 5, range(length + 4, length + 6)))]:
                if state == 0:
                    X[0:3, inner_col] = True                   # join 1-2
                    X[4:7, inner_col] = True                   # join 3-4
                else:
                    X[0, list(stub)] = X[6, list(stub)] = True # outer strands reach out...
                    X[0:7, outer_col] = True                   # ...and join 1-4
                    X[2:5, inner_col] = True                   # join 2-3
            figs[(a, b)] = X
    return figs


def local_windows(shape, d):
    """Every d x d square window on the retina. A predicate whose support has
    diameter < d always fits inside one of them."""
    return [(slice(i, i + d), slice(j, j + d))
            for i in range(max(1, shape[0] - d + 1)) for j in range(max(1, shape[1] - d + 1))]


def locality_identity_holds(figs, d):
    """The key step of the proof of Theorem 0.8.

    For every window W of size d, check that W can't see BOTH ends: either the left
    end looks the same in X0b and X1b, or the right end looks the same in Xa0 and
    Xa1. Then EVERY predicate phi inside W, whatever its rule, satisfies
        phi(X11) - phi(X10) - phi(X01) + phi(X00) = 0,
    and so does any weighted sum of such predicates. That makes a correct perceptron
    impossible:
        S(X11) = S(X10) + S(X01) - S(X00) > theta + theta - theta = theta,
    so it would call the disconnected X11 connected.
    Returns True if the identity holds for every window.
    """
    X = figs
    for W in local_windows(X[0, 0].shape, d):
        left_blind = all(np.array_equal(X[0, b][W], X[1, b][W]) for b in (0, 1))
        right_blind = all(np.array_equal(X[a, 0][W], X[a, 1][W]) for a in (0, 1))
        if not (left_blind or right_blind):
            return False
    return True


def window_features(figs, d):
    """Diameter-limited partial predicates for the four figures: for every window
    W and every pattern seen there, phi(X) = 1 iff X looks exactly like that
    pattern inside W. Any diameter-limited predicate is a function of such
    window contents, so these span everything a diameter-limited perceptron could use.
    Returns a (4, n_features) 0/1 matrix, rows in the order 00, 01, 10, 11."""
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    cols = []
    for W in local_windows(figs[0, 0].shape, d):
        patterns = {figs[k][W].tobytes() for k in keys}
        for pat in patterns:
            cols.append([int(figs[k][W].tobytes() == pat) for k in keys])
    return np.array(cols).T


def train_perceptron(F, y, epochs=1000):
    """Plain perceptron learning (Rosenblatt's rule) on feature matrix F, labels y.
    Returns the number of mistakes in each epoch; ends early at 0 mistakes."""
    w, b = np.zeros(F.shape[1]), 0.0
    history = []
    for _ in range(epochs):
        mistakes = 0
        for f, t in zip(F, y):
            out = int(f @ w + b > 0)
            if out != t:
                w += (t - out) * f
                b += (t - out)
                mistakes += 1
        history.append(mistakes)
        if mistakes == 0:
            break
    return history


# ---------------------------------------------------------------------------
# Extension (Chapter 3 of the book, not in this Introduction): parity's order
# ---------------------------------------------------------------------------

def representable(n, k, target):
    """Can a perceptron whose partial predicates are masks of at most k of the n
    points compute `target` (a function of n bits)? Solved exactly as a linear
    feasibility problem: find alpha_A for all |A| <= k with
        sum_A alpha_A [A subset of X] >= 1  when target(X) = 1
        sum_A alpha_A [A subset of X] <= -1 when target(X) = 0.
    (Allowing only masks loses nothing: every predicate on k points is a weighted
    sum of masks on those points.)"""
    from scipy.optimize import linprog
    masks = [A for size in range(k + 1) for A in combinations(range(n), size)]
    A_ub, b_ub = [], []
    for bits in range(2 ** n):
        X = [(bits >> i) & 1 for i in range(n)]
        row = [int(all(X[i] for i in A)) for A in masks]
        if target(X):
            A_ub.append([-v for v in row]); b_ub.append(-1)
        else:
            A_ub.append(row); b_ub.append(-1)
    res = linprog(np.zeros(len(masks)), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * len(masks), method="highs")
    return res.status == 0


def order_of(n, target):
    """The smallest k for which `target` on n points is representable."""
    return next(k for k in range(n + 1) if representable(n, k, target))
