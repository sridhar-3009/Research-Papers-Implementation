"""A guided tour of the Introduction to *Perceptrons* (1969).  Run:  python3 demo.py"""

import numpy as np

from minsky import (convexity_perceptron, fooling_figure, four_figures, is_connected,
                    is_line_convex, ladder, locality_identity_holds, order_of,
                    seesaw_perceptron, tips_right, train_perceptron, window_features)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(X, indent="   "):
    for row in X:
        print(indent + "".join("#" if v else "." for v in row))


# ---------------------------------------------------------------------------
section("Section 0.8, Figure 0.3 - the seesaw: an order-1 perceptron")
S = seesaw_perceptron()
print("   psi = 1  iff  sum (i - 4) * [pebble at i] > 0     (pivot at position 4)")
for pebbles in ["1000000", "0000001", "1000011", "0110010"]:
    X = np.array([[c == "1" for c in pebbles]])
    print(f"   pebbles {pebbles}: perceptron says {'tips right' if S(X) else 'does not':10s}"
          f"  physics says {'tips right' if tips_right(X) else 'does not'}")
print(f"   order = {S.order} (each partial predicate looks at ONE point)")

# ---------------------------------------------------------------------------
section("Section 0.6 - CONVEX is local: an order-3 perceptron")
P = convexity_perceptron((6, 6))
print(f"   {len(P.phis)} partial predicates, each looking at 3 points p, q, r (q between p and r).")
print("   One fires if p and r are black but q is white. psi = 1 iff NONE fires.\n")
sq = np.zeros((6, 6), bool); sq[1:5, 1:5] = True
L = sq.copy(); L[1:3, 3:5] = False
ring = sq.copy(); ring[2:4, 2:4] = False
for name, X in [("square", sq), ("L-shape", L), ("ring", ring)]:
    print(f"   {name}: convex = {P(X)}")
    show(X, "      ")
rng = np.random.default_rng(0)
agree = sum(P(X) == is_line_convex(X) for X in (rng.random((6, 6)) < rng.uniform(.1, .9) for _ in range(300)))
print(f"   On 300 random figures the perceptron agrees with the direct check {agree}/300 times.")

# ---------------------------------------------------------------------------
section("Theorem 0.6.1 - CONNECTED is not conjunctively local, of ANY order")
n = 6
Y0 = ladder(n)
print("   Y0: two separate rails (NOT connected). Some local test must say 'no' to it.")
show(Y0)
support = [(2, 1), (2, 3), (1, 5), (3, 9), (2, 11)]      # a test that looks at 5 points
j, Y2 = fooling_figure(support, n)
print(f"\n   Say that test looks at 5 points: {support}.")
print(f"   With {n} rungs, rung {j} contains none of them. Add just that rung -> Y2:")
show(Y2)
print(f"   Y2 is connected ({is_connected(Y2)}), but the test sees exactly what it saw on Y0,")
print("   so it still says 'no'. Whatever k is, use more than k rungs: no order works.")

# ---------------------------------------------------------------------------
section("Theorem 0.8 - no DIAMETER-LIMITED perceptron can compute CONNECTED")
figs = four_figures(30)
for key, label in [((0, 0), "X00"), ((0, 1), "X01"), ((1, 0), "X10"), ((1, 1), "X11")]:
    print(f"   {label}: connected = {is_connected(figs[key])}")
    show(figs[key], "      ")
print("\n   A local test can't see both ends at once, so for EVERY local test:")
print("      phi(X11) - phi(X10) - phi(X01) + phi(X00) = 0")
print("   and any weighted sum S obeys the same identity. If S > theta on X01 and X10")
print("   but S <= theta on X00, then S(X11) > theta: X11 would be called connected. Wrong!\n")
y = np.array([0, 1, 1, 0])                                   # connected? for 00, 01, 10, 11
for d in [5, 15, 30, 40]:
    F = window_features(figs, d)
    hist = train_perceptron(F, y, epochs=2000)
    verdict = f"learned in {len(hist)} epochs" if hist[-1] == 0 else f"still {hist[-1]} mistakes after {len(hist)} epochs"
    print(f"   windows {d:2d} wide ({F.shape[1]:3d} local features): identity holds = "
          f"{str(locality_identity_holds(figs, d)):5s} -> perceptron rule: {verdict}")
print("   Only windows as wide as the whole figure (36 columns) make it learnable.")

# ---------------------------------------------------------------------------
section("Extension (Chapter 3 of the book): the ORDER of parity grows with the retina")
for n in [2, 3, 4]:
    print(f"   {n} points:  AND has order {order_of(n, lambda X: all(X))},  "
          f"PARITY has order {order_of(n, lambda X: sum(X) % 2 == 1)}")
print("   XOR is parity on 2 points: order 2, so a perceptron that sees one input")
print("   per partial predicate (like a single neuron on raw inputs) can't compute it.")
