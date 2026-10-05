"""Matrix Relational Embedding in ~30 seconds: objects and relations as 4x4 matrices trained with the discriminative
cost (Eq. 1) on mod-12 arithmetic and the two family trees; then relations understood ONLY from higher-order facts
like (3, +3) in plus or (has_father, has_mother) in higher_oppsex, with the squared-error cost (Eq. 2) vs the
discriminative one, and learning a new relation incrementally with everything else frozen."""

import time

import numpy as np

import mre as M

T0 = time.time()
SEEDS = range(3)


def section(t):
    print(f"\n=== {t} ===")


section("1. The cost and its gradient")
m = M.MRE(5, 3, N=3, seed=0)
basic = np.array([(m.rel(0), 1, 2), (m.rel(1), 0, 3), (m.rel(2), 4, 4)])
higher = np.array([(m.rel(0), m.rel(1), m.rel(2)), (m.rel(2), 0, m.rel(1))])
c, g = m.cost_grad(m.E, basic, higher)
num = np.zeros_like(m.E)
for idx in np.ndindex(m.E.shape):
    Ep, Em = m.E.copy(), m.E.copy()
    Ep[idx] += 1e-6; Em[idx] -= 1e-6
    num[idx] = (m.cost_grad(Ep, basic, higher)[0] - m.cost_grad(Em, basic, higher)[0]) / 2e-6
print(f"  analytic vs finite-difference gradient of Eq. 1 + Eq. 2 + weight decay: max difference {np.abs(num - g).max():.1e}")

section("2. Modular arithmetic: 12 numbers, relations +0..+11 and x0..x11 (288 facts), 4x4 matrices (cf. Table 1)")
for n in (30, 60, 90):
    res = [M.run_arithmetic(n, seed=s) for s in SEEDS]
    print(f"    {n} held out: test errors {[r[0] for r in res]} (mean {np.mean([r[0] for r in res]):.1f}); training "
          f"errors {[r[1] for r in res]}")
print("  paper (5 runs): 30 -> 0.0, 60 -> 6.8, 90 -> 24.0; random guessing gets >= 90% wrong")

section("3. Family trees: 24 people in two isomorphic trees, 12 relations, 112 cases (cf. Table 2)")
people, cases = M.family_task()
print(f"  our tree definitions give {len(cases)} cases (the paper: 112)")
for n in (10, 20, 30):
    res = [M.run_family(n, seed=s) for s in SEEDS]
    print(f"    {n} held out: test errors {[r[0] for r in res]} (mean {np.mean([r[0] for r in res]):.1f}); training "
          f"errors {[r[1] for r in res]}")
print("  paper: 10 -> 0.4, 20 -> 1.2, 30 -> 2.0")

section("4. Higher-order arithmetic: ALL basic facts of +k removed, +k known only from plus / minus / inverse (Table 3)")
means = {"eq2": [], "disc": []}
for k in (4, 10):
    eq2 = [M.higher_order_arithmetic(k, seed=s)[0] for s in SEEDS]
    disc = [M.higher_order_arithmetic(k, seed=s, discriminative_higher=True)[0] for s in SEEDS]
    inc = [M.higher_order_arithmetic(k, seed=s, incremental=True)[0] for s in SEEDS]
    means["eq2"] += eq2; means["disc"] += disc
    print(f"    +{k:<2d} (12 queries): Eq. 2 errors {eq2}   discriminative higher-order cost {disc}   incremental {inc}")
print(f"  mean errors over both relations: Eq. 2 {np.mean(means['eq2']):.1f}, discriminative {np.mean(means['disc']):.1f} of 12")
print("  paper: +4 2.6, +10 3.6 mean errors (Eq. 2); the discriminative version was 'only slightly better than chance';")
print("  incremental (Table 5): +4 5.8, +10 4.4")

section("5. Higher-order family trees: has_father known only from higher_oppsex (Table 4)")
for r in ("has_father", "has_sister"):
    joint = [M.higher_order_family(r, seed=s)[0] for s in SEEDS]
    inc = [M.higher_order_family(r, seed=s, incremental=True)[0] for s in SEEDS]
    n = M.higher_order_family(r, seed=0)[1]
    print(f"    {r:11s} ({n} queries): joint training errors {joint}   incremental {inc}")
print("  paper: has_father 2.4 (runs 0, 12, 0, 0, 0), has_sister 0.4; incremental 2.0 and 0.0")
print("  -> as in the paper, most runs get every query right and the occasional run fails completely (a poor local")
print("     minimum of the non-convex cost); the discriminative cost for higher-order facts is worse on average because")
print("     it only needs '3 x plus' to be CLOSER to +3 than to other relations, not equal to it (in our runs it lands")
print("     between Eq. 2 and chance, which is ~11 of 12 wrong; the paper called its first attempt 'only slightly better")
print("     than chance')")

section("What the paper reports (verified against the PDF)")
for k, v in M.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
