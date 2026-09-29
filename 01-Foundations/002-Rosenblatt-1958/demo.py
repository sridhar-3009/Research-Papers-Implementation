"""A short tour of Rosenblatt's perceptron.  Run:  python3 demo.py
(For every figure and conclusion of the paper, run experiments.py.)
"""

import numpy as np

from experiments import ideal_environment, prototype_classes
from perceptron import Photoperceptron
from theory import Pa, Pc, overlap_to_LG


def section(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def show_side_by_side(*images, side=20, titles=()):
    """Print 20x20 stimuli next to each other: # = lit S-point, . = dark."""
    print("   " + "   ".join(t.ljust(side) for t in titles))
    rows = [np.asarray(img).reshape(side, side) for img in images]
    for r in range(side):
        print("   " + "   ".join("".join("#" if v else "." for v in img[r]) for img in rows))


rng = np.random.default_rng(0)

section("1. The retina (20 x 20 S-points): a class prototype and two noisy members")
S, lab = prototype_classes(300, rng)
examples = S[lab == 0][:2]
from experiments import PROTOTYPES
show_side_by_side(PROTOTYPES[0], examples[0], examples[1],
                  titles=("prototype 0", "example (30% flipped)", "another example"))
print("   Members of a class share most points with the prototype, but no two are identical.")

section("2. A-units: random wiring, fixed threshold (Eq. 1 theory vs the real machine)")
p = Photoperceptron(400, 2000, x=5, y=5, theta=3, rng=1)
A = p.activate(S)
print(f"   each A-unit: 5 excitatory + 5 inhibitory random connections, fires if e - i >= 3")
print(f"   theory  Pa = {Pa(0.5, 5, 5, 3):.4f}   (stimuli light about half the retina)")
print(f"   machine Pa = {A.mean():.4f}   (fraction of A-units firing, averaged)")
print(f"   theory  Pc for two stimuli sharing 70% of points = {Pc(0.5, *overlap_to_LG(0.5, 0.7), 5, 5, 3):.4f}")

section("3. Learning (gamma system): forced training on 300 examples, test on 400 NEW ones")
T, lt = prototype_classes(400, rng)
p.train_forced(S, lab, "gamma")
print(f"   P_r (recall of training examples) = {p.p_correct(S, lab):.3f}")
print(f"   P_g (brand-new examples)          = {p.p_correct(T, lt):.3f}")
print("   It learned the CLASSES, not just the examples: the A-units that tend to fire")
print("   for class 0 gained value in R0's source-set, and the rest lost.")

section("4. Random stimuli: memorizing is possible, generalizing is not")
for n in [20, 200, 2000]:
    q = Photoperceptron(400, 2000, x=5, y=5, theta=3, rng=2)
    R_ = ideal_environment(2 * n, rng)
    lab_r = np.repeat([0, 1], n)
    q.train_forced(R_, lab_r, "gamma")
    print(f"   {2 * n:5d} random patterns learned: recall {q.p_correct(R_, lab_r):.3f},"
          f"  new random patterns {q.p_correct(ideal_environment(400, rng), rng.integers(0, 2, 400)):.3f}")
print("   Recall falls toward 50% as memory fills up; new patterns are always ~50%.")

section("5. Distributed memory: remove A-units after learning")
for frac in [0.0, 0.25, 0.5, 0.75]:
    p.alive[:] = True
    p.alive[rng.permutation(2000)[:int(frac * 2000)]] = False
    print(f"   {frac:4.0%} of A-units removed -> P_g = {p.p_correct(T, lt):.3f}")
p.alive[:] = True
print("   No single memory is stored in one place, so damage costs a little everywhere.")
