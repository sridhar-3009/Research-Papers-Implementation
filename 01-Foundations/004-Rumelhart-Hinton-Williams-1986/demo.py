"""A guided tour of back-propagation (Rumelhart, Hinton & Williams, 1986).

Run:  python3 demo.py        (~5 seconds)
(For every figure and number of the paper, run experiments.py.)
"""

import numpy as np

from backprop import LayeredNet, train
from experiments import correct, solves_symmetry, weight_structure
from tasks import FAMILY_NET, PEOPLE, RELATIONS, family_cases, symmetry_data

np.set_printoptions(precision=2, suppress=True)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
section("1. One backward pass by hand: Eqs. (4)-(7) on a 2-1-1 net")
net = LayeredNet([("in", 2, []), ("hidden", 1, ["in"]), ("out", 1, ["hidden"])], rng=0)
x, d = np.array([[1.0, 0.0]]), np.array([[1.0]])
y = net.forward({"in": x})
print(f"   forward:  hidden y = {y['hidden'][0, 0]:.3f},  output y = {y['out'][0, 0]:.3f},  target d = 1")
dE_dy = y["out"] - d                                              # Eq. (4)
dE_dx = dE_dy * y["out"] * (1 - y["out"])                         # Eq. (5)
print(f"   Eq.(4)  dE/dy_out = y - d                = {dE_dy[0, 0]:+.4f}")
print(f"   Eq.(5)  dE/dx_out = dE/dy * y(1 - y)     = {dE_dx[0, 0]:+.4f}")
print(f"   Eq.(6)  dE/dw(hidden->out) = dE/dx * y_hidden = {(dE_dx * y['hidden'])[0, 0]:+.4f}")
dE_dy_h = dE_dx @ net.W[("hidden", "out")].T                      # Eq. (7)
print(f"   Eq.(7)  dE/dy_hidden = dE/dx * w       = {dE_dy_h[0, 0]:+.4f}   <- the error sent DOWN")
g = net.gradients({"in": x}, d)
print(f"   the code's gradients() gives the same:  {g[('W', 'hidden', 'out')][0, 0]:+.4f}")

# ---------------------------------------------------------------------------
section("2. Figure 1: learning to detect mirror symmetry (6 inputs, 2 hidden units)")
X, d = symmetry_data()
net = LayeredNet([("in", 6, []), ("hidden", 2, ["in"]), ("out", 1, ["hidden"])], init_range=0.3, rng=3)
hist = train(net, {"in": X}, d, sweeps=5000, eps=0.1, alpha=0.9, stop=lambda n: solves_symmetry(n, X, d))
print(f"   Solved all 64 input vectors after {len(hist)} sweeps (the paper's run took 1,425).")
W = net.W[("in", "hidden")].T
for j in range(2):
    print(f"   hidden unit {j + 1} weights: {W[j]}   bias {net.b['hidden'][j]:+.2f}")
anti, corr, ratio = weight_structure(net)
print(f"   -> mirror positions have equal and opposite weights (error {anti:.3f})")
print(f"   -> sizes on one side are 1 : {ratio[1]:.2f} : {ratio[2]:.2f}  (paper: 1 : 2 : 4)")
print("   Why it works: a symmetric input sends 0 to both hidden units (the halves cancel),")
print("   their negative biases keep them off, and the output's positive bias says 'yes'.")
for pattern in ["101101", "110011", "100000", "011010"]:
    xv = np.array([[int(c) for c in pattern]], float)
    out = net.forward({"in": xv})["out"][0, 0]
    print(f"      {pattern}: output {out:.2f}  -> {'symmetric' if out > 0.5 else 'not symmetric'}")
print("   (Symmetric ones are only just above 0.5 because training stopped the moment all 64")
print("   were on the right side. More sweeps push them toward 1.)")

section("   ...and without a hidden layer it can't be done")
flat = LayeredNet([("in", 6, []), ("out", 1, ["in"])], init_range=0.3, rng=0)
train(flat, {"in": X}, d, sweeps=3000, eps=0.1, alpha=0.9)
right = int(np.sum(np.abs(flat.forward({"in": X})["out"] - d) < 0.5))
print(f"   Best it manages: {right}/64, about the same as always saying 'not symmetric' (56/64).")
print("   Each input alone tells you nothing about symmetry, so adding up evidence fails:")
print("   the limit Minsky & Papert proved (Paper 003). Hidden units fix it.")

# ---------------------------------------------------------------------------
section("3. Figures 2-4: family trees - learning relationships")
P, R, D, keys = family_cases()
rng = np.random.default_rng(100)
test = rng.choice(len(keys), 4, replace=False)
tr = np.setdiff1d(np.arange(len(keys)), test)
net = LayeredNet(FAMILY_NET, init_range=1.0, rng=0)
train(net, {"person": P[tr], "relation": R[tr]}, D[tr], sweeps=1500,
      schedule=lambda t: (0.025, 0.5) if t < 20 else (0.05, 0.9), decay=0.0002, margin=(0.2, 0.8))
acc = correct(net.forward({"person": P[tr], "relation": R[tr]})["out"], D[tr]).mean()
print(f"   Network: 24 people + 12 relations -> 6 + 6 -> 12 -> 6 -> 24 people.  Training cases right: {acc:.0%}")
print("   Questions it was NEVER trained on:")
yt = net.forward({"person": P[test], "relation": R[test]})["out"]
for k, i in enumerate(test):
    who, rel = keys[i]
    answer = [PEOPLE[j] for j in np.flatnonzero(yt[k] > 0.5)]
    truth = [PEOPLE[j] for j in np.flatnonzero(D[i])]
    print(f"      {who}'s {rel}?  net says {answer or ['(nobody)']}, truth {truth}"
          f"  {'RIGHT' if answer == truth else 'wrong'}")
