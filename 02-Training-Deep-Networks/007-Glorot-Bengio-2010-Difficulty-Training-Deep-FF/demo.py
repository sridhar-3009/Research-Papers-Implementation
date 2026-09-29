"""A guided tour of Glorot & Bengio (2010).  Run:  python3 demo.py   (~1 minute)
(For Table 1 and every figure, run experiments.py.)
"""

import numpy as np
import torch

from glorot import CLASSES, SHAPES, DeepNet, gradient_stats, jacobian_singular_values, nll, shapeset

torch.manual_seed(0)
np.set_printoptions(precision=3, suppress=True)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


section("1. Shapeset-3x2: which shapes are in the picture? (Section 2.1)")
X, y = shapeset(4, rng=3)
for k in range(2):
    print(f"   label: {' + '.join(SHAPES[s] for s in CLASSES[y[k]])}")
    img = X[k].reshape(32, 32)
    for r in img[::2]:
        print("      " + "".join("#" if v > 0.6 else "+" if v > 0 else "." for v in r))

section("2. At initialization: standard vs normalized init (5 tanh layers of 1000)")
X, y = shapeset(300, rng=7)
X, y = torch.tensor(X), torch.tensor(y)
for init in ("standard", "normalized"):
    net = DeepNet(1024, 9, act="tanh", init=init)
    act, back, wgrad = gradient_stats(net, X, y)
    print(f"\n   {init} init")
    print(f"     activation spread,  layer 1 -> 5 : {np.array(act)}")
    print(f"     gradient flowing DOWN, layer 1 -> 5 (x1e-5): {np.array(back[:5]) * 1e5}")
    print(f"     average Jacobian singular value : {np.mean(jacobian_singular_values(net, X)):.2f}")
print("\n   Standard init: each layer multiplies the signal by ~0.5, so activations fade going up")
print("   and gradients fade going down. Normalized init keeps it near 0.8 and both stay level.")

section("3. Why? The variance argument (Section 4.2.1)")
n = 1000
print(f"   Standard init: n Var[W] = n * 1/(3n) = {1/3:.3f}  -> after 5 layers: {(1/3)**5:.4f} of the variance left")
print(f"   Normalized init: Var[W] = 2/(n_in + n_out) = {2/(2*n):.4f}, so n Var[W] = 1 when layers are equal")

section("4. A few hundred updates on Shapeset: sigmoid vs tanh (Section 3.1)")
for act in ("sigmoid", "tanh"):
    net = DeepNet(1024, 9, act=act, init="normalized" if act == "tanh" else "standard")
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    means = []
    for k in range(401):
        if k % 100 == 0:
            with torch.no_grad():
                means.append(float(net(X, keep=True)[2][-1].mean()))
        Xb, yb = shapeset(10, rng=1000 + k)
        opt.zero_grad(); nll(net(torch.tensor(Xb)), torch.tensor(yb)).backward(); opt.step()
    print(f"   {act:8s}: mean activation of the TOP hidden layer after 0,100,..,400 updates: "
          + ", ".join(f"{m:+.2f}" for m in means))
print("   The sigmoid's top layer is pushed from ~0.5 down toward 0, which is its saturated")
print("   floor: there f'(s) ~ 0 and gradients stop flowing back (Figure 2). tanh sits near 0")
print("   too, but 0 is the MIDDLE of tanh, where it's steepest, so nothing gets stuck.")
