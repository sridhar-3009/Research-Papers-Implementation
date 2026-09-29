"""A guided tour of "Efficient BackProp".  Run:  python3 demo.py   (~5 seconds)
(For every experiment and table, run experiments.py.)
"""

import numpy as np
from sklearn.datasets import load_digits

from efficient_backprop import (MLP, InputTransform, lecun_tanh, lms_batch, lms_hessian,
                                power_method, train_sgd, two_gaussians)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


section("1. The recommended sigmoid f(x) = 1.7159 tanh(2x/3)  (Section 4.4)")
for x in (-3, -1, 0, 1, 3):
    print(f"   f({x:+d}) = {lecun_tanh(x):+.3f}")
print("   f(+-1) = +-1: so +-1 are good targets (inside the range, where the curve bends most).")

section("2. Why learning rates must stay below 2 / lambda_max  (Section 5)")
X, D = two_gaussians(100, rng=0)
ev = np.linalg.eigvalsh(lms_hessian(X))
print(f"   Linear net on the 2-Gaussian data. Hessian eigenvalues: {np.round(ev, 3)}")
print(f"   lambda_max = {ev.max():.2f}  ->  fastest single rate 1/lambda_max = {1 / ev.max():.2f},"
      f" divergence above {2 / ev.max():.2f}")
for lr in (0.5, 1.0, 1.9, 2.1):
    h, _ = lms_batch(X, D, lr, epochs=60)
    print(f"   eta = {lr}: MSE {h[0]:.3f} -> {h[-1]:.3g}  {'(DIVERGES)' if h[-1] > h[0] else ''}")
print("   (The paper says the limit is 2/0.84 = 2.38, but forgets the bias's eigenvalue of ~1.)")

section("3. Shift the inputs away from zero -> one huge eigenvalue  (Section 5.3)")
for shift in (0, 3):
    ev = np.linalg.eigvalsh(lms_hessian(X + shift))
    print(f"   inputs + {shift}: eigenvalues {np.round(ev, 3)}   spread (condition number) {ev.max() / ev.min():.0f}")
print("   Bigger spread = the safe learning rate is tiny for the steep direction and")
print("   learning crawls along the flat one. Subtracting the mean fixes it.")

section("4. The whole recipe on real handwritten digits  (Section 10)")
d = load_digits()
idx = np.random.default_rng(0).permutation(len(d.target))
tr, te = idx[:1000], idx[1000:]
T = lambda y: np.where(np.eye(10)[y] == 1, 1.0, -1.0)
tf = InputTransform().fit(d.data[tr])
for name, A, B in (("raw inputs (0-16)", d.data[tr], d.data[te]),
                   ("normalized inputs", tf(d.data[tr]), tf(d.data[te]))):
    net = MLP([64, 30, 10], rng=0)                       # LeCun sigmoid + LeCun init
    train_sgd(net, A, T(d.target[tr]), epochs=3, lr=0.003 if name.startswith("norm") else 0.001, rng=0)
    acc = (net.predict(B).argmax(1) == d.target[te]).mean()
    print(f"   {name:18s}: test accuracy after 3 epochs = {acc:.1%}")

section("5. Estimating the best learning rate without computing the Hessian  (Section 9.2)")
net = MLP([64, 30, 10], rng=1)
A, Dd = tf(d.data[tr[:300]]), T(d.target[tr[:300]])
lam, _ = power_method(net, A, Dd, iters=40, rng=0)
print(f"   Power method with Hessian-vector products (2 gradients each): lambda_max = {lam:.2f}")
print(f"   Predicted optimal learning rate 1/lambda_max = {1 / lam:.3f}")
print("   (See results.md: right order of magnitude for batch learning; ~10x too big for SGD.)")
