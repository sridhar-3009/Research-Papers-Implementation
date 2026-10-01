"""A light tour of Zhang et al. (2017). Only small matrix computations; about a second.

    python3 demo.py
"""

import torch

from randomization import (MLP, SmallAlexNet, SmallInception, count_params, finite_sample_network,
                           finite_sample_predict, min_norm_interpolant, rademacher_fit, sgd_least_squares)

g = lambda s=0: torch.Generator().manual_seed(s)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. More parameters than training examples (Table 1)")
for name, m in (("small Inception", SmallInception()), ("small AlexNet", SmallAlexNet()),
                ("MLP 3x512", MLP(3)), ("MLP 1x512", MLP(1))):
    p = count_params(m)
    print(f"{name:16s} {p:>10,} parameters = {p / 50000:5.1f} per CIFAR-10 training image")
print("-> plenty of room to memorize every image, whatever its label.")

line("2. Theorem 1: 2n + d weights can fit ANY labels")
n, d = 500, 20
X = torch.randn(n, d)
for name, y in (("smooth labels  y = sin(x_1)", torch.sin(X[:, 0])),
                ("random labels  y ~ N(0, 1)", torch.randn(n, generator=g(1)))):
    a, b, w = finite_sample_network(X, y, g())
    err = (finite_sample_predict(a, b, w, X) - y.double()).abs().max().item()
    print(f"{name}: max training error {err:.1e} with {a.numel() + b.numel() + w.numel()} weights (2n + d = {2 * n + d})")
print("-> the construction: project every point onto one random direction, put a ReLU 'kink'")
print("   between consecutive projections, then solve a triangular linear system.")

line("3. Rademacher complexity: can the learner fit random +-1 labels?")
for n, d in ((100, 10), (100, 50), (100, 100), (100, 400)):
    Xr = torch.randn(n, d)
    if d >= n:
        fit = lambda X, s: X.double() @ min_norm_interpolant(X, s)
    else:
        fit = lambda X, s: X.double() @ torch.linalg.lstsq(X.double(), s.double()[:, None]).solution[:, 0]
    print(f"linear model, {n} points, {d:3d} features: Rademacher fit = {rademacher_fit(fit, Xr, 10, g()):.2f}")
print("-> once parameters >= data points, the score is 1.00: the complexity bound becomes useless.")
print("   Deep nets are in that regime (they fit random labels), yet they generalize on real labels.")

line("4. Section 5: which of the many perfect fits does SGD pick?")
X = torch.randn(30, 100)
y = torch.randn(30)
w_min = min_norm_interpolant(X, y)
w_sgd = sgd_least_squares(X, y, epochs=300, generator=g())
P = torch.eye(100, dtype=torch.float64) - torch.linalg.pinv(X.double()) @ X.double()   # moves inside the null space
w_other = w_min + P @ torch.randn(100, dtype=torch.float64, generator=g(3)) * 2
for name, w in (("SGD started at 0", w_sgd), ("kernel solution X^T (XX^T)^-1 y", w_min), ("another exact fit", w_other)):
    print(f"{name:32s} training error {(X.double() @ w - y.double()).abs().max():.1e}   ||w|| = {w.norm():.3f}")
print("-> 30 equations, 100 unknowns: infinitely many perfect fits. SGD from zero lands on the one")
print("   with the SMALLEST norm, an implicit regularizer nobody asked for.")
