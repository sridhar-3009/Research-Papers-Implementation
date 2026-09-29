"""A guided tour of Sutskever et al. (2013). Everything here is tiny: runs in a second.
Run:  python3 demo.py
(For the autoencoder and RNN experiments, run experiments.py on a strong machine.)
"""

import numpy as np
import torch

from momentum import RNN, addition_problem, mu_schedule, quadratic_run, sparse_init_


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


section("1. Classical momentum vs Nesterov on a long, narrow valley (Section 2.1)")
A = np.diag([1.0, 100.0])                  # flat direction curvature 1, steep direction 100
for name, nes in (("classical momentum (CM)", False), ("Nesterov (NAG)", True)):
    p = quadratic_run(A, np.zeros(2), [-10, 1], lr=0.01, mu=0.9, steps=100, nesterov=nes)
    flips = int(np.sum(np.diff(np.sign(p[:, 1])) != 0))
    print(f"   {name:24s}: zig-zags across the valley {flips:3d} times, distance to minimum after 100 steps "
          f"{np.linalg.norm(p[-1]):.4f}")
print("   NAG looks ahead (gradient at theta + mu*v), sees it's about to overshoot, and brakes.")

section("2. Theorem 2.1: NAG = CM with smaller momentum in steep directions")
eps, mu = 0.01, 0.9
for lam in (1, 10, 50, 100):
    print(f"   curvature {lam:3d}: NAG's effective momentum mu(1 - lambda*eps) = {mu * (1 - lam * eps):.3f}   (CM: {mu})")
print("   Steep directions (big lambda) get less momentum -> fewer oscillations.")

section("3. ...but careful: when lambda*eps > 1, the effective momentum turns NEGATIVE")
p_cm = quadratic_run(A, np.zeros(2), [-10, 1], 0.015, 0.95, 100, False)
p_nag = quadratic_run(A, np.zeros(2), [-10, 1], 0.015, 0.95, 100, True)
print(f"   lr 0.015, mu 0.95 (lambda*eps = 1.5): CM ends {np.linalg.norm(p_cm[-1]):.3f} from the minimum,"
      f" NAG ends {np.linalg.norm(p_nag[-1]):.2e}  <- NAG diverged")

section("4. The momentum schedule of Eq. (5)")
for t in (0, 250, 500, 1000, 2500, 12500, 50000, 250000):
    print(f"   update {t:6d}: mu = {mu_schedule(t, 0.999):.4f}")
print("   Start gentle (0.5), grow toward 1 as training goes on, capped at mu_max.")

section("5. Initializations")
W = sparse_init_(torch.empty(784, 1000), 15)
print(f"   Sparse init (Section 3.1): every unit has exactly {int((W[:, 0] != 0).sum())} non-zero incoming weights,"
      f" out of 784 possible.")
r = RNN(2, 100, 1, radius=1.1)
print(f"   RNN init (Table 4): spectral radius of the recurrent weights = "
      f"{float(torch.linalg.eigvals(r.W_hh).abs().max()):.3f} (just above 1: echoes last, but don't explode)")

section("6. The addition problem (Section 4)")
x, y = addition_problem(1, 20, torch.Generator().manual_seed(0))
vals, marks = (x[0, :, 0] + 0.5).numpy(), (x[0, :, 1] + 2 / 20).round().numpy()
print("   values :", " ".join(f"{v:.2f}" for v in vals))
print("   markers:", " ".join(f"{int(m):4d}" for m in marks))
print(f"   target = sum of the two marked values = {float(y[0, 0]) + 1:.3f}  (the net must remember across time)")
