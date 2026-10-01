"""A light tour of the dropout journal paper (Srivastava et al., 2014). Runs in seconds.
Run:  python3 demo.py
(For the MNIST experiments, run experiments.py on a strong machine.)
"""

import numpy as np
import torch

from dropout import Net, dropout_linear_regression_closed_form


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


section("1. Bernoulli dropout: multiply each unit by r ~ Bernoulli(p)  (Section 4)")
torch.manual_seed(0)
h = torch.tensor([[2.0, 0.5, 1.0, 3.0, 0.0, 1.5]])
for k in range(3):
    r = (torch.rand_like(h) < 0.5).float()
    print(f"   sample {k + 1}: r = {r.int().tolist()[0]}   ->  thinned outputs {(h * r).tolist()[0]}")
print(f"   at test time: multiply by p instead  ->  {(h * 0.5).tolist()[0]}  (the average of all samples)")

section("2. Gaussian dropout: multiply by r ~ N(1, (1-p)/p)  (Section 10)")
r = 1 + np.sqrt((1 - 0.5) / 0.5) * torch.randn(1, 6)
print(f"   r = {r.numpy().round(2)[0]}   (mean 1, so NO rescaling needed at test time)")

section("3. Dropout in linear regression = ridge regression  (Section 9.1)")
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)) * np.array([1.0, 3.0, 0.3])
y = X @ np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.2, 200)
print("   true weights: [1, 1, 1];  input spreads: [1.0, 3.0, 0.3]")
for p in (1.0, 0.8, 0.5, 0.2):
    w = p * dropout_linear_regression_closed_form(X, y, p) if p < 1 else np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"   p = {p}: effective weights (p * w) = {np.round(w, 3)}")
print("   Smaller p = stronger shrinkage: each effective weight is pulled toward p * (true weight).")
print("   The penalty p(1-p)||Gamma w||^2 grows with each input's spread, which exactly cancels the")
print("   spread's effect on the fit: with uncorrelated inputs EVERY weight shrinks by the same factor p")
print("   (plain ridge would squeeze the low-spread column 3 hardest). Column 3 strays here only because")
print("   with 200 samples it happens to correlate with the big column 2; with 200,000 samples all three")
print("   come out at 0.50 for p = 0.5.")

section("4. A dropout net in one line")
net = Net([784, 1024, 1024, 2048, 10], p_input=0.8, p_hidden=0.5, act="relu")
n_params = sum(p.numel() for p in net.parameters())
print(f"   Net([784, 1024, 1024, 2048, 10], p_input=0.8, p_hidden=0.5): {n_params:,} weights")
print("   Trained with max-norm (||w|| <= c per unit), momentum 0.95 - see experiments.py.")
