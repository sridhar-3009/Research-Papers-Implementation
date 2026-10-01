"""A light tour of Szegedy et al. (2013). No dataset and no real training; a couple of seconds.

    python3 demo.py
"""

import torch
import torch.nn as nn

from adversarial import (conv_operator_norm_fft, distortion, fc_operator_norm, gaussian_distort, minimal_adversarial,
                         predict)

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. The paper's attack on a toy classifier (Section 4.1)")
m = nn.Linear(2, 2)
with torch.no_grad():
    m.weight.copy_(torch.tensor([[20.0, 0.0], [-20.0, 0.0]])); m.bias.copy_(torch.tensor([-10.0, 10.0]))
x = torch.tensor([0.3, 0.5])
z, c = minimal_adversarial(m, x, target=0)
print(f"x = {x.tolist()} is class {predict(m, x)}; the boundary is the line x0 = 0.5")
print(f"minimal adversarial point: {[round(v, 4) for v in z.tolist()]} (class {predict(m, z)}), found with c = {c:.3g}")
print(f"||r|| = {(z - x).norm():.4f}; the exact answer is the distance to the boundary, 0.2")
print("-> L-BFGS + a search over c finds (almost) the closest point of the target class.")

line("2. Why tiny changes work in high dimensions")
print("A random linear 'classifier' score s = w.x, |w_i| ~ 1. Change every pixel by eps = 0.01,")
print("each in the direction of sign(w_i): the score moves by eps * sum|w_i|.")
for n in (10, 100, 784, 150528):
    w = torch.randn(n)
    typical = (w @ (torch.rand(n) - 0.5)).abs()                        # the score of a random centred image
    print(f"  {n:6d} inputs: per-pixel change 0.01 moves the score by {0.01 * w.abs().sum():8.2f}   (a typical score: {typical:7.2f})")
print("-> the same imperceptible per-pixel change adds up over many pixels.")
print("   (Goodfellow et al. 2015, the next paper in this line, made this 'linear' explanation precise.)")

line("3. Random noise vs a chosen perturbation of the same size")
n = 784
w = torch.randn(10, n) * 0.1
clf = nn.Linear(n, 10, bias=False)
with torch.no_grad():
    clf.weight.copy_(w)
img = torch.rand(n)
label = predict(clf, img)
g = torch.Generator().manual_seed(0)
z, _ = minimal_adversarial(clf, img, target=(label + 1) % 10)
d = distortion(img, z)
flips = sum(predict(clf, gaussian_distort(img, d, g)) != label for _ in range(200))
print(f"chosen perturbation: stddev {d:.4f}, label {label} -> {predict(clf, z)}")
print(f"Gaussian noise of the SAME stddev {d:.4f}: the label changed in {flips} of 200 tries")
print("-> at equal size, a random direction rarely crosses the boundary; the attack picks the one that does.")

line("4. Section 4.3: how much can each layer amplify a perturbation?")
W = torch.randn(256, 784) * 0.05
print(f"fully connected 784->256 (random): operator norm {fc_operator_norm(W):.2f}")
k = torch.randn(32, 16, 3, 3) * 0.1
print(f"3x3 conv 16->32 on 28x28 (random, Fourier formula): operator norm {conv_operator_norm_fft(k, 28):.2f}")
print("-> ||phi(x) - phi(x + r)|| <= (product of the layer norms) * ||r||. If each norm is above 1")
print("   (AlexNet's: 2.75, 10, 7, 7.5, 11, 3.12, 4, 4), a small r can grow a lot. The bound is loose,")
print("   but it shows instability can start in the very first layer.")
