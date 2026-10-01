"""A light tour of LeNet-5 (LeCun et al. 1998). No training; runs in about a second.

    python3 demo.py
"""

import numpy as np
import torch

from lenet5 import C3_TABLE, LeNet5, conv2d_naive, map_loss, mse_loss, rbf_prototypes


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def show(a, chars=" .:#"):
    lo, hi = a.min(), a.max()
    for row in a:
        print("   " + "".join(chars[int(round((v - lo) / (hi - lo + 1e-9) * (len(chars) - 1)))] * 2 for v in row))


line("1. A convolution by hand: one 3x3 kernel slid over the whole image")
img = np.zeros((1, 10, 10))
img[0, 2:8, 4] = 1.0                    # a vertical stroke
img[0, 2, 4:8] = 1.0                    # with a bar on top
print("input image:")
show(img[0])
vertical = np.array([[[-1, 2, -1]] * 3], dtype=float)[None]      # responds to vertical lines
horizontal = vertical.transpose(0, 1, 3, 2)                        # responds to horizontal lines
out = conv2d_naive(img, np.concatenate([vertical, horizontal]), np.zeros(2))
print("feature map 1 (vertical-line kernel):")
show(out[0])
print("feature map 2 (horizontal-line kernel):")
show(out[1])
print("-> each map finds ONE kind of feature, everywhere in the image, with only 9 + 1 weights.")

line("2. Shift the input -> the feature map shifts the same way")
shifted = np.roll(img, (1, 2), axis=(1, 2))
out_s = conv2d_naive(shifted, vertical, np.zeros(1))
peak = lambda m: tuple(int(i) for i in np.unravel_index(m.argmax(), m.shape))
print("strongest response before and after shifting the image by (1, 2):", peak(out[0]), "->", peak(out_s[0]))
print("-> moved by exactly (1, 2). Subsampling then blurs the exact position away.")

line("3. Weight sharing: parameters vs connections (Section II.B)")
net = LeNet5()
rows = [("C1", 156, 122304), ("S2", 12, 5880), ("C3", 1516, 151600), ("S4", 32, 2000),
        ("C5", 48120, 48120), ("F6", 10164, 10164)]
for name, p, c in rows:
    print(f"{name}: {p:6d} parameters, {c:7d} connections  ({c / p:5.0f} connections per parameter)")
print(f"total trainable parameters in our model: {net.n_trainable()} (paper: 60,000)")
print(f"a fully connected layer from 32x32 to C1's 6x28x28 units would need {1024 * 6 * 28 * 28 + 6 * 28 * 28:,} weights")

line("4. Table I: which S2 maps each C3 map reads")
print("       " + " ".join(f"{i:2d}" for i in range(16)))
for s in range(6):
    print(f"S2 {s}:  " + " ".join(" X" if s in t else "  " for t in C3_TABLE))
print("-> different inputs force different C3 maps to learn different features (symmetry breaking).")

line("5. The output codes: 7x12 bitmaps of the digits (+1 = ink)")
P = rbf_prototypes().view(10, 12, 7)
for r in range(12):
    print("  " + "   ".join("".join("#" if P[d, r, c] > 0 else "." for c in range(7)) for d in range(10)))
print("-> output unit i computes the squared distance from F6's 84 numbers to bitmap i.")
print("   The prediction is the CLOSEST bitmap (smallest penalty).")

line("6. Why the MAP loss (Eq. 9) instead of plain MSE (Eq. 8)")
labels = torch.tensor([2])
collapsed = torch.zeros(1, 10)                          # every penalty 0: "all RBF centers equal"
good = torch.full((1, 10), 50.0); good[0, 2] = 1.0
for name, y in (("collapsed (ignores the input)", collapsed), ("correct class close, others far", good)):
    print(f"{name:34s} MSE {mse_loss(y, labels).item():6.3f}   MAP {map_loss(y, labels).item():6.3f}")
print("-> MSE rates the useless collapsed network as PERFECT (0). MAP also pushes the wrong")
print("   classes' penalties up, so it prefers the network that separates the classes.")
