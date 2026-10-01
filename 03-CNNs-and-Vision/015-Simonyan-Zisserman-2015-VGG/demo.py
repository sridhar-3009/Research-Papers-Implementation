"""A light tour of VGG (Simonyan & Zisserman 2015). No training; a couple of seconds.

    python3 demo.py
"""

import torch

from vgg import CONFIGS, VGG, count_params, receptive_field, to_fully_convolutional, weight_layers

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Table 1: six networks, one design, more and more depth")
for name, cfg in CONFIGS.items():
    pretty = " ".join("|" if v == "M" else ("LRN" if v == "LRN" else (f"1x1-{v[1]}" if isinstance(v, tuple) else str(v))) for v in cfg)
    print(f"{name:6s} {weight_layers(name):2d} layers  {count_params(name) / 1e6:6.1f}M params   {pretty}")
print("('|' = 2x2 max-pool. Every number is a 3x3 conv with that many channels.)")
print("-> 8 more conv layers (A -> E) add only 11M parameters: most of the 133M+ sit in FC-4096.")

line("2. Why stacks of 3x3 filters? (Section 2.3)")
C = 512
for n in (1, 2, 3):
    print(f"{n} stacked 3x3 conv(s): sees {receptive_field(n)}x{receptive_field(n)}, "
          f"{n * 9 * C * C / 1e6:5.2f}M weights, {n} ReLU(s)")
for k in (5, 7):
    print(f"one {k}x{k} conv:          sees {k}x{k}, {k * k * C * C / 1e6:5.2f}M weights, 1 ReLU")
print("-> three 3x3 layers see as much as one 7x7, with 27C^2 instead of 49C^2 weights")
print("   and three non-linearities instead of one.")

line("3. Initialising a deep net (Section 3.1): activation size through 16 conv layers of E")
x = torch.randn(4, 3, 32, 32)
for init in ("paper", "paper001", "glorot"):
    net = VGG("E", init=init, num_classes=10, input_size=32, fc=64).eval()   # full-width convs
    h, sizes = x, []
    with torch.no_grad():
        for m in net.features:
            h = m(h)
            if isinstance(m, torch.nn.ReLU):
                sizes.append(h.std().item())
    print(f"init {init:9s} std after conv 1, 4, 8, 12, 16: " + "  ".join(f"{sizes[i]:9.2e}" for i in (0, 3, 7, 11, 15)))
print("-> std 0.1 ('10^-2 variance', as written) EXPLODES (x10^7); std 0.01 VANISHES (x10^-9).")
print("   Glorot is far better but still shrinks ~1000x: every ReLU zeroes half its inputs, which")
print("   Glorot's formula (made for tanh) ignores. He et al. (2015) fix it with an extra factor 2.")
print("   This is why the paper first trained the shallow net A and copied its layers into D and E.")

line("4. Dense evaluation: the fc layers become convolutions (Section 3.2)")
net = VGG("A", num_classes=10, input_size=64, fc=32, width_div=16).eval()
fcn = to_fully_convolutional(net).eval()
with torch.no_grad():
    for size in ((64, 64), (96, 128), (160, 160)):
        out = fcn(torch.randn(1, 3, *size))
        print(f"image {size[0]}x{size[1]} -> class score map {tuple(out.shape[1:])}")
    x = torch.randn(2, 3, 64, 64)
    print("on the training size, FCN output == original output:", torch.allclose(fcn(x)[:, :, 0, 0], net(x), atol=1e-5))
print("-> one pass over the whole image scores every position; average the map to classify.")
