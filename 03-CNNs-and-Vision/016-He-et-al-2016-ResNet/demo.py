"""A light tour of ResNet (He et al. 2016). No training; a few seconds.

    python3 demo.py
"""

import torch
import torch.nn as nn

from resnet import PAPER_GFLOPS, BasicBlock, CifarResNet, ImageNetResNet, count_macs, count_params

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. The residual block: y = relu(F(x) + x)")
blk = BasicBlock(16, 16).eval()
x = torch.relu(torch.randn(1, 16, 8, 8))
with torch.no_grad():
    print(f"normal block: output differs from input by {(blk(x) - x).abs().mean():.3f} on average")
    nn.init.zeros_(blk.bn2.weight); nn.init.zeros_(blk.bn2.bias)       # make F(x) = 0
    print(f"F(x) = 0:     output differs from input by {(blk(x) - x).abs().mean():.3f}")
print("-> to become an identity, a residual block only has to push F to ZERO (easy: shrink weights).")
print("   A plain block would have to LEARN the identity through two convs and ReLUs (hard).")

line("2. Table 1: depth, cost, parameters (ImageNet, measured on 'meta' tensors: no memory used)")
for d in (18, 34, 50, 101, 152):
    with torch.device("meta"):
        net = ImageNetResNet(d)
    print(f"ResNet-{d:<3d}  {count_macs(net) / 1e9:5.2f} GFLOPs (paper {PAPER_GFLOPS[d]:4.1f})   {count_params(net) / 1e6:5.1f}M parameters")
print("VGG-16 for comparison: 15.3 GFLOPs, 138M parameters")
print("-> ResNet-152 is 8x deeper than VGG but cheaper: thin 3x3 layers, bottlenecks, no big FC layers.")

line("3. Gradient reaching the FIRST layer, plain vs residual (CIFAR nets at initialization)")
x, y = torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
table = {}
for n in (3, 9, 18):
    norms = {}
    for residual in (False, True):
        net = CifarResNet(n, residual=residual)
        loss = nn.functional.cross_entropy(net(x), y)
        loss.backward()
        norms[residual] = net.conv1.weight.grad.norm().item()
    table[n] = norms
    print(f"{6 * n + 2:3d} layers: plain {norms[False]:.4f}   residual {norms[True]:.4f}")
growth = lambda k, n: table[n][k] / table[3][k]
print(f"growth from 20 to 110 layers: plain x{growth(False, 18):,.0f}, residual x{growth(True, 18):.0f}")
print("-> the plain net's gradient does not vanish (the paper checked that): it EXPLODES with depth.")
print("   The residual net's grows only mildly.")
print("   (The paper only said the plain gradients had 'healthy norms' and guessed at 'exponentially")
print("   low convergence rates'. Yang et al. (2019) later proved that BatchNorm makes gradients")
print("   explode at initialization in deep plain nets; shortcuts tame this.)")

line("4. Shortcut options (ResNet-34)")
for opt, what in (("A", "zero-padding, no parameters"), ("B", "projection only where the shape changes"),
                  ("C", "projection on every block")):
    with torch.device("meta"):
        net = ImageNetResNet(34, option=opt)
    print(f"option {opt} ({what}): {count_params(net) / 1e6:.2f}M parameters")
print("paper top-1 error: A 25.03%, B 24.52%, C 24.19%: small differences, so projections")
print("are NOT what fixes degradation; the identity path is.")
