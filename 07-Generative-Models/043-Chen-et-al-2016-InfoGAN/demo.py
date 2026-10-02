"""A ~10-second tour of InfoGAN.

  1. The variational bound on mutual information (Eqs. 4-5) on an exact discrete channel
  2. Figure 1 in miniature: L_I over training for InfoGAN vs a regular GAN with the same Q
  3. Disentangling: a 4-cluster 2-D dataset where each cluster is a short segment. Does the categorical code pick
     the cluster and the continuous code the position along the segment?
  4. Section 7.2's 'exaggerated' codes: push the continuous code to +-2, outside its training range
"""

import math
import time

import torch

from infogan import (LatentSpec, MLPDiscriminatorQ, MLPGenerator, cluster_accuracy, infogan_step,
                     mutual_information_discrete)

torch.manual_seed(0)
T0 = time.time()
CENTRES = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
DIRS = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])           # each cluster: a segment


def section(t):
    print(f"\n=== {t} ===")


def data(n):
    k, t = torch.randint(0, 4, (n,)), torch.rand(n) * 2 - 1
    return CENTRES[k] + 0.8 * t[:, None] * DIRS[k] + 0.05 * torch.randn(n, 2)


# --------------------------------------------------------------------------------------------- 1
section("1. I(c; x) >= E[log Q(c|x)] + H(c) on an exact channel (c uniform over 3 values, x over 5)")
torch.manual_seed(0)
Pxc = torch.rand(3, 5) ** 3
Pxc /= Pxc.sum(1, keepdim=True)
joint = Pxc / 3
I = mutual_information_discrete(joint).item()
post = joint / joint.sum(0, keepdim=True)
for name, Q in (("Q = true posterior P(c|x)", post), ("Q = posterior blurred 50% toward uniform", 0.5 * post + 0.5 / 3),
                ("Q = uniform (knows nothing)", torch.full((3, 5), 1 / 3))):
    print(f"  {name:>42}: L_I = {(joint * Q.log()).sum().item() + math.log(3):.4f}")
print(f"  {'true mutual information I(c; x)':>42}: {I:.4f}   (H(c) = log 3 = {math.log(3):.4f})")
print("  -> L_I never exceeds I, equals it when Q is the true posterior, and is 0 when Q is clueless. Training Q")
print("    tightens the bound; training G raises I itself.")

# --------------------------------------------------------------------------------------------- 2-3
section("2. Figure 1 in miniature: categorical part of L_I during training (H(c) = log 4 = 1.386 is the maximum)")
spec = LatentSpec(4, (4,), 1)
models = {}
curves = {}
for info in (True, False):
    torch.manual_seed(0)
    G, DQ = MLPGenerator(spec.dim), MLPDiscriminatorQ(2, 4 + 2)
    og = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))              # Appendix C: lr 1e-3 for G,
    od = torch.optim.Adam(DQ.parameters(), 2e-4, betas=(0.5, 0.999))             # 2e-4 for D/Q
    curves[info] = []
    for step in range(1, 2401):
        infogan_step(G, DQ, og, od, data(256), spec, info=info)
        if step in (50, 300, 1200, 2400):
            with torch.no_grad():
                z, cat, cont = spec.sample(2000)
                _, q = DQ(G(spec.pack(z, cat, cont)))
                l_cat = -torch.nn.functional.cross_entropy(q[:, :4], cat[:, 0]).item() + math.log(4)
            curves[info].append(l_cat)
    models[info] = G
print(f"  {'iteration':>12} {'50':>7} {'300':>7} {'1200':>7} {'2400':>7}")
for info in (True, False):
    print(f"  {'InfoGAN' if info else 'regular GAN':>12} " + " ".join(f"{v:7.3f}" for v in curves[info]))
print("  -> InfoGAN pushes L_I to ~H(c) quickly (the bound is tight: x reveals c); in the regular GAN the same Q finds")
print("     almost nothing, because nothing forces G to make its output depend on c.")

section("3. What did the codes learn? (2000 generated points per model)")
for info in (True, False):
    G = models[info]
    with torch.no_grad():
        z, cat, cont = spec.sample(2000)
        x = G(spec.pack(z, cat, cont))
    cluster = torch.cdist(x, CENTRES).argmin(1)
    acc = cluster_accuracy(cluster, cat[:, 0], 4)
    along = ((x - CENTRES[cluster]) * DIRS[cluster]).sum(1)                        # position along the segment
    corr = torch.stack([torch.corrcoef(torch.stack([cont[cluster == k, 0], along[cluster == k]]))[0, 1]
                        for k in range(4) if (cluster == k).sum() > 10]).abs().mean().item()
    on_data = (torch.cdist(x, CENTRES).min(1).values < 1.0).float().mean().item()
    print(f"  {'InfoGAN' if info else 'regular GAN':>12}: categorical code -> cluster accuracy {acc:.3f}, "
          f"|corr(continuous code, position on segment)| = {corr:.3f}, {100 * on_data:.0f}% of points on a cluster")
print("  -> both models generate the clusters (sample quality fluctuates during GAN training; see the % above), but only")
print("     InfoGAN's codes MEAN something: its c1 is a cluster index (cf. the paper's MNIST c1: 5% error as a digit")
print("     classifier) and its continuous code is the position along each segment. The regular GAN's c1 is near")
print("     chance (0.25).")

# --------------------------------------------------------------------------------------------- 4
section("4. Exaggerated continuous code (trained on [-1, 1]); cluster 0, fixed noise")
G = models[True]
z, _, _ = spec.sample(1)
row = []
for v in (-2.0, -1.0, 0.0, 1.0, 2.0):
    with torch.no_grad():
        x = G(spec.pack(z, torch.tensor([[0]]), torch.tensor([[v]])))
    row.append(f"c2={v:+.0f}: ({x[0, 0]:+.2f}, {x[0, 1]:+.2f})")
print("  " + "   ".join(row))
print("  -> the code keeps moving the point in a consistent direction past the trained range (the paper's -2..2 plots);")
print("     how far it extrapolates sensibly depends on the network, so look at the numbers rather than assume.")

print(f"\n(total {time.time() - T0:.1f} s)")
