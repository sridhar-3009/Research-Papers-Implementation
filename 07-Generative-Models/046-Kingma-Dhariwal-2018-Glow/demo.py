"""A ~6-second tour of Glow.

  1. Change of variables, layer by layer: each layer's cheap log-determinant equals the full Jacobian's
  2. Actnorm's data-dependent initialisation
  3. The invertible 1x1 convolution: a learned generalisation of a permutation; LU form makes log|det| O(c)
  4. Train a small flow on 'two moons' hidden in 6-D: exact log-likelihood, exact inverse
  5. Temperature (Section 6): shrinking the latent spread trades diversity for 'typical' samples
  6. Latent arithmetic (Figure 6): move a point along z_pos - z_neg and decode
"""

import math
import time

import torch

torch.set_num_threads(1)                                                          # tiny tensors: 1 thread is fastest
from glow import ActNorm, Glow, InvConv1x1                                       # noqa: E402

torch.manual_seed(0)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. log p(x) = log p(z) + sum of log-dets: cheap per-layer formulas vs the full Jacobian (1 x 4 x 4 input)")
g = Glow(C=1, K=2, L=2, hidden=8)
x = torch.randn(4, 1, 4, 4)
g.log_prob(x)                                                                    # initialise actnorm
with torch.no_grad():
    for p in g.parameters():
        p.add_(torch.randn_like(p) * 0.1)
zs, logdet, logpz = g.encode(x[:1])
flat = lambda v: torch.cat([z.flatten(1) for z in g.encode(v)[0]], 1)
J = torch.autograd.functional.jacobian(lambda v: flat(v.view(1, 1, 4, 4))[0], x[:1].flatten())
print(f"  sum of per-layer log-dets: {logdet.item():.6f}   log|det| of the full 16x16 Jacobian: "
      f"{torch.linalg.slogdet(J)[1].item():.6f}")
print(f"  reconstruction error |decode(encode(x)) - x|: {(g.decode(zs) - x[:1]).abs().max().item():.1e}")
print("  -> every layer is chosen so its Jacobian is triangular or block-structured (Eq. 8): the expensive determinant")
print("     is never computed, yet the likelihood is EXACT, not a bound (unlike a VAE).")

# --------------------------------------------------------------------------------------------- 2
section("2. Actnorm: initialised from the first minibatch")
a = ActNorm(3)
batch = torch.randn(256, 3, 4, 4) * torch.tensor([5.0, 0.2, 1.0]).view(1, 3, 1, 1) + torch.tensor([3.0, -1.0, 0.0]).view(1, 3, 1, 1)
y, _ = a(batch)
print(f"  input  per-channel mean {batch.mean((0, 2, 3)).numpy().round(2)}, std {batch.std((0, 2, 3)).numpy().round(2)}")
print(f"  output per-channel mean {y.mean((0, 2, 3)).detach().numpy().round(3)}, std {y.std((0, 2, 3)).detach().numpy().round(3)}")
print("  -> like batch norm at step 0, but afterwards plain trainable parameters: works with batch size 1 per GPU.")

# --------------------------------------------------------------------------------------------- 3
section("3. Invertible 1x1 convolution")
P = torch.eye(4)[[3, 2, 1, 0]]
print(f"  the channel reversal used by RealNVP is the 1x1 conv with W = a permutation matrix: det = {torch.det(P):+.0f}")
c = InvConv1x1(4)
print(f"  random-rotation init: log|det W| = {torch.linalg.slogdet(c.W)[1].item():+.1e} (volume preserving at the start)")
for C in (64, 512):
    W = torch.linalg.qr(torch.randn(C, C))[0]
    t = time.time()
    for _ in range(20):
        torch.linalg.slogdet(W)
    t_full = (time.time() - t) / 20
    lu = InvConv1x1(C, lu=True)
    t = time.time()
    for _ in range(20):
        lu.log_s.sum()
    t_lu = (time.time() - t) / 20
    print(f"  c = {C:3d}: log|det W| via slogdet {1e3 * t_full:.3f} ms (O(c^3)) vs LU form sum(log|s|) {1e3 * t_lu:.4f} ms (O(c))")
print("  -> a learned mixing of channels instead of a fixed shuffle; the paper measured lower NLL and only ~7% more")
print("     wall-clock time. With LU, even the log-determinant is free.")

# --------------------------------------------------------------------------------------------- 4
section("4. Train a small flow: 'two moons' rotated into 6-D (4 near-flat noise directions), K = 4 steps")
D = 6
R = torch.linalg.qr(torch.randn(D, D))[0]


def moons(n, which=None):
    t = torch.rand(n) * math.pi
    k = torch.randint(0, 2, (n,)).float() if which is None else torch.full((n,), float(which))
    xy = torch.stack([torch.where(k > 0, 1 - torch.cos(t), torch.cos(t)),
                      torch.where(k > 0, 0.5 - torch.sin(t), torch.sin(t))], 1) + 0.05 * torch.randn(n, 2)
    return (torch.cat([xy, 0.05 * torch.randn(n, D - 2)], 1) @ R.T).view(n, D, 1, 1)


def to_plane(v):
    return (v.view(-1, D) @ R)[:, :2]                                            # undo the rotation, keep the moons


torch.manual_seed(0)
flow = Glow(C=D, K=4, L=1, hidden=32, perm="conv", squeeze=False)
flow.log_prob(moons(512))
opt = torch.optim.Adam(flow.parameters(), 2e-3)
test = moons(4000)
for step in range(1, 151):
    loss = -flow.log_prob(moons(256)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 50, 150):
        with torch.no_grad():
            print(f"  step {step:3d}: test NLL {-flow.log_prob(test).mean().item():7.3f} nats per point")
with torch.no_grad():
    zs, _, _ = flow.encode(test[:500])
    rec = flow.decode(zs)
print(f"  max reconstruction error over 500 test points: {(rec - test[:500]).abs().max().item():.1e}")
print("  (negative NLL is fine: the data are thin in 4 of the 6 directions, so the density there is > 1)")

# --------------------------------------------------------------------------------------------- 5
section("5. Temperature: sample with the latent standard deviation scaled by T")
shape = flow.top_shape(D, 1, 1)
with torch.no_grad():
    for T in (0.0, 0.5, 0.7, 1.0):
        s = flow.decode(n=2000, shape=shape, temperature=T)
        p2 = to_plane(s)
        print(f"  T = {T:.1f}: spread of samples in the moon plane {p2.std(0).mean().item():.3f}, "
              f"mean log p(sample) {flow.log_prob(s).mean().item():8.2f}")
print("  -> lower T concentrates samples in higher-density regions (T = 0 decodes the single point z = 0). Scaling the")
print("     latent spread gives exactly the distribution proportional to p(x)^(1/T^2) only for additive coupling (the")
print("     paper's remark); with affine coupling, as here, it is an approximation. The paper uses T = 0.7 for faces.")

# --------------------------------------------------------------------------------------------- 6
section("6. Latent arithmetic (Figure 6): direction = mean z of moon 1 - mean z of moon 0")
with torch.no_grad():
    z_pos = flow.encode(moons(2000, 1))[0][0].mean(0)
    z_neg = flow.encode(moons(2000, 0))[0][0].mean(0)
    direction = z_pos - z_neg
    start = moons(5, 0)
    z = flow.encode(start)[0][0]
    for alpha in (0.0, 0.5, 1.0):
        out = to_plane(flow.decode([z + alpha * direction]))
        print(f"  alpha = {alpha:.1f}: mean point in the moon plane ({out[:, 0].mean():+.2f}, {out[:, 1].mean():+.2f})")
print("  -> moon 0 sits around (0, +0.64), moon 1 around (1, -0.14): walking along one latent direction (computed from")
print("     labels AFTER training) moves points from one moon toward the other, the toy version of 'add a smile'.")

print(f"\n(total {time.time() - T0:.1f} s)")
