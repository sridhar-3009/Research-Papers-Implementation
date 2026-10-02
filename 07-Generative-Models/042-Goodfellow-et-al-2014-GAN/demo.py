"""A ~10-second tour of Generative Adversarial Nets.

  1. The theory on a grid: D* = p_data / (p_data + p_g) and C(G) = -log 4 + 2 JSD (Proposition 1, Theorem 1)
  2. Figure 1 in 1-D: a GAN learns N(3, 0.5^2) from uniform noise; D(x) drifts to ~1/2 on the data
  3. Saturation (Section 3): with a confident D, log(1 - D(G(z))) gives far weaker generator gradients
  4. 'The Helvetica scenario' (Section 6): train G without updating D and it collapses onto one mode
  5. Parzen-window evaluation (Table 1): how sigma is chosen, and why the number can mislead
"""

import math
import time

import torch

from gan import (Discriminator, Generator, d_loss, g_loss, jsd, optimal_discriminator, parzen_log_likelihood,
                 select_sigma, train_step, virtual_criterion)

torch.manual_seed(0)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


def adam(m, lr=1e-3):
    return torch.optim.Adam(m.parameters(), lr, betas=(0.5, 0.999))


# --------------------------------------------------------------------------------------------- 1
section("1. Theory on a grid: data N(0, 1) vs generator N(mu, 1)")
x = torch.linspace(-8, 8, 4001)
dx = (x[1] - x[0]).item()
pdf = lambda m, s: torch.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
p_data = pdf(0, 1)
print(f"  {'p_g':>11} {'D*(0)':>6} {'D*(1)':>6} {'C(G)':>8} {'-log4 + 2 JSD':>14}")
for mu in (3.0, 1.0, 0.3, 0.0):
    p_g = pdf(mu, 1)
    D = optimal_discriminator(p_data, p_g)
    i0, i1 = 2000, 2250                                                          # x = 0 and x = 1
    print(f"  {f'N({mu}, 1)':>11} {D[i0]:6.3f} {D[i1]:6.3f} {virtual_criterion(p_data, p_g, dx):8.4f} "
          f"{-math.log(4) + 2 * jsd(p_data, p_g, dx):14.4f}")
print(f"  -> C(G) always equals -log 4 + 2 JSD and bottoms out at -log 4 = {-math.log(4):.4f} exactly when p_g = p_data,")
print("     where the best possible discriminator can only say 1/2 everywhere.")

# --------------------------------------------------------------------------------------------- 2
section("2. Figure 1 in 1-D: data N(3, 0.5^2), z ~ U(-1, 1), G and D small MLPs, k = 1")
torch.manual_seed(0)
G = Generator(1, (32, 32), 1, out="linear")
D = Discriminator(1, (64, 64), pieces=3, dropout=0.0, input_dropout=0.0)
with torch.no_grad():
    G.net[-1].bias.fill_(-3.0)                                                   # start far from the data
og, od = adam(G), adam(D)
real = lambda m: 3 + 0.5 * torch.randn(m, 1)
grid = torch.tensor([[2.0], [2.5], [3.0], [3.5], [4.0]])
print(f"  {'step':>5} {'G mean':>7} {'G std':>6}   D(x) at x = 2, 2.5, 3, 3.5, 4")
for step in range(1, 2001):
    train_step(G, D, og, od, real, m=256)
    if step in (1, 200, 1000, 2000):
        with torch.no_grad():
            s = G(G.sample_z(5000))
            d = torch.sigmoid(D(grid)).squeeze(-1)
        print(f"  {step:5d} {s.mean():7.3f} {s.std():6.3f}   " + " ".join(f"{v:.2f}" for v in d))
print("  -> the generated distribution moves onto the data (target mean 3, std 0.5) and D(x) flattens toward 1/2,")
print("     panel (d) of Figure 1. But it does not settle: the spread swings (0.56 at step 1000, 0.31 at 2000) because")
print("     G and D keep chasing each other. Simultaneous gradient steps on a minimax game need not converge.")
print("     (Adam with beta1 = 0.5 here; the paper used SGD with momentum.)")

# --------------------------------------------------------------------------------------------- 3
section("3. Saturation: G starts at mean -6, D pre-trained to reject it; size of the first generator gradient")
torch.manual_seed(1)
G = Generator(1, (32, 32), 1, out="linear")
D = Discriminator(1, (64, 64), pieces=3, dropout=0.0, input_dropout=0.0)
with torch.no_grad():
    G.net[-1].bias.fill_(-6.0)
od = adam(D)
for _ in range(300):
    with torch.no_grad():
        f = G(G.sample_z(256))
    loss = d_loss(D, real(256), f)
    od.zero_grad(); loss.backward(); od.step()
z = G.sample_z(256)
with torch.no_grad():
    dfake = torch.sigmoid(D(G(z))).mean().item()
norms = {}
for sat in (True, False):
    G.zero_grad()
    g_loss(D, G(z), saturating=sat).backward()
    norms[sat] = torch.sqrt(sum((p.grad ** 2).sum() for p in G.parameters())).item()
print(f"  D(G(z)) = {dfake:.4f} on average (D is confident)")
print(f"  minimax loss       log(1 - D(G(z))):  |grad| = {norms[True]:.2e}")
print(f"  non-saturating     -log D(G(z)):      |grad| = {norms[False]:.2e}   ({norms[False] / norms[True]:.0f}x larger)")
print("  -> d/dlogit log(1 - D) = -D, which vanishes when D ~ 0; d/dlogit (-log D) = -(1 - D) ~ -1. Same fixed point,")
print("     but the second actually moves G early in training.")

# --------------------------------------------------------------------------------------------- 4
section("4. The Helvetica scenario: 8 Gaussians on a ring")


def ring(m):
    a = 2 * math.pi * torch.randint(0, 8, (m,)) / 8
    return torch.stack([2 * a.cos(), 2 * a.sin()], 1) + 0.05 * torch.randn(m, 2)


def coverage(G):
    with torch.no_grad():
        s = G(G.sample_z(2000))
    mode = ((torch.atan2(s[:, 1], s[:, 0]) / (2 * math.pi / 8)).round() % 8).long()
    centre = torch.stack([2 * torch.cos(2 * math.pi * mode / 8), 2 * torch.sin(2 * math.pi * mode / 8)], 1)
    good = (s - centre).norm(dim=1) < 0.3
    counts = torch.bincount(mode[good], minlength=8)
    return (counts > 40).sum().item(), good.float().mean().item(), counts.tolist()


torch.manual_seed(0)
G = Generator(2, (64, 64), 2, out="linear")
D = Discriminator(2, (64, 64), pieces=3, dropout=0.0, input_dropout=0.0)
og, od = adam(G, 2e-3), adam(D, 2e-3)
for _ in range(1500):
    train_step(G, D, og, od, ring, m=256)
n, q, c = coverage(G)
print(f"  alternating (k = 1), 1500 steps: {n}/8 modes covered, {100 * q:.0f}% of samples near a mode, counts {c}")
sgd = torch.optim.SGD(G.parameters(), 0.05)
for step in range(1, 501):
    loss = g_loss(D, G(G.sample_z(256)))
    sgd.zero_grad(); loss.backward(); sgd.step()
    if step in (100, 500):
        n, q, c = coverage(G)
        print(f"  + {step:3d} generator-only steps (D frozen): {n}/8 modes, counts {c}")
print("  -> with D out of date, G simply moves every z to wherever the stale D says 'real': many z, one x.")
print("     This is why Algorithm 1 keeps D in step with G (k discriminator steps per generator step).")

# --------------------------------------------------------------------------------------------- 5
section("5. Parzen-window log-likelihood (the paper's Table 1 metric), on 2-D data N(0, I)")
torch.manual_seed(0)
samples, valid, test = torch.randn(1000, 2), torch.randn(500, 2), torch.randn(1000, 2)
best, scores = select_sigma(samples, valid, [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
exact = (-0.5 * (2 * math.log(2 * math.pi) + (test ** 2).sum(1))).mean().item()
print("  sigma -> validation log-lik: " + ", ".join(f"{s}: {v:.2f}" for s, v in scores.items()))
print(f"  chosen sigma = {best}; test estimate {parzen_log_likelihood(samples, test, best).mean():.3f} vs exact "
      f"log-likelihood {exact:.3f}")
bad = torch.randn(1000, 2) * 0.3                                                 # a generator too concentrated
s_bad = select_sigma(bad, valid, list(scores))[0]
print(f"  a too-narrow 'generator' (std 0.3 instead of 1): chosen sigma = {s_bad}, test estimate "
      f"{parzen_log_likelihood(bad, test, s_bad).mean():.3f}")
print("  -> close to the truth for good samples, but the wrong generator scores about as well: cross-validation widens")
print("     sigma until the kernels cover for the missing spread. The metric measures the samples PLUS a tuned blur,")
print("     and in 784-D it is far worse; the paper itself warns of its high variance. It was the best tool in 2014.")

print(f"\n(total {time.time() - T0:.1f} s)")
