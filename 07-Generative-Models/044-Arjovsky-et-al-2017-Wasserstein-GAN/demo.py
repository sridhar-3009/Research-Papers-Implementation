"""A ~8-second tour of Wasserstein GAN.

  1. Example 1 (parallel lines): W = |theta| shrinks smoothly; JS, KL and TV jump
  2. Kantorovich-Rubinstein (Eq. 1 = Eq. 2): the transport LP and the 1-Lipschitz-function LP give the same number
  3. Figures 1-2: train a GAN discriminator and a WGAN critic at fixed theta; estimates and gradients w.r.t. theta
  4. WGAN learning Example 1: theta goes to 0
  5. A meaningful loss (Section 4.2)? The critic's estimate vs the true W during training
  6. Weight clipping: what it does to the Lipschitz constant (and to the scale of the estimate)
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgan import (Critic, clip_weights, critic_objective, example1_distances, lipschitz_bound, max_grad_norm,
                  wasserstein_1d_samples, wasserstein_dual_lp, wasserstein_lp, wgan_step)

torch.manual_seed(0)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


real_lines = lambda m: torch.stack([torch.zeros(m), torch.rand(m)], 1)            # P_0 = (0, Z)

# --------------------------------------------------------------------------------------------- 1
section("1. Example 1: P_0 = (0, Z) vs P_theta = (theta, Z), Z ~ U[0, 1]")
print(f"  {'theta':>7} {'W (EM)':>7} {'JS':>6} {'KL':>5} {'TV':>4}")
for theta in (1.0, 0.5, 0.1, 0.01, 0.0):
    d = example1_distances(theta)
    print(f"  {theta:7.2f} {d['W']:7.2f} {d['JS']:6.3f} {d['KL']:>5} {d['TV']:4.0f}")
print("  -> as theta -> 0 only W goes to 0 smoothly. JS sits at log 2 and KL at infinity until the lines coincide:")
print("     no gradient tells the generator which way to move. Low-dimensional supports (images!) behave like this.")

# --------------------------------------------------------------------------------------------- 2
section("2. Primal (cheapest transport plan, Eq. 1) = dual (best 1-Lipschitz f, Eq. 2), 9-point grid")
torch.manual_seed(0)
x = torch.linspace(0, 4, 9).double()
p, q = torch.rand(9).double(), torch.rand(9).double()
p, q = p / p.sum(), q / q.sum()
wp, plan = wasserstein_lp(p, q, x)
wd, f = wasserstein_dual_lp(p, q, x)
print(f"  primal W = {wp:.6f}   dual W = {wd:.6f}")
print("  optimal f on the grid: " + " ".join(f"{v:+.2f}" for v in (f - f[0])))
print(f"  largest slope |f_i+1 - f_i| / dx = {((f[1:] - f[:-1]).abs() / (x[1] - x[0])).max():.3f} (must be <= 1)")
print("  -> the critic's job: find the 1-Lipschitz function that best separates the two distributions on average.")

# --------------------------------------------------------------------------------------------- 3
section("3. Figures 1-2: GAN discriminator vs WGAN critic, each trained 300 steps at a fixed theta")
print(f"  {'theta':>6} {'JS estimate':>12} {'|d G-loss/d theta|':>19} {'W estimate':>11} {'|d G-loss/d theta|':>19}")
for theta in (2.0, 1.0, 0.5, 0.2, 0.05):
    torch.manual_seed(0)
    th = torch.tensor(theta, requires_grad=True)
    fake = lambda m: torch.stack([th.detach().expand(m), torch.rand(m)], 1)
    D = Critic(2, (32, 32))
    od = torch.optim.Adam(D.parameters(), 1e-3)
    for _ in range(300):
        loss = -(F.logsigmoid(D(real_lines(128))).mean() + F.logsigmoid(-D(fake(128))).mean())
        od.zero_grad(); loss.backward(); od.step()
    js_est = 0.5 * (-loss.item()) + math.log(2)                                  # Section 4.2's lower bound of JS
    z = torch.rand(512)
    gj, = torch.autograd.grad(F.logsigmoid(-D(torch.stack([th.expand(512), z], 1))).mean(), th)
    fc = Critic(2, (32, 32))
    of = torch.optim.RMSprop(fc.parameters(), 5e-3)
    for _ in range(300):
        obj = critic_objective(fc, real_lines(128), fake(128))
        of.zero_grad(); (-obj).backward(); of.step(); clip_weights(fc, 0.1)
    gw, = torch.autograd.grad(-fc(torch.stack([th.expand(512), z], 1)).mean(), th)
    print(f"  {theta:6.2f} {js_est:12.3f} {abs(gj.item()):19.2e} {obj.item():11.4f} {abs(gw.item()):19.2e}")
print("  -> the JS estimate is stuck near log 2 = 0.693 for every theta >= 0.2: the discriminator separates the lines")
print("     perfectly and its number says nothing about HOW far apart they are; its gradient swings ~1000x (tiny far")
print("     away, huge near 0). The W estimate grows ~linearly with theta (up to the clipping scale) and its gradient")
print("     stays roughly steady, the clean, linear critic of Figure 2.")

# --------------------------------------------------------------------------------------------- 4
section("4. WGAN learning Example 1 (Algorithm 1: n_critic = 5, clip c = 0.01, RMSProp)")


class Line(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(1.0))

    def forward(self, z):
        return torch.stack([self.theta.expand_as(z), z], 1)


torch.manual_seed(0)
G, fc = Line(), Critic(2, (64, 64))
og, of = torch.optim.RMSprop(G.parameters(), 5e-3), torch.optim.RMSprop(fc.parameters(), 5e-4)
traj = []
for step in range(1, 301):
    wgan_step(G, fc, og, of, real_lines, lambda m: torch.rand(m), m=64, n_critic=5, clip=0.01)
    if step in (1, 25, 50, 100, 300):
        traj.append(f"step {step}: theta = {G.theta.item():+.3f}")
print("  " + "   ".join(traj))
print("  -> the generator walks the line onto the data even though the two supports never overlap on the way.")

# --------------------------------------------------------------------------------------------- 5
section("5. A meaningful loss: critic estimate vs the true W1 while a WGAN learns a 1-D two-bump distribution")
torch.manual_seed(0)
real_bumps = lambda m: 3 + (torch.randint(0, 2, (m, 1)).float() * 2 - 1) + 0.3 * torch.randn(m, 1)
G = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
fc = Critic(1, (64, 64))
og, of = torch.optim.RMSprop(G.parameters(), 5e-4), torch.optim.RMSprop(fc.parameters(), 1e-3)
est, true = [], []
for step in range(1, 1501):
    wgan_step(G, fc, og, of, real_bumps, lambda m: torch.randn(m, 4), m=128, n_critic=5, clip=0.1)
    if step % 50 == 0:
        with torch.no_grad():
            fake = G(torch.randn(4000, 4))
            est.append(critic_objective(fc, real_bumps(4000), fake).item())
            true.append(wasserstein_1d_samples(real_bumps(4000)[:, 0], fake[:, 0]).item())
for k in (0, 5, 11, 17, 23, 29):
    print(f"  step {50 * (k + 1):5d}: critic estimate {est[k]:.4f}   true W1 {true[k]:.3f}")
r = torch.corrcoef(torch.tensor([est, true]))[0, 1].item()
r_late = torch.corrcoef(torch.tensor([est[4:], true[4:]]))[0, 1].item()
print(f"  -> correlation over the whole run: {r:.2f}, but only {r_late:.2f} after step 200. The critic's number tracks the")
print("     big early drop (true W 1.1 -> 0.36), then hovers around 0, even negative, while the true W stays ~0.3-0.4.")
print("     The estimate is K * W with a small, unknown K set by clipping, plus critic noise: it reports coarse")
print("     progress, not fine differences. (The paper's Figure 3 curves are median-filtered, too.)")

# --------------------------------------------------------------------------------------------- 6
section("6. Weight clipping and the Lipschitz constant (critic 2 -> 64 -> 64 -> 1)")
torch.manual_seed(0)
for c in (1.0, 0.1, 0.01):
    f = Critic(2, (64, 64))
    for p_ in f.parameters():
        nn.init.uniform_(p_, -1, 1)
    clip_weights(f, c)
    print(f"  c = {c:5.2f}: spectral-norm bound K <= {lipschitz_bound(f):10.3e}, observed max |grad f| = "
          f"{max_grad_norm(f, torch.randn(500, 2)):.3e}")
print("  -> clipping makes f K-Lipschitz, but K shrinks like c^3 for 3 layers: the paper itself calls clipping")
print("     'a clearly terrible way to enforce a Lipschitz constraint' (too small c -> vanishing gradients).")

print(f"\n(total {time.time() - T0:.1f} s)")
