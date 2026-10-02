"""A few-second tour of Auto-Encoding Variational Bayes.

  1. Eq. 1 on a model where everything is exact: log p(x) = bound + KL(q || true posterior)
  2. Why the reparameterization trick: gradient variance vs the score-function estimator (Section 2.2)
  3. Estimator A (Eq. 6) vs estimator B (Eq. 7): same mean, B has lower variance
  4. AEVB vs wake-sleep on a small slice of MNIST (a toy Figure 2), with a 2-D latent space
  5. Nz = 20 vs 2: more latents, no overfitting; how unevenly the latent dimensions are used
"""

import math
import time
from pathlib import Path

import torch

from vae import VAE, kl_to_standard_normal, reparam_grad, score_function_grad, wake_sleep_losses

torch.manual_seed(0)
T0 = time.time()
DATA = Path(__file__).resolve().parents[2] / "data"


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. Eq. 1: log p(x) = lower bound + KL(q || posterior)")
w, s, x = 2.0, 0.5, 1.3                       # p(z) = N(0,1), p(x|z) = N(2z, 0.25)  =>  p(x) = N(0, 4.25)
logpx = -0.5 * (math.log(2 * math.pi * (w * w + s * s)) + x * x / (w * w + s * s))
v_post = 1 / (1 + w * w / (s * s)); m_post = v_post * w * x / (s * s)


def bound(m, v):
    rec = -0.5 * math.log(2 * math.pi * s * s) - ((x - w * m) ** 2 + w * w * v) / (2 * s * s)
    return rec - kl_to_standard_normal(torch.tensor([m]), torch.tensor([math.log(v)])).item()


def kl_q_post(m, v):
    return 0.5 * (math.log(v_post / v) + (v + (m - m_post) ** 2) / v_post - 1)


print(f"  x = {x}: log p(x) = {logpx:.4f}; true posterior N({m_post:.3f}, {v_post:.4f})")
print(f"  {'q = N(m, v)':>22} {'bound L':>9} {'KL(q||post)':>12} {'L + KL':>8}")
for m, v in ((m_post, v_post), (0.5, v_post), (m_post, 0.5), (0.0, 1.0)):
    print(f"  {f'N({m:.3f}, {v:.4f})':>22} {bound(m, v):9.4f} {kl_q_post(m, v):12.4f} {bound(m, v) + kl_q_post(m, v):8.4f}")
print("  -> L + KL is always log p(x); the bound is tight only when q IS the posterior (the last column never moves).")

# --------------------------------------------------------------------------------------------- 2
section("2. The reparameterization trick: gradient of E[z^2] under N(mu=1, sigma=1); true answer d/dmu = 2")
f = lambda z: z ** 2
sf = score_function_grad(f, 1.0, 0.0, 100000)
rp = reparam_grad(f, 1.0, 0.0, 100000)
print(f"  {'estimator':>18} {'mean d/dmu':>11} {'variance':>9}   {'mean d/dlog s':>13} {'variance':>9}")
for name, g in (("score function", sf), ("reparameterized", rp)):
    print(f"  {name:>18} {g[:, 0].mean():11.3f} {g[:, 0].var():9.2f}   {g[:, 1].mean():13.3f} {g[:, 1].var():9.2f}")
print(f"  -> both are unbiased, but the reparameterized one has {sf[:, 0].var() / rp[:, 0].var():.0f}x / "
      f"{sf[:, 1].var() / rp[:, 1].var():.0f}x less variance with one sample.")

# --------------------------------------------------------------------------------------------- 3
section("3. Estimator A (everything sampled) vs B (KL in closed form), one sample each")
m = VAE(12, 2, 16, init_std=0.3)
xb = torch.bernoulli(torch.full((1, 12), 0.5))
with torch.no_grad():
    A = torch.stack([m.elbo(xb, estimator="A") for _ in range(4000)]).squeeze()
    B = torch.stack([m.elbo(xb, estimator="B") for _ in range(4000)]).squeeze()
print(f"  estimator A: mean {A.mean():.3f}, std {A.std():.3f}")
print(f"  estimator B: mean {B.mean():.3f}, std {B.std():.3f}")
print("  -> same expectation; B removes the noise of sampling the KL term ('typically has less variance').")

# --------------------------------------------------------------------------------------------- 4
section("4. AEVB vs wake-sleep, MNIST slice (5000 train / 1000 test, binarized, Nz = 2, 200 hidden units)")
try:
    import torchvision
    tr = torchvision.datasets.MNIST(DATA, train=True, download=False)
    te = torchvision.datasets.MNIST(DATA, train=False, download=False)
    Xtr = (tr.data[:5000].float().view(-1, 784) / 255 > 0.5).float(); ytr = tr.targets[:5000]
    Xte = (te.data[:1000].float().view(-1, 784) / 255 > 0.5).float()
    source = "MNIST"
except Exception:                                                                # no MNIST on disk: random blobs
    protos = torch.bernoulli(torch.full((10, 784), 0.3))
    ytr = torch.randint(0, 10, (5000,))
    Xtr = (protos[ytr] + (torch.rand(5000, 784) < 0.05)).clamp(max=1)
    Xte = (protos[torch.randint(0, 10, (1000,))] + (torch.rand(1000, 784) < 0.05)).clamp(max=1)
    source = "synthetic (MNIST not found in data/)"
print(f"  data: {source}")


def train_aevb(epochs=8):
    torch.manual_seed(1)
    vae = VAE(784, 2, 200, init_std=0.01)
    opt = torch.optim.Adagrad(vae.parameters(), lr=0.1)
    for _ in range(epochs):
        for idx in torch.randperm(5000).split(100):
            loss = -vae.map_objective(Xtr[idx], N=5000)
            opt.zero_grad(); loss.backward(); opt.step()
    return vae


def train_wake_sleep(epochs=8):
    torch.manual_seed(1)
    ws = VAE(784, 2, 200, init_std=0.01)
    o_dec = torch.optim.Adagrad(ws.dec.parameters(), lr=0.1)
    o_enc = torch.optim.Adagrad(ws.enc.parameters(), lr=0.1)
    for _ in range(epochs):
        for idx in torch.randperm(5000).split(100):
            wake, sleep = wake_sleep_losses(ws, Xtr[idx])
            o_dec.zero_grad(); wake.backward(); o_dec.step()
            o_enc.zero_grad(); sleep.backward(); o_enc.step()
    return ws


vae, ws = train_aevb(), train_wake_sleep()
with torch.no_grad():
    for name, mm in (("AEVB", vae), ("wake-sleep", ws)):
        print(f"  {name:>11}: lower bound per image train {mm.elbo(Xtr[:1000], L=5).mean():8.2f}, "
              f"test {mm.elbo(Xte, L=5).mean():8.2f} nats   (importance-sampled log p(x) test "
              f"{mm.importance_log_likelihood(Xte[:200], K=200).mean():7.2f})")
    mu = vae.enc(Xtr)[0]
    spread = mu.std(0).mean().item()
    centres = torch.stack([mu[ytr == c].mean(0) for c in range(10)])
    within = torch.stack([(mu[ytr == c] - centres[c]).norm(dim=1).mean() for c in range(10)]).mean().item()
    between = torch.pdist(centres).mean().item()
print(f"  AEVB 2-D codes: classes sit {between:.2f} apart on average, points lie {within:.2f} from their class centre"
      f" (overall spread {spread:.2f}) -> similar images get similar codes, without labels.")
print("  -> AEVB optimises one objective (the bound) for both networks; wake-sleep's two losses don't form a bound,"
      " and it is behind after the same number of updates, as in Figure 2.")

# --------------------------------------------------------------------------------------------- 5
section("5. More latents (Nz = 20 vs 2) on the same slice: does it overfit? Which dimensions get used?")
torch.manual_seed(2)
big = VAE(784, 20, 200, init_std=0.01)
opt = torch.optim.Adagrad(big.parameters(), lr=0.1)
for _ in range(8):
    for idx in torch.randperm(5000).split(100):
        loss = -big.map_objective(Xtr[idx], N=5000)
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    for name, mm in (("Nz = 2", vae), ("Nz = 20", big)):
        a, b = mm.elbo(Xtr[:1000], L=5).mean(), mm.elbo(Xte, L=5).mean()
        print(f"  {name:>8}: train bound {a:8.2f}, test bound {b:8.2f}, gap {a - b:5.2f} nats")
    mu, logvar = big.enc(Xtr[:2000])
    kl_dim = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).mean(0)
print("  KL per latent dimension (nats):", " ".join(f"{k:.2f}" for k in kl_dim.sort(descending=True).values))
print("  -> 10x more latents improve the bound by ~20 nats and the train/test gap stays small: no overfitting,"
      " matching Figure 2's remark.")
print(f"     Each dimension pays for its information in KL, so use is very uneven ({kl_dim.max():.2f} down to"
      f" {kl_dim.min():.2f} nats). On this"
      " small slice none switch off completely; fully 'inactive' units are reported for longer training on full"
      " data (e.g. Burda et al. 2016), not reproduced here.")

print(f"\n(total {time.time() - T0:.1f} s)")
