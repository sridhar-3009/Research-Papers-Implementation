"""Reproduce the experiments of Wasserstein GAN (Arjovsky et al. 2017), with CIFAR-10 (32x32) standing in for
LSUN bedrooms (64x64, ~42 GB).

  E1  Figure 3: WGAN training curves (the critic's estimate of W) for three set-ups: MLP generator + DCGAN critic,
      DCGAN generator + DCGAN critic, MLP generator + MLP critic with a too-high learning rate. Sample grids are
      saved along the way so the curve can be compared with sample quality.
  E2  Figure 4: the same generators trained as standard GANs (-log D trick); the JS lower bound 1/2 L(D, g) + log 2
      over training.
  E3  Figures 5-7: robustness to the generator architecture: DCGAN, DCGAN without batch norm and with a constant
      number of filters, 4-layer 512-unit ReLU MLP; WGAN vs GAN sample grids (+ a mode-collapse proxy: the mean
      pairwise distance between samples).
  E4  Section 4.2's negative result: Adam (beta1 = 0.5) vs RMSProp on the critic; logs the cosine between Adam's
      step and the gradient (the paper saw it turn negative when training became unstable).
  E5  The clipping constant c in {0.001, 0.01, 0.1}: estimate scale, and samples.

Algorithm 1 defaults: RMSProp, lr 5e-5, c = 0.01, m = 64, n_critic = 5.

!! HEAVY: GPU-days in the paper. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import math
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgan import clip_weights, critic_objective, generator_loss

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
FIG = HERE / "figures"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cifar10():
    d = DATA / "cifar10"; d.mkdir(parents=True, exist_ok=True)
    t = d / "cifar-10-python.tar.gz"
    if not t.exists():
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", t)
        tarfile.open(t).extractall(d)
    xs = []
    for i in range(1, 6):
        with open(d / "cifar-10-batches-py" / f"data_batch_{i}", "rb") as f:
            xs.append(torch.tensor(pickle.load(f, encoding="bytes")[b"data"]).view(-1, 3, 32, 32))
    return torch.cat(xs).float().div(127.5).sub(1)                                 # tanh range


# ---------------------------------------------------------------------------------------------------- networks
class DCGANG(nn.Module):
    """DCGAN generator for 32x32; bn=False and constant=True gives Figure 6's weakened version."""

    def __init__(self, z_dim=100, ch=64, bn=True, constant=False):
        super().__init__()
        c = [ch * 4, ch * 2, ch] if not constant else [ch, ch, ch]
        norm = (lambda k: nn.BatchNorm2d(k)) if bn else (lambda k: nn.Identity())
        self.z_dim = z_dim
        self.net = nn.Sequential(nn.ConvTranspose2d(z_dim, c[0], 4, 1, 0), norm(c[0]), nn.ReLU(),
                                 nn.ConvTranspose2d(c[0], c[1], 4, 2, 1), norm(c[1]), nn.ReLU(),
                                 nn.ConvTranspose2d(c[1], c[2], 4, 2, 1), norm(c[2]), nn.ReLU(),
                                 nn.ConvTranspose2d(c[2], 3, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z.view(-1, self.z_dim, 1, 1))


class MLPG(nn.Module):
    """Figure 7: 4 hidden layers of 512 ReLU units."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        layers, d = [], z_dim
        for _ in range(4):
            layers += [nn.Linear(d, 512), nn.ReLU()]
            d = 512
        self.net = nn.Sequential(*layers, nn.Linear(512, 3 * 32 * 32), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(-1, 3, 32, 32)


class DCGANCritic(nn.Module):
    """The DCGAN discriminator WITHOUT the final sigmoid (the paper uses it as critic and, with a sigmoid in the
    loss, as GAN discriminator, so both losses are comparable)."""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.BatchNorm2d(ch * 4), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch * 4, 1, 4, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


class MLPCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3072, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        return self.net(x).view(-1)


# ---------------------------------------------------------------------------------------------------- training
def train(G, f, X, a, mode="wgan", lr=None, clip=None, critic_opt="rmsprop", tag=""):
    """mode='wgan': Algorithm 1. mode='gan': standard GAN with the -log D trick (k = 1, Adam as in DCGAN).
    Logs the critic's W estimate (WGAN) or the JS lower bound 1/2 L + log 2 (GAN) and saves sample grids."""
    G, f = G.to(DEV), f.to(DEV)
    lr = lr or a.lr
    clip = a.clip if clip is None else clip
    if mode == "wgan":
        og = torch.optim.RMSprop(G.parameters(), lr)
        of = torch.optim.RMSprop(f.parameters(), lr) if critic_opt == "rmsprop" else \
            torch.optim.Adam(f.parameters(), lr, betas=(0.5, 0.999))
    else:
        og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
        of = torch.optim.Adam(f.parameters(), 2e-4, betas=(0.5, 0.999))
    sample_real = lambda: X[torch.randint(0, len(X), (a.m,))].to(DEV)
    sample_z = lambda n: torch.randn(n, G.z_dim, device=DEV)
    fixed_z = sample_z(64)
    log, cosines = [], []
    for it in range(1, a.iters + 1):
        n_critic = a.n_critic if mode == "wgan" else 1
        if mode == "wgan" and (it <= 25 or it % 500 == 0):
            n_critic = 100                                                         # the reference code's warm start
        for _ in range(n_critic):
            with torch.no_grad():
                fake = G(sample_z(a.m))
            if mode == "wgan":
                obj = critic_objective(f, sample_real(), fake)
                of.zero_grad(); (-obj).backward()
                if critic_opt == "adam":
                    before = [p.detach().clone() for p in f.parameters()]
                    grads = torch.cat([p.grad.flatten() for p in f.parameters()])
                of.step()
                if critic_opt == "adam":
                    step = torch.cat([(p.detach() - b).flatten() for p, b in zip(f.parameters(), before)])
                    cosines.append(F.cosine_similarity(step, -grads, dim=0).item())  # ascent direction is -grad of -obj
                clip_weights(f, clip)
                value = obj.item()
            else:
                L = F.logsigmoid(f(sample_real())).mean() + F.logsigmoid(-f(fake)).mean()
                of.zero_grad(); (-L).backward(); of.step()
                value = 0.5 * L.item() + math.log(2)
        fake = G(sample_z(a.m))
        lg = generator_loss(f, fake) if mode == "wgan" else -F.logsigmoid(f(fake)).mean()
        og.zero_grad(); lg.backward(); og.step()
        if it % a.log_every == 0:
            with torch.no_grad():
                s = G(sample_z(256))
                spread = torch.pdist(s.flatten(1)).mean().item()                   # mode-collapse proxy
            log.append({"iter": it, "estimate": value, "sample spread": spread})
        if tag and it % a.save_every == 0:
            with torch.no_grad():
                save(grid(G(fixed_z).cpu()), f"{tag}_iter{it}.png")
    return {"log": log, "adam cosines (last 100)": cosines[-100:] if cosines else None}


def grid(x, rows=8):
    x = (x.clamp(-1, 1) + 1) / 2
    B, C, H, W = x.shape
    return x.view(rows, -1, C, H, W).permute(0, 3, 1, 4, 2).reshape(rows * H, -1, C)


def save(img, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.imsave(FIG / name, img.numpy())


def e1(a):
    X = cifar10()
    out = {}
    for name, G, f, lr in (("MLP G, DCGAN critic", MLPG(), DCGANCritic(), None),
                           ("DCGAN G, DCGAN critic", DCGANG(), DCGANCritic(), None),
                           ("MLP G, MLP critic, high lr", MLPG(), MLPCritic(), 1e-3)):
        torch.manual_seed(0)
        out[name] = train(G, f, X, a, "wgan", lr=lr, tag="e1_" + name.split(",")[0].replace(" ", ""))
        print("  E1", name, out[name]["log"][-1], flush=True)
    return out


def e2(a):
    X = cifar10()
    out = {}
    for name, G, D in (("MLP G, DCGAN D", MLPG(), DCGANCritic()), ("DCGAN G, DCGAN D", DCGANG(), DCGANCritic()),
                       ("MLP G, MLP D", MLPG(), MLPCritic())):
        torch.manual_seed(0)
        out[name] = train(G, D, X, a, "gan", tag="e2_" + name.split(",")[0].replace(" ", ""))
        print("  E2", name, out[name]["log"][-1], flush=True)
    return out


def e3(a):
    X = cifar10()
    out = {}
    for gname, make in (("DCGAN", lambda: DCGANG()), ("no-BN constant-filter DCGAN", lambda: DCGANG(bn=False, constant=True)),
                        ("MLP 4x512", lambda: MLPG())):
        for mode in ("wgan", "gan"):
            torch.manual_seed(0)
            r = train(make(), DCGANCritic(), X, a, mode, tag=f"e3_{mode}_{gname.split()[0]}")
            out[f"{gname} | {mode}"] = r["log"][-1]
            print("  E3", gname, mode, r["log"][-1], flush=True)
    return out


def e4(a):
    X = cifar10()
    out = {}
    for opt in ("rmsprop", "adam"):
        torch.manual_seed(0)
        r = train(DCGANG(), DCGANCritic(), X, a, "wgan", critic_opt=opt)
        out[opt] = {"final": r["log"][-1], "adam cosine (mean of last 100)":
                    (sum(r["adam cosines (last 100)"]) / len(r["adam cosines (last 100)"])) if r["adam cosines (last 100)"] else None}
        print("  E4", opt, out[opt], flush=True)
    return out


def e5(a):
    X = cifar10()
    out = {}
    for c in (0.001, 0.01, 0.1):
        torch.manual_seed(0)
        r = train(DCGANG(), DCGANCritic(), X, a, "wgan", clip=c, tag=f"e5_clip{c}")
        out[f"c={c}"] = r["log"][-1]
        print("  E5", c, r["log"][-1], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Figure 3, WGAN curves"), ("e2", "E2: Figure 4, GAN JS estimates"),
                 ("e3", "E3: Figures 5-7, robustness"), ("e4", "E4: Adam vs RMSProp"), ("e5", "E5: clipping constant")):
        if R.get(k):
            body = R[k]
            if k in ("e1", "e2"):
                body = {kk: v["log"][-5:] for kk, v in body.items()}
            L += [f"## {t}", "", "```", json.dumps(body, indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.lr, a.clip, a.m, a.n_critic = 5e-5, 0.01, 64, 5
    a.iters, a.log_every, a.save_every = 100000, 500, 10000
    if a.quick:
        a.iters, a.log_every, a.save_every = 60, 20, 60
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
