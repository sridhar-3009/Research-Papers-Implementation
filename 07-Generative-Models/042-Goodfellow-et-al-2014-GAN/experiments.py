"""Reproduce the experiments of Generative Adversarial Nets (Goodfellow et al. 2014).

  E1  Table 1 (MNIST): the paper's MLPs (generator 100 -> 1200 -> 1200 -> 784, ReLU + sigmoid; discriminator maxout
      240x5 -> 240x5 with dropout), SGD with momentum, k = 1; Parzen-window test log-likelihood with sigma chosen
      on a validation set from 10,000 generated samples. Paper: 225 +- 2.
  E2  Section 3: minimax (saturating) vs non-saturating generator loss, Parzen score over training.
  E3  Algorithm 1's k: k = 1 (paper) vs k = 5 discriminator steps per generator step.
  E4  Figure 2c/d: CIFAR-10 with a fully connected model and with a convolutional discriminator +
      'deconvolutional' generator; sample grids.
  E5  Figures 2-3: MNIST samples next to their nearest training example, and z-space interpolations.
  (The Toronto Face Database used in the paper is not freely downloadable, so TFD is skipped.)

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

from gan import (Discriminator, Generator, interpolate, nearest_neighbours, parzen_log_likelihood, select_sigma,
                 train_step)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
FIG = HERE / "figures"
SIGMAS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3]


def mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    f = lambda d: d.data.float().view(-1, 784) / 255                               # real-valued, as in Table 1
    X = f(tr)
    return X[:50000], X[50000:], f(te)


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
    return torch.cat(xs).float() / 255


class ConvG(nn.Module):
    """A 'deconvolutional' generator for 32x32x3 (Figure 2d)."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(nn.Linear(z_dim, 256 * 4 * 4), nn.ReLU())
        self.net = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    def sample_z(self, n):
        return torch.rand(n, self.z_dim) * 2 - 1

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 4, 4)).flatten(1)


class ConvD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Unflatten(1, (3, 32, 32)), nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                                 nn.Dropout(0.3), nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                                 nn.Flatten(), nn.Linear(128 * 8 * 8, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def opts(G, D, a):
    """The paper used SGD with momentum (and decaying learning rates in its released code)."""
    return (torch.optim.SGD(G.parameters(), a.lr, momentum=0.5), torch.optim.SGD(D.parameters(), a.lr, momentum=0.5))


def train(G, D, X, a, iters, k=1, saturating=False, log_every=None, Xv=None, Xt=None):
    og, od = opts(G, D, a)
    sample_real = lambda m: X[torch.randint(0, len(X), (m,))]
    curve = []
    for it in range(1, iters + 1):
        for g in og.param_groups + od.param_groups:                               # momentum ramp-up, lr decay
            g["lr"] = a.lr * (a.lr_decay ** it)
            g["momentum"] = min(0.7, 0.5 + 0.2 * it / max(1, iters // 2))
        train_step(G, D, og, od, sample_real, m=100, k=k, saturating=saturating)
        if log_every and it % log_every == 0:
            curve.append({"iter": it, **parzen_score(G, Xv, Xt, a)})
    return curve


@torch.no_grad()
def parzen_score(G, Xv, Xt, a):
    G.eval()
    s = G(G.sample_z(a.n_samples))
    G.train()
    sigma, _ = select_sigma(s, Xv[:a.n_valid], SIGMAS)
    ll = parzen_log_likelihood(s, Xt[:a.n_test], sigma)
    return {"sigma": sigma, "log-lik": ll.mean().item(), "std err": (ll.std() / len(ll) ** 0.5).item()}


def mnist_models():
    return Generator(100, (1200, 1200), 784), Discriminator(784, (240, 240), pieces=5)


def e1(a):
    X, Xv, Xt = mnist()
    torch.manual_seed(0)
    G, D = mnist_models()
    train(G, D, X, a, a.iters)
    torch.save(G.state_dict(), HERE / "g_mnist.pt")
    out = parzen_score(G, Xv, Xt, a)
    out["paper"] = "225 +- 2"
    print("  E1", out, flush=True)
    return out


def e2(a):
    X, Xv, Xt = mnist()
    out = {}
    for sat in (True, False):
        torch.manual_seed(0)
        G, D = mnist_models()
        out["minimax log(1 - D(G(z)))" if sat else "non-saturating -log D(G(z))"] = \
            train(G, D, X, a, a.iters, saturating=sat, log_every=a.iters // 5, Xv=Xv, Xt=Xt)
        print("  E2", sat, flush=True)
    return out


def e3(a):
    X, Xv, Xt = mnist()
    out = {}
    for k in (1, 5):
        torch.manual_seed(0)
        G, D = mnist_models()
        train(G, D, X, a, a.iters // k, k=k)                                       # the same number of D updates
        out[f"k={k}"] = parzen_score(G, Xv, Xt, a)
        print("  E3", k, out[f"k={k}"], flush=True)
    return out


def grid(x, rows, C, H, W):
    x = x.view(-1, C, H, W)[: rows * (len(x) // rows)]
    return x.view(rows, -1, C, H, W).permute(0, 3, 1, 4, 2).reshape(rows * H, -1, C).squeeze(-1)


def save(img, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.imsave(FIG / name, img.clamp(0, 1).numpy(), cmap="gray" if img.dim() == 2 else None)


def e4(a):
    X = cifar10()
    out = {}
    for name, G, D in (("fully connected", Generator(100, (8000, 8000), 3072), Discriminator(3072, (1600, 1600))),
                       ("convolutional", ConvG(), ConvD())):
        torch.manual_seed(0)
        train(G, D, X.view(len(X), -1), a, a.iters)
        with torch.no_grad():
            save(grid(G(G.sample_z(64)), 8, 3, 32, 32), f"e4_cifar_{name.replace(' ', '_')}.png")
        out[name] = f"figures/e4_cifar_{name.replace(' ', '_')}.png"
    return out


def e5(a):
    X, _, _ = mnist()
    G, _ = mnist_models()
    if not (HERE / "g_mnist.pt").exists():
        return {"note": "run e1 first"}
    G.load_state_dict(torch.load(HERE / "g_mnist.pt"))
    torch.manual_seed(0)
    with torch.no_grad():
        s = G(G.sample_z(30))
        nn_ = nearest_neighbours(s, X)
        rows = torch.stack([torch.cat([s[i * 5:(i + 1) * 5], nn_[i * 5 + 4:i * 5 + 5]]) for i in range(6)]).view(-1, 784)
        save(grid(rows, 6, 1, 28, 28), "e5_samples_with_nearest_train.png")
        lines = torch.cat([interpolate(G, G.sample_z(1)[0], G.sample_z(1)[0], 10) for _ in range(6)])
        save(grid(lines, 6, 1, 28, 28), "e5_interpolations.png")
    return {"figures": ["figures/e5_samples_with_nearest_train.png", "figures/e5_interpolations.png"]}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Table 1, MNIST Parzen log-likelihood"), ("e2", "E2: saturating vs non-saturating"),
                 ("e3", "E3: k"), ("e4", "E4: CIFAR-10 samples"), ("e5", "E5: figures")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.lr, a.lr_decay, a.iters = 0.1, 1 - 4e-6, 100000
    a.n_samples, a.n_valid, a.n_test = 10000, 10000, 10000
    if a.quick:
        a.iters, a.n_samples, a.n_valid, a.n_test = 2000, 2000, 1000, 1000
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
