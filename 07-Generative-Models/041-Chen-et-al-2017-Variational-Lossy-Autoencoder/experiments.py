"""Reproduce the experiments of the Variational Lossy Autoencoder (Chen et al. 2017), scaled to one machine.

  E1  Section 4.1 lossy compression: statically binarized MNIST. VAE with a factorized decoder vs VLAE (AF prior,
      6-layer 3x3 PixelCNN decoder). Average KL in bits (paper: 37.3 vs 19.2) and 'decompressions'
      z ~ q(z|x), x' ~ p(x|z) saved as a figure (Figure 1a), plus VLAE samples (Figure 1b).
  E2  Table 1: statically binarized MNIST NLL of IAF-posterior VAE vs AF-prior VAE (both factorized decoders) vs VLAE.
  E3  Tables 2-4: VLAE vs the same PixelCNN without the VAE part ('Unconditional Decoder') on dynamically
      binarized MNIST, OMNIGLOT and Caltech-101 Silhouettes, with the hyperparameters of E2.
  E4  Figure 3: CIFAR-10 decoders with receptive fields 4x2, 5x3, 7x4 and 7x4 grayscale context: nats in the code
      and decompressions.
  E5  The demo's information-preference toy at full length: KL in z vs decoder window 0..15, several seeds.

Common settings: conv encoder (strided convs, ELU), 32 latents (64 for CIFAR), free bits on the total KL
(lambda per latent dimension x number of dimensions, because an AF prior's KL does not split per dimension),
Adamax, importance-sampled NLL (paper: 4096 samples on binary data, 512 on CIFAR-10).

!! HEAVY: PixelCNN decoders + thousands of importance samples. Not run on the author's laptop.
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

from vlae import VLAE, LocalPixelCNN, iaf040, log_standard_normal, toy_data, toy_true_log_prob, train_toy

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
FIG = HERE / "figures"
LOG2PI = math.log(2 * math.pi)


# ---------------------------------------------------------------------------------------------------- data
def fetch(url, name):
    d = DATA / "vlae"; d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    return p


def static_mnist():
    """Hugo Larochelle's fixed binarization (the paper's footnote 1)."""
    base = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat"
    load = lambda s: torch.tensor([[float(v) for v in l.split()] for l in
                                   fetch(base.format(s), f"bmnist_{s}.amat").read_text().splitlines()]).view(-1, 1, 28, 28)
    return torch.cat([load("train"), load("valid")]), load("test")


def dynamic_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    f = lambda d: d.data.float().div(255).unsqueeze(1)
    return f(tr), f(te)                                                            # binarized per minibatch


def omniglot():
    import scipy.io
    m = scipy.io.loadmat(fetch("https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat",
                               "chardata.mat"))
    f = lambda a: torch.tensor(a.T, dtype=torch.float32).view(-1, 28, 28).transpose(1, 2).unsqueeze(1)
    return f(m["data"]), f(m["testdata"])                                          # grey -> binarized per minibatch


def caltech():
    import scipy.io
    m = scipy.io.loadmat(fetch("https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat",
                               "caltech101_silhouettes_28_split1.mat"))
    f = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 28, 28).transpose(1, 2).unsqueeze(1)
    return torch.cat([f(m["train_data"]), f(m["val_data"])]), f(m["test_data"])


def cifar10():
    d = DATA / "cifar10"; d.mkdir(parents=True, exist_ok=True)
    t = d / "cifar-10-python.tar.gz"
    if not t.exists():
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", t)
        tarfile.open(t).extractall(d)

    def load(names):
        xs = []
        for nm in names:
            with open(d / "cifar-10-batches-py" / nm, "rb") as f:
                xs.append(torch.tensor(pickle.load(f, encoding="bytes")[b"data"]).view(-1, 3, 32, 32))
        return torch.cat(xs).float() / 256
    return load([f"data_batch_{i}" for i in range(1, 6)]), load(["test_batch"])


# ---------------------------------------------------------------------------------------------------- models
def conv_encoder(C=1, width=64):
    return nn.Sequential(nn.Conv2d(C, width, 3, 2, 1), nn.ELU(), nn.Conv2d(width, width, 3, 2, 1), nn.ELU(),
                         nn.Conv2d(width, width, 3, 2, 1), nn.ELU(), nn.Flatten(), nn.LazyLinear(450), nn.ELU())


class FactorizedDecoder(nn.Module):
    """p(x|z) = prod_i Bernoulli(x_i | z): an MLP + transposed convs, no autoregression."""

    def __init__(self, z_dim=32, width=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, 450), nn.ELU(), nn.Linear(450, width * 7 * 7), nn.ELU())
        self.net = nn.Sequential(nn.ConvTranspose2d(width, width, 4, 2, 1), nn.ELU(),
                                 nn.ConvTranspose2d(width, 1, 4, 2, 1))
        self.width = width

    def log_prob(self, x, z):
        lead = x.shape[:-3]
        o = self.net(self.fc(z.reshape(-1, z.shape[-1])).view(-1, self.width, 7, 7))
        xf = x.reshape(-1, *x.shape[-3:])
        return -F.binary_cross_entropy_with_logits(o, xf, reduction="none").flatten(1).sum(-1).view(lead)

    @torch.no_grad()
    def sample(self, z):
        return torch.bernoulli(torch.sigmoid(self.net(self.fc(z).view(-1, self.width, 7, 7))))


class IAFVAE(nn.Module):
    """Table 1's 'IAF VAE': N(0, I) prior, IAF posterior (040's Algorithm 1), factorized decoder."""

    def __init__(self, z_dim=32, T=2, width=320):
        super().__init__()
        self.encoder, self.decoder, self.z_dim = conv_encoder(), FactorizedDecoder(z_dim), z_dim
        self.mu, self.logvar, self.h = nn.Linear(450, z_dim), nn.Linear(450, z_dim), nn.Linear(450, 64)
        self.flow = iaf040.IAFPosterior(z_dim, T, width, 1, 64)

    def terms(self, x, n=1):
        a = self.encoder(x)
        mu, logvar, h = (t.expand(n, *t.shape) for t in (self.mu(a), self.logvar(a), self.h(a)))
        z, log_q = self.flow(mu, logvar, h)
        return self.decoder.log_prob(x.expand(n, *x.shape), z), log_standard_normal(z), log_q

    def elbo(self, x, n=1):
        rec, lp, lq = self.terms(x, n)
        return (rec + lp - lq).mean(0)

    def kl(self, x, n=16):
        _, lp, lq = self.terms(x, n)
        return (lq - lp).mean(0)

    @torch.no_grad()
    def log_likelihood(self, x, n=128):
        rec, lp, lq = self.terms(x, n)
        return torch.logsumexp(rec + lp - lq, 0) - math.log(n)


def make(kind, z_dim=32, C=1, size=28, window=None, gray=False, likelihood="bernoulli"):
    if kind == "iaf-vae":
        return _init(IAFVAE(z_dim), C, size)
    enc = conv_encoder(C)
    if kind == "af-vae":
        dec = FactorizedDecoder(z_dim)
    elif kind == "vae":
        dec = FactorizedDecoder(z_dim)
        return _init(VLAE(enc, 450, z_dim, dec, prior="normal"), C, size)
    else:                                                                          # 'vlae'
        dec = LocalPixelCNN(C, size, z_dim, 64, window=window, layers=6, gray_context=gray, likelihood=likelihood)
    return _init(VLAE(enc, 450, z_dim, dec, prior="af", af_steps=2, af_hidden=640), C, size)


def _init(m, C, size):
    with torch.no_grad():
        m.encoder(torch.zeros(1, C, size, size))                                  # materialise LazyLinear
    return m


class Unconditional(nn.Module):
    """The 'Unconditional Decoder' rows: the same PixelCNN with no z at all."""

    def __init__(self):
        super().__init__()
        self.decoder = LocalPixelCNN(1, 28, 0, 64, window=None, layers=6)

    def log_likelihood(self, x, n=1):
        return self.decoder.log_prob(x)

    def elbo(self, x, n=1):
        return self.decoder.log_prob(x)


# ---------------------------------------------------------------------------------------------------- training
def train(model, X, a, epochs, binarize, free_bits):
    opt = torch.optim.Adamax(model.parameters(), lr=a.lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(a.batch):
            x = torch.bernoulli(X[idx]) if binarize else X[idx]
            if hasattr(model, "terms"):
                rec, lp, lq = model.terms(x)
                kl = (lq - lp).mean()
                lam = free_bits * model.z_dim
                loss = -rec.mean() + torch.clamp(kl, min=lam)                       # free bits on the total KL
            else:
                loss = -model.elbo(x).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 100.0); opt.step()


@torch.no_grad()
def evaluate(model, X, binarize, n_is, chunk=50):
    g = torch.Generator().manual_seed(0)
    nll, kl = [], []
    for k in range(0, len(X), chunk):
        x = X[k:k + chunk]
        x = torch.bernoulli(x, generator=g) if binarize else x
        nll.append(-model.log_likelihood(x, n_is))
        if hasattr(model, "kl"):
            kl.append(model.kl(x))
    out = {"NLL (nats)": torch.cat(nll).mean().item()}
    if kl:
        out["KL in code (bits)"] = torch.cat(kl).mean().item() / math.log(2)
    return out


def grid(x, rows):
    x = x[: rows * (len(x) // rows)]
    B, C, H, W = x.shape
    return x.view(rows, -1, C, H, W).permute(0, 3, 1, 4, 2).reshape(rows * H, -1, C).squeeze(-1)


def save(img, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.imsave(FIG / name, img.clamp(0, 1).numpy(), cmap="gray" if img.dim() == 2 else None)


@torch.no_grad()
def decompress(model, x):
    a = model.encoder(x)
    z = model.mu(a) + (0.5 * model.logvar(a)).exp() * torch.randn(len(x), model.z_dim)
    return model.decoder.sample(z)


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    Xtr, Xte = static_mnist()
    out = {}
    for kind in ("vae", "vlae"):
        torch.manual_seed(0)
        m = make(kind)
        train(m, Xtr, a, a.epochs, False, a.free_bits)
        out[kind] = evaluate(m, Xte[:a.n_test], False, a.n_is)
        x = Xte[:32]
        save(grid(torch.cat([x, decompress(m, x)]), 8), f"e1_decompress_{kind}.png")
        if kind == "vlae":
            save(grid(m.decoder.sample(m.prior.sample(32)), 4), "e1_vlae_samples.png")
        print("  E1", kind, out[kind], flush=True)
    out["paper"] = {"vae KL bits": 37.3, "vlae KL bits": 19.2}
    return out


def e2(a):
    Xtr, Xte = static_mnist()
    out = {}
    for name, kind in (("IAF VAE", "iaf-vae"), ("AF VAE", "af-vae"), ("VLAE", "vlae")):
        torch.manual_seed(0)
        m = make(kind)
        train(m, Xtr, a, a.epochs, False, a.free_bits)
        out[name] = evaluate(m, Xte[:a.n_test], False, a.n_is)
        print("  E2", name, out[name], flush=True)
    out["paper NLL"] = {"IAF VAE": 79.88, "AF VAE": 79.30, "VLAE": 79.03}
    return out


def e3(a):
    out = {}
    paper = {"dynamic MNIST": (87.55, 78.53), "OMNIGLOT": (95.02, 90.98), "Caltech-101": (89.26, 77.36)}
    for name, loader in (("dynamic MNIST", dynamic_mnist), ("OMNIGLOT", omniglot), ("Caltech-101", caltech)):
        Xtr, Xte = loader()
        binar = name != "Caltech-101"
        res = {}
        for label, m in (("Unconditional Decoder", Unconditional()), ("VLAE", make("vlae"))):
            torch.manual_seed(0)
            train(m, Xtr, a, a.epochs, binar, a.free_bits)
            res[label] = evaluate(m, Xte[:a.n_test], binar, a.n_is if label == "VLAE" else 1)
        res["paper NLL (unconditional, VLAE)"] = paper[name]
        out[name] = res
        print("  E3", name, res, flush=True)
    return out


def e4(a):
    Xtr, Xte = cifar10()
    out = {}
    for name, window, gray in (("4x2", (4, 2), False), ("5x3", (5, 3), False), ("7x4", (7, 4), False),
                               ("7x4 grayscale", (7, 4), True)):
        torch.manual_seed(0)
        m = make("vlae", z_dim=64, C=3, size=32, window=window, gray=gray, likelihood="logistic")
        train(m, Xtr, a, a.epochs_cifar, False, a.free_bits)
        r = evaluate(m, Xte[:a.n_test_cifar], False, a.n_is_cifar)
        r["bits/dim"] = r["NLL (nats)"] / (3 * 32 * 32 * math.log(2))
        out[name] = r
        x = Xte[:16]
        save(grid(torch.cat([x, decompress(m, x)]), 4), f"e4_decompress_{name.replace(' ', '_')}.png")
        print("  E4", name, r, flush=True)
    return out


def e5(a):
    X, Xt = toy_data(20000), toy_data(5000)
    H = -toy_true_log_prob(Xt).mean().item()
    out = {"H(data) nats": H}
    for w in a.toy_windows:
        kls, gaps = [], []
        for seed in range(a.toy_seeds):
            m = train_toy(w, a.toy_steps, seed, X)
            with torch.no_grad():
                kls.append(m.kl(Xt).mean().item()); gaps.append(-m.elbo(Xt, 16).mean().item() - H)
        out[f"window {w}"] = {"KL in z (nats)": sum(kls) / len(kls), "-ELBO - H (nats)": sum(gaps) / len(gaps)}
        print("  E5", w, out[f"window {w}"], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: lossy compression (KL bits)"), ("e2", "E2: Table 1"), ("e3", "E3: Tables 2-4"),
                 ("e4", "E4: Figure 3 receptive fields (CIFAR-10)"), ("e5", "E5: information preference sweep")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.lr, a.batch, a.free_bits = 2e-3, 64, 0.03
    a.epochs, a.n_test, a.n_is = 100, 10000, 4096
    a.epochs_cifar, a.n_test_cifar, a.n_is_cifar = 50, 10000, 512
    a.toy_windows, a.toy_seeds, a.toy_steps = (0, 1, 2, 4, 8, 15), 5, 5000
    if a.quick:
        a.epochs, a.n_test, a.n_is = 1, 200, 32
        a.epochs_cifar, a.n_test_cifar, a.n_is_cifar = 1, 50, 8
        a.toy_windows, a.toy_seeds, a.toy_steps = (0, 1, 15), 1, 500
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
