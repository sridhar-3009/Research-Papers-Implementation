"""Reproduce the experiments of Inverse Autoregressive Flow (Kingma et al. 2016), scaled to one machine.

  E1  Table 1: dynamically binarized MNIST, convolutional VAE with 32 latents (Appendix B, simplified: plain
      conv/ELU blocks, no weight norm), posterior = diagonal / IAF (depth 2, width 320) / IAF (depth 2, 4, 8,
      width 1920). Reports the bound (VLB) and an importance-sampled log p(x) with 128 samples, over seeds.
  E2  Table 3: IAF location-only (sigma = 1) vs location+scale, on the E1 setup.
  E3  Tables 2/4 in miniature: CIFAR-10, a conv VAE with one 16x8x8 latent tensor (flattened to 1024 dims),
      discretized logistic likelihood (App. C.5), free bits (App. C.8), posterior = diagonal / linear IAF /
      IAF with 1 hidden layer (1 and 2 steps). Reports bits/dim. (The paper's ResNet VAE with up to 20 stochastic
      layers and bidirectional inference is far beyond this script.)
  E4  Figure 7 / App. C.8: free bits lambda in {0, 0.125, 0.5, 2}; KL per latent group over training.
  E5  Figure 1: VAE with a 2-D latent on four datapoints; scatter plots of each datapoint's posterior samples
      for diagonal vs IAF posteriors.
  E6  Section 6.2 'synthesis speed': one parallel decoder pass (VAE) vs pixel-by-pixel autoregressive sampling
      (a MADE over 784 binary pixels), seconds per image.

!! HEAVY. Not run on the author's laptop.
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

from iaf import (MADE, IAFPosterior, LinearIAF, FlowVAE, discretized_logistic_log_prob, free_bits_kl,
                 log_standard_normal)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
FIG = HERE / "figures"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG2PI = math.log(2 * math.pi)


def mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    return tr.data.float().div(255).unsqueeze(1), te.data.float().div(255).unsqueeze(1)


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
        return torch.cat(xs).float() / 256                                         # values k/256, k = 0..255
    return load([f"data_batch_{i}" for i in range(1, 6)]), load(["test_batch"])


# ---------------------------------------------------------------------------------------------------- models
class ConvVAE(nn.Module):
    """Encoder: strided convs (ELU) -> FC -> (mu, log sigma^2, h); decoder mirrors it with transposed convs.
    posterior: 'diag' | 'iaf' | 'linear'; likelihood: 'bernoulli' (MNIST) or 'logistic' (CIFAR-10)."""

    def __init__(self, C=1, size=28, z_dim=32, fc=450, posterior="iaf", T=2, width=320, mode="gated",
                 likelihood="bernoulli", ctx=64, chans=(16, 32, 32)):
        super().__init__()
        self.z_dim, self.posterior, self.likelihood, self.C = z_dim, posterior, likelihood, C
        layers, c = [], C
        for k in chans:
            layers += [nn.Conv2d(c, k, 3, 2, 1), nn.ELU()]
            c = k
        self.enc = nn.Sequential(*layers, nn.Flatten())
        self.s = math.ceil(size / 2 ** len(chans))
        flat = c * self.s * self.s
        self.fc = nn.Sequential(nn.Linear(flat, fc), nn.ELU())
        self.mu, self.logvar, self.h = nn.Linear(fc, z_dim), nn.Linear(fc, z_dim), nn.Linear(fc, ctx)
        self.dfc = nn.Sequential(nn.Linear(z_dim, fc), nn.ELU(), nn.Linear(fc, flat), nn.ELU())
        dl, rev = [], list(chans[::-1]) + [C]
        for i in range(len(chans)):
            dl += [nn.ConvTranspose2d(rev[i], rev[i + 1] if i < len(chans) - 1 else 32, 4, 2, 1), nn.ELU()]
        dl[-1] = nn.Identity()
        self.dec = nn.Sequential(*dl, nn.ELU(), nn.Conv2d(32, C, 3, 1, 1))
        self.top_c, self.size = chans[-1], size
        self.log_scale = nn.Parameter(torch.full((C,), -3.0))                     # per-channel logistic log-scale
        if posterior == "iaf":
            self.flow = IAFPosterior(z_dim, T, width, 1, ctx, mode=mode)
        elif posterior == "linear":
            self.flow = LinearIAF(z_dim, ctx)

    def q(self, x, n=1):
        a = self.fc(self.enc(x))
        mu, logvar, h = (t.expand(n, *t.shape) for t in (self.mu(a), self.logvar(a), self.h(a)))
        if self.posterior == "diag":
            eps = torch.randn_like(mu)
            return mu + (0.5 * logvar).exp() * eps, -(0.5 * eps ** 2 + 0.5 * LOG2PI + 0.5 * logvar), None
        z, log_q = self.flow(mu, logvar, h)
        return z, log_q, "total"

    def decode(self, z):
        lead = z.shape[:-1]
        f = self.dfc(z.reshape(-1, self.z_dim)).view(-1, self.top_c, self.s, self.s)
        out = self.dec(f)[..., :self.size, :self.size]
        return out.reshape(*lead, *out.shape[1:])

    def log_px(self, x, z):
        out = self.decode(z)
        if self.likelihood == "bernoulli":
            return -F.binary_cross_entropy_with_logits(out, x.expand_as(out), reduction="none").flatten(-3).sum(-1)
        mu = torch.sigmoid(out)
        return discretized_logistic_log_prob(x.expand_as(mu), mu, self.log_scale[:, None, None]).flatten(-3).sum(-1)

    def terms(self, x, n=1):
        """(log p(x|z), per-dim log p(z) - log q(z|x) or its total) for n samples."""
        z, log_q, kind = self.q(x, n)
        rec = self.log_px(x, z)
        if kind is None:                                                           # diagonal: per-dimension KL terms
            kl = log_q - (-0.5 * (LOG2PI + z ** 2))                                # n, B, D (single-sample estimate)
        else:
            kl = (log_q - log_standard_normal(z)).unsqueeze(-1)                     # n, B, 1
        return rec, kl

    def elbo(self, x, n=1):
        rec, kl = self.terms(x, n)
        return (rec - kl.sum(-1)).mean(0)

    @torch.no_grad()
    def log_likelihood(self, x, n=128):
        rec, kl = self.terms(x, n)
        return torch.logsumexp(rec - kl.sum(-1), 0) - math.log(n)


# ---------------------------------------------------------------------------------------------------- training
def train(model, X, a, epochs, binarize=False, lam=0.0, track_kl=False):
    model.to(DEV)
    opt = torch.optim.Adamax(model.parameters(), lr=a.lr)                         # the paper's code uses Adamax
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=a.lr_decay)
    hist = []
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(X)).split(a.batch):
            x = X[idx].to(DEV)
            if binarize:
                x = torch.bernoulli(x)                                             # dynamic binarization
            rec, kl = model.terms(x)
            if lam > 0:
                loss = -(rec.mean()) + free_bits_kl(kl.mean(0), lam)               # Eq. 15
            else:
                loss = -(rec - kl.sum(-1)).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 100.0); opt.step()
        sched.step()
        if track_kl:
            with torch.no_grad():
                _, kl = model.terms(X[:500].to(DEV))
                hist.append(kl.mean((0, 1)).tolist())
    return hist


@torch.no_grad()
def evaluate(model, X, binarize=False, n=128, chunk=100, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    vlb, ll = [], []
    for k in range(0, len(X), chunk):
        x = X[k:k + chunk]
        if binarize:
            x = torch.bernoulli(x, generator=g)
        x = x.to(DEV)
        vlb.append(model.elbo(x, 16).cpu()); ll.append(model.log_likelihood(x, n).cpu())
    return torch.cat(vlb).mean().item(), torch.cat(ll).mean().item()


def e1(a, variants=None):
    Xtr, Xte = mnist()
    variants = variants or [("Diagonal covariance", dict(posterior="diag")),
                            ("IAF (Depth = 2, Width = 320)", dict(T=2, width=320)),
                            ("IAF (Depth = 2, Width = 1920)", dict(T=2, width=1920)),
                            ("IAF (Depth = 4, Width = 1920)", dict(T=4, width=1920)),
                            ("IAF (Depth = 8, Width = 1920)", dict(T=8, width=1920))]
    out = {}
    for name, kw in variants:
        runs = []
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = ConvVAE(**kw)
            train(m, Xtr, a, a.epochs, binarize=True)
            runs.append(evaluate(m, Xte[:a.n_test], binarize=True))
        v = torch.tensor(runs)
        out[name] = {"VLB": v[:, 0].mean().item(), "VLB std": v[:, 0].std().item() if len(runs) > 1 else 0.0,
                     "log p(x)": v[:, 1].mean().item(), "log p(x) std": v[:, 1].std().item() if len(runs) > 1 else 0.0}
        print("  E1", name, out[name], flush=True)
    return out


def e2(a):
    return e1(a, [("IAF location-only", dict(T=2, width=320, mode="location")),
                  ("IAF location+scale", dict(T=2, width=320, mode="gated"))])


def bits_per_dim(nats, dims=3 * 32 * 32):
    return -nats / (dims * math.log(2))


def cifar_vae(**kw):
    return ConvVAE(C=3, size=32, z_dim=1024, fc=1024, likelihood="logistic", ctx=256, chans=(64, 128, 128), **kw)


def e3(a):
    Xtr, Xte = cifar10()
    out = {}
    for name, kw in (("factorized Gaussian", dict(posterior="diag")),
                     ("IAF, linear (0 hidden layers), 1 step", dict(posterior="linear")),
                     ("IAF, 1 hidden layer, 1 step", dict(T=1, width=2048)),
                     ("IAF, 1 hidden layer, 2 steps", dict(T=2, width=2048))):
        torch.manual_seed(0)
        m = cifar_vae(**kw)
        train(m, Xtr, a, a.epochs_cifar, lam=a.free_bits)
        vlb, ll = evaluate(m, Xte[:a.n_test], n=a.is_samples_cifar)
        out[name] = {"bits/dim (bound)": bits_per_dim(vlb), "bits/dim (importance sampled)": bits_per_dim(ll)}
        print("  E3", name, out[name], flush=True)
    return out


def e4(a):
    Xtr, Xte = cifar10()
    out = {}
    for lam in (0.0, 0.125, 0.5, 2.0):
        torch.manual_seed(0)
        m = cifar_vae(posterior="diag")
        hist = train(m, Xtr, a, a.epochs_cifar, lam=lam, track_kl=True)
        vlb, _ = evaluate(m, Xte[:a.n_test], n=8)
        last = torch.tensor(hist[-1])
        out[f"lambda={lam}"] = {"bits/dim (bound)": bits_per_dim(vlb), "total KL (nats)": last.sum().item(),
                                "dims with KL > 0.01": int((last > 0.01).sum()), "KL history (sum per epoch)":
                                [sum(h) for h in hist]}
        print("  E4", lam, {k: v for k, v in out[f"lambda={lam}"].items() if "history" not in k}, flush=True)
    return out


def e5(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = torch.zeros(4, 20)
    for i in range(4):
        X[i, 5 * i:5 * i + 5] = 1
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(*torch.randn(2000, 2).T, s=2, c="gray"); ax[0].set_title("(a) prior")
    out = {}
    for k, post in ((1, "diag"), (2, "iaf")):
        torch.manual_seed(0)
        m = FlowVAE(20, 2, 64, posterior=post, T=4, iaf_hidden=64, ctx=16)
        opt = torch.optim.Adam(m.parameters(), 3e-3)
        for _ in range(a.toy_steps):
            loss = -m.elbo(X, n=8).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            z, _ = m.q_sample(X, 500)
            out[post] = {"bound": m.elbo(X, 5000).mean().item(), "log p(x)": m.log_likelihood(X, 5000).mean().item()}
        for i in range(4):
            ax[k].scatter(*z[:, i].T, s=2)
        ax[k].set_title("(b) diagonal posteriors" if post == "diag" else "(c) IAF posteriors")
    fig.savefig(FIG / "e5_figure1.png", dpi=100); plt.close(fig)
    out["figure"] = "figures/e5_figure1.png"
    return out


def e6(a):
    """Seconds per image: VAE (z -> decoder, one pass) vs an autoregressive model sampling 784 pixels one by one."""
    torch.manual_seed(0)
    vae = ConvVAE(posterior="iaf").to(DEV).eval()
    ar = MADE(784, 2000, 2, n_out=1).to(DEV).eval()
    with torch.no_grad():
        t = time.time()
        for _ in range(a.speed_reps):
            torch.sigmoid(vae.decode(torch.randn(1, 32, device=DEV)))
        t_vae = (time.time() - t) / a.speed_reps
        t = time.time()
        x = torch.zeros(1, 784, device=DEV)
        for i in range(784):
            x[:, i] = torch.bernoulli(torch.sigmoid(ar(x)[0][:, i]))
        t_ar = time.time() - t
    return {"VAE seconds/image": t_vae, "autoregressive seconds/image": t_ar, "ratio": t_ar / t_vae}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Table 1 (MNIST)"), ("e2", "E2: Table 3, location-only vs location+scale"),
                 ("e3", "E3: CIFAR-10 posteriors (bits/dim)"), ("e4", "E4: free bits"),
                 ("e5", "E5: Figure 1 toy"), ("e6", "E6: synthesis speed")):
        if R.get(k):
            body = R[k]
            if k == "e4":
                body = {kk: {x: y for x, y in v.items() if "history" not in x} for kk, v in body.items()}
            L += [f"## {t}", "", "```", json.dumps(body, indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.lr, a.lr_decay, a.batch = 2e-3, 0.995, 100
    a.epochs, a.seeds, a.n_test = 200, 5, 10000
    a.epochs_cifar, a.free_bits, a.is_samples_cifar = 100, 0.125, 128
    a.toy_steps, a.speed_reps = 3000, 100
    if a.quick:
        a.epochs, a.seeds, a.n_test = 3, 1, 1000
        a.epochs_cifar, a.is_samples_cifar = 1, 16
        a.toy_steps, a.speed_reps = 1000, 10
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
