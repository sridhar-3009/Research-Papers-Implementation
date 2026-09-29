"""Reproduce the experiments of Kingma & Ba (2015) with the optimizers in optimizers.py.

  fig1a   MNIST L2-regularized logistic regression, minibatch 128: Adam (alpha/sqrt(t)),
          SGD with Nesterov momentum, AdaGrad                                       (Section 6.1)
  fig1b   sparse bag-of-words logistic regression with 50% input dropout: Adam, AdaGrad,
          RMSProp, SGD Nesterov. The paper used IMDB; this script uses 20 Newsgroups
          (scikit-learn, also sparse 10,000-word bag-of-words) as a stand-in.
  fig2    MNIST MLP 784-1000-1000-10 ReLU with dropout: AdaGrad, RMSProp, SGD Nesterov,
          AdaDelta, Adam                                                             (Section 6.2)
  fig3    CIFAR-10 CNN c64-c64-c128-1000 (5x5 conv, 3x3 max-pool stride 2), with and
          without dropout: Adam, AdaGrad, SGD Nesterov                                (Section 6.3)
  fig4    VAE (500 softplus units, 50-d latent) on MNIST: bias correction on/off, over
          beta1 in {0, 0.9}, beta2 in {0.99, 0.999, 0.9999}, log10(alpha) in -5..-1 (Section 6.4)

Every optimizer's learning rate is picked from a small grid (the paper: "a dense grid").

!! HEAVY. Not run on the author's laptop. Each part can take from minutes (fig1a) to
!! hours (fig3, fig4) on a CPU. Use --quick for a small version, and run one part at a time:
       python3 experiments.py fig1a --quick
       python3 experiments.py fig2 --epochs 20
Results are saved to results_<part>.json and figures/<part>.png.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from optimizers import AdaDelta, AdaGrad, Adam, RMSProp, SGDNesterov

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# A tiny training loop that applies OUR optimizers to torch parameters
# ---------------------------------------------------------------------------

def fit(params, loss_fn, batches, opt, epochs, eval_fn):
    """params: list of leaf tensors (requires_grad). loss_fn(batch) -> scalar.
    batches() yields minibatches for one epoch. Returns eval_fn() after every epoch."""
    history = [eval_fn()]
    for _ in range(epochs):
        for batch in batches():
            loss = loss_fn(batch)
            grads = torch.autograd.grad(loss, params)
            new = opt.step([p.detach() for p in params], [g.detach() for g in grads])
            with torch.no_grad():
                for p, n in zip(params, new):
                    p.copy_(n)
            if not torch.isfinite(loss):
                return history + [float("nan")]
        history.append(eval_fn())
    return history


def minibatches(X, y, size, seed):
    gen = torch.Generator().manual_seed(seed)

    def batches():
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), size):
            idx = perm[i:i + size].to(X.device)
            yield X[idx], y[idx]
    return batches


def best_over_grid(make_params, loss_fn, batches, make_opt, lrs, epochs, eval_fn_for):
    """Run each learning rate from the same initialization; keep the lowest final cost."""
    best = None
    for lr in lrs:
        params = make_params()
        hist = fit(params, loss_fn(params), batches, make_opt(lr), epochs, eval_fn_for(params))
        final = hist[-1] if np.isfinite(hist[-1]) else np.inf
        if best is None or final < best[0]:
            best = (final, lr, hist)
    return dict(lr=best[1], history=best[2])


def mnist(flat=True):
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    X = tr.data.float().to(DEV) / 255
    return (X.reshape(-1, 784) if flat else X[:, None]), tr.targets.to(DEV)


# ---------------------------------------------------------------------------

def fig1a(epochs, quick):
    X, y = mnist()
    if quick:
        X, y = X[:5000], y[:5000]
    torch.manual_seed(0)
    W0, b0 = torch.randn(784, 10, device=DEV) * 0.01, torch.zeros(10, device=DEV)
    make = lambda: [W0.clone().requires_grad_(), b0.clone().requires_grad_()]
    loss_fn = lambda P: lambda b: F.cross_entropy(b[0] @ P[0] + P[1], b[1]) + 1e-4 * (P[0] ** 2).sum()
    evalf = lambda P: lambda: float(F.cross_entropy(X @ P[0] + P[1], y).detach())
    batches = minibatches(X, y, 128, 0)
    opts = {"Adam": (lambda lr: Adam(lr, decay_sqrt_t=True), (0.01, 0.003, 0.001)),
            "SGDNesterov": (lambda lr: SGDNesterov(lr, 0.9), (0.1, 0.03, 0.01)),
            "AdaGrad": (lambda lr: AdaGrad(lr), (0.1, 0.03, 0.01))}
    return {k: best_over_grid(make, loss_fn, batches, mk, lrs, epochs, evalf) for k, (mk, lrs) in opts.items()}


def fig1b(epochs, quick):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), data_home=str(DATA))
    vec = CountVectorizer(max_features=10000, binary=True)
    X = torch.tensor(vec.fit_transform(d.data).toarray(), dtype=torch.float32, device=DEV)
    y = torch.tensor(d.target, device=DEV)
    if quick:
        X, y = X[:2000], y[:2000]
    torch.manual_seed(0)
    W0 = torch.zeros(10000, 20, device=DEV)
    make = lambda: [W0.clone().requires_grad_(), torch.zeros(20, device=DEV, requires_grad=True)]
    drop = lambda x: x * (torch.rand_like(x) < 0.5).float() * 2              # 50% dropout on features
    loss_fn = lambda P: lambda b: F.cross_entropy(drop(b[0]) @ P[0] + P[1], b[1])
    evalf = lambda P: lambda: float(F.cross_entropy(X @ P[0] + P[1], y).detach())
    batches = minibatches(X, y, 128, 0)
    opts = {"Adam": (lambda lr: Adam(lr, decay_sqrt_t=True), (0.03, 0.01, 0.003)),
            "AdaGrad": (lambda lr: AdaGrad(lr), (0.3, 0.1, 0.03)),
            "RMSProp": (lambda lr: RMSProp(lr), (0.003, 0.001, 0.0003)),
            "SGDNesterov": (lambda lr: SGDNesterov(lr, 0.9), (0.3, 0.1, 0.03))}
    return {k: best_over_grid(make, loss_fn, batches, mk, lrs, epochs, evalf) for k, (mk, lrs) in opts.items()}


def mlp_params(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        P.append((torch.randn(a, b, generator=g) * np.sqrt(2 / a)).to(DEV))       # He-style init for ReLU
        P.append(torch.zeros(b, device=DEV))
    return P


def fig2(epochs, quick):
    X, y = mnist()
    if quick:
        X, y = X[:5000], y[:5000]
    P0 = mlp_params([784, 1000, 1000, 10])
    make = lambda: [p.clone().requires_grad_() for p in P0]

    def forward(P, x, train):
        h = x
        for i in range(0, len(P), 2):
            h = h @ P[i] + P[i + 1]
            if i < len(P) - 2:
                h = torch.relu(h)
                if train:
                    h = h * (torch.rand_like(h) < 0.5).float() * 2             # dropout
        return h
    loss_fn = lambda P: lambda b: F.cross_entropy(forward(P, b[0], True), b[1])
    evalf = lambda P: lambda: float(torch.stack([F.cross_entropy(forward(P, X[i:i + 5000], False), y[i:i + 5000])
                                                 for i in range(0, len(X), 5000)]).mean().detach())
    batches = minibatches(X, y, 128, 0)
    opts = {"AdaGrad": (AdaGrad, (0.03, 0.01)), "RMSProp": (RMSProp, (0.001, 0.0003)),
            "SGDNesterov": (lambda lr: SGDNesterov(lr, 0.9), (0.1, 0.03)),
            "AdaDelta": (lambda lr: AdaDelta(), (None,)), "Adam": (Adam, (0.001, 0.0003))}
    return {k: best_over_grid(make, loss_fn, batches, mk, lrs, epochs, evalf) for k, (mk, lrs) in opts.items()}


def fig3(epochs, quick):
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    X = torch.tensor(tr.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255
    X = (X - X.mean((0, 2, 3), keepdim=True)) / X.std((0, 2, 3), keepdim=True)   # per-channel standardizing
    X, y = X.to(DEV), torch.tensor(tr.targets, device=DEV)
    if quick:
        X, y = X[:3000], y[:3000]
    g = torch.Generator().manual_seed(0)
    shapes = [(64, 3, 5, 5), (64,), (64, 64, 5, 5), (64,), (128, 64, 5, 5), (128,), (128 * 4 * 4, 1000), (1000,), (1000, 10), (10,)]
    P0 = [(torch.randn(*s, generator=g) * (np.sqrt(2 / np.prod(s[1:])) if len(s) > 1 else 0)).to(DEV) for s in shapes]

    def forward(P, x, dropout):
        if dropout:
            x = x * (torch.rand_like(x) < 0.8).float() / 0.8                   # input dropout
        for i in (0, 2, 4):
            x = F.max_pool2d(torch.relu(F.conv2d(x, P[i], P[i + 1], padding=2)), 3, 2, padding=1)
        h = torch.relu(x.flatten(1) @ P[6] + P[7])                              # 32 -> 16 -> 8 -> 4, so 128*4*4 inputs
        if dropout:
            h = h * (torch.rand_like(h) < 0.5).float() * 2
        return h @ P[8] + P[9]
    out = {}
    for drop in (False, True):
        make = lambda: [p.clone().requires_grad_() for p in P0]
        loss_fn = lambda P, d=drop: lambda b: F.cross_entropy(forward(P, b[0], d), b[1])
        evalf = lambda P: lambda: float(F.cross_entropy(forward(P, X[:5000], False), y[:5000]).detach())
        batches = minibatches(X, y, 128, 0)
        for k, (mk, lrs) in {"Adam": (Adam, (0.001, 0.0003)), "AdaGrad": (AdaGrad, (0.01, 0.003)),
                             "SGDNesterov": (lambda lr: SGDNesterov(lr, 0.9), (0.01, 0.003))}.items():
            out[k + ("+dropout" if drop else "")] = best_over_grid(make, loss_fn, batches, mk, lrs, epochs, evalf)
    return out


def fig4(epochs, quick):
    """VAE of Kingma & Welling (2014): encoder 784-500(softplus)-(mu, logvar) of 50,
    decoder 50-500(softplus)-784 Bernoulli. Loss = negative ELBO per image."""
    X, _ = mnist()
    X = (X > 0.5).float()
    if quick:
        X = X[:5000]
    g = torch.Generator().manual_seed(0)
    shapes = [(784, 500), (500,), (500, 100), (100,), (50, 500), (500,), (500, 784), (784,)]
    P0 = [(torch.randn(*s, generator=g) * 0.01).to(DEV) for s in shapes]

    def neg_elbo(P, x):
        h = F.softplus(x @ P[0] + P[1])
        mu, logvar = (h @ P[2] + P[3]).chunk(2, dim=1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)                 # reparameterization
        logits = F.softplus(z @ P[4] + P[5]) @ P[6] + P[7]
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum() / len(x)
        return rec + kl
    grid = {}
    alphas = (-4, -3, -2) if quick else (-5, -4, -3, -2, -1)
    for beta1 in (0.0, 0.9):
        for beta2 in (0.99, 0.999, 0.9999):
            for la in alphas:
                for bc in (True, False):
                    params = [p.clone().requires_grad_() for p in P0]
                    hist = fit(params, lambda b, P=params: neg_elbo(P, b[0]), minibatches(X, X, 128, 0),
                               Adam(10.0 ** la, beta1, beta2, bias_correction=bc), epochs,
                               lambda P=params: float(neg_elbo(P, X[:5000]).detach()))
                    grid[f"b1={beta1},b2={beta2},log_a={la},bc={bc}"] = hist
                    print(f"fig4 beta1 {beta1} beta2 {beta2} log10(alpha) {la} bias-correction {bc}: {hist[-1]:.1f}", flush=True)
    return grid


PARTS = {"fig1a": fig1a, "fig1b": fig1b, "fig2": fig2, "fig3": fig3, "fig4": fig4}
DEFAULT_EPOCHS = {"fig1a": 45, "fig1b": 45, "fig2": 50, "fig3": 45, "fig4": 10}


def plot(part, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if part == "fig4":
        for bc, ls in ((True, "-"), (False, "--")):
            for beta2 in (0.99, 0.999, 0.9999):
                keys = [k for k in r if f"b1=0.9,b2={beta2}," in k and k.endswith(f"bc={bc}")]
                xs = [int(k.split("log_a=")[1].split(",")[0]) for k in keys]
                ys = [min(r[k][-1], 300) for k in keys]
                ax.plot(xs, ys, ls, marker="o", label=f"beta2={beta2}, {'bias corr.' if bc else 'no correction'}")
        ax.set_xlabel("log10(alpha)"); ax.set_ylabel("negative ELBO after training")
    else:
        for name, v in r.items():
            ax.semilogy(v["history"], label=f"{name} (lr {v['lr']})")
        ax.set_xlabel("epochs"); ax.set_ylabel("training cost")
    ax.set_title(part); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "figures" / f"{part}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=list(PARTS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="small data, few epochs")
    a = ap.parse_args()
    epochs = a.epochs or (2 if a.quick else DEFAULT_EPOCHS[a.part])
    t0 = time.time()
    r = PARTS[a.part](epochs, a.quick)
    (HERE / f"results_{a.part}.json").write_text(json.dumps(r))
    plot(a.part, r)
    print(f"done in {time.time() - t0:.0f}s; saved results_{a.part}.json and figures/{a.part}.png")


if __name__ == "__main__":
    main()
