"""Reproduce the CIFAR-10 experiments of Zhang et al. (2017).

  E1  Figure 1a: training loss of the small Inception on true labels, random labels, shuffled
      pixels, random pixels and Gaussian images. Paper: ALL reach ~zero training loss.
  E2  Figures 1b, 1c: label corruption p in {0, 0.2, 0.4, 0.6, 0.8, 1.0} for Inception, AlexNet,
      MLP 1x512. Time to fit (relative to p = 0) and test error. Paper: test error rises to 90%,
      training time grows only by a small factor.
  E3  Table 1: Inception (with/without BN), AlexNet, MLP 3x512, MLP 1x512, with/without random-crop
      augmentation and weight decay, plus random labels.
  E4  Section 5: fit the MNIST training labels EXACTLY with a kernel and no regularization.
      Paper: 1.2% test error (a 60,000 x 60,000 kernel matrix = 30 GB in float64!).
      This script uses --kernel-n examples (default 10,000 -> 0.8 GB in float64).

Paper's CIFAR-10 setup: center crop 28x28, per-image whitening, SGD momentum 0.9, lr 0.1
(Inception) or 0.01 (AlexNet, MLP), decayed by 0.95 per epoch, no regularization unless stated.
"Fitted" here means training accuracy 100% (or the epoch limit).

!! HEAVY. Not run on the author's laptop. Fitting random labels takes many epochs.
       python3 experiments.py --quick          # 5,000 images, ~30-60 minutes
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from randomization import (MLP, SmallAlexNet, SmallInception, center_crop, corrupt_labels, gaussian_images,
                           per_image_whitening, random_pixels, shuffle_pixels)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {"Inception": (SmallInception, 0.1), "AlexNet": (SmallAlexNet, 0.01), "MLP 1x512": (lambda: MLP(1), 0.01),
          "MLP 3x512": (lambda: MLP(3), 0.01), "Inception w/o BN": (lambda: SmallInception(bn=False), 0.1)}


def load_cifar(n=None):
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    f = lambda d: (torch.tensor(d.data).permute(0, 3, 1, 2).float() / 255, torch.tensor(d.targets))
    (X, y), (Xt, yt) = f(tr), f(te)
    if n:
        X, y, Xt, yt = X[:n], y[:n], Xt[:n // 5], yt[:n // 5]
    return (X, y), (Xt, yt)


def prep(X):
    """Appendix A: centre crop to 28x28, then per-image whitening."""
    return per_image_whitening(center_crop(X))


def random_crop(X32, g):
    """Data augmentation for Table 1: a random 28x28 crop of the 32x32 image (instead of the centre)."""
    N = len(X32)
    i, j = torch.randint(0, 5, (2, N), generator=g)
    return torch.stack([X32[k, :, i[k]:i[k] + 28, j[k]:j[k] + 28] for k in range(N)])


@torch.no_grad()
def accuracy(net, X, y, bs=1000):
    net.eval()
    acc = sum((net(X[i:i + bs].to(DEV)).argmax(1).cpu() == y[i:i + bs]).sum().item() for i in range(0, len(X), bs))
    net.train()
    return acc / len(X)


def fit(name, X_train, y_train, X_test, y_test, epochs, wd=0.0, augment_from=None, seed=0):
    """Train until 100% training accuracy or `epochs`. X_* are already preprocessed (28x28), except
    when augment_from (the 32x32 originals) is given: then each epoch takes fresh random crops."""
    make, lr = MODELS[name]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = make().to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)                 # x0.95 per epoch
    log = {"loss": [], "train_acc": [], "epochs_to_fit": None}
    for ep in range(epochs):
        Xe = per_image_whitening(random_crop(augment_from, g)) if augment_from is not None else X_train
        perm = torch.randperm(len(Xe), generator=g)
        total = 0.0
        for i in range(0, len(Xe), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(net(Xe[idx].to(DEV)), y_train[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        sched.step()
        log["loss"].append(total / len(Xe))
        log["train_acc"].append(accuracy(net, X_train, y_train))
        print(f"    {name} epoch {ep + 1}: loss {log['loss'][-1]:.3f}  train acc {log['train_acc'][-1]:.4f}", flush=True)
        if log["train_acc"][-1] >= 0.999:
            log["epochs_to_fit"] = ep + 1
            break
    log["test_acc"] = accuracy(net, X_test, y_test)
    return log


def e1(data, a):
    (X, y), (Xt, yt) = data
    g = torch.Generator().manual_seed(0)
    Xp, Xtp = prep(X), prep(Xt)
    shuf, perm = shuffle_pixels(Xp, g)
    settings = {"true labels": (Xp, y, Xtp, yt),
                "random labels": (Xp, corrupt_labels(y, 1.0, generator=g), Xtp, yt),
                "shuffled pixels": (shuf, y, shuffle_pixels(Xtp, perm=perm)[0], yt),
                "random pixels": (random_pixels(Xp, g), y, random_pixels(Xtp, g), yt),
                "gaussian": (gaussian_images(Xp, g), y, gaussian_images(Xtp, g), yt)}
    return {k: fit("Inception", *v, a.epochs) for k, v in settings.items()}


def e2(data, a):
    (X, y), (Xt, yt) = data
    Xp, Xtp = prep(X), prep(Xt)
    out = {}
    for name in ("Inception", "AlexNet", "MLP 1x512"):
        out[name] = {}
        for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            yc = corrupt_labels(y, p, generator=torch.Generator().manual_seed(1))
            out[name][str(p)] = fit(name, Xp, yc, Xtp, yt, a.epochs)
    return out


def e3(data, a):
    (X, y), (Xt, yt) = data
    Xp, Xtp = prep(X), prep(Xt)
    out = {}
    for name in ("Inception", "Inception w/o BN", "AlexNet", "MLP 3x512", "MLP 1x512"):
        crops = (True, False) if name in ("Inception", "AlexNet") else (False,)
        for crop in crops:
            for wd in (5e-4, 0.0):
                key = f"{name} | crop {crop} | wd {wd > 0}"
                out[key] = fit(name, Xp, y, Xtp, yt, a.epochs, wd=wd, augment_from=X if crop else None)
        out[f"{name} | random labels"] = fit(name, Xp, corrupt_labels(y, 1.0, generator=torch.Generator().manual_seed(2)),
                                            Xtp, yt, a.epochs)
    return out


def e4(_data, a):
    """Interpolate the MNIST training labels exactly with a kernel, NO regularization: solve K alpha = Y.
    Note: Eq. (3) with the LINEAR kernel K = X X^T cannot interpolate raw MNIST: d = 784 < n, so
    X X^T has rank <= 784. An exact fit needs a nonlinear kernel; we use a Gaussian kernel
    (bandwidth from the median pairwise distance). Linear least squares is shown for reference."""
    mn = torchvision.datasets.MNIST(DATA, train=True, download=True)
    mt = torchvision.datasets.MNIST(DATA, train=False, download=True)
    n = a.kernel_n
    X = mn.data[:n].double().view(n, -1) / 255
    Xt = mt.data.double().view(len(mt.data), -1) / 255
    Y = F.one_hot(mn.targets[:n], 10).double()
    out = {}
    W = torch.linalg.lstsq(X, Y).solution                       # best linear fit (cannot be exact)
    out["linear least squares"] = dict(test_err=(Xt @ W).argmax(1).ne(mt.targets).double().mean().item(),
                                       train_fit=(X @ W).argmax(1).eq(mn.targets[:n]).double().mean().item())
    D = torch.cdist(X, X)
    s2 = D[D > 0].median().item() ** 2
    K = torch.exp(-D ** 2 / s2)
    alpha = torch.linalg.solve(K, Y)                             # exact interpolation, no ridge
    Kt = torch.exp(-torch.cdist(Xt, X) ** 2 / s2)
    out["gaussian kernel interpolation"] = dict(test_err=(Kt @ alpha).argmax(1).ne(mt.targets).double().mean().item(),
                                                train_fit=(K @ alpha).argmax(1).eq(mn.targets[:n]).double().mean().item())
    for k, v in out.items():
        print(f"  E4 {k}, n={n}: train fit {v['train_fit']:.4f}  test error {v['test_err']:.4f}", flush=True)
    return out


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


def report(R, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    L = ["# Results", "", f"{a.n or 50000} training images, up to {a.epochs} epochs" + (" (QUICK run)" if a.quick else ""), ""]
    if R.get("e1"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, v in R["e1"].items():
            ax.plot(v["loss"], label=k)
        ax.set_xlabel("epochs"); ax.set_ylabel("average training loss"); ax.legend()
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e1_fig1a.png", dpi=130); plt.close(fig)
        L += ["## E1: Figure 1a (Inception)", "", "| setting | epochs to fit | final train acc | test acc |", "|---|---|---|---|"]
        L += [f"| {k} | {v['epochs_to_fit']} | {100 * v['train_acc'][-1]:.1f}% | {100 * v['test_acc']:.1f}% |" for k, v in R["e1"].items()]
        L += ["", "![](figures/e1_fig1a.png)", ""]
    if R.get("e2"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        L += ["## E2: Figures 1b-1c (label corruption)", ""]
        for name, runs in R["e2"].items():
            ps = [float(p) for p in runs]
            base = runs["0.0"]["epochs_to_fit"] or a.epochs
            ax[0].plot(ps, [(r["epochs_to_fit"] or a.epochs) / base for r in runs.values()], "o-", label=name)
            ax[1].plot(ps, [1 - r["test_acc"] for r in runs.values()], "o-", label=name)
            L.append(f"- {name}: test error " + ", ".join(f"p={p}: {100 * (1 - r['test_acc']):.1f}%" for p, r in runs.items()))
        ax[0].set_xlabel("label corruption"); ax[0].set_ylabel("time to fit (relative)")
        ax[1].set_xlabel("label corruption"); ax[1].set_ylabel("test error")
        for x in ax:
            x.legend()
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e2_fig1bc.png", dpi=130); plt.close(fig)
        L += ["", "![](figures/e2_fig1bc.png)", ""]
    if R.get("e3"):
        L += ["## E3: Table 1", "", "| model / setting | train acc | test acc |", "|---|---|---|"]
        L += [f"| {k} | {100 * v['train_acc'][-1]:.2f}% | {100 * v['test_acc']:.2f}% |" for k, v in R["e3"].items()] + [""]
    if R.get("e4"):
        L += ["## E4: exact kernel interpolation on MNIST, no regularization (paper, all 60k: 1.2%)", ""]
        L += [f"- {k}: test error {100 * v['test_err']:.2f}% (train fit {100 * v['train_fit']:.1f}%)" for k, v in R["e4"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="number of training images (default all 50,000)")
    ap.add_argument("--kernel-n", type=int, default=10000)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.epochs = a.epochs or (15 if a.quick else 100)
    a.n = a.n or (5000 if a.quick else None)
    if a.quick:
        a.kernel_n = min(a.kernel_n, 3000)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        data = load_cifar(a.n) if a.only != "e4" else None
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(data, a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
