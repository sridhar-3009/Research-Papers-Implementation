"""Reproduce the findings of Krizhevsky, Sutskever & Hinton (2012).

ImageNet training (5-6 days on 2 GPUs in 2012) is out of reach, so the design choices are
tested on CIFAR-10 with a small 4-layer CNN, the way the paper itself did for Figure 1 and
the LRN claim, plus an optional full-AlexNet run on any ImageNet-style folder.

  E1  Figure 1: ReLU vs tanh, iterations to reach 25% training error on CIFAR-10
      (paper: ReLU ~6x faster). No regularization, each with its own best learning rate.
  E2  Section 3.3: local response normalization on/off on CIFAR-10 (paper: 13% -> 11% test error)
  E3  Section 3.4: overlapping (3x3, stride 2) vs plain (2x2, stride 2) max-pooling
  E4  Section 4.1: data augmentation (random crops + flips, PCA colour) on/off
  E5  Section 4.2: dropout on/off in a bigger fully connected head (train vs test error gap)
  E6  The full AlexNet on an ImageNet-style folder (--imagenet DIR, with train/ and val/
      sub-folders of class folders; e.g. Imagenette, a 10-class ImageNet subset).
      Paper recipe: batch 128, momentum 0.9, weight decay 5e-4, lr 0.01 divided by 10 on
      plateau, ~90 epochs; 10-crop testing. Also saves Figure 3 (the conv1 kernels).
      (Simplification: one random crop position per batch, not per image, for speed.)

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick                  # CIFAR parts, small, ~10 minutes
       python3 experiments.py --only e2                # one experiment
       python3 experiments.py --only e6 --imagenet ~/data/imagenette2   # days for real ImageNet
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from alexnet import (AlexNet, DivideOnPlateau, PaperDropout, SmallCifarNet, paper_sgd_step, pca_color_augment,
                     rgb_pca, ten_crop)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# CIFAR-10 helpers
# ---------------------------------------------------------------------------

def load_cifar(quick):
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    f = lambda d: (torch.tensor(d.data).permute(0, 3, 1, 2).float() / 255, torch.tensor(d.targets))
    (X, y), (Xt, yt) = f(tr), f(te)
    mean = X.mean((0, 2, 3), keepdim=True)                     # "subtracting the mean activity" (Section 2)
    X, Xt = X - mean, Xt - mean
    if quick:
        X, y, Xt, yt = X[:10000], y[:10000], Xt[:2000], yt[:2000]
    return (X, y), (Xt, yt)


def augment_cifar(xb, pca=None, g=None):
    """Section 4.1 at CIFAR scale: pad 4 + random 32x32 crop (translations), random mirror,
    and PCA colour noise."""
    N = len(xb)
    padded = F.pad(xb, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2, N), generator=g)
    out = torch.stack([padded[k, :, i[k]:i[k] + 32, j[k]:j[k] + 32] for k in range(N)])
    flip = torch.rand(N, generator=g) < 0.5
    out[flip] = out[flip].flip(-1)
    if pca is not None:
        out = torch.stack([pca_color_augment(im, *pca, generator=g) for im in out])
    return out


@torch.no_grad()
def error(net, X, y, bs=1000):
    net.eval()
    wrong = sum((net(X[i:i + bs].to(DEV)).argmax(1).cpu() != y[i:i + bs]).sum().item() for i in range(0, len(X), bs))
    net.train()
    return wrong / len(X)


def train_cifar(net, train, test, epochs, lr, wd=0.0, augment=False, pca=None, stop_at_train_error=None, seed=0):
    """SGD with momentum 0.9 and batch 128 (the paper's update rule). Logs errors every epoch.
    stop_at_train_error: return as soon as the running training error drops below it (Figure 1)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = net.to(DEV)
    params = list(net.parameters())
    vel = [torch.zeros_like(p) for p in params]
    X, y = train
    log = {"train_err": [], "test_err": [], "iterations_to_target": None}
    it = 0
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        wrong = 0
        for s in range(0, len(X), 128):
            idx = perm[s:s + 128]
            xb = augment_cifar(X[idx], pca, g) if augment else X[idx]
            out = net(xb.to(DEV))
            loss = F.cross_entropy(out, y[idx].to(DEV))
            for p in params:
                p.grad = None
            loss.backward()
            paper_sgd_step(params, vel, lr, weight_decay=wd)
            wrong += (out.argmax(1).cpu() != y[idx]).sum().item()
            it += 1
            if not torch.isfinite(loss):
                log["diverged"] = True
                return log
        log["train_err"].append(wrong / len(X))
        log["test_err"].append(error(net, *test))
        print(f"    epoch {ep + 1}: train {log['train_err'][-1]:.3f}  test {log['test_err'][-1]:.3f}", flush=True)
        if stop_at_train_error and log["train_err"][-1] <= stop_at_train_error:
            log["iterations_to_target"] = it
            break
    return log


# ---------------------------------------------------------------------------

def e1(train, test, a):
    out = {}
    for act in ("relu", "tanh"):
        print(f"  E1 {act}")
        out[act] = fastest_lr(act, train, test, a)
    r, t = out["relu"]["iterations_to_target"], out["tanh"]["iterations_to_target"]
    out["speedup"] = (t / r) if (r and t) else None
    return out


def fastest_lr(act, train, test, a):
    """The paper chose each network's learning rate 'independently to make training as fast as
    possible': try a few and keep the one that reaches 25% training error in the fewest iterations."""
    runs = {}
    for lr in (0.03, 0.01, 0.003):
        runs[lr] = train_cifar(SmallCifarNet(act=act), train, test, epochs=a.epochs * 3, lr=lr, stop_at_train_error=0.25)
    best = min(runs, key=lambda lr: runs[lr]["iterations_to_target"] or float("inf"))
    return dict(lr=best, **runs[best])


def e2(train, test, a):
    return {name: train_cifar(SmallCifarNet(lrn=lrn), train, test, a.epochs, lr=0.01, wd=5e-4)
            for name, lrn in (("no LRN", False), ("LRN", True))}


def e3(train, test, a):
    return {name: train_cifar(SmallCifarNet(overlap=ov), train, test, a.epochs, lr=0.01, wd=5e-4)
            for name, ov in (("plain 2x2/2", False), ("overlapping 3x3/2", True))}


def e4(train, test, a):
    pca = rgb_pca(train[0][:5000])
    return {name: train_cifar(SmallCifarNet(), train, test, a.epochs, lr=0.01, wd=5e-4, augment=aug, pca=p)
            for name, aug, p in (("none", False, None), ("crops + flips", True, None), ("crops + flips + PCA colour", True, pca))}


class CifarWithHead(nn.Module):
    """SmallCifarNet's conv layers + a big 2-layer fully connected head (like fc6/fc7), to overfit."""

    def __init__(self, dropout):
        super().__init__()
        base = SmallCifarNet()
        self.convs = nn.ModuleList([base.c1, base.c2, base.c3])
        self.pool = base.pool
        self.fc1, self.fc2, self.fc3 = nn.Linear(1024, 1024), nn.Linear(1024, 1024), nn.Linear(1024, 10)
        self.drop = PaperDropout() if dropout else nn.Identity()

    def forward(self, x):
        for c in self.convs:
            x = self.pool(F.relu(c(x)))
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)


def e5(train, test, a):
    return {name: train_cifar(CifarWithHead(d), train, test, a.epochs * 2, lr=0.01, wd=5e-4)
            for name, d in (("no dropout", False), ("dropout", True))}


# ---------------------------------------------------------------------------
# E6: the real network on an ImageNet-style folder
# ---------------------------------------------------------------------------

def e6(_train, _test, a):
    if not a.imagenet:
        print("  E6 skipped: pass --imagenet DIR (with train/ and val/ class folders)")
        return None
    T = torchvision.transforms
    to256 = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])      # Section 2: shorter side 256, center crop
    tr = torchvision.datasets.ImageFolder(Path(a.imagenet) / "train", transform=to256)
    va = torchvision.datasets.ImageFolder(Path(a.imagenet) / "val", transform=to256)
    loader = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    sample = torch.stack([tr[i][0] for i in np.random.default_rng(0).choice(len(tr), 500, replace=False)])
    mean = sample.mean((0, 2, 3)).view(1, 3, 1, 1)
    pca = rgb_pca(sample - mean)
    net = AlexNet(num_classes=len(tr.classes)).to(DEV)
    params = list(net.parameters())
    vel = [torch.zeros_like(p) for p in params]
    sched = DivideOnPlateau(lr=0.01, patience=2)
    log = {"val_top1": [], "val_top5": [], "lr": []}
    for ep in range(a.imagenet_epochs):
        net.train()
        for xb, yb in loader:
            xb = xb - mean
            i, j = np.random.randint(0, 33, 2)
            xb = xb[:, :, i:i + 224, j:j + 224]                              # random 224 crop (one per batch for speed)
            if np.random.rand() < 0.5:
                xb = xb.flip(-1)
            xb = torch.stack([pca_color_augment(im, *pca) for im in xb])
            loss = F.cross_entropy(net(xb.to(DEV)), yb.to(DEV))
            for p in params:
                p.grad = None
            loss.backward()
            paper_sgd_step(params, vel, sched.lr)
        top1, top5 = ten_crop_eval(net, va, mean)
        log["val_top1"].append(top1); log["val_top5"].append(top5); log["lr"].append(sched.lr)
        sched.update(top1)
        print(f"  epoch {ep + 1}: val top-1 {top1:.3f} top-5 {top5:.3f} lr {sched.lr:g}", flush=True)
    torch.save(net.state_dict(), HERE / "alexnet.pt")
    figure3_kernels(net)
    return log


@torch.no_grad()
def ten_crop_eval(net, dataset, mean, limit=2000):
    """Average the softmax over 10 crops (Section 4.1)."""
    net.eval()
    wrong1 = wrong5 = 0
    n = min(limit, len(dataset))
    for i in range(n):
        img, label = dataset[i]
        probs = F.softmax(net(ten_crop(img - mean[0]).to(DEV)), 1).mean(0)
        top5 = probs.topk(5).indices.cpu()
        wrong1 += int(top5[0] != label); wrong5 += int(label not in top5)
    return wrong1 / n, wrong5 / n


def figure3_kernels(net):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = net.conv1.weight.detach().cpu()
    w = (w - w.amin((1, 2, 3), keepdim=True)) / (w.amax((1, 2, 3), keepdim=True) - w.amin((1, 2, 3), keepdim=True))
    fig, axes = plt.subplots(6, 16, figsize=(12, 4.8))
    for k, ax in enumerate(axes.flat):
        ax.imshow(w[k].permute(1, 2, 0)); ax.axis("off")
    fig.suptitle("conv1 kernels (top 3 rows: 'GPU 1', bottom 3 rows: 'GPU 2')")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "e6_conv1_kernels.png", dpi=120); plt.close(fig)


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}


def report(R, a):
    L = ["# Results", "", "CIFAR-10 parts: SmallCifarNet, batch 128, momentum 0.9" + (" (QUICK run)" if a.quick else ""), ""]
    fin = lambda r: f"train {100 * r['train_err'][-1]:.1f}% / test {100 * r['test_err'][-1]:.1f}%"
    if R.get("e1"):
        e = R["e1"]
        L += ["## E1: ReLU vs tanh (Figure 1; paper: ReLU ~6x faster to 25% training error)", "",
              f"- ReLU: {e['relu']['iterations_to_target']} iterations (lr {e['relu']['lr']})",
              f"- tanh: {e['tanh']['iterations_to_target']} iterations (lr {e['tanh']['lr']})",
              f"- speed-up: {e['speedup']:.1f}x" if e.get("speedup") else "- speed-up: n/a (a run didn't reach 25%)", ""]
    for key, title in (("e2", "E2: LRN (paper: 13% -> 11% on CIFAR-10)"), ("e3", "E3: overlapping pooling"),
                       ("e4", "E4: data augmentation"), ("e5", "E5: dropout in the fully connected head")):
        if R.get(key):
            L += [f"## {title}", ""] + [f"- {k}: {fin(v)}" for k, v in R[key].items()] + [""]
    if R.get("e6"):
        L += ["## E6: AlexNet on the given folder", "",
              f"best val top-1 {100 * min(R['e6']['val_top1']):.1f}%, top-5 {100 * min(R['e6']['val_top5']):.1f}%",
              "", "![](figures/e6_conv1_kernels.png)"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imagenet", default=None)
    ap.add_argument("--imagenet-epochs", type=int, default=90)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.epochs = a.epochs or (3 if a.quick else 30)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        train, test = load_cifar(a.quick) if a.only != "e6" else (None, None)
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(train, test, a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
