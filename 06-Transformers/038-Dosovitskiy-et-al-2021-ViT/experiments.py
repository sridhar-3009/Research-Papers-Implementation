"""Reproduce the Vision Transformer's main findings (Dosovitskiy et al. 2021) at CIFAR-10 scale.

The paper pre-trains on ImageNet (1.3M), ImageNet-21k (14M) and JFT-300M with TPUv3 pods. One GPU cannot; instead
we test the paper's CLAIMS where they can be tested small:

  E1  Figures 3-4, 'large scale training trumps inductive bias': ViT vs a ResNet trained from scratch on 5%, 20%,
      50% and 100% of CIFAR-10. Paper: ResNets win on small data, ViT overtakes as data grows.
  E2  Sequence length: ViT with 8x8 vs 4x4 patches (N = 16 vs 64 tokens on 32x32 images); accuracy and time.
  E3  Hybrid (a CNN stem whose feature map gives 1x1 'patches') vs pure ViT (Figure 5: hybrids help small models).
  E4  Section 3.2, fine-tuning at higher resolution: train at 32x32, then fine-tune at 48x48 with the position
      embeddings 2-D interpolated, vs keeping 32x32.
  E5  Figure 7: patch-embedding filters (PCA), position-embedding similarity and mean attention distance of the
      E1 ViT trained on all data, saved as figures.
  E6  Section 4.6: masked patch prediction (predict the 3-bit mean colour of 50% corrupted patches) as
      self-supervised pre-training, then supervised fine-tuning, vs from scratch with the same budget.

Training follows Section 4.1 where it applies: Adam (0.9, 0.999), weight decay 0.1, linear warm-up then decay, with
light augmentation (random crops, flips) because CIFAR-10 is tiny.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # a few epochs, ~1 hour on CPU
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

from vit import (ViT, corrupt_patches, count_params, embedding_filters_pca, mean_attention_distance,
                 mean_colour_targets, patchify, position_similarity)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
STD = torch.tensor([0.2470, 0.2435, 0.2616])[:, None, None]


def cifar10():
    d = DATA / "cifar10"; d.mkdir(parents=True, exist_ok=True)
    t = d / "cifar-10-python.tar.gz"
    if not t.exists():
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", t)
        tarfile.open(t).extractall(d)

    def load(names):
        xs, ys = [], []
        for nm in names:
            with open(d / "cifar-10-batches-py" / nm, "rb") as f:
                b = pickle.load(f, encoding="bytes")
            xs.append(torch.tensor(b[b"data"]).view(-1, 3, 32, 32)); ys += b[b"labels"]
        return torch.cat(xs).float() / 255, torch.tensor(ys)
    return load([f"data_batch_{i}" for i in range(1, 6)]), load(["test_batch"])


def augment(x):
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,))
    x = x[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(len(x)) < 0.5
    x[flip] = x[flip].flip(-1)
    return x


def norm(x):
    return (x - MEAN) / STD


class ResNetSmall(nn.Module):
    """A ResNet-18-style CIFAR network with GroupNorm (the paper's 'BiT' ResNets use GroupNorm + weight
    standardisation)."""

    def __init__(self, classes=10, w=64):
        super().__init__()

        def block(cin, cout, stride):
            return nn.ModuleDict({"c1": nn.Conv2d(cin, cout, 3, stride, 1, bias=False), "n1": nn.GroupNorm(8, cout),
                                  "c2": nn.Conv2d(cout, cout, 3, 1, 1, bias=False), "n2": nn.GroupNorm(8, cout),
                                  "sc": nn.Conv2d(cin, cout, 1, stride, bias=False) if (stride > 1 or cin != cout) else nn.Identity()})
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.GroupNorm(8, w), nn.ReLU())
        chans = [(w, w, 1), (w, w, 1), (w, 2 * w, 2), (2 * w, 2 * w, 1), (2 * w, 4 * w, 2), (4 * w, 4 * w, 1),
                 (4 * w, 8 * w, 2), (8 * w, 8 * w, 1)]
        self.blocks = nn.ModuleList([block(*c) for c in chans])
        self.fc = nn.Linear(8 * w, classes)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            h = F.relu(b["n1"](b["c1"](x)))
            x = F.relu(b["n2"](b["c2"](h)) + b["sc"](x))
        return self.fc(x.mean((2, 3)))


def train(model, X, Y, a, epochs, res=32, lr=None):
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or a.lr, betas=(0.9, 0.999), weight_decay=0.1)
    steps = epochs * math.ceil(len(X) / a.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for k in range(0, len(X), a.batch):
            idx = perm[k:k + a.batch]
            x = norm(augment(X[idx]))
            if res != 32:
                x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(model(x.to(DEV)), Y[idx].to(DEV))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return time.time() - t0


@torch.no_grad()
def accuracy(model, X, Y, res=32):
    model.eval(); right = 0
    for k in range(0, len(X), 500):
        x = norm(X[k:k + 500])
        if res != 32:
            x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        right += (model(x.to(DEV)).argmax(-1).cpu() == Y[k:k + 500]).sum().item()
    return 100 * right / len(X)


def vit(a, patch=4, **kw):
    return ViT(32, patch, 3, D=a.D, layers=a.layers, heads=a.heads, mlp=4 * a.D, classes=10, dropout=0.1, head="mlp", **kw)


def e1(a):
    (X, Y), (Xt, Yt) = cifar10()
    out = {}
    for frac in a.fracs:
        n = int(frac * len(X))
        epochs = max(a.epochs, int(a.epochs / frac / 4))                         # roughly equal update counts
        for name, make in (("ResNet (GroupNorm)", lambda: ResNetSmall()), ("ViT", lambda: vit(a))):
            torch.manual_seed(0)
            m = make()
            t = train(m, X[:n], Y[:n], a, epochs)
            out[f"{frac:.0%} {name}"] = {"test acc": accuracy(m, Xt, Yt), "params": count_params(m), "train s": t}
            print("  E1", frac, name, out[f"{frac:.0%} {name}"], flush=True)
            if frac == 1.0 and name == "ViT":
                torch.save(m.state_dict(), HERE / "vit_cifar.pt")
    return out


def e2(a):
    (X, Y), (Xt, Yt) = cifar10()
    out = {}
    for P in (8, 4):
        torch.manual_seed(0)
        m = vit(a, patch=P)
        t = train(m, X, Y, a, a.epochs)
        out[f"patch {P} (N = {(32 // P) ** 2})"] = {"test acc": accuracy(m, Xt, Yt), "train s": t}
        print("  E2", P, out, flush=True)
    return out


def e3(a):
    (X, Y), (Xt, Yt) = cifar10()
    out = {}
    for name, kw in (("ViT patch 4", dict(patch=4)), ("hybrid (CNN stem, /4)", dict(patch=1, hybrid_downsample=4))):
        torch.manual_seed(0)
        m = vit(a, **kw)
        train(m, X, Y, a, a.epochs)
        out[name] = {"test acc": accuracy(m, Xt, Yt), "params": count_params(m)}
        print("  E3", name, out[name], flush=True)
    return out


def e4(a):
    (X, Y), (Xt, Yt) = cifar10()
    torch.manual_seed(0)
    base = vit(a)
    train(base, X, Y, a, a.epochs)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    out = {"pre-trained at 32": accuracy(base, Xt, Yt)}
    keep = vit(a); keep.load_state_dict(state)
    train(keep, X, Y, a, a.ft_epochs, lr=a.lr / 10)
    out["fine-tuned at 32"] = accuracy(keep, Xt, Yt)
    hi = vit(a); hi.load_state_dict(state)
    hi.resize_positions(48 // 4)                                                 # 8x8 grid -> 12x12 grid
    train(hi, X, Y, a, a.ft_epochs, res=48, lr=a.lr / 10)
    out["fine-tuned at 48 (interpolated positions)"] = accuracy(hi, Xt, Yt, res=48)
    return out


def e5(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (HERE / "vit_cifar.pt").exists():
        return {"note": "run e1 first"}
    (X, Y), (Xt, Yt) = cifar10()
    m = vit(a); m.load_state_dict(torch.load(HERE / "vit_cifar.pt", map_location="cpu")); m.eval()
    (HERE / "figures").mkdir(exist_ok=True)
    comps, _ = embedding_filters_pca(m, k=16)
    fig, ax = plt.subplots(2, 8, figsize=(10, 3))
    for k, axx in enumerate(ax.flat):
        f = comps[k].reshape(4, 4, 3)
        axx.imshow(((f - f.min()) / (f.max() - f.min() + 1e-9)).numpy()); axx.axis("off")
    fig.savefig(HERE / "figures" / "e5_filters.png", dpi=100); plt.close(fig)
    S = position_similarity(m)
    g = m.grid
    fig, ax = plt.subplots(g, g, figsize=(6, 6))
    for k, axx in enumerate(ax.flat):
        axx.imshow(S[k].reshape(g, g).numpy(), vmin=-1, vmax=1); axx.axis("off")
    fig.savefig(HERE / "figures" / "e5_positions.png", dpi=100); plt.close(fig)
    d = mean_attention_distance(m, norm(Xt[:256]))
    fig, axx = plt.subplots(figsize=(5, 4))
    for h in range(d.shape[1]):
        axx.scatter(range(1, d.shape[0] + 1), d[:, h], s=10)
    axx.set_xlabel("layer"); axx.set_ylabel("mean attention distance (px)")
    fig.savefig(HERE / "figures" / "e5_attention_distance.png", dpi=100); plt.close(fig)
    return {"figures": ["figures/e5_filters.png", "figures/e5_positions.png", "figures/e5_attention_distance.png"],
            "attention distance per layer (mean over heads)": d.mean(1).tolist()}


def e6(a):
    (X, Y), (Xt, Yt) = cifar10()
    out = {}
    torch.manual_seed(0)
    m = vit(a).to(DEV)
    mask_tok = nn.Parameter(torch.zeros(4 * 4 * 3, device=DEV))
    head = nn.Linear(a.D, 512).to(DEV)
    opt = torch.optim.AdamW(list(m.parameters()) + [mask_tok] + list(head.parameters()), lr=a.lr, weight_decay=0.1)
    for ep in range(a.ssl_epochs):
        perm = torch.randperm(len(X))
        for k in range(0, len(X), a.batch):
            img = augment(X[perm[k:k + a.batch]])
            target = mean_colour_targets(img, 4).to(DEV)
            patches = patchify(norm(img), 4)
            corrupted, chosen, masked = corrupt_patches(patches)
            corrupted = corrupted.to(DEV)
            corrupted[masked.to(DEV)] = mask_tok
            z = m.E(corrupted)
            z = torch.cat([m.cls.expand(len(z), -1, -1), z], 1) + m.pos
            for b in m.blocks:
                z = b(z)
            logits = head(m.ln(z)[:, 1:])
            loss = F.cross_entropy(logits[chosen.to(DEV)], target[chosen.to(DEV)])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"    SSL epoch {ep + 1}: loss {loss.item():.3f}", flush=True)
    m.new_head(10)
    train(m, X, Y, a, a.epochs, lr=a.lr / 3)
    out["masked patch prediction, then fine-tune"] = accuracy(m, Xt, Yt)
    torch.manual_seed(0)
    s = vit(a)
    train(s, X, Y, a, a.epochs)
    out["from scratch"] = accuracy(s, Xt, Yt)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: ViT vs ResNet as data grows"), ("e2", "E2: patch size"), ("e3", "E3: hybrid"),
                 ("e4", "E4: higher-resolution fine-tuning"), ("e5", "E5: Figure 7 analyses"),
                 ("e6", "E6: masked patch prediction")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.D, a.layers, a.heads, a.lr, a.batch, a.epochs, a.ft_epochs, a.ssl_epochs = 384, 7, 6, 1e-3, 256, 100, 10, 50
    a.fracs = (0.05, 0.2, 0.5, 1.0)
    if a.quick:
        a.D, a.layers, a.heads, a.batch, a.epochs, a.ft_epochs, a.ssl_epochs = 128, 4, 4, 128, 3, 1, 2
        a.fracs = (0.1, 1.0)
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
