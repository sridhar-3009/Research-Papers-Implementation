"""Reproduce the findings of Simonyan & Zisserman (2015) at CIFAR-10 scale.

Training VGG on ImageNet took 2-3 weeks on 4 GPUs per network, so every claim is tested on
CIFAR-10 with VGG networks shrunk in width (channels / 4, fc 512) but with the paper's DEPTHS
and layer patterns (Table 1). 32x32 input -> five 2x2 pools -> 1x1 at the end.

  E1  Table 3: depth. Configs A, A-LRN, B, C, D, E, same training. Paper: error falls from
      A (11 layers) to D/E (16-19); LRN doesn't help; C (1x1 convs) < D (3x3 convs).
  E2  Section 4.1: B vs a "shallow" B where each pair of 3x3 convs is one 5x5 conv
      (same receptive field, fewer non-linearities). Paper: shallow is 7% worse (top-1).
  E3  Section 3.1: initialising a deep net. E from N(0, std 0.1) as written, N(0, std 0.01),
      Glorot, and "pre-initialised from a trained A". Paper: random init can stall deep nets.
  E4  Section 3.1 / Table 3: scale jittering. Train with a fixed scale vs random rescale
      (S in [32, 48], then a random 32x32 crop). Paper: jittering is clearly better.
  E5  Section 3.2 / Table 5: dense (fully convolutional) vs multi-crop vs both, on test
      images upscaled to 40x40.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick                 # small, ~15 minutes
       python3 experiments.py --only e1               # one experiment (~1-2 h on a CPU)
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from vgg import CONFIGS, VGG, dense_predict, init_from_A, to_fully_convolutional

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SMALL = dict(num_classes=10, input_size=32, fc=512, width_div=4)


def load_cifar(quick):
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    f = lambda d: (torch.tensor(d.data).permute(0, 3, 1, 2).float() / 255, torch.tensor(d.targets))
    (X, y), (Xt, yt) = f(tr), f(te)
    mean = X.mean((0, 2, 3), keepdim=True)                     # "subtracting the mean RGB value"
    X, Xt = X - mean, Xt - mean
    if quick:
        X, y, Xt, yt = X[:10000], y[:10000], Xt[:2000], yt[:2000]
    return (X, y), (Xt, yt)


def jitter_batch(xb, s_min, s_max, rng):
    """Rescale each image so its side is S ~ U[s_min, s_max], then a random 32x32 crop + flip."""
    out = []
    for im in xb:
        S = rng.randint(s_min, s_max)
        big = F.interpolate(im[None], size=(S, S), mode="bilinear", align_corners=False)[0]
        i, j = rng.randint(0, S - 32), rng.randint(0, S - 32)
        c = big[:, i:i + 32, j:j + 32]
        out.append(c.flip(-1) if rng.random() < 0.5 else c)
    return torch.stack(out)


@torch.no_grad()
def error(net, X, y, bs=1000):
    net.eval()
    wrong = sum((net(X[i:i + bs].to(DEV)).argmax(1).cpu() != y[i:i + bs]).sum().item() for i in range(0, len(X), bs))
    net.train()
    return wrong / len(X)


def train(net, data, epochs, lr=0.01, jitter=None, seed=0):
    """Section 3.1 recipe: SGD, momentum 0.9, weight decay 5e-4, dropout 0.5 in fc1/fc2,
    lr divided by 10 when the validation error stops improving. Batch 128 (paper: 256)."""
    (X, y), (Xt, yt) = data
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = net.to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)
    log = {"train_loss": [], "test_err": []}
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for s in range(0, len(X), 128):
            idx = perm[s:s + 128]
            xb = jitter_batch(X[idx], *jitter, rng) if jitter else X[idx]
            loss = F.cross_entropy(net(xb.to(DEV)), y[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        log["train_loss"].append(total / len(X))
        log["test_err"].append(error(net, Xt, yt))
        sched.step(log["test_err"][-1])
        print(f"    train loss {log['train_loss'][-1]:.3f}  test error {log['test_err'][-1]:.3f}", flush=True)
    return net, log


def e1(data, a):
    out = {}
    for name in CONFIGS:
        print(f"  E1 {name}")
        out[name] = train(VGG(name, **SMALL), data, a.epochs)[1]
    return out


class ShallowB(nn.Module):
    """Config B with each pair of 3x3 convs replaced by ONE 5x5 conv (Section 4.1)."""

    def __init__(self, width_div=4, fc=512):
        super().__init__()
        chans, layers, c_in = [64, 128, 256, 512, 512], [], 3
        for c in chans:
            c //= width_div
            layers += [nn.Conv2d(c_in, c, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_in, fc), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(fc, fc), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def e2(data, a):
    return {"B (pairs of 3x3)": train(VGG("B", **SMALL), data, a.epochs)[1],
            "shallow B (single 5x5)": train(ShallowB(), data, a.epochs)[1]}


def e3(data, a):
    out = {}
    for init in ("paper", "paper001", "glorot"):
        print(f"  E3 init {init}")
        out[init] = train(VGG("E", init=init, **SMALL), data, a.epochs)[1]
    print("  E3 pre-initialised from A")
    net_a, _ = train(VGG("A", init="paper001", **SMALL), data, a.epochs)
    e = VGG("E", init="paper001", **SMALL).to(DEV)
    init_from_A(e, net_a)
    out["from A"] = train(e, data, a.epochs)[1]
    return out


def e4(data, a):
    return {"fixed scale (32)": train(VGG("D", **SMALL), data, a.epochs, jitter=(32, 32))[1],
            "scale jitter [32, 48]": train(VGG("D", **SMALL), data, a.epochs, jitter=(32, 48))[1]}


@torch.no_grad()
def e5(data, a):
    """Train D with jitter, then evaluate test images resized to 40x40 three ways."""
    net, _ = train(VGG("D", **SMALL), data, a.epochs, jitter=(32, 48))
    net.eval()
    fcn = to_fully_convolutional(net).eval()
    Xt, yt = data[1]
    big = F.interpolate(Xt, size=(40, 40), mode="bilinear", align_corners=False)
    dense, crops = [], []
    for i in range(0, len(big), 500):
        xb = big[i:i + 500].to(DEV)
        dense.append(dense_predict(fcn, xb).cpu())
        p = 0
        for y0 in (0, 4, 8):                                          # a 3x3 grid of 32x32 crops (+ flips)
            for x0 in (0, 4, 8):
                c = xb[:, :, y0:y0 + 32, x0:x0 + 32]
                p = p + F.softmax(net(c), 1) + F.softmax(net(c.flip(-1)), 1)
        crops.append((p / 18).cpu())
    dense, crops = torch.cat(dense), torch.cat(crops)
    err = lambda p: (p.argmax(1) != yt).float().mean().item()
    return {"dense": err(dense), "multi-crop": err(crops), "multi-crop & dense": err(dense + crops)}


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}


def report(R, a):
    L = ["# Results", "", f"CIFAR-10, VGG shrunk to width/4 and fc 512, {a.epochs} epochs" + (" (QUICK run)" if a.quick else ""), ""]
    fin = lambda r: f"{100 * min(r['test_err']):.1f}%"
    paper_top1 = {"A": 29.6, "A-LRN": 29.7, "B": 28.7, "C": 28.1, "D": 27.0, "E": 27.3}
    if R.get("e1"):
        L += ["## E1: depth (Table 3)", "", "| config | weight layers | best test error (ours) | ImageNet top-1, S=256 (paper) |", "|---|---|---|---|"]
        depth = {"A": 11, "A-LRN": 11, "B": 13, "C": 16, "D": 16, "E": 19}
        L += [f"| {k} | {depth[k]} | {fin(v)} | {paper_top1[k]}% |" for k, v in R["e1"].items()] + [""]
    for key, title in (("e2", "E2: 3x3 pairs vs one 5x5 (paper: shallow 7% worse)"),
                       ("e3", "E3: initialising the 19-layer net"), ("e4", "E4: scale jittering")):
        if R.get(key):
            L += [f"## {title}", ""] + [f"- {k}: best test error {fin(v)}" for k, v in R[key].items()] + [""]
    if R.get("e5"):
        L += ["## E5: evaluation methods (Table 5)", ""] + [f"- {k}: {100 * v:.2f}%" for k, v in R["e5"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.epochs = a.epochs or (2 if a.quick else 30)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        data = load_cifar(a.quick)
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
