"""Reproduce the experiments of PixelCNN++ (Salimans et al. 2017).

  E1  Table 1: unconditional CIFAR-10 (5 gated ResNet layers per block, 160-192 filters, dropout 0.5, K = 10
      logistics). Paper: 2.92 bits per sub-pixel.
  E2  Section 3.2: class-conditional CIFAR-10 (one-hot class -> bias in every gated layer). Paper: 2.94.
  E3  Table 2 (simplified): 'Small PixelCNN' without downsampling, depth chosen to limit the receptive field; we
      measure the field with autograd and report bits/dim. (The NIN / autoregressive-channel variants are not
      implemented.) Paper: 3.03-3.11 for 11x5 / 15x8 fields.
  E4  Ablations (Section 3.4): 256-way softmax head (1536 outputs per pixel, logits linear in earlier sub-pixels)
      vs the logistic mixture (Figure 6); dequantized continuous mixture (paper: 3.11 vs 2.92); no short-cut
      connections (Figure 7: fails to train); no dropout (Section 3.4.4: train < 2.0, test > 6.0 bits).
  E5  Laptop-sized version of the short-cut ablation: MNIST resized to 12x12 (1 channel). In a single run during
      development (300 Adam steps, batch 64, 32 filters, K = 5) this gave 2.31 bits/dim with short-cuts and 2.76
      without; this function repeats it over seeds.

!! HEAVY: the paper trains for days on 8 GPUs. Not run on the author's laptop (except the one E5 run noted above).
       python3 experiments.py --quick
       python3 experiments.py --only e5
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

from pixelcnnpp import (PixelCNNpp, bits_per_dim, dequantized_log_density, discretized_mix_logistic_log_prob,
                        softmax_log_prob, to_pm1)

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

    def load(names):
        xs, ys = [], []
        for nm in names:
            with open(d / "cifar-10-batches-py" / nm, "rb") as f:
                b = pickle.load(f, encoding="bytes")
            xs.append(torch.tensor(b[b"data"]).view(-1, 3, 32, 32)); ys += b[b"labels"]
        return to_pm1(torch.cat(xs).long()), torch.tensor(ys)
    return load([f"data_batch_{i}" for i in range(1, 6)]), load(["test_batch"])


def mnist12():
    import torchvision
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    f = lambda d: to_pm1(F.interpolate(d.data.float().unsqueeze(1), size=(12, 12), mode="area").round().clamp(0, 255).long())
    return f(tr), f(te)


class SoftmaxHead(nn.Module):
    """Section 3.4.1: per pixel, 3 x 256 base logits + 256 coefficients for each dependency (r->g, r->b, g->b):
    6 x 256 = 1536 outputs. Logits of a later channel = base + coefficient * (observed earlier value)."""

    def __init__(self, nf=160):
        super().__init__()
        self.out = nn.Conv2d(nf, 6 * 256, 1)

    def log_prob(self, h, x):
        B, _, H, W = h.shape
        o = self.out(F.elu(h)).view(B, 6, 256, H, W)
        r, g = x[:, 0:1].unsqueeze(2), x[:, 1:2].unsqueeze(2)
        logits = torch.stack([o[:, 0], o[:, 1] + o[:, 3] * r[:, 0], o[:, 2] + o[:, 4] * r[:, 0] + o[:, 5] * g[:, 0]], 1)
        return softmax_log_prob(x, logits).flatten(1).sum(1)


class SoftmaxPixelCNN(PixelCNNpp):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.head = SoftmaxHead(self.out.in_channels)

    def log_prob(self, x, h=None):
        feats = {}
        hook = self.out.register_forward_hook(lambda m, i, o: feats.__setitem__("h", i[0]))
        self(x, h)
        hook.remove()
        return self.head.log_prob(feats["h"], x)


def train(model, X, a, epochs, Y=None, mode="mixture", Xt=None, Yt=None, n_classes=0):
    """Adam (lr 1e-3, decay 0.999995 per step as in the released code). mode: 'mixture' | 'dequantized'."""
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, a.lr_decay)
    curve = []
    D = X[0].numel()
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(X)).split(a.batch):
            x = X[idx].to(DEV)
            h = F.one_hot(Y[idx], n_classes).float().to(DEV) if n_classes else None
            if mode == "dequantized":
                xn = x + (torch.rand_like(x) - 0.5) * (2 / 255)
                lp = dequantized_log_density(xn, model(xn, h), model.K).flatten(1).sum(1)
            else:
                lp = model.log_prob(x, h)
            loss = -lp.mean() / D
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        curve.append({"epoch": ep + 1, "train bits/dim": bits_per_dim(-loss.item() * D, D),
                      "test bits/dim": evaluate(model, Xt, a, Yt, n_classes, mode) if Xt is not None else None})
        print(f"    epoch {ep + 1}: {curve[-1]}", flush=True)
    return curve


@torch.no_grad()
def evaluate(model, X, a, Y=None, n_classes=0, mode="mixture"):
    """Test bits/dim; for the dequantized model, the variational bound with one noise draw per image (as reported
    in Section 3.4.2)."""
    model.eval()
    tot, D = 0.0, X[0].numel()
    for k in range(0, len(X), a.batch):
        x = X[k:k + a.batch].to(DEV)
        h = F.one_hot(Y[k:k + a.batch], n_classes).float().to(DEV) if n_classes else None
        if mode == "dequantized":
            xn = x + (torch.rand_like(x) - 0.5) * (2 / 255)
            tot += dequantized_log_density(xn, model(xn, h), model.K).sum().item()
        else:
            tot += model.log_prob(x, h).sum().item()
    return bits_per_dim(tot / len(X), D)


def big(a, **kw):
    return PixelCNNpp(3, a.nr_resnet, a.nf, 10, a.dropout, **kw)


def e1(a):
    (X, _), (Xt, _) = cifar10()
    torch.manual_seed(0)
    curve = train(big(a), X, a, a.epochs, Xt=Xt[:a.n_test])
    return {"curve": curve, "paper": 2.92}


def e2(a):
    (X, Y), (Xt, Yt) = cifar10()
    torch.manual_seed(0)
    curve = train(big(a, n_classes=10), X, a, a.epochs, Y=Y, Xt=Xt[:a.n_test], Yt=Yt[:a.n_test], n_classes=10)
    return {"curve": curve, "paper": 2.94}


def receptive_field_size(model, size=32):
    x = torch.zeros(1, model.C, size, size, requires_grad=True)
    model.eval()
    model(x)[0, :, size // 2, size // 2].sum().backward()
    seen = (x.grad.abs().sum(1)[0] > 0).nonzero()
    rows = (size // 2 - seen[:, 0].min()).item()
    cols = (seen[:, 1].max() - seen[:, 1].min() + 1).item()
    return f"{cols}x{rows}"


def e3(a):
    (X, _), (Xt, _) = cifar10()
    out = {}
    for nr in (1, 2):
        torch.manual_seed(0)
        m = PixelCNNpp(3, nr, a.nf, 10, a.dropout, downsample=False)
        field = receptive_field_size(m)
        out[f"no downsampling, {nr} layer(s) per block (field {field})"] = train(m, X, a, a.epochs, Xt=Xt[:a.n_test])[-1]
    out["paper"] = {"11x5 plain": 3.11, "15x8 plain": 3.07, "15x8 autoregressive channel": 3.03}
    return out


def e4(a):
    (X, _), (Xt, _) = cifar10()
    out = {}
    for name, make, mode in (("logistic mixture (baseline)", lambda: big(a), "mixture"),
                             ("256-way softmax", lambda: SoftmaxPixelCNN(C=3, nr_resnet=a.nr_resnet, nf=a.nf, K=10,
                                                                         dropout=a.dropout), "mixture"),
                             ("dequantized continuous mixture", lambda: big(a), "dequantized"),
                             ("no short-cuts", lambda: big(a, shortcuts=False), "mixture"),
                             ("no dropout", lambda: PixelCNNpp(3, a.nr_resnet, a.nf, 10, 0.0), "mixture")):
        torch.manual_seed(0)
        out[name] = train(make(), X, a, a.epochs_ablation, mode=mode, Xt=Xt[:a.n_test])
        print("  E4", name, out[name][-1], flush=True)
    out["paper"] = {"dequantized": 3.11, "baseline": 2.92, "no dropout": "train < 2.0, test > 6.0"}
    return out


def e5(a):
    X, Xt = mnist12()
    out = {}
    for sc in (True, False):
        res = []
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = PixelCNNpp(1, 1, 32, 5, 0.0, shortcuts=sc)
            opt = torch.optim.Adam(m.parameters(), 2e-3)
            for _ in range(a.small_steps):
                x = X[torch.randint(0, len(X), (64,))]
                loss = -m.log_prob(x).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            res.append(evaluate(m, Xt[:1000], argparse.Namespace(batch=100)))
        out[f"shortcuts={sc}"] = {"bits/dim per seed": res, "mean": sum(res) / len(res)}
        print("  E5", sc, out[f"shortcuts={sc}"], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Table 1"), ("e2", "E2: class-conditional"), ("e3", "E3: small receptive fields"),
                 ("e4", "E4: ablations"), ("e5", "E5: short-cuts on 12x12 MNIST")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.lr, a.lr_decay, a.batch, a.nr_resnet, a.nf, a.dropout = 1e-3, 0.999995, 16, 5, 160, 0.5
    a.epochs, a.epochs_ablation, a.n_test = 500, 100, 10000
    a.seeds, a.small_steps = 5, 2000
    if a.quick:
        a.nr_resnet, a.nf, a.epochs, a.epochs_ablation, a.n_test = 1, 32, 1, 1, 200
        a.seeds, a.small_steps = 1, 300
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
