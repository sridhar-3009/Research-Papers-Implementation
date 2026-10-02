"""Reproduce the experiments of Glow (Kingma & Dhariwal 2018), scaled to one machine.

  E1  Figure 3: CIFAR-10, K = 32, L = 3; channel reversal vs fixed shuffle vs invertible 1x1 convolution, each with
      additive and affine coupling, 3 seeds; test bits/dim over training (and wall-clock per epoch).
  E2  Table 2 (CIFAR-10 column): the best model's bits/dim. Paper: Glow 3.35 vs RealNVP 3.49.
  E3  Table 3: 5-bit CIFAR-10. Paper: 1.67 bits/dim.
  E4  Figure 8: samples at temperatures 0, 0.25, 0.6, 0.7, 0.8, 0.9, 1.0.
  E5  Figures 5-6: CelebA at 64x64, 5-bit (smaller than the paper's 256x256, L = 6): interpolation between encoded
      real images, and attribute manipulation along z_pos - z_neg for Smiling, Blond hair, Young, Male, Pale skin,
      Narrow eyes (directions computed from labels AFTER training, as in the paper).
  E6  LU-decomposed vs plain 1x1 convolution: wall-clock per step and bits/dim.
  ImageNet 32/64 and LSUN (the other Table 2 columns) need large downloads and are not included.

Settings (Appendix C): Adam lr 1e-3, coupling NN with 512 hidden channels, batch 512 for CIFAR-10 (64 here by
default to fit one GPU), affine coupling for the main results.

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

from glow import Glow, bits_per_dim, dequantize

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
FIG = HERE / "figures"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATTRS = ["Smiling", "Blond_Hair", "Young", "Male", "Pale_Skin", "Narrow_Eyes"]


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
        return torch.cat(xs).to(torch.uint8)
    return load([f"data_batch_{i}" for i in range(1, 6)]), load(["test_batch"])


def celeba64():
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.CenterCrop(148), T.Resize(64), T.PILToTensor()])
    ds = torchvision.datasets.CelebA(DATA, split="train", target_type="attr", download=True, transform=tf)
    idx = {n: i for i, n in enumerate(ds.attr_names)}
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    A = ds.attr
    return X, {a: A[:, idx[a]] for a in ATTRS}


def train(model, X, Xt, a, n_bits=8, epochs=None, log=True):
    model.to(DEV)
    with torch.no_grad():                                                          # actnorm data-dependent init
        model.log_prob(dequantize(X[:a.init_batch], n_bits)[0].to(DEV))
    opt = torch.optim.Adam(model.parameters(), a.lr)
    curve = []
    for ep in range(epochs or a.epochs):
        t0 = time.time()
        model.train()
        for idx in torch.randperm(len(X)).split(a.batch):
            x, c = dequantize(X[idx], n_bits)
            loss = bits_per_dim(model.log_prob(x.to(DEV)), c, x[0].numel()).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
            opt.step()
        curve.append({"epoch": ep + 1, "test bits/dim": evaluate(model, Xt, a, n_bits), "seconds": time.time() - t0})
        if log:
            print(f"    {curve[-1]}", flush=True)
    return curve


@torch.no_grad()
def evaluate(model, X, a, n_bits=8):
    model.eval()
    tot = 0.0
    for k in range(0, len(X), a.batch):
        x, c = dequantize(X[k:k + a.batch], n_bits)
        tot += bits_per_dim(model.log_prob(x.to(DEV)), c, x[0].numel()).sum().item()
    return tot / len(X)


def save_grid(x, rows, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    x = (x.clamp(-0.5, 0.5) + 0.5).cpu()
    B, C, H, W = x.shape
    img = x.view(rows, -1, C, H, W).permute(0, 3, 1, 4, 2).reshape(rows * H, -1, C)
    plt.imsave(FIG / name, img.numpy())


def make(a, perm="conv", additive=False, C=3, L=3):
    return Glow(C=C, K=a.K, L=L, hidden=a.hidden, perm=perm, additive=additive)


def e1(a):
    X, Xt = cifar10()
    out = {}
    for additive in (True, False):
        for perm in ("reverse", "shuffle", "conv"):
            runs = []
            for seed in range(a.seeds):
                torch.manual_seed(seed)
                runs.append(train(make(a, perm, additive), X, Xt[:a.n_test], a))
            name = f"{'additive' if additive else 'affine'} | {perm}"
            out[name] = runs
            print("  E1", name, [r[-1]["test bits/dim"] for r in runs], flush=True)
    return out


def e2(a):
    X, Xt = cifar10()
    torch.manual_seed(0)
    m = make(a)
    curve = train(m, X, Xt[:a.n_test], a, epochs=a.epochs_best)
    torch.save(m.state_dict(), HERE / "glow_cifar.pt")
    return {"final test bits/dim": curve[-1]["test bits/dim"], "paper": {"Glow": 3.35, "RealNVP": 3.49}}


def e3(a):
    X, Xt = cifar10()
    torch.manual_seed(0)
    curve = train(make(a), X, Xt[:a.n_test], a, n_bits=5)
    return {"final test bits/dim (5-bit)": curve[-1]["test bits/dim"], "paper": 1.67}


def e4(a):
    X, Xt = cifar10()
    m = make(a).to(DEV)
    if (HERE / "glow_cifar.pt").exists():
        with torch.no_grad():
            m.log_prob(dequantize(X[:16])[0].to(DEV))
        m.load_state_dict(torch.load(HERE / "glow_cifar.pt", map_location=DEV))
    else:
        train(m, X, Xt[:a.n_test], a, epochs=1)
    m.eval()
    rows = []
    torch.manual_seed(0)
    with torch.no_grad():
        for T in (0.0, 0.25, 0.6, 0.7, 0.8, 0.9, 1.0):
            rows.append(m.decode(n=8, shape=m.top_shape(3, 32, 32), temperature=T))
    save_grid(torch.stack(rows, 1).flatten(0, 1), 8, "e4_temperature.png")
    return {"figure": "figures/e4_temperature.png", "columns": "T = 0, 0.25, 0.6, 0.7, 0.8, 0.9, 1.0"}


def e5(a):
    X, attrs = celeba64()
    n_test = 1000
    Xtr, Xte = X[:-n_test], X[-n_test:]
    torch.manual_seed(0)
    m = Glow(C=3, K=a.K, L=4, hidden=a.hidden)
    train(m, Xtr, Xte[:a.n_test], a, n_bits=5, epochs=a.epochs_celeba)
    m.eval()
    out = {}
    with torch.no_grad():
        def enc(x):
            return [z.cpu() for z in m.encode(dequantize(x, 5)[0].to(DEV))[0]]
        zs = [enc(X[k:k + 256]) for k in range(0, a.n_attr, 256)]
        zs = [torch.cat([b[i] for b in zs]) for i in range(len(zs[0]))]
        base = enc(Xte[:5])
        a0, a1 = enc(Xte[5:6]), enc(Xte[6:7])                                      # Figure 5: interpolation
        line = [m.decode([((1 - t) * u + t * v).to(DEV) for u, v in zip(a0, a1)]) for t in torch.linspace(0, 1, 8)]
        save_grid(torch.cat(line), 1, "e5_interpolation.png")
        for attr in ATTRS:                                                         # Figure 6: manipulation
            lab = attrs[attr][:a.n_attr] == 1
            direction = [z[lab].mean(0) - z[~lab].mean(0) for z in zs]
            row = [m.decode([(b + alpha * d).to(DEV) for b, d in zip(base, direction)]) for alpha in (-1, -0.5, 0, 0.5, 1)]
            save_grid(torch.stack(row, 1).flatten(0, 1), 5, f"e5_{attr}.png")
            out[attr] = f"figures/e5_{attr}.png"
    out["interpolation"] = "figures/e5_interpolation.png"
    return out


def e6(a):
    X, Xt = cifar10()
    out = {}
    for perm in ("conv", "lu"):
        torch.manual_seed(0)
        curve = train(make(a, perm), X, Xt[:a.n_test], a, epochs=a.epochs_lu)
        out[perm] = {"final test bits/dim": curve[-1]["test bits/dim"],
                     "seconds per epoch": sum(c["seconds"] for c in curve) / len(curve)}
        print("  E6", perm, out[perm], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Figure 3, permutations"), ("e2", "E2: Table 2, CIFAR-10"), ("e3", "E3: 5-bit CIFAR-10"),
                 ("e4", "E4: temperature"), ("e5", "E5: CelebA interpolation and attributes"), ("e6", "E6: LU")):
        if R.get(k):
            body = R[k]
            if k == "e1":
                body = {n: [r[-1]["test bits/dim"] for r in runs] for n, runs in body.items()}
            L += [f"## {t}", "", "```", json.dumps(body, indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.lr, a.batch, a.init_batch, a.K, a.hidden = 1e-3, 64, 512, 32, 512
    a.epochs, a.epochs_best, a.epochs_celeba, a.epochs_lu, a.seeds = 100, 1000, 50, 20, 3
    a.n_test, a.n_attr = 10000, 30000
    if a.quick:
        a.K, a.hidden, a.init_batch = 4, 64, 64
        a.epochs, a.epochs_best, a.epochs_celeba, a.epochs_lu, a.seeds = 1, 1, 1, 1, 1
        a.n_test, a.n_attr = 256, 512
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
