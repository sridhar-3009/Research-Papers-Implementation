"""Reproduce the experiments of InfoGAN (Chen et al. 2016).

  E1  Figure 1: MNIST, c ~ Cat(10); L_I over the first 1000 iterations for InfoGAN vs a regular GAN with the same
      Q network (trained but not fed back into G). Paper: InfoGAN reaches H(c) = 2.30 quickly, GAN stays near 0.
  E2  Figure 2 + Section 7.2: MNIST with c1 ~ Cat(10), c2, c3 ~ U(-1, 1), 62 noise variables (Table 1 networks).
      Saves the four manipulation grids (c1 rows; c2 and c3 from -2 to 2) for InfoGAN and c1 for a regular GAN, and
      measures c1 as an unsupervised classifier on the MNIST test set (best one-to-one matching; paper: 5% error).
  E3  Figure 5: SVHN with 4 ten-way categorical codes, 4 continuous codes, 124 noise variables (Table 2).
  E4  Figure 6: CelebA (32x32) with 10 ten-way categorical codes and 128 noise variables (Table 3).
  E5  Lambda for continuous codes: lambda_cont in {0, 0.05, 0.1, 0.5, 1} on MNIST; how strongly c2/c3 control the
      output (mean pixel change per unit of code) and the sample quality proxy (D's real/fake accuracy).
  (3D faces and 3D chairs need rendered datasets that are not freely redistributable; they are skipped.)

Settings (Appendix C): Adam, lr 2e-4 for D/Q, 1e-3 for G, batch norm, leaky ReLU 0.1 in D, lambda = 1 (discrete).

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from infogan import (LatentSpec, MNISTDiscriminatorQ, MNISTGenerator, cluster_accuracy, infogan_step, q_out_dim)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
FIG = HERE / "figures"


def mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    return tr.data.float().div(255).unsqueeze(1), te.data.float().div(255).unsqueeze(1), te.targets


def svhn():
    import torchvision
    tr = torchvision.datasets.SVHN(DATA / "svhn", split="train", download=True)
    return torch.tensor(tr.data).float().div(127.5).sub(1)                         # tanh range


def celeba32():
    import torchvision
    import torchvision.transforms as T
    ds = torchvision.datasets.CelebA(DATA, split="train", download=True,
                                     transform=T.Compose([T.CenterCrop(140), T.Resize(32), T.ToTensor()]))
    return torch.stack([ds[i][0] for i in range(len(ds))]).mul(2).sub(1)


class ColorGenerator(nn.Module):
    """Tables 2-3: FC 2x2x448 -> upconv 256 -> 128 -> 64 -> 3 (tanh), 32x32 output."""

    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 2 * 2 * 448), nn.BatchNorm1d(2 * 2 * 448), nn.ReLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(448, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                                nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, v):
        return self.up(self.fc(v).view(-1, 448, 2, 2))


class ColorDiscriminatorQ(nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.1),
                                  nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
                                  nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.Flatten())
        self.d = nn.Linear(256 * 4 * 4, 1)
        self.q = nn.Sequential(nn.Linear(256 * 4 * 4, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Linear(128, q_dim))

    def forward(self, x):
        h = self.body(x)
        return self.d(h).squeeze(-1), self.q(h)


def train(G, DQ, X, spec, a, iters, info=True, lam_cont=1.0, log_every=None):
    og = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))
    od = torch.optim.Adam(DQ.parameters(), 2e-4, betas=(0.5, 0.999))
    curve = []
    for it in range(1, iters + 1):
        real = X[torch.randint(0, len(X), (a.batch,))]
        _, _, li = infogan_step(G, DQ, og, od, real, spec, lam_cat=1.0, lam_cont=lam_cont, info=info)
        if log_every and it % log_every == 0:
            curve.append({"iter": it, "L_I": li})
    return curve


def grid(x, rows):
    B, C, H, W = x.shape
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x.view(rows, -1, C, H, W).permute(0, 3, 1, 4, 2).reshape(rows * H, -1, C).squeeze(-1)


def save(img, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.imsave(FIG / name, img.numpy(), cmap="gray" if img.dim() == 2 else None)


@torch.no_grad()
def manipulate(G, spec, which, values, rows=5):
    """Figure 2 convention: one code varies left to right, everything else fixed per row."""
    G.eval()
    out = []
    for r in range(rows):
        z, cat, cont = spec.sample(1)
        for v in values:
            c2, k2 = cat.clone(), cont.clone()
            if which == "cat":
                c2[0, 0] = int(v)
            else:
                k2[0, which] = v
            out.append(G(spec.pack(z, c2, k2)))
    G.train()
    return torch.cat(out)


def e1(a):
    X, _, _ = mnist()
    spec = LatentSpec(62, (10,), 0)
    out = {}
    for info in (True, False):
        torch.manual_seed(0)
        G, DQ = MNISTGenerator(spec.dim), MNISTDiscriminatorQ(q_out_dim(spec))
        out["InfoGAN" if info else "GAN"] = train(G, DQ, X, spec, a, a.iters_fig1, info, log_every=a.iters_fig1 // 20)
    out["H(c)"] = math.log(10)
    return out


def e2(a):
    X, Xt, yt = mnist()
    spec = LatentSpec(62, (10,), 2)
    out = {}
    for info in (True, False):
        torch.manual_seed(0)
        G, DQ = MNISTGenerator(spec.dim), MNISTDiscriminatorQ(q_out_dim(spec))
        train(G, DQ, X, spec, a, a.iters, info, lam_cont=a.lam_cont)
        tag = "infogan" if info else "gan"
        save(grid(manipulate(G, spec, "cat", range(10)), 5), f"e2_{tag}_c1.png")
        if info:
            for j in (0, 1):
                save(grid(manipulate(G, spec, j, torch.linspace(-2, 2, 10).tolist()), 5), f"e2_infogan_c{j + 2}.png")
        DQ.eval()
        with torch.no_grad():
            pred = torch.cat([DQ(Xt[k:k + 500])[1][:, :10].argmax(1) for k in range(0, len(Xt), 500)])
        DQ.train()
        out[tag] = {"c1 as classifier: accuracy": cluster_accuracy(pred, yt, 10)}
        print("  E2", tag, out[tag], flush=True)
    out["paper"] = "InfoGAN c1: 5% error (95% accuracy)"
    return out


def e3(a):
    X = svhn()
    spec = LatentSpec(124, (10, 10, 10, 10), 4)
    torch.manual_seed(0)
    G, DQ = ColorGenerator(spec.dim), ColorDiscriminatorQ(q_out_dim(spec))
    train(G, DQ, X, spec, a, a.iters, lam_cont=a.lam_cont)
    save(grid(manipulate(G, spec, "cat", range(10)), 5), "e3_svhn_c1.png")
    save(grid(manipulate(G, spec, 0, torch.linspace(-1, 1, 10).tolist()), 5), "e3_svhn_cont1.png")
    return {"figures": ["figures/e3_svhn_c1.png", "figures/e3_svhn_cont1.png"]}


def e4(a):
    X = celeba32()
    spec = LatentSpec(128, (10,) * 10, 0)
    torch.manual_seed(0)
    G, DQ = ColorGenerator(spec.dim), ColorDiscriminatorQ(q_out_dim(spec))
    train(G, DQ, X, spec, a, a.iters)
    figs = []
    for k in range(10):                                                           # one figure per categorical code
        G.eval()
        with torch.no_grad():
            rows = []
            for r in range(5):
                z, cat, cont = spec.sample(1)
                for v in range(10):
                    c2 = cat.clone(); c2[0, k] = v
                    rows.append(G(spec.pack(z, c2, cont)))
        G.train()
        save(grid(torch.cat(rows), 5), f"e4_celeba_code{k}.png")
        figs.append(f"figures/e4_celeba_code{k}.png")
    return {"figures": figs}


def e5(a):
    X, _, _ = mnist()
    spec = LatentSpec(62, (10,), 2)
    out = {}
    for lam in (0.0, 0.05, 0.1, 0.5, 1.0):
        torch.manual_seed(0)
        G, DQ = MNISTGenerator(spec.dim), MNISTDiscriminatorQ(q_out_dim(spec))
        train(G, DQ, X, spec, a, a.iters // 2, lam_cont=lam)
        G.eval(); DQ.eval()
        with torch.no_grad():
            z, cat, cont = spec.sample(500)
            lo, hi = cont.clone(), cont.clone()
            lo[:, 0], hi[:, 0] = -1, 1
            effect = (G(spec.pack(z, cat, hi)) - G(spec.pack(z, cat, lo))).abs().mean().item() / 2
            d_real = (DQ(X[:500])[0] > 0).float().mean().item()
            d_fake = (DQ(G(spec.pack(z, cat, cont)))[0] < 0).float().mean().item()
        out[f"lambda_cont={lam}"] = {"mean |dx| per unit of c2": effect, "D accuracy (real, fake)": [d_real, d_fake]}
        print("  E5", lam, out[f"lambda_cont={lam}"], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Figure 1, L_I curves"), ("e2", "E2: Figure 2, MNIST codes"), ("e3", "E3: SVHN"),
                 ("e4", "E4: CelebA"), ("e5", "E5: lambda for continuous codes")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.batch, a.iters, a.iters_fig1, a.lam_cont = 128, 30000, 1000, 0.1
    if a.quick:
        a.iters, a.iters_fig1 = 300, 100
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
