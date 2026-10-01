"""Reproduce the MNIST experiments of Szegedy et al. (2013).

  E1  Table 1: train FC10(1e-4), FC10(1e-2), FC10(1), FC100-100-10, FC200-200-10, AE400-10 and
      measure the average minimal distortion needed to make EVERY attacked example wrong.
      Paper: 0.062, 0.1, 0.14, 0.058, 0.065, 0.086.
  E2  Table 2: cross-model generalization. Adversarial examples made for model A, fed to model B.
      Plus Gaussian-noise rows (stddev 0.1, 0.3) as a control.
  E3  Tables 3-4: cross-training-set generalization. FC100-100-10 and FC123-456-10 trained on half P1,
      FC100-100-10' on the other half P2; attack test images; also "amplified to stddev 0.1".
  E4  Section 3 (Figures 1-2): images that most excite single hidden units vs random directions.
  E5  Section 4.3 (Table 5): upper Lipschitz bound of each layer of the trained nets.
      --alexnet also downloads torchvision's pretrained AlexNet (~233 MB) and bounds its layers.
  E6  Section 4.2: training 100-100-10 with a pool of adversarial examples mixed in
      (input level only; the paper also attacked hidden layers). Paper: below 1.2% test error,
      vs 1.6% with weight decay alone.

Each attack is one L-BFGS-B run per value of c (~10 per example), so attacking thousands of
examples takes hours. --n-attack sets how many examples to attack (paper: whole sets).

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # 100 attacked examples, ~20-40 minutes
       python3 experiments.py --only e1
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

from adversarial import (AE400, amplify, conv_operator_norm_power, decay_penalty, distortion, fc_net,
                         fc_operator_norm, gaussian_distort, minimal_adversarial, random_direction, top_activating,
                         unit_direction)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"


def load_mnist():
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    f = lambda d: (d.data.float().view(-1, 784) / 255, d.targets)            # pixels in [0, 1]
    return f(tr), f(te)


def train(model, X, y, lambdas, epochs, lr=0.1, seed=0):
    """Plain SGD on cross-entropy + the paper's weight decay."""
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 100):
            idx = perm[i:i + 100]
            loss = F.cross_entropy(model(X[idx]), y[idx]) + decay_penalty(model, lambdas)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def train_ae400(X, y, epochs, seed=0):
    """Sparse autoencoder (sigmoid, KL sparsity towards mean activation 0.05), then a softmax on
    the frozen 400 features."""
    torch.manual_seed(seed)
    ae = AE400()
    opt = torch.optim.Adam(list(ae.enc.parameters()) + list(ae.dec.parameters()), 1e-3)
    for _ in range(epochs):
        for i in range(0, len(X), 100):
            xb = X[i:i + 100]
            h = ae.encode(xb)
            rho, rho_hat = 0.05, h.mean(0).clamp(1e-4, 1 - 1e-4)
            kl = (rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))).sum()
            loss = F.binary_cross_entropy(ae.reconstruct(xb), xb) + 1e-3 * kl
            opt.zero_grad(); loss.backward(); opt.step()
    opt = torch.optim.SGD(ae.cls.parameters(), lr=0.1, momentum=0.9)
    for _ in range(epochs):
        for i in range(0, len(X), 100):
            loss = F.cross_entropy(ae(X[i:i + 100]), y[i:i + 100]) + 1e-6 * (ae.cls.weight ** 2).sum() / 10
            opt.zero_grad(); loss.backward(); opt.step()
    return ae.eval()


def make_models(X, y, epochs):
    return {
        "FC10(1e-4)": train(fc_net(()), X, y, [1e-4], epochs),
        "FC10(1e-2)": train(fc_net(()), X, y, [1e-2], epochs),
        "FC10(1)": train(fc_net(()), X, y, [1.0], epochs),
        "FC100-100-10": train(fc_net((100, 100)), X, y, [1e-5, 1e-5, 1e-6], epochs),
        "FC200-200-10": train(fc_net((200, 200)), X, y, [1e-5, 1e-5, 1e-6], epochs),
        "AE400-10": train_ae400(X, y, epochs),
    }


@torch.no_grad()
def err(model, X, y):
    return (model(X).argmax(1) != y).float().mean().item()


def attack_set(model, X, y, rng):
    """For each example: a random wrong target label, then the minimal-distortion attack."""
    advs, dists = [], []
    for x, label in zip(X, y):
        target = rng.choice([t for t in range(10) if t != int(label)])
        z, _ = minimal_adversarial(model, x, target)
        z = x if z is None else z
        advs.append(z); dists.append(distortion(x, z))
    return torch.stack(advs), sum(dists) / len(dists)


def e1_e2(data, a):
    (X, y), _ = data
    models = make_models(X, y, a.epochs)
    rng = random.Random(0)
    Xa, ya = X[:a.n_attack], y[:a.n_attack]
    advs, out = {}, {"models": {}, "transfer": {}}
    for name, m in models.items():
        advs[name], d = attack_set(m, Xa, ya, rng)
        out["models"][name] = dict(train_err=err(m, X, y), test_err=err(m, *data[1]), distortion=d)
        print(f"  {name}: train {out['models'][name]['train_err']:.3f} test {out['models'][name]['test_err']:.3f} "
              f"min distortion {d:.3f}", flush=True)
    g = torch.Generator().manual_seed(0)
    rows = dict(advs)
    rows["Gaussian noise, stddev=0.1"] = gaussian_distort(Xa, 0.1, g)
    rows["Gaussian noise, stddev=0.3"] = gaussian_distort(Xa, 0.3, g)
    for src, Xadv in rows.items():
        out["transfer"][src] = {dst: err(m, Xadv, ya) for dst, m in models.items()}
    torch.save({k: v.state_dict() for k, v in models.items()}, HERE / "models.pt")
    return out


def e3(data, a):
    (X, y), (Xt, yt) = data
    P1, P2 = (X[:30000], y[:30000]), (X[30000:], y[30000:])
    models = {"FC100-100-10 (P1)": train(fc_net((100, 100)), *P1, [1e-5, 1e-5, 1e-6], a.epochs),
              "FC123-456-10 (P1)": train(fc_net((123, 456)), *P1, [1e-5, 1e-5, 1e-6], a.epochs),
              "FC100-100-10' (P2)": train(fc_net((100, 100)), *P2, [1e-5, 1e-5, 1e-6], a.epochs, seed=1)}
    rng = random.Random(1)
    Xa, ya = Xt[:a.n_attack], yt[:a.n_attack]
    out = {"baseline": {n: dict(P1=err(m, *P1), P2=err(m, *P2), test=err(m, Xt, yt)) for n, m in models.items()},
           "transfer": {}, "amplified": {}}
    for src, m in models.items():
        Xadv, d = attack_set(m, Xa, ya, rng)
        out["transfer"][f"{src} (stddev {d:.3f})"] = {n: err(mm, Xadv, ya) for n, mm in models.items()}
        Xamp = torch.stack([amplify(x, z, 0.1) for x, z in zip(Xa, Xadv)])
        out["amplified"][src] = {n: err(mm, Xamp, ya) for n, mm in models.items()}
    g = torch.Generator().manual_seed(0)
    for s in (0.06, 0.1):
        out["transfer"][f"Gaussian noise {s}"] = {n: err(mm, gaussian_distort(Xa, s, g), ya) for n, mm in models.items()}
    return out


def e4(data, a):
    """Save image grids: top-8 test images for 4 single units and 4 random directions of the first
    hidden layer of a trained FC100-100-10."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (X, y), (Xt, _) = data
    m = train(fc_net((100, 100)), X, y, [1e-5, 1e-5, 1e-6], a.epochs)
    with torch.no_grad():
        feats = m[2](m[1](Xt))                                  # sigmoid(W1 x + b1): 100 hidden units
    g = torch.Generator().manual_seed(0)
    rows = [("unit %d" % i, unit_direction(100, i)) for i in (0, 1, 2, 3)]
    rows += [("random %d" % i, random_direction(100, g)) for i in range(4)]
    fig, axes = plt.subplots(8, 8, figsize=(8, 8.5))
    for r, (name, v) in enumerate(rows):
        for c, idx in enumerate(top_activating(feats, v, 8)):
            axes[r, c].imshow(Xt[idx].view(28, 28), cmap="gray"); axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=7, loc="left")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "e4_units_vs_random.png", dpi=110); plt.close(fig)
    return {"figure": "figures/e4_units_vs_random.png"}


def e5(data, a):
    out = {}
    path = HERE / "models.pt"
    if path.exists():
        states = torch.load(path)
        for name in ("FC100-100-10", "FC200-200-10"):
            hidden = (100, 100) if "100" in name else (200, 200)
            m = fc_net(hidden); m.load_state_dict(states[name])
            out[name] = [fc_operator_norm(l.weight) for l in m if isinstance(l, nn.Linear)]
    if a.alexnet:
        net = torchvision.models.alexnet(weights="DEFAULT").eval()
        sizes = [224, 27, 13, 13, 13]
        convs = [l for l in net.features if isinstance(l, nn.Conv2d)]
        bounds = [conv_operator_norm_power(c.weight, n, c.stride[0], iters=100) for c, n in zip(convs, sizes)]
        bounds += [fc_operator_norm(l.weight) for l in net.classifier if isinstance(l, nn.Linear)]
        out["AlexNet (torchvision)"] = bounds
    return out


def e6(data, a):
    """Adversarial training with a pool (input level): every few batches, regenerate adversarial
    examples for a random subset of training images and mix them into the batches."""
    (X, y), (Xt, yt) = data
    torch.manual_seed(0)
    rng = random.Random(0)
    m = fc_net((100, 100))
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    pool_x, pool_y = X[:0], y[:0]
    for ep in range(a.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 100):
            idx = perm[i:i + 100]
            xb, yb = X[idx], y[idx]
            if len(pool_x):
                j = torch.randint(0, len(pool_x), (50,))
                xb, yb = torch.cat([xb, pool_x[j]]), torch.cat([yb, pool_y[j]])
            loss = F.cross_entropy(m(xb), yb) + decay_penalty(m, [1e-5, 1e-5, 1e-6])
            opt.zero_grad(); loss.backward(); opt.step()
            if i % (100 * a.refresh) == 0:                      # refresh part of the pool
                m.eval()
                k = torch.randint(0, len(X), (a.pool_add,))
                new, _ = attack_set(m, X[k], y[k], rng)
                pool_x, pool_y = torch.cat([pool_x, new])[-a.pool_size:], torch.cat([pool_y, y[k]])[-a.pool_size:]
                m.train()
        print(f"  epoch {ep + 1}: test error {err(m.eval(), Xt, yt):.4f} (pool {len(pool_x)})", flush=True)
        m.train()
    base = train(fc_net((100, 100)), X, y, [1e-5, 1e-5, 1e-6], a.epochs)
    return {"adversarial training": err(m.eval(), Xt, yt), "weight decay only": err(base, Xt, yt)}


EXPS = {"e1": e1_e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}


def report(R, a):
    L = ["# Results", "", f"{a.epochs} epochs, {a.n_attack} attacked examples" + (" (QUICK run)" if a.quick else ""), ""]
    paper = {"FC10(1e-4)": 0.062, "FC10(1e-2)": 0.1, "FC10(1)": 0.14, "FC100-100-10": 0.058, "FC200-200-10": 0.065, "AE400-10": 0.086}
    if R.get("e1"):
        L += ["## E1: Table 1", "", "| model | train err | test err | min distortion | paper |", "|---|---|---|---|---|"]
        L += [f"| {k} | {100 * v['train_err']:.1f}% | {100 * v['test_err']:.1f}% | {v['distortion']:.3f} | {paper[k]} |"
              for k, v in R["e1"]["models"].items()]
        names = list(R["e1"]["models"])
        L += ["", "## E2: Table 2 (error of column model on examples made for row model)", "",
              "| made for \\ fed to | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        L += [f"| {src} | " + " | ".join(f"{100 * row[n]:.1f}%" for n in names) + " |" for src, row in R["e1"]["transfer"].items()]
        L += [""]
    if R.get("e3"):
        L += ["## E3: Tables 3-4 (cross training set)", ""]
        for part in ("transfer", "amplified"):
            for src, row in R["e3"][part].items():
                L.append(f"- {part}: {src}: " + ", ".join(f"{n} {100 * v:.1f}%" for n, v in row.items()))
        L += [""]
    if R.get("e4"):
        L += ["## E4: units vs random directions", "", f"![]({R['e4']['figure']})", ""]
    if R.get("e5"):
        L += ["## E5: per-layer upper Lipschitz bounds (paper's AlexNet: 2.75, 10, 7, 7.5, 11, 3.12, 4, 4)", ""]
        L += [f"- {k}: " + ", ".join(f"{b:.2f}" for b in v) for k, v in R["e5"].items()] + [""]
    if R.get("e6"):
        L += ["## E6: adversarial training (paper: <1.2% vs 1.6%)", ""] + [f"- {k}: {100 * v:.2f}%" for k, v in R["e6"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-attack", type=int, default=None)
    ap.add_argument("--alexnet", action="store_true")
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--pool-add", type=int, default=20)
    ap.add_argument("--refresh", type=int, default=50, help="refresh the adversarial pool every N batches")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.epochs = a.epochs or (2 if a.quick else 20)
    a.n_attack = a.n_attack or (100 if a.quick else 1000)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        data = load_mnist()
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
