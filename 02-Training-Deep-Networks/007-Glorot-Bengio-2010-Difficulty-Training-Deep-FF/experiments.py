"""Reproduce Glorot & Bengio (2010).

  E1  Section 4.2, Figures 6-8, Eq. (17): at initialization - activation spread,
      back-propagated gradients and weight gradients per layer, and the Jacobian's
      average singular value, for standard vs normalized init
  E2  Section 4.1, Figure 5: plateaus in the cross-entropy vs quadratic cost surface
  E3  Section 3, Figures 2, 3, 10: activations per layer DURING training
      (sigmoid top-layer saturation; tanh layer-by-layer saturation; softsign)
  E4  Section 5, Table 1, Figures 11-12: test error of 5-hidden-layer nets,
      {softsign, tanh} x {standard, normalized} + sigmoid, on Shapeset-3x2 and MNIST

Scale (the paper trained for 5 million updates): Shapeset 300,000 online examples
(30,000 updates of 10), MNIST 5 epochs. Hidden layers: 1000 units, as in the paper.
Learning rate picked from {0.01, 0.03, 0.1} on a validation set, as in the paper.

Run:  python3 experiments.py         (~30-40 minutes; saves results.json, results.md, figures/)
      python3 experiments.py --quick (~3 minutes, smaller runs, for checking)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from glorot import (DeepNet, activation_stats, gradient_stats, jacobian_singular_values, nll,
                    quadratic, shapeset)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
CONFIGS = [("softsign", "standard"), ("softsign", "normalized"), ("tanh", "standard"),
           ("tanh", "normalized"), ("sigmoid", "standard")]
NAME = lambda act, init: act.capitalize() + (" N" if init == "normalized" else "")
torch.set_num_threads(8)


def mnist():
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    X = tr.data.reshape(-1, 784).float() / 255
    Xte = te.data.reshape(-1, 784).float() / 255
    # the paper's split: 50,000 train, 10,000 validation, 10,000 test
    return (X[:50000], tr.targets[:50000]), (X[50000:], tr.targets[50000:]), (Xte, te.targets)


def shapeset_split(seed=123):
    Xv, yv = shapeset(5000, rng=seed)
    Xt, yt = shapeset(10000, rng=seed + 1)
    return (torch.tensor(Xv), torch.tensor(yv)), (torch.tensor(Xt), torch.tensor(yt))


@torch.no_grad()
def error(net, X, y):
    return float((net(X).argmax(1) != y).float().mean())


def train(act, init, lr, stream, n_updates, val, monitor=None, monitor_every=500, seed=0):
    """SGD on mini-batches of 10 (Section 2.3). stream(k) returns batch number k.
    Returns (net, validation-error curve, activation-stats history)."""
    torch.manual_seed(seed)
    net = DeepNet(val[0].shape[1], 10 if val[0].shape[1] == 784 else 9, act=act, init=init, seed=seed)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    curve, stats = [], []
    for k in range(n_updates):
        if monitor is not None and k % monitor_every == 0:
            stats.append((k, activation_stats(net, monitor)))
        if k % max(1, n_updates // 20) == 0:
            curve.append((k, error(net, *val)))
        X, y = stream(k)
        opt.zero_grad()
        loss = nll(net(X), y)
        if not torch.isfinite(loss):
            curve.append((k, 1.0))
            break
        loss.backward()
        opt.step()
    curve.append((n_updates, error(net, *val)))
    return net, curve, stats


# ---------------------------------------------------------------------------

def e1_initialization():
    X, y = shapeset(300, rng=7)
    X, y = torch.tensor(X), torch.tensor(y)
    out = {}
    for act in ("tanh", "softsign", "sigmoid"):
        for init in ("standard", "normalized"):
            net = DeepNet(1024, 9, act=act, init=init, seed=0)
            a, b, w = gradient_stats(net, X, y)
            out[f"{act} {init}"] = dict(act_std=a, backprop_std=b, weight_grad_std=w,
                                        jacobian_sv=jacobian_singular_values(net, X))
    return out


def e2_cost_surface(n=41, span=4.0, seed=0):
    """Figure 5: a 1-input, 1-hidden-tanh-unit, 1-output network on random inputs
    and random binary targets. Cost as a function of the two weights W1, W2
    (biases fixed at 0). A 'plateau' point = gradient norm < 1% of its maximum."""
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=200), dtype=torch.float32)
    t = torch.tensor(rng.integers(0, 2, 200), dtype=torch.float32)
    grid = np.linspace(-span, span, n)
    res = {}
    for name in ("cross-entropy", "quadratic"):
        C = np.zeros((n, n)); G = np.zeros((n, n))
        for i, w1 in enumerate(grid):
            for j, w2 in enumerate(grid):
                W = torch.tensor([w1, w2], requires_grad=True)
                p = torch.sigmoid(W[1] * torch.tanh(W[0] * x))
                if name == "cross-entropy":
                    c = torch.nn.functional.binary_cross_entropy(p, t)
                else:
                    c = 0.5 * ((p - t) ** 2).mean()
                c.backward()
                C[i, j], G[i, j] = float(c.detach()), float(W.grad.norm())
        res[name] = dict(cost=C.tolist(), grad=G.tolist(),
                         plateau_fraction=float((G < 0.01 * G.max()).mean()),
                         grad_ratio_max_min=float(G.max() / max(G.min(), 1e-12)))
    return res, grid.tolist()


def shapeset_stream(seed):
    buf = {}

    def stream(k):
        block = k // 100                              # 1,000 fresh images per block of 100 updates
        if block not in buf:
            buf.clear()
            X, y = shapeset(1000, rng=seed * 1_000_003 + block)
            buf[block] = (torch.tensor(X), torch.tensor(y))
        X, y = buf[block]
        i = (k % 100) * 10
        return X[i:i + 10], y[i:i + 10]
    return stream


def mnist_stream(X, y, seed):
    gen = torch.Generator().manual_seed(seed)
    perms = {}

    def stream(k):
        epoch, pos = divmod(k * 10, len(X))
        if epoch not in perms:
            perms.clear()
            perms[epoch] = torch.randperm(len(X), generator=gen)
        idx = perms[epoch][pos:pos + 10]
        return X[idx], y[idx]
    return stream


def e3_e4(quick=False):
    lrs = (0.01, 0.03, 0.1)
    n_shape = 3000 if quick else 30000
    n_mnist = 2500 if quick else 25000                 # 5 epochs of 50,000 in batches of 10
    (Xv, yv), (Xt, yt) = shapeset_split()
    monitor = Xv[:300]
    (Mtr, ytr), (Mv, yvm), (Mte, yte) = mnist()
    results = {"shapeset": {}, "mnist": {}}
    for act, init in CONFIGS:
        name = NAME(act, init)
        for dataset in ("shapeset", "mnist"):
            best = None
            for lr in lrs:
                t0 = time.time()
                if dataset == "shapeset":
                    net, curve, stats = train(act, init, lr, shapeset_stream(1), n_shape, (Xv, yv),
                                              monitor=monitor, monitor_every=max(1, n_shape // 30))
                    test = error(net, Xt, yt)
                else:
                    net, curve, stats = train(act, init, lr, mnist_stream(Mtr, ytr, 1), n_mnist, (Mv, yvm))
                    test = error(net, Mte, yte)
                val = curve[-1][1]
                print(f"{dataset:8s} {name:11s} lr {lr:<5} val {val:.4f} test {test:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if best is None or val < best["val"]:
                    best = dict(lr=lr, val=val, test=test, curve=curve, stats=stats)
            results[dataset][name] = best
    return results


# ---------------------------------------------------------------------------

def report(r):
    L = ["# Glorot & Bengio (2010): reproduced results", "", "Generated by `python3 experiments.py`.", ""]
    L += ["## E1. At initialization: 5 hidden layers of 1000 units, 300 Shapeset images", "",
          "Standard init = U[-1/sqrt(n), 1/sqrt(n)] (Eq. 1); normalized = U[-sqrt(6/(n_in+n_out)), ...] (Eq. 16).", "",
          "| network | activation std, layers 1-5 | back-prop gradient std, layers 1-5 (x1e-5) | weight-gradient std, layers 1-5 (x1e-5) | mean Jacobian singular value |",
          "|---|---|---|---|---|"]
    for k, v in r["e1"].items():
        L.append(f"| {k} | {' '.join(f'{a:.3f}' for a in v['act_std'])} | "
                 f"{' '.join(f'{b * 1e5:.1f}' for b in v['backprop_std'][:5])} | "
                 f"{' '.join(f'{w * 1e5:.1f}' for w in v['weight_grad_std'][:5])} | "
                 f"{np.mean(v['jacobian_sv']):.2f} |")
    e2 = r["e2"]["surface"]
    L += ["", "## E2. Cost surface plateaus (Figure 5)", "",
          "1 input -> 1 tanh unit -> 1 sigmoid output, random inputs and binary targets; cost over W1, W2 in [-4, 4].", "",
          "| cost | fraction of the surface that is a plateau (gradient < 1% of max) | max / min gradient |", "|---|---|---|"]
    for k in ("cross-entropy", "quadratic"):
        L.append(f"| {k} | {e2[k]['plateau_fraction']:.1%} | {e2[k]['grad_ratio_max_min']:.0f} |")
    for ds, title in (("shapeset", "Shapeset-3x2 (online)"), ("mnist", "MNIST")):
        L += ["", f"## E4. Test error, 5 hidden layers of 1000 units: {title}", "",
              "| network | best lr | test error | paper (Table 1) |", "|---|---|---|---|"]
        paper = {"shapeset": {"Softsign": 16.27, "Softsign N": 16.06, "Tanh": 27.15, "Tanh N": 15.60, "Sigmoid": 82.61},
                 "mnist": {"Softsign": 1.64, "Softsign N": 1.72, "Tanh": 1.76, "Tanh N": 1.64, "Sigmoid": 2.21}}[ds]
        for name, b in sorted(r["e34"][ds].items(), key=lambda kv: kv[1]["test"]):
            L.append(f"| {name} | {b['lr']} | {b['test']:.2%} | {paper[name]:.2f}% |")
    # saturation during training (E3)
    L += ["", "## E3. Activations during training on Shapeset (Figures 2, 3, 10)", "",
          "Mean activation of each hidden layer (layers 1-5) at the start, 1/3, 2/3 and end of training:", ""]
    for name in ("Sigmoid", "Tanh", "Tanh N", "Softsign"):
        stats = r["e34"]["shapeset"][name]["stats"]
        L.append(f"**{name}**")
        L.append("")
        L.append("| updates | " + " | ".join(f"layer {i + 1} mean / 98th pct" for i in range(5)) + " |")
        L.append("|---|" + "---|" * 5)
        for k, s in (stats[0], stats[len(stats) // 3], stats[2 * len(stats) // 3], stats[-1]):
            L.append(f"| {k} | " + " | ".join(f"{m:+.2f} / {q:.2f}" for m, sd, q in s) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def figures(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = HERE / "figures"; path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for init, ls in (("standard", "--"), ("normalized", "-")):
        v = r["e1"][f"tanh {init}"]
        axes[0].plot(range(1, 6), v["act_std"], ls, marker="o", label=init)
        axes[1].plot(range(1, 7), v["backprop_std"], ls, marker="o", label=init)
        axes[2].plot(range(1, 7), v["weight_grad_std"], ls, marker="o", label=init)
    for ax, t in zip(axes, ("activation std (Fig. 6)", "back-propagated gradient std (Fig. 7)", "weight gradient std (Fig. 8)")):
        ax.set_title(t, fontsize=9); ax.set_xlabel("layer"); ax.set_yscale("log"); ax.legend(fontsize=8)
    fig.suptitle("tanh net at initialization: standard vs normalized init", fontsize=10)
    fig.tight_layout(); fig.savefig(path / "fig6_7_8_init.png", dpi=130); plt.close(fig)

    e2, grid = r["e2"]["surface"], r["e2"]["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, ("cross-entropy", "quadratic")):
        im = ax.contourf(grid, grid, np.array(e2[k]["cost"]).T, levels=30, cmap="viridis")
        ax.set_title(f"{k} cost (Fig. 5)", fontsize=9); ax.set_xlabel("W1 (layer 1)"); ax.set_ylabel("W2 (layer 2)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path / "fig5_cost_surfaces.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    for ax, name in zip(axes, ("Sigmoid", "Tanh", "Softsign")):
        stats = r["e34"]["shapeset"][name]["stats"]
        ks = [k for k, _ in stats]
        for layer in range(5):
            key = 0 if name == "Sigmoid" else 2         # sigmoid: mean (Fig. 2); others: 98th pct (Fig. 3)
            ax.plot(ks, [s[layer][key] for _, s in stats], label=f"layer {layer + 1}")
        ax.set_title(f"{name}: {'mean activation (Fig. 2)' if name == 'Sigmoid' else '98th percentile |activation| (Fig. 3)'}", fontsize=9)
        ax.set_xlabel("updates"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path / "fig2_3_saturation.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ds in zip(axes, ("shapeset", "mnist")):
        for name, b in r["e34"][ds].items():
            k, e = zip(*b["curve"])
            ax.plot(k, e, label=name)
        ax.set_title(f"{ds}: validation error during training (Figs. 11-12)", fontsize=9)
        ax.set_xlabel("updates (mini-batches of 10)"); ax.set_ylabel("error"); ax.legend(fontsize=8)
        if ds == "mnist":
            ax.set_ylim(0, 0.1)
    fig.tight_layout(); fig.savefig(path / "fig11_12_error_curves.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild results.md/figures from results.json")
    args = ap.parse_args()
    if args.report_only:
        r = json.loads((HERE / "results.json").read_text())
    else:
        surface, grid = e2_cost_surface()
        r = {"e1": e1_initialization(), "e2": {"surface": surface, "grid": grid}, "e34": e3_e4(args.quick)}
        (HERE / "results.json").write_text(json.dumps(r))
    text = report(r)
    (HERE / "results.md").write_text(text)
    print(text)
    figures(r)


if __name__ == "__main__":
    main()
