"""Reproduce the MNIST experiments of Ioffe & Szegedy (2015), plus small-scale versions of the
ImageNet findings (the ImageNet runs themselves need a cluster).

  E1  Figure 1: 784-100-100-100-10 sigmoid, 50,000 steps of 60 examples, with and without BN.
      (a) test accuracy vs steps; (b, c) the 15th/50th/85th percentiles of one sigmoid input
      in the last hidden layer over training (the "internal covariate shift").
  E2  Figures 2-3 in miniature: learning rate x1, x5, x30 with and without BN (ReLU and sigmoid).
      Steps to reach the baseline's best accuracy, and the best accuracy.
  E3  "Inception with sigmoid never beats chance, BN-x5-Sigmoid reaches 69.8%": a DEEP
      (10 hidden layers) sigmoid net, with and without BN.

The paper does not give the learning rate or init scale for Figure 1; we use SGD with
lr 0.5 and N(0, 0.1^2) weights for both networks (change with --lr / --init-std).

!! HEAVY. Not run on the author's laptop. The full script is roughly 30-60 minutes on a CPU.
       python3 experiments.py --quick         # a few minutes
       python3 experiments.py --only e1       # one experiment
       python3 experiments.py --report-only   # rebuild results.md from results.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from batchnorm import MLP

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_mnist():
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    f = lambda d: (d.data.float().reshape(-1, 784).to(DEV) / 255, d.targets.to(DEV))
    return f(tr), f(te)


@torch.no_grad()
def accuracy(net, X, y):
    net.eval()
    acc = (net(X).argmax(1) == y).float().mean().item()
    net.train()
    return acc


def train(net, train_set, test_set, steps, lr, batch=60, eval_every=500, track_unit=None, seed=0):
    """Plain SGD on cross-entropy. Returns accuracies (and percentiles of one sigmoid input if asked)."""
    (X, y), (Xt, yt) = train_set, test_set
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    log = {"step": [], "acc": [], "pct": []}
    for step in range(steps + 1):
        if step % eval_every == 0:
            log["step"].append(step)
            log["acc"].append(accuracy(net, Xt, yt))
            if track_unit is not None:
                with torch.no_grad():
                    net.eval()
                    _, pre = net(Xt, return_preact=True)
                    net.train()
                    v = pre[-1][:, track_unit]                        # input of one last-layer sigmoid
                    log["pct"].append([torch.quantile(v, q).item() for q in (0.15, 0.5, 0.85)])
            if not np.isfinite(log["acc"][-1]):
                break
        if step == steps:
            break
        idx = torch.randint(0, len(X), (batch,), generator=g).to(DEV)
        loss = F.cross_entropy(net(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if not torch.isfinite(loss):
            log["diverged_at"] = step
            break
    return log


def e1_figure1(data, a):
    out = {}
    for bn in (False, True):
        torch.manual_seed(0)
        net = MLP(bn=bn, init_std=a.init_std).to(DEV)
        out["BN" if bn else "baseline"] = train(net, *data, a.steps, a.lr, track_unit=0)
        print(f"E1 {'BN' if bn else 'baseline'}: final accuracy {out['BN' if bn else 'baseline'] ['acc'][-1]:.4f}", flush=True)
    return out


def steps_to(log, target):
    for s, acc in zip(log["step"], log["acc"]):
        if acc >= target:
            return s
    return None


def e2_learning_rates(data, a):
    out = {}
    for act in ("relu", "sigmoid"):
        base_lr = 0.1 if act == "relu" else 0.5
        runs = {}
        for bn in (False, True):
            for mult in (1, 5, 30):
                torch.manual_seed(0)
                net = MLP(act=act, bn=bn, init_std=a.init_std).to(DEV)
                log = train(net, *data, a.steps, base_lr * mult)
                runs[f"{'BN' if bn else 'plain'}-x{mult}"] = log
                print(f"E2 {act} {'BN' if bn else 'plain'} lr x{mult}: best {max(log['acc']):.4f}", flush=True)
        target = max(runs["plain-x1"]["acc"])
        out[act] = {k: dict(best=max(v["acc"]), steps_to_baseline_best=steps_to(v, target),
                            diverged=("diverged_at" in v), curve=v) for k, v in runs.items()}
    return out


def e3_deep_sigmoid(data, a):
    out = {}
    sizes = (784,) + (100,) * 10 + (10,)
    for bn in (False, True):
        torch.manual_seed(0)
        net = MLP(sizes, act="sigmoid", bn=bn, init_std=a.init_std).to(DEV)
        log = train(net, *data, a.steps, a.lr)
        out["BN" if bn else "baseline"] = log
        print(f"E3 10-layer sigmoid {'BN' if bn else 'baseline'}: final accuracy {log['acc'][-1]:.4f}", flush=True)
    return out


def plots(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    if "e1" in R:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for k, v in R["e1"].items():
            ax[0].plot(v["step"], v["acc"], label=k)
        ax[0].set_title("(a) test accuracy"); ax[0].set_xlabel("steps"); ax[0].legend(); ax[0].set_ylim(0.8, 1)
        for i, k in enumerate(("baseline", "BN")):
            p = np.array(R["e1"][k]["pct"])
            for j in range(3):
                ax[i + 1].plot(R["e1"][k]["step"], p[:, j], "k-" if j == 1 else "k--")
            ax[i + 1].set_title(f"({'bc'[i]}) {k}: sigmoid input, 15/50/85th pct"); ax[i + 1].set_xlabel("steps")
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e1_figure1.png", dpi=130); plt.close(fig)
    if "e2" in R:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for i, act in enumerate(R["e2"]):
            for k, v in R["e2"][act].items():
                ax[i].plot(v["curve"]["step"], v["curve"]["acc"], "-" if k.startswith("BN") else ":", label=k)
            ax[i].set_title(f"{act}: learning-rate multipliers"); ax[i].set_ylim(0.8, 1); ax[i].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e2_learning_rates.png", dpi=130); plt.close(fig)
    if "e3" in R:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for k, v in R["e3"].items():
            ax.plot(v["step"], v["acc"], label=k)
        ax.set_title("10-layer sigmoid net"); ax.set_xlabel("steps"); ax.legend()
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e3_deep_sigmoid.png", dpi=130); plt.close(fig)


def report(R, a):
    L = ["# Results", "", f"Settings: {a.steps} steps, batch 60, lr {a.lr}, init std {a.init_std}"
         + (" (QUICK run: not the paper's scale)" if a.quick else ""), ""]
    if "e1" in R:
        L += ["## E1: Figure 1", "", "| network | final test accuracy |", "|---|---|"]
        L += [f"| {k} | {v['acc'][-1]:.4f} |" for k, v in R["e1"].items()]
        for k, v in R["e1"].items():
            p = np.array(v["pct"])
            L += ["", f"{k}: median sigmoid input moved from {p[0, 1]:+.2f} to {p[-1, 1]:+.2f}; "
                      f"15-85% spread from {p[0, 2] - p[0, 0]:.2f} to {p[-1, 2] - p[-1, 0]:.2f}"]
        L += ["", "![](figures/e1_figure1.png)", ""]
    if "e2" in R:
        L += ["## E2: higher learning rates (Figures 2-3 in miniature)", ""]
        for act, runs in R["e2"].items():
            L += [f"**{act}**", "", "| run | best accuracy | steps to reach plain-x1's best | diverged |", "|---|---|---|---|"]
            L += [f"| {k} | {v['best']:.4f} | {v['steps_to_baseline_best']} | {v['diverged']} |" for k, v in runs.items()]
            L += [""]
        L += ["![](figures/e2_learning_rates.png)", ""]
    if "e3" in R:
        L += ["## E3: 10-layer sigmoid net", "", "| network | final test accuracy |", "|---|---|"]
        L += [f"| {k} | {v['acc'][-1]:.4f} |" for k, v in R["e3"].items()]
        L += ["", "![](figures/e3_deep_sigmoid.png)"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="5,000 steps instead of 50,000")
    ap.add_argument("--only", choices=["e1", "e2", "e3"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--init-std", type=float, default=0.1)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.steps = a.steps or (5000 if a.quick else 50000)
    path = HERE / "results.json"
    if a.report_only:
        R = json.loads(path.read_text())
    else:
        data = load_mnist()
        R = json.loads(path.read_text()) if path.exists() else {}
        t0 = time.time()
        for name, fn in (("e1", e1_figure1), ("e2", e2_learning_rates), ("e3", e3_deep_sigmoid)):
            if a.only in (None, name):
                R[name] = fn(data, a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    plots(R)
    report(R, a)


if __name__ == "__main__":
    main()
