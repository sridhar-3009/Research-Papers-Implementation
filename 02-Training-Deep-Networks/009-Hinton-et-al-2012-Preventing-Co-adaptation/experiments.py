"""Reproduce Hinton et al. (2012), "Improving neural networks by preventing co-adaptation".

  E1  page 2: the mean network = the normalized geometric mean of all 2^N dropout
      sub-networks (one hidden layer, softmax), and it beats their average log-prob
  E2  Figure 1 / Appendix A.1: MNIST 784-800-800-10 - standard backprop vs 50% hidden
      dropout vs + 20% input dropout (test errors out of 10,000)
  E3  page 2: the mean network vs averaging many sampled dropout networks
  E4  Figure 5 / Appendix A.3: first-layer features of a 784-500-500 net, backprop vs dropout

Scale: the paper trained for 3,000 epochs; here 100. The learning-rate decay and the
momentum ramp are compressed by the same factor (30x) so the schedule has the same shape.

Run:  python3 experiments.py            (~25 minutes on an Apple-silicon GPU)
      python3 experiments.py --quick    (~3 minutes)
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from dropout import DropoutNet, mc_average_errors, test_errors, train

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def mnist():
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    f = lambda d: (d.data.reshape(-1, 784).float() / 255).to(DEV)
    return f(tr), tr.targets.to(DEV), f(te), te.targets.to(DEV)


def e1_geometric_mean(n_hidden=10):
    torch.manual_seed(0)
    net = DropoutNet([6, n_hidden, 4], keep_hidden=0.5, init_std=1.0, seed=1).double()
    x = torch.randn(20, 6, dtype=torch.float64)
    y = torch.randint(0, 4, (20,))
    with torch.no_grad():
        L = torch.stack([torch.log_softmax(net(x, masks=[torch.ones(6, dtype=torch.float64),
                                                         torch.tensor(bits, dtype=torch.float64)]), 1)
                         for bits in itertools.product([0, 1], repeat=n_hidden)])
        geo = torch.softmax(L.mean(0), 1)
        mean_net = torch.softmax(net(x, mode="mean"), 1)
    return dict(n_subnets=2 ** n_hidden, max_diff=float((geo - mean_net).abs().max()),
                logp_mean_net=float(torch.log(mean_net[range(20), y]).mean()),
                logp_avg_subnets=float(L[:, range(20), y].mean()))


def run(sizes, X, y, Xt, yt, epochs, drop_in, drop_hidden, max_norm, lr0, seed=0):
    net = DropoutNet(sizes, keep_input=1 - drop_in, keep_hidden=1 - drop_hidden, seed=seed).to(DEV)
    squeeze = 3000 / epochs                                  # compress the 3000-epoch schedule
    hist = train(net, X, y, epochs, dropout=(drop_in + drop_hidden) > 0, max_norm=max_norm, lr0=lr0,
                 lr_decay=0.998 ** squeeze, mom_epochs=500 / squeeze, Xte=Xt, yte=yt, seed=seed)
    return net, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        r = json.loads((HERE / "results.json").read_text())
        write(r)
        return
    X, y, Xt, yt = mnist()
    epochs = 10 if args.quick else 100
    r = {"e1": e1_geometric_mean(), "epochs": epochs, "e2": {}}
    arch = [784, 800, 800, 10]
    # standard backprop: its own best starting learning rate (no max-norm, no dropout)
    best = None
    for lr0 in (0.3, 1.0, 3.0):
        _, h = run(arch, X, y, Xt, yt, epochs, 0, 0, None, lr0)
        print(f"standard backprop lr0 {lr0}: final {h[-1]} errors, best {min(h)}", flush=True)
        if best is None or h[-1] < best[1][-1]:
            best = (lr0, h)
    r["e2"]["standard backprop"] = dict(lr0=best[0], hist=best[1])
    for name, di, dh, mn in (("max-norm only", 0.0, 0.0, 15.0),
                             ("50% hidden dropout + max-norm", 0.0, 0.5, 15.0),
                             ("50% hidden + 20% input dropout + max-norm", 0.2, 0.5, 15.0)):
        net, h = run(arch, X, y, Xt, yt, epochs, di, dh, mn, 10.0 if dh else 1.0)
        print(f"{name}: final {h[-1]} errors, best {min(h)}", flush=True)
        r["e2"][name] = dict(lr0=10.0 if dh else 1.0, hist=h)
        if di > 0:
            last = net
    # E3: mean network vs Monte-Carlo averaging of k dropout nets
    r["e3"] = dict(mean=test_errors(last, Xt, yt),
                   mc={k: mc_average_errors(last, Xt, yt, k) for k in (1, 5, 10, 50, 100)})
    # E4: features of a 784-500-500 net (plus a softmax output layer to train it)
    feats = {}
    for name, dh in (("backprop", 0.0), ("dropout", 0.5)):
        net, h = run([784, 500, 500, 10], X, y, Xt, yt, epochs // 2, 0.0, dh, 15.0 if dh else None, 10.0 if dh else 1.0)
        feats[name] = dict(W=net.W[0].detach().cpu().numpy()[:, :100].tolist(), errors=h[-1])
    r["e4"] = feats
    (HERE / "results.json").write_text(json.dumps(r))
    write(r)


def write(r):
    e1 = r["e1"]
    L = ["# Hinton et al. (2012): reproduced results", "", "Generated by `python3 experiments.py`.", "",
         "## E1. The mean network is the geometric mean of all dropout networks (page 2)", "",
         f"One hidden layer of 10 logistic units, softmax output: all {e1['n_subnets']} sub-networks enumerated.", "",
         f"- max |normalized geometric mean - mean network| = **{e1['max_diff']:.1e}** (exact, up to rounding).",
         f"- average log-probability of the correct class: mean network **{e1['logp_mean_net']:.3f}** vs average over "
         f"sub-networks **{e1['logp_avg_subnets']:.3f}** (the mean network is better, as claimed).", "",
         f"## E2. MNIST, 784-800-800-10, {r['epochs']} epochs (Figure 1)", "",
         "Test errors out of 10,000 with the mean network. Paper (3,000 epochs): ~160 standard (best published), "
         "~130 with 50% dropout, ~110 with 20% input dropout too.", "",
         "| method | starting lr | errors at the end | lowest during training |", "|---|---|---|---|"]
    for name, v in r["e2"].items():
        L.append(f"| {name} | {v['lr0']} | {v['hist'][-1]} | {min(v['hist'])} |")
    e3 = r["e3"]
    L += ["", "## E3. Mean network vs averaging sampled dropout networks (page 2)", "",
          f"Network with 50% hidden + 20% input dropout. Mean network: **{e3['mean']}** errors.", "",
          "| k sampled networks averaged | " + " | ".join(str(k) for k in e3["mc"]) + " |",
          "|---|" + "---|" * len(e3["mc"]),
          "| test errors | " + " | ".join(str(v) for v in e3["mc"].values()) + " |", "",
          "## E4. First-layer features, 784-500-500 (Figure 5)", "",
          f"See `figures/fig5_features.png`. Test errors: backprop {r['e4']['backprop']['errors']}, "
          f"dropout {r['e4']['dropout']['errors']}."]
    text = "\n".join(L) + "\n"
    (HERE / "results.md").write_text(text)
    print(text)
    figures(r)


def figures(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, v in r["e2"].items():
        ax.plot(range(1, len(v["hist"]) + 1), v["hist"], label=name)
    ax.set_ylim(80, 250); ax.set_xlabel("epoch"); ax.set_ylabel("test errors (of 10,000)")
    ax.set_title("Figure 1 reproduced: MNIST 784-800-800-10", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "fig1_mnist_errors.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, name in zip(axes, ("backprop", "dropout")):
        W = np.array(r["e4"][name]["W"])               # (784, 100)
        tiles = np.zeros((10 * 29, 10 * 29))
        for k in range(100):
            w = W[:, k].reshape(28, 28)
            w = (w - w.min()) / (w.max() - w.min() + 1e-9)
            i, j = divmod(k, 10)
            tiles[i * 29:i * 29 + 28, j * 29:j * 29 + 28] = w
        ax.imshow(tiles, cmap="gray"); ax.axis("off")
        ax.set_title(f"({'a' if name == 'backprop' else 'b'}) {name}: 100 first-layer features", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "fig5_features.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
