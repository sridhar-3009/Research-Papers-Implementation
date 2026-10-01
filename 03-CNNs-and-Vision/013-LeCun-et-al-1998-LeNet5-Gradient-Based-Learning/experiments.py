"""Reproduce Section III of LeCun et al. (1998) on MNIST.

  E1  Figure 5: LeNet-5, 20 passes, stochastic diagonal Levenberg-Marquardt, the paper's
      learning-rate schedule, Hessian re-estimated on 500 samples before each pass.
      Paper: test error 0.95% (stable after ~10 passes), training error 0.35%.
  E2  Figure 6: training set size 15,000 / 30,000 / 60,000.
  E3  Distortions: 90% of the patterns shown are randomly distorted (60,000 originals +
      540,000 distortions in the paper). Paper: 0.95% -> 0.8%.
  E4  Figure 9 baselines: linear classifier (paper 12%), K-NN Euclidean (5.0%),
      MLP 784-300-10 (4.7%), MLP 784-300-100-10 (3.05%).
  E5  Shift robustness (Section II.A's claim): test error when every test digit is moved
      by 0..4 pixels, LeNet-5 vs the 784-300-100-10 MLP.
  E6  Section II.C: MSE vs MAP loss when the RBF centers are LEARNED (MSE should collapse).

The paper updates after EVERY example (batch size 1). That is what --batch 1 does (default),
and it is slow: 20 passes = 1.2 million updates. --batch 32 is ~20x faster and close in result.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick              # small, a few minutes
       python3 experiments.py --only e1            # one experiment (hours at batch 1 on a CPU)
       python3 experiments.py --report-only        # rebuild results.md from results.json
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

from lenet5 import LeNet5, distort, gauss_newton_diag, map_loss, mse_loss, paper_eta, prepare, sdlm_step

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    return (prepare(tr.data).to(DEV), tr.targets.to(DEV)), (prepare(te.data).to(DEV), te.targets.to(DEV))


@torch.no_grad()
def error(net, X, y, bs=2000):
    wrong = 0
    for i in range(0, len(X), bs):
        wrong += (net.predict(X[i:i + bs]) != y[i:i + bs]).sum().item()
    return wrong / len(X)


def train_lenet(train, test, a, n_train=None, distortions=False, loss_fn=mse_loss, learn_prototypes=False, seed=0):
    """The paper's training procedure. Returns per-pass training error (on the fly) and test error."""
    torch.manual_seed(seed)
    X, y = train
    if n_train:
        X, y = X[:n_train], y[:n_train]
    net = LeNet5(learn_prototypes=learn_prototypes).to(DEV)
    g = torch.Generator().manual_seed(seed)
    log = {"train_err": [], "test_err": []}
    for epoch in range(a.epochs):
        h = gauss_newton_diag(net, X[torch.randperm(len(X), generator=g)[:a.hessian_samples].to(DEV)])
        eta = paper_eta(epoch)
        wrong, perm = 0, torch.randperm(len(X), generator=g).to(DEV)
        for i in range(0, len(X), a.batch):
            idx = perm[i:i + a.batch]
            xb = X[idx]
            if distortions:                                         # 540k of 600k patterns are distorted
                keep = torch.rand(len(xb), generator=g).to(DEV) < 0.1
                xb = torch.where(keep[:, None, None, None], xb, distort(xb, generator=g))
            out = net(xb)
            wrong += (out.argmin(1) != y[idx]).sum().item()
            sdlm_step(net, loss_fn(out, y[idx]), h, eta)
        log["train_err"].append(wrong / len(X))                     # measured on the fly, like Figure 5
        log["test_err"].append(error(net, *test))
        print(f"  pass {epoch + 1}: train {log['train_err'][-1]:.4f}  test {log['test_err'][-1]:.4f}", flush=True)
    return net, log


def e1(train, test, a):
    net, log = train_lenet(train, test, a)
    torch.save(net.state_dict(), HERE / "lenet5.pt")
    return log


def e2(train, test, a):
    return {n: train_lenet(train, test, a, n_train=n)[1] for n in ((1500, 3000, 6000) if a.quick else (15000, 30000, 60000))}


def e3(train, test, a):
    return train_lenet(train, test, a, distortions=True)[1]


def mlp(sizes):
    layers = []
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(m, n))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(nn.Flatten(), *layers).to(DEV)


def train_simple(net, train, epochs, lr=0.05, seed=0):
    torch.manual_seed(seed)
    X, y = train
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        perm = torch.randperm(len(X)).to(DEV)
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def knn_error(train, test, k=3, n_test=10000):
    """Euclidean K-NN on raw pixels (Section III.C.2)."""
    (X, y), (Xt, yt) = train, test
    X, Xt = X.flatten(1), Xt[:n_test].flatten(1)
    wrong = 0
    for i in range(0, len(Xt), 500):
        d = torch.cdist(Xt[i:i + 500], X)
        votes = y[d.topk(k, largest=False).indices]
        wrong += (votes.mode(1).values != yt[i:i + 500]).sum().item()
    return wrong / len(Xt)


def e4(train, test, a):
    ep = 2 if a.quick else 20
    out = {}
    for name, sizes in (("linear", [1024, 10]), ("300-10", [1024, 300, 10]), ("300-100-10", [1024, 300, 100, 10])):
        net = train_simple(mlp(sizes), train, ep)
        out[name] = torch.mean((net(test[0]).argmax(1) != test[1]).float()).item()
        print(f"  {name}: {out[name]:.4f}", flush=True)
    out["knn-3"] = knn_error(train, test, n_test=1000 if a.quick else 10000)
    print(f"  knn-3: {out['knn-3']:.4f}", flush=True)
    return out


def e5(train, test, a):
    """Error on shifted test digits. Needs E1's saved network (or trains a quick one)."""
    lenet = LeNet5().to(DEV)
    if (HERE / "lenet5.pt").exists():
        lenet.load_state_dict(torch.load(HERE / "lenet5.pt", map_location=DEV))
    else:
        lenet = train_lenet(train, test, a)[0]
    net_mlp = train_simple(mlp([1024, 300, 100, 10]), train, 2 if a.quick else 20)
    Xt, yt = test
    out = {"shift": [], "LeNet-5": [], "MLP 300-100": []}
    for s in range(5):
        shifted = torch.roll(Xt, shifts=(s, s), dims=(2, 3))         # background rolls in at the edges: fine, it's -0.1
        out["shift"].append(s)
        out["LeNet-5"].append(error(lenet, shifted, yt))
        with torch.no_grad():
            out["MLP 300-100"].append(torch.mean((net_mlp(shifted).argmax(1) != yt).float()).item())
        print(f"  shift {s}px: LeNet-5 {out['LeNet-5'][-1]:.4f}  MLP {out['MLP 300-100'][-1]:.4f}", flush=True)
    return out


def e6(train, test, a):
    out = {}
    for name, fn in (("MSE", mse_loss), ("MAP", map_loss)):
        net, log = train_lenet(train, test, a, loss_fn=fn, learn_prototypes=True, n_train=6000 if a.quick else 20000)
        spread = torch.pdist(net.prototypes.detach()).min().item()
        out[name] = dict(test_err=log["test_err"], min_prototype_distance=spread)
        print(f"  {name}: final test error {log['test_err'][-1]:.4f}, closest two RBF centers {spread:.3f}", flush=True)
    return out


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}


def report(R, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    L = ["# Results", "", f"passes {a.epochs}, batch {a.batch}" + ("  (QUICK run: not the paper's scale)" if a.quick else ""), ""]
    if "e1" in R:
        L += ["## E1: Figure 5 (paper: test 0.95%, train 0.35%)", "",
              f"final test error {100 * R['e1']['test_err'][-1]:.2f}%, training error {100 * R['e1']['train_err'][-1]:.2f}%", "",
              "![](figures/e1_fig5.png)", ""]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(np.arange(1, len(R["e1"]["test_err"]) + 1), 100 * np.array(R["e1"]["test_err"]), "o-", label="test")
        ax.plot(np.arange(1, len(R["e1"]["train_err"]) + 1), 100 * np.array(R["e1"]["train_err"]), "s-", label="training (on the fly)")
        ax.set_xlabel("passes"); ax.set_ylabel("error (%)"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e1_fig5.png", dpi=130); plt.close(fig)
    if "e2" in R:
        L += ["## E2: Figure 6 (training set size)", "", "| examples | test error | training error |", "|---|---|---|"]
        L += [f"| {n} | {100 * v['test_err'][-1]:.2f}% | {100 * v['train_err'][-1]:.2f}% |" for n, v in R["e2"].items()]
        L += [""]
    if "e3" in R:
        L += ["## E3: distortions (paper: 0.8%)", "", f"test error {100 * R['e3']['test_err'][-1]:.2f}%", ""]
    if "e4" in R:
        paper = {"linear": 12.0, "300-10": 4.7, "300-100-10": 3.05, "knn-3": 5.0}
        L += ["## E4: Figure 9 baselines", "", "| method | ours | paper |", "|---|---|---|"]
        L += [f"| {k} | {100 * v:.2f}% | {paper[k]}% |" for k, v in R["e4"].items()]
        L += ["", "(ours use SGD + momentum + cross-entropy on 32x32 inputs, not the paper's exact recipe)", ""]
    if "e5" in R:
        L += ["## E5: shifted test digits", "", "| shift (px) | LeNet-5 | MLP 300-100 |", "|---|---|---|"]
        L += [f"| {s} | {100 * l:.2f}% | {100 * m:.2f}% |" for s, l, m in zip(R["e5"]["shift"], R["e5"]["LeNet-5"], R["e5"]["MLP 300-100"])]
        L += [""]
    if "e6" in R:
        L += ["## E6: learned RBF centers, MSE vs MAP", "", "| loss | test error | closest two centers |", "|---|---|---|"]
        L += [f"| {k} | {100 * v['test_err'][-1]:.2f}% | {v['min_prototype_distance']:.3f} |" for k, v in R["e6"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--hessian-samples", type=int, default=500)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.epochs = a.epochs or (2 if a.quick else 20)
    a.batch = a.batch or (32 if a.quick else 1)
    if a.quick:
        a.hessian_samples = 100
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        train, test = load()
        if a.quick:
            train = (train[0][:6000], train[1][:6000])
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
