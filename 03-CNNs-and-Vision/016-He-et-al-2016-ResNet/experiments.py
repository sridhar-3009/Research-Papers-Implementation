"""Reproduce the CIFAR-10 experiments of He et al. (2016), Section 4.2.

  E1  Fig. 6 + Table 6: plain vs ResNet with 20, 32, 44, 56 layers (and ResNet-110).
      Paper: plain nets get WORSE with depth (training error too); ResNets get better:
      8.75 / 7.51 / 7.17 / 6.97 / 6.43 % test error.
  E2  Fig. 7: std of the layer responses (after BN, before the addition) for plain-20/56 and
      ResNet-20/56/110 after training. Paper: ResNet responses are smaller, and smaller with depth.
  E3  Table 3 at CIFAR scale: shortcut options A / B / C on ResNet-32.
  E4  ResNet-1202 (19.4M parameters). Paper: trains fine (<0.1% training error) but 7.93% test
      error, worse than ResNet-110: overfitting. VERY heavy; only with --with-1202.

Paper recipe: batch 128, SGD momentum 0.9, weight decay 1e-4, lr 0.1, divided by 10 at 32k and 48k
iterations, stop at 64k (= 164 epochs of 45k images); augmentation = pad 4 + random 32x32 crop + flip.
ResNet-110 warms up at lr 0.01 until the training error is below 80% (~400 iterations).

!! HEAVY. Not run on the author's laptop. The full E1 is ~10 networks x 64k iterations.
       python3 experiments.py --quick          # a few thousand iterations, ~20-30 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from resnet import CifarResNet, count_params, layer_responses

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_cifar():
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    f = lambda d: (torch.tensor(d.data).permute(0, 3, 1, 2).float() / 255, torch.tensor(d.targets))
    (X, y), (Xt, yt) = f(tr), f(te)
    mean, std = X.mean((0, 2, 3), keepdim=True), X.std((0, 2, 3), keepdim=True)
    return ((X - mean) / std, y), ((Xt - mean) / std, yt)


def augment(xb, g):
    """Pad 4 pixels on each side, take a random 32x32 crop, flip half of them."""
    N = len(xb)
    p = F.pad(xb, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2, N), generator=g)
    out = torch.stack([p[k, :, i[k]:i[k] + 32, j[k]:j[k] + 32] for k in range(N)])
    flip = torch.rand(N, generator=g) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


@torch.no_grad()
def error(net, X, y, bs=1000):
    net.eval()
    wrong = sum((net(X[i:i + bs].to(DEV)).argmax(1).cpu() != y[i:i + bs]).sum().item() for i in range(0, len(X), bs))
    net.train()
    return wrong / len(X)


def train(net, data, iters, warmup=False, eval_every=1000, seed=0):
    """The paper's schedule, scaled to `iters` total iterations (32k/48k/64k -> 50%/75%/100%)."""
    (X, y), (Xt, yt) = data
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = net.to(DEV).train()
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    milestones = (int(iters * 0.5), int(iters * 0.75))
    log = {"iter": [], "train_err": [], "test_err": []}
    running, warm = [], warmup
    for it in range(1, iters + 1):
        lr = 0.01 if warm else (0.1 if it < milestones[0] else 0.01 if it < milestones[1] else 0.001)
        for group in opt.param_groups:
            group["lr"] = lr
        idx = torch.randint(0, len(X), (128,), generator=g)
        out = net(augment(X[idx], g).to(DEV))
        loss = F.cross_entropy(out, y[idx].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        running.append((out.argmax(1).cpu() != y[idx]).float().mean().item())
        running = running[-200:]
        if warm and len(running) >= 50 and sum(running) / len(running) < 0.8:   # ResNet-110's warm-up
            warm = False
        if it % eval_every == 0 or it == iters:
            log["iter"].append(it)
            log["train_err"].append(sum(running) / len(running))
            log["test_err"].append(error(net, Xt, yt))
            print(f"    iter {it}: train {log['train_err'][-1]:.3f} test {log['test_err'][-1]:.3f}", flush=True)
    return net, log


def e1(data, a):
    out, nets = {}, {}
    for n in (3, 5, 7, 9):
        for residual in (False, True):
            name = f"{'ResNet' if residual else 'plain'}-{6 * n + 2}"
            print(f"  E1 {name}")
            nets[name], out[name] = train(CifarResNet(n, residual=residual), data, a.iters)
    print("  E1 ResNet-110")
    nets["ResNet-110"], out["ResNet-110"] = train(CifarResNet(18), data, a.iters, warmup=True)
    for name in ("plain-20", "plain-56", "ResNet-20", "ResNet-56", "ResNet-110"):        # for E2
        torch.save(nets[name].state_dict(), HERE / f"{name}.pt")
    return out


def e2(data, a):
    """Needs E1's saved networks."""
    (Xt, _) = data[1]
    out = {}
    for name, n, residual in (("plain-20", 3, False), ("plain-56", 9, False), ("ResNet-20", 3, True),
                              ("ResNet-56", 9, True), ("ResNet-110", 18, True)):
        path = HERE / f"{name}.pt"
        if not path.exists():
            print(f"  E2: {path.name} missing, run e1 first"); continue
        net = CifarResNet(n, residual=residual).to(DEV)
        net.load_state_dict(torch.load(path, map_location=DEV))
        r = layer_responses(net, Xt[:1000].to(DEV))
        out[name] = r
        print(f"  E2 {name}: mean response std {sum(r) / len(r):.3f}", flush=True)
    return out


def e3(data, a):
    return {opt: train(CifarResNet(5, option=opt), data, a.iters)[1] for opt in "ABC"}


def e4(data, a):
    if not a.with_1202:
        print("  E4 skipped (pass --with-1202; it is 19.4M parameters and very slow)")
        return None
    return train(CifarResNet(200), data, a.iters, warmup=True)[1]


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


def report(R, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    paper = {"ResNet-20": 8.75, "ResNet-32": 7.51, "ResNet-44": 7.17, "ResNet-56": 6.97, "ResNet-110": 6.43}
    L = ["# Results", "", f"{a.iters} iterations per network" + (" (QUICK run: far from the paper's 64k)" if a.quick else ""), ""]
    if R.get("e1"):
        L += ["## E1: plain vs residual (Fig. 6, Table 6)", "", "| network | params | final train error | final test error | paper test error |", "|---|---|---|---|---|"]
        for name, v in R["e1"].items():
            depth = int(name.split("-")[1])
            n = (depth - 2) // 6
            p = count_params(CifarResNet(n, residual=name.startswith("ResNet")))
            L.append(f"| {name} | {p / 1e6:.2f}M | {100 * v['train_err'][-1]:.2f}% | {100 * v['test_err'][-1]:.2f}% | {paper.get(name, '-')} |")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for name, v in R["e1"].items():
            k = 0 if name.startswith("plain") else 1
            ax[k].plot(v["iter"], [100 * e for e in v["train_err"]], "--")
            ax[k].plot(v["iter"], [100 * e for e in v["test_err"]], label=name)
        for k, t in enumerate(("plain", "ResNet")):
            ax[k].set_title(t); ax[k].set_xlabel("iterations"); ax[k].legend(fontsize=7); ax[k].set_ylim(0, 25)
        ax[0].set_ylabel("error % (dashed: training)")
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e1_fig6.png", dpi=130); plt.close(fig)
        L += ["", "![](figures/e1_fig6.png)", ""]
    if R.get("e2"):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for name, r in R["e2"].items():
            ax.plot(r, label=name)
        ax.set_xlabel("3x3 layer index"); ax.set_ylabel("std of response"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(HERE / "figures" / "e2_fig7.png", dpi=130); plt.close(fig)
        L += ["## E2: layer responses (Fig. 7)", ""] + [f"- {k}: mean std {sum(r) / len(r):.3f}" for k, r in R["e2"].items()]
        L += ["", "![](figures/e2_fig7.png)", ""]
    if R.get("e3"):
        L += ["## E3: shortcut options on ResNet-32 (paper, ImageNet ResNet-34: A 25.03, B 24.52, C 24.19 top-1)", ""]
        L += [f"- option {k}: test error {100 * v['test_err'][-1]:.2f}%" for k, v in R["e3"].items()] + [""]
    if R.get("e4"):
        L += ["## E4: ResNet-1202 (paper: 7.93% test, training error < 0.1%)", "",
              f"train {100 * R['e4']['train_err'][-1]:.2f}%, test {100 * R['e4']['test_err'][-1]:.2f}%"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--with-1202", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.iters = a.iters or (2000 if a.quick else 64000)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        data = load_cifar()
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
