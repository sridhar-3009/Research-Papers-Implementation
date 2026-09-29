"""Reproduce Srivastava et al. (2014), "Dropout", on MNIST.

  E1  Table 9 (Section 6.5): dropout vs other regularizers, 784-1024-1024-2048-10 ReLU
  E2  Figure 9 (Section 7.3): the retention probability p, with n fixed and with p*n fixed
  E3  Figure 10 (Section 7.4): training-set size 100 ... 50,000
  E4  Figure 11 (Section 7.5): Monte-Carlo averaging of k nets vs weight scaling
  E5  Figures 7-8 (Sections 7.1-7.2): ReLU autoencoder features and sparsity
  E6  Table 10 (Section 10): Bernoulli vs Gaussian dropout
  E7  Section 9.1: dropout in linear regression = a form of ridge regression

Scale: the paper trained for up to 10^6 updates. Here 40-60 epochs (24k-36k updates)
per run, and hyperparameters picked on a 10,000-image validation split from a small grid.

Run:  python3 experiments.py            (~1 hour on an Apple-silicon GPU)
      python3 experiments.py --quick    (~5 minutes)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from dropout import (Autoencoder, Net, dropout_linear_regression_closed_form,
                     dropout_linear_regression_sgd, error_rate, monte_carlo_error, train)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def mnist():
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    X = (tr.data.reshape(-1, 784).float() / 255).to(DEV)
    Xt = (te.data.reshape(-1, 784).float() / 255).to(DEV)
    return X[:50000], tr.targets[:50000].to(DEV), X[50000:], tr.targets[50000:].to(DEV), Xt, te.targets.to(DEV)


def fit(sizes, X, y, Xv, yv, Xt, yt, epochs, p_in=1.0, p_h=1.0, noise="bernoulli", seed=0, **kw):
    """Train; return (net, validation error, test error)."""
    net = Net(sizes, p_in, p_h, noise=noise, seed=seed).to(DEV)
    lr = 0.1 if (p_in < 1 or p_h < 1) else 0.05       # dropout nets like a larger step (Appendix A.2)
    train(net, X, y, epochs, lr=lr, momentum=0.95, lr_decay=0.96, seed=seed, **kw)
    return net, error_rate(net, Xv, yv), error_rate(net, Xt, yt)


def e1_regularizers(D, epochs, quick):
    X, y, Xv, yv, Xt, yt = D
    arch = [784, 1024, 1024, 2048, 10]
    grid = {
        "no regularization": [dict()],
        "L2": [dict(l2=1e-5), dict(l2=1e-4)],
        "L2 + L1": [dict(l2=1e-5, l1=1e-6)],
        "max-norm": [dict(max_norm=3.0), dict(max_norm=4.0)],
        "dropout + L2": [dict(p_in=0.8, p_h=0.5, l2=1e-5)],
        "dropout + max-norm": [dict(p_in=0.8, p_h=0.5, max_norm=3.0), dict(p_in=0.8, p_h=0.5, max_norm=4.0)],
    }
    out, best_net = {}, None
    for name, options in grid.items():
        options = options[:1] if quick else options
        best = None
        for kw in options:
            net, v, t = fit(arch, X, y, Xv, yv, Xt, yt, epochs, **kw)
            print(f"E1 {name:20s} {kw}: val {v:.4f} test {t:.4f}", flush=True)
            if best is None or v < best[0]:
                best = (v, t, kw, net)
        out[name] = dict(val=best[0], test=best[1], setting=str(best[2]))
        if name == "dropout + max-norm":
            best_net = best[3]
    return out, best_net


def e2_retention(D, epochs):
    X, y, Xv, yv, Xt, yt = D
    ps = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    fixed_n, fixed_pn = {}, {}
    for p in ps:
        net, _, t = fit([784, 1024, 1024, 1024, 10], X, y, Xv, yv, Xt, yt, epochs, p_h=p, max_norm=3.0)
        fixed_n[p] = dict(test=t, train=error_rate(net, X[:10000], y[:10000]))
        n1, n3 = int(256 / p), int(512 / p)                    # p*n = 256, 256, 512 (paper)
        net, _, t = fit([784, n1, n1, n3, 10], X, y, Xv, yv, Xt, yt, epochs, p_h=p, max_norm=3.0)
        fixed_pn[p] = dict(test=t, train=error_rate(net, X[:10000], y[:10000]), sizes=[n1, n1, n3])
        print(f"E2 p={p}: fixed n test {fixed_n[p]['test']:.4f}, fixed pn test {fixed_pn[p]['test']:.4f}", flush=True)
    return dict(fixed_n=fixed_n, fixed_pn=fixed_pn)


def e3_dataset_size(D, updates=6000):
    X, y, Xv, yv, Xt, yt = D
    out = {}
    for n in (100, 500, 1000, 5000, 10000, 50000):
        epochs = max(1, updates * 100 // n)
        res = {}
        for name, kw in (("without dropout", dict()), ("with dropout", dict(p_in=0.8, p_h=0.5, max_norm=3.0))):
            _, _, t = fit([784, 1024, 1024, 2048, 10], X[:n], y[:n], Xv, yv, Xt, yt, min(epochs, 3000),
                          batch=min(100, n), **kw)
            res[name] = t
        out[n] = res
        print(f"E3 n={n}: {res}", flush=True)
    return out


def e4_monte_carlo(net, D):
    *_, Xt, yt = D
    ws = error_rate(net, Xt, yt)
    mc = {k: float(np.mean([monte_carlo_error(net, Xt, yt, k, seed=s) for s in range(3)]))
          for k in (1, 2, 5, 10, 20, 50, 100, 120)}
    return dict(weight_scaling=ws, monte_carlo=mc)


def e5_autoencoder(D, epochs):
    X = D[0]
    out = {}
    for name, p in (("without dropout", 1.0), ("dropout p=0.5", 0.5)):
        ae = Autoencoder(256, p, seed=0).to(DEV)
        opt = torch.optim.SGD(ae.parameters(), lr=0.1, momentum=0.9)
        gen = torch.Generator().manual_seed(0)
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=gen).to(DEV)
            for i in range(0, len(X), 100):
                xb = X[perm[i:i + 100]]
                rec, _ = ae(xb, train=True)
                loss = torch.nn.functional.binary_cross_entropy(rec, xb, reduction="sum") / len(xb)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            rec, h = ae(D[4][:1000])                               # test images
            if p < 1:
                h = h / p                                         # activations as the net computes them
            err = float(((rec - D[4][:1000]) ** 2).sum(1).mean())
        out[name] = dict(recon_sq_error=err, mean_activation=float(h.mean()),
                         frac_near_zero=float((h < 0.01).float().mean()),
                         unit_means=h.mean(0).cpu().numpy().tolist(),
                         acts_sample=h[:100].flatten().cpu().numpy().tolist(),
                         W=ae.W1.detach().cpu().numpy()[:, :100].tolist())
        print(f"E5 {name}: recon {err:.2f}, mean activation {out[name]['mean_activation']:.3f}", flush=True)
    return out


def e6_gaussian(D, epochs, seeds):
    X, y, Xv, yv, Xt, yt = D
    out = {}
    for noise in ("bernoulli", "gaussian"):
        errs = [fit([784, 1024, 1024, 10], X, y, Xv, yv, Xt, yt, epochs, p_in=0.8, p_h=0.5, noise=noise,
                    max_norm=3.0, seed=s)[2] for s in range(seeds)]
        out[noise] = dict(mean=float(np.mean(errs)), std=float(np.std(errs)), runs=errs)
        print(f"E6 {noise}: {out[noise]}", flush=True)
    return out


def e7_linear_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * np.array([1, 2, 0.5, 3, 1])
    y = X @ np.array([1, -2, 0.5, 1, 0]) + rng.normal(0, 0.5, 200)
    p = 0.7
    wc = dropout_linear_regression_closed_form(X, y, p)
    ws = dropout_linear_regression_sgd(X, y, p, steps=400000, lr=2e-4)
    R = rng.random((20000,) + X.shape) < p
    obj = lambda w: float(np.mean(np.sum((y - (R * X) @ w) ** 2, axis=1)))
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(closed=wc.tolist(), sgd=ws.tolist(), ols=ols.tolist(),
                obj_closed=obj(wc), obj_sgd=obj(ws), obj_ols=obj(ols))


# ---------------------------------------------------------------------------

def report(r):
    L = ["# Srivastava et al. (2014): reproduced results", "", "Generated by `python3 experiments.py`.", "",
         f"## E1. Dropout vs standard regularizers (Table 9), 784-1024-1024-2048-10 ReLU, {r['epochs']} epochs", "",
         "Best setting per row chosen on a 10,000-image validation split.", "",
         "| method | test error | chosen setting | paper |", "|---|---|---|---|"]
    paper9 = {"no regularization": "–", "L2": "1.62%", "L2 + L1": "1.60%", "max-norm": "1.35%",
              "dropout + L2": "1.25%", "dropout + max-norm": "1.05%"}
    for k, v in r["e1"].items():
        L.append(f"| {k} | {v['test']:.2%} | {v['setting']} | {paper9[k]} |")
    e2 = r["e2"]
    L += ["", "## E2. Retention probability p (Figure 9)", "",
          "(a) n fixed: 784-1024-1024-1024-10. (b) p*n fixed = 256, 256, 512 (so small p = wider net).", "",
          "| p | (a) test error | (a) training error | (b) test error | (b) layer sizes |", "|---|---|---|---|---|"]
    for p in e2["fixed_n"]:
        a, b = e2["fixed_n"][p], e2["fixed_pn"][p]
        L.append(f"| {p} | {a['test']:.2%} | {a['train']:.2%} | {b['test']:.2%} | {b['sizes']} |")
    L += ["", "## E3. Training-set size (Figure 10)", "",
          "| training images | without dropout | with dropout | gain |", "|---|---|---|---|"]
    for n, v in r["e3"].items():
        L.append(f"| {n} | {v['without dropout']:.2%} | {v['with dropout']:.2%} | "
                 f"{(v['without dropout'] - v['with dropout']) * 100:+.2f} pts |")
    e4 = r["e4"]
    L += ["", "## E4. Monte-Carlo averaging vs weight scaling (Figure 11)", "",
          f"Weight scaling (one pass of the mean network): **{e4['weight_scaling']:.2%}**. Averaging k sampled "
          "networks (mean of 3 repeats):", "",
          "| k | " + " | ".join(str(k) for k in e4["monte_carlo"]) + " |", "|---|" + "---|" * len(e4["monte_carlo"]),
          "| error | " + " | ".join(f"{v:.2%}" for v in e4["monte_carlo"].values()) + " |"]
    e5 = r["e5"]
    L += ["", "## E5. Autoencoder features and sparsity (Figures 7-8), 256 ReLU hidden units", "",
          "| model | reconstruction error | mean activation | activations < 0.01 |", "|---|---|---|---|"]
    for k, v in e5.items():
        L.append(f"| {k} | {v['recon_sq_error']:.2f} | {v['mean_activation']:.3f} | {v['frac_near_zero']:.1%} |")
    L += ["", "Paper: mean activation ~2.0 without dropout, ~0.7 with dropout."]
    e6 = r["e6"]
    L += ["", "## E6. Bernoulli vs Gaussian dropout (Table 10), 784-1024-1024-10", "",
          "| noise | test error (mean ± std over seeds) |", "|---|---|"]
    for k, v in e6.items():
        L.append(f"| {k} | {v['mean']:.2%} ± {v['std']:.2%} |")
    L += ["", "Paper: Bernoulli 1.08 ± 0.04 %, Gaussian 0.95 ± 0.04 %."]
    e7 = r["e7"]
    L += ["", "## E7. Dropout in linear regression = ridge regression (Section 9.1)", "",
          f"- closed form (ridge with Gamma = diag(XᵀX)^½): w = {np.round(e7['closed'], 3).tolist()}",
          f"- SGD with real random dropout masks:            w = {np.round(e7['sgd'], 3).tolist()}",
          f"- expected dropout loss: closed form {e7['obj_closed']:.1f}, SGD {e7['obj_sgd']:.1f}, "
          f"ordinary least squares {e7['obj_ols']:.1f}"]
    return "\n".join(L) + "\n"


def figures(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    e2 = r["e2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, t in zip(axes, ("fixed_n", "fixed_pn"), ("(a) keeping n fixed", "(b) keeping pn fixed")):
        ps = [float(p) for p in e2[key]]
        ax.plot(ps, [v["test"] * 100 for v in e2[key].values()], "o-", label="test")
        if key == "fixed_n":
            ax.plot(ps, [v["train"] * 100 for v in e2[key].values()], "s--", label="training")
        ax.set_title(f"Fig. 9 {t}", fontsize=9); ax.set_xlabel("probability of retaining a unit (p)")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    axes[0].set_ylabel("classification error %")
    fig.tight_layout(); fig.savefig(HERE / "figures" / "fig9_retention.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [int(n) for n in r["e3"]]
    for name in ("without dropout", "with dropout"):
        ax.semilogx(ns, [v[name] * 100 for v in r["e3"].values()], "o-", label=name)
    ax.set_xlabel("data set size"); ax.set_ylabel("classification error %")
    ax.set_title("Fig. 10 reproduced", fontsize=10); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "fig10_dataset_size.png", dpi=130); plt.close(fig)

    e4 = r["e4"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = [int(k) for k in e4["monte_carlo"]]
    ax.plot(ks, [v * 100 for v in e4["monte_carlo"].values()], "o-", label="Monte-Carlo model averaging")
    ax.axhline(e4["weight_scaling"] * 100, color="k", ls="--", label="weight scaling")
    ax.set_xlabel("number of samples k"); ax.set_ylabel("test error %"); ax.set_ylim(top=min(3, ax.get_ylim()[1]))
    ax.set_title("Fig. 11 reproduced", fontsize=10); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "fig11_monte_carlo.png", dpi=130); plt.close(fig)

    e5 = r["e5"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for col, name in enumerate(e5):
        W = np.array(e5[name]["W"])
        tiles = np.zeros((10 * 29, 10 * 29))
        for k in range(100):
            w = W[:, k].reshape(28, 28); w = (w - w.min()) / (w.max() - w.min() + 1e-9)
            i, j = divmod(k, 10); tiles[i * 29:i * 29 + 28, j * 29:j * 29 + 28] = w
        axes[0, col].imshow(tiles, cmap="gray"); axes[0, col].axis("off")
        axes[0, col].set_title(f"Fig. 7: features, {name}", fontsize=9)
        axes[1, col].hist(e5[name]["acts_sample"], bins=60, color="tab:blue")
        axes[1, col].set_yscale("log"); axes[1, col].set_xlabel("activation")
        axes[1, col].set_title(f"Fig. 8: activations, {name} (mean {e5[name]['mean_activation']:.2f})", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "fig7_8_features_sparsity.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        r = json.loads((HERE / "results.json").read_text())
    else:
        D = mnist()
        epochs = 3 if a.quick else 50
        e1, best = e1_regularizers(D, epochs, a.quick)
        r = dict(epochs=epochs, e1=e1,
                 e2=e2_retention(D, 3 if a.quick else 30),
                 e3=e3_dataset_size(D, updates=300 if a.quick else 6000),
                 e4=e4_monte_carlo(best, D),
                 e5=e5_autoencoder(D, 2 if a.quick else 20),
                 e6=e6_gaussian(D, 3 if a.quick else 40, 1 if a.quick else 3),
                 e7=e7_linear_regression())
        (HERE / "results.json").write_text(json.dumps(r))
    text = report(r)
    (HERE / "results.md").write_text(text)
    print(text)
    figures(r)


if __name__ == "__main__":
    main()
